import os
import json
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from flask import Blueprint, make_response, jsonify, request
from app.services.content_service import ContentService, ChapterService
from app.models.content import Content
from ..utils.response import make_response as api_response
from ..utils.auth import token_required

import subprocess
import threading

# https://alphago.ltd/backend/rtmp/adm

bp = Blueprint('rtmp', __name__, url_prefix='/api/backend')

        # 设置基础路径
BASE_DIR = Path(__file__).parent.parent
RESOURCE_DIR = BASE_DIR / 'resource'
STATIC_DIR = BASE_DIR / 'static'

def push_to_rtmp_server(file_path, filename):
    global stream_process
    logging.info(f'开始处理视频文件: {filename}')
    logging.info(f'视频文件路径: {file_path}')

    # 构建完整的文件路径
    input_file = f"{file_path}"

    ## TODO 临时固定一个直播的地址，在Living.vue中直接引用
    online_file = "rtmp://47.98.62.98:1935/live/stream"
    logging.info(f'RTMP推流地址: {online_file}')
    
    # TODO 可以考虑直接使用ffmpeg包的方式来处理，而不是使用子进程调用ffmpeg指令的方式
    # 使用 ffmpeg 将视频文件推送到 RTMP 服务器
    command = [
        'ffmpeg',
        '-re',  # 以输入文件的原始帧率读取输入文件
        '-i', input_file,  # 输入文件路径
        '-c:v', 'libx264',  # 视频编解码器
        '-preset', 'ultrafast',  # 编码速度预设
        '-f', 'flv',  # 输出格式
        f'{online_file}'  # RTMP 服务器地址
    ]
    logging.info(f'FFmpeg命令: {" ".join(command)}')

    # 定义一个函数来运行 ffmpeg 命令
    def run_ffmpeg():
        try:
            logging.info('开始执行FFmpeg命令...')
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8')
                logging.error(f'FFmpeg命令执行失败: {error_msg}')
                print(f"Error: {error_msg}")
            else:
                logging.info('视频推流成功')
                print("Video pushed to RTMP server successfully.")
        except Exception as e:
            logging.error(f'FFmpeg命令执行异常: {str(e)}')
            print(f"Exception during FFmpeg execution: {str(e)}")

    # 启动一个线程来运行 ffmpeg 命令
    try:
        logging.info('创建FFmpeg执行线程...')
        thread = threading.Thread(target=run_ffmpeg)
        thread.start()
        logging.info('FFmpeg执行线程已启动')
    except Exception as e:
        logging.error(f'创建FFmpeg执行线程失败: {str(e)}')
        raise Exception(f'创建FFmpeg执行线程失败: {str(e)}')

    return online_file
@bp.route('/rtmp/upload_video', methods=['POST'])
@token_required
def upload_video(current_user):
    '''
    上传直播视频
    参数说明：
    - files: 文件列表，支持多文件上传
    '''
    try:
        logging.warn('接收到视频上传请求', current_user)
        # 获取文件
        files = request.files.getlist('files')
        logging.info(f'上传文件数量: {len(files)}')
        
        if not files:
            logging.warning('未接收到上传文件')
            return api_response(message='files参数不能为空', code=400)
        
        # 文件类型配置
        FILE_TYPE_CONFIG = {
            'images': {
                'extensions': ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'],
                'max_size': 10 * 1024 * 1024  # 10MB
            },
            'audios': {
                'extensions': ['.mp3', '.wav', '.ogg', '.m4a', '.flac'],
                'max_size': 50 * 1024 * 1024  # 50MB
            },
            'videos': {
                'extensions': ['.mp4', '.webm', '.mkv', '.avi'],
                'max_size': 200 * 1024 * 1024  # 200MB
            },
            'documents': {
                'extensions': ['.pdf', '.mobi', '.epub'],
                'max_size': 100 * 1024 * 1024  # 100MB
            },
            'texts': {
                'extensions': ['.txt'],
                'max_size': 5 * 1024 * 1024  # 5MB
            }
        }
        
        # 构建允许的文件类型集合
        ALLOWED_EXTENSIONS = set()
        for config in FILE_TYPE_CONFIG.values():
            ALLOWED_EXTENSIONS.update(config['extensions'])
        
        uploaded_files = []
        # 根据文件类型分类存储上传的文件
        file_type_mapping = {}
        
        online_file = None
        for file in files:
            file_ext = os.path.splitext(file.filename)[1].lower()
            
            # 验证文件类型
            if file_ext not in ALLOWED_EXTENSIONS:
                logging.warning(f'不支持的文件类型: {file.filename}')
                return api_response(message=f'不支持的文件类型: {file_ext}', code=400)
            
            # 根据文件扩展名确定文件类型
            file_type = None
            for type_name, config in FILE_TYPE_CONFIG.items():
                if file_ext in config['extensions']:
                    file_type = type_name
                    break
            
            if file_type is None:
                file_type = 'others'
                logging.warning(f'文件 {file.filename} 未能匹配到具体类型，归类为others')
            
            # 检查文件大小
            if file_type in FILE_TYPE_CONFIG:
                max_size = FILE_TYPE_CONFIG[file_type]['max_size']
                # 获取文件大小（注意：这里假设文件对象有tell()方法）
                file.seek(0, os.SEEK_END)
                file_size = file.tell()
                file.seek(0)  # 重置文件指针
                logging.info(f'文件大小检查: {file.filename}, 大小: {file_size/1024/1024:.2f}MB, 限制: {max_size/1024/1024:.2f}MB')
                
                if file_size > max_size:
                    logging.warning(f'文件过大: {file.filename}, 大小: {file_size}, 限制: {max_size}')
                    return api_response(
                        message=f'文件 {file.filename} 超过大小限制 {max_size/(1024*1024)}MB',
                        code=400
                    )
            
            # 初始化文件类型列表
            if file_type not in file_type_mapping:
                file_type_mapping[file_type] = []
            
            # 构建文件URL和顺序信息
            relative_path = f'uploads/{file.filename}'
            logging.info(f'构建文件保存路径: {relative_path}')

            # 保存文件
            try:
                save_path = os.path.join(STATIC_DIR, relative_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                logging.info(f'开始保存文件: {save_path}')
                file.save(save_path)
                logging.info(f'文件保存成功: {save_path}')
            except Exception as e:
                logging.error(f'文件保存失败: {save_path}, 错误: {str(e)}')
                return api_response(message=f'文件保存失败: {str(e)}', code=500)

            try:
                logging.info(f'开始推流处理: {file.filename}')
                online_file = push_to_rtmp_server(save_path, file.filename)
                logging.info(f'推流处理成功: {online_file}')
            except Exception as e:
                logging.error(f'推流处理失败: {file.filename}, 错误: {str(e)}')
                return api_response(message=f'推流处理失败: {str(e)}', code=500)
        
        return api_response(data={
            'files': uploaded_files,
            'online_url': online_file
        }, message='文件上传成功...')
        
    except Exception as e:
        logging.error(f'文件上传失败：{str(e)}')
        return api_response(message=f'文件上传失败：{str(e)}', code=500)