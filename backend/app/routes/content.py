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

bp = Blueprint('content', __name__, url_prefix='/api/backend')

        # 设置基础路径
BASE_DIR = Path(__file__).parent.parent
RESOURCE_DIR = BASE_DIR / 'resource'
STATIC_DIR = BASE_DIR / 'static'

@bp.route('/content/tree', methods=['GET'])
def get_content_try():
    '''
    获取content的树，返回所有目录结构
    '''
    tree_data = []
    root_node = {
        'id': 'static',  # 将根节点ID设置为'static'
        'name': 'static',
        'type': 'FOLDER',
        'children': []
    }
    tree_data.append(root_node)

    def build_directory_tree(parent_node, current_path):
        """递归构建目录树"""
        if not os.path.exists(current_path) or not os.path.isdir(current_path):
            return

        for item in os.listdir(current_path):
            item_path = os.path.join(current_path, item)
            if os.path.isdir(item_path):
                # 计算相对于STATIC_DIR的路径作为id
                relative_path = os.path.relpath(item_path, STATIC_DIR)
                node = {
                    'id': relative_path,
                    'name': item,
                    'type': 'FOLDER',
                    'children': []
                }
                parent_node['children'].append(node)
                # 递归处理子目录
                build_directory_tree(node, item_path)

    # 从根目录开始构建目录树
    build_directory_tree(root_node, STATIC_DIR)
    return api_response(data=tree_data)
    
@bp.route('/content/node', methods=['POST'])
def add_content_node():
    '''
    添加目录节点
    '''
    # 获取请求参数
    data = request.get_json()
    name = data.get('name')
    parent_id = data.get('parentId')
    
    if not name or not parent_id:
        return api_response(message='name和parentId参数不能为空', code=400)
    
    # 构造新节点的路径
    if parent_id == 'static':
        # 如果是根目录，直接在STATIC_DIR下创建
        new_node_path = os.path.join(STATIC_DIR, name)
    else:
        # 否则在父目录下创建
        node_path = os.path.join(STATIC_DIR, parent_id)
        new_node_path = os.path.join(node_path, name)
    
    # 创建新目录
    os.makedirs(new_node_path, exist_ok=True)
    
    # 返回成功响应
    return api_response(data={
        'id': os.path.relpath(new_node_path, STATIC_DIR),
        'name': name,
        'type': 'FOLDER'
    }, message='创建目录成功')

@bp.route('/content/upload_chapters', methods=['POST'])
def upload_chapters():
    '''
    上传章节
    参数说明：
    - content_id: 内容ID，格式为 "二级分类/内容名称（三级分类）/章节名称（四级分类）"，例如 "comic/海贼王/海贼王-第001章"
    - files: 文件列表，支持多文件上传
    '''
    try:
        # 获取并记录请求参数
        content_id = request.form.get('content_id')
        logging.info(f'接收到上传章节请求，content_id={content_id}')
        
        if not content_id:
            return api_response(message='content_id参数不能为空', code=400)
        
        # 解析content_id获取内容名称和章节名称
        parts = content_id.split('/')
        if len(parts) != 3:
            return api_response(message='content_id格式错误，应为 "二级分类/内容名称/章节名称，请创建到三级分类，选择三级分类再上传内容！"', code=400)
        
        category, content_name, chapter_name = parts
        logging.info(f'解析content_id: 分类={category}, 内容名称={content_name}, 章节名称={chapter_name}')
        
        # 获取文件
        files = request.files.getlist('files')
        logging.info(f'上传文件数量: {len(files)}')
        
        if not files:
            return api_response(message='files参数不能为空', code=400)
        
        # 检查或创建Content记录
        contents = ContentService.get_contents_by_name_and_type(content_name, category.upper())
        content = None
        if not contents or len(contents) == 0:
            # 创建新的Content记录
            content = ContentService.create_content(
                name=content_name,
                author_id='Unknown', # TODO
                cover_url='/static/covers/default_cover.jpg',
                description='f{category} {content_name} {chapter_name}', # TODO
                title=content_name,
                content_type=category.upper(),
                status='DRAFT'
            )
            logging.info(f'创建新的Content记录: {content.id}')
        else:
            content = contents[0]  # 获取第一个匹配的记录
            logging.info(f'使用现有Content记录: {content.id}')
        
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
            relative_path = f'{category}/{content_name}/{chapter_name}/{file.filename}'
            file_type_mapping[file_type].append({
                'url': f'/static/{relative_path}',
                'order': len(file_type_mapping[file_type]) + 1,
                'filename': file.filename,
                'extension': file_ext
            })

            # 保存文件
            save_path = os.path.join(STATIC_DIR, relative_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            file.save(save_path)
            uploaded_files.append(relative_path)

        # 检查Chapter是否已存在
        existing_chapters = ChapterService.get_chapters_by_content_and_title(content.id, chapter_name)
        if existing_chapters:
            # 更新现有Chapter的pages数据
            logging.info(f'更新已存在的Chapter记录: {existing_chapters[0].id}')
            chapter = existing_chapters[0]
            
            # 获取原有的pages数据
            original_pages = chapter.pages or {}
            
            logging.info(f'原有pages数据: {json.dumps(original_pages, indent=2)}')
            logging.info(f'新上传的文件映射: {json.dumps(file_type_mapping, indent=2)}')
            
            # 合并新旧数据
            for file_type, files in file_type_mapping.items():
                logging.info(f'处理文件类型: {file_type}')
                if file_type not in original_pages:
                    logging.info(f'文件类型 {file_type} 在原有数据中不存在，创建新列表')
                    original_pages[file_type] = []
                
                # 检查文件是否已存在
                existing_urls = {file['url'] for file in original_pages[file_type]}
                logging.info(f'当前类型已存在的URL: {existing_urls}')
                
                for file in files:
                    if file['url'] not in existing_urls:
                        # 设置新文件的order为当前列表长度加1
                        file['order'] = len(original_pages[file_type]) + 1
                        original_pages[file_type].append(file)
                        logging.info(f'添加新文件: {file}')
                    else:
                        logging.info(f'文件已存在，跳过: {file["url"]}')
            
            logging.info(f'最终合并后的pages数据: {json.dumps(original_pages, indent=2)}')
            chapter = ChapterService.update_chapter_obj(chapter, pages=original_pages)
        else:
            # 创建新的Chapter记录
            logging.info(f'未找到对应的章节{content_id} {category} {content} {chapter_name}，所以要创建新的Chapter记录: {chapter_name}')
            # 使用当前时间戳作为chapter_no
            current_time = datetime.now()
            chapter_no = current_time.strftime('%Y%m%d%H%M%S')
            
            chapter = ChapterService.create_chapter(
                content_id=content.id,
                chapter_no=chapter_no,
                title=chapter_name,
                pages=file_type_mapping,
                is_free=True
            )
        
        return api_response(data={
            'files': uploaded_files,
            'content_id': content.id,
            'chapter_id': chapter.id
        }, message='文件上传成功，并已更新内容记录')
        
    except Exception as e:
        logging.error(f'文件上传失败：{str(e)}')
        return api_response(message=f'文件上传失败：{str(e)}', code=500)