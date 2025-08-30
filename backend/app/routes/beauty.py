import os
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from flask import Blueprint, make_response, jsonify, request

bp = Blueprint('beauty', __name__, url_prefix='/api/beauty')
@bp.route('/beaulist', methods=['GET'])
def get_beauty_list():
    """
    获取某个分类下的beauty列表
    
    返回：
    - 固定数量的beauty列表信息（按分页数量来定义的）
    - 如果没有找到配置文件或出现错误，返回空列表
    """
    try:
        # 获取请求参数
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 10, type=int)
        refer_id = request.args.get('refer_id', type=str)
        
        # 构建文件路径
        resource_dir = Path(__file__).parent.parent / 'resource'
        image_mapping_file = resource_dir / 'image_mapping.json'
        
        if not image_mapping_file.exists():
            return make_response(jsonify({
                'code': 404,
                'message': '未找到图片映射文件',
                'data': []
            }), 505)
        
        # 读取并解析文件内容
        with open(image_mapping_file, 'r', encoding='utf-8') as f:
            image_mapping = json.load(f)
        
        def find_directory_by_refer_id(directory, target_refer_id):
            # 检查目录项，不检查到目录下的子文件
            for item in directory.get('children', []):
                if item.get('refer_id') == target_refer_id:
                    return item
            
            return None
        
        # 如果指定了refer_id，查找对应的目录
        target_directory = None
        if refer_id:
            target_directory = find_directory_by_refer_id(image_mapping, refer_id)
            if not target_directory:
                return make_response(jsonify({
                    'code': 404,
                    'message': '未找到指定的目录',
                    'data': []
                }), 404)
        else:
            target_directory = image_mapping
        
        # 将目录结构转换为扁平的图片列表
        images_list = []
        current_dir_name = target_directory.get('dir_name', '')
        
        for item in target_directory.get('children', []):
            if item['type'] == 'file':
                image_path = os.path.join(current_dir_name, item['name'])
                images_list.append({
                    'id': item.get('refer_id', str(len(images_list) + 1)),
                    'name': item['name'],
                    'url': f'/static/beauty/{image_path}',
                    'directory': current_dir_name,
                    'type': item['type'],
                    'views': random.randint(1000, 10000)  # 随机访问次数
                })
            else:  # directory
                '''
                # 非文件不加进来（即：目录等不会加进来的）
                images_list.append({
                    'id': str(len(images_list) + 1),
                    'name': item['dir_name'],
                    'type': item['type'],
                    'directory': current_dir_name
                })
                '''
        
        # 计算分页信息
        total = len(images_list)
        total_pages = (total + page_size - 1) // page_size
        page = min(max(1, page), total_pages)  # 确保页码在有效范围内
        
        # 获取当前页的数据
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total)
        current_page_data = images_list[start_idx:end_idx]
        
        return make_response(jsonify({
            'code': 200,
            'message': '获取图片列表成功',
            'data': {
                'list': current_page_data,
                'current_directory': current_dir_name,
                'pagination': {
                    'total': total,
                    'page': page,
                    'page_size': page_size,
                    'total_pages': total_pages
                }
            }
        }), 200)
        
    except Exception as e:
        return make_response(jsonify({
            'code': 500,
            'message': f'获取图片列表失败: {str(e)}',
            'data': []
        }), 500)
    