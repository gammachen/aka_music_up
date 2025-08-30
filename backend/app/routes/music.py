import os
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from flask import Blueprint, make_response, jsonify, request

bp = Blueprint('music', __name__, url_prefix='/api/music')
@bp.route('/mulist', methods=['GET'])
def get_music_list():
    """
    获取某个分类下的音乐列表
    
    返回：
    - 固定数量的音乐列表信息（按分页数量来定义的）
    - 如果没有找到配置文件或出现错误，返回空列表
    """
    try:
        # 获取请求参数
        category_id = request.args.get('category_id', type=str)
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 10, type=int)
        
        if not category_id:
            return make_response(jsonify({
                'code': 400,
                'message': '缺少分类ID参数',
                'data': []
            }), 400)
        
        # 构建文件路径
        resource_dir = Path(__file__).parent.parent / 'resource'
        song_mapping_file = resource_dir / f"{category_id}_song_mapping.json"
        
        if not song_mapping_file.exists():
            return make_response(jsonify({
                'code': 404,
                'message': f'未找到分类 {category_id} 的音乐列表',
                'data': []
            }), 404)
        
        # 读取并解析文件内容
        with open(song_mapping_file, 'r', encoding='utf-8') as f:
            song_mapping = json.load(f)
        
        # 将字典转换为列表，便于分页
        songs_list = []
        for song_info, url in song_mapping.items():
            # 解析歌曲信息
            parts = song_info.split('.')
            if len(parts) >= 2:
                song_id = parts[0]
                artist_title = '.'.join(parts[1:]).split('-')
                artist = artist_title[0].strip() if len(artist_title) > 1 else '未知歌手'
                title = artist_title[1].strip() if len(artist_title) > 1 else artist_title[0].strip()
            else:
                song_id = '0'
                artist = '未知歌手'
                title = song_info
            
            songs_list.append({
                'id': song_id,
                'title': title,
                'artist': artist,
                'url': f'/static/videos/{category_id}{url}', # 注意构造的路径，要与static目录下的目录结构保持一致
                'coverUrl': f'/static/covers/{random.randint(1, 10)}.png',  # 随机封面
                'plays': random.randint(1000, 10000)  # 随机播放次数
            })
        
        # 计算分页信息
        total = len(songs_list)
        total_pages = (total + page_size - 1) // page_size
        page = min(max(1, page), total_pages)  # 确保页码在有效范围内
        
        # 获取当前页的数据
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total)
        
        # current_page_data = songs_list[start_idx:end_idx]
        # TODO 对map无法正确分页的，每次都会随机变换数组的，暂时全部返回 性能会有比较大的影响（如果数据量大的情况）
        current_page_data = songs_list
        
        return make_response(jsonify({
            'code': 200,
            'message': '获取音乐列表成功',
            'data': {
                'list': current_page_data,
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
            'message': f'获取音乐列表失败: {str(e)}',
            'data': []
        }), 500)
    
        
