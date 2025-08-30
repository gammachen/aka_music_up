import os
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from flask import Blueprint, make_response, jsonify

recommend_bp = Blueprint('recommend', __name__, url_prefix='/api/recommend')
@recommend_bp.route('/online_musics', methods=['GET'])
def get_recommendations():
    """
    获取推荐音乐列表
    
    返回：
    - 随机选择的12首音乐信息
    - 如果没有找到配置文件或出现错误，返回空列表
    """
    try:
        # 获取最新的配置文件
        latest_file = get_latest_processed_songs_file()
        if not latest_file:
            return make_response(jsonify({
                'code': 404,
                'message': '没有找到可用的音乐配置文件',
                'data': []
            }), 404)
        
        # 构造缓存文件名
        cache_filename = f"online_musics_{latest_file.name}"
        cache_file = latest_file.parent / cache_filename
        
        # 检查缓存文件是否存在
        if cache_file.exists():
            # 直接读取缓存文件
            with open(cache_file, 'r', encoding='utf-8') as f:
                recommended_songs = json.load(f)
        else:
            # 读取原始文件内容
            with open(latest_file, 'r', encoding='utf-8') as f:
                all_songs = json.load(f)
            
            # 随机选择12首歌曲
            recommended_songs = random.sample(all_songs, min(12, len(all_songs)))
            
            # 保存到缓存文件
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(recommended_songs, f, ensure_ascii=False, indent=2)
        
        return make_response(jsonify({
            'code': 200,
            'message': '获取推荐音乐成功',
            'data': recommended_songs
        }), 200)
        
    except Exception as e:
        return make_response(jsonify({
            'code': 500,
            'message': f'获取推荐音乐失败: {str(e)}',
            'data': []
        }), 500)
        

def get_latest_processed_songs_file():
    """
    获取最新的processed_songs配置文件
    
    算法说明：
    1. 获取resource目录下所有文件
    2. 筛选出符合日期_processed_songs.json格式的文件
    3. 解析文件名中的日期部分
    4. 按日期从大到小排序
    5. 返回最新的文件路径
    
    如果当天没有配置文件，会依次往前查找，直到找到最近的一个配置文件
    """
    resource_dir = Path(__file__).parent.parent / 'resource'
    processed_files = []
    
    # 遍历resource目录下的所有文件
    for file in resource_dir.glob('*_processed_songs.json'):
        try:
            # 解析文件名中的日期部分
            date_str = file.name.split('_')[0]
            file_date = datetime.strptime(date_str, '%Y%m%d')
            processed_files.append((file_date, file))
        except (ValueError, IndexError):
            continue
    
    # 按日期从大到小排序
    processed_files.sort(key=lambda x: x[0], reverse=True)
    
    # 返回最新的文件路径
    return processed_files[0][1] if processed_files else None


'''
获取推荐漫画列表（首页推荐使用）
'''
@recommend_bp.route('/online_comics', methods=['GET'])
def get_recommendations_of_comics():
    """
    获取推荐漫画列表
    
    返回：
    - 随机选择的12部漫画信息
    - 如果没有找到配置文件或出现错误，返回空列表
    """
    try:
        
        # 构造缓存文件名
        cache_filename = f"comic_landing_recomments.json"
        cache_file = Path(__file__).parent.parent / 'resource' / cache_filename
        
        # 检查缓存文件是否存在
        if cache_file.exists():
            # 直接读取缓存文件
            with open(cache_file, 'r', encoding='utf-8') as f:
                recommended_songs = json.load(f)
        else:
            # 从数据库中获取最新的12部漫画
            from app.models.content import Content
            from sqlalchemy import desc
            
            # , status='PUBLISHED'
            latest_comics = Content.query.filter_by(type='COMIC')\
                .order_by(desc(Content.updated_at))\
                .limit(12)\
                .all()
            
            # 将查询结果转换为字典列表
            recommended_songs = [comic.to_dict() for comic in latest_comics]
            
            # 保存到缓存文件
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(recommended_songs, f, ensure_ascii=False, indent=2)
        
        return make_response(jsonify({
            'code': 200,
            'message': '获取推荐Comic成功',
            'data': recommended_songs
        }), 200)
        
    except Exception as e:
        return make_response(jsonify({
            'code': 500,
            'message': f'获取推荐Comic失败: {str(e)}',
            'data': []
        }), 500)
        
