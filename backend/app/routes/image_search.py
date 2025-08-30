from flask import Blueprint, request, jsonify
from app.utils.response import make_response
import json
import os
from app.scripts.match_images_what_you_want import google_image_search, google_image_search_by_count
import logging
import requests
import datetime
import secrets

bp = Blueprint('image_search', __name__, url_prefix='/api/image-search')

# 加载图片配置
with open(os.path.join(os.path.dirname(__file__), '../scripts/image_dims.json'), 'r', encoding='utf-8') as f:
    IMAGE_CONFIG = json.load(f)

def validate_ratio(ratio):
    """验证图片比例是否在支持列表中"""
    return any(aspect['ratio'] == ratio for aspect in IMAGE_CONFIG['image_aspect_ratios'])

def get_recommended_pixel(ratio):
    """获取指定比例的推荐像素值"""
    for aspect in IMAGE_CONFIG['image_aspect_ratios']:
        if aspect['ratio'] == ratio:
            return aspect['recommended_pixels'][0]
    return None

def generate_layout_combinations():
    """生成所有可能的布局组合"""
    layouts = IMAGE_CONFIG.get('supported_layouts', [])
    card_styles = IMAGE_CONFIG.get('supported_card_styles', [])
    animation_classes = IMAGE_CONFIG.get('supported_animation_class', [])
    
    combinations = []
    for layout in layouts:
        for style in card_styles:
            for animation in animation_classes:
                combinations.append({
                    'layout': layout,
                    'card_style': style,
                    'animation_class': animation
                })
    return combinations

@bp.route('/search', methods=['POST'])
def search_images():
    try:
        data = request.get_json()
        if not data:
            return make_response(code=400, message='请求参数不能为空')
            
        # 获取并验证查询参数
        query = data.get('query')
        ratios = data.get('ratios', [])
        
        if not query:
            return make_response(code=400, message='查询关键词不能为空')
        if not ratios:
            return make_response(code=400, message='图片比例不能为空')
            
        # 验证所有比例是否支持
        invalid_ratios = [ratio for ratio in ratios if not validate_ratio(ratio)]
        if invalid_ratios:
            return make_response(
                code=400,
                message=f'不支持的图片比例：{", ".join(invalid_ratios)}'
            )
            
        # 处理查询关键词（支持中英文逗号分割）
        keywords = [k.strip() for k in query.replace('，', ',').split(',') if k.strip()]
        # 对关键词作特殊的处理：
        # 1. 去除重复
        # 2. 去除前后空格
        keywords = list(set(keywords))
        
        # 3. 智能分配搜索数量
        total_images = 5
        keyword_counts = []
        remaining = total_images
        
        '''
        改进的算法：

        针对不足5个关键字的内容，通过分配关键字与搜索的count值来满足需求

        给关键字进行去重，将数量逐量的分配给每个关键字，但是总的分配数量是5，并且必须保证都能够分配到数量

        从第一个分配一个最大数量的值，第二个分配剩余的最大的数量，递归的进行分配

        分配完成之后，在进行googlesearch的调用时，传递对应的数量
        '''
        for i, _ in enumerate(keywords):
            # 计算当前关键字应分配的图片数量
            count = max(1, remaining // (len(keywords) - i))
            keyword_counts.append(count)
            remaining -= count
        
        # 存储搜索结果
        search_results = []
        
        # 为每个比例和关键词进行搜索
        for ratio in ratios:
            recommended_pixel = get_recommended_pixel(ratio)
            if not recommended_pixel:
                continue
                
            for idx, keyword in enumerate(keywords):
                # 构建搜索查询
                search_query = f'{keyword}'
                
                try:
                    # 调用Google搜索服务，使用分配的数量
                    image_paths = google_image_search_by_count(search_query, count=keyword_counts[idx])
                    
                    if not image_paths:
                        # 导入logging模块
                        logging.warning(f'未找到匹配的图片: {search_query}')
                        continue
                    
                    # 构建本地存储路径
                    local_paths = []
                    for i, url in enumerate(image_paths):
                        # 构建本地文件名
                        filename = f"{keyword}_{ratio}_{i}.jpg"
                        filepath = os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads', 'images', filename)
                        
                        try:
                            # 确保目录存在
                            os.makedirs(os.path.dirname(filepath), exist_ok=True)
                            
                            # 生成唯一文件名
                            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                            unique_id = secrets.token_hex(8)
                            filename = f"{keyword}_{ratio}_{timestamp}_{unique_id}.jpg"
                            filepath = os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads', 'images', filename)
                            
                            # 下载并保存图片
                            from app.utils.http_config import DEFAULT_HEADERS
                            headers = DEFAULT_HEADERS
                            response = requests.get(url, headers=headers)
                            response.raise_for_status()
                            
                            with open(filepath, 'wb') as f:
                                f.write(response.content)
                            
                            # 构建可访问的URL路径
                            local_url = f'/static/uploads/images/{filename}'
                            local_paths.append(local_url)
                            
                        except Exception as e:
                            logging.error(f'保存图片失败 {url} -> {filepath}: {str(e)}')
                            continue
                    
                    # 添加搜索结果
                    if local_paths:
                        # 获取所有可能的布局组合
                        layout_combinations = generate_layout_combinations()
                        
                        # 为每个布局组合创建一个图片结果
                        for layout in layout_combinations:
                            search_results.append({
                                'keyword': keyword,
                                'ratio': ratio,
                                'recommended_pixel': recommended_pixel,
                                'image_paths': local_paths,
                                'layout': layout['layout'],
                                'card_style': layout['card_style'],
                                'animation_class': layout['animation_class']
                            })
                    
                except Exception as e:
                    logging.error(f'图片搜索失败: {str(e)}')
                    continue
        
        return make_response(data={
            'results': search_results
        })
        
    except Exception as e:
        return make_response(code=500, message=f'搜索图片失败：{str(e)}')