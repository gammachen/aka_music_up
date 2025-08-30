#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from typing import List, Dict, Any

def load_poems_from_file(file_path: str) -> List[Dict]:
    """从文件加载诗歌数据
    
    Args:
        file_path: 文件路径
        
    Returns:
        诗歌数据列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载文件失败: {e}")
        return []

def get_resource_paths() -> Dict[str, str]:
    """获取资源文件路径
    
    Returns:
        包含各种资源路径的字典
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resource_dir = os.path.join(script_dir, 'resource')
    
    return {
        'resource_dir': resource_dir,
    }

def generate_timeline_json(poems: List[Dict], poet_name: str, poems_with_images: List[Dict] = None) -> Dict[str, Any]:
    """根据诗歌数据生成时间线JSON
    
    Args:
        poems: 诗歌数据列表
        poet_name: 诗人名称
        
    Returns:
        时间线JSON数据
    """
    # 找出有效的最大年份值
    max_valid_year = 0
    for poem in poems:
        year = poem.get('creation_year', None)
        if year is not None and year != "未知":
            try:
                year_int = int(year) if not isinstance(year, int) else year
                if year_int < 9999:  # 排除默认值
                    max_valid_year = max(max_valid_year, year_int)
            except (ValueError, TypeError):
                pass
    
    # 如果没有找到有效年份，使用默认值
    if max_valid_year == 0:
        max_valid_year = 9999
    
    # 按创作年份排序
    def get_year_value(poem):
        year = poem.get('creation_year', max_valid_year)
        if isinstance(year, int):
            return year
        try:
            return int(year)
        except (ValueError, TypeError):
            return max_valid_year
    
    sorted_poems = sorted(poems, key=get_year_value)
    
    # 生成事件列表
    events = []
    
    # 创建标题到图片URL的映射
    title_to_image_url = {}
    if poems_with_images:
        for poem in poems_with_images:
            if 'title' in poem and 'localImgUrl' in poem and poem['localImgUrl']:
                title_to_image_url[poem['title']] = poem['localImgUrl']
                print(f"已加载《{poem['title']}》的图片URL: {poem['localImgUrl']}")
    
    # 添加诗人出生事件（示例数据，实际应根据实际情况修改）
    birth_event = {
        "media": {
            "url": "/img/libai_1.jpg"
        },
        "start_date": {
            "year": "701"
        },
        "text": {
            "headline": f"{poet_name} 出生于蜀郡绵州昌隆县（一说出生于西域碎叶",
            "text": "<p>出生于蜀郡绵州昌隆县（一说出生于西域碎叶）。李白先世曾迁居碎叶（今吉尔吉斯斯坦托克马克市），后其父逃归于蜀，定居绵州昌隆县青莲乡，李白即出生于此。祖籍为陇西郡成纪县（今甘肃省天水市秦安县）。其家世、家族皆不详。</p>"
        }
    }
    events.append(birth_event)
    
    # 为每首诗生成一个事件
    for i, poem in enumerate(sorted_poems):
        title = poem.get('title', '无题')
        year = poem.get('creation_year', '')
        if not year or year == '未知':  # 对于没有年份的诗，使用最大有效年份
            year = max_valid_year
            
        place = poem.get('creation_place', '未知')
        scene = poem.get('creation_scene', '')
        background = poem.get('historical_background', '')
        content = '\n'.join(poem.get('paragraphs', []))
        
        # 生成事件描述
        headline = f"{poet_name} 创作《{title}》"
        text_content = f"<p><strong>创作地点：</strong>{place}</p>"
        
        if scene:
            text_content += f"<p><strong>创作场景：</strong>{scene}</p>"
            
        if background:
            text_content += f"<p><strong>历史背景：</strong>{background}</p>"
            
        # 处理换行符
        formatted_content = content.replace('\n', '<br>')
        text_content += f"<p><strong>诗歌内容：</strong><br>{formatted_content}</p>"
        
        # 生成事件对象
        # 查找对应的图片URL
        image_url = "/img/libai_1.jpg"  # 默认图片URL
        if title in title_to_image_url:
            image_url = title_to_image_url[title]
            print(f"已为《{title}》设置图片URL: {image_url}")
        
        event = {
            "media": {
                "url": image_url
            },
            "start_date": {
                "year": str(year)
            },
            "text": {
                "headline": headline,
                "text": text_content
            }
        }
        
        events.append(event)
    
    # 构造完整的时间线数据
    timeline_data = {
        "events": events
    }
    
    return timeline_data

def generate_simple_timeline_json(poems: List[Dict], poet_name: str) -> List[Dict[str, Any]]:
    """根据诗歌数据生成简化的时间线JSON
    
    Args:
        poems: 诗歌数据列表
        poet_name: 诗人名称
        
    Returns:
        简化的时间线JSON数据列表
    """
    # 找出有效的最大年份值
    max_valid_year = 0
    for poem in poems:
        year = poem.get('creation_year', None)
        if year is not None and year != "未知":
            try:
                year_int = int(year) if not isinstance(year, int) else year
                if year_int < 9999:  # 排除默认值
                    max_valid_year = max(max_valid_year, year_int)
            except (ValueError, TypeError):
                pass
    
    # 如果没有找到有效年份，使用默认值
    if max_valid_year == 0:
        max_valid_year = 9999
    
    # 按创作年份排序
    def get_year_value(poem):
        year = poem.get('creation_year', max_valid_year)
        if isinstance(year, int):
            return year
        try:
            return int(year)
        except (ValueError, TypeError):
            return max_valid_year
    
    sorted_poems = sorted(poems, key=get_year_value)
    
    # 生成简化的时间线数据
    simple_timeline = []
    
    # 添加诗人出生事件
    birth_event = {
        "content": ["出生于蜀郡绵州昌隆县（一说出生于西域碎叶）。李白先世曾迁居碎叶（今吉尔吉斯斯坦托克马克市），后其父逃归于蜀，定居绵州昌隆县青莲乡，李白即出生于此。祖籍为陇西郡成纪县（今甘肃省天水市秦安县）。"],
        "time": "701"
    }
    simple_timeline.append(birth_event)
    
    # 为每首诗生成一个事件
    for poem in sorted_poems:
        title = poem.get('title', '无题')
        year = poem.get('creation_year', '')
        if not year or year == '未知':  # 对于没有年份的诗，使用最大有效年份
            year = max_valid_year
            
        place = poem.get('creation_place', '未知')
        scene = poem.get('creation_scene', '')
        background = poem.get('historical_background', '')
        
        # 生成事件描述
        content_items = []
        content_items.append(f"创作《{title}》于{place}。")
        
        if scene:
            content_items.append(scene)
            
        if background:
            content_items.append(background)
        
        # 生成事件对象
        event = {
            "content": content_items,
            "time": str(year)
        }
        
        simple_timeline.append(event)
    
    return simple_timeline

def save_timeline_json(timeline_data: Dict, output_path: str) -> None:
    """保存时间线JSON数据到文件
    
    Args:
        timeline_data: 时间线JSON数据
        output_path: 输出文件路径
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(timeline_data, f, ensure_ascii=False, indent=2)
        print(f"已保存时间线JSON数据到: {output_path}")
    except Exception as e:
        print(f"保存时间线JSON数据失败: {e}")

def main():
    paths = get_resource_paths()
    
    # 设置参数
    poet_name = "李白"
    
    # 输入和输出文件路径
    input_json_path = os.path.join(paths['resource_dir'], f'{poet_name}_enriched.json')
    output_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'frontend', 'public', f'{poet_name}_timeline_data.json')
    output_simple_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'frontend', 'public', f'{poet_name}_simple_timeline_data.json')
    
    # 加载诗歌数据
    poems = load_poems_from_file(input_json_path)
    
    if not poems:
        print(f"未找到{poet_name}的诗歌数据")
        return
    
    # 检查是否存在带图片的enriched文件
    poems_with_images_data = None
    enriched_with_images_path = os.path.join(paths['resource_dir'], f'{poet_name}_enriched_with_images.json')
    if os.path.exists(enriched_with_images_path):
        print(f"找到带图片的数据文件: {enriched_with_images_path}")
        try:
            # 加载带图片的数据
            with open(enriched_with_images_path, 'r', encoding='utf-8') as f:
                poems_with_images_data = json.load(f)
            print(f"已成功加载带图片的数据")
        except Exception as e:
            print(f"加载带图片的数据时出错: {e}")
    
    # 生成标准时间线JSON，直接传入带图片的数据
    timeline_data = generate_timeline_json(poems, poet_name, poems_with_images_data)
    
    # 保存标准时间线JSON
    save_timeline_json(timeline_data, output_json_path)
    
    # 生成简化时间线JSON
    simple_timeline_data = generate_simple_timeline_json(poems, poet_name)
    
    # 保存简化时间线JSON
    save_timeline_json(simple_timeline_data, output_simple_json_path)
    
    print(f"已成功生成{poet_name}的标准和简化时间线JSON数据")
    print(f"标准时间线JSON保存至: {output_json_path}")
    print(f"简化时间线JSON保存至: {output_simple_json_path}")

if __name__ == "__main__":
    main()