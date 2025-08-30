import os
import json
from pathlib import Path
from match_images_what_you_want import baidu_images_search_and_restore_for_covers

def update_comic_crawler_categories():
    # 设置基础路径
    BASE_DIR = Path(__file__).parent.parent
    RESOURCE_DIR = BASE_DIR / 'resource'
    CRAWLER_CATEGORIES_DIR = Path(__file__).parent / 'crawler_categories'
    
    # 确保crawler_categories目录存在
    CRAWLER_CATEGORIES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 用于存储所有漫画名称的集合（使用集合自动去重）
    comic_names = set()
    
    # 遍历resource目录下的所有comic_mapping文件
    for file_path in RESOURCE_DIR.glob('comic_mapping_*.json'):
        try:
            print(f'处理文件: {file_path.name}')
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 处理每个分类下的漫画
                for category, comics in data.items():
                    for comic in comics:
                        if isinstance(comic, dict) and 'name' in comic:
                            comic_names.add(comic['name'])
        except Exception as e:
            print(f'处理文件 {file_path} 时出错: {str(e)}')
    
    # 将去重后的漫画名称写入comic.txt文件
    comic_txt_path = CRAWLER_CATEGORIES_DIR / 'comic.txt'
    with open(comic_txt_path, 'w', encoding='utf-8') as f:
        for name in sorted(comic_names):  # 排序以保持稳定的顺序
            f.write(f'{name}\n')
    
    print(f'已将 {len(comic_names)} 个唯一的漫画名称写入 {comic_txt_path}')
    
    # 调用封面下载函数
    print('开始下载漫画封面...')
    baidu_images_search_and_restore_for_covers()
    print('漫画封面下载完成')

if __name__ == '__main__':
    update_comic_crawler_categories()