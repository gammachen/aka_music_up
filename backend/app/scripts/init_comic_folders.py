import os
import json
from pathlib import Path

# 设置基础路径
BASE_DIR = Path(__file__).parent.parent
RESOURCE_DIR = BASE_DIR / 'resource'
STATIC_COMIC_DIR = BASE_DIR / 'static' / 'comic'

def create_comic_folders():
    # 确保static/comic目录存在
    os.makedirs(STATIC_COMIC_DIR, exist_ok=True)
    
    # 扫描resource目录下的comic_mapping文件
    for file_path in RESOURCE_DIR.glob('comic_mapping_*.json'):
        try:
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 处理每个漫画分类
            for category, comics in data.items():
                # 创建分类目录
                category_path = STATIC_COMIC_DIR / category
                os.makedirs(category_path, exist_ok=True)
                print(f'Created category directory: {category_path}')
                
                # 为每个漫画创建目录
                for comic in comics:
                    if isinstance(comic, dict) and 'name' in comic:
                        # comic_path = category_path / comic['name']
                        comic_path = STATIC_COMIC_DIR / comic['name'] # 不用带上分类名
                        os.makedirs(comic_path, exist_ok=True)
                        print(f'Created comic directory: {comic_path}')
                
        except json.JSONDecodeError as e:
            print(f'Error parsing {file_path}: {e}')
        except Exception as e:
            print(f'Error processing {file_path}: {e}')

if __name__ == '__main__':
    create_comic_folders()