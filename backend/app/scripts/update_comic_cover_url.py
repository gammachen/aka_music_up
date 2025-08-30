import os
import json
import random
from pathlib import Path

# 设置基础路径
BASE_DIR = Path(__file__).parent.parent
RESOURCE_DIR = BASE_DIR / 'resource'
STATIC_COMIC_COVERS_DIR = BASE_DIR / 'static' / 'covers' / 'comic'

def update_comic_cover_urls():
    """更新comic_mapping_*.json文件中的漫画对象，添加cover_url属性 TODO 这个脚本其实可以由抓取封面脚本之后进行触发等来更新的， 作一些消息更新机制的 """
    print(f"开始更新漫画封面URL...")
    
    # 获取所有可用的漫画封面目录
    available_covers = {}
    if STATIC_COMIC_COVERS_DIR.exists():
        for comic_dir in STATIC_COMIC_COVERS_DIR.iterdir():
            if comic_dir.is_dir():
                comic_name = comic_dir.name
                cover_images = list(comic_dir.glob('*.jpg')) + list(comic_dir.glob('*.png'))
                if cover_images:
                    available_covers[comic_name] = cover_images
    
    print(f"找到 {len(available_covers)} 个漫画有封面图片")
    
    # 扫描resource目录下的comic_mapping文件
    updated_files_count = 0
    updated_comics_count = 0
    
    for file_path in RESOURCE_DIR.glob('comic_mapping_*.json'):
        try:
            print(f"处理文件: {file_path.name}")
            
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_updated = False
            
            # 处理每个漫画分类
            for category, comics in data.items():
                for i, comic in enumerate(comics):
                    if isinstance(comic, dict) and 'name' in comic:
                        comic_name = comic['name']
                        
                        # 如果漫画已经有cover_url属性且不为空，则跳过
                        if 'cover_url' in comic and comic['cover_url'] and comic['cover_url'] != "/static/covers/comic/default_cover.jpg":
                            continue
                        
                        # 检查是否有对应的封面图片
                        if comic_name in available_covers:
                            # 随机选择一张图片作为封面
                            cover_image = random.choice(available_covers[comic_name])
                            relative_path = f"/static/covers/comic/{comic_name}/{cover_image.name}"
                            
                            # 添加cover_url属性
                            comic['cover_url'] = relative_path
                            file_updated = True
                            updated_comics_count += 1
                            print(f"  - 为 '{comic_name}' 添加封面: {relative_path}")
                        else:
                            # 如果没有找到封面，设置一个默认的封面URL
                            comic['cover_url'] = "/static/covers/comic/default_cover.jpg"
                            file_updated = True
                            updated_comics_count += 1
                            print(f"  - 为 '{comic_name}' 添加默认封面")
            
            # 如果文件有更新，写回文件
            if file_updated:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                updated_files_count += 1
                print(f"  文件 {file_path.name} 已更新")
            else:
                print(f"  文件 {file_path.name} 无需更新")
                
        except Exception as e:
            print(f"处理 {file_path} 时出错: {e}")
    
    print(f"更新完成! 共更新了 {updated_files_count} 个文件中的 {updated_comics_count} 个漫画封面URL")

if __name__ == '__main__':
    update_comic_cover_urls()