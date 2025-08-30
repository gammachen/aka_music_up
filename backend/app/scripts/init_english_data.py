import os
import sys
import json
from pathlib import Path
from datetime import datetime
from PyPDF2 import PdfReader
from PIL import Image
import fitz  # PyMuPDF

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from app import create_app
from app.services.content_service import ContentService, ChapterService
from app.utils.pdf_utils import convert_pdf_to_images

# 创建Flask应用实例
app = create_app()

def load_comic_mappings(resource_dir):
    """加载所有comic_mapping开头的JSON文件内容"""
    comic_data = []
    for file in os.listdir(resource_dir):
        if file.startswith('comic_mapping_') and file.endswith('.json'):
            with open(os.path.join(resource_dir, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 处理每个分类下的漫画列表
                for category, comics in data.items():
                    for comic in comics:
                        comic['category'] = category
                        comic_data.append(comic)
    return comic_data

def scan_comic_chapters(comic_dir, comic_name):
    """扫描漫画章节目录结构"""
    comic_path = os.path.join(comic_dir, comic_name)
    chapters = []
    
    if not os.path.exists(comic_path):
        print(f"警告: 未找到漫画 '{comic_name}' 的目录")
        return chapters
    
    # 如果发现目录下有后缀是pdf文件的章节，将其进行转成图片的形式，并且创建对应的章节目录
    for chapter_name in sorted(os.listdir(comic_path)):
        chapter_path = os.path.join(comic_path, chapter_name)
        if chapter_path.lower().endswith('.pdf'):
            print(f"警告: 漫画 '{comic_name}' 的章节 '{chapter_name}' 是pdf文件，将其进行转成图片的形式，并且创建对应的章节目录")
            
            # 创建章节目录（使用PDF文件名，去掉.pdf后缀）
            chapter_dir = os.path.join(comic_path, os.path.splitext(chapter_name)[0])
            os.makedirs(chapter_dir, exist_ok=True)
            
            # 转换PDF为图片
            convert_pdf_to_images(chapter_path, chapter_dir)
            
            # 转换完成后删除PDF文件
            # os.remove(chapter_path)
            
            print(f"成功将PDF文件 '{chapter_name}' 转换为图片并保存到目录 '{chapter_dir}'")
            continue
    
    # 遍历漫画目录下的所有章节文件夹
    for chapter_name in sorted(os.listdir(comic_path)):
        chapter_path = os.path.join(comic_path, chapter_name)
        if os.path.isdir(chapter_path):
            contents = []
            # 遍历章节目录下的所有图片
            for content in sorted(os.listdir(chapter_path)):
                # 处理图片类型的数据
                if content.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    # 构建相对URL路径
                    image_url = f"/static/comic/{comic_name}/{chapter_name}/{content}"
                    contents.append({
                        'url': image_url,
                        'order': len(contents) + 1
                    })
                # 处理pdf类型的数据（暂时不这样处理，依赖上面将pdf转图片的步骤，还是图片处理的比较合适）
                # if content.lower().endswith(('.pdf')):
                #     # 构建相对URL路径
                #     image_url = f"/static/comic/{comic_name}/{chapter_name}/{content}"
                #     contents.append({
                #         'url': image_url,
                #         'order': len(contents) + 1
                #     })
            if contents:
                chapters.append({
                    'name': chapter_name,
                    'order': len(chapters) + 1,
                    'images': contents
                })
    
    return chapters

def init_comic_data():
    """初始化漫画数据 注意这里面并不是非常严谨"""
    # 定义相关目录路径
    base_dir = Path(__file__).parent.parent
    resource_dir = base_dir / 'resource'
    comic_dir = base_dir / 'static' / 'comic'
    
    # 加载所有漫画映射数据
    comics = load_comic_mappings(resource_dir)
    
    # 在应用上下文中执行数据库操作
    with app.app_context():
        # 处理每个漫画的数据
        for comic in comics:
            comic_name = comic['name']
            print(f"正在处理漫画: {comic_name}")
            
            contents = ContentService.get_contents_by_name_and_type(name=comic_name, content_type='COMIC')
            if contents:
                print(f"警告: 漫画 '{comic_name}' 的内容记录已存在，跳过创建")
                continue
            else:
                print(f"漫画 '{comic_name}' 的内容记录不存在，开始创建")
                # 创建漫画内容记录
                content = ContentService.create_comic_content(
                    title=comic_name, # comic['title'] TODO 暂时使用name作为title，后续json文件中补充了title的信息之后再将其补充完整，理论上这个title与description重复了
                    name=comic_name,
                    content_type='COMIC',
                    author_id=comic['author'],
                    description=f"作者: {comic['author']}\n发布日期: {comic['publish_date']}\n状态: {'已完结' if comic['is_completed'] else '连载中'}\n国家: {comic['country']}\n标签: {', '.join(comic['tags'])}",
                    publish_date=datetime.strptime(comic['publish_date'], '%Y-%m-%d') if len(comic['publish_date']) == 10 else datetime.strptime(f"{comic['publish_date']}-01-01", '%Y-%m-%d'),
                    price_strategy='FREE',
                    status='PUBLISHED' if comic['is_completed'] else 'DRAFT'
                )
            
            # 扫描并创建章节记录
            chapters = scan_comic_chapters(comic_dir, comic_name)
            
            ## TODO 这里面并没有检查是否已经创建过该章节，后续需要优化，所以如果重复初始化的话，可能导致章节的数据重复
            ## 严谨的逻辑在import函数中，请使用import_comic_data函数
            for chapter_data in chapters:
                ChapterService.create_chapter(
                    content_id=content.id,
                    chapter_no=chapter_data['order'],
                    title=chapter_data['name'],
                    pages={'images': chapter_data['images']},
                    is_free=True,
                    unlock_type='FREE'
                )
        
        print(f"\n数据初始化完成！")
        print(f"总共处理了 {len(comics)} 部漫画")

def import_comic_data(comic_name):
    """
    导入指定漫画的数据
    
    args:
        comic_name: 漫画名称，必须放置在static/comic目录下才会被扫描到（章节信息将回被排序的）
    
    returns:
        无返回值，直接在控制台输出结果
    """
    
    # 定义相关目录路径
    base_dir = Path(__file__).parent.parent
    resource_dir = base_dir /'resource'
    comic_dir = base_dir /'static' / 'comic'

    # 在应用上下文中执行数据库操作
    with app.app_context():
        contents = ContentService.get_contents_by_name_and_type(name=comic_name, content_type='COMIC')
        if not contents:
            print(f"警告: 未找到漫画 '{comic_name}' 的内容记录")
            return
        
        content_id = contents[0].id
        # 扫描并创建章节记录
        chapters = scan_comic_chapters(comic_dir, comic_name)
        for chapter_data in chapters:
            # 先检查，避免重复创建
            if ChapterService.get_chapters_by_content_and_title(content_id, chapter_data['name']):
                print(f"警告: 漫画 '{comic_name}' 的章节 '{chapter_data['name']}' 已存在，跳过创建")
                continue
            
            chapter = ChapterService.create_chapter(
                content_id=content_id,
                chapter_no=chapter_data['order'],
                title=chapter_data['name'],
                pages={'images': chapter_data['images']},
                is_free=True,
                unlock_type='FREE'
            )
            
            print(f"成功创建漫画 '{comic_name}' 的章节 '{chapter}'")

if __name__ == '__main__':
    # init_comic_data()
    # import_comic_data('海贼王')
    # import_comic_data('灌篮高手')
    # import_comic_data('银魂')
    import_comic_data('杀戮都市')