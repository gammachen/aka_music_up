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

bp = Blueprint('comic', __name__, url_prefix='/api/comic')

        # 设置基础路径
BASE_DIR = Path(__file__).parent.parent
RESOURCE_DIR = BASE_DIR / 'resource'
STATIC_COMIC_DIR = BASE_DIR / 'static' / 'comic'
STATIC_COMIC_COVERS_DIR = BASE_DIR / 'static' / 'covers' / 'comic'

@bp.route('/all_comic_genre', methods=['GET'])
def get_all_comic_genre():
    """
    获取所有漫画的分类
    返回：
    - 所有漫画的分类列表
    """
    try:
        # 获取所有漫画的分类（直接透出的话是没有分类信息的，因为分类信息没有落到Coneten中的）
        # 有两种做法能够满足这个目录或者content所属的分类到底是热血漫画还是冷门漫画：
        # 1. 在Content模型中增加扩展属性的方式，或者额外的增加Tag这种（可以作成这样的，就是实施成本会高一点，要多一套体系）
        # 2. 使用Category内部的映射，但是这个映射可能不是那么通用，涉及到要作三层的映射，并且还需要一些硬编码或者使用名称作映射，将来可能不能简单的修改名称等问题，需要将refer_id映射到Content中的id
        # 3. 使用映射Category与Content之间的映射
        # 其实就是加一层映射，这个映射使用配置文件作还是说使用数据库表（使用Category还是增加一层CategoryContent映射，就是两种实现方式）
        
        # 当前我们使用resource中的comic_mapping_x.json 来获取所有漫画的分类
        '''
        from app.models.content import Content
        from sqlalchemy import desc
        
        comics = Content.query.filter_by(type='COMIC')\
            .order_by(desc(Content.updated_at))\
            .all()
        
        # 将查询结果转换为字典列表
        all_comic_genres = [comic.to_dict() for comic in comics]
        
        return api_response(message='获取所有漫画的分类成功', data=all_comic_genres)
        '''
        
        # 存储所有漫画分类数据的字典
        genres_dict = {}
        
        # 统计使用默认封面的漫画数量
        default_cover_count = 0
        total_comics_count = 0
        
        # 扫描resource目录下的comic_mapping文件
        for file_path in RESOURCE_DIR.glob('comic_mapping_*.json'):
            try:
                # 读取JSON文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 处理每个漫画分类
                for category, comics in data.items():
                    # 如果分类不存在于字典中，则初始化为空列表
                    if category not in genres_dict:
                        genres_dict[category] = []
                    
                    # 将漫画添加到对应分类下
                    for comic in comics:
                        if isinstance(comic, dict) and 'name' in comic:
                            # 检查是否有封面URL，如果没有或者是默认封面，尝试从static/covers/comic目录获取
                            comic_name = comic['name']
                            total_comics_count += 1
                            
                            # 检查是否已有非默认封面
                            has_custom_cover = ('cover_url' in comic and 
                                               comic['cover_url'] and 
                                               not comic['cover_url'].endswith('default_cover.jpg'))
                            
                            if not has_custom_cover:
                                # 检查是否存在对应的封面目录
                                comic_cover_dir = STATIC_COMIC_COVERS_DIR / comic_name
                                if comic_cover_dir.exists() and comic_cover_dir.is_dir():
                                    # 获取目录下所有jpg和png文件
                                    cover_images = list(comic_cover_dir.glob('*.jpg')) + list(comic_cover_dir.glob('*.png'))
                                    if cover_images:
                                        # 随机选择一张图片作为封面
                                        cover_image = random.choice(cover_images)
                                        relative_path = f"/static/covers/comic/{comic_name}/{cover_image.name}"
                                        comic['cover_url'] = relative_path
                                    else:
                                        # 目录存在但没有图片，使用默认封面
                                        comic['cover_url'] = "/static/covers/comic/default_cover.jpg"
                                        default_cover_count += 1
                                else:
                                    # 目录不存在，使用默认封面
                                    comic['cover_url'] = "/static/covers/comic/default_cover.jpg"
                                    default_cover_count += 1
                            elif comic['cover_url'].endswith('default_cover.jpg'):
                                default_cover_count += 1
                                
                            # 不需要在每个漫画对象中添加category字段，因为现在已经按分类组织了
                            genres_dict[category].append(comic)
        
            except Exception as e:
                print(f'Error processing {file_path}: {e}')
        
        # 检查默认封面的比例，如果超过50%，输出警告日志
        if total_comics_count > 0:
            default_cover_ratio = default_cover_count / total_comics_count
            if default_cover_ratio > 0.5:  # 如果超过50%使用默认封面
                logging.warning(f"警告：有{default_cover_count}/{total_comics_count}({default_cover_ratio:.2%})的漫画使用默认封面，请管理员更新封面资源！")
        
        # 将字典转换为前端需要的格式：[{category: '分类名', comics: [漫画列表]}]
        formatted_genres = []
        for category, comics in genres_dict.items():
            formatted_genres.append({
                'category': category,
                'comics': comics
            })
        
        return api_response(message='获取所有漫画的分类成功', data=formatted_genres)
        
    except Exception as e:
        return api_response(message=f'获取漫画分类失败: {str(e)}', code=500)

@bp.route('/contentDetail', methods=['GET'])
def get_content_detail():
    '''
    获取内容详情

    返回：
    - 内容详情信息
    '''
    try:
        # 获取请求参数
        content_id = request.args.get('id', type=str)

        if not content_id:
            return api_response(message='缺少内容ID参数', code=400)
        
        # 这里的content_id可能是content中的name（特别是漫画的Category导航的逻辑中，将漫画的名称作为content name，并且传递到这里面来作为导航项的，所以这里必须可能的漫画名称作一个特殊的处理）
        # 设计的Content的id是通过uuid生成的，如果能够判断是uuid，就直接使用get_content_by_id查询，如果非uuid，就使用get_content_by_name_and_type查询    
        
        content = ContentService.get_content_by_id(content_id)
        
        if not content:
            # 尝试使用name查询
            contents = ContentService.get_contents_by_name_and_type(content_id, 'COMIC')
            
            if not contents and len(contents) == 0:    
                return api_response(message='内容不存在', code=404)
            
            content = contents[0]
            
        # 构建响应数据
        response_data = {
            'id': content.id,
            'title': content.title,
            'name': content.name,
            'type': content.type,
            'author_id': content.author_id,
            'cover_url': content.cover_url,
            'description': content.description,
            'publish_date': content.publish_date,
            'status': content.status,
            'price_strategy': content.price_strategy
        }
        return api_response(data=response_data, message='获取内容详情成功')
    except Exception as e:
        return api_response(message=f'获取内容详情失败: {str(e)}', code=500)

@bp.route('/chapterList', methods=['GET'])
def get_content_list():
    """
    获取某个分类下的内容列表
    
    返回：
    - 内容列表信息
    - 分页信息（预留）
    """
    try:
        # 获取请求参数
        content_id = request.args.get('id', type=str)
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 10, type=int)
        
        if not content_id:
            return api_response(message='缺少分类ID参数', code=400)
        
        content = ContentService.get_content_by_id(content_id)
        
        if not content:
            # 尝试使用name查询
            contents = ContentService.get_contents_by_name_and_type(content_id, 'COMIC')
            
            if not contents and len(contents) == 0:    
                return api_response(message='内容已经下架！请浏览其他内容，谢谢！', code=404)
            
            content = contents[0]
            
        if not content:
            return api_response(message='内容已经下架！请浏览其他内容，谢谢！', code=404)
        
        content_id = content.id
        # 获取章节列表
        chapters = ChapterService.get_chapters_by_content(content_id)
        if not chapters:
            chapters = []
        else:
            # 将Chapter对象转换为字典格式
            chapters = [chapter.to_dict() for chapter in chapters]
            
        # 预留分页信息
        total = len(chapters)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        
        return api_response(
            message='获取内容列表成功',
            data={
                'list': chapters,
                'pagination': {
                    'total': total,
                    'page': page,
                    'page_size': page_size,
                    'total_pages': total_pages
                }
            }
        )
        
    except Exception as e:
        return api_response(message=f'获取内容列表失败: {str(e)}', code=500)

@bp.route('/chapterDetail', methods=['GET'])
def get_chapter_detail():
    """
    获取某个内容的章节详情(Important, 不做登录验证，这个接口就是给非登录用户使用的，但是内部会做强校验，验证这个章节是否是free的，避免free的章节也要用户登录的场景发生，虽然分开了两个接口，但是实现会比较容易，交互会比较友好)
    
    返回：
    - 章节详情信息
    """
    try:
        # 获取请求参数
        chapter_id = request.args.get('id', type=str)
        
        if not chapter_id:
            return api_response(message='缺少ChapterID参数', code=400)
        
        # 获取章节详情
        chapter = ChapterService.get_chapter_by_id(chapter_id)
        if not chapter:
            return api_response(message='章节不存在', code=404)
        
        print(chapter.content.status)
        print(chapter.is_free)
        
        if chapter.is_free == False:
            return api_response(message='章节不是免费的，需要登录', code=401)
            
        # 将Chapter对象转换为字典格式
        chapter_data = chapter.to_dict()
            
        return api_response(
            message='获取章节详情成功',
            data=chapter_data
        )
        
    except Exception as e:
        return api_response(message=f'获取章节详情失败: {str(e)}', code=500)
           
@bp.route('/chapterDetailForLogin', methods=['GET'])
@token_required
def get_chapter_detail_for_login(crrent_user):
    """
    获取某个内容的章节详情
    
    返回：
    - 章节详情信息
    """
    try:
        # 获取请求参数
        chapter_id = request.args.get('id', type=str)
        
        if not chapter_id:
            return api_response(message='缺少ChapterID参数', code=400)
        
        # 检查用户是否购买了该章节的内容
        ordered = ChapterService.check_order_chapter_status(chapter_id, crrent_user.id)
        print(ordered)
        if not ordered:
            return api_response(message='用户没有购买该章节的内容，请先购买！', code=1101) # 1101 code 特殊定义的code
        
        # 获取章节详情
        chapter = ChapterService.get_chapter_by_id(chapter_id)
        if not chapter:
            return api_response(message='章节不存在', code=404)
            
        # 将Chapter对象转换为字典格式
        chapter_data = chapter.to_dict()
            
        return api_response(
            message='获取章节详情成功',
            data=chapter_data
        )
        
    except Exception as e:
        return api_response(message=f'获取章节详情失败: {str(e)}', code=500)

@bp.route('/orderChapter', methods=['POST'])
@token_required
def order_chapter(current_user):
    """
    购买章节

    返回：
    - 购买结果
    """
    try:
        # 获取请求参数
        chapter_id = request.json.get('chapter_id')

        if not chapter_id:
            return api_response(message='缺少ChapterID参数', code=400)    
        
        chapter = ChapterService.get_chapter_by_id(chapter_id)
        if not chapter:
            return api_response(message='章节不存在', code=404)
            
        # 检查用户是否购买了该章节的内容
        order = ChapterService.order_chapter(chapter_id, current_user.id, amount=chapter.price)
        
        return api_response(
            message='购买章节成功',
            data={
                'order_no': order['order_no']
            }
        )

    except Exception as e:
        return api_response(message=f'购买章节失败: {str(e)}', code=500)