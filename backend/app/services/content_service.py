from typing import List, Optional
from datetime import datetime
import uuid

from app.models.content import Content, Chapter
from app.models.user import db

from app.models.user import User
from app.models.topic import Topic
from app.models.payment import GoldTransaction
import logging
from sqlalchemy import text
from sqlalchemy.orm.attributes import flag_modified

import json

class ContentService:
    @staticmethod
    def get_content_by_id(content_id: str) -> Optional[Content]:
        """根据ID获取内容"""
        return Content.query.get(content_id)
    
    @staticmethod
    def get_contents_by_author(author_id: str) -> List[Content]:
        """获取作者的所有内容"""
        return Content.query.filter_by(author_id=author_id).all()
    
    @staticmethod
    def get_contents_by_type(content_type: str) -> List[Content]:
        """根据类型获取内容列表"""
        return Content.query.filter_by(type=content_type).all()
    
    @staticmethod
    def get_contents_by_name_and_type(name: str, content_type: str) -> List[Content]:
        """获取名字与类型相关的所有内容"""
        return Content.query.filter_by(name=name).filter_by(type=content_type).all() # TODO 可能这里面拥有对content_type作一个大小写的处理，外部传递进来的可能是小写，所以这里可能是一个小写的，要转成大写
    
    @staticmethod
    def create_content(title: str, content_type: str, author_id: str, 
                      name: Optional[str] = None,
                      cover_url: Optional[str] = None, 
                      description: Optional[str] = None,
                      publish_date: Optional[datetime] = None,
                      price_strategy: str = 'FREE',
                      status: str='DRAFT') -> Content:
        
        # 对cover_url作兜底的处理
        if not cover_url:
            # 查询static/cover/{name}目录下的是否有.jpg文件，如果有则随机选一张.jpg的文件为cover_url
            import os
            import random
            cover_dir = f'static/covers/{content_type.lower()}/{name}'
            if os.path.exists(cover_dir) and os.path.isdir(cover_dir):
                jpg_files = [f for f in os.listdir(cover_dir) if f.endswith('.jpg')]
                if jpg_files:
                    cover_url = f'/{cover_dir}/{random.choice(jpg_files)}'
            cover_url = '/static/covers/default.jpg'
        
        """创建新内容"""
        content = Content(
            id=str(uuid.uuid4()),
            title=title,
            name=name or title[:30],  # 如果未提供name，则使用title的前30个字符
            type=content_type,
            author_id=author_id,
            cover_url=cover_url,
            description=description,
            publish_date=publish_date,
            status=status,
            price_strategy=price_strategy
        )
        db.session.add(content)
        db.session.commit()
        return content
    
    @staticmethod
    def create_comic_content(title: str, content_type: str, author_id: str, 
                      name: Optional[str] = None,
                      cover_url: Optional[str] = None, 
                      description: Optional[str] = None,
                      publish_date: Optional[datetime] = None,
                      price_strategy: str = 'FREE',
                      status: str='DRAFT') -> Content:
        
        # 对cover_url作兜底的处理
        if not cover_url:
            # 查询static/cover/{name}目录下的是否有.jpg文件，如果有则随机选一张.jpg的文件为cover_url
            import os
            import random
            cover_dir = f'static/covers/comic/{name}'
            if os.path.exists(cover_dir) and os.path.isdir(cover_dir):
                jpg_files = [f for f in os.listdir(cover_dir) if f.endswith('.jpg')]
                if jpg_files:
                    cover_url = f'/{cover_dir}/{random.choice(jpg_files)}'
            cover_url = '/static/covers/default.jpg'
        
        """创建新内容"""
        content = Content(
            id=str(uuid.uuid4()),
            title=title,
            name=name or title[:30],  # 如果未提供name，则使用title的前30个字符
            type=content_type,
            author_id=author_id,
            cover_url=cover_url,
            description=description,
            publish_date=publish_date,
            status=status,
            price_strategy=price_strategy
        )
        db.session.add(content)
        db.session.commit()
        return content
    
    @staticmethod
    def update_content_status(content_id: str, status: str) -> Optional[Content]:
        """更新内容状态"""
        content = Content.query.get(content_id)
        if content:
            content.status = status
            content.updated_at = datetime.utcnow()
            db.session.commit()
        return content
    
    @staticmethod
    def update_content(content_id: str, **kwargs) -> Optional[Content]:
        """更新内容信息"""
        content = Content.query.get(content_id)
        if content:
            for key, value in kwargs.items():
                if hasattr(content, key):
                    setattr(content, key, value)
            content.updated_at = datetime.utcnow()
            db.session.commit()
        return content

class ChapterService:
    @staticmethod
    def get_chapter_by_id(chapter_id: str) -> Optional[Chapter]:
        """根据ID获取章节"""
        return Chapter.query.get(chapter_id)
    
    @staticmethod
    def get_chapters_by_content(content_id: str) -> List[Chapter]:
        """获取内容的所有章节"""
        return Chapter.query.filter_by(content_id=content_id).order_by(Chapter.chapter_no).all()
    
    @staticmethod
    def get_chapters_by_content_and_title(content_id: str, title: str) -> Optional[Chapter]:
        """根据title过滤并获取内容的所有章节"""
        return Chapter.query.filter_by(content_id=content_id).filter_by(title=title).all()
    
    @staticmethod
    def create_chapter(content_id: str, chapter_no: int, title: str, 
                      pages: dict, price: float = 0.0, 
                      is_free: bool = False,
                      unlock_type: str = 'FREE') -> Chapter:
        """创建新章节"""
        logging.info(f'开始创建新章节 - 内容ID: {content_id}, 章节号: {chapter_no}, 标题: {title}')
        logging.info(f'章节参数 - 页面数据: {json.dumps(pages)}, 价格: {price}, 是否免费: {is_free}, 解锁类型: {unlock_type}')
        
        try:
            chapter_id = str(uuid.uuid4())
            chapter = Chapter(
                id=chapter_id,
                content_id=content_id,
                chapter_no=chapter_no,
                title=title,
                pages=pages,  # 直接存储dict对象，不需要json.dumps
                price=price,
                is_free=is_free,
                unlock_type=unlock_type
            )
            logging.info(f'创建章节对象成功 - 章节ID: {chapter_id}')
            
            db.session.add(chapter)
            db.session.flush()
            logging.info(f'章节数据已添加到会话 - 章节ID: {chapter_id}')
            
            db.session.commit()
            logging.info(f'章节创建成功 - 章节ID: {chapter_id}, 标题: {title}')
            logging.info(f'章节详细信息: {json.dumps(chapter.to_dict(), indent=2)}')
        except Exception as e:
            db.session.rollback()
            logging.error(f'章节创建失败 - 错误信息: {str(e)}')
            raise e
        return chapter
    
    @staticmethod
    def update_chapter(chapter_id: str, **kwargs) -> Optional[Chapter]:
        """更新章节信息"""
        logging.info(f'开始更新章节信息 - 章节ID: {chapter_id}')
        logging.info(f'更新参数: {json.dumps(kwargs, indent=2)}')
        
        chapter = Chapter.query.get(chapter_id)
        return ChapterService.update_chapter_obj(chapter, **kwargs)
    
    @staticmethod
    def update_chapter_obj(chapter: Chapter, **kwargs) -> Optional[Chapter]:
        if chapter:
            logging.info(f'找到章节记录，原始数据: {json.dumps(chapter.to_dict(), indent=2)}')
            
            for key, value in kwargs.items():
                if hasattr(chapter, key):
                    logging.info(f'更新字段 {key}: {value}')
                    if key == 'pages':
                        # 确保pages是dict类型
                        if isinstance(value, str):
                            try:
                                value = json.loads(value)
                            except json.JSONDecodeError:
                                logging.error(f'无效的pages JSON数据: {value}')
                                continue
                        
                        if not isinstance(value, dict):
                            logging.error(f'pages必须是dict类型: {value}')
                            continue
                        
                        # 更新pages字段
                        chapter.pages = value
                        flag_modified(chapter, 'pages')
                        logging.info(f'更新后的pages数据: {value}')
                    else:
                        setattr(chapter, key, value)
            
            logging.info(f'数据库更新成功，待更新后的数据: {json.dumps(chapter.to_dict(), indent=2)}')
            
            # 直接使用datetime对象更新时间
            chapter.updated_at = datetime.now()
            
            # chapter.pages = pages

            logging.info(f'pages: {chapter.pages}')
            try:
                # 开启SQL语句日志记录
                # from sqlalchemy import event
                # @event.listens_for(db.engine, 'before_cursor_execute')
                # def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
                #     logging.info(f'执行的SQL语句: {statement}')
                #     logging.info(f'SQL参数: {parameters}')

                db.session.commit()
                # 直接使用已更新的对象，不需要重新查询
                logging.info(f'数据库更新成功，更新后的数据: {json.dumps(chapter.to_dict(), indent=2)}')
            except Exception as e:
                db.session.rollback()
                logging.error(f'数据库更新失败: {str(e)}')
                raise e
        else:
            logging.error(f'未找到章节记录')
        
        return chapter
    
    @staticmethod
    def delete_chapter(chapter_id: str) -> bool:
        """删除章节"""
        chapter = Chapter.query.get(chapter_id)
        if chapter:
            db.session.delete(chapter)
            db.session.commit()
            return True
        return False
    
    @staticmethod
    def check_order_chapter_status(content_id: str, user_id: int) -> bool:
        """
        检查章节购买记录是否存在
        """
        logging.info(f'检查章节购买记录 - 用户ID: {user_id}, 内容ID: {content_id}')
        
        try:
            # 使用具体的查询条件
            transactions = GoldTransaction.query.filter_by(
                user_id=user_id,
                reference_id=content_id,
                reference_type='ContentChapter',
                transaction_type='ChapterConsume'
            ).first()
            
            # valid_transactions = [t for t in transactions if t is not None]
            # logging.info(f'查询到的有效记录数: {len(valid_transactions)}')
            
            # 检查数据库连接状态
            # if not db.session.is_active:
            #     logging.error('数据库会话未激活')
            #     return False
            
            # 返回是否存在有效记录
            # return len(valid_transactions) > 0
            return transactions is not None
            
        except Exception as e:
            logging.error(f'查询过程发生异常: {str(e)}')
            return False
    
    @staticmethod
    def order_chapter(content_id: str, user_id: int, amount: int, message: str = '') -> dict:
        """
        创建章节购买记录
        
        Args:
            content_id: 章节ID
            user_id: 用户ID
            amount: 金币数量
            message: 购买留言
            
        Returns:
            dict: 购买记录信息
            
        Raises:
            ValueError: 参数验证失败
            Exception: 数据库操作失败
        """
        # 打印输入参数
        logging.info(f'章节购买请求参数 - 章节ID: {content_id}, 用户ID: {user_id}, 金币数量: {amount}, 留言: {message}')
        
        try:
            # 检查用户是否有足够的金币
            user = User.query.get(user_id)
            if not user:
                logging.error(f'用户不存在 - 用户ID: {user_id}')
                raise ValueError('用户不存在')
                
            if user.gold_balance < amount:
                logging.error(f'用户金币不足 - 用户ID: {user_id}, 当前余额: {user.gold_balance}, 需要金币: {amount}')
                raise ValueError('用户金币不足')
                
            # 检查内容是否存在
            content = Chapter.query.get(content_id)
            if not content:
                logging.error(f'内容不存在 - 内容ID: {content_id}')
                raise ValueError('内容不存在')

            # 检查是否重复购买
            record = ChapterService.check_order_chapter_status(content_id, user_id)
            if record:
                logging.error(f'用户已经购买过 - 用户ID: {user_id}, 内容ID: {content_id}')
                raise ValueError('用户已经购买过')
            
            # 生成订单号
            order_no = f'CH{datetime.now().strftime("%Y%m%d%H%M%S")}{uuid.uuid4().hex[:8]}'
            logging.info(f'生成章节购买订单 - 订单号: {order_no}')
            
            # 开始事务处理
            try:
                # 扣减用户金币
                user.gold_balance -= amount
                db.session.flush()  # 确保金币扣减成功
                
                # 创建购买记录
                order = GoldTransaction(
                    user_id=user_id,
                    order_no=order_no,
                    reference_id=content_id,
                    reference_type='ContentChapter',
                    amount=-amount,  # 使用负数表示支出
                    order_status='已完成',
                    transaction_type='ChapterConsume',
                    created_at=datetime.now(),
                    # updated_at=datetime.utcnow() # TODO 缺失了这个字段，将来补充
                )
                
                db.session.add(order)
                db.session.commit()
                
                logging.info(f'章节购买成功 - 订单号: {order_no}, 用户ID: {user_id}, 内容ID: {content_id}, 金币: {amount}')
                
                return {
                    'order_no': order_no,
                    'status': '已完成',
                    'amount': amount
                }
                
            except Exception as e:
                db.session.rollback()
                logging.error(f'章节购买失败 - 订单号: {order_no}, 错误信息: {str(e)}')
                raise Exception('订单处理失败')
                
        except ValueError as ve:
            raise ve
        except Exception as e:
            logging.error(f'章节购买异常 - 错误信息: {str(e)}')
            raise Exception('系统异常，请稍后重试')
        
