from werkzeug.security import generate_password_hash
from datetime import datetime, timedelta
from ..models.user import db, User
from ..models.topic import Topic
from ..models.topic_comment import TopicComment
from ..models.category import Category
from .init_postgres import init_postgres
from ..utils.db_factory import DBFactory

def init_db():
    # 初始化PostgreSQL数据库和pgvector扩展
    # init_postgres()
    
    # 创建所有表
    db.create_all()

    # 检查是否已经初始化
    if User.query.first() is not None:
        return

    # 创建管理员用户
    admin = User(
        username='admin',
        email='admin@example.com',
        password_hash=generate_password_hash('admin123'),
        is_admin=True,
        score=1000,
        level=10,
        avatar='https://api.dicebear.com/7.x/avataaars/svg?seed=admin'
    )
    db.session.add(admin)

    # 创建测试用户
    test_user = User(
        username='test_user',
        email='test@example.com',
        password_hash=generate_password_hash('test123'),
        score=100,
        level=2,
        avatar='https://api.dicebear.com/7.x/avataaars/svg?seed=test'
    )
    db.session.add(test_user)

    # 提交用户数据
    db.session.commit()

    # 创建示例主题
    topics = [
        {
            'title': '推荐一首经典摇滚',
            'content': '推荐Queen的Bohemian Rhapsody，这是一首融合了摇滚、歌剧等多种元素的经典之作。',
            'category': 'music',
            'user_id': admin.id,
            'views': 100,
            'likes': 50,
            'favorites': 30
        },
        {
            'title': '关于新专辑的乐评',
            'content': '最近发行的这张专辑在制作上非常用心，编曲精良，歌词深刻。',
            'category': 'review',
            'user_id': test_user.id,
            'views': 80,
            'likes': 30,
            'favorites': 20
        }
    ]

    for topic_data in topics:
        topic = Topic(**topic_data)
        db.session.add(topic)

    db.session.commit()

    # 创建音乐分类
    categories = [
        {
            'name': '流行音乐',
            'level': 1,
            'sort_order': 1,
            'background_style': 'background: linear-gradient(45deg, #FF6B6B, #FFE66D)',
            'desc_image': 'https://example.com/images/pop-music.jpg'
        },
        {
            'name': '摇滚音乐',
            'level': 1,
            'sort_order': 2,
            'background_style': 'background: linear-gradient(45deg, #4A90E2, #50E3C2)',
            'desc_image': 'https://example.com/images/rock-music.jpg'
        }
    ]

    for category_data in categories:
        category = Category(**category_data)
        db.session.add(category)
        
        # 添加子分类
        if category.name == '流行音乐':
            subcategories = [
                {
                    'name': '华语流行',
                    'level': 2,
                    'parent_id': category.id,
                    'sort_order': 1,
                    'background_style': 'background: linear-gradient(45deg, #FF9A9E, #FAD0C4)',
                    'desc_image': 'https://example.com/images/chinese-pop.jpg'
                },
                {
                    'name': '欧美流行',
                    'level': 2,
                    'parent_id': category.id,
                    'sort_order': 2,
                    'background_style': 'background: linear-gradient(45deg, #A18CD1, #FBC2EB)',
                    'desc_image': 'https://example.com/images/western-pop.jpg'
                }
            ]
            for subcat_data in subcategories:
                subcat = Category(**subcat_data)
                db.session.add(subcat)
        
        elif category.name == '摇滚音乐':
            subcategories = [
                {
                    'name': '经典摇滚',
                    'level': 2,
                    'parent_id': category.id,
                    'sort_order': 1,
                    'background_style': 'background: linear-gradient(45deg, #6B8DD6, #8E37D7)',
                    'desc_image': 'https://example.com/images/classic-rock.jpg'
                },
                {
                    'name': '重金属',
                    'level': 2,
                    'parent_id': category.id,
                    'sort_order': 2,
                    'background_style': 'background: linear-gradient(45deg, #434343, #000000)',
                    'desc_image': 'https://example.com/images/heavy-metal.jpg'
                }
            ]
            for subcat_data in subcategories:
                subcat = Category(**subcat_data)
                db.session.add(subcat)
    
    db.session.commit()

    # 创建示例评论
    comments = [
        {
            'content': '完全同意你的观点，这确实是一首经典！',
            'user_id': test_user.id,
            'topic_id': 1,
            'likes': 10
        },
        {
            'content': '分析得很到位，期待更多优质乐评。',
            'user_id': admin.id,
            'topic_id': 2,
            'likes': 8
        }
    ]

    for comment_data in comments:
        comment = TopicComment(**comment_data)
        db.session.add(comment)

    db.session.commit()