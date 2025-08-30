from app.models.base import Base
from app.models.content import Content, Chapter
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

def init_db():
    # 获取数据库文件路径
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'instance', 'aka_music.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # 创建数据库引擎
    engine = create_engine(f'sqlite:///{db_path}')
    
    # 创建所有表
    Base.metadata.create_all(engine)
    
    # 创建会话
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # 这里可以添加一些初始数据
        pass
        
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

if __name__ == '__main__':
    init_db()
    print('数据库初始化完成')