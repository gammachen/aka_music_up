import os
from flask import current_app
from sqlalchemy import text, create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from ..utils.db_factory import DBFactory

def init_postgres():
    """
    初始化PostgreSQL数据库，启用pgvector扩展
    注意：数据库连接已在DBFactory.init_app中初始化，这里只执行pgvector扩展的启用
    """
    try:
        # 获取PostgreSQL连接URI
        postgres_uri = current_app.config.get('POSTGRES_DATABASE_URI')
        if not postgres_uri:
            # 这里不需要设置默认值，因为DBFactory.init_app已经处理了
            return
        
        # 直接创建数据库引擎，仅用于启用pgvector扩展
        engine = create_engine(postgres_uri)
        
        # 执行SQL脚本启用pgvector扩展
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.execute(text("ALTER EXTENSION vector ADD OPERATOR <=> (vector, vector);"))
            # Verify operator existence
            operator_check = conn.execute(text("SELECT oprname FROM pg_operator WHERE oprname = '<=>' AND oprleft = 'vector'::regtype AND oprright = 'vector'::regtype;")).scalar()
            if not operator_check:
                conn.execute(text("CREATE OPERATOR <=> (LEFTARG = vector, RIGHTARG = vector, PROCEDURE = vector_l2sq_operator)"))
            conn.commit()
            print("pgvector扩展已启用")
        
        # 获取PostgreSQL数据库实例
        postgres_db = DBFactory.get_db('postgres')
        
        # 创建必要的表结构
        # 使用with_bind方法指定使用postgres绑定
        postgres_db.create_all(bind='postgres')
        print("Vector comparison operators successfully verified/created")
        print("PostgreSQL数据库表结构已创建")
        
    except Exception as e:
        print(f"初始化PostgreSQL数据库时出错: {str(e)}")
        raise