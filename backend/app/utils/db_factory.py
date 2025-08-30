from flask_sqlalchemy import SQLAlchemy
from flask import current_app
import os

# 创建数据库实例
sqlite_db = SQLAlchemy()
postgres_db = SQLAlchemy()

# 数据库连接工厂类
class DBFactory:
    @staticmethod
    def get_db(db_type='sqlite'):
        """
        根据数据库类型返回相应的数据库实例
        :param db_type: 'sqlite' 或 'postgres'
        :return: 数据库实例
        """
        if db_type.lower() == 'postgres':
            return postgres_db
        else:
            return sqlite_db

    @staticmethod
    def init_app(app):
        """
        初始化所有数据库连接
        :param app: Flask应用实例
        """
        # 配置SQLite数据库
        sqlite_uri = app.config.get('SQLALCHEMY_DATABASE_URI')
        
        # 配置PostgreSQL数据库
        postgres_uri = app.config.get('POSTGRES_DATABASE_URI')
        if not postgres_uri:
            # 如果没有配置PostgreSQL连接，使用默认配置
            postgres_uri = 'postgresql://postgres:postgres468028475@localhost:5432/immich'
            app.config['POSTGRES_DATABASE_URI'] = postgres_uri
        
        # 初始化SQLite数据库
        sqlite_db.init_app(app)
        
        # 为PostgreSQL创建一个单独的配置键，避免与默认的SQLALCHEMY_DATABASE_URI冲突
        app.config['SQLALCHEMY_BINDS'] = {
            'postgres': postgres_uri
        }
        
        # 初始化PostgreSQL数据库，使用相同的app实例但不会冲突
        # postgres_db.init_app(app)
        
        return {'sqlite': sqlite_db, 'postgres': postgres_db}