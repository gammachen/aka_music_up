from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from ..utils.db_factory import DBFactory, sqlite_db

# 保持向后兼容性，使用SQLite数据库实例
db = sqlite_db

class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    phone = db.Column(db.String(20), unique=True, nullable=True)
    password_hash = db.Column(db.String(128), nullable=False)
    avatar = db.Column(db.String(200))
    is_admin = db.Column(db.Boolean, default=False)
    score = db.Column(db.Integer, default=0)  # 用户积分
    level = db.Column(db.Integer, default=1)  # 用户等级
    gold_balance = db.Column(db.Integer, default=0)  # 用户金币余额
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<User {self.username}>'

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'phone': self.phone,
            'avatar': self.avatar,
            'is_admin': self.is_admin,
            'score': self.score,
            'level': self.level,
            'gold_balance': self.gold_balance,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S')
        }