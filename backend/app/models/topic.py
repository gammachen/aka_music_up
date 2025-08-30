from datetime import datetime
from .user import db

class Topic(db.Model):
    __tablename__ = 'topics'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    hidden_content = db.Column(db.Text)  # 隐藏内容
    category = db.Column(db.String(50), nullable=False)  # 分类：music, review, discussion
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    views = db.Column(db.Integer, default=0)  # 浏览量
    likes = db.Column(db.Integer, default=0)  # 点赞数
    favorites = db.Column(db.Integer, default=0)  # 收藏数
    comments_count = db.Column(db.Integer, default=0)  # 评论数
    status = db.Column(db.String(20), default='active')  # active, deleted
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<Topic {self.title}>'