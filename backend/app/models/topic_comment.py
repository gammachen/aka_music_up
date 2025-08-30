from datetime import datetime
from .user import db

class TopicComment(db.Model):
    __tablename__ = 'topic_comments'

    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    topic_id = db.Column(db.Integer, db.ForeignKey('topics.id'), nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey('topic_comments.id'))  # 父评论ID，用于回复
    likes = db.Column(db.Integer, default=0)  # 点赞数
    status = db.Column(db.String(20), default='active')  # active, deleted
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关联
    user = db.relationship('User', backref=db.backref('comments', lazy=True))
    topic = db.relationship('Topic', backref=db.backref('comments', lazy=True))
    replies = db.relationship('TopicComment', backref=db.backref('parent', remote_side=[id]), lazy=True)

    def __repr__(self):
        return f'<TopicComment {self.id}>'