from datetime import datetime
from app import db

class RewardRecord(db.Model):
    '''打赏记录单'''
    __tablename__ = 'rewards'

    id = db.Column(db.Integer, primary_key=True)
    amount = db.Column(db.Integer, nullable=False, comment='打赏金额（金币）')
    from_user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, comment='打赏用户ID')
    to_user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, comment='被打赏用户ID')
    topic_id = db.Column(db.Integer, db.ForeignKey('topics.id'), nullable=False, comment='主题ID')
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, comment='打赏时间')
    message = db.Column(db.String(200), nullable=True, comment='打赏留言')

    # 关联关系
    from_user = db.relationship('User', foreign_keys=[from_user_id], backref=db.backref('rewards_sent', lazy='dynamic'))
    to_user = db.relationship('User', foreign_keys=[to_user_id], backref=db.backref('rewards_received', lazy='dynamic'))
    topic = db.relationship('Topic', backref=db.backref('rewards', lazy='dynamic'))

    def to_dict(self):
        return {
            'id': self.id,
            'amount': self.amount,
            'from_user_id': self.from_user_id,
            'to_user_id': self.to_user_id,
            'topic_id': self.topic_id,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'message': self.message,
            'from_user': self.from_user.to_dict() if self.from_user else None,
            'to_user': self.to_user.to_dict() if self.to_user else None
        }