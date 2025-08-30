from datetime import datetime

# 创建content表的SQL语句
CREATE_CONTENT_TABLE = """
CREATE TABLE IF NOT EXISTS content (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(30) NOT NULL,
    title VARCHAR(255) NOT NULL,
    type VARCHAR(10) NOT NULL,
    author_id VARCHAR(36) NOT NULL,
    cover_url VARCHAR(255),
    description TEXT,
    status VARCHAR(10) NOT NULL DEFAULT 'DRAFT',
    price_strategy VARCHAR(10) NOT NULL,
    publish_date DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

# 创建chapter表的SQL语句
CREATE_CHAPTER_TABLE = """
CREATE TABLE IF NOT EXISTS chapter (
    id VARCHAR(36) PRIMARY KEY,
    content_id VARCHAR(36) NOT NULL,
    chapter_no INTEGER NOT NULL,
    title VARCHAR(255) NOT NULL,
    pages JSON NOT NULL,
    price DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    is_free BOOLEAN NOT NULL DEFAULT FALSE,
    unlock_type VARCHAR(10) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (content_id) REFERENCES content(id) ON DELETE CASCADE
);
"""

from .user import db
from sqlalchemy import Column, JSON
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.ext.declarative import declarative_base

'''
TODO 增加Tags的字段（理论上要增加Tags与关联表的，简单起见可以用字段来表示，后续再考虑增加Tags表）
'''
class Content(db.Model):
    __tablename__ = 'content'

    id = db.Column(db.String(36), primary_key=True)
    name = db.Column(db.String(30), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    type = db.Column(db.String(10), nullable=False)
    author_id = db.Column(db.String(36), nullable=False)
    cover_url = db.Column(db.String(255))
    description = db.Column(db.Text)
    status = db.Column(db.String(10), nullable=False, default='DRAFT')
    price_strategy = db.Column(db.String(10), nullable=False)
    publish_date = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 建立与Chapter的一对多关系
    chapters = db.relationship('Chapter', backref='content', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Content {self.title}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'title': self.title,
            'type': self.type,
            'author_id': self.author_id,
            'cover_url': self.cover_url,
            'description': self.description,
            'status': self.status,
            'price_strategy': self.price_strategy,
            'publish_date': self.publish_date.isoformat() if self.publish_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Chapter(db.Model):
    __tablename__ = 'chapter'

    id = db.Column(db.String(36), primary_key=True)
    content_id = db.Column(db.String(36), db.ForeignKey('content.id'), nullable=False)
    chapter_no = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String(255), nullable=False)
    pages = db.Column(MutableDict.as_mutable(db.JSON), nullable=False)
    price = db.Column(db.Integer, nullable=False, default=0.00)
    is_free = db.Column(db.Boolean, nullable=False, default=False)
    unlock_type = db.Column(db.String(10), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<Chapter {self.title}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'content_id': self.content_id,
            'chapter_no': self.chapter_no,
            'title': self.title,
            'pages': self.pages,
            'price': float(self.price),
            'is_free': self.is_free,
            'unlock_type': self.unlock_type,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }