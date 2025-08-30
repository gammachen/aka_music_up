from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, Index
from sqlalchemy.ext.mutable import MutableList
from ..models.user import db
from ..utils.db_factory import DBFactory
import json
from flask import current_app

# 使用PostgreSQL数据库实例
postgres_db = DBFactory.get_db('postgres')

# 根据数据库类型导入不同的模块
try:
    from sqlalchemy.dialects.postgresql import ARRAY, FLOAT
except ImportError:
    # 如果不支持PostgreSQL，创建替代类
    pass

class FaceImage(postgres_db.Model):
    __tablename__ = 'face_images'
    __bind_key__ = 'postgres'  # 指定使用postgres绑定

    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255), nullable=False)  # 原始图片路径
    face_location = db.Column(MutableList.as_mutable(ARRAY(db.Integer)), nullable=True)  # 人脸位置 [top, right, bottom, left]
    feature_vector = db.Column(MutableList.as_mutable(ARRAY(FLOAT)), nullable=True)  # CLIP特征向量
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)  # 关联用户ID（如果有）
    image_metadata = db.Column(db.JSON, nullable=True)  # 额外元数据，如EXIF信息
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 创建向量索引（需要在PostgreSQL中启用pgvector扩展）
    __table_args__ = (
        # 使用PostgreSQL的pgvector扩展创建向量索引
        Index('idx_feature_vector', feature_vector, postgresql_using='ivfflat'),
    )

    def __repr__(self):
        return f'<FaceImage {self.id} - {self.image_path}>'

    def to_dict(self):
        return {
            'id': self.id,
            'image_path': self.image_path,
            'face_location': self.face_location,
            'user_id': self.user_id,
            'metadata': self.image_metadata,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    def save_to_file(self, output_dir=None):
        """
        将FaceImage对象保存到本地JSON文件
        
        Args:
            output_dir: 输出目录，如果为None则使用默认目录
        
        Returns:
            保存的文件路径
        """
        import os
        import json
        from datetime import datetime
        
        # 确定输出目录
        if output_dir is None:
            # 使用与人脸图像相同的目录结构，但在static/face_data下
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(base_dir, 'static', 'face_data')
        
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名（使用时间戳和ID确保唯一性）
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"face_{self.id or timestamp}.json"
        file_path = os.path.join(output_dir, filename)
        
        # 准备数据
        data = self.to_dict()
        
        # 特殊处理feature_vector（如果存在）
        if hasattr(self, 'feature_vector') and self.feature_vector:
            data['feature_vector'] = self.feature_vector
        
        # 保存到文件
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"已将人脸数据保存到: {file_path}")
            return file_path
        except Exception as e:
            print(f"保存人脸数据时出错: {str(e)}")
            return None