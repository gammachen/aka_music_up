from app import db
from datetime import datetime

class Category(db.Model):
    """分类模型
    采用邻接表方案实现二级分类树结构
    """
    __tablename__ = 'category'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, comment='分类名称')
    parent_id = db.Column(db.Integer, db.ForeignKey('category.id'), nullable=True, comment='父分类ID')
    level = db.Column(db.Integer, default=1, comment='分类层级: 1-一级分类, 2-二级分类')
    sort_order = db.Column(db.Integer, default=0, comment='排序权重')
    background_style = db.Column(db.String(100), nullable=True, comment='背景样式')
    desc_image = db.Column(db.String(255), nullable=True, comment='辅助图片URL')
    refer_id = db.Column(db.String(19), nullable=True, comment='关联的目录ID，与本地目录ID对应') # TODO 在beauty中这个id是通过md5算法对目录名称计算生成的，主要还是因为这个id要暴露出去给外部访问，否则直接将目录名字暴露出去都可以的，但是这个id对维护人员其实是非常不友好的，都没地方看这个值是什么（@resouce中的image_mapping.json中），如果非要自己计算的话，也能够使用md5的工具直接计算的）
    prefix = db.Column(db.String(50), nullable=True, comment='分类页面中渲染路径的前缀，比如mulist、beaulist、movlist等')
    created_at = db.Column(db.DateTime, default=datetime.utcnow, comment='创建时间')
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment='更新时间')

    # 建立父子关系
    parent = db.relationship('Category', remote_side=[id], backref=db.backref('children', lazy='dynamic'))

    @staticmethod
    def get_all_categories():
        """获取所有分类，按层级和排序返回"""
        return Category.query.order_by(Category.level, Category.sort_order).all()

    @staticmethod
    def get_category_tree():
        """获取分类树结构"""
        categories = Category.query.filter_by(level=1).order_by(Category.sort_order).all()
        result = []
        for category in categories:
            category_dict = {
                'id': category.id,
                'name': category.name,
                'prefix': category.prefix,
                'children': [{
                    'id': child.id,
                    'name': child.name,
                    'background_style': child.background_style,
                    'desc_image': child.desc_image,
                    'refer_id': child.refer_id,
                    'prefix': child.prefix
                } for child in category.children.order_by(Category.sort_order)]
            }
            result.append(category_dict)
        return result

    def add_child(self, name, sort_order=0):
        """添加子分类
        
        Args:
            name: 子分类名称
            sort_order: 排序权重
            
        Returns:
            Category: 新创建的子分类
            
        Raises:
            ValueError: 当前分类已是二级分类，无法添加子分类
        """
        if self.level >= 2:
            raise ValueError('不能创建超过二级的分类')
            
        child = Category(
            name=name,
            parent_id=self.id,
            level=self.level + 1,
            sort_order=sort_order
        )
        db.session.add(child)
        return child

    def __repr__(self):
        return f'<Category {self.name}>'