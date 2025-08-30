from flask import Blueprint, jsonify
from app.models.category import Category
from ..utils.response import make_response

bp = Blueprint('category', __name__, url_prefix='/api/category')

@bp.route('/tree', methods=['GET'])
def get_category_tree():
    """获取分类树结构"""
    try:
        categories = Category.get_category_tree()
        
        return make_response(data=categories)
    except Exception as e:
        return make_response(data=None, message='获取分类树失败', code=500)

@bp.route('/categories', methods=['GET'])
def get_all_categories():
    """获取所有分类列表（扁平结构）"""
    try:
        categories = Category.get_all_categories()
        data = [{
                'id': category.id,
                'name': category.name,
                'parent_id': category.parent_id,
                'level': category.level,
                'sort_order': category.sort_order
            } for category in categories]
        return make_response(data=data)
    except Exception as e:
        return make_response(data=None, message=f'获取分类列表失败: {str(e)}', code=500)