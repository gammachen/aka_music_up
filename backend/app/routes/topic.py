from flask import Blueprint, jsonify, request
from ..utils.auth import token_required
from ..models.topic import Topic
from ..models.topic_comment import TopicComment
from ..models.user import db
from ..utils.response import make_response

bp = Blueprint('topic', __name__, url_prefix='/api/topics')

@bp.route('/', methods=['GET'])
def get_topics():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        category = request.args.get('category')
        
        query = Topic.query
        if category:
            query = query.filter_by(category=category)
        
        pagination = query.order_by(Topic.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        topics = pagination.items
        data = {
            'topics': [{
                'id': topic.id,
                'title': topic.title,
                'content': topic.content,
                'category': topic.category,
                'views': topic.views,
                'likes': topic.likes,
                'favorites': topic.favorites,
                'comments_count': topic.comments_count,
                'created_at': topic.created_at.isoformat()
            } for topic in topics],
            'pagination': {
                'total': pagination.total,
                'pages': pagination.pages,
                'current_page': page,
                'per_page': per_page
            }
        }
        return make_response(data=data)
    except Exception as e:
        return make_response(message=str(e), code=500)

@bp.route('/<int:topic_id>', methods=['GET'])
def get_topic(topic_id):
    try:
        topic = Topic.query.get_or_404(topic_id)
        # 增加浏览量
        topic.views += 1
        db.session.commit()
        
        data = {
            'id': topic.id,
            'title': topic.title,
            'content': topic.content,
            'category': topic.category,
            'views': topic.views,
            'likes': topic.likes,
            'favorites': topic.favorites,
            'comments_count': topic.comments_count,
            'created_at': topic.created_at.isoformat()
        }
        return make_response(data=data)
    except Exception as e:
        return make_response(message=str(e), code=500)

@bp.route('/create', methods=['POST'])
@token_required
def create_topic(current_user):
    try:
        data = request.get_json()
        title = data.get('title')
        content = data.get('content')
        category = data.get('category')
        hidden_content = data.get('hidden_content')

        print(f'[DEBUG] 收到创建主题请求: {data}')
        print(f'[DEBUG] 当前用户ID: {current_user.id}')

        if not all([title, content, category]):
            return make_response(message='缺少必要参数', code=400)

        topic = Topic(
            title=title,
            content=content,
            category=category,
            hidden_content=hidden_content,
            user_id=current_user.id
        )
        print(f'[DEBUG] 准备创建主题: {topic.__dict__}')

        try:
            db.session.add(topic)
            print('[DEBUG] 主题已添加到session')
            db.session.flush()
            print(f'[DEBUG] 主题ID: {topic.id}')
            db.session.commit()
            print('[DEBUG] 数据库事务已提交')
        except Exception as db_error:
            db.session.rollback()
            print(f'[ERROR] 数据库操作失败: {str(db_error)}')
            raise db_error

        return make_response(data={'id': topic.id}, message='主题创建成功')
    except Exception as e:
        print(f'[ERROR] 创建主题失败: {str(e)}')
        return make_response(message=str(e), code=500)

@bp.route('/<int:topic_id>', methods=['PUT'])
@token_required
def update_topic(current_user, topic_id):
    try:
        topic = Topic.query.get_or_404(topic_id)
        if topic.user_id != current_user.id:
            return make_response(message='无权限修改此主题', code=403)

        data = request.get_json()
        topic.title = data.get('title', topic.title)
        topic.content = data.get('content', topic.content)
        topic.category = data.get('category', topic.category)
        topic.hidden_content = data.get('hidden_content', topic.hidden_content)

        db.session.commit()
        return make_response(message='主题更新成功')
    except Exception as e:
        return make_response(message=str(e), code=500)

@bp.route('/<int:topic_id>', methods=['DELETE'])
@token_required
def delete_topic(current_user, topic_id):
    try:
        topic = Topic.query.get_or_404(topic_id)
        if topic.user_id != current_user.id:
            return make_response(message='无权限删除此主题', code=403)

        topic.status = 'deleted'
        db.session.commit()
        return make_response(message='主题删除成功')
    except Exception as e:
        return make_response(message=str(e), code=500)