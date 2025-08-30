from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from ..models.topic_comment import TopicComment
from ..models.topic import Topic
from ..models.user import db
from ..utils.response import make_response
from ..utils.auth import token_required

bp = Blueprint('comment', __name__, url_prefix='/api/comments')

@bp.route('/', methods=['GET'])
def get_comments():
    try:
        topic_id = request.args.get('topic_id', type=int)
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        if not topic_id:
            return make_response(message='缺少主题ID参数', code=400)
        
        # 获取主题的评论，按时间倒序排列
        query = TopicComment.query.filter_by(
            topic_id=topic_id,
            parent_id=None,  # 只获取顶级评论
            status='active'
        )
        
        pagination = query.order_by(TopicComment.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        comments = pagination.items
        data = {
            'comments': [{
                'id': comment.id,
                'content': comment.content,
                'user_id': comment.user_id,
                'username': comment.user.username,
                'likes': comment.likes,
                'replies_count': len(comment.replies),
                'created_at': comment.created_at.isoformat(),
                'replies': [{
                    'id': reply.id,
                    'content': reply.content,
                    'user_id': reply.user_id,
                    'username': reply.user.username,
                    'likes': reply.likes,
                    'created_at': reply.created_at.isoformat()
                } for reply in comment.replies if reply.status == 'active']
            } for comment in comments],
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

@bp.route('/reply', methods=['POST'])
@token_required
def create_comment(current_user):
    try:
        data = request.get_json()
        topic_id = data.get('topic_id')
        content = data.get('content')
        parent_id = data.get('parent_id')  # 可选，回复其他评论时使用
        
        if not all([topic_id, content]):
            return make_response(message='缺少必要参数', code=400)
        
        # 检查主题是否存在
        try:
            topic_id = int(topic_id)
        except (ValueError, TypeError):
            return make_response(message='主题ID必须为整数', code=400)
            
        topic = Topic.query.get_or_404(topic_id)
        
        # 如果是回复其他评论，检查父评论是否存在
        if parent_id:
            parent_comment = TopicComment.query.get_or_404(parent_id)
            # print(parent_comment.topic_id, topic_id, parent_id)
            
            if parent_comment.topic_id != topic_id:
                return make_response(message='父评论不属于该主题', code=400)
        
        comment = TopicComment(
            content=content,
            topic_id=topic_id,
            user_id=current_user.id,
            parent_id=parent_id
        )
        
        db.session.add(comment)
        topic.comments_count += 1
        db.session.commit()
        
        return make_response(data={'id': comment.id}, message='评论创建成功')
    except Exception as e:
        return make_response(message=str(e), code=500)

@bp.route('/<int:comment_id>', methods=['PUT'])
@token_required
def update_comment(comment_id):
    try:
        comment = TopicComment.query.get_or_404(comment_id)
        if comment.user_id != current_user.id:
            return make_response(message='无权限修改此评论', code=403)
        
        data = request.get_json()
        content = data.get('content')
        
        if not content:
            return make_response(message='缺少评论内容', code=400)
        
        comment.content = content
        db.session.commit()
        
        return make_response(message='评论更新成功')
    except Exception as e:
        return make_response(message=str(e), code=500)

@bp.route('/<int:comment_id>', methods=['DELETE'])
@token_required
def delete_comment(comment_id):
    try:
        comment = TopicComment.query.get_or_404(comment_id)
        if comment.user_id != current_user.id:
            return make_response(message='无权限删除此评论', code=403)
        
        comment.status = 'deleted'
        topic = Topic.query.get(comment.topic_id)
        topic.comments_count -= 1
        db.session.commit()
        
        return make_response(message='评论删除成功')
    except Exception as e:
        return make_response(message=str(e), code=500)