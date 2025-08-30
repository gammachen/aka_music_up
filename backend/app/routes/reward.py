from flask import Blueprint, request, jsonify
# from flask_jwt_extended import jwt_required, get_jwt_identity
from ..utils.auth import token_required
from app import db
from app.models.reward import RewardRecord
from app.models.user import User
from app.models.topic import Topic
from app.models.payment import GoldTransaction
from datetime import datetime
from ..utils.response import make_response
from app.services.reward import RewardService

bp = Blueprint('rewards', __name__, url_prefix='/api/rewards')

@bp.route('/create', methods=['POST', 'OPTIONS'])
@token_required
def create_reward(current_user):
    """创建打赏记录"""
    data = request.get_json()

    # 验证必要参数
    if not all(key in data for key in ['target_id', 'amount', 'target_type']):
        return make_response(message='缺少必要参数', code=400)

    # TODO target_type可以扩展，现在只是一个topic，表明是对topic进行打赏的
    try:
        # 调用服务层处理打赏逻辑
        reward_data = RewardService.create_reward(
            from_user_id=current_user.id,
            topic_id=data['target_id'],
            amount=data['amount'],
            message=data.get('message', '')
        )
        return make_response(message='打赏成功', data=reward_data)
    except ValueError as e:
        return make_response(message=str(e), code=400)
    except Exception as e:
        return make_response(message=str(e), code=500)

@bp.route('/rewards/topic/<int:topic_id>', methods=['GET'])
def get_topic_rewards(topic_id):
    """获取主题的打赏记录"""
    try:
        rewards = RewardRecord.query.filter_by(topic_id=topic_id).order_by(RewardRecord.created_at.desc()).all()
        return make_response(data=[reward.to_dict() for reward in rewards])
    except Exception as e:
        return make_response(message=str(e), code=500)

@bp.route('/rewards/user/<int:user_id>', methods=['GET'])
@token_required
def get_user_rewards(user_id):
    """获取用户的打赏记录（包括发出和收到的打赏）"""
    try:
        sent_rewards = RewardRecord.query.filter_by(from_user_id=user_id).order_by(RewardRecord.created_at.desc()).all()
        received_rewards = RewardRecord.query.filter_by(to_user_id=user_id).order_by(RewardRecord.created_at.desc()).all()

        return make_response(data={
            'sent': [reward.to_dict() for reward in sent_rewards],
            'received': [reward.to_dict() for reward in received_rewards]
        })
    except Exception as e:
        return make_response(message=str(e), code=500)