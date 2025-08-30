from functools import wraps
from flask import request, jsonify
import jwt
from ..models.user import User
from .response import make_response
import logging

JWT_SECRET_KEY = 'your-secret-key'  # 在实际应用中应该从配置文件中读取
logger = logging.getLogger(__name__)

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        logger.info(f'收到请求，认证头信息：{auth_header}')

        if auth_header:
            try:
                token = auth_header.split(' ')[1]
                logger.info(f'成功提取token：{token[:10]}...')
            except IndexError:
                logger.error('认证头格式错误')
                return make_response(message='无效的认证头', code=401)

        if not token:
            logger.error('请求中缺少token')
            return make_response(message='缺少认证Token', code=401)

        try:
            logger.info('开始解析token...')
            data = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
            logger.info(f'token解析成功，用户ID：{data.get("user_id")}')
            
            current_user = User.query.get(data['user_id'])
            if not current_user:
                logger.error(f'用户不存在，ID：{data.get("user_id")}')
                return make_response(message='用户不存在', code=401)
            
            logger.info(f'用户认证成功，ID：{current_user.id}')
        except jwt.ExpiredSignatureError:
            logger.error('token已过期')
            return make_response(message='Token已过期', code=401)
        except jwt.InvalidTokenError as e:
            logger.error(f'无效的token：{str(e)}')
            return make_response(message='无效的Token', code=401)

        return f(current_user, *args, **kwargs)
    return decorated