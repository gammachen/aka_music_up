from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from datetime import datetime, timedelta
from passlib.context import CryptContext
from random import randint
import secrets
import jwt
from typing import Optional
from app.models.user import User, db
from ..utils.response import make_response
import os

# JWT配置
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key')  # 建议在生产环境中使用环境变量
JWT_ALGORITHM = "HS256"

bp = Blueprint('auth', __name__, url_prefix='/api/auth/v1')

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 验证码存储
verification_codes = {}

def generate_code() -> str:
    """生成6位数字验证码"""
    return str(randint(100000, 999999))

def store_code(key: str, code: str) -> None:
    """存储验证码，设置5分钟有效期"""
    verification_codes[key] = {
        'code': code,
        'expire_time': datetime.now() + timedelta(minutes=5)
    }

def verify_code(key: str, code: str) -> bool:
    """验证验证码"""
    if key not in verification_codes:
        return False
    stored = verification_codes[key]
    if datetime.now() > stored['expire_time']:
        del verification_codes[key]
        return False
    if stored['code'] != code:
        return False
    del verification_codes[key]  # 使用后删除验证码
    return True

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(username: str):
    return db.session.query(User).filter(User.username == str(username)).first()

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.password_hash):
        return False
    return user

@bp.route('/phone-code', methods=['POST'])
def send_phone_code():
    """发送手机验证码"""
    data = request.get_json()
    if not data or 'phone' not in data:
        return jsonify({
            "code": 1001,
            "message": "缺少手机号参数",
            "data": None
        }), 400

    phone = data['phone']
    code = generate_code()
    store_code(phone, code)
    # TODO: 接入短信服务发送验证码（验证手机号的正确性再生成啥的）
    
    return jsonify({
        "code": 0,
        "message": "验证码发送成功",
        "data": {"code": code}  # 开发环境直接返回验证码
    })

@bp.route('/email-code', methods=['POST'])
def send_email_code():
    """发送邮箱验证码"""
    data = request.get_json()
    if not data or 'email' not in data:
        return jsonify({
            "code": 1001,
            "message": "缺少邮箱参数",
            "data": None
        }), 400

    code = generate_code()
    store_code(data['email'], code)
    # TODO: 接入邮件服务发送验证码
    return jsonify({
        "code": 0,
        "message": "验证码发送成功",
        "data": {"code": code}  # 开发环境直接返回验证码
    })

@bp.route('/<platform>/url', methods=['GET'])
def get_auth_url(platform):
    """获取第三方平台授权URL"""
    if platform not in ["wechat", "qq"]:
        return jsonify({
            "code": 1001,
            "message": "不支持的平台",
            "data": None
        }), 400
    
    # 生成state参数
    state = secrets.token_urlsafe(32)
    
    # TODO: 保存state到Redis或数据库，用于回调验证
    
    # 根据平台返回对应的授权URL
    auth_urls = {
        "wechat": "https://open.weixin.qq.com/connect/qrconnect",
        "qq": "https://graph.qq.com/oauth2.0/authorize"
    }
    
    return jsonify({
        "code": 0,
        "message": "获取授权URL成功",
        "data": {
            "auth_url": auth_urls[platform],
            "state": state
        }
    })

@bp.route('/register/phone', methods=['POST'])
def register_by_phone():
    """手机号注册"""
    data = request.get_json()
    if not data or not all([data.get('phone'), data.get('code'), data.get('password')]):
        return jsonify({
            "code": 1001,
            "message": "缺少必要参数",
            "data": None
        }), 400
    
    # 验证验证码
    if not verify_code(data['phone'], data['code']):
        return jsonify({
            "code": 1003,
            "message": "验证码错误或已过期",
            "data": None
        }), 400
    
    # 验证手机号是否已被注册（TODO 验证手机号的正确性）
    if db.session.query(User).filter(User.phone == data['phone']).first():
        return jsonify({
            "code": 1002,
            "message": "手机号已被注册",
            "data": None
        }), 400
    
    # 生成用户名（使用手机号后4位）
    username = f"user_{data['phone'][-4:]}"
    
    # 为了避免重复，如果用户名已存在，添加随机后缀（一直搞）
    while get_user(username):
        username = f"user_{data['phone'][-4:]}_{randint(1000, 9999)}"
    
    new_user = User(
        username=username,
        phone=data['phone'],
        password_hash=get_password_hash(data['password']),
        avatar="/uploads/avatars/default.jpg"
    )
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({
        "code": 0,
        "message": "注册成功",
        "data": {
            "username": username,
            "phone": data['phone']
        }
    })

@bp.route('/register/email', methods=['POST'])
def register():
    """用户注册"""
    data = request.get_json()
    if not data or not all([data.get('password'), data.get('email'), data.get('code')]):
        return jsonify({
            "code": 1001,
            "message": "缺少必要参数",
            "data": None
        }), 400

    # 验证验证码
    if not verify_code(data.get('email'), data['code']):
        return jsonify({
            "code": 1003,
            "message": "验证码错误或已过期",
            "data": None
        }), 400

    if db.session.query(User).filter(User.email == data['email']).first():
        return jsonify({
            "code": 1002,
            "message": "邮箱已被注册",
            "data": None
        }), 400
    
    username=data['email'].split('@')[0]
    
    while get_user(username):
        username = f"user_{data['email'][-4:]}_{randint(1000, 9999)}"
        
    new_user = User(
        # 使用邮箱的前缀作为用户名
        username=str(username),
        email=data['email'],
        password_hash=get_password_hash(data['password']),
        avatar="/uploads/avatars/default.jpg"
    )
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({
        "code": 0,
        "message": "注册成功",
        "data": {
            "username": username,
            "email": data['email']
        }
    })

@bp.route('/login/email', methods=['POST', 'GET'])
def login_by_email():
    if request.method == 'GET':
        next_url = request.args.get('next', '')
        return make_response(code=401, message='请先登录', data={'redirect_url': f'http://localhost:5173/login?next={next_url}'})
    
    """邮箱密码登录"""
    data = request.get_json()
    if not data or not all([data.get('email'), data.get('password')]):
        return jsonify({
            "code": 1001,
            "message": "缺少必要参数",
            "data": None
        }), 400

    user = db.session.query(User).filter(User.email == data['email']).first()
    if not user or not verify_password(data['password'], user.password_hash):
        return jsonify({
            "code": 1002,
            "message": "邮箱或密码错误",
            "data": None
        }), 401
    
    access_token = create_access_token(data={"user_id": user.id})
    
    return jsonify({
        "code": 0,
        "message": "登录成功",
        "data": {
            "token": access_token,
            "user": {
                "username": user.username,
                "email": user.email,
                "avatar_path": user.avatar
            }
        }
    })

@bp.route('/login/phone', methods=['POST'])
def login_by_phone():
    """手机验证码登录"""
    data = request.get_json()
    if not data or not all([data.get('phone'), data.get('code')]):
        return jsonify({
            "code": 1001,
            "message": "缺少必要参数",
            "data": None
        }), 400

    # 验证验证码
    if not verify_code(data['phone'], data['code']):
        return jsonify({
            "code": 1003,
            "message": "验证码错误或已过期",
            "data": None
        }), 400

    user = db.session.query(User).filter(User.phone == data['phone']).first()
    if not user:
        return jsonify({
            "code": 1002,
            "message": "该手机号未注册",
            "data": None
        }), 401
    
    access_token = create_access_token(data={"user_id": user.id})
    
    return jsonify({
        "code": 0,
        "message": "登录成功",
        "data": {
            "token": access_token,
            "user": {
                "username": user.username,
                "phone": user.phone,
                "avatar_path": user.avatar
            }
        }
    })

@bp.route('/login/oauth/<platform>', methods=['POST'])
def login_by_oauth(platform):
    """第三方登录"""
    if platform not in ["wechat", "qq"]:
        return jsonify({
            "code": 1001,
            "message": "不支持的平台",
            "data": None
        }), 400

    data = request.get_json()
    if not data or not all([data.get('code'), data.get('state')]):
        return jsonify({
            "code": 1001,
            "message": "缺少必要参数",
            "data": None
        }), 400

    try:
        # TODO: 验证state参数
        # TODO: 根据platform和code获取access_token和用户信息
        
        # 示例用户数据
        user_info = {
            "id": 1,
            "username": "test_user",
            "avatar_path": "/uploads/avatars/default.jpg"
        }
        
        access_token = create_access_token(data={"user_id": user_info["id"]})
        
        return jsonify({
            "code": 0,
            "message": "登录成功",
            "data": {
                "token": access_token,
                "user": user_info
            }
        })
    except Exception as e:
        return jsonify({
            "code": 1003,
            "message": str(e),
            "data": None
        }), 400

@bp.route('/<platform>/callback', methods=['GET'])
def auth_callback(platform):
    """处理第三方平台的授权回调"""
    code = request.args.get('code')
    state = request.args.get('state')
    
    if not all([code, state]):
        return jsonify({
            "code": 1001,
            "message": "缺少必要参数",
            "data": None
        }), 400
    
    try:
        # TODO: 验证state参数
        # TODO: 根据platform和code获取access_token和用户信息
        
        # 示例用户数据
        user_info = {
            "id": 1,
            "username": "test_user",
            "avatar_path": "/uploads/avatars/default.jpg"
        }
        
        # 生成JWT token
        access_token = create_access_token(data={"user_id": user_info["id"]})
        
        return jsonify({
            "code": 0,
            "message": "授权成功",
            "data": {
                "token": access_token,
                "user": user_info
            }
        })
    except Exception as e:
        return jsonify({
            "code": 1003,
            "message": str(e),
            "data": None
        }), 400

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """创建JWT token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=300)  # 默认30分钟过期
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    return encoded_jwt

def get_current_user(token):
    """验证JWT token并返回当前用户"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")
        if user_id is None:
            return None
        
        user = db.session.query(User).get(user_id)
        if user is None:
            return None
        
        return user
    except jwt.JWTError:
        return None