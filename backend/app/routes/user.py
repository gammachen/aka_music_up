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
from app.utils.auth import token_required

bp = Blueprint('user', __name__, url_prefix='/api/user')

@bp.route('/balance', methods=['GET'])
@token_required
def get_current_user_info(current_user):
    """获取当前用户信息"""
    '''
    return jsonify({
        "code": 0,
        "message": "获取用户信息成功",
        "data": {
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email,
            "avatar_path": current_user.avatar,
            "balance": current_user.gold_balance
        }
    })
    '''
    
    response_data = {
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email,
            "avatar_path": current_user.avatar,
            "balance": current_user.gold_balance
        }
    
    return make_response(data=response_data)