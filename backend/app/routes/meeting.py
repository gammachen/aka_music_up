from flask import Blueprint, render_template, request, jsonify
from flask_socketio import SocketIO, join_room, leave_room
from flask import current_app
import logging

# 创建蓝图
bp = Blueprint('meeting', __name__, url_prefix='/api/meeting')

# 获取logger
logger = logging.getLogger(__name__)

# 存储用户信息
users = {}

# 从app/__init__.py导入socketio实例
from .. import socketio

# WebRTC信令服务器事件处理
@socketio.on('offer')
def handle_offer(data):
    user_id = data['user_id']
    other_user_id = data['other_user_id']
    offer = data['offer']
    
    logger.info(f'WebRTC信令: 用户 {user_id} 向用户 {other_user_id} 发送offer')
    logger.debug(f'Offer详情: {offer}')

    # 发送offer给另一用户
    socketio.emit('offer', data, room=other_user_id)
    logger.info(f'WebRTC信令: offer已转发至用户 {other_user_id}')

    return jsonify({'status': 'success'})

@socketio.on('answer')
def handle_answer(data):
    user_id = data['user_id']
    other_user_id = data['other_user_id']
    answer = data['answer']
    
    logger.info(f'WebRTC信令: 用户 {user_id} 向用户 {other_user_id} 发送answer')
    logger.debug(f'Answer详情: {answer}')

    # 发送answer给另一用户
    socketio.emit('answer', data, room=other_user_id)
    logger.info(f'WebRTC信令: answer已转发至用户 {other_user_id}')

    return jsonify({'status': 'success'})

@socketio.on('ice_candidate')
def handle_ice_candidate(data):
    user_id = data['user_id']
    other_user_id = data['other_user_id']
    candidate = data['candidate']
    
    logger.info(f'WebRTC信令: 用户 {user_id} 向用户 {other_user_id} 发送ICE候选')
    logger.debug(f'ICE候选详情: {candidate}')

    # 发送ICE候选给另一用户
    socketio.emit('ice_candidate', data, room=other_user_id)
    logger.info(f'WebRTC信令: ICE候选已转发至用户 {other_user_id}')

    return jsonify({'status': 'success'})

@socketio.on('connect')
def on_connect():
    user_id = request.sid
    users[user_id] = {'sid': user_id, 'room': None}
    logger.info(f'WebSocket连接: 用户 {user_id} 已连接')

@socketio.on('join_room')
def on_join_room(data):
    user_id = request.sid
    room = data['room']
    users[user_id]['room'] = room
    join_room(room)
    print(f'User {user_id} joined room {room}')

    # 向房间内的所有用户发送信令
    for user in users.values():
        if user['room'] == room and user['sid'] != user_id:
            socketio.emit('new_user', {'user_id': user_id}, room=user['sid'])

@socketio.on('disconnect')
def on_disconnect():
    user_id = request.sid
    if user_id in users:
        room = users[user_id]['room']
        leave_room(room)
        del users[user_id]
        print(f'User {user_id} disconnected')

@socketio.on('message')
def handle_message(data):
    print('Received message: ' + data)
    socketio.emit('message', data)