from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, join_room, leave_room
from flask_cors import CORS
from flask_talisman import Talisman

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)  # 允许所有域名访问

# 设置CSP头，允许内联脚本
csp = {
    'default-src': '\'self\'',
    'script-src': ['\'self\'', 'https://cdnjs.cloudflare.com', '\'unsafe-inline\''],
    'style-src': ['\'self\'', '\'unsafe-inline\'']
}

Talisman(app, content_security_policy=csp)

socketio = SocketIO(app)

# 存储用户信息
users = {}

@app.route('/')
def index():
    return render_template('42_webrtc_multi_user_meeting.html')

@app.route('/v2')
def index2():
    return render_template('42_webrtc_multi_user_meeting_v2.html')


@socketio.on('offer')
def handle_offer(data):
    # data = request.json
    user_id = data['user_id']
    other_user_id = data['other_user_id']
    offer = data['offer']
    
    print(f'User {user_id} offer other_user_id: {other_user_id} offer: {offer}')

    # 发送offer给另一用户
    socketio.emit('offer', data, room=other_user_id)

    return jsonify({'status': 'success'})

@socketio.on('answer')
def handle_answer(data):
    # data = request.json
    user_id = data['user_id']
    other_user_id = data['other_user_id']
    answer = data['answer']
    
    print(f'User {user_id} answer other_user_id: {other_user_id} answer: {answer}')

    # 发送answer给另一用户
    # socketio.emit('answer', data, room=other_user_id)
    socketio.emit('answer', data, room=other_user_id)

    return jsonify({'status': 'success'})

@socketio.on('ice_candidate')
def handle_ice_candidate(data):
    # data = request.json
    user_id = data['user_id']
    other_user_id = data['other_user_id']
    candidate = data['candidate']
    
    print(f'User {user_id} ice_candidate')

    # 发送ICE候选给另一用户
    socketio.emit('ice_candidate', data, room=other_user_id)

    return jsonify({'status': 'success'})

@socketio.on('connect')
def on_connect():
    user_id = request.sid
    users[user_id] = {'sid': user_id, 'room': None}
    print(f'User {user_id} connected')

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

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=6001, ssl_context='adhoc')

