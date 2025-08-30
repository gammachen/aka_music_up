import eventlet
eventlet.monkey_patch()
import os
import socket

from app import create_app, socketio

app = create_app()

if __name__ == '__main__':
    # 确保使用eventlet作为服务器
    socketio.server_options['async_mode'] = 'eventlet'
    
    # 检查是否需要SSL
    use_ssl = False
    cert_file = None
    key_file = None
    
    # 首先检查环境变量
    if os.environ.get('FLASK_CERT') and os.environ.get('FLASK_KEY'):
        use_ssl = True
        cert_file = os.environ.get('FLASK_CERT')
        key_file = os.environ.get('FLASK_KEY')
        print(f"Using SSL certificates from environment: {cert_file}, {key_file}")
    # 然后检查本地证书文件
    elif os.path.exists('certs/cert.pem') and os.path.exists('certs/key.pem'):
        use_ssl = True
        cert_file = 'certs/cert.pem'
        key_file = 'certs/key.pem'
        print(f"Using local SSL certificates: {cert_file}, {key_file}")
    
    if use_ssl:
        # 使用Flask-SocketIO的run方法，但不直接传递ssl_context
        # 在内部，它会正确处理eventlet的SSL配置
        socketio.run(app, host='0.0.0.0', debug=True, keyfile=key_file, certfile=cert_file)
    else:
        socketio.run(app, host='0.0.0.0', debug=True)