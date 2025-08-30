import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_login import LoginManager
from flask_socketio import SocketIO
from .models.user import db, User
from .utils.init_db import init_db
from .logging_config import configure_logging
from .routes.upload import upload_bp
from .utils.db_factory import DBFactory

from flask_socketio import SocketIO

socketio = SocketIO(async_mode='eventlet', cors_allowed_origins="*")

# from opentelemetry import trace
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.resources import Resource
# from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
# from opentelemetry.instrumentation.flask import FlaskInstrumentor

def create_app(test_config=None, config_name=None):
    # 配置日志
    configure_logging()
    
    # 创建资源（可选）
    # resource = Resource.create({
    #     "service.name": "aka-music-service",
    #     "service.version": "1.0.0",
    # })
    # 将资源添加到 TracerProvider
    
    # 初始化 Tracer
    ''' 使用opentelemetry-sdk 可以使用以下代码进行初始化，没有跑通，暂时废弃掉，使用skywalking-python
    provider = TracerProvider(resource=resource)

    # 配置数据导出到 SkyWalking 的 OTLP 接收端
    otlp_exporter = OTLPSpanExporter(
        endpoint="http://127.0.0.1:11800/v1/traces",  # SkyWalking OAP 地址
        insecure=False  # 如果使用 HTTPS 需配置证书
    )
    
    provider.add_span_processor(SimpleSpanProcessor(otlp_exporter))
    
    trace.set_tracer_provider(provider)
    # 获取全局 Tracer
    tracer = trace.get_tracer(__name__)
    
    # with tracer.start_as_current_span("example-span"):
    #     print("This is a traced operation.")
    '''
    
    # https://skywalking.apache.org/docs/skywalking-python/next/en/setup/configuration/
    from skywalking import agent, config
    try:
        config.init(agent_collector_backend_services='127.0.0.1:11800', agent_name='aka-music-service')
        # agent.start() # TODO 0225，暂时不监控
    except Exception as e:
        print(f"Failed to initialize SkyWalking agent: {e}")
    
    app = Flask(__name__, instance_relative_config=True)
    
    # 初始化 FlaskInstrumentor
    # FlaskInstrumentor().instrument_app(app)
    
    CORS(app, resources={r"/api/*": {
        "origins": [
                "http://127.0.0.1:5173", 
                "https://127.0.0.1:5173", 
                "http://127.0.0.1:11800", 
                "https://127.0.0.1:11800", 
                "http://localhost:5173", 
                "https://localhost:5173", 
                "http://alphago.ltd:5173", 
                "https://alphago.ltd:5173", 
                "http://1851-112-10-202-68.ngrok-free.app",
                "https://1851-112-10-202-68.ngrok-free.app",
                "http://47.98.62.98:5173",
                "https://47.98.62.98:5173",
                "http://47.98.62.98",
                "https://47.98.62.98",
                "http://alphago.ltd",
                "https://alphago.ltd",
            ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"], # TODO DELETE要考虑删除掉
        "allow_headers": ["Content-Type", "Authorization", "traceparent", "tracestate"], # Important!
        "supports_credentials": True
    }})
    
    # 确保实例文件夹存在
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    
    if test_config is None:
        # 加载配置文件
        app.config.from_mapping(
            SECRET_KEY='dev',
            # 设置SQLite数据库文件路径
            SQLALCHEMY_DATABASE_URI=f'sqlite:///{os.path.join(app.instance_path, "aka_music.db")}',
            # 设置PostgreSQL数据库连接
            # 修改连接字符串，确保正确处理无密码认证
            POSTGRES_DATABASE_URI='postgresql://postgres:postgres468028475@localhost:5432/immich',
            SQLALCHEMY_TRACK_MODIFICATIONS=False
        )
    else:
        app.config.update(test_config)
    
    # 初始化数据库
    # 使用DBFactory初始化SQLite和PostgreSQL数据库连接
    # DBFactory.init_app已经初始化了SQLite数据库，不需要再次初始化db
    DBFactory.init_app(app)
    
    # 注意：db对象已经在DBFactory.init_app中被初始化，这里不需要再次初始化
    # db.init_app(app)
    
    # 初始化Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    
    @login_manager.request_loader
    def load_user_from_request(request):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return None
        
        try:
            token_type, token = auth_header.split()
            if token_type.lower() != 'bearer':
                return None
            
            from .routes.auth import get_current_user
            return get_current_user(token)
        except:
            return None
    
    with app.app_context():
        init_db()
    
    # 初始化socketio
    socketio.init_app(app, path='/api/meeting/socket.io')
    
    # 注册蓝图
    from .routes import topic, comment, auth, payment, user, reward, recommend, category, music, beauty, comic, image_search, content, education, rtmp, meeting, assets, face_recognition
    app.register_blueprint(topic.bp)
    app.register_blueprint(comment.bp)
    app.register_blueprint(auth.bp)
    app.register_blueprint(payment.bp)
    app.register_blueprint(user.bp)
    app.register_blueprint(upload_bp, url_prefix='/api/upload')
    app.register_blueprint(reward.bp)
    app.register_blueprint(recommend.recommend_bp)
    app.register_blueprint(category.bp)
    app.register_blueprint(music.bp)
    app.register_blueprint(beauty.bp)
    app.register_blueprint(comic.bp)
    app.register_blueprint(image_search.bp)
    app.register_blueprint(content.bp)
    app.register_blueprint(education.bp)
    app.register_blueprint(rtmp.bp)
    app.register_blueprint(meeting.bp)
    app.register_blueprint(assets.bp)
    app.register_blueprint(face_recognition.bp)

    # 配置静态文件目录
    app.static_folder = 'static'
    os.makedirs(os.path.join(app.static_folder, 'uploads', 'topics'), exist_ok=True)
        
    return app