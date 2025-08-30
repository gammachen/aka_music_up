from flask import Blueprint, send_from_directory, current_app
import os

bp = Blueprint('assets', __name__)

# 通过/api路径请求到后端的资源文件的内容的吐出去（不需要这个特殊的处理，因为我们的flask中定义了static_folder配置，直接配置到uploads目录下的，详情见__init__.py）
# deprecated 2025.02.21
@bp.route('/api/assets/<path:filename>')
def serve_asset(filename):
    # 设置资源文件的基础目录
    assets_dir = os.path.join(current_app.root_path, 'static', 'assets')
    
    # 确保目录存在
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
    
    try:
        # 使用 send_from_directory 安全地发送文件
        return send_from_directory(assets_dir, filename)
    except Exception as e:
        current_app.logger.error(f"Error serving asset {filename}: {str(e)}")
        return {'error': 'Asset not found'}, 404