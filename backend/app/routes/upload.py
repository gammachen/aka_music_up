from flask import Blueprint, request, current_app
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import secrets
from ..utils.response import make_response

upload_bp = Blueprint('upload', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@upload_bp.route('/image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return make_response(code=500, message=f'没有文件被上传')
    
    file = request.files['file']
    if file.filename == '':
        return make_response(code=500, message=f'没有选择文件')
    
    if not allowed_file(file.filename):
        return make_response(code=500, message=f'不支持的文件类型')
    
    # 检查文件大小
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    
    if size > MAX_FILE_SIZE:
        return make_response(code=500, message=f'文件大小超过限制')
    
    try:
        # 获取原始文件名和扩展名
        original_filename = file.filename
        _, ext = os.path.splitext(original_filename)
        
        # 生成包含时间戳的唯一文件名
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        unique_filename = f"{timestamp}_{secrets.token_hex(8)}{ext}"
        safe_filename = secure_filename(unique_filename)
        
        # 确保上传目录存在
        upload_folder = os.path.join(current_app.static_folder, 'uploads', 'topics')
        os.makedirs(upload_folder, exist_ok=True)
        
        # 保存文件
        file_path = os.path.join(upload_folder, safe_filename)
        file.save(file_path)
        
        # 返回文件URL
        url = f'/static/uploads/topics/{safe_filename}'
        
        respond = {
            'url': url,
            'alt': original_filename,
            'href': url
        }
        return make_response(data=respond)
    except Exception as e:
        return make_response(code=500, message=f'文件上传失败: {str(e)}')