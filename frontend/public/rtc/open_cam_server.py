from flask import Flask, request, jsonify
from datetime import datetime
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/cap_picture', methods=['POST'])
def capture_picture():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{timestamp}.jpg"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)
    
    # 返回响应
    return jsonify({
        "status": "success",
        "message": "Image received and saved",
        "filename": filename,
        "path": save_path
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)