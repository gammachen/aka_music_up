import cv2
import requests
import time
import os

# 配置参数
SERVER_URL = "http://192.168.31.150:5001/cap_picture"
CAMERA_INDEX = 0  # 后置摄像头通常为0，前置为1
CAPTURE_INTERVAL = 5  # 捕获间隔（秒）
MAX_RETRIES = 3  # 最大重试次数

def grant_camera_permission():
    """确保有摄像头权限"""
    # os.system("termux-camera-photo -c 0 /dev/null >/dev/null 2>&1")
    print("已授予摄像头权限")

def capture_frame(cap):
    """捕获一帧图像并返回JPEG字节流"""
    ret, frame = cap.read()
    if not ret:
        return None
    
    # 竖屏设备可能需要旋转图像（根据实际设备调整）
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    # 将图像转换为JPEG格式
    _, jpeg_data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg_data.tobytes()

def send_image_to_server(image_data):
    """发送图片到服务器"""
    try:
        files = {'image': ('capture.jpg', image_data, 'image/jpeg')}
        response = requests.post(SERVER_URL, files=files, timeout=10)
        
        if response.status_code == 200:
            print(f"图片发送成功! 响应: {response.text}")
            return True
        else:
            print(f"服务器返回错误: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"发送失败: {str(e)}")
        return False

def main():
    grant_camera_permission()
    
    # 打开摄像头
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("无法打开摄像头，请检查权限和设备")
        return
    
    print(f"开始捕获图片，每{CAPTURE_INTERVAL}秒发送到服务器...")
    print("按Ctrl+C停止程序")
    
    try:
        while True:
            # 捕获图像
            image_data = capture_frame(cap)
            if image_data is None:
                print("捕获图像失败，等待重试...")
                time.sleep(1)
                continue
            
            # 发送图像（带重试机制）
            success = False
            for attempt in range(MAX_RETRIES):
                print(f"尝试发送图片 ({attempt+1}/{MAX_RETRIES})")
                if send_image_to_server(image_data):
                    success = True
                    break
                time.sleep(1)
            
            if not success:
                print(f"发送失败，已达到最大重试次数 ({MAX_RETRIES})")
            
            # 等待下一次捕获
            time.sleep(CAPTURE_INTERVAL)
            
    except KeyboardInterrupt:
        print("用户中断")
    finally:
        # 清理资源
        cap.release()
        print("摄像头已关闭")

if __name__ == "__main__":
    main()