import cv2
import subprocess
import os

# 配置参数
RTMP_URL = "rtmp://192.168.31.150:1935/live/test"
CAMERA_INDEX = 0  # 后置摄像头通常为0，前置为1
WIDTH, HEIGHT = 640, 480
FPS = 15  # 降低帧率以适应移动设备性能

def grant_camera_permission():
    """确保有摄像头权限"""
    os.system("termux-camera-photo -c 0 /dev/null >/dev/null 2>&1")

def get_ffmpeg_command():
    """返回FFmpeg推流命令"""
    return [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{WIDTH}x{HEIGHT}',
        '-r', str(FPS),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-g', str(FPS * 2),  # 关键帧间隔(2秒)
        '-b:v', '500k',      # 降低比特率
        '-f', 'flv',
        RTMP_URL
    ]

def main():
    grant_camera_permission()
    
    # 打开摄像头
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    
    if not cap.isOpened():
        print("无法打开摄像头，请检查权限和设备")
        return
    
    # 启动FFmpeg进程
    ffmpeg_cmd = get_ffmpeg_command()
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    
    print(f"开始推流到 {RTMP_URL} (按Ctrl+C停止)...")
    print("摄像头参数:", f"{WIDTH}x{HEIGHT}@{FPS}fps")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("读取帧失败，尝试重新打开摄像头...")
                cap.release()
                cap = cv2.VideoCapture(CAMERA_INDEX)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, FPS)
                continue
            
            # 竖屏设备需要旋转图像（根据实际设备调整）
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            # 将帧写入FFmpeg
            process.stdin.write(frame.tobytes())
            
    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        # 清理资源
        cap.release()
        process.stdin.close()
        process.terminate()
        process.wait()
        print("推流已停止")

if __name__ == "__main__":
    main()