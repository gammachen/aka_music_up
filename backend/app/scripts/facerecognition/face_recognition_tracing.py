import cv2
import datetime
import os
import glob
from tqdm import tqdm

def smart_surveillance(video_source=0, output_dir="detected_faces"):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 初始化视频捕获
    cap = cv2.VideoCapture(video_source)  # 0表示默认摄像头
    
    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 初始化变量
    face_detected = False
    last_save_time = datetime.datetime.now()
    min_save_interval = datetime.timedelta(seconds=5)  # 每5秒最多保存一次
    frame_count = 0
    
    print("智能监控系统已启动。按'q'键退出。")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        # 每3帧处理一次（提高性能）
        if frame_count % 3 != 0:
            continue
            
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 在画面上显示时间
        cv2.putText(frame, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 处理检测结果
        if len(faces) > 0:
            face_detected = True
            # 在画面上标记人脸并显示数量
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            cv2.putText(frame, f"Reco {len(faces)} Faces", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 定期保存检测到人脸的图片
            if current_time - last_save_time > min_save_interval:
                filename = os.path.join(output_dir, f"face_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(filename, frame)
                print(f"[{time_str}] 检测到 {len(faces)} 个人脸，已保存到 {filename}")
                last_save_time = current_time
        else:
            if face_detected:
                print(f"[{time_str}] 人脸已离开画面")
                face_detected = False
        
        # 显示结果
        cv2.imshow('智能监控', frame)
        
        # 按q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("监控系统已关闭")

def detect_faces_in_images(input_dir, output_dir="detected_faces_images"):
    """
    检测多张图片中的人脸并标注
    
    参数:
        input_dir: 输入图片目录或图片路径列表
        output_dir: 输出标注后图片的目录
    
    返回:
        标注后的图片保存路径列表
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 处理输入路径
    if isinstance(input_dir, str):
        if os.path.isdir(input_dir):
            image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                         glob.glob(os.path.join(input_dir, "*.jpeg")) + \
                         glob.glob(os.path.join(input_dir, "*.png"))
        else:
            image_paths = [input_dir]
    else:
        image_paths = input_dir
    
    output_paths = []
    
    print(f"开始处理 {len(image_paths)} 张图片...")
    
    # 遍历处理每张图片
    for img_path in tqdm(image_paths):
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # 在图片上标记人脸
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # 添加检测到的人脸数量信息
        cv2.putText(img, f"Reco {len(faces)} Faces", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 保存标注后的图片
        base_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"detected_{base_name}")
        cv2.imwrite(output_path, img)
        output_paths.append(output_path)
        
        print(f"图片 {base_name} 中检测到 {len(faces)} 个人脸，已保存到 {output_path}")
    
    print(f"所有图片处理完成，结果保存在 {output_dir} 目录")
    return output_paths

def analyze_video_faces(video_path, output_dir="video_face_analysis", sample_interval=30):
    """
    分析视频文件中的人脸并统计人数
    
    参数:
        video_path: 视频文件路径
        output_dir: 输出目录，用于保存截图和分析结果
        sample_interval: 采样间隔（帧数），默认每30帧采样一次
    
    返回:
        包含时间戳和人脸数量的字典列表
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 初始化视频捕获
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return []
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"视频信息: {frame_count} 帧, {fps:.2f} FPS, 时长 {duration:.2f} 秒")
    
    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 初始化结果列表
    results = []
    frame_idx = 0
    
    # 创建进度条
    pbar = tqdm(total=frame_count)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        pbar.update(1)
        
        # 按指定间隔采样
        if frame_idx % sample_interval != 0:
            continue
        
        # 计算当前时间点
        time_sec = frame_idx / fps if fps > 0 else 0
        time_str = str(datetime.timedelta(seconds=int(time_sec)))
        
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        face_count = len(faces)
        
        # 记录结果
        result = {
            "frame": frame_idx,
            "time": time_str,
            "time_seconds": time_sec,
            "face_count": face_count
        }
        results.append(result)
        
        # 在画面上标记人脸
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # 添加时间和人脸数量信息
        # cv2.putText(frame, f"时间: {time_str}", (10, 30), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Reco {face_count} Faces", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 每检测到人脸时保存截图
        if face_count > 0:
            filename = os.path.join(output_dir, f"frame_{frame_idx}_faces_{face_count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[{time_str}] 检测到 {face_count} 个人脸，已保存到 {filename}")
    
    # 关闭进度条
    pbar.close()
    
    # 释放资源
    cap.release()
    
    # 输出统计信息
    if results:
        max_faces = max(results, key=lambda x: x["face_count"])
        print(f"分析完成! 视频中最多同时出现 {max_faces['face_count']} 个人脸，出现在 {max_faces['time']}")
        
        # 计算平均人脸数
        total_faces = sum(r["face_count"] for r in results)
        avg_faces = total_faces / len(results) if results else 0
        print(f"平均每帧检测到 {avg_faces:.2f} 个人脸")
    else:
        print("未检测到任何人脸")
    
    print(f"视频分析完成，结果保存在 {output_dir} 目录")
    return results

def detect_persons_in_images(input_dir, output_dir="detected_persons_images"):
    """
    检测多张图片中的人体并标注
    
    参数:
        input_dir: 输入图片目录或图片路径列表
        output_dir: 输出标注后图片的目录
    
    返回:
        标注后的图片保存路径列表
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载HOG人体检测器
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # 处理输入路径
    if isinstance(input_dir, str):
        if os.path.isdir(input_dir):
            image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                         glob.glob(os.path.join(input_dir, "*.jpeg")) + \
                         glob.glob(os.path.join(input_dir, "*.png"))
        else:
            image_paths = [input_dir]
    else:
        image_paths = input_dir
    
    output_paths = []
    
    print(f"开始处理 {len(image_paths)} 张图片...")
    
    # 遍历处理每张图片
    for img_path in tqdm(image_paths):
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue
        
        # 调整图像大小以提高检测速度和准确性
        height, width = img.shape[:2]
        if width > 800:
            scale = 800 / width
            img = cv2.resize(img, (int(width * scale), int(height * scale)))
        
        # 检测人体
        # 使用HOG检测器检测人体
        # winStride - 步长，padding - 填充，scale - 缩放比例
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8), padding=(4, 4), scale=1.05)
        
        # 在图片上标记人体
        for (x, y, w, h) in boxes:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 添加检测到的人体数量信息
        cv2.putText(img, f"Reco {len(boxes)} Bodys", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 保存标注后的图片
        base_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"detected_person_{base_name}")
        cv2.imwrite(output_path, img)
        output_paths.append(output_path)
        
        print(f"图片 {base_name} 中检测到 {len(boxes)} 个人体，已保存到 {output_path}")
    
    print(f"所有图片处理完成，结果保存在 {output_dir} 目录")
    return output_paths

def analyze_video_bodys(video_path, output_dir="video_body_analysis", sample_interval=30):
    """
    分析视频文件中的人体并统计人流量
    
    参数:
        video_path: 视频文件路径
        output_dir: 输出目录，用于保存截图和分析结果
        sample_interval: 采样间隔（帧数），默认每30帧采样一次
    
    返回:
        包含时间戳和人体数量的字典列表
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 初始化视频捕获
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return []
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"视频信息: {frame_count} 帧, {fps:.2f} FPS, 时长 {duration:.2f} 秒")
    
    # 加载HOG人体检测器
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # 初始化结果列表
    results = []
    frame_idx = 0
    
    # 创建进度条
    pbar = tqdm(total=frame_count)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        pbar.update(1)
        
        # 按指定间隔采样
        if frame_idx % sample_interval != 0:
            continue
        
        # 计算当前时间点
        time_sec = frame_idx / fps if fps > 0 else 0
        time_str = str(datetime.timedelta(seconds=int(time_sec)))
        
        # 调整图像大小以提高检测速度和准确性
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800 / width
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        
        # 检测人体
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
        body_count = len(boxes)
        
        # 记录结果
        result = {
            "frame": frame_idx,
            "time": time_str,
            "time_seconds": time_sec,
            "body_count": body_count
        }
        results.append(result)
        
        # 在画面上标记人体
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 添加时间和人体数量信息
        cv2.putText(frame, f"Reco {body_count} Bodys", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 每检测到人体时保存截图
        if body_count > 0:
            filename = os.path.join(output_dir, f"frame_{frame_idx}_bodys_{body_count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[{time_str}] 检测到 {body_count} 个人体，已保存到 {filename}")
    
    # 关闭进度条
    pbar.close()
    
    # 释放资源
    cap.release()
    
    # 输出统计信息
    if results:
        max_bodys = max(results, key=lambda x: x["body_count"])
        print(f"分析完成! 视频中最多同时出现 {max_bodys['body_count']} 个人体，出现在 {max_bodys['time']}")
        
        # 计算平均人体数
        total_bodys = sum(r["body_count"] for r in results)
        avg_bodys = total_bodys / len(results) if results else 0
        print(f"平均每帧检测到 {avg_bodys:.2f} 个人体")
        
        # 分析人流趋势
        if len(results) > 1:
            time_points = [r["time_seconds"] for r in results]
            body_counts = [r["body_count"] for r in results]
            
            # 计算人流变化率
            changes = [body_counts[i] - body_counts[i-1] for i in range(1, len(body_counts))]
            avg_change = sum(abs(c) for c in changes) / len(changes) if changes else 0
            print(f"人流量平均变化率: {avg_change:.2f} 人/采样")
            
            # 找出人流量突增和突减的时间点
            significant_changes = [(results[i+1]["time"], changes[i]) 
                                for i in range(len(changes)) 
                                if abs(changes[i]) > avg_change * 2]
            
            if significant_changes:
                print("显著人流量变化时间点:")
                for time, change in significant_changes:
                    change_type = "增加" if change > 0 else "减少"
                    print(f"  - {time}: {change_type} {abs(change)} 人")
    else:
        print("未检测到任何人体")
    
    print(f"视频分析完成，结果保存在 {output_dir} 目录")
    return results

# 运行示例
if __name__ == "__main__":
    # smart_surveillance()
    
    # detect_faces_in_images("office")
    detect_persons_in_images("office", output_dir="detected_office")  # 取消注释以运行人体检测
    
    detect_persons_in_images("office_2", output_dir="detected_office_2")  # 取消注释以运行人体检测

    detect_faces_in_images("office", output_dir="detected_faces_office")  # 取消注释以运行人体检测

    detect_faces_in_images("office_2", output_dir="detected_faces_office_2")  # 取消注释以运行人体检测
    
    detect_faces_in_images("persons", output_dir="detected_faces_persons")  # 取消注释以运行人体检测
    
    analyze_video_faces("sheng_ri_face_video.mp4", output_dir="detected_sheng_ri_face_video")
    
    analyze_video_faces("/Users/shhaofu/Downloads/xin_wen_lian_bo.mp4", output_dir="xin_wen_lian_bo_face_video")
    '''
    分析完成! 视频中最多同时出现 6 个人体，出现在 0:01:34
    平均每帧检测到 1.55 个人体
    人流量平均变化率: 1.07 人/采样
    显著人流量变化时间点:
    - 0:00:07: 增加 3 人
    - 0:00:08: 减少 4 人
    - 0:00:16: 增加 3 人
    - 0:00:18: 减少 3 人
    - 0:00:55: 减少 3 人
    - 0:00:56: 增加 3 人
    - 0:01:15: 增加 3 人
    - 0:01:28: 减少 3 人
    - 0:01:31: 增加 4 人
    - 0:01:34: 增加 3 人
    - 0:01:42: 减少 3 人
    - 0:02:25: 增加 3 人
    - 0:02:27: 减少 3 人
    - 0:02:33: 增加 5 人
    - 0:02:37: 减少 3 人
    '''
    # 使用新函数分析视频中的人体
    analyze_video_bodys("/Users/shhaofu/Downloads/shang_chang_ren_liu_1080p.mp4", output_dir="shang_chang_ren_liu_1080p_body_video")
    
    
    
    