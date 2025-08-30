import cv2
import datetime
import os
import glob
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np

# 加载YOLO模型
def load_yolo_model(model_type="yolov8n.pt", face_detection=False):
    """
    加载YOLO模型
    
    参数:
        model_type: 模型类型，可选值包括'yolov8n.pt'(最小),'yolov8s.pt'(小),'yolov8m.pt'(中),'yolov8l.pt'(大),'yolov8x.pt'(超大)
        face_detection: 是否使用专门的人脸检测模型，如果为True，将使用YOLOv8-Face
    
    返回:
        加载好的YOLO模型
    """
    try:
        if face_detection:
            # 使用专门的人脸检测模块加载YOLOv8-Face模型
            from face_detection import load_yolov8_face_model
            model = load_yolov8_face_model()
            if model is not None:
                print("成功加载YOLOv8-Face模型")
                return model
            else:
                print("警告: 无法加载YOLOv8-Face模型，将尝试使用标准YOLOv8模型")
                # 如果无法加载专门的人脸检测模型，尝试使用标准YOLOv8模型
                model = YOLO(model_type)
                print(f"成功加载标准YOLOv8模型: {model_type}")
                return model
        else:
            # 使用标准YOLOv8模型进行人体和物体检测
            model = YOLO(model_type)
            print(f"成功加载YOLOv8模型: {model_type}")
            return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

def smart_surveillance_yolo(video_source=0, output_dir="detected_faces_yolo", conf_threshold=0.5):
    """
    使用YOLOv8进行智能监控，检测人脸和人体
    
    参数:
        video_source: 视频源，可以是摄像头索引或视频文件路径
        output_dir: 输出目录，用于保存检测结果
        conf_threshold: 置信度阈值，只有置信度高于此值的检测结果才会被保留
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 初始化视频捕获
    cap = cv2.VideoCapture(video_source)  # 0表示默认摄像头
    if not cap.isOpened():
        print(f"无法打开视频源: {video_source}")
        return
    
    # 加载专门的人脸检测模型
    face_model = load_yolo_model(face_detection=True)
    if face_model is None:
        print("人脸检测模型加载失败，无法继续")
        return
    
    # 加载标准模型用于检测人体和其他物体
    standard_model = load_yolo_model(face_detection=False)
    
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
        
        # 获取当前时间
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 在画面上显示时间
        cv2.putText(frame, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 处理检测结果
        faces = []
        persons = []
        objects = []
        
        # 使用专门的人脸检测模型检测人脸
        from face_detection import detect_faces
        faces = detect_faces(frame, face_model, conf_threshold)
        
        # 使用标准模型检测人体和其他物体
        if standard_model is not None:
            results = standard_model(frame, conf=conf_threshold, verbose=False)  # 检测所有类别
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    # 获取类别和置信度
                    cls = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    cls_name = standard_model.names[cls]
                    
                    # 根据类别分类
                    if cls_name == "person" or cls == 0:  # 人
                        color = (0, 255, 0)  # 绿色
                        persons.append((x1, y1, x2-x1, y2-y1, conf))
                        label = f"Person: {conf:.2f}"
                    else:  # 其他物体
                        color = (0, 255, 255)  # 黄色
                        objects.append((x1, y1, x2-x1, y2-y1, conf))
                        label = f"{cls_name}: {conf:.2f}"
                    
                    # 绘制边界框和标签
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 显示检测到的人脸、人数和物体数量
        face_count = len(faces)
        person_count = len(persons)
        object_count = len(objects)
        
        if face_count > 0 or person_count > 0 or object_count > 0:
            detection_text = f"Reco {face_count} Faces, {person_count} Persons, {object_count} Objects"
            cv2.putText(frame, detection_text, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 如果检测到人脸，更新状态并定期保存图片
            if face_count > 0 or person_count > 0 or object_count > 0:
                face_detected = True
                if current_time - last_save_time > min_save_interval:
                    filename = os.path.join(output_dir, f"face_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"[{time_str}] 检测到 {face_count} 个人脸，{person_count} 个人体，{object_count} 个物体，已保存到 {filename}")
                    last_save_time = current_time
        else:
            if face_detected:
                print(f"[{time_str}] 人脸已离开画面")
                face_detected = False
        
        # 显示结果
        cv2.imshow('YOLO智能监控', frame)
        
        # 按q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("监控系统已关闭")

def detect_objects_in_images(input_dir, output_dir="detected_objects_yolo", conf_threshold=0.5, classes=None):
    """
    使用YOLOv8检测多张图片中的目标并标注
    
    参数:
        input_dir: 输入图片目录或图片路径列表
        output_dir: 输出标注后图片的目录
        conf_threshold: 置信度阈值
        classes: 要检测的类别列表，默认为None表示检测所有类别。例如[0, 1]表示只检测人和脸
    
    返回:
        标注后的图片保存路径列表
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载专门的人脸检测模型
    face_model = load_yolo_model(face_detection=True)
    if face_model is None:
        print("人脸检测模型加载失败，无法继续")
        return []
    
    # 加载标准模型用于检测人体和其他物体
    standard_model = load_yolo_model(face_detection=False)
    
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
    
    # 导入人脸检测模块
    from face_detection import detect_faces
    
    # 遍历处理每张图片
    for img_path in tqdm(image_paths):
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue
        
        # 处理检测结果
        face_count = 0
        person_count = 0
        object_count = 0
        
        # 使用专门的人脸检测模型检测人脸
        faces = detect_faces(img, face_model, conf_threshold)
        face_count = len(faces)
        
        # 使用标准模型检测人体和其他物体
        if standard_model is not None:
            results = standard_model(img, conf=conf_threshold, classes=classes, verbose=False)
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    # 获取类别和置信度
                    cls = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    cls_name = standard_model.names[cls]
                    
                    # 根据类别绘制不同颜色的边界框
                    if cls == 0 or cls_name == "person":  # 人
                        color = (0, 255, 0)  # 绿色
                        person_count += 1
                        label = f"Person: {conf:.2f}"
                        
                        # 绘制边界框和标签
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    elif cls != 1 and cls_name != "face" and cls_name != "head":  # 排除人脸，因为已经由专门的模型检测
                        # 其他类别
                        color = (0, 255, 255)  # 黄色
                        object_count += 1
                        label = f"{cls_name}: {conf:.2f}"
                        
                        # 绘制边界框和标签
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 添加检测到的目标数量信息
        detection_text = f"Reco {face_count} Faces, {person_count} Persons, {object_count} Objects"
        cv2.putText(img, detection_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 保存标注后的图片
        base_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"yolo_detected_{base_name}")
        cv2.imwrite(output_path, img)
        output_paths.append(output_path)
        
        print(f"图片 {base_name} 中检测到 {face_count} 个人脸，{person_count} 个人体，{object_count} 个物体，已保存到 {output_path}")
    
    print(f"所有图片处理完成，结果保存在 {output_dir} 目录")
    return output_paths

def analyze_video_with_yolo(video_path, output_dir="video_yolo_analysis", sample_interval=30, conf_threshold=0.5, classes=None):
    """
    使用YOLOv8分析视频文件中的目标并统计数量
    
    参数:
        video_path: 视频文件路径
        output_dir: 输出目录，用于保存截图和分析结果
        sample_interval: 采样间隔（帧数），默认每30帧采样一次
        conf_threshold: 置信度阈值
        classes: 要检测的类别列表，默认为None表示检测所有类别
    
    返回:
        包含时间戳和目标数量的字典列表
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
    
    # 加载专门的人脸检测模型
    face_model = load_yolo_model(face_detection=True)
    if face_model is None:
        print("人脸检测模型加载失败，无法继续")
        return []
    
    # 加载标准模型用于检测人体和其他物体
    standard_model = load_yolo_model(face_detection=False)
    
    # 导入人脸检测模块
    from face_detection import detect_faces
    
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
        
        # 处理检测结果
        face_count = 0
        person_count = 0
        object_count = 0
        
        # 使用专门的人脸检测模型检测人脸
        faces = detect_faces(frame, face_model, conf_threshold)
        face_count = len(faces)
        
        # 使用标准模型检测人体和其他物体
        if standard_model is not None:
            detections = standard_model(frame, conf=conf_threshold, classes=classes, verbose=False)
            
            for r in detections:
                boxes = r.boxes
                for box in boxes:
                    # 获取类别
                    cls = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    cls_name = standard_model.names[cls]
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # 根据类别统计和绘制
                    if cls == 0 or cls_name == "person":  # 人
                        person_count += 1
                        color = (0, 255, 0)  # 绿色
                        label = f"Person: {conf:.2f}"
                        # 绘制边界框和标签
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    elif cls != 1 and cls_name != "face" and cls_name != "head":  # 排除人脸，因为已经由专门的模型检测
                        object_count += 1
                        color = (0, 255, 255)  # 黄色
                        label = f"{cls_name}: {conf:.2f}"
                        # 绘制边界框和标签
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 记录结果
        result = {
            "frame": frame_idx,
            "time": time_str,
            "time_seconds": time_sec,
            "face_count": face_count,
            "person_count": person_count,
            "object_count": object_count
        }
        results.append(result)
        
        # 添加时间和检测数量信息
        detection_text = f"Reco {face_count} Faces, {person_count} Persons, {object_count} Objects"
        cv2.putText(frame, detection_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 每检测到目标时保存截图
        if face_count > 0 or person_count > 0:
            filename = os.path.join(output_dir, f"frame_{frame_idx}_faces_{face_count}_persons_{person_count}_objects_{object_count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[{time_str}] 检测到 {face_count} 个人脸，{person_count} 个人体，{object_count} 个物体，已保存到 {filename}")
    
    # 关闭进度条
    pbar.close()
    
    # 释放资源
    cap.release()
    
    # 输出统计信息
    if results:
        max_faces = max(results, key=lambda x: x["face_count"])
        max_persons = max(results, key=lambda x: x["person_count"])
        max_objects = max(results, key=lambda x: x["object_count"])
        
        print(f"分析完成! 视频中最多同时出现 {max_faces['face_count']} 个人脸，出现在 {max_faces['time']}")
        print(f"视频中最多同时出现 {max_persons['person_count']} 个人体，出现在 {max_persons['time']}")
        print(f"视频中最多同时出现 {max_objects['object_count']} 个物体，出现在 {max_objects['time']}")
        
        # 计算平均人脸数、人体数和物体数
        total_faces = sum(r["face_count"] for r in results)
        total_persons = sum(r["person_count"] for r in results)
        total_objects = sum(r["object_count"] for r in results)
        avg_faces = total_faces / len(results) if results else 0
        avg_persons = total_persons / len(results) if results else 0
        avg_objects = total_objects / len(results) if results else 0
        
        print(f"平均每帧检测到 {avg_faces:.2f} 个人脸，{avg_persons:.2f} 个人体，{avg_objects:.2f} 个物体")
        
        # 分析人流趋势
        if len(results) > 1:
            person_counts = [r["person_count"] for r in results]
            
            # 计算人流变化率
            changes = [person_counts[i] - person_counts[i-1] for i in range(1, len(person_counts))]
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
        print("未检测到任何目标")
    
    print(f"视频分析完成，结果保存在 {output_dir} 目录")
    return results

# 运行示例
if __name__ == "__main__":
    # 图片检测示例 - 取消注释以运行
    detect_objects_in_images("office", output_dir="yolo_detected_office")
    
    # 检测另一个文件夹中的图片
    detect_objects_in_images("office_2", output_dir="yolo_detected_office_2")
    
    # 检测人脸与人体的示例 (class 1 是人脸)
    detect_objects_in_images("persons", output_dir="yolo_detected_faces_persons", classes=None)
    
    # 只检测人体的示例 (class 0 是人体)
    # detect_objects_in_images("persons", output_dir="yolo_detected_persons", classes=[1])
    # detect_objects_in_images("persons", output_dir="yolo_detected_persons", classes=[0])
    
    # 视频分析示例 - 取消注释以运行
    analyze_video_with_yolo("sheng_ri_face_video.mp4", output_dir="yolo_detected_sheng_ri_face_video")
    
    # 分析另一个视频文件
    analyze_video_with_yolo("/Users/shhaofu/Downloads/xin_wen_lian_bo.mp4", output_dir="yolo_detected_xin_wen_lian_bo")
    
    # 分析人流视频
    analyze_video_with_yolo("/Users/shhaofu/Downloads/shang_chang_ren_liu_1080p.mp4", output_dir="yolo_detected_shang_chang_ren_liu")
    '''
    视频中最多同时出现 14 个人体，出现在 0:06:19
    平均每帧检测到 0.02 个人脸，6.77 个人体
    人流量平均变化率: 1.79 人/采样
    显著人流量变化时间点:
    - 0:00:04: 减少 5 人
    - 0:00:38: 增加 5 人
    - 0:00:40: 减少 4 人
    - 0:00:41: 减少 4 人
    - 0:00:42: 增加 5 人
    - 0:00:45: 减少 4 人
    - 0:00:49: 增加 7 人
    - 0:01:12: 减少 5 人
    - 0:01:14: 增加 6 人
    - 0:01:35: 增加 8 人
    - 0:01:38: 减少 5 人
    - 0:01:59: 减少 5 人
    - 0:02:15: 增加 4 人
    - 0:02:25: 增加 4 人
    - 0:02:45: 增加 5 人
    - 0:02:46: 减少 6 人
    - 0:02:55: 减少 4 人
    - 0:02:57: 增加 4 人
    - 0:02:58: 减少 4 人
    - 0:03:09: 增加 4 人
    - 0:03:10: 减少 4 人
    - 0:03:18: 增加 4 人
    - 0:03:20: 减少 4 人
    - 0:03:22: 增加 6 人
    - 0:03:23: 减少 4 人
    - 0:03:24: 减少 5 人
    - 0:03:29: 增加 4 人
    - 0:03:34: 减少 4 人
    - 0:03:36: 增加 4 人
    - 0:03:38: 减少 4 人
    - 0:04:12: 增加 5 人
    - 0:04:13: 减少 4 人
    - 0:04:18: 减少 4 人
    - 0:04:28: 减少 4 人
    - 0:04:29: 增加 4 人
    - 0:04:53: 增加 4 人
    - 0:05:00: 增加 5 人
    - 0:05:13: 增加 6 人
    - 0:05:16: 减少 4 人
    - 0:05:23: 减少 4 人
    - 0:05:24: 增加 6 人
    - 0:05:40: 减少 4 人
    - 0:05:48: 减少 4 人
    - 0:05:54: 减少 4 人
    - 0:05:55: 增加 5 人
    - 0:06:19: 增加 4 人
    - 0:06:21: 减少 5 人
    - 0:06:24: 增加 4 人
    - 0:06:25: 减少 4 人
    - 0:06:27: 减少 4 人
    - 0:06:28: 增加 5 人
    - 0:06:34: 增加 4 人
    - 0:06:40: 增加 6 人
    - 0:06:41: 减少 4 人
    - 0:06:44: 减少 4 人
    - 0:07:12: 增加 4 人
    - 0:07:20: 减少 6 人
    - 0:07:22: 增加 5 人
    - 0:07:31: 减少 4 人
    - 0:07:38: 增加 4 人
    - 0:07:42: 减少 5 人
    - 0:07:43: 增加 5 人
    - 0:07:49: 减少 6 人
    - 0:07:52: 增加 5 人
    - 0:07:53: 减少 6 人
    '''
    
    # 实时监控示例 - 取消注释以运行
    # smart_surveillance_yolo(0, output_dir="yolo_detected_faces_realtime")  # 使用默认摄像头
    
    # 使用视频文件作为监控源
    # smart_surveillance_yolo("sheng_ri_face_video.mp4", output_dir="yolo_surveillance_sheng_ri")
    
    print("\n程序执行完毕！")