import cv2
import numpy as np
import os
import torch

def load_yolov8_face_model(model_path=None):
    """
    加载YOLOv8-Face模型用于人脸检测
    
    参数:
        model_path: 模型文件路径，如果为None则使用默认路径
    
    返回:
        加载好的YOLOv8-Face模型
    """
    from ultralytics import YOLO
    
    # 如果未指定模型路径，使用默认路径
    if model_path is None:
        # 默认模型目录和文件名
        model_dir = os.path.join(os.path.expanduser("~"), ".yolov8face")
        model_path = os.path.join(model_dir, "yolov8n-face.pt")
        
        # 确保模型目录存在
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    # 检查模型文件是否存在
    if os.path.exists(model_path):
        print(f"找到本地YOLOv8-Face模型: {model_path}")
        try:
            # 加载模型
            model = YOLO(model_path)
            print("成功加载YOLOv8-Face模型")
            return model
        except Exception as e:
            print(f"加载YOLOv8-Face模型失败: {e}")
    else:
        print(f"本地YOLOv8-Face模型不存在: {model_path}")
        try:
            # 尝试从在线资源加载模型
            print("尝试从在线资源加载YOLOv8-Face模型...")
            # 这里可以添加从在线资源下载模型的代码
            # 例如使用ultralytics提供的预训练模型
            model = YOLO('yolov8n-face.pt')
            
            # 如果加载成功，保存模型到本地
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            try:
                # 保存模型
                model.save(model_path)
                print(f"已将模型保存到本地: {model_path}")
            except Exception as save_err:
                print(f"模型保存失败: {save_err}")
                
            return model
        except Exception as e:
            print(f"从在线资源加载YOLOv8-Face模型失败: {e}")
            return None

def detect_faces(image, model, conf_threshold=0.5):
    """
    使用YOLOv8-Face模型检测图像中的人脸
    
    参数:
        image: 输入图像
        model: 已加载的YOLOv8-Face模型
        conf_threshold: 置信度阈值
    
    返回:
        检测到的人脸列表，每个人脸为(x, y, w, h, conf)格式
    """
    if model is None:
        print("人脸检测模型未加载，无法进行人脸检测")
        return []
    
    faces = []
    
    # 使用模型进行检测
    results = model(image, conf=conf_threshold, verbose=False)
    
    # 处理检测结果
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            # 获取置信度
            conf = box.conf[0].item()
            
            # 添加到人脸列表
            faces.append((x1, y1, x2-x1, y2-y1, conf))
            
            # 在图像上绘制人脸边界框
            color = (255, 0, 0)  # 蓝色
            label = f"Face: {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return faces

def detect_faces_in_video_frame(frame, model, conf_threshold=0.5):
    """
    在视频帧中检测人脸
    
    参数:
        frame: 视频帧
        model: 已加载的YOLOv8-Face模型
        conf_threshold: 置信度阈值
    
    返回:
        检测到的人脸列表，每个人脸为(x, y, w, h, conf)格式
    """
    return detect_faces(frame, model, conf_threshold)