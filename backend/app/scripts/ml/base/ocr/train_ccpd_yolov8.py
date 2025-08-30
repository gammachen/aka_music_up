#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用CCPD数据集训练YOLOv8模型进行车牌检测

数据集路径: /User/shhaofu/Downloads/CCPD2020/ccpd_green
数据集格式: 图片文件名包含标注信息
例如: 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg

训练后的模型保存为best.pt
"""

import os
import shutil
import random
from pathlib import Path
import yaml
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# 定义常量
CCPD_ROOT = "/User/shhaofu/Downloads/CCPD2020/ccpd_green"
OUTPUT_DIR = "./ccpd_yolo_dataset"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 定义车牌字符映射字典
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def parse_ccpd_filename(filename):
    """
    解析CCPD数据集的文件名，提取标注信息
    
    Args:
        filename: CCPD数据集的文件名
        
    Returns:
        bbox: 车牌边界框坐标 [x_min, y_min, x_max, y_max]
        points: 车牌四个角点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        plate_number: 车牌号码
    """
    parts = filename.split('-')
    
    # 提取边界框坐标
    bbox_str = parts[2].split('_')
    x_min, y_min = map(int, bbox_str[0].split('&'))
    x_max, y_max = map(int, bbox_str[1].split('&'))
    bbox = [x_min, y_min, x_max, y_max]
    
    # 提取四个角点坐标
    points_str = parts[3].split('_')
    points = []
    for point in points_str:
        x, y = map(int, point.split('&'))
        points.append([x, y])
    
    # 提取车牌号码
    plate_idx = parts[4].split('_')
    province_idx = int(plate_idx[0])
    alphabet_idx = int(plate_idx[1])
    ads_idx = [int(idx) for idx in plate_idx[2:]]
    
    plate_number = PROVINCES[province_idx] + ALPHABETS[alphabet_idx]
    for idx in ads_idx:
        plate_number += ADS[idx]
    
    return bbox, points, plate_number


def convert_to_yolo_format(img_path, output_dir, split):
    """
    将CCPD数据集转换为YOLO格式
    
    Args:
        img_path: 图片路径
        output_dir: 输出目录
        split: 数据集划分（train/val/test）
    """
    filename = os.path.basename(img_path)
    
    try:
        # 解析文件名获取标注信息
        bbox, points, plate_number = parse_ccpd_filename(filename)
        
        # 读取图片获取宽高
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            return
        
        height, width, _ = img.shape
        
        # 计算YOLO格式的边界框（归一化坐标）
        x_min, y_min, x_max, y_max = bbox
        x_center = (x_min + x_max) / (2 * width)
        y_center = (y_min + y_max) / (2 * height)
        box_width = (x_max - x_min) / width
        box_height = (y_max - y_min) / height
        
        # 创建YOLO格式的标注文件
        img_output_path = os.path.join(output_dir, split, 'images', filename)
        label_output_path = os.path.join(output_dir, split, 'labels', os.path.splitext(filename)[0] + '.txt')
        
        # 复制图片到目标目录
        os.makedirs(os.path.dirname(img_output_path), exist_ok=True)
        shutil.copy(img_path, img_output_path)
        
        # 写入标注文件（类别为0，表示车牌）
        os.makedirs(os.path.dirname(label_output_path), exist_ok=True)
        with open(label_output_path, 'w') as f:
            f.write(f"0 {x_center} {y_center} {box_width} {box_height}\n")
            
    except Exception as e:
        print(f"处理文件 {filename} 时出错: {e}")


def create_dataset_yaml(output_dir):
    """
    创建数据集的YAML配置文件
    
    Args:
        output_dir: 输出目录
    """
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {
            0: 'license_plate'
        }
    }
    
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)


def prepare_dataset():
    """
    准备YOLO格式的数据集
    """
    print("开始准备数据集...")
    
    # 创建输出目录
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'), exist_ok=True)
    
    # 收集所有数据集目录
    all_images = []
    for split_dir in ['train', 'val', 'test']:
        split_path = os.path.join(CCPD_ROOT, split_dir)
        if os.path.exists(split_path):
            for img_file in os.listdir(split_path):
                if img_file.endswith('.jpg'):
                    all_images.append(os.path.join(split_path, img_file))
    
    # 如果数据集已经分割好了，直接使用原始分割
    if all_images:
        print(f"使用原始数据集分割，共找到 {len(all_images)} 张图片")
        
        # 处理训练集
        train_dir = os.path.join(CCPD_ROOT, 'train')
        if os.path.exists(train_dir):
            for img_file in tqdm(os.listdir(train_dir), desc="处理训练集"):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(train_dir, img_file)
                    convert_to_yolo_format(img_path, OUTPUT_DIR, 'train')
        
        # 处理验证集
        val_dir = os.path.join(CCPD_ROOT, 'val')
        if os.path.exists(val_dir):
            for img_file in tqdm(os.listdir(val_dir), desc="处理验证集"):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(val_dir, img_file)
                    convert_to_yolo_format(img_path, OUTPUT_DIR, 'val')
        
        # 处理测试集
        test_dir = os.path.join(CCPD_ROOT, 'test')
        if os.path.exists(test_dir):
            for img_file in tqdm(os.listdir(test_dir), desc="处理测试集"):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(test_dir, img_file)
                    convert_to_yolo_format(img_path, OUTPUT_DIR, 'test')
    else:
        # 如果没有预先分割，则自己进行分割
        print("未找到预分割的数据集，将自动进行分割")
        
        # 收集所有图片
        all_images = []
        for img_file in os.listdir(CCPD_ROOT):
            if img_file.endswith('.jpg'):
                all_images.append(os.path.join(CCPD_ROOT, img_file))
        
        # 随机打乱
        random.shuffle(all_images)
        
        # 计算分割点
        num_images = len(all_images)
        train_end = int(num_images * TRAIN_RATIO)
        val_end = train_end + int(num_images * VAL_RATIO)
        
        # 分割数据集
        train_images = all_images[:train_end]
        val_images = all_images[train_end:val_end]
        test_images = all_images[val_end:]
        
        # 处理训练集
        for img_path in tqdm(train_images, desc="处理训练集"):
            convert_to_yolo_format(img_path, OUTPUT_DIR, 'train')
        
        # 处理验证集
        for img_path in tqdm(val_images, desc="处理验证集"):
            convert_to_yolo_format(img_path, OUTPUT_DIR, 'val')
        
        # 处理测试集
        for img_path in tqdm(test_images, desc="处理测试集"):
            convert_to_yolo_format(img_path, OUTPUT_DIR, 'test')
    
    # 创建数据集YAML文件
    create_dataset_yaml(OUTPUT_DIR)
    
    print("数据集准备完成！")


def train_yolov8():
    """
    使用YOLOv8训练车牌检测模型
    """
    print("开始训练YOLOv8模型...")
    
    # 加载预训练模型
    model = YOLO('yolov8n.pt')  # 使用YOLOv8n作为基础模型
    
    # 开始训练
    results = model.train(
        data=os.path.join(OUTPUT_DIR, 'dataset.yaml'),
        epochs=100,  # 训练轮次
        imgsz=640,   # 图像大小
        batch=16,    # 批次大小
        name='ccpd_license_plate',  # 实验名称
        patience=15,  # 早停耐心值
        save=True,    # 保存模型
        device='0',   # 使用GPU
        project='ccpd_yolo_runs'  # 项目名称
    )
    
    # 复制最佳模型到当前目录
    best_model_path = model.best
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, 'best.pt')
        print(f"最佳模型已保存为 best.pt")
    else:
        print(f"警告：未找到最佳模型 {best_model_path}")
    
    print("训练完成！")


def validate_model():
    """
    验证训练好的模型
    """
    print("开始验证模型...")
    
    # 加载训练好的模型
    model = YOLO('best.pt')
    
    # 在测试集上验证
    results = model.val(
        data=os.path.join(OUTPUT_DIR, 'dataset.yaml'),
        split='test'
    )
    
    print(f"验证结果: mAP@0.5 = {results.box.map50:.4f}, mAP@0.5:0.95 = {results.box.map:.4f}")


def test_on_image(image_path):
    """
    在单张图片上测试模型
    
    Args:
        image_path: 测试图片路径
    """
    # 加载训练好的模型
    model = YOLO('best.pt')
    
    # 预测
    results = model(image_path)
    
    # 显示结果
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            print(f"检测到车牌: 置信度={conf:.4f}, 类别={cls}")
    
    # 保存标注后的图片
    result_img = results[0].plot()
    output_path = os.path.join(os.path.dirname(image_path), 'result_' + os.path.basename(image_path))
    cv2.imwrite(output_path, result_img)
    print(f"结果已保存到 {output_path}")


def main():
    """
    主函数
    """
    # 准备数据集
    prepare_dataset()
    
    # 训练模型
    train_yolov8()
    
    # 验证模型
    validate_model()
    
    # 测试模型（可选）
    test_dir = os.path.join(OUTPUT_DIR, 'test', 'images')
    if os.path.exists(test_dir) and os.listdir(test_dir):
        test_image = os.path.join(test_dir, os.listdir(test_dir)[0])
        test_on_image(test_image)


if __name__ == "__main__":
    main()