#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试训练好的YOLOv8车牌检测模型

使用方法:
    python test_ccpd_model.py --model best.pt --image /path/to/image.jpg
    python test_ccpd_model.py --model best.pt --dir /path/to/images/
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# 定义车牌字符映射字典
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def parse_ccpd_filename(filename):
    """
    解析CCPD数据集的文件名，提取车牌号码
    
    Args:
        filename: CCPD数据集的文件名
        
    Returns:
        plate_number: 车牌号码
    """
    try:
        parts = os.path.basename(filename).split('-')
        if len(parts) < 5:
            return None
        
        # 提取车牌号码
        plate_idx = parts[4].split('_')
        province_idx = int(plate_idx[0])
        alphabet_idx = int(plate_idx[1])
        ads_idx = [int(idx) for idx in plate_idx[2:]]
        
        plate_number = PROVINCES[province_idx] + ALPHABETS[alphabet_idx]
        for idx in ads_idx:
            plate_number += ADS[idx]
        
        return plate_number
    except:
        return None


def detect_license_plate(model, image_path, save_dir=None, conf_threshold=0.25):
    """
    使用YOLOv8模型检测图像中的车牌
    
    Args:
        model: 加载的YOLOv8模型
        image_path: 图像路径
        save_dir: 保存结果的目录，如果为None则不保存
        conf_threshold: 置信度阈值
        
    Returns:
        result_img: 标注后的图像
        detections: 检测结果列表 [(x1, y1, x2, y2, conf), ...]
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None, []
    
    # 获取真实车牌号（如果文件名符合CCPD格式）
    true_plate = parse_ccpd_filename(image_path)
    if true_plate:
        print(f"图像 {os.path.basename(image_path)} 的真实车牌号: {true_plate}")
    
    # 预测
    results = model(image_path, conf=conf_threshold)
    
    # 提取检测结果
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            detections.append((x1, y1, x2, y2, conf))
            print(f"检测到车牌: 坐标=({int(x1)},{int(y1)},{int(x2)},{int(y2)}), 置信度={conf:.4f}")
    
    # 绘制结果
    result_img = results[0].plot()
    
    # 保存结果
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, 'result_' + os.path.basename(image_path))
        cv2.imwrite(output_path, result_img)
        print(f"结果已保存到 {output_path}")
    
    return result_img, detections


def process_directory(model, dir_path, save_dir, conf_threshold=0.25):
    """
    处理目录中的所有图像
    
    Args:
        model: 加载的YOLOv8模型
        dir_path: 图像目录路径
        save_dir: 保存结果的目录
        conf_threshold: 置信度阈值
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(dir_path).glob(f"*{ext}")))
    
    if not image_files:
        print(f"目录 {dir_path} 中未找到图像文件")
        return
    
    print(f"在目录 {dir_path} 中找到 {len(image_files)} 个图像文件")
    
    # 处理每个图像
    for img_path in image_files:
        print(f"\n处理图像: {img_path}")
        detect_license_plate(model, str(img_path), save_dir, conf_threshold)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="测试YOLOv8车牌检测模型")
    parser.add_argument("--model", type=str, default="best.pt", help="模型路径")
    parser.add_argument("--image", type=str, default=None, help="测试图像路径")
    parser.add_argument("--dir", type=str, default=None, help="测试图像目录")
    parser.add_argument("--save_dir", type=str, default="./results", help="保存结果的目录")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    args = parser.parse_args()
    
    # 检查参数
    if args.image is None and args.dir is None:
        print("错误: 必须指定 --image 或 --dir 参数")
        return
    
    # 加载模型
    print(f"加载模型: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 处理单张图像或目录
    if args.image is not None:
        if not os.path.exists(args.image):
            print(f"图像文件不存在: {args.image}")
            return
        
        print(f"处理图像: {args.image}")
        result_img, detections = detect_license_plate(model, args.image, args.save_dir, args.conf)
        
        # 显示结果（可选）
        if result_img is not None:
            cv2.imshow("检测结果", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    elif args.dir is not None:
        if not os.path.isdir(args.dir):
            print(f"目录不存在: {args.dir}")
            return
        
        process_directory(model, args.dir, args.save_dir, args.conf)


if __name__ == "__main__":
    main()