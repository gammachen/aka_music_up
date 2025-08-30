#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
准备CCPD数据集用于YOLOv8训练

使用方法:
    python prepare_ccpd_dataset.py --data_dir /path/to/ccpd_dataset --output_dir /path/to/output

该脚本将:
1. 解析CCPD数据集文件名，提取车牌位置和车牌号码
2. 将数据转换为YOLO格式（归一化坐标的边界框）
3. 自动划分训练集、验证集和测试集
4. 创建YOLOv8所需的数据集配置文件(dataset.yaml)
"""

import os
import sys
import glob
import shutil
import random
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 定义车牌字符映射字典
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def parse_ccpd_filename(filename):
    """
    解析CCPD数据集的文件名，提取车牌位置和车牌号码
    
    Args:
        filename: CCPD数据集的文件名
        
    Returns:
        bbox: 车牌边界框坐标 [x1, y1, x2, y2]
        plate_number: 车牌号码
    """
    try:
        parts = os.path.basename(filename).split('-')
        if len(parts) < 5:
            return None, None
        
        # 提取边界框坐标
        bbox_str = parts[2].split('_')
        if len(bbox_str) != 2:
            return None, None
        
        # 解析左上角和右下角坐标
        top_left = bbox_str[0].split('&')
        bottom_right = bbox_str[1].split('&')
        if len(top_left) != 2 or len(bottom_right) != 2:
            return None, None
        
        x1, y1 = int(top_left[0]), int(top_left[1])
        x2, y2 = int(bottom_right[0]), int(bottom_right[1])
        bbox = [x1, y1, x2, y2]
        
        # 提取车牌号码
        plate_idx = parts[4].split('_')
        if len(plate_idx) < 7:
            return bbox, None
        
        province_idx = int(plate_idx[0])
        alphabet_idx = int(plate_idx[1])
        ads_idx = [int(idx) for idx in plate_idx[2:7]]
        
        plate_number = PROVINCES[province_idx] + ALPHABETS[alphabet_idx]
        for idx in ads_idx:
            plate_number += ADS[idx]
        
        return bbox, plate_number
    except Exception as e:
        print(f"解析文件名 {filename} 时出错: {e}")
        return None, None


def convert_to_yolo_format(image_path, image_width, image_height):
    """
    将CCPD格式的标注转换为YOLO格式
    
    Args:
        image_path: 图像路径
        image_width: 图像宽度
        image_height: 图像高度
        
    Returns:
        yolo_annotation: YOLO格式的标注字符串 "class x_center y_center width height"
    """
    bbox, _ = parse_ccpd_filename(image_path)
    if bbox is None:
        return None
    
    x1, y1, x2, y2 = bbox
    
    # 计算归一化后的中心点坐标和宽高
    x_center = (x1 + x2) / (2 * image_width)
    y_center = (y1 + y2) / (2 * image_height)
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    
    # YOLO格式: class x_center y_center width height
    # 车牌检测只有一个类别，所以class=0
    yolo_annotation = f"0 {x_center} {y_center} {width} {height}"
    
    return yolo_annotation


def prepare_dataset(data_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    准备YOLO格式的数据集
    
    Args:
        data_dir: CCPD数据集目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        
    Returns:
        dataset_yaml: 数据集配置文件路径
    """
    # 确保比例之和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "比例之和必须为1"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建数据集子目录
    dataset_dir = os.path.join(output_dir, "dataset")
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
    
    # 获取所有图像文件
    image_files = []
    for ext in [".jpg", ".jpeg", ".png"]:
        image_files.extend(glob.glob(os.path.join(data_dir, f"*{ext}")))
    
    if not image_files:
        raise ValueError(f"在目录 {data_dir} 中未找到图像文件")
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 检查是否有预先划分的数据集
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    
    if os.path.isdir(train_dir) and os.path.isdir(val_dir) and os.path.isdir(test_dir):
        print("检测到预先划分的数据集，使用现有划分")
        
        # 处理训练集
        train_files = []
        for ext in [".jpg", ".jpeg", ".png"]:
            train_files.extend(glob.glob(os.path.join(train_dir, f"*{ext}")))
        
        # 处理验证集
        val_files = []
        for ext in [".jpg", ".jpeg", ".png"]:
            val_files.extend(glob.glob(os.path.join(val_dir, f"*{ext}")))
        
        # 处理测试集
        test_files = []
        for ext in [".jpg", ".jpeg", ".png"]:
            test_files.extend(glob.glob(os.path.join(test_dir, f"*{ext}")))
        
        print(f"训练集: {len(train_files)} 张图像")
        print(f"验证集: {len(val_files)} 张图像")
        print(f"测试集: {len(test_files)} 张图像")
    else:
        print("未检测到预先划分的数据集，自动划分数据集")
        
        # 随机打乱文件列表
        random.shuffle(image_files)
        
        # 划分数据集
        num_train = int(len(image_files) * train_ratio)
        num_val = int(len(image_files) * val_ratio)
        
        train_files = image_files[:num_train]
        val_files = image_files[num_train:num_train+num_val]
        test_files = image_files[num_train+num_val:]
        
        print(f"训练集: {len(train_files)} 张图像")
        print(f"验证集: {len(val_files)} 张图像")
        print(f"测试集: {len(test_files)} 张图像")
    
    # 处理每个数据集
    for split, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        print(f"处理{split}集...")
        for image_path in tqdm(files):
            # 获取图像文件名
            image_filename = os.path.basename(image_path)
            base_filename = os.path.splitext(image_filename)[0]
            
            # 复制图像文件
            dst_image_path = os.path.join(images_dir, split, image_filename)
            shutil.copy(image_path, dst_image_path)
            
            # 读取图像尺寸
            try:
                img = cv2.imread(image_path)
                if img is None:
                    print(f"无法读取图像: {image_path}")
                    continue
                height, width = img.shape[:2]
            except Exception as e:
                print(f"处理图像 {image_path} 时出错: {e}")
                continue
            
            # 转换为YOLO格式
            yolo_annotation = convert_to_yolo_format(image_path, width, height)
            if yolo_annotation is None:
                print(f"无法解析文件名: {image_path}")
                continue
            
            # 保存标签文件
            label_path = os.path.join(labels_dir, split, f"{base_filename}.txt")
            with open(label_path, "w") as f:
                f.write(yolo_annotation)
    
    # 创建数据集配置文件
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(dataset_dir)}\n")
        f.write(f"train: {os.path.join('images', 'train')}\n")
        f.write(f"val: {os.path.join('images', 'val')}\n")
        f.write(f"test: {os.path.join('images', 'test')}\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['license_plate']\n")
    
    print(f"数据集配置文件已保存到 {yaml_path}")
    
    return yaml_path


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="准备CCPD数据集用于YOLOv8训练")
    parser.add_argument("--data_dir", type=str, required=True, help="CCPD数据集目录")
    parser.add_argument("--output_dir", type=str, default="./ccpd_yolo_dataset", help="输出目录")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例")
    args = parser.parse_args()
    
    # 检查数据目录是否存在
    if not os.path.isdir(args.data_dir):
        print(f"错误: 数据目录 {args.data_dir} 不存在")
        return
    
    # 准备数据集
    print("准备数据集...")
    dataset_yaml = prepare_dataset(
        args.data_dir, 
        args.output_dir, 
        args.train_ratio, 
        args.val_ratio, 
        args.test_ratio
    )
    
    print("\n数据集准备完成!")
    print(f"数据集配置文件: {dataset_yaml}")
    print("现在可以使用以下命令训练YOLOv8模型:")
    print(f"yolo train model=yolov8n.pt data={dataset_yaml} epochs=100 imgsz=640 batch=16")
    print("或者使用train_ccpd_yolov8.py脚本进行训练")


if __name__ == "__main__":
    main()