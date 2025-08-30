#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CCPD数据集YOLOv8车牌检测完整工作流

使用方法:
    python ccpd_yolov8_workflow.py --data_dir /path/to/ccpd_dataset --output_dir /path/to/output

该脚本将执行完整的工作流程:
1. 准备数据集: 解析CCPD数据集并转换为YOLOv8格式
2. 训练模型: 使用YOLOv8训练车牌检测模型
3. 测试模型: 在测试集上验证模型性能
"""

import os
import sys
import argparse
import subprocess
import time


def run_command(command):
    """
    运行命令并实时输出结果
    
    Args:
        command: 要运行的命令列表
    """
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode


def prepare_dataset(data_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    准备CCPD数据集
    
    Args:
        data_dir: CCPD数据集目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        
    Returns:
        dataset_yaml: 数据集配置文件路径
    """
    print("\n" + "=" * 50)
    print("第1步: 准备数据集")
    print("=" * 50)
    
    # 构建命令
    command = [
        sys.executable,
        "prepare_ccpd_dataset.py",
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--train_ratio", str(train_ratio),
        "--val_ratio", str(val_ratio),
        "--test_ratio", str(test_ratio)
    ]
    
    # 运行命令
    returncode = run_command(command)
    if returncode != 0:
        print("数据集准备失败!")
        sys.exit(1)
    
    # 返回数据集配置文件路径
    dataset_yaml = os.path.join(output_dir, "dataset.yaml")
    return dataset_yaml


def train_model(dataset_yaml, output_dir, epochs=100, batch_size=16, img_size=640, pretrained="yolov8n.pt"):
    """
    训练YOLOv8模型
    
    Args:
        dataset_yaml: 数据集配置文件路径
        output_dir: 输出目录
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 图像大小
        pretrained: 预训练权重路径
        
    Returns:
        best_model_path: 最佳模型路径
    """
    print("\n" + "=" * 50)
    print("第2步: 训练模型")
    print("=" * 50)
    
    # 构建命令
    command = [
        sys.executable,
        "train_yolov8_ccpd.py",
        "--data_yaml", dataset_yaml,
        "--output_dir", output_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--img_size", str(img_size),
        "--pretrained", pretrained
    ]
    
    # 运行命令
    returncode = run_command(command)
    if returncode != 0:
        print("模型训练失败!")
        sys.exit(1)
    
    # 返回最佳模型路径
    best_model_path = os.path.join(output_dir, "best.pt")
    return best_model_path


def test_model(model_path, test_dir, save_dir):
    """
    测试YOLOv8模型
    
    Args:
        model_path: 模型路径
        test_dir: 测试图像目录
        save_dir: 保存结果的目录
    """
    print("\n" + "=" * 50)
    print("第3步: 测试模型")
    print("=" * 50)
    
    # 构建命令
    command = [
        sys.executable,
        "test_ccpd_model.py",
        "--model", model_path,
        "--dir", test_dir,
        "--save_dir", save_dir
    ]
    
    # 运行命令
    returncode = run_command(command)
    if returncode != 0:
        print("模型测试失败!")
        sys.exit(1)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="CCPD数据集YOLOv8车牌检测完整工作流")
    parser.add_argument("--data_dir", type=str, required=True, help="CCPD数据集目录")
    parser.add_argument("--output_dir", type=str, default="./ccpd_yolov8_workflow", help="输出目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--img_size", type=int, default=640, help="图像大小")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--pretrained", type=str, default="yolov8n.pt", help="预训练权重路径")
    args = parser.parse_args()
    
    # 检查数据目录是否存在
    if not os.path.isdir(args.data_dir):
        print(f"错误: 数据目录 {args.data_dir} 不存在")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    # 步骤1: 准备数据集
    dataset_yaml = prepare_dataset(
        args.data_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )
    
    # 步骤2: 训练模型
    best_model_path = train_model(
        dataset_yaml,
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.img_size,
        args.pretrained
    )
    
    # 步骤3: 测试模型
    test_dir = os.path.join(args.output_dir, "dataset", "images", "test")
    save_dir = os.path.join(args.output_dir, "results")
    if os.path.exists(test_dir) and os.listdir(test_dir):
        test_model(best_model_path, test_dir, save_dir)
    else:
        print("\n警告: 测试集为空，跳过测试步骤")
    
    # 计算总耗时
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "=" * 50)
    print("工作流完成!")
    print("=" * 50)
    print(f"总耗时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
    print(f"最佳模型保存在: {best_model_path}")
    print(f"测试结果保存在: {save_dir}")
    print("\n使用以下命令测试单张图像:")
    print(f"python test_ccpd_model.py --model {best_model_path} --image /path/to/image.jpg")


if __name__ == "__main__":
    main()