#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CCPD数据集YOLOv8车牌检测完整工作流（修复版）

使用方法:
    python ccpd_yolov8_workflow_fixed.py --data_dir /path/to/ccpd_dataset --output_dir /path/to/output

该脚本将执行完整的工作流程:
1. 准备数据集: 解析CCPD数据集并转换为YOLOv8格式
2. 训练模型: 使用YOLOv8训练车牌检测模型
3. 测试模型: 在测试集上验证模型性能

该脚本会自动处理依赖问题，特别是numpy和pandas的兼容性问题
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path


def fix_dependencies():
    """
    修复依赖关系，特别是numpy和pandas的兼容性问题以及PyTorch 2.6兼容性问题
    """
    print("检查并修复依赖关系...")
    try:
        # 安装/更新依赖
        requirements = [
            "numpy>=1.22.0,<1.25.0",  # 限制numpy版本，避免兼容性问题
            "pandas>=1.4.0,<2.0.0",   # 限制pandas版本，避免兼容性问题
            "ultralytics>=8.0.0",
            "opencv-python>=4.5.0",
            "tqdm>=4.60.0",
            "torch>=1.8.0"           # 确保PyTorch版本兼容
        ]
        
        for req in requirements:
            print(f"安装/更新 {req}...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", req],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        
        # 配置PyTorch 2.6兼容性
        try:
            import torch.serialization
            from ultralytics.nn.tasks import DetectionModel
            print("配置PyTorch 2.6兼容性...")
            # 将DetectionModel添加到安全全局变量列表中
            torch.serialization.add_safe_globals([DetectionModel])
            print("PyTorch 2.6兼容性配置完成")
        except (ImportError, AttributeError) as e:
            print(f"PyTorch 2.6兼容性配置跳过: {e}")
        
        print("依赖关系修复完成")
        return True
    except Exception as e:
        print(f"修复依赖关系时出错: {e}")
        return False


def fix_pytorch_compatibility():
    """
    修复PyTorch 2.6兼容性问题，特别是weights_only参数默认值变更问题
    """
    try:
        import torch
        import torch.serialization
        from ultralytics.nn.tasks import DetectionModel
        
        # 将DetectionModel添加到安全全局变量列表中
        print("配置PyTorch 2.6兼容性...")
        torch.serialization.add_safe_globals([DetectionModel])
        
        # 修改torch.load的默认行为
        original_torch_load = torch.load
        
        def patched_torch_load(f, *args, **kwargs):
            # 确保weights_only参数为False
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_torch_load(f, *args, **kwargs)
        
        # 替换torch.load函数
        torch.load = patched_torch_load
        print("PyTorch 2.6兼容性配置完成")
        return True
    except (ImportError, AttributeError) as e:
        print(f"PyTorch 2.6兼容性配置失败: {e}")
        return False

def run_command(command, description=None):
    """
    运行命令并实时输出结果
    
    Args:
        command: 要运行的命令列表
        description: 命令描述
    
    Returns:
        returncode: 命令返回码
    """
    if description:
        print(f"\n执行: {description}")
    
    print(f"运行命令: {' '.join(command)}")
    
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
    returncode = run_command(command, "准备数据集")
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
    
    # 应用PyTorch兼容性修复
    fix_pytorch_compatibility()
    
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
    returncode = run_command(command, "训练模型")
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
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        print("跳过测试步骤")
        return
    
    # 构建命令
    command = [
        sys.executable,
        "test_ccpd_model.py",
        "--model", model_path,
        "--dir", test_dir,
        "--save_dir", save_dir
    ]
    
    # 运行命令
    returncode = run_command(command, "测试模型")
    if returncode != 0:
        print("模型测试失败!")
        # 不退出，因为这是最后一步


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="CCPD数据集YOLOv8车牌检测完整工作流（修复版）")
    parser.add_argument("--data_dir", type=str, required=True, help="CCPD数据集目录")
    parser.add_argument("--output_dir", type=str, default="./ccpd_yolov8_workflow", help="输出目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--img_size", type=int, default=640, help="图像大小")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--pretrained", type=str, default="yolov8n.pt", help="预训练权重路径")
    parser.add_argument("--skip_dependency_fix", action="store_true", help="跳过依赖修复")
    args = parser.parse_args()
    
    # 检查数据目录是否存在
    if not os.path.isdir(args.data_dir):
        print(f"错误: 数据目录 {args.data_dir} 不存在")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 修复依赖关系
    if not args.skip_dependency_fix:
        if not fix_dependencies():
            print("警告: 依赖关系修复失败，尝试继续执行工作流...")
    
    # 应用PyTorch兼容性修复
    fix_pytorch_compatibility()
    
    # 记录开始时间
    start_time = time.time()
    
    try:
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
        test_model(best_model_path, test_dir, save_dir)
        
        # 计算总耗时
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n" + "=" * 50)
        print(f"工作流完成! 总耗时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
        print("=" * 50)
        print(f"数据集: {dataset_yaml}")
        print(f"最佳模型: {best_model_path}")
        print(f"测试结果: {save_dir}")
        
    except KeyboardInterrupt:
        print("\n工作流被用户中断!")
    except Exception as e:
        print(f"\n工作流执行出错: {e}")


if __name__ == "__main__":
    main()