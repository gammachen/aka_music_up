#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用CCPD数据集训练YOLOv8车牌检测模型

使用方法:
    python train_yolov8_ccpd.py --data_yaml /path/to/dataset.yaml --epochs 100 --batch_size 16 --img_size 640

该脚本将:
1. 加载预处理好的CCPD数据集（通过prepare_ccpd_dataset.py生成）
2. 使用YOLOv8训练车牌检测模型
3. 验证模型性能并保存最佳模型
"""

import os
import sys
import shutil
import argparse
import subprocess
import time

# 尝试解决numpy和pandas兼容性问题
def fix_numpy_pandas_compatibility():
    """
    尝试解决numpy和pandas的兼容性问题
    """
    try:
        print("尝试修复numpy和pandas的兼容性问题...")
        # 尝试更新numpy
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "numpy"], check=True)
        print("已更新numpy")
        
        # 尝试更新pandas
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pandas"], check=True)
        print("已更新pandas")
        
        # 尝试安装ultralytics
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "ultralytics"], check=True)
        print("已安装/更新ultralytics")
        
        return True
    except Exception as e:
        print(f"修复依赖关系时出错: {e}")
        return False

# 尝试导入必要的库
try:
    import torch
    # 添加PyTorch 2.6兼容性配置，解决weights_only参数默认值变更问题
    from torch.serialization import add_safe_globals
    from ultralytics.nn.tasks import DetectionModel
    # 将DetectionModel添加到安全全局变量列表中
    add_safe_globals([DetectionModel])
    from ultralytics import YOLO
    print("成功导入所需库")
except ImportError as e:
    print(f"导入库失败: {e}")
    print("尝试修复依赖关系...")
    if fix_numpy_pandas_compatibility():
        try:
            import torch
            # 添加PyTorch 2.6兼容性配置，解决weights_only参数默认值变更问题
            from torch.serialization import add_safe_globals
            from ultralytics.nn.tasks import DetectionModel
            # 将DetectionModel添加到安全全局变量列表中
            add_safe_globals([DetectionModel])
            from ultralytics import YOLO
            print("成功导入所需库")
        except ImportError as e:
            print(f"修复后仍然无法导入所需库: {e}")
            print("请手动安装依赖: pip install numpy pandas torch ultralytics")
            sys.exit(1)
    else:
        print("无法修复依赖关系")
        print("请手动安装依赖: pip install numpy pandas torch ultralytics")
        sys.exit(1)


def train_yolov8(dataset_yaml, output_dir, epochs=100, batch_size=16, img_size=640, pretrained_weights="yolov8n.pt"):
    """
    训练YOLOv8模型
    
    Args:
        dataset_yaml: 数据集配置文件路径
        output_dir: 输出目录
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 图像大小
        pretrained_weights: 预训练权重路径
    """
    # 创建模型输出目录
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # 加载预训练模型
    # 使用weights_only=False参数解决PyTorch 2.6兼容性问题
    try:
        model = YOLO(pretrained_weights)
    except Exception as e:
        print(f"使用默认参数加载模型失败: {e}")
        print("尝试使用weights_only=False参数加载模型...")
        # 修改torch.load的默认行为，设置weights_only=False
        import torch.serialization
        original_torch_load = torch.load
        
        def patched_torch_load(f, *args, **kwargs):
            # 确保weights_only参数为False
            kwargs['weights_only'] = False
            return original_torch_load(f, *args, **kwargs)
        
        # 替换torch.load函数
        torch.load = patched_torch_load
        
        # 重新尝试加载模型
        model = YOLO(pretrained_weights)
        
        # 恢复原始torch.load函数
        torch.load = original_torch_load
    
    # 训练模型
    print(f"开始训练YOLOv8模型，使用数据集: {dataset_yaml}")
    model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=model_dir,
        name="train",
        exist_ok=True,
        patience=20,  # 早停耐心值
        save=True,  # 保存最佳模型
        device="0" if torch.cuda.is_available() else "cpu",  # 使用GPU或CPU
    )
    
    # 复制最佳模型到输出目录
    best_model_path = os.path.join(model_dir, "train", "weights", "best.pt")
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, os.path.join(output_dir, "best.pt"))
        print(f"最佳模型已保存到 {os.path.join(output_dir, 'best.pt')}")
    
    # 验证模型
    print("\n验证模型性能:")
    model.val(data=dataset_yaml)
    
    return os.path.join(output_dir, "best.pt")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="使用CCPD数据集训练YOLOv8车牌检测模型")
    parser.add_argument("--data_yaml", type=str, required=True, help="数据集配置文件路径")
    parser.add_argument("--output_dir", type=str, default="./ccpd_yolov8_output", help="输出目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--img_size", type=int, default=640, help="图像大小")
    parser.add_argument("--pretrained", type=str, default="yolov8n.pt", help="预训练权重路径")
    args = parser.parse_args()
    
    # 检查数据集配置文件是否存在
    if not os.path.exists(args.data_yaml):
        print(f"错误: 数据集配置文件 {args.data_yaml} 不存在")
        print("请先运行 prepare_ccpd_dataset.py 准备数据集")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练模型
    print("\n开始训练模型...")
    best_model_path = train_yolov8(
        args.data_yaml, 
        args.output_dir, 
        args.epochs, 
        args.batch_size, 
        args.img_size, 
        args.pretrained
    )
    
    print("\n训练完成!")
    print(f"最佳模型保存在: {best_model_path}")
    print("可以使用 test_ccpd_model.py 脚本测试模型性能:")
    print(f"python test_ccpd_model.py --model {best_model_path} --image /path/to/test/image.jpg")


if __name__ == "__main__":
    # 检查CUDA可用性
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    
    main()