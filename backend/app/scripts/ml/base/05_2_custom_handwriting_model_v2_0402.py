#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
自定义手写识别模型训练脚本
本脚本实现了一个基于PyTorch的深度学习模型，用于手写字符识别
包含完整的训练流程：
1. 数据加载与预处理
2. 模型构建
3. 损失函数与优化器定义
4. 训练过程
5. 模型评估与可视化
'''

# 导入必要的库
import torch
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器
import torch.nn.functional as F  # 激活函数等功能性组件
from torch.utils.data import DataLoader, Dataset, ConcatDataset  # 数据加载器和数据集类

import torchvision  # 计算机视觉库
import torchvision.transforms as transforms  # 数据变换
from torchvision.datasets import MNIST  # MNIST数据集

import matplotlib.pyplot as plt  # 可视化
import matplotlib
from matplotlib.font_manager import FontProperties
import numpy as np
import time
import os
import cv2
from PIL import Image

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'Songti SC', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建中文字体属性对象，用于单独设置某些元素的字体
chineseFont = FontProperties(family='Arial Unicode MS')

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")

# 定义标签字典，确保与目录结构一致
label_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 
                 10: '=', 11: '+', 12: '-', 13: '×', 14: '÷'}
# ====================== 第一步：自定义数据集类 ======================

class HandwritingDataset(Dataset):
    '''
    自定义数据集类，用于加载手写字符图像
    '''
    def __init__(self, root_dir, transform=None, train=True, split_ratio=0.8):
        '''
        初始化数据集
        参数:
            root_dir: 数据集根目录，包含分类子文件夹
            transform: 数据变换
            train: 是否为训练集
            split_ratio: 训练集比例
        '''
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.split_ratio = split_ratio
        
        # 获取所有类别（文件夹名称）
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.classes.sort()  # 确保类别顺序一致
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 收集所有图像路径和标签
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.png') or img_name.endswith('.jpg') or img_name.endswith('.jpeg'):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
        
        # 打乱数据
        np.random.shuffle(self.samples)
        
        # 划分训练集和测试集
        split_idx = int(len(self.samples) * split_ratio)
        if train:
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
    
    def __len__(self):
        '''
        返回数据集大小
        '''
        return len(self.samples)
    
    def __getitem__(self, idx):
        '''
        获取数据集中的一个样本
        '''
        img_path, label = self.samples[idx]
        
        # 读取图像
        image = Image.open(img_path).convert('L')  # 转换为灰度图
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 调试样本加载
        if idx % 500 == 0:
            print(f"加载样本 {idx+1}/{len(self.samples)} | 路径: {img_path} | 标签: {label}")
        return image, label

# ====================== 第二步：数据加载与预处理 ======================

# 创建MNIST数据集包装类，使其与自定义数据集兼容
class MNISTWrapper(Dataset):
    '''
    MNIST数据集包装类，使其与自定义数据集的标签兼容
    '''
    def __init__(self, mnist_dataset, custom_classes):
        '''
        初始化MNIST包装类
        参数:
            mnist_dataset: 原始MNIST数据集
            custom_classes: 自定义数据集的类别列表，用于确定哪些标签需要特殊处理
        '''
        self.mnist_dataset = mnist_dataset
        self.custom_classes = custom_classes
        
        # 创建标签映射字典
        self.label_map = {}
        for i in range(10):  # MNIST只有0-9这10个标签
            # 数字标签0-9必须保持一致，确保MNIST的数字标签映射到自定义数据集中相同的数字标签
            if str(i) in self.custom_classes:
                # 如果自定义数据集中有这个数字标签，则直接映射到对应位置
                self.label_map[i] = self.custom_classes.index(str(i))
            else:
                # 如果自定义数据集中没有这个数字标签，则放在自定义数据集后面
                # 这种情况在实际应用中应该不会发生，因为我们的自定义数据集应该包含所有数字0-9
                self.label_map[i] = len(self.custom_classes) + i
                print(f"警告：自定义数据集中没有数字标签 {i}，这可能会导致标签映射问题")
    
    def __len__(self):
        return len(self.mnist_dataset)
    
    def __getitem__(self, idx):
        image, label = self.mnist_dataset[idx]
        # 返回图像和映射后的标签
        # 调试样本加载
        if idx % 500 == 0:
            print(f"加载MNIST样本 {idx+1}/{len(self.mnist_dataset)} | 原始标签: {label}")
        
        # 使用映射字典获取调整后的标签
        adjusted_label = self.label_map[label]
        
        if idx % 1000 == 0:
            print(f"MNIST样本 {idx} | 原始标签: {label} -> 调整后标签: {adjusted_label} (保持数字0-9一致)")
        return image, adjusted_label

def load_data(data_dir='./dataset', batch_size=64, use_mnist=True, mnist_label_offset=None):
    '''
    加载数据集并进行预处理，可选择是否包含MNIST数据集
    参数:
        data_dir: 自定义数据集目录
        batch_size: 批量大小
        use_mnist: 是否使用MNIST数据集
        mnist_label_offset: MNIST数据集标签偏移量（已废弃，保留参数是为了兼容性）
    返回:
        train_loader, test_loader: 训练和测试数据加载器
    '''
    
    '''
    加载数据集并进行预处理，可选择是否包含MNIST数据集
    参数:
        data_dir: 自定义数据集目录
        batch_size: 批量大小
        use_mnist: 是否使用MNIST数据集
        mnist_label_offset: MNIST数据集标签偏移量
    返回:
        train_loader, test_loader: 训练和测试数据加载器
    '''
    # 定义自定义数据集的变换：调整大小、转换为张量、标准化
    custom_transform = transforms.Compose([
        transforms.Resize((24, 24)),  # 调整为24x24大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.5,), (0.5,))  # 标准化到[-1,1]范围
    ])
    
    # 定义MNIST数据集的变换：调整大小、标准化
    mnist_transform = transforms.Compose([
        transforms.Resize((24, 24)),  # 调整为与自定义数据集相同的大小
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
    ])
    
    # 创建自定义训练集和测试集
    custom_train_dataset = HandwritingDataset(
        root_dir=data_dir,
        transform=custom_transform,
        train=True
    )
    
    custom_test_dataset = HandwritingDataset(
        root_dir=data_dir,
        transform=custom_transform,
        train=False
    )
    
    num_classes = len(custom_train_dataset.classes)
    
    # 如果使用MNIST数据集，则加载并合并
    if use_mnist:
        # 加载MNIST训练集和测试集
        mnist_train = MNIST(
            root='./data',
            train=True,
            download=True,
            transform=mnist_transform
        )
        
        mnist_test = MNIST(
            root='./data',
            train=False,
            download=True,
            transform=mnist_transform
        )
        
        # 打印自定义数据集的类别
        print("自定义数据集类别:", custom_train_dataset.classes)
        
        # 包装MNIST数据集，传递自定义类别列表而不是偏移量
        mnist_train_wrapped = MNISTWrapper(mnist_train, custom_train_dataset.classes)
        mnist_test_wrapped = MNISTWrapper(mnist_test, custom_test_dataset.classes)
        
        # 合并数据集
        train_dataset = ConcatDataset([custom_train_dataset, mnist_train_wrapped])
        test_dataset = ConcatDataset([custom_test_dataset, mnist_test_wrapped])
        
        # 计算实际的类别数量（考虑到数字0-9可能在两个数据集中重叠）
        # 自定义数据集的类别数 + MNIST中不在自定义数据集中的类别数
        mnist_unique_classes = sum(1 for i in range(10) if str(i) not in custom_train_dataset.classes)
        total_classes = num_classes + mnist_unique_classes
        
        print(f"合并后的训练集大小: {len(train_dataset)} (自定义: {len(custom_train_dataset)}, MNIST: {len(mnist_train)})")
        print(f"总类别数: {total_classes} (自定义类别: {num_classes}, MNIST独有类别: {mnist_unique_classes})")
        print("="*50 + " 数据集详情 " + "="*50)
        print(f"自定义训练集: {len(custom_train_dataset)} 个样本")
        print(f"自定义测试集: {len(custom_test_dataset)} 个样本")
        print(f"MNIST训练集: {len(mnist_train)} 个样本")
        print(f"MNIST测试集: {len(mnist_test)} 个样本")
    else:
        # 只使用自定义数据集
        train_dataset = custom_train_dataset
        test_dataset = custom_test_dataset
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # 验证第一个测试批次
    test_batch = next(iter(test_loader))
    t_images, t_labels = test_batch
    print(f'\n首测试批次验证 -> 图像尺寸: {t_images.shape} 标签形状: {t_labels.shape}')
    print(f'测试标签范围: {t_labels.min()} ~ {t_labels.max()} (应覆盖0-{total_classes-1 if use_mnist else num_classes-1})')
    print(f"合并后的测试集大小: {len(test_dataset)} (自定义: {len(custom_test_dataset)}, MNIST: {len(mnist_test) if use_mnist else 0})")
    
    # ====================== 数据验证：抽取样本并显示 ======================
    print("\n" + "="*20 + " 数据验证：抽取样本并显示 " + "="*20)
    
    # 创建一个函数来显示图像和标签
    def show_samples(dataset, title, num_samples=5, class_names=None, is_mnist=False, label_map=None):
        # 创建数据加载器，每次只加载一个样本
        sample_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        # 获取指定数量的样本
        samples = []
        for _ in range(min(num_samples, len(dataset))):
            try:
                img, label = next(iter(sample_loader))
                samples.append((img, label))
            except StopIteration:
                break
        
        # 显示样本
        plt.figure(figsize=(15, 3))
        for i, (img, label) in enumerate(samples):
            plt.subplot(1, len(samples), i+1)
            
            # 反标准化图像
            img = img.squeeze().numpy()  # 移除批次和通道维度
            img = img * 0.5 + 0.5  # 从[-1,1]转换回[0,1]
            
            plt.imshow(img, cmap='gray')
            
            # 获取标签文本
            label_idx = label.item()
            if is_mnist and label_map is not None:
                # 对于MNIST，显示原始标签和映射后的标签
                # 反向查找原始标签
                for orig_label, mapped_label in label_map.items():
                    if mapped_label == label_idx:
                        original_label = orig_label
                        break
                else:
                    original_label = '?'
                label_text = f"{original_label} → {label_idx}"
            elif class_names is not None and label_idx < len(class_names):
                label_text = f"{label_idx} ({class_names[label_idx]})"
            else:
                label_text = str(label_idx)
                
            plt.title(f"标签: {label_text}")
            plt.axis('off')
        
        plt.suptitle(title, fontproperties=chineseFont, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    # 显示自定义数据集样本
    if len(custom_train_dataset) > 0:
        print("\n自定义数据集样本:")
        show_samples(custom_train_dataset, "自定义数据集样本", 
                    num_samples=5, class_names=custom_train_dataset.classes)
    
    # 如果使用MNIST数据集，显示MNIST样本
    if use_mnist:
        print("\nMNIST数据集样本 (带标签映射):")
        # 对于MNIST，我们需要传递标签映射字典
        show_samples(mnist_train_wrapped, "MNIST数据集样本 (带标签映射)", 
                    num_samples=5, is_mnist=True, label_map=mnist_train_wrapped.label_map)
    
    # 显示合并后的数据集样本（如果有合并）
    if use_mnist:
        print("\n合并数据集样本:")
        # 创建合并后的类别名称列表
        combined_class_names = list(custom_train_dataset.classes)
        
        # 添加MNIST中不在自定义数据集中的类别
        for i in range(10):
            if str(i) not in custom_train_dataset.classes:
                # 找到这个数字在合并后的位置
                for orig_label, mapped_label in mnist_train_wrapped.label_map.items():
                    if orig_label == i:
                        # 确保列表长度足够
                        while len(combined_class_names) <= mapped_label:
                            combined_class_names.append("")
                        combined_class_names[mapped_label] = f"{i}"
                        break
        
        show_samples(train_dataset, "合并数据集样本", num_samples=10, class_names=combined_class_names)
    
    return train_loader, test_loader, num_classes

# ====================== 第三步：构建神经网络模型 ======================

class EnhancedCNN(nn.Module):
    '''
    增强版卷积神经网络模型，适用于MNIST和自定义数据集的组合
    '''
    def __init__(self, num_classes=15):
        '''
        初始化模型
        参数:
            num_classes: 类别数量
        '''
        super(EnhancedCNN, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 批归一化，提高训练稳定性
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三个卷积块（增加网络深度）
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 计算全连接层的输入维度
        # 经过三次池化后，图像尺寸从24x24变为3x3
        fc_input_dim = 128 * 3 * 3
        
        # 全连接层
        self.fc1 = nn.Linear(fc_input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout层，防止过拟合
        self.dropout1 = nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout(p=0.4)
    
    def forward(self, x):
        '''
        前向传播过程
        参数:
            x: 输入图像张量，形状为[batch_size, 1, height, width]
        返回:
            输出预测张量，形状为[batch_size, num_classes]
        '''
        # 图像已在数据加载时进行了缩放和标准化
        
        # 第一个卷积块：卷积 -> ReLU激活 -> 池化
        x = self.pool1(F.relu(self.conv1(x)))  # 输出尺寸: [batch_size, 24, 12, 12]
        
        # 第二个卷积块：卷积 -> ReLU激活 -> 池化
        x = self.pool2(F.relu(self.conv2(x)))  # 输出尺寸: [batch_size, 64, 6, 6]
        
        # 第三个卷积块：卷积 -> ReLU激活 -> 池化
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)  # 输出尺寸: [batch_size, 128, 3, 3]
        
        # 展平操作
        x = x.view(x.size(0), -1)  # 输出尺寸: [batch_size, 128*3*3=1152]
        
        # 全连接层1：线性变换 -> ReLU激活 -> Dropout
        x = self.dropout1(F.relu(self.fc1(x)))  # 输出尺寸: [batch_size, 128]
        
        # 全连接层2：线性变换（输出层）
        x = self.fc2(x)  # 输出尺寸: [batch_size, num_classes]
        
        return x

# ====================== 第四步：训练模型 ======================

def train(model, train_loader, test_loader, epochs=10, resume_from=None, checkpoint_dir='./checkpoints'):
    '''
    训练模型函数
    参数:
        model: 待训练的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        epochs: 训练轮数
        resume_from: 恢复训练的检查点路径，如果为None则从头开始训练
        checkpoint_dir: 检查点保存目录
    返回:
        训练历史记录（损失和准确率）
    '''
    # 将模型移动到指定设备（GPU或CPU）
    model = model.to(device)
    
    # 定义损失函数：交叉熵损失（带logits）
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器：Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 学习率调度器：当验证集性能停止提升时，降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    # 创建检查点目录（如果不存在）
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 初始化训练状态
    start_epoch = 0
    best_acc = 0.0
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # 如果提供了恢复点，则加载之前的训练状态
    if resume_from and os.path.isfile(resume_from):
        print(f"加载检查点: {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        history = checkpoint['history']
        print(f"恢复训练从 epoch {start_epoch}，最佳准确率: {best_acc:.2f}%")
    
    # 训练开始时间
    start_time = time.time()
    
    # 训练循环
    for epoch in range(start_epoch, epochs):
        # 设置为训练模式
        model.train()
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 遍历训练数据批次
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 将数据移动到指定设备
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播：计算梯度
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 累计损失
            train_loss += loss.item()
            
            # 计算准确率
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # 每10个批次打印一次进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f'Epoch: {epoch+1}/{epochs} | Batch: {batch_idx+1}/{len(train_loader)} | '
                      f'Loss: {train_loss/(batch_idx+1):.4f} | '
                      f'Acc: {100.*train_correct/train_total:.2f}%')
        
        # 计算平均训练损失和准确率
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # 评估模型
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        
        # 更新学习率
        scheduler.step(test_acc)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # 打印epoch结果
        print(f'Epoch: {epoch+1}/{epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
        
        # 保存检查点
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_acc': best_acc
        }
        torch.save(checkpoint, checkpoint_path)
        print(f'检查点已保存: {checkpoint_path}')
        
        # 如果当前模型性能超过之前最佳，保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'history': history
            }, best_model_path)
            print(f'新的最佳模型已保存! 准确率: {test_acc:.2f}%')
    
    # 训练结束，打印总用时
    total_time = time.time() - start_time
    print(f'训练完成！总用时: {total_time:.2f}秒')
    
    return history

# ====================== 第五步：评估模型 ======================

def evaluate(model, data_loader, criterion):
    '''
    评估模型函数
    参数:
        model: 待评估的模型
        data_loader: 数据加载器
        criterion: 损失函数
    返回:
        平均损失和准确率
    '''
    # 设置为评估模式
    model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    # 禁用梯度计算
    with torch.no_grad():
        for inputs, targets in data_loader:
            # 将数据移动到指定设备
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            # 计算准确率
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # 计算平均损失和准确率
    test_loss = test_loss / len(data_loader)
    accuracy = 100. * correct / total
    
    return test_loss, accuracy

# ====================== 第六步：可视化结果 ======================

def visualize_results(history, model, test_loader, num_classes):
    '''
    可视化训练结果和模型预测
    参数:
        history: 训练历史记录
        model: 训练好的模型
        test_loader: 测试数据加载器
        num_classes: 类别数量
    '''
    # 1. 绘制训练曲线
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['test_loss'], label='测试损失')
    plt.xlabel('Epoch', fontproperties=chineseFont)
    plt.ylabel('损失', fontproperties=chineseFont)
    plt.legend(prop=chineseFont)
    plt.title('训练和测试损失', fontproperties=chineseFont)
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['test_acc'], label='测试准确率')
    plt.xlabel('Epoch', fontproperties=chineseFont)
    plt.ylabel('准确率 (%)', fontproperties=chineseFont)
    plt.legend(prop=chineseFont)
    plt.title('训练和测试准确率', fontproperties=chineseFont)
    
    plt.tight_layout()
    plt.savefig('custom_training_curves.png')
    plt.show()
    
    # 2. 可视化模型预测
    model.eval()
    
    # 获取一批测试数据
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # 选择前25张图片（如果有的话）
    num_images = min(25, len(images))
    images = images[:num_images]
    labels = labels[:num_images]
    
    # 获取模型预测
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()
    
    # 显示图片和预测结果
    images = images.numpy()
    
    # 创建网格显示图片
    grid_size = int(np.ceil(np.sqrt(num_images)))
    plt.figure(figsize=(12, 12))
    for i in range(num_images):
        plt.subplot(grid_size, grid_size, i+1)
        # 将图像从[1,24,24]转换为[24,24]并反转标准化
        plt.imshow(images[i][0] * 0.5 + 0.5, cmap='gray')
        
        # 使用label_dict转换标签
        predicted_label = label_dict.get(predicted[i], str(predicted[i]))
        true_label = label_dict.get(labels[i], str(labels[i]))
        
        # 设置标题：绿色表示预测正确，红色表示预测错误
        if predicted[i] == labels[i]:
            plt.title(f'预测: {predicted[i]} -> {predicted_label}', color='green')
        else:
            plt.title(f'预测: {predicted_label}\n实际: {true_label}', color='red')
        plt.axis('off')  # 隐藏坐标轴
    
    plt.tight_layout()
    plt.savefig('custom_predictions.png')
    plt.show()

# ====================== 第七步：保存和加载模型 ======================

def save_model(model, filepath='custom_cnn_model.pth'):
    '''
    保存模型
    参数:
        model: 要保存的模型
        filepath: 保存路径
    '''
    torch.save(model.state_dict(), filepath)
    print(f'模型已保存至: {filepath}')

def load_model(model, filepath='custom_cnn_model.pth'):
    '''
    加载模型
    参数:
        model: 模型实例
        filepath: 模型文件路径
    返回:
        加载了权重的模型
    '''
    model.load_state_dict(torch.load(filepath))
    model.eval()  # 设置为评估模式
    print(f'模型已从 {filepath} 加载')
    return model

# ====================== 第八步：预测功能 ======================

def preprocess_image(image_path):
    '''
    预处理单张图像用于预测
    参数:
        image_path: 图像文件路径
    返回:
        预处理后的图像张量
    '''
    # 定义与训练时相同的变换
    transform = transforms.Compose([
        transforms.Resize((24, 24)),  # 调整为24x24大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.5,), (0.5,))  # 标准化到[-1,1]范围
    ])
    
    # 读取图像并转换为灰度图
    image = Image.open(image_path).convert('L')
    
    # 应用变换
    image_tensor = transform(image)
    
    # 添加批次维度
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def predict(model, image_paths, class_names=None):
    '''
    使用训练好的模型预测图像
    参数:
        model: 训练好的模型
        image_paths: 图像文件路径列表
        class_names: 类别名称列表，如果为None则使用索引作为类别
    返回:
        预测结果列表
    '''
    # 确保模型处于评估模式
    model.eval()
    model = model.to(device)
    
    results = []
    original_images = []
    
    # 处理每张图像
    for img_path in image_paths:
        # 预处理图像
        img_tensor = preprocess_image(img_path)
        img_tensor = img_tensor.to(device)
        
        # 保存原始图像用于可视化
        original_img = cv2.imread(img_path)
        if original_img is None:  # 如果OpenCV无法读取，尝试使用PIL
            original_img = np.array(Image.open(img_path))
            if len(original_img.shape) == 2:  # 如果是灰度图，转换为BGR
                original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        original_images.append(original_img)
        
        # 进行预测
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            # 获取预测类别
            predicted_idx = predicted_class.item()
            confidence_value = confidence.item()
            
            # 如果提供了类别名称，则使用类别名称
            if class_names is not None and predicted_idx < len(class_names):
                predicted_label = class_names[predicted_idx]
            else:
                predicted_label = str(predicted_idx)
            
            results.append({
                'path': img_path,
                'predicted_class': predicted_idx,
                'predicted_label': predicted_label,
                'confidence': confidence_value
            })
    
    # 可视化预测结果
    visualize_predictions(original_images, results)
    
    return results

def visualize_predictions(images, results):
    '''
    可视化预测结果
    参数:
        images: 原始图像列表
        results: 预测结果列表
    '''
    n = len(images)
    if n == 0:
        return
    
    # 确定图像网格的行列数
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    plt.figure(figsize=(4*cols, 4*rows))
    
    for i, (img, result) in enumerate(zip(images, results)):
        plt.subplot(rows, cols, i+1)
        
        # 显示图像
        if len(img.shape) == 3 and img.shape[2] == 3:  # 彩色图像
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:  # 灰度图像
            plt.imshow(img, cmap='gray')
        
                # 使用label_dict转换标签
        try:
            # 将字符串标签转换为整数索引
            predicted_idx = int(result['predicted_label'])
            predicted_symbol = label_dict.get(predicted_idx, str(result['predicted_label']))
        except ValueError:
            predicted_symbol = str(result['predicted_label'])
        
        # 设置标题
        title = f"预测: {result['predicted_label']} -> {predicted_symbol}\n置信度: {result['confidence']:.2f}"
        plt.title(title, fontproperties=chineseFont)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.show()

def predict_batch_from_array(model, images_array, class_names=None):
    '''
    使用训练好的模型预测图像数组
    参数:
        model: 训练好的模型
        images_array: numpy数组，形状为[batch_size, height, width]或[batch_size, height, width, channels]
        class_names: 类别名称列表，如果为None则使用索引作为类别
    返回:
        预测结果列表
    '''
    # 确保模型处于评估模式
    model.eval()
    model = model.to(device)
    
    # 预处理图像数组
    processed_images = []
    for img in images_array:
        # 如果是彩色图像，转换为灰度图
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 显示原始图像和调整大小后的图像
        plt.figure(figsize=(10, 4))
        
        # 显示原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('原始图像', fontproperties=chineseFont)
        plt.axis('off')
        
        # 调整大小并显示
        resized_img = cv2.resize(img, (24, 24))
        plt.subplot(1, 2, 2)
        plt.imshow(resized_img, cmap='gray')
        plt.title('调整大小后 (24x24)', fontproperties=chineseFont)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 更新图像变量
        img = resized_img
        
        # 转换为张量并标准化
        img_tensor = torch.from_numpy(img).float() / 255.0  # 归一化到[0,1]
        img_tensor = img_tensor.unsqueeze(0)  # 添加通道维度
        img_tensor = img_tensor * 2 - 1  # 标准化到[-1,1]
        
        processed_images.append(img_tensor)
    
    # 堆叠所有图像为一个批次
    batch_tensor = torch.stack(processed_images).to(device)
    
    # 进行预测
    results = []
    with torch.no_grad():
        outputs = model(batch_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_classes = torch.max(probabilities, 1)
        
        # 处理每个预测结果
        for i, (pred_idx, conf) in enumerate(zip(predicted_classes, confidence)):
            pred_idx = pred_idx.item()
            conf_value = conf.item()
            
            # 如果提供了类别名称，则使用类别名称
            if class_names is not None and pred_idx < len(class_names):
                pred_label = class_names[pred_idx]
            else:
                pred_label = str(pred_idx)
            
            results.append({
                'index': i,
                'predicted_class': pred_idx,
                'predicted_label': pred_label,
                'confidence': conf_value
            })
    
    return results

# ====================== 主函数 ======================

def main():
    '''
    主函数：执行完整的训练和评估流程
    '''
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='自定义手写字符识别模型训练')
    parser.add_argument('--resume', type=str, default=None, help='从指定检查点恢复训练')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='检查点保存目录')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--data_dir', type=str, default='./dataset', help='数据集目录')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'], help='运行模式：训练或预测')
    parser.add_argument('--model_path', type=str, default=None, help='用于预测的模型路径')
    # 支持多种方式传入测试图像路径:
    # 1. 空格分隔: --test_images image1.jpg image2.jpg image3.jpg
    # 2. 逗号分隔: --test_images image1.jpg,image2.jpg,image3.jpg
    parser.add_argument('--test_images', type=str, nargs='+', default=[], help='用于预测的测试图像路径列表。可用空格或逗号分隔多个路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果使用逗号分隔的路径，将其分割为列表
    if len(args.test_images) == 1 and ',' in args.test_images[0]:
        args.test_images = args.test_images[0].split(',')
    
    if args.mode == 'train':
        print("开始自定义手写字符识别任务...")
        
        # 1. 加载数据
        print("\n步骤1: 加载数据集...")
        train_loader, test_loader, num_classes = load_data(data_dir=args.data_dir, batch_size=args.batch_size)
        print(f"数据加载完成。类别数量: {num_classes}, 训练批次数: {len(train_loader)}, 测试批次数: {len(test_loader)}")
        
        # 2. 创建模型
        print("\n步骤2: 创建卷积神经网络模型...")
        model = EnhancedCNN(num_classes=num_classes)
        print("模型结构:")
        print(model)
        
        # 3. 训练模型
        print("\n步骤3: 开始训练模型...")
        if args.resume:
            print(f"从检查点恢复训练: {args.resume}")
        
        history = train(model, train_loader, test_loader, 
                       epochs=args.epochs, 
                       resume_from=args.resume, 
                       checkpoint_dir=args.checkpoint_dir)
        
        # 4. 评估最终模型
        print("\n步骤4: 评估最终模型...")
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f"最终测试结果 - 损失: {test_loss:.4f}, 准确率: {test_acc:.2f}%")
        
        # 5. 可视化结果
        print("\n步骤5: 可视化训练结果和模型预测...")
        visualize_results(history, model, test_loader, num_classes)
        
        # 6. 保存最终模型
        print("\n步骤6: 保存最终模型...")
        final_model_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
        save_model(model, final_model_path)
        
        print("\n任务完成！")
        print(f"检查点和模型已保存在: {args.checkpoint_dir}")
        print(f"最佳模型: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")
        print(f"最终模型: {final_model_path}")
    
    elif args.mode == 'predict':
        print("开始手写字符识别预测...")
        
        # 1. 加载数据集信息（仅获取类别信息）
        print("\n步骤1: 加载数据集信息...")
        try:
            # 尝试加载数据集以获取类别信息
            _, _, num_classes = load_data(data_dir=args.data_dir, batch_size=1)
            
            # 获取类别名称
            dataset = HandwritingDataset(root_dir=args.data_dir, transform=None, train=True)
            class_names = dataset.classes
            print(f"加载了 {len(class_names)} 个类别: {class_names}")
            # 通过label_dict的映射关系打印出类别对应的真实值
            try:
                # 将字符串类型的类别转换为整数索引，并获取对应的符号
                mapped_labels = [label_dict[int(cls)] for cls in class_names]
                print("类别符号映射:", ", ".join([f"{cls}→'{sym}'" for cls, sym in zip(class_names, mapped_labels)]))
            except (ValueError, KeyError) as e:
                print(f"标签映射异常: {e}，请检查label_dict定义")
            
        except Exception as e:
            print(f"警告: 无法加载数据集信息: {e}")
            print("将使用数字索引作为类别标签")
            num_classes = 15  # 默认类别数量
            class_names = None
        
        # 2. 加载模型
        print("\n步骤2: 加载模型...")
        model_path = args.model_path
        if not model_path:
            # 如果未指定模型路径，尝试使用最佳模型
            best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                model_path = best_model_path
                print(f"使用最佳模型: {model_path}")
            else:
                # 尝试使用最终模型
                final_model_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
                if os.path.exists(final_model_path):
                    model_path = final_model_path
                    print(f"使用最终模型: {model_path}")
                else:
                    print("错误: 未找到可用的模型文件")
                    return
        
        # 创建模型实例
        model = EnhancedCNN(num_classes=num_classes)
        
        # 加载模型权重
        try:
            # 尝试加载完整的检查点（包含模型状态字典）
            checkpoint = torch.load(model_path)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"模型已从 {model_path} 加载")
        except Exception as e:
            print(f"错误: 无法加载模型: {e}")
            return
        
        # 3. 准备测试图像
        print("\n步骤3: 准备测试图像...")
        test_images = args.test_images
        
        # 如果未提供测试图像，使用示例图像
        if not test_images:
            print("未提供测试图像，将使用示例图像")
            # 尝试从测试集中获取一些图像
            try:
                test_dataset = HandwritingDataset(
                    root_dir=args.data_dir,
                    transform=None,
                    train=False
                )
                
                # 获取三个不同类别的样本
                sample_indices = [0, len(test_dataset)//3, 2*len(test_dataset)//3]
                test_images = []
                true_labels = []
                
                for idx in sample_indices:
                    if idx < len(test_dataset):
                        img_path, label = test_dataset.samples[idx]
                        test_images.append(img_path)
                        if class_names:
                            true_labels.append(class_names[label])
                        else:
                            true_labels.append(str(label))
                
                print(f"从测试集中选择了 {len(test_images)} 张图像")
                print(f"真实标签: {true_labels}")
            except Exception as e:
                print(f"警告: 无法从测试集获取图像: {e}")
                print("请提供测试图像路径")
                return
        
        # 4. 进行预测
        print("\n步骤4: 进行预测...")
        results = predict(model, test_images, class_names)
        
        # 5. 输出预测结果
        print("\n步骤5: 预测结果...")
        for i, result in enumerate(results):
            print(f"图像 {i+1}: {result['path']}")
            print(f"  预测类别: {result['predicted_label']}")
            print(f"  置信度: {result['confidence']:.4f}")
        
        print("\n预测完成！结果已保存为 'prediction_results.png'")
        
        # 6. 批量预测示例（使用numpy数组）
        print("\n步骤6: 批量预测示例...")
        try:
            # 读取图像为numpy数组
            images_array = []
            for img_path in test_images:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images_array.append(img)
            
            if images_array:
                batch_results = predict_batch_from_array(model, images_array, class_names)
                
                print("批量预测结果:")
                for i, result in enumerate(batch_results):
                    print(f"图像 {i+1}:")
                    print(f"  预测类别: {result['predicted_label']}")
                    print(f"  置信度: {result['confidence']:.4f}")
        except Exception as e:
            print(f"批量预测示例出错: {e}")

# 当作为脚本运行时执行主函数
if __name__ == "__main__":
    main()