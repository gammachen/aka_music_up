#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
基于PyTorch的MNIST手写数字识别
本脚本详细展示了深度学习的完整流程，包括：
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
from torch.utils.data import DataLoader  # 数据加载器

import torchvision  # 计算机视觉库
import torchvision.transforms as transforms  # 数据变换
from torchvision.datasets import MNIST  # MNIST数据集

import matplotlib.pyplot as plt  # 可视化
import matplotlib
from matplotlib.font_manager import FontProperties
import numpy as np
import time

# 设置中文字体支持
# 尝试使用macOS常见的中文字体
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

# ====================== 第一步：数据加载与预处理 ======================

def load_data():
    '''
    加载MNIST数据集并进行预处理
    '''
    # 定义数据变换：将图像转换为PyTorch张量，并标准化像素值到[-1,1]区间
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将PIL图像或numpy.ndarray转换为张量
        transforms.Normalize((0.1307,), (0.3081,))  # 使用MNIST的均值和标准差进行标准化
    ])
    
    # 加载训练集
    train_dataset = MNIST(
        root='./data',  # 数据存储路径
        train=True,     # 指定为训练集
        download=True,  # 如果数据不存在，则下载
        transform=transform  # 应用上面定义的变换
    )
    
    # 加载测试集
    test_dataset = MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # 创建数据加载器，用于批量加载数据
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,     # 每批加载64张图片
        shuffle=True,      # 打乱数据，减少模型对数据顺序的依赖
        num_workers=2      # 使用2个子进程加载数据
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,   # 测试时使用更大的批量
        shuffle=False,     # 测试时不需要打乱数据
        num_workers=2
    )
    
    return train_loader, test_loader

# ====================== 第二步：构建神经网络模型 ======================

class CNN(nn.Module):
    '''
    卷积神经网络模型定义
    '''
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积层：输入1通道(灰度图)，输出32通道，卷积核大小3x3
        # 卷积层提取图像的局部特征
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        
        # 第二个卷积层：输入32通道，输出64通道，卷积核大小3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # 池化层：最大池化，窗口大小2x2
        # 池化层降低特征图的分辨率，减少参数数量，提高模型的平移不变性
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层1：输入维度7*7*64，输出维度128
        # 全连接层将特征图转换为分类所需的特征向量
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        
        # 全连接层2：输入维度128，输出维度10（对应10个数字类别）
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout层：以0.5的概率随机丢弃神经元，防止过拟合
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        '''
        前向传播过程
        参数:
            x: 输入图像，形状为[batch_size, 1, 28, 28]
        返回:
            输出预测，形状为[batch_size, 10]
        '''
        # 第一个卷积块：卷积 -> ReLU激活 -> 池化
        # ReLU激活函数引入非线性，使网络能够学习复杂模式
        x = self.pool(F.relu(self.conv1(x)))  # 输出尺寸: [batch_size, 32, 14, 14]
        
        # 第二个卷积块：卷积 -> ReLU激活 -> 池化
        x = self.pool(F.relu(self.conv2(x)))  # 输出尺寸: [batch_size, 64, 7, 7]
        
        # 将特征图展平为向量
        x = x.view(-1, 7 * 7 * 64)  # 输出尺寸: [batch_size, 7*7*64]
        
        # 全连接层1：线性变换 -> ReLU激活 -> Dropout
        x = self.dropout(F.relu(self.fc1(x)))  # 输出尺寸: [batch_size, 128]
        
        # 全连接层2：线性变换（不使用激活函数，因为后面会用交叉熵损失）
        x = self.fc2(x)  # 输出尺寸: [batch_size, 10]
        
        return x

# ====================== 第三步：训练模型 ======================

def train(model, train_loader, test_loader, epochs=10):
    '''
    训练模型函数
    参数:
        model: 待训练的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        epochs: 训练轮数
    返回:
        训练历史记录（损失和准确率）
    '''
    # 将模型移动到指定设备（GPU或CPU）
    model = model.to(device)
    
    # 定义损失函数：交叉熵损失
    # 交叉熵损失适合多分类问题，测量预测分布与真实分布的差异
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器：Adam优化器
    # 优化器通过计算梯度来更新模型参数，Adam自适应调整学习率
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 学习率调度器：当验证集性能停止提升时，降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # 训练开始时间
    start_time = time.time()
    
    # 训练循环
    for epoch in range(epochs):
        # 设置为训练模式（启用Dropout等）
        model.train()
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 遍历训练数据批次
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 将数据移动到指定设备
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 清零梯度
            # 每个批次开始前清除之前的梯度，防止梯度累积
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播：计算梯度
            # 通过自动微分计算损失函数对各参数的梯度
            loss.backward()
            
            # 更新参数
            # 根据计算的梯度和优化算法更新模型参数
            optimizer.step()
            
            # 累计损失
            train_loss += loss.item()
            
            # 计算准确率
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # 每100个批次打印一次进度
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
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
    
    # 训练结束，打印总用时
    total_time = time.time() - start_time
    print(f'训练完成！总用时: {total_time:.2f}秒')
    
    return history

# ====================== 第四步：评估模型 ======================

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
    # 设置为评估模式（禁用Dropout等）
    model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    # 禁用梯度计算，减少内存使用并加速推理
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
            _, predicted = outputs.max(1)  # 获取最大概率的索引
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # 计算平均损失和准确率
    test_loss = test_loss / len(data_loader)
    accuracy = 100. * correct / total
    
    return test_loss, accuracy

# ====================== 第五步：可视化结果 ======================

def visualize_results(history, model, test_loader):
    '''
    可视化训练结果和模型预测
    参数:
        history: 训练历史记录
        model: 训练好的模型
        test_loader: 测试数据加载器
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
    plt.savefig('mnist_training_curves.png')
    plt.show()
    
    # 2. 可视化模型预测
    model.eval()
    
    # 获取一批测试数据
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # 选择前25张图片
    images = images[:25]
    labels = labels[:25]
    
    # 获取模型预测
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()
    
    # 显示图片和预测结果
    images = images.numpy()
    
    # 创建5x5网格显示图片
    plt.figure(figsize=(12, 12))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        # 将图像从[1,28,28]转换为[28,28]并反转标准化
        plt.imshow(images[i][0] * 0.3081 + 0.1307, cmap='gray')
        
        # 设置标题：绿色表示预测正确，红色表示预测错误
        if predicted[i] == labels[i]:
            plt.title(f'预测: {predicted[i]}', color='green')
        else:
            plt.title(f'预测: {predicted[i]}\n实际: {labels[i]}', color='red')
            
        plt.axis('off')  # 隐藏坐标轴
    
    plt.tight_layout()
    plt.savefig('mnist_predictions.png')
    plt.show()

# ====================== 主函数 ======================

def main():
    '''
    主函数：执行完整的训练和评估流程
    '''
    print("开始MNIST手写数字识别任务...")
    
    # 1. 加载数据
    print("\n步骤1: 加载MNIST数据集...")
    train_loader, test_loader = load_data()
    print(f"数据加载完成。训练批次数: {len(train_loader)}, 测试批次数: {len(test_loader)}")
    
    # 2. 创建模型
    print("\n步骤2: 创建卷积神经网络模型...")
    model = CNN()
    print("模型结构:")
    print(model)
    
    # 3. 训练模型
    print("\n步骤3: 开始训练模型...")
    history = train(model, train_loader, test_loader, epochs=5)
    
    # 4. 评估最终模型
    print("\n步骤4: 最终模型评估...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"最终测试损失: {test_loss:.4f}, 最终测试准确率: {test_acc:.2f}%")
    
    # 5. 可视化结果
    print("\n步骤5: 可视化结果...")
    visualize_results(history, model, test_loader)
    
    # 6. 保存模型
    print("\n步骤6: 保存模型...")
    torch.save(model.state_dict(), 'mnist_cnn_model.pth')
    print("模型已保存为 'mnist_cnn_model.pth'")
    
    print("\n完成所有步骤！")

# 当脚本直接运行时执行主函数
if __name__ == "__main__":
    main()