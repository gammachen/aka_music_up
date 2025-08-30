import torch
import numpy as np
import matplotlib.pyplot as plt

"""
PyTorch自动微分(Automatic Differentiation)示例

本示例展示了PyTorch中自动微分的基本原理和使用方法。
自动微分是深度学习框架的核心功能，它能够自动计算复杂函数的导数，
而无需手动推导数学公式。
"""

# 第一部分：标量计算的自动微分示例
print("\n=== 标量计算的自动微分示例 ===")

# 创建需要计算梯度的张量，requires_grad=True表示需要追踪计算历史
x = torch.tensor(2.0, requires_grad=True)
print(f"初始值 x = {x.item()}")

# 定义一个简单的函数 y = x^2 + 3x + 1
y = x**2 + 3*x + 1
print(f"函数 y = x^2 + 3x + 1，计算结果 y = {y.item()}")

# 反向传播以计算梯度
y.backward()

# 打印 x 的梯度 dy/dx
print(f"导数 dy/dx = 2x + 3，在 x = 2 处的值为 {x.grad.item()}")
print("验证：2*2 + 3 = 7")


# 第二部分：多次反向传播
print("\n=== 多次反向传播示例 ===")

# 重置梯度
x.grad.zero_()

# 创建一个新的计算图
z = x**3 - 5*x + 1
print(f"函数 z = x^3 - 5x + 1，计算结果 z = {z.item()}")

# 反向传播
z.backward()

# 打印梯度
print(f"导数 dz/dx = 3x^2 - 5，在 x = 2 处的值为 {x.grad.item()}")
print("验证：3*2^2 - 5 = 3*4 - 5 = 12 - 5 = 7")


# 第三部分：矩阵运算的自动微分
print("\n=== 矩阵运算的自动微分示例 ===")

# 创建矩阵
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = torch.tensor([2.0, 1.0])

print("矩阵 A:")
print(A)
print("向量 b:")
print(b)

# 矩阵乘法
c = A @ b  # 等价于 torch.matmul(A, b)
print("矩阵乘法结果 c = A @ b:")
print(c)

# 计算损失函数（假设为所有元素的和）
loss = c.sum()
print(f"损失函数（元素和）: {loss.item()}")

# 反向传播
loss.backward()

# 打印梯度
print("矩阵 A 的梯度 d(loss)/dA:")
print(A.grad)
print("验证：梯度应该是 [[2, 1], [2, 1]]，因为每个元素的梯度就是向量b的对应元素")


# 第四部分：使用自动微分进行简单优化
print("\n=== 使用自动微分进行简单优化示例 ===")

# 创建一个需要优化的参数
w = torch.tensor([1.0, 2.0], requires_grad=True)
print(f"初始参数 w = {w}")

# 定义学习率
learning_rate = 0.1

# 存储损失值用于绘图
loss_values = []

# 简单的梯度下降优化
for i in range(10):
    # 前向传播：计算预测值和损失
    pred = w[0]**2 + w[1]**2  # 简单的二次函数，最小值在(0,0)
    loss = pred
    loss_values.append(loss.item())
    
    # 打印当前状态
    print(f"迭代 {i+1}: w = {w.detach().numpy()}, 损失 = {loss.item():.4f}")
    
    # 反向传播：计算梯度
    loss.backward()
    
    # 更新参数（需要使用torch.no_grad()以避免记录梯度更新操作）
    with torch.no_grad():
        w -= learning_rate * w.grad
        
    # 重置梯度
    w.grad.zero_()

print(f"优化后的参数 w = {w.detach().numpy()}")

# 绘制损失曲线
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), loss_values, marker='o')
plt.title('优化过程中的损失变化')
plt.xlabel('迭代次数')
plt.ylabel('损失值')
plt.grid(True)

# 保存图像（可选）
# plt.savefig('loss_curve.png')

# 显示图像（如果在支持图形界面的环境中运行）
# plt.show()

print("\n自动微分示例完成！")

'''
=== 标量计算的自动微分示例 ===
初始值 x = 2.0
函数 y = x^2 + 3x + 1，计算结果 y = 11.0
导数 dy/dx = 2x + 3，在 x = 2 处的值为 7.0
验证：2*2 + 3 = 7

=== 多次反向传播示例 ===
函数 z = x^3 - 5x + 1，计算结果 z = -1.0
导数 dz/dx = 3x^2 - 5，在 x = 2 处的值为 7.0
验证：3*2^2 - 5 = 3*4 - 5 = 12 - 5 = 7

=== 矩阵运算的自动微分示例 ===
矩阵 A:
tensor([[1., 2.],
        [3., 4.]], requires_grad=True)
向量 b:
tensor([2., 1.])
矩阵乘法结果 c = A @ b:
tensor([ 4., 10.], grad_fn=<MvBackward0>)
损失函数（元素和）: 14.0
矩阵 A 的梯度 d(loss)/dA:
tensor([[2., 1.],
        [2., 1.]])
验证：梯度应该是 [[2, 1], [2, 1]]，因为每个元素的梯度就是向量b的对应元素

=== 使用自动微分进行简单优化示例 ===
初始参数 w = tensor([1., 2.], requires_grad=True)
迭代 1: w = [1. 2.], 损失 = 5.0000
迭代 2: w = [0.8 1.6], 损失 = 3.2000
迭代 3: w = [0.64 1.28], 损失 = 2.0480
迭代 4: w = [0.51199996 1.0239999 ], 损失 = 1.3107
迭代 5: w = [0.40959996 0.8191999 ], 损失 = 0.8389
迭代 6: w = [0.32767996 0.6553599 ], 损失 = 0.5369
迭代 7: w = [0.26214397 0.52428794], 损失 = 0.3436
迭代 8: w = [0.20971517 0.41943035], 损失 = 0.2199
迭代 9: w = [0.16777214 0.3355443 ], 损失 = 0.1407
迭代 10: w = [0.13421771 0.26843542], 损失 = 0.0901
优化后的参数 w = [0.10737417 0.21474834]

自动微分示例完成！
'''