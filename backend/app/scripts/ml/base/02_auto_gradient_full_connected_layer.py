import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties

# 设置中文字体支持
# 尝试使用macOS常见的中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'Songti SC', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建中文字体属性对象，用于单独设置某些元素的字体
chineseFont = FontProperties(family='Arial Unicode MS')


"""
全连接层神经网络自动微分示例

本示例展示了如何使用PyTorch构建一个简单的全连接层神经网络，
通过自动微分和梯度下降来估算线性回归参数(w,b)。
我们将随机生成一些数据点，这些数据点满足线性关系 y = wx + b + noise，
然后训练神经网络来学习参数w和b。
"""

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 第一部分：生成随机数据
print("\n=== 生成随机数据 ===\n")

# 设置真实参数
true_w = 2.5
true_b = 4.0

# 样本数量
n_samples = 100

# 生成随机x值
x_data = np.random.rand(n_samples, 1) * 10  # 0到10之间的随机数

# 根据 y = wx + b + noise 生成y值
y_data = true_w * x_data + true_b + np.random.normal(0, 1, size=(n_samples, 1))

print(f"生成了{n_samples}个数据点")
print(f"真实参数: w = {true_w}, b = {true_b}")

# 将NumPy数组转换为PyTorch张量
x_tensor = torch.FloatTensor(x_data)
y_tensor = torch.FloatTensor(y_data)

# 绘制数据点
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, alpha=0.7, label='数据点')

# 绘制真实的线性关系
x_line = np.linspace(0, 10, 100).reshape(-1, 1)
y_line = true_w * x_line + true_b
plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'真实模型: y = {true_w}x + {true_b}')
plt.title('随机生成的线性回归数据', fontproperties=chineseFont)
plt.xlabel('x', fontproperties=chineseFont)
plt.ylabel('y', fontproperties=chineseFont)
plt.legend(prop=chineseFont)
plt.grid(True)

# 保存随机生成的线性回归数据图
plt.savefig('linear_regression_data.png')

# 第二部分：定义神经网络模型
print("\n=== 定义神经网络模型 ===\n")

class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # 定义一个全连接层，输入维度为1，输出维度为1
        self.linear = torch.nn.Linear(1, 1)
    
    def forward(self, x):
        # 前向传播
        return self.linear(x)

# 实例化模型
model = LinearRegressionModel()

# 打印模型结构
print("模型结构:")
print(model)

# 打印初始参数
print("\n初始参数:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data.numpy().flatten()}")

# 第三部分：定义损失函数和优化器
print("\n=== 定义损失函数和优化器 ===\n")

# 使用均方误差损失函数
criterion = torch.nn.MSELoss()

# 使用随机梯度下降优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print(f"损失函数: {criterion}")
print(f"优化器: {optimizer}")

# 第四部分：训练模型
print("\n=== 训练模型 ===\n")

# 训练轮数
n_epochs = 100

# 存储损失值用于绘图
loss_values = []

# 存储参数变化用于绘图
w_values = []
b_values = []

# 训练循环
for epoch in range(n_epochs):
    # 前向传播
    y_pred = model(x_tensor)
    
    # 计算损失
    loss = criterion(y_pred, y_tensor)
    loss_values.append(loss.item())
    
    # 存储当前参数
    w_value = model.linear.weight.item()
    b_value = model.linear.bias.item()
    w_values.append(w_value)
    b_values.append(b_value)
    
    # 反向传播
    optimizer.zero_grad()  # 清除之前的梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数
    
    # 每10轮打印一次进度
    if (epoch + 1) % 10 == 0:
        print(f"轮次 {epoch+1}/{n_epochs}, 损失: {loss.item():.4f}, w: {w_value:.4f}, b: {b_value:.4f}")

# 获取最终学习到的参数
final_w = model.linear.weight.item()
final_b = model.linear.bias.item()

print("\n训练完成!")
print(f"学习到的参数: w = {final_w:.4f}, b = {final_b:.4f}")
print(f"真实参数: w = {true_w}, b = {true_b}")
print(f"参数误差: w_error = {abs(final_w - true_w):.4f}, b_error = {abs(final_b - true_b):.4f}")

# 第五部分：可视化结果
print("\n=== 可视化结果 ===\n")

# 绘制损失曲线
plt.figure(figsize=(12, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, n_epochs+1), loss_values, 'b-')
plt.title('训练过程中的损失变化', fontproperties=chineseFont)
plt.xlabel('训练轮次', fontproperties=chineseFont)
plt.ylabel('损失值', fontproperties=chineseFont)
plt.grid(True)

# 参数变化曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, n_epochs+1), w_values, 'r-', label='w')
plt.plot(range(1, n_epochs+1), b_values, 'g-', label='b')
plt.axhline(y=true_w, color='r', linestyle='--', alpha=0.5, label='真实w')
plt.axhline(y=true_b, color='g', linestyle='--', alpha=0.5, label='真实b')
plt.title('参数变化曲线', fontproperties=chineseFont)
plt.xlabel('训练轮次', fontproperties=chineseFont)
plt.ylabel('参数值', fontproperties=chineseFont)
plt.legend(prop=chineseFont)
plt.grid(True)

plt.tight_layout()

# 保存损失和参数变化曲线图
plt.savefig('training_curves.png')

# 绘制最终的回归线
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, alpha=0.7, label='数据点')

# 绘制真实的线性关系
plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'真实模型: y = {true_w}x + {true_b}')

# 绘制学习到的线性关系
y_pred_line = final_w * x_line + final_b
plt.plot(x_line, y_pred_line, 'b--', linewidth=2, label=f'学习模型: y = {final_w:.4f}x + {final_b:.4f}')

plt.title('线性回归结果对比', fontproperties=chineseFont)
plt.xlabel('x', fontproperties=chineseFont)
plt.ylabel('y', fontproperties=chineseFont)
plt.legend(prop=chineseFont)
plt.grid(True)

# 保存图像（可选）
plt.savefig('linear_regression_result.png')

# 显示图像（如果在支持图形界面的环境中运行）
plt.show()

print("\n全连接层神经网络自动微分示例完成！")

'''
=== 生成随机数据 ===

生成了100个数据点
真实参数: w = 2.5, b = 4.0

=== 定义神经网络模型 ===

模型结构:
LinearRegressionModel(
  (linear): Linear(in_features=1, out_features=1, bias=True)
)

初始参数:
linear.weight: [0.7645385]
linear.bias: [0.8300079]

=== 定义损失函数和优化器 ===

损失函数: MSELoss()
优化器: SGD (
Parameter Group 0
    dampening: 0
    differentiable: False
    foreach: None
    fused: None
    lr: 0.01
    maximize: False
    momentum: 0
    nesterov: False
    weight_decay: 0
)

=== 训练模型 ===

轮次 10/100, 损失: 3.2010, w: 2.9001, b: 1.3109
轮次 20/100, 损失: 2.9490, w: 2.8762, b: 1.4680
轮次 30/100, 损失: 2.7234, w: 2.8534, b: 1.6166
轮次 40/100, 损失: 2.5216, w: 2.8318, b: 1.7572
轮次 50/100, 损失: 2.3410, w: 2.8114, b: 1.8902
轮次 60/100, 损失: 2.1795, w: 2.7920, b: 2.0160
轮次 70/100, 损失: 2.0350, w: 2.7737, b: 2.1350
轮次 80/100, 损失: 1.9056, w: 2.7564, b: 2.2475
轮次 90/100, 损失: 1.7899, w: 2.7401, b: 2.3540
轮次 100/100, 损失: 1.6864, w: 2.7246, b: 2.4547

训练完成!
学习到的参数: w = 2.7231, b = 2.4644
真实参数: w = 2.5, b = 4.0
参数误差: w_error = 0.2231, b_error = 1.5356
'''