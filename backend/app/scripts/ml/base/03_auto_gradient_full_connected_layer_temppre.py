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
全连接层神经网络自动微分示例 - 温度转换

本示例展示了如何使用PyTorch构建一个简单的全连接层神经网络，
通过自动微分和梯度下降来学习摄氏度(C)与华氏度(F)之间的转换公式。
真实的转换公式为：华氏度(F) = 摄氏度(C) * (9/5) + 32
我们将生成一些摄氏温度数据点，计算对应的华氏温度，
然后训练神经网络来学习参数w(9/5)和b(32)。
"""

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 第一部分：生成温度数据
print("\n=== 生成温度数据 ===\n")

# 设置真实参数 (根据华氏度F = 摄氏度C * (9/5) + 32)
true_w = 9/5  # 1.8
true_b = 32.0

# 样本数量
n_samples = 100

# 生成随机摄氏温度值 (-30°C 到 50°C 之间的随机数)
x_data = np.random.uniform(-30, 50, (n_samples, 1))

# 根据 F = C*(9/5) + 32 + noise 生成华氏温度值 (添加少量噪声以模拟真实测量误差)
y_data = true_w * x_data + true_b + np.random.normal(0, 0.5, size=(n_samples, 1))

print(f"生成了{n_samples}个温度数据点")
print(f"真实参数: w = {true_w}, b = {true_b}")
print(f"转换公式: 华氏度(F) = 摄氏度(C) * {true_w} + {true_b}")

# 将NumPy数组转换为PyTorch张量
x_tensor = torch.FloatTensor(x_data)
y_tensor = torch.FloatTensor(y_data)

# 绘制数据点
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, alpha=0.7, label='温度数据点')

# 绘制真实的线性关系
x_line = np.linspace(-40, 60, 100).reshape(-1, 1)
y_line = true_w * x_line + true_b
plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'真实转换: F = {true_w}C + {true_b}')
plt.title('摄氏度与华氏度转换数据', fontproperties=chineseFont)
plt.xlabel('摄氏度 (°C)', fontproperties=chineseFont)
plt.ylabel('华氏度 (°F)', fontproperties=chineseFont)
plt.legend(prop=chineseFont)
plt.grid(True)

# 保存温度转换数据图
plt.savefig('temperature_conversion_data.png')

# 第二部分：定义神经网络模型
print("\n=== 定义神经网络模型 ===\n")

class TemperatureConversionModel(torch.nn.Module):
    def __init__(self):
        super(TemperatureConversionModel, self).__init__()
        # 定义一个全连接层，输入维度为1(摄氏度)，输出维度为1(华氏度)
        self.linear = torch.nn.Linear(1, 1)
    
    def forward(self, x):
        # 前向传播
        return self.linear(x)

# 实例化模型
model = TemperatureConversionModel()

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
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

print(f"损失函数: {criterion}")
print(f"优化器: {optimizer}")

# 第四部分：训练模型
print("\n=== 训练模型 ===\n")

# 训练轮数
n_epochs = 1000

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
    
    # 每100轮打印一次进度
    if (epoch + 1) % 100 == 0:
        print(f"轮次 {epoch+1}/{n_epochs}, 损失: {loss.item():.6f}, w: {w_value:.6f}, b: {b_value:.6f}")

# 获取最终学习到的参数
final_w = model.linear.weight.item()
final_b = model.linear.bias.item()

print("\n训练完成!")
print(f"学习到的参数: w = {final_w:.6f}, b = {final_b:.6f}")
print(f"真实参数: w = {true_w}, b = {true_b}")
print(f"参数误差: w_error = {abs(final_w - true_w):.6f}, b_error = {abs(final_b - true_b):.6f}")

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
plt.axhline(y=true_w, color='r', linestyle='--', alpha=0.5, label='真实w (9/5)')
plt.axhline(y=true_b, color='g', linestyle='--', alpha=0.5, label='真实b (32)')
plt.title('参数变化曲线', fontproperties=chineseFont)
plt.xlabel('训练轮次', fontproperties=chineseFont)
plt.ylabel('参数值', fontproperties=chineseFont)
plt.legend(prop=chineseFont)
plt.grid(True)

plt.tight_layout()

# 保存损失和参数变化曲线图
plt.savefig('temperature_training_curves.png')

# 绘制最终的转换曲线
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, alpha=0.7, label='温度数据点')

# 绘制真实的线性关系
plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'真实转换: F = {true_w}C + {true_b}')

# 绘制学习到的线性关系
y_pred_line = final_w * x_line + final_b
plt.plot(x_line, y_pred_line, 'b--', linewidth=2, label=f'学习转换: F = {final_w:.4f}C + {final_b:.4f}')

plt.title('摄氏度与华氏度转换结果对比', fontproperties=chineseFont)
plt.xlabel('摄氏度 (°C)', fontproperties=chineseFont)
plt.ylabel('华氏度 (°F)', fontproperties=chineseFont)
plt.legend(prop=chineseFont)
plt.grid(True)

# 保存图像
plt.savefig('temperature_conversion_result.png')

# 显示图像（如果在支持图形界面的环境中运行）
plt.show()

# 第六部分：使用模型进行温度转换预测
print("\n=== 使用模型进行温度转换预测 ===\n")

# 测试一些常见和极端的温度值
test_temps_c = np.array([-40, -17.8, 0, 10, 20, 25, 30, 37, 40, 50, 100, 200])
print("测试摄氏温度值:")
print("摄氏度(°C)\t华氏度(°F)(真实)\t华氏度(°F)(预测)\t误差(°F)")
print("-" * 70)
for temp_c in test_temps_c:
    # 使用真实公式计算
    temp_f_true = temp_c * (9/5) + 32
    
    # 使用模型预测
    temp_c_tensor = torch.FloatTensor([[temp_c]])
    temp_f_pred = model(temp_c_tensor).item()
    
    # 添加一些特殊温度点的说明
    note = ""
    if temp_c == -40: note = "(摄氏度与华氏度相等点)"
    elif temp_c == -17.8: note = "(0°F)"
    elif temp_c == 0: note = "(水的冰点)"
    elif temp_c == 37: note = "(人体正常体温)"
    elif temp_c == 100: note = "(水的沸点)"
    
    print(f"{temp_c:6.1f}\t\t{temp_f_true:6.2f}\t\t{temp_f_pred:6.2f}\t\t{abs(temp_f_true - temp_f_pred):6.4f}\t{note}")

print("\n全连接层神经网络学习温度转换公式示例完成！")