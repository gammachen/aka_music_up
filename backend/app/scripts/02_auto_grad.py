import torch

# 创建需要计算梯度的张量
x = torch.tensor(2.0, requires_grad=True)

# 定义一个简单的函数 y = x^2 + 3x + 1
y = x ** 2 + 3 * x + 1

# 反向传播以计算梯度
y.backward()

# 打印 x 的梯度 dy/dx
print(x.grad)


# tensor(7.)
# 对函数进行求导：dy/dx = 2x + 3
# 将2带入进去，得到7
# 2x + 3 = 7
# 但是系统将这个梯度的计算泛化了，无需更新公式的情况，内部给于实现了自动微分