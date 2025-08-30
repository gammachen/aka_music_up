import torch

# 创建需要计算梯度的张量
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 定义函数 y = 2 * x^T * x
y = 2 * torch.dot(x, x)

# 反向传播以计算梯度
y.backward()

# 打印 x 的梯度 dy/dx
print(x.grad)
# tensor([ 4.,  8., 12.])