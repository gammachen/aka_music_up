以下是关于图像处理中"自适应二值化"的详细讲解：

### 一、核心逻辑
在代码中体现的关键逻辑：
```python
binary_img = cv2.adaptiveThreshold(
    resized_img, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,  # 实际代码中该参数被注释
    21, 10
)
```
1. **动态阈值计算**：不同于全局阈值，对图像每个像素点周围的邻域（21x21区域）计算局部阈值
2. **高斯加权平均**：使用`ADAPTIVE_THRESH_GAUSSIAN_C`方法，邻域内像素的权重按高斯分布计算
3. **阈值反转处理**：当使用`cv2.THRESH_BINARY_INV`时，生成白底黑字的二值图像（适合文字识别场景）

### 二、使用场景
基于代码上下文分析：
1. **光照不均匀图像**：处理音乐相关图像（乐谱、乐器等）时常见的光照差异问题
2. **小尺寸图像处理**：配合24x24尺寸调整（适合机器学习模型输入）
3. **特征增强预处理**：为后续形态学操作和特征提取做准备

### 三、核心参数详解
| 参数 | 代码示例值 | 作用 | 调整建议 |
|------|-----------|------|----------|
| `blockSize` | 21 | 计算阈值的邻域大小 | 需为奇数，建议值：11-31（根据图像尺寸调整） |
| `C` | 10 | 从均值/加权均值中减去的常数 | 范围：5-15（过大会导致噪声增加） |
| `adaptiveMethod` | ADAPTIVE_THRESH_GAUSSIAN_C | 阈值计算方法 | 高斯法（降噪更好）或均值法（计算更快） |
| `maxValue` | 255 | 二值化后的最大值 | 通常保持255（白色） |

### 四、效果对比建议
可在代码中尝试以下组合观察效果：
```python
# 不同参数组合示例
cv2.adaptiveThreshold(..., cv2.ADAPTIVE_THRESH_MEAN_C, 31, 15)
cv2.adaptiveThreshold(..., cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 15, 7)
```

**最佳实践**：对于乐谱等含细密线条的图像，建议保持`ADAPTIVE_THRESH_GAUSSIAN_C`配合中等大小的blockSize（如21），C值根据具体光照条件微调。