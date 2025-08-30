# 手写算式识别系统技术文档

## 目录

- [手写算式识别系统技术文档](#手写算式识别系统技术文档)
  - [目录](#目录)
  - [1. 系统概述](#1-系统概述)
  - [2. 技术架构](#2-技术架构)
  - [3. 图像分割模块](#3-图像分割模块)
    - [3.1 预处理](#31-预处理)
    - [3.2 投影分割](#32-投影分割)
      - [水平投影](#水平投影)
      - [垂直投影](#垂直投影)
      - [基于投影的切割](#基于投影的切割)
    - [3.3 膨胀分组](#33-膨胀分组)
    - [3.4 分割结果可视化](#34-分割结果可视化)
  - [4. 手写识别模型](#4-手写识别模型)
    - [4.1 模型架构](#41-模型架构)
    - [4.2 数据集处理](#42-数据集处理)
      - [4.2.1 自建数据集生成](#421-自建数据集生成)
      - [4.2.2 数据集加载与处理](#422-数据集加载与处理)
    - [4.3 训练流程](#43-训练流程)
    - [4.4 预测接口](#44-预测接口)
  - [5. 实验结果与分析](#5-实验结果与分析)
    - [5.1 分割效果](#51-分割效果)
    - [5.2 识别准确率](#52-识别准确率)
    - [5.3 完整流程示例](#53-完整流程示例)
  - [6. 总结与展望](#6-总结与展望)

## 1. 系统概述

本文档详细介绍了一个专门用于识别小学生手写口算题的图像处理与文本识别系统。该系统能够自动分割手写算式图像，识别其中的数字和运算符号，并输出结构化的识别结果。系统主要由两个核心模块组成：图像分割模块和手写识别模型。

系统处理流程如下：

1. 输入手写算式图像
2. 图像预处理（灰度转换、二值化）
3. 基于投影法进行行列分割
4. 使用膨胀算法进行算式分组
5. 对分割后的单个字符进行识别
6. 输出结构化的识别结果

## 2. 技术架构

系统采用模块化设计，主要包含以下技术组件：

- **图像处理库**：OpenCV (cv2)，用于图像读取、预处理和分割
- **数值计算库**：NumPy，用于高效的数组操作
- **可视化工具**：Matplotlib，用于结果可视化
- **深度学习框架**：PyTorch，用于构建和训练手写识别模型
- **图像处理库**：PIL (Python Imaging Library)，用于图像格式转换

系统的核心代码文件包括：

- `image_segmentation.py`：实现图像分割和预处理功能
- `a_05_2_custom_handwriting_model_v2.py`：实现手写字符识别模型

## 3. 图像分割模块

图像分割模块是系统的前端处理部分，负责将输入的手写算式图像切分为单个字符，为后续的识别做准备。

### 3.1 预处理

预处理阶段主要完成图像的灰度转换和二值化处理：

```python
def preprocess_image(self):
    '''
    预处理图像，包括灰度转换和二值化
    '''
    # 读取图像并转换为灰度图
    img = cv2.imread(self.image_path)
    if len(img.shape) == 3 and img.shape[2] == 3:
        self.gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        self.gray_img = img.copy()

    # 添加二值化处理
    _, self.binary_img = cv2.threshold(self.gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print("预处理完成，二值化图像已生成")
```

二值化采用OTSU自适应阈值方法，能够自动确定最佳阈值，将图像转换为黑白二值图像，便于后续处理。同时使用THRESH_BINARY_INV反转图像，使得文字为白色（255），背景为黑色（0）。

### 3.2 投影分割

系统采用投影法进行图像分割，包括水平投影和垂直投影：

#### 水平投影

```python
def horizontal_projection(self):
    '''
    计算图像的水平投影
    返回:
        水平投影数组
    '''
    h, w = self.binary_img.shape  # 使用二值化图像
    horizontal_projection = np.zeros(h)

    for i in range(h):
        horizontal_projection[i] = np.count_nonzero(self.binary_img[i, :])

    return horizontal_projection
```

水平投影是将图像在水平方向上的像素分布投影到垂直轴上，形成一个一维数组。数组中的每个值表示对应行中非零像素的数量。下图展示了一个手写算式图像的水平投影结果：

![水平投影示例](projections/projection_x_20250402_211900.png)

从上图可以看出，水平投影能够清晰地显示出文本行的位置，投影值较高的区域对应着文字所在的行，而投影值为零或较低的区域则对应着行间的空白区域。通过分析这些峰谷，系统可以准确地确定每一行的边界位置。

#### 垂直投影

```python
def vertical_projection(self, binary_img=None):
    '''
    计算图像的垂直投影
    参数:
        binary_img: 二值化图像，默认为类属性中的二值化图像
    返回:
        垂直投影数组
    '''
    if binary_img is None:
        binary_img = self.binary_img

    h, w = binary_img.shape
    vertical_projection = np.zeros(w)

    for j in range(w):
        vertical_projection[j] = np.count_nonzero(binary_img[:, j])

    return vertical_projection
```

垂直投影是将图像在垂直方向上的像素分布投影到水平轴上，形成一个一维数组。数组中的每个值表示对应列中非零像素的数量。下图展示了一个手写算式图像的垂直投影结果：

![垂直投影示例](projections/projection_y_20250402_211901.png)

从上图可以看出，垂直投影能够清晰地显示出字符的位置，投影值较高的区域对应着字符所在的位置，而投影值为零或较低的区域则对应着字符间的空白区域。通过分析这些峰谷，系统可以准确地确定每个字符的边界位置。

投影法的原理是统计图像在水平或垂直方向上的像素分布，通过分析投影数组中的峰谷，可以确定字符的边界位置。下图展示了水平和垂直投影的组合效果：

![水平垂直投影组合](projections/merged_projections.png)

通过组合使用水平和垂直投影，系统能够实现对手写算式的二维分割，首先通过水平投影分割出行，然后通过垂直投影分割出每行中的字符。这种方法简单高效，对于结构规整的手写算式具有良好的分割效果。

#### 基于投影的切割

```python
def cut_by_projection(self, projection, axis='horizontal', input_img=None):
    '''
    根据投影结果切割图像，并记录每个子图像的坐标
    '''
    sub_images = []  # 存储切割后的子图像
    coordinates = []  # 存储每个子图像的坐标
    start = 0  # 记录当前切割区域的起始位置
    in_object = False  # 标记是否处于一个切割区域内

    # 遍历投影数组，寻找切割点
    for i, value in enumerate(projection):
        if not in_object and value > 0:  # 进入一个切割区域
            start = i
            in_object = True
        elif in_object and value == 0:  # 离开一个切割区域
            if axis == 'horizontal':
                # 水平切割，提取从 start 到 i 的行
                sub_images.append(input_img[start:i, :])
                coordinates.append((start, i, 0, input_img.shape[1]))
            elif axis == 'vertical':
                # 垂直切割，提取从 start 到 i 的列
                sub_images.append(input_img[:, start:i])
                coordinates.append((0, input_img.shape[0], start, i))
            in_object = False

    # 处理最后一个切割区域
    if in_object:
        # 类似逻辑处理最后一个区域...

    return list(zip(sub_images, coordinates))  # 返回子图像及其坐标的列表
```

### 3.3 膨胀分组

为了更好地分组算式，系统采用了图像膨胀技术，将相邻的字符连接起来，形成完整的算式组：

```python
# 使用图像膨胀的方法来分组
kernel = np.ones((5, 5), np.uint8)  # 膨胀核大小
row_img_b = cv2.dilate(row, kernel, iterations=9)  # 图像膨胀9次

# 计算膨胀后的垂直投影
vertical_projection_dilated = self.vertical_projection(row_img_b)
dilated_segments = self.cut_by_projection(vertical_projection_dilated, axis='vertical', input_img=row_img_b)
```

膨胀操作使得相邻的字符连接在一起，形成更大的连通区域，便于将属于同一个算式的字符分为一组。下图展示了膨胀前后的对比：

![膨胀前后对比](formula_1_segments_v2_20250404_210451/row_1_dilation.png)

膨胀后，系统根据膨胀图像的分割结果，将原始图像中的小分割块映射到对应的大分割块中，完成算式的分组：

```python
# 将未膨胀之前切分出来的小块丢到大块里面，组装成组
for dilated_segment, dilated_coords in dilated_segments:
    group = []
    for segment_idx, (segment, coords) in enumerate(adjusted_segments):
        # 修正判断条件：小块的列范围应在大块范围内
        if (coords[2] >= dilated_coords[2]) and (coords[3] <= dilated_coords[3]):
            group.append((segment, coords))
    
    if group:
        grouped_segments.append(group)
```

### 3.4 分割结果可视化

系统提供了多种可视化方法，用于展示分割结果：

```python
def visualize_segments(self, segments, output_dir=None):
    # 将嵌套列表展平
    flat_segments = []  # 初始化展平后的分割块列表
    for row in segments:  # 遍历每一行
        for group in row:  # 遍历每一组
            for segment in group:  # 遍历每个分割块
                if isinstance(segment, tuple) and len(segment) > 0:  # 检查分割块是否有效
                    flat_segments.append(segment[0])  # 只取子图像部分
    
    # 创建图形并显示
    plt.figure(figsize=(cols * 2, rows * 2))  # 设置图形大小
    for i, segment in enumerate(flat_segments):  # 遍历每个分割块
        plt.subplot(rows, cols, i + 1)  # 创建子图
        plt.imshow(segment, cmap='gray')  # 显示图像
        plt.title(f'Segment {i+1}')  # 设置标题
        plt.axis('off')  # 关闭坐标轴
```

分割结果示例：

![分割结果示例](formula_1_segments_v2_20250404_210451/row_1_final_groups.png)

## 4. 手写识别模型

手写识别模型是系统的核心部分，负责将分割后的单个字符图像识别为对应的数字或符号。

### 4.1 模型架构

系统采用了一个增强版的卷积神经网络（CNN）模型，具有以下特点：

- 多层卷积结构，提取图像特征
- 批归一化层，加速训练并提高稳定性
- Dropout层，防止过拟合
- 残差连接，缓解梯度消失问题

模型定义如下：

```python
class EnhancedCNN(nn.Module):
    '''
    增强版卷积神经网络模型，适用于MNIST和自定义数据集的组合
    '''
    def __init__(self, num_classes=15):
        super(EnhancedCNN, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)
        
        # 第三个卷积块（带残差连接）
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 第三个卷积块（带残差连接）
        identity = x
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x + identity)  # 残差连接
        
        # 全连接层
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x
```

### 4.2 数据集处理

系统使用了两种数据集：

1. **自定义手写数据集**：包含数字0-9和运算符号（+、-、×、÷、=）
2. **MNIST数据集**：标准手写数字数据集，用于增强模型对数字的识别能力

#### 4.2.1 自建数据集生成

为了获得足够多样化的训练样本，系统使用了基于字体文件的数据集生成方法。这种方法可以快速生成大量带标签的训练数据，并通过各种变换增强数据多样性。

自建数据集生成的核心流程如下：

1. **字体加载**：从系统和自定义字体目录中加载多种字体文件
2. **图像生成**：为每种字体、每个字符生成图像
3. **数据增强**：应用旋转、位移、噪声等变换增强数据多样性

```python
# 定义要生成的字符类别
label_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 
             7: '7', 8: '8', 9: '9', 10: '=', 11: '+', 12: '-', 
             13: '×', 14: '÷'}

# 为每个类别创建对应的文件夹
for value, char in label_dict.items():
    train_images_dir = "dataset"+"/"+str(value)
    os.makedirs(train_images_dir, exist_ok=True)
```

图像生成的核心函数实现如下：

```python
def makeImage(label_dict, font_path, width=28, height=28, rotate=0, 
           noise_level=0, apply_transforms=True, show_debug_info=False):
    """生成手写数字和符号图片"""
    # 从字典中取出键值对
    for value, char in label_dict.items():
        # 创建黑色背景图像
        img = Image.new("L", (width, height), 0)  # L模式表示灰度图
        draw = ImageDraw.Draw(img)
        
        # 动态调整字体大小，确保字符适合图像尺寸
        font_size = int(width * 1)  # 初始字体大小
        font = ImageFont.truetype(font_path, font_size)
        
        # 获取文本边界框和度量参数
        text_bbox = draw.textbbox((0, 0), char, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        ascent, descent = font.getmetrics()
        text_height = ascent + descent
        
        # 调整字体大小，确保字符不超出图像边界
        while text_width > width * max_width_ratio or text_height > height * max_height_ratio:
            font_size -= 1
            if font_size <= 8:  # 设置最小字体大小限制
                break
            font = ImageFont.truetype(font_path, font_size)
            text_bbox = draw.textbbox((0, 0), char, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            ascent, descent = font.getmetrics()
            text_height = ascent + descent
        
        # 计算字符位置，使其居中
        x = (width - text_width) / 2
        baseline_compensation = (ascent - descent) * 0.15
        y = (height - text_height) / 2 - baseline_compensation
        
        # 绘制字符
        draw.text((x, y), char, 255, font)  # 255表示白色
        
        # 应用旋转变换
        if rotate != 0:
            img = img.rotate(rotate, fillcolor=0)
        
        # 应用数据增强变换
        if apply_transforms:
            # 随机位移
            if random.random() < 0.3:
                shift_x = random.randint(-2, 2)
                shift_y = random.randint(-2, 2)
                img = img.transform(
                    img.size, 
                    Image.AFFINE, 
                    (1, 0, shift_x, 0, 1, shift_y),
                    resample=Image.BICUBIC
                )
            
            # 随机对比度调整
            if random.random() < 0.4:
                enhancer = ImageEnhance.Contrast(img)
                factor = random.uniform(0.8, 1.2)
                img = enhancer.enhance(factor)
            
            # 添加轻微的高斯模糊
            if random.random() < 0.5:
                blur_radius = random.uniform(0.3, 0.7)
                img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # 添加随机噪点
        if noise_level > 0:
            img_array = np.array(img)
            noise = np.random.randint(0, noise_level, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        
        # 保存图像
        time_value = int(round(time.time() * 1000))
        img_path = f"dataset/{value}/img-{value}_r-{rotate}_{time_value}.png"
        img.save(img_path)
```

系统支持多种字体来源：

1. **系统字体**：通过`matplotlib.font_manager.findSystemFonts()`获取系统中的字体
2. **自定义字体**：从指定目录加载自定义手写风格字体

```python
def generate_images(mode='mixed', apply_noise=True, show_debug_info=False):
    """生成图片主函数"""
    # 加载自定义字体
    script_dir = os.path.dirname(os.path.abspath(__file__))
    font_dir = os.path.join(script_dir, "data/fonts")
    custom_fonts = []
    
    if (mode == 'mixed' or mode == 'custom') and os.path.exists(font_dir):
        for font_name in os.listdir(font_dir):
            path_font_file = os.path.join(font_dir, font_name)
            if os.path.isfile(path_font_file) and path_font_file.lower().endswith(('.ttf', '.otf')):
                custom_fonts.append(path_font_file)
    
    # 加载系统字体
    system_fonts = [] if mode == 'custom' else get_system_fonts()
    
    # 合并字体列表
    if mode == 'mixed':
        all_fonts = custom_fonts + system_fonts
    elif mode == 'custom':
        all_fonts = custom_fonts
    else:  # standard模式
        all_fonts = system_fonts
    
    # 为每种字体生成图片
    for font_path in all_fonts:
        try:
            # 对系统字体和自定义字体使用不同的参数
            is_system_font = font_path not in custom_fonts
            
            # 设置旋转角度范围
            angle_range = range(-12, 12, 2) if is_system_font else range(-10, 10, 2)
            
            # 为每个角度生成图片
            for k in angle_range:
                noise_level = random.randint(0, 15) if apply_noise else 0
                makeImage(label_dict, font_path, rotate=k, noise_level=noise_level, 
                          apply_transforms=True, show_debug_info=show_debug_info)
        except Exception as e:
            print(f"使用字体 {font_path} 生成图片时出错: {e}")
            continue
```

通过这种方法，系统可以生成大量多样化的训练样本，如下图所示：

<!-- ![自建数据集样例](dataset/dataset_samples.png) -->

[自建数据集样例](./dataset_samples.pdf)

生成的数据集具有以下特点：

1. **多样性**：通过使用不同字体、旋转角度和变换，生成多样化的样本
2. **平衡性**：每个类别生成相似数量的样本，避免类别不平衡
3. **可控性**：可以通过参数控制生成过程，如噪声级别、变换类型等
4. **可扩展性**：可以轻松添加新的字体或字符类别

#### 4.2.2 数据集加载与处理

```python
# 自定义数据集类
class HandwritingDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, split_ratio=0.8):
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
        
        # 划分训练集和测试集
        np.random.shuffle(self.samples)
        split_idx = int(len(self.samples) * split_ratio)
        if train:
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
```

为了合并MNIST和自定义数据集，系统创建了一个MNIST包装类，确保标签映射一致：

```python
class MNISTWrapper(Dataset):
    def __init__(self, mnist_dataset, custom_classes):
        self.mnist_dataset = mnist_dataset
        self.custom_classes = custom_classes
        
        # 创建标签映射字典
        self.label_map = {}
        for i in range(10):  # MNIST只有0-9这10个标签
            if str(i) in self.custom_classes:
                # 如果自定义数据集中有这个数字标签，则直接映射到对应位置
                self.label_map[i] = self.custom_classes.index(str(i))
            else:
                # 如果自定义数据集中没有这个数字标签，则放在自定义数据集后面
                self.label_map[i] = len(self.custom_classes) + i
```

### 4.3 训练流程

模型训练流程包括以下步骤：

1. 数据加载与预处理
2. 定义损失函数和优化器
3. 训练循环
4. 模型评估与保存

```python
def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001):
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()  # 设置为训练模式
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # 每个epoch结束后评估模型
        model.eval()  # 设置为评估模式
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%')
```

### 4.4 预测接口

系统提供了两个预测接口，用于对分割后的图像进行识别：

```python
def predict_image_v1(images_array):
    """
    图像预测接口
    :param images_array: 要预测的图片对象数组
    :return: 预测结果字符串
    """
    # 确保模型已初始化
    if _model_instance is None:
        initialize_model(model_path='./best_model.pth')
    
    class_names = ['0', '1', '10', '11', '12', '13', '14', '2', '3', '4', '5', '6', '7', '8', '9']
    results = predict_batch_from_array(_model_instance, images_array, class_names, adaptiveThreashold=False)
    
    return results

def predict_image_v2(images_array):
    """
    图像预测接口
    :param images_array: 要预测的图片对象数组
    :return: 预测结果字符串
    """
    # 确保模型已初始化
    if _model_instance is None:
        initialize_model()
    
    # 预处理图像数组
    processed_images = []
    for img in images_array:
        # 如果是彩色图像，转换为灰度图
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 调整大小并显示
        resized_img = cv2.resize(img, (24, 24))
        
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
        outputs = _model_instance(batch_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_classes = torch.max(probabilities, 1)
        
        # 处理每个预测结果
        for i, (pred_idx, conf) in enumerate(zip(predicted_classes, confidence)):
            pred_idx = pred_idx.item()
            conf_value = conf.item()
            
            # 获取预测标签
            pred_label = label_dict[pred_idx]
            
            results.append({
                'index': i,
                'predicted_class': pred_idx,
                'predicted_label': pred_label,
                'confidence': conf_value
            })
    
    return results
```

## 5. 实验结果与分析

### 5.1 分割效果

系统对手写算式图像的分割效果良好，能够准确地将算式分割为行、组和单个字符。下面是一个分割结果示例：

![分割结果示例](formula_1_segments_v2_20250404_210451/row_1_final_groups.png)

从上图可以看出，系统成功地将第一行算式分割为多个组，每个组包含一个完整的算式。每个算式内的字符也被正确分割，为后续的识别提供了良好的基础。

### 5.2 识别准确率

根据处理结果的元数据，系统对分割后的字符识别准确率很高。以下是部分识别结果：

```json
{
  "/Users/shhaofu/Code/cursor-projects/aka_music/backend/app/scripts/ml/base/formula_1_output_20250404_210451/images/row1_group1_seg1.png": {
    "row": 1,
    "group": 1,
    "coordinates": [22, 71, 14, 48],
    "content": {
      "index": 0,
      "predicted_class": 1,
      "predicted_label": "1",
      "confidence": 1.0
    },
    "confidence": 1.0,
    "expression": "10-6=4",
    "result": 4
  }
}
```

从上述结果可以看出，系统不仅能够识别单个字符，还能够组合成完整的算式表达式，并计算出结果。识别置信度普遍很高，表明模型对训练数据有良好的拟合。

### 5.3 完整流程示例

下面是一个完整的处理流程示例，展示了从输入图像到最终识别结果的全过程：

1. **输入图像**：手写算式图像
2. **预处理**：灰度转换和二值化
3. **行分割**：基于水平投影分割行
4. **字符分割**：基于垂直投影分割字符
5. **算式分组**：使用膨胀算法分组
6. **字符识别**：使用CNN模型识别字符
7. **结果输出**：生成结构化的识别结果

处理结果包括：

- 分割后的字符图像
- 识别结果及置信度
- 完整的算式表达式
- 算式计算结果

## 6. 总结与展望

本文档详细介绍了一个用于识别小学生手写口算题的图像处理与文本识别系统。系统采用了投影法进行图像分割，使用膨胀算法进行算式分组，并使用卷积神经网络进行字符识别。实验结果表明，系统能够有效地处理手写算式图像，并输出准确的识别结果。

未来的改进方向包括：

1. **提高分割鲁棒性**：优化分割算法，使其能够处理更复杂的排版和更差的书写质量
2. **扩展识别范围**：增加对分数、小数等更复杂数学符号的支持
3. **优化模型性能**：使用更先进的网络架构和训练技术，提高识别准确率
4. **实现端到端训练**：将分割和识别模块整合，实现端到端的训练和优化
5. **添加错误检测与纠正**：增加对识别结果的验证机制，自动检测和纠正可能的错误

总的来说，该系统为手写算式的自动识别提供了一个有效的解决方案，具有广泛的应用前景，如教育辅助、作业批改自动化等领域。