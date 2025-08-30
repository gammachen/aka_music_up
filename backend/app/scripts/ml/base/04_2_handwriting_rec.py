import cv2  # OpenCV库，用于图像处理
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.functional as F  # PyTorch函数式API
import numpy as np  # 数值计算库
import os  # 操作系统接口，用于文件路径操作

# 定义与训练时相同的CNN模型结构（卷积神经网络）
class CNN(nn.Module):
    """手写数字识别的卷积神经网络模型
    
    该模型基于MNIST数据集训练，用于识别手写数字（0-9）
    模型结构：2个卷积层，2个池化层，2个全连接层，使用Dropout防止过拟合
    """
    def __init__(self):
        """初始化模型的各层结构"""
        super(CNN, self).__init__()
        # 第一个卷积层：输入1通道(灰度图)，输出32通道，卷积核大小3x3
        # 步长为1，填充为1，保持输入输出尺寸一致
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        
        # 第二个卷积层：输入32通道，输出64通道，卷积核大小3x3
        # 步长为1，填充为1，保持输入输出尺寸一致
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # 最大池化层：窗口大小2x2，步长为2
        # 用于降低特征图的分辨率，减少参数数量，提高模型的平移不变性
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第一个全连接层：输入维度7*7*64，输出维度128
        # 7*7是经过两次池化后的特征图大小，64是第二个卷积层的输出通道数
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        
        # 第二个全连接层：输入维度128，输出维度10（对应10个数字类别0-9）
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout层：以0.5的概率随机丢弃神经元，防止过拟合
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        """前向传播过程
        
        Args:
            x: 输入图像张量，形状为[batch_size, 1, height, width]
               其中1表示单通道灰度图像
        
        Returns:
            输出预测张量，形状为[batch_size, 10]，表示10个数字类别的概率分布
        """
        # 第一个卷积块：卷积 -> ReLU激活 -> 池化
        # 卷积提取特征，ReLU引入非线性，池化降低分辨率
        x = self.pool(F.relu(self.conv1(x)))  # 输出尺寸: [batch_size, 32, height/2, width/2]
        
        # 第二个卷积块：卷积 -> ReLU激活 -> 池化
        x = self.pool(F.relu(self.conv2(x)))  # 输出尺寸: [batch_size, 64, height/4, width/4]
        
        # 展平操作：将特征图转换为一维向量，用于全连接层
        x = x.view(-1, 7 * 7 * 64)  # 输出尺寸: [batch_size, 7*7*64]
        
        # 全连接层1：线性变换 -> ReLU激活 -> Dropout
        x = self.dropout(F.relu(self.fc1(x)))  # 输出尺寸: [batch_size, 128]
        
        # 全连接层2：线性变换得到最终的类别分数
        x = self.fc2(x)  # 输出尺寸: [batch_size, 10]
        
        return x

# 创建CNN模型实例
model = CNN()

# 加载预训练模型参数（基于MNIST数据集训练的权重）
# 注意：确保mnist_cnn_model.pth文件存在于当前目录
model.load_state_dict(torch.load('mnist_cnn_model.pth'))

# 设置模型为评估模式（关闭Dropout等训练特性）
# 在推理阶段使用评估模式可以提高模型性能和稳定性
model.eval()

def preprocess(img_path):
    """图像预处理函数
    
    将输入图像转换为适合模型识别的格式，包括灰度化、二值化和降噪等步骤
    
    Args:
        img_path: 输入图像的文件路径
        
    Returns:
        tuple: (处理后的图像, 原始图像)
            - 处理后的图像：用于后续的数字识别
            - 原始图像：用于结果可视化
    """
    # 读取原始图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}，请检查文件路径是否正确")
    
    # 将彩色图像转换为灰度图像
    # 手写数字识别只需要形状信息，不需要颜色信息
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 自适应阈值二值化处理
    # 参数说明:
    # - 255: 最大像素值
    # - cv2.ADAPTIVE_THRESH_GAUSSIAN_C: 使用高斯加权的局部阈值
    # - cv2.THRESH_BINARY_INV: 反转二值化结果，使数字为白色(255)，背景为黑色(0)
    # - 11: 计算阈值的区域大小(11x11像素)
    # - 2: 从计算出的加权平均值减去的常数
    binary = cv2.adaptiveThreshold(gray, 255, 
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # 中值滤波去噪
    # 使用3x3的滤波窗口，可以有效去除椒盐噪声，保留边缘信息
    denoised = cv2.medianBlur(binary, 3)
    
    # 返回预处理后的图像和原始图像
    return denoised, img

def recognize(img_path):
    """识别图像中的手写数字
    
    处理流程：
    1. 图像预处理
    2. 检测数字轮廓
    3. 提取每个数字区域
    4. 调整大小并归一化
    5. 使用CNN模型进行预测
    6. 按位置排序结果
    
    Args:
        img_path: 输入图像的文件路径
        
    Returns:
        tuple: (识别结果列表, 原始图像)
            - 识别结果列表：每个元素为(x, y, w, h, digit)，表示数字的位置、大小和识别结果
            - 原始图像：用于结果可视化
    """
    # 调用预处理函数，获取处理后的图像和原始图像
    processed, original_img = preprocess(img_path)
    
    # 查找图像中的轮廓
    # cv2.RETR_EXTERNAL: 只检测最外层轮廓，忽略内部轮廓
    # cv2.CHAIN_APPROX_SIMPLE: 压缩水平、垂直和对角线段，只保留端点
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 存储识别结果的列表
    results = []
    
    # 处理每个检测到的轮廓
    for cnt in contours:
        # 获取轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 过滤掉太小的轮廓，这些可能是噪点而非数字
        # 宽度>10且高度>20的轮廓才被视为可能的数字
        if w > 10 and h > 20:
            # 提取感兴趣区域(ROI)
            roi = processed[y:y+h, x:x+w]
            
            # 将ROI调整为28x28大小，与MNIST训练数据一致
            resized = cv2.resize(roi, (28, 28))
            
            # 将像素值归一化到[0,1]范围，并转换为浮点型
            normalized = resized.astype(np.float32) / 255.0
            
            # 转换为PyTorch张量，并添加批次维度和通道维度
            # unsqueeze(0)添加批次维度，unsqueeze(0)添加通道维度
            # 最终形状为[1, 1, 28, 28]，表示1个样本，1个通道，28x28大小
            tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
            
            # 使用torch.no_grad()关闭梯度计算，提高推理速度和减少内存使用
            with torch.no_grad():
                # 将张量输入模型，获取预测结果
                output = model(tensor)
                
                # torch.max返回每行的最大值及其索引
                # prob是置信度（最大值），pred是预测的数字（索引）
                prob, pred = torch.max(output, 1)
            
            # 只保留置信度大于0.7的预测结果，过滤掉不确定的预测
            if prob > 0.4:  
                # 将结果添加到列表中：(x坐标, y坐标, 宽度, 高度, 预测的数字)
                results.append((x, y, w, h, pred.item()))
    
    # 按y坐标（从上到下）排序结果
    # 这对于多行数字的图像很有用，可以按行顺序输出结果
    results.sort(key=lambda x: x[1])
    
    # 返回识别结果列表和原始图像
    return results, original_img

def draw_results(img_path, save_path=None):
    """在图像上标注识别结果并保存
    
    将识别到的数字在原图上用矩形框标出，并在框上方或下方显示识别结果
    
    Args:
        img_path: 输入图像路径，要处理的原始图像
        save_path: 保存结果图像的路径，如果为None则自动生成
    
    Returns:
        str: 标注后的图像保存路径
    """
    # 调用recognize函数获取识别结果和原始图像
    results, img = recognize(img_path)
    
    # 在原图上绘制矩形框和标注识别结果
    for x, y, w, h, digit in results:
        # 绘制绿色矩形框(0,255,0)，线宽为2像素
        # 参数说明: 图像, 左上角坐标, 右下角坐标, 颜色(BGR), 线宽
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 设置文本内容和样式
        text = str(int(digit))  # 将预测结果转换为字符串
        font = cv2.FONT_HERSHEY_SIMPLEX  # 设置字体
        font_scale = 0.8  # 字体大小
        font_thickness = 2  # 字体粗细
        
        # 计算文本大小，用于确定文本位置
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        
        # 确定文本的x坐标（与矩形左上角对齐）
        text_x = x
        
        # 确定文本的y坐标
        # 如果矩形上方有足够空间，则将文本放在上方，否则放在下方
        text_y = y - 5 if y - 5 > text_size[1] else y + h + text_size[1] + 5
        
        # 绘制文本背景（绿色填充矩形）
        # 参数说明: 图像, 左上角坐标, 右下角坐标, 颜色(BGR), -1表示填充
        cv2.rectangle(img, (text_x, text_y - text_size[1]), 
                     (text_x + text_size[0], text_y), (0, 255, 0), -1)
        
        # 在背景上绘制黑色文本
        # 参数说明: 图像, 文本, 位置, 字体, 大小, 颜色, 粗细
        cv2.putText(img, text, (text_x, text_y), font, font_scale, 
                   (0, 0, 0), font_thickness)
    
    # 如果未指定保存路径，则根据原图路径自动生成
    if save_path is None:
        # 分离文件名和扩展名
        filename, ext = os.path.splitext(img_path)
        # 生成新的文件名，添加"_annotated"后缀
        save_path = f"{filename}_annotated{ext}"
    
    # 保存标注后的图像到指定路径
    cv2.imwrite(save_path, img)
    print(f"标注结果已保存至: {save_path}")
    
    # 返回保存的文件路径，便于后续使用
    return save_path

# 使用示例代码
# 注意：请确保图像文件存在后再运行以下代码
if __name__ == "__main__":
    """主函数：程序入口点
    
    展示了如何使用本模块进行手写数字识别的完整流程：
    1. 加载预训练模型
    2. 读取测试图像
    3. 识别图像中的数字
    4. 可视化识别结果
    5. 保存标注后的图像
    
    注意：运行前请确保以下条件满足：
    - mnist_cnn_model.pth 模型文件存在
    - data目录下有测试图像
    """
    print("模型加载成功！现在可以使用recognize函数识别手写数字。")
    print("示例用法：results = recognize('your_image_path.jpg')")
    print("本程序可以识别手写数字图像，并在原图上标注识别结果")
    # 示例1：处理第一张测试图像
    img_path = "data/homework_1.jpeg"
    print(f"\n处理图像: {img_path}")
    
    # 调用recognize函数获取识别结果
    results, _ = recognize(img_path)
    
    # 打印每个识别到的数字及其位置
    print("识别结果:")
    for x, y, w, h, digit in results:
        print(f"位置({x}, {y}) 识别结果：{digit}")
    
    # 在图像上标注识别结果并保存
    annotated_img_path = draw_results(img_path)
    print(f"标注图像已保存至: {annotated_img_path}")
    print(f"请查看标注后的图像并进行人工比对")
    
    # 示例2：处理第二张测试图像
    img_path = "data/homework_2.jpeg"
    print(f"\n处理图像: {img_path}")
    
    # 调用recognize函数获取识别结果
    results, _ = recognize(img_path)
    
    # 打印每个识别到的数字及其位置
    print("识别结果:")
    for x, y, w, h, digit in results:
        print(f"位置({x}, {y}) 识别结果：{digit}")
    
    # 在图像上标注识别结果并保存
    annotated_img_path = draw_results(img_path)
    print(f"标注图像已保存至: {annotated_img_path}")
    print(f"请查看标注后的图像并进行人工比对")
    
    # 示例3：处理第三张测试图像
    img_path = "data/homework_3.jpeg"
    print(f"\n处理图像: {img_path}")
    
    # 调用recognize函数获取识别结果
    results, _ = recognize(img_path)
    
    # 打印每个识别到的数字及其位置
    print("识别结果:")
    for x, y, w, h, digit in results:
        print(f"位置({x}, {y}) 识别结果：{digit}")
    
    # 在图像上标注识别结果并保存
    annotated_img_path = draw_results(img_path)
    print(f"标注图像已保存至: {annotated_img_path}")
    print(f"请查看标注后的图像并进行人工比对")
    
    
    # 示例4：处理第四张测试图像
    img_path = "data/homework_4.jpeg"
    print(f"\n处理图像: {img_path}")
    
    # 调用recognize函数获取识别结果
    results, _ = recognize(img_path)
    
    # 打印每个识别到的数字及其位置
    print("识别结果:")
    for x, y, w, h, digit in results:
        print(f"位置({x}, {y}) 识别结果：{digit}")
    
    # 在图像上标注识别结果并保存
    annotated_img_path = draw_results(img_path)
    print(f"标注图像已保存至: {annotated_img_path}")
    print(f"请查看标注后的图像并进行人工比对")
    
    # 示例5：处理第五张测试图像
    img_path = "data/homework_5.jpeg"
    print(f"\n处理图像: {img_path}")
    
    # 调用recognize函数获取识别结果
    results, _ = recognize(img_path)
    
    # 打印每个识别到的数字及其位置
    print("识别结果:")
    for x, y, w, h, digit in results:
        print(f"位置({x}, {y}) 识别结果：{digit}")
    
    # 在图像上标注识别结果并保存
    annotated_img_path = draw_results(img_path)
    print(f"标注图像已保存至: {annotated_img_path}")
    print(f"请查看标注后的图像并进行人工比对")
    
    # 示例6：处理第六张测试图像
    img_path = "data/homework_6.jpeg"
    print(f"\n处理图像: {img_path}")
    
    # 调用recognize函数获取识别结果
    results, _ = recognize(img_path)
    
    # 打印每个识别到的数字及其位置
    print("识别结果:")
    for x, y, w, h, digit in results:
        print(f"位置({x}, {y}) 识别结果：{digit}")
    
    # 在图像上标注识别结果并保存
    annotated_img_path = draw_results(img_path)
    print(f"标注图像已保存至: {annotated_img_path}")
    print(f"请查看标注后的图像并进行人工比对")
    
    # 示例7：处理第六张测试图像
    img_path = "data/handwriting_images/number_6.jpg"
    print(f"\n处理图像: {img_path}")
    
    # 调用recognize函数获取识别结果
    results, _ = recognize(img_path)
    
    # 打印每个识别到的数字及其位置
    print("识别结果:")
    for x, y, w, h, digit in results:
        print(f"位置({x}, {y}) 识别结果：{digit}")
    
    # 在图像上标注识别结果并保存
    annotated_img_path = draw_results(img_path)
    print(f"标注图像已保存至: {annotated_img_path}")
    print(f"请查看标注后的图像并进行人工比对")
    
    # 示例7：处理第六张测试图像
    img_path = "data/handwriting_images/number_9.jpg"
    print(f"\n处理图像: {img_path}")
    
    # 调用recognize函数获取识别结果
    results, _ = recognize(img_path)
    
    # 打印每个识别到的数字及其位置
    print("识别结果:")
    for x, y, w, h, digit in results:
        print(f"位置({x}, {y}) 识别结果：{digit}")
    
    # 在图像上标注识别结果并保存
    annotated_img_path = draw_results(img_path)
    print(f"标注图像已保存至: {annotated_img_path}")
    print(f"请查看标注后的图像并进行人工比对")
    
    # 示例7：处理第六张测试图像
    img_path = "data/handwriting_images/number_6_and_9.jpg"
    print(f"\n处理图像: {img_path}")
    
    # 调用recognize函数获取识别结果
    results, _ = recognize(img_path)
    
    # 打印每个识别到的数字及其位置
    print("识别结果:")
    for x, y, w, h, digit in results:
        print(f"位置({x}, {y}) 识别结果：{digit}")
    
    # 在图像上标注识别结果并保存
    annotated_img_path = draw_results(img_path)
    print(f"标注图像已保存至: {annotated_img_path}")
    print(f"请查看标注后的图像并进行人工比对")
    
    print("\n所有测试图像处理完成！")