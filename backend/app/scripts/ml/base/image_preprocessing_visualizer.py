#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
图像预处理可视化脚本
本脚本用于可视化图像预处理的效果，包括以下步骤：
1. 灰度转换
2. 调整大小
3. 自适应二值化
4. 形态学操作
5. 直方图均衡化
6. 边缘保留滤波

使用方法：
python image_preprocessing_visualizer.py --image_path 图片路径
'''

# 导入必要的库
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
import argparse
from datetime import datetime
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'Songti SC', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建中文字体属性对象，用于单独设置某些元素的字体
chineseFont = FontProperties(family='Arial Unicode MS')

def preprocess_image(image_path):
    '''
    图像预处理函数
    参数:
        image_path: 图像路径
    返回:
        原始图像和处理后的图像
    '''
    logger.info(f"开始预处理图像: {image_path}")
    # 读取图像
    img = cv2.imread(image_path)
    
    # 检查图像是否成功加载
    if img is None:
        logger.error(f"无法加载图像: {image_path}")
        raise ValueError(f"无法加载图像: {image_path}")
    
    # 如果是彩色图像，转换为灰度图
    if len(img.shape) == 3 and img.shape[2] == 3:
        logger.debug("图像为彩色，转换为灰度图")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        logger.debug("图像已经是灰度图")
        gray_img = img.copy()
    
    # 保存原始灰度图像用于显示
    original_img = gray_img.copy()
    
    # 调整大小
    logger.debug("调整图像大小至 24x24")
    resized_img = cv2.resize(gray_img, (24, 24))
    
    # 1. 高斯模糊去噪
    logger.debug("应用高斯模糊去噪")
    blurred_img = cv2.GaussianBlur(resized_img, (5, 5), 0)
    
    # 2. 自适应二值化
    logger.debug("应用自适应二值化")
    binary_img = cv2.adaptiveThreshold(
        blurred_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 15  # 调整参数以更好地处理对比度
    )
    
    # 3. 形态学操作（闭运算去除小噪声点）
    logger.debug("应用形态学操作（闭运算）")
    kernel = np.ones((3,3), np.uint8)
    morphology_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    
    # 4. 直方图均衡化
    logger.debug("应用直方图均衡化")
    equalized_img = cv2.equalizeHist(morphology_img)
    
    # 5. 边缘保留滤波
    logger.debug("应用边缘保留滤波")
    filtered_img = cv2.bilateralFilter(equalized_img, 5, 75, 75)
    
    logger.info("图像预处理完成")
    return original_img, filtered_img

def get_filename_without_extension(file_path):
    '''
    从文件路径中提取文件名（不含扩展名）
    参数:
        file_path: 文件路径
    返回:
        文件名（不含扩展名）
    '''
    # 获取基本文件名（含扩展名）
    base_name = os.path.basename(file_path)
    # 分离文件名和扩展名
    file_name, _ = os.path.splitext(base_name)
    return file_name

def visualize_preprocessing(original_img, processed_img, image_path, output_dir=None):
    '''
    可视化预处理效果
    参数:
        original_img: 原始图像
        processed_img: 处理后的图像
        image_path: 原始图像路径，用于提取文件名
        output_dir: 输出目录，如果为None则保存在当前目录
    '''
    # 创建图像
    plt.figure(figsize=(12, 5))
    
    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('原始图像', fontproperties=chineseFont)
    plt.axis('off')
    
    # 显示处理后的图像
    plt.subplot(1, 2, 2)
    plt.imshow(processed_img, cmap='gray')
    plt.title('处理后的图像 (24x24)', fontproperties=chineseFont)
    plt.axis('off')
    
    plt.tight_layout()
    
    # 添加时间戳避免覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 获取原始文件名
    original_filename = get_filename_without_extension(image_path)
    
    # 确定输出路径
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{original_filename}_preprocessing_steps_{timestamp}.png')
    else:
        output_path = f'{original_filename}_preprocessing_steps_{timestamp}.png'
    
    # 保存图像
    plt.savefig(output_path)
    plt.close()
    
    print(f"预处理可视化已保存到: {output_path}")

def visualize_preprocessing_steps(image_path, output_dir=None):
    '''
    可视化图像预处理的各个步骤
    参数:
        image_path: 图像路径
        output_dir: 输出目录，如果为None则保存在当前目录
    '''
    logger.info(f"开始可视化图像预处理的各个步骤: {image_path}")
    # 读取图像
    img = cv2.imread(image_path)
    
    # 检查图像是否成功加载
    if img is None:
        logger.error(f"无法加载图像: {image_path}")
        raise ValueError(f"无法加载图像: {image_path}")
    
    # 如果是彩色图像，转换为灰度图
    if len(img.shape) == 3 and img.shape[2] == 3:
        logger.debug("图像为彩色，转换为灰度图")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        logger.debug("图像已经是灰度图")
        gray_img = img.copy()
    
    # 创建图像
    plt.figure(figsize=(15, 10))
    
    # 1. 显示原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(gray_img, cmap='gray')
    plt.title('1. 原始灰度图像', fontproperties=chineseFont)
    plt.axis('off')
    
    # 2. 调整大小
    logger.debug("调整图像大小至 24x24")
    resized_img = cv2.resize(gray_img, (24, 24))
    plt.subplot(2, 3, 2)
    plt.imshow(resized_img, cmap='gray')
    plt.title('2. 调整大小 (24x24)', fontproperties=chineseFont)
    plt.axis('off')
    
    # 3. 自适应二值化
    logger.debug("应用自适应二值化")
    binary_img = cv2.adaptiveThreshold(
        resized_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        9, 10
    )
    plt.subplot(2, 3, 3)
    plt.imshow(binary_img, cmap='gray')
    plt.title('3. 自适应二值化', fontproperties=chineseFont)
    plt.axis('off')
    
    # 4. 形态学操作
    logger.debug("应用形态学操作（开运算）")
    kernel = np.ones((2,2), np.uint8)
    morphology_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    plt.subplot(2, 3, 4)
    plt.imshow(morphology_img, cmap='gray')
    plt.title('4. 形态学操作', fontproperties=chineseFont)
    plt.axis('off')
    
    # 5. 直方图均衡化
    logger.debug("应用直方图均衡化")
    equalized_img = cv2.equalizeHist(morphology_img)
    plt.subplot(2, 3, 5)
    plt.imshow(equalized_img, cmap='gray')
    plt.title('5. 直方图均衡化', fontproperties=chineseFont)
    plt.axis('off')
    
    # 6. 边缘保留滤波
    logger.debug("应用边缘保留滤波")
    filtered_img = cv2.bilateralFilter(equalized_img, 5, 75, 75)
    plt.subplot(2, 3, 6)
    plt.imshow(filtered_img, cmap='gray')
    plt.title('6. 边缘保留滤波', fontproperties=chineseFont)
    plt.axis('off')
    
    plt.tight_layout()
    
    # 添加时间戳避免覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 获取原始文件名
    original_filename = get_filename_without_extension(image_path)
    
    # 确定输出路径
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{original_filename}_preprocessing_detailed_steps_{timestamp}.png')
    else:
        output_path = f'{original_filename}_preprocessing_detailed_steps_{timestamp}.png'
    
    # 保存图像
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"详细预处理步骤可视化已保存到: {output_path}")

def visualize_preprocessing_steps_v2(image_path, output_dir=None):
    '''
    可视化图像预处理的各个步骤
    参数:
        image_path: 图像路径
        output_dir: 输出目录，如果为None则保存在当前目录
    '''
    # 读取图像
    img = cv2.imread(image_path)
    
    # 检查图像是否成功加载
    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")
    
    # 如果是彩色图像，转换为灰度图
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img.copy()
    
    # 创建图像
    plt.figure(figsize=(15, 10))
    
    # 1. 显示原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(gray_img, cmap='gray')
    plt.title('1. 原始灰度图像', fontproperties=chineseFont)
    plt.axis('off')
    
    # 2. 调整大小
    resized_img = cv2.resize(gray_img, (24, 24))
    plt.subplot(2, 3, 2)
    plt.imshow(resized_img, cmap='gray')
    plt.title('2. 调整大小 (24x24)', fontproperties=chineseFont)
    plt.axis('off')
    
    # 3. 高斯模糊去噪
    blurred_img = cv2.GaussianBlur(resized_img, (5, 5), 0)
    plt.subplot(2, 3, 3)
    plt.imshow(blurred_img, cmap='gray')
    plt.title('3. 高斯模糊去噪', fontproperties=chineseFont)
    plt.axis('off')
    
    # 4. 自适应二值化
    binary_img = cv2.adaptiveThreshold(
        blurred_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 15
    )
    plt.subplot(2, 3, 4)
    plt.imshow(binary_img, cmap='gray')
    plt.title('4. 自适应二值化', fontproperties=chineseFont)
    plt.axis('off')
    
    # 5. 形态学操作（闭运算去除小噪声点）
    kernel = np.ones((3,3), np.uint8)
    morphology_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    plt.subplot(2, 3, 5)
    plt.imshow(morphology_img, cmap='gray')
    plt.title('5. 形态学操作', fontproperties=chineseFont)
    plt.axis('off')
    
    plt.tight_layout()
    
    # 添加时间戳避免覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 获取原始文件名
    original_filename = get_filename_without_extension(image_path)
    
    # 确定输出路径
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{original_filename}_preprocessing_detailed_steps_{timestamp}.png')
    else:
        output_path = f'{original_filename}_preprocessing_detailed_steps_{timestamp}.png'
    
    # 保存图像
    plt.savefig(output_path)
    plt.close()
    
    print(f"详细预处理步骤可视化已保存到: {output_path}")

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='图像预处理可视化工具')
    parser.add_argument('--image_path', type=str, required=True, help='输入图像的路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录，默认为当前目录')
    parser.add_argument('--detailed', action='store_true', help='是否显示详细的预处理步骤')
    parser.add_argument('--detailed2', action='store_true', help='是否显示详细的预处理步骤')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    try:
        if args.detailed2:
            # 显示详细的预处理步骤
            visualize_preprocessing_steps_v2(args.image_path, args.output_dir)
        elif args.detailed:
            # 显示详细的预处理步骤
            visualize_preprocessing_steps(args.image_path, args.output_dir)
        else:
            # 只显示原始图像和最终处理后的图像
            original_img, processed_img = preprocess_image(args.image_path)
            visualize_preprocessing(original_img, processed_img, args.image_path, args.output_dir)
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()