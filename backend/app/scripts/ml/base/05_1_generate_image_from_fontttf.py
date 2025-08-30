from __future__ import print_function
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageEnhance
import os
import shutil
import time
import platform
import random
import numpy as np
from PIL import ImageFilter
from matplotlib.font_manager import FontProperties, findSystemFonts
 
## %% 要生成的文本
label_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '=', 11: '+', 12: '-', 13: '×', 14: '÷'}
# label_dict = {0: '0', 1: '1', 10: '='}
 
# 文本对应的文件夹，给每一个分类建一个文件
for value,char in label_dict.items():
    train_images_dir = "dataset"+"/"+str(value)
    # if os.path.isdir(train_images_dir):
    #     shutil.rmtree(train_images_dir)
    os.makedirs(train_images_dir, exist_ok=True)
 
##  %% 生成图片
def makeImage(label_dict, font_path, width=28, height=28, rotate=0, noise_level=0, apply_transforms=True, show_debug_info=False):
    """生成手写数字和符号图片
    
    参数:
        label_dict: 标签字典，键为类别编号，值为对应的字符
        font_path: 字体文件路径
        width: 图片宽度
        height: 图片高度
        rotate: 旋转角度
        noise_level: 噪声级别(0-15)
        apply_transforms: 是否应用随机变换
        show_debug_info: 是否显示调试信息（边框和辅助线）
    """
    # 从字典中取出键值对
    for value,char in label_dict.items():
        # 如果字体宽度或高度超过图片尺寸的80%，则按比例缩小字体大小
        # 降低到80%确保字符周围有更多边距，避免裁剪
        # max_width_ratio = 0.8
        max_width_ratio = 1 # 尽可能的填充，不要边距
        # max_height_ratio = 0.95
        max_height_ratio = 1 # 尽可能的填充，不要边距
        
        # 创建一个黑色背景的图片，大小是28*28，使用"L"模式创建灰度图
        img = Image.new("L", (width, height), 0)  # 230表示浅灰色背景，原来是0表示黑色
        draw = ImageDraw.Draw(img)
        
        # 动态调整字体大小，确保字体尽可能大且不超出图片边界
        # 初始字体大小设置为图片宽度的65%，避免过大导致字符被裁剪
        # 对于数字1这样的窄字符，可以使用更大的初始字体大小
        if char == '1':
            font_size = int(width * 1)  # 数字1可以稍小一些，避免太高
        else:
            font_size = int(width * 1)  # 其他字符使用65%的宽度作为初始大小
            
        font = ImageFont.truetype(font_path, font_size)
        
        # 使用textbbox获取更准确的文本边界框
        text_bbox = draw.textbbox((0, 0), char, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        
        if text_width < font_size * max_width_ratio:
            text_width = int(width * 1)  # 如果宽度小于最大宽度的80%，则使用原始宽度作为字体大小
        else:
            text_width = int(width * 1)  # 其他字符使用65%的宽度作为初始大小
        
        # 获取字体度量参数 - 更准确反映字体设计参数
        ascent, descent = font.getmetrics()
        text_height = ascent + descent  # 替代原text_bbox高度计算
        
        # 打印边界框和字体度量信息，帮助调试
        print(f"Character '{char}' bbox: {text_bbox}， width:{text_width}, ascent: {ascent}, descent: {descent} -> height:{text_height}")
        # 计算字体的基线位置（baseline）- 这对于垂直对齐很重要
        # 基线通常位于字体底部上方一点的位置
        
        # 对于数字1，可以使用更严格的高度限制
        # if char == '1':
        #     max_height_ratio = 0.75  # 数字1高度限制更严格
        
        while text_width > width * max_width_ratio:
            print(f"Adajust text width for text_width > width * max_width_ratio-> {text_width > width * max_width_ratio}  text_width:", text_width, "width:", width, "text_height:", text_height, "height:", height)
            font_size -= 1
            if font_size <= 8:  # 设置最小字体大小限制
                break
            font = ImageFont.truetype(font_path, font_size)
            text_bbox = draw.textbbox((0, 0), char, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            
        while text_height > height * max_height_ratio:
            print(f"Adajust text height for text_height > height * max_height_ratio -> {text_height > height * max_height_ratio} text_width:", text_width, "width:", width, "text_height:", text_height, "height:", height)
            font_size -= 1
            if font_size <= 8:  # 设置最小字体大小限制
                break
            font = ImageFont.truetype(font_path, font_size)
            # text_height = text_bbox[3] - text_bbox[1]
            ascent_t, descent_t = font.getmetrics()
            text_height = ascent_t + descent_t  # 替代原text_bbox高度计算
        
        print(f"final text width and height: {text_width}, {text_height}")
        
        # 计算字体绘制的x,y坐标，确保字符居中且完全显示在图片中
        x = (width - text_width) / 2
        
        # 对于特殊字符进行垂直位置调整
        # 使用混合定位策略，结合字体度量参数和基线补偿
        
        # 计算基线补偿（关键改进）
        # 基线补偿考虑了字体的上升部分和下降部分的不平衡
        baseline_compensation = (ascent - descent) * 0.15
        
        # 垂直位置计算新公式
        base_y = (height - text_height) / 2 - baseline_compensation
        
        # 确保坐标不会导致文本超出边界
        x = max(0, min(x, width - text_width))
        
        # 为底部预留更多边距，确保字符不会被裁剪
        bottom_margin = 2  # 底部预留2像素边距
        y = max(0, min(base_y, height - text_height - bottom_margin))
        
        # 打印调试信息
        print(f"Drawing '{char}' at ({x:.1f}, {y:.1f}) with size {font_size}, dimensions: {text_width}x{text_height}")
        
        # 绘制图片：位置，画啥，颜色，字体
        draw.text((x, y), char, 255, font)  # 255表示白色
        
        # 如果启用了调试信息，绘制字体边框和辅助线
        if show_debug_info:
            # 绘制文本边界框
            bbox_left = x + text_bbox[0]
            bbox_top = y + text_bbox[1]
            bbox_right = x + text_bbox[2]
            bbox_bottom = y + text_bbox[3]
            
            # 绘制边界框矩形
            draw.rectangle([bbox_left, bbox_top, bbox_right, bbox_bottom], outline=180)  # 灰色边框
            
            # 绘制图片中心十字线，帮助判断居中情况
            draw.line([(width//2, 0), (width//2, height)], fill=180, width=1)  # 垂直中心线
            draw.line([(0, height//2), (width, height//2)], fill=180, width=1)  # 水平中心线
            
            # 绘制字符基线位置的水平线
            baseline_y = y + ascent  # 使用ascent计算基线位置
            draw.line([(0, baseline_y), (width, baseline_y)], fill=150, width=1)  # 基线
            
            # 绘制字体度量信息
            ascent_line_y = y
            descent_line_y = y + text_height
            draw.line([(0, ascent_line_y), (width, ascent_line_y)], fill=120, width=1)  # 上升线
            draw.line([(0, descent_line_y), (width, descent_line_y)], fill=100, width=1)  # 下降线
        
        # 在绘制文本前应用倾斜变换
        # 创建一个临时图像用于绘制倾斜后的文本
        if rotate != 0:
            # 创建一个临时图像，用于绘制旋转后的文本
            temp_img = Image.new("L", (width, height), 0)  # 黑色背景
            temp_draw = ImageDraw.Draw(temp_img)
            
            # 在临时图像上绘制文本
            temp_draw.text((x, y), char, 255, font)  # 255表示白色
            
            # 只旋转文本内容，不旋转整个图像
            rotated_text = temp_img.rotate(rotate, fillcolor=0)  # 旋转文本，背景填充黑色
            
            # 将旋转后的文本合并到原始图像上
            img.paste(rotated_text, (0, 0), rotated_text)
        else:
            # 如果不需要旋转，直接在原图上绘制文本
            draw.text((x, y), char, 255, font)  # 255表示白色
        
        # 在处理图像前检查图像是否有内容（防止空白图像）
        extrema = img.getextrema()
        if extrema[1] > 0:  # 确保图像不是全黑的
            # 应用随机变换增强数据多样性
            if apply_transforms:
                # 随机位移（小范围）
                if random.random() < 0.3:  # 30%的概率应用位移
                    shift_x = random.randint(-2, 2)
                    shift_y = random.randint(-2, 2)
                    img = img.transform(
                        img.size, 
                        Image.AFFINE, 
                        (1, 0, shift_x, 0, 1, shift_y),
                        resample=Image.BICUBIC
                    )
                
                # 随机对比度调整
                if random.random() < 0.4:  # 40%的概率调整对比度
                    enhancer = ImageEnhance.Contrast(img)
                    factor = random.uniform(0.8, 1.2)  # 对比度因子
                    img = enhancer.enhance(factor)
                
                # 添加轻微的高斯模糊，使字体看起来更像手写
                if random.random() < 0.5:  # 50%的概率添加模糊
                    blur_radius = random.uniform(0.3, 0.7)
                    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # 添加随机噪点，增加数据多样性
            if noise_level > 0:
                img_array = np.array(img)
                noise = np.random.randint(0, noise_level, img_array.shape)
                img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(img_array)
            
            # 命名文件保存，命名规则：dataset/编号/img-编号_r-选择角度_时间戳.png
            time_value = int(round(time.time() * 1000))
            img_path = "dataset/{}/img-{}_r-{}_{}.png".format(value,value,rotate,time_value)
            img.save(img_path)
            
            # 打印成功信息
            print(f"Saved image for '{char}' with rotation {rotate}°")
        else:
            print(f"Warning: Empty image detected for '{char}' with rotation {rotate}°, skipping save")
        
## %% 获取系统字体
def get_system_fonts():
    """获取系统字体路径列表"""
    system_fonts = []
    try:
        # 使用matplotlib的findSystemFonts函数获取系统字体
        system_fonts = findSystemFonts()
        print(f"找到 {len(system_fonts)} 个系统字体")
        
        # 过滤出常用字体类型
        valid_fonts = [f for f in system_fonts if f.lower().endswith(('.ttf', '.otf'))]
        print(f"其中有 {len(valid_fonts)} 个有效字体(.ttf/.otf)")
        
        # 随机选择一部分字体，避免使用太多
        if len(valid_fonts) > 20:
            valid_fonts = random.sample(valid_fonts, 20)
            print(f"随机选择了 20 个系统字体")
            
        return valid_fonts
    except Exception as e:
        print(f"获取系统字体时出错: {e}")
        return []

## %% 生成图片主函数
# 执行图片生成
# generate_images()
        
## %%

import argparse

def main():
    """主函数，处理命令行参数并执行图片生成"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='生成手写数字和符号图片数据集')
    
    # 添加命令行参数
    parser.add_argument('--mode', type=str, default='mixed', choices=['standard', 'mixed', 'custom'],
                        help='字体选择模式: standard(仅使用系统默认字体), mixed(使用所有字体，默认), 或 custom(仅使用自定义字体)')
    parser.add_argument('--noise', type=int, default=1, choices=[0, 1],
                        help='是否应用随机噪声处理: 0(不应用) 或 1(应用，默认)')
    parser.add_argument('--debug', type=int, default=0, choices=[0, 1],
                        help='是否显示调试信息(边框和辅助线): 0(不显示，默认) 或 1(显示)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 根据参数执行图片生成
    generate_images(mode=args.mode, apply_noise=(args.noise == 1), show_debug_info=(args.debug == 1))
    
    # 根据不同模式显示完成信息
    mode_desc = {
        'standard': '仅使用系统字体',
        'mixed': '使用系统和自定义字体',
        'custom': '仅使用自定义字体'
    }
    print(f"\n图片生成完成! 模式: {args.mode} ({mode_desc.get(args.mode, '')}), 噪声处理: {'开启' if args.noise == 1 else '关闭'}, 调试信息: {'显示' if args.debug == 1 else '隐藏'}")

# 修改generate_images函数，添加参数
def generate_images(mode='mixed', apply_noise=True, show_debug_info=False):
    """生成图片主函数
    
    参数:
        mode: 字体选择模式，'standard'仅使用系统默认字体，'mixed'使用所有字体，'custom'仅使用自定义字体
        apply_noise: 是否应用随机噪声处理
        show_debug_info: 是否显示调试信息（边框和辅助线）
    """
    # 1. 使用自定义字体目录中的字体（在mixed或custom模式下）
    # 使用脚本所在目录的相对路径，确保能正确找到字体目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    font_dir = os.path.join(script_dir, "data/fonts")
    custom_fonts = []
    
    if (mode == 'mixed' or mode == 'custom') and os.path.exists(font_dir):
        for font_name in os.listdir(font_dir):
            # 把每种字体都取出来，每种字体都生成一批图片
            path_font_file = os.path.join(font_dir, font_name)
            if os.path.isfile(path_font_file) and (path_font_file.lower().endswith('.ttf') or path_font_file.lower().endswith('.otf')):
                custom_fonts.append(path_font_file)
    
        print(f"找到 {len(custom_fonts)} 个自定义字体")
    elif mode == 'standard':
        print("标准模式: 仅使用系统字体")
    
    # 2. 获取系统字体（在standard或mixed模式下）
    system_fonts = [] if mode == 'custom' else get_system_fonts()
    
    # 3. 合并所有字体
    if mode == 'mixed':
        all_fonts = custom_fonts + system_fonts
    elif mode == 'custom':
        all_fonts = custom_fonts
        if not all_fonts:
            print("警告：未找到自定义字体，请确保 'data/fonts' 目录存在并包含字体文件")
            return
    else:  # standard模式
        all_fonts = system_fonts
    
    print(f"总共将使用 {len(all_fonts)} 个字体生成图片")
    
    # 4. 为每种字体生成图片
    for font_path in all_fonts:
        try:
            font_name = os.path.basename(font_path)
            print(f"正在使用字体: {font_name}")
            
            # 对系统字体和自定义字体使用不同的参数
            is_system_font = font_path not in custom_fonts
            
            # 系统字体生成更多变化
            if is_system_font:
                print(f"系统字体: {font_name} - 应用更多随机变换")
                # 倾斜角度范围更大，步长更小
                angle_range = range(-12, 12, 2)
                # 更高概率应用变换
                apply_transforms = True
            else:
                print(f"自定义字体: {font_name} - 应用标准变换")
                # 标准倾斜角度范围
                angle_range = range(-10, 10, 2)
                # 标准概率应用变换
                apply_transforms = random.random() > 0.2  # 80%概率应用变换
            
            # 为每个角度生成图片
            for k in angle_range:
                # 随机噪声级别 (根据apply_noise参数决定是否应用)
                noise_level = random.randint(0, 15) if apply_noise else 0
                # 每个字符都生成图片
                makeImage(label_dict, font_path, rotate=k, noise_level=noise_level, apply_transforms=apply_transforms, show_debug_info=show_debug_info)
        except Exception as e:
            print(f"使用字体 {font_path} 生成图片时出错: {e}")
            continue

# 如果直接运行此脚本，则执行main函数
if __name__ == "__main__":
    main()
    # python 05_1_generate_image_from_fontttf.py --mode custom --noise 0 # 自定义字体、不要噪声处理