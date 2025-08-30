import cv2
import numpy as np
import argparse
import os

def apply_perspective_transform(img, strength=0.3):
    """
    对图片应用透视变换，创造向后倒的3D效果
    
    参数:
        img: 输入图片
        strength: 透视强度，值越大效果越明显
    """
    height, width = img.shape[:2]
    
    # 定义源点和目标点
    # 源点是图像的四个角
    src_points = np.float32([
        [0, 0],               # 左上
        [width - 1, 0],       # 右上
        [0, height - 1],      # 左下
        [width - 1, height - 1] # 右下
    ])
    
    # 目标点：顶部向后倾斜，底部保持不变，创造向后倒的3D效果
    # 计算顶部Y轴的偏移量，使图片顶部向后倾斜
    y_offset_top = int(height * strength * 0.8)  # 顶部Y轴偏移量（正值使其向后倾斜）
    
    # 同时添加一些水平方向的收缩，增强3D效果
    x_offset = int(width * strength * 0.3)  # 水平方向的收缩量
    
    # 调整四个角点的位置，创造更真实的向后倒的3D效果
    # 顶部向后倾斜（Y值增大），底部保持不变，增强3D效果
    dst_points = np.float32([
        [x_offset, y_offset_top],                # 左上（Y值增大，向下移动）
        [width - 1 - x_offset, y_offset_top],    # 右上（Y值增大，向下移动）
        [0, height - 1],                         # 左下（保持原位置）
        [width - 1, height - 1]                  # 右下（保持原位置）
    ])
    
    # 计算透视变换矩阵
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # 应用透视变换，确保使用白色背景
    # 增加输出图像的高度，以容纳向后倒的效果
    # 由于是向后倒，需要更大的高度来容纳下方延伸的部分
    output_height = int(height * 1.5)
    # 调整输出图像的起始位置，使图像主体保持在视野中
    output_matrix = perspective_matrix.copy()
    # 向上移动图像，确保图像主体在视野中
    output_matrix[1, 2] -= y_offset_top // 2  # 向上移动图像，使图像主体居中
    result = cv2.warpPerspective(img, output_matrix, (width, output_height), borderValue=(255, 255, 255))
    
    return result

def skew_and_rotate_image(input_path, output_path, skew_factor=0.3, rotation_angle=30, bg_color=(255, 255, 255), perspective_strength=0.0):
    """
    对图片进行斜切、透视和旋转处理
    
    参数:
        input_path: 输入图片路径
        output_path: 输出图片路径
        skew_factor: 斜切因子，值越大斜切效果越明显
        rotation_angle: 旋转角度（度）
        bg_color: 背景颜色，RGB格式的元组
        perspective_strength: 透视变换强度，值越大效果越明显
    """
    try:
        # 读取图片
        img = cv2.imread(input_path)
        if img is None:
            print(f"错误：无法读取图片 {input_path}")
            return False
            
        # 获取图片尺寸
        height, width = img.shape[:2]
        
        # 1. 创建白色背景画布
        # 计算斜切后的图片尺寸
        skew_width = int(width + height * abs(skew_factor))
        
        # 创建一个白色背景的画布（或指定的背景颜色）
        canvas = np.ones((height, skew_width, 3), dtype=np.uint8)
        canvas[:] = bg_color  # 设置背景颜色
        
        # 将原图放在画布上
        canvas[0:height, 0:width] = img
        
        # 创建斜切变换矩阵
        skew_matrix = np.float32([
            [1, skew_factor, 0],
            [0, 1, 0]
        ])
        
        # 应用斜切变换
        skewed_img = cv2.warpAffine(canvas, skew_matrix, (skew_width, height), borderValue=bg_color)
        
        # 应用透视变换（如果需要）- 现在在斜切之后应用
        if perspective_strength > 0:
            skewed_img = apply_perspective_transform(skewed_img, perspective_strength)
            # 重新获取尺寸，因为透视变换可能改变了图片尺寸
            skew_height, skew_width = skewed_img.shape[:2]
        
        # 2. 进行旋转变换
        # 计算旋转中心点
        center = (skew_width // 2, height // 2)
        
        # 创建旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        
        # 计算旋转后的图片尺寸
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        
        new_width = int((height * sin) + (skew_width * cos))
        new_height = int((height * cos) + (skew_width * sin))
        
        # 调整旋转矩阵以考虑新的尺寸
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # 应用旋转变换
        result_img = cv2.warpAffine(skewed_img, rotation_matrix, (new_width, new_height), borderValue=bg_color)
        
        # 保存结果图片
        cv2.imwrite(output_path, result_img)
        print(f"处理完成：图片已保存至 {output_path}")
        return True
        
    except Exception as e:
        print(f"处理图片时发生错误: {str(e)}")
        return False

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='对图片进行斜切、旋转和透视变换处理')
    parser.add_argument('-i', '--input', required=True, help='输入图片路径')
    parser.add_argument('-o', '--output', required=True, help='输出图片路径')
    parser.add_argument('-s', '--skew', type=float, default=0.7, help='斜切因子 (默认: 0.7)')
    parser.add_argument('-r', '--rotate', type=float, default=-30, help='旋转角度 (默认: -30度)')
    parser.add_argument('-p', '--perspective', type=float, default=0.0, help='透视变换强度 (默认: 0.0，不应用透视变换)')
    parser.add_argument('-b', '--background', type=str, default='255,255,255', help='背景颜色，RGB格式，用逗号分隔 (默认: 255,255,255 白色)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.isfile(args.input):
        print(f"错误：输入文件 {args.input} 不存在")
        return
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 解析背景颜色
    try:
        bg_color = tuple(map(int, args.background.split(',')))
        if len(bg_color) != 3 or not all(0 <= c <= 255 for c in bg_color):
            print(f"警告：背景颜色格式错误，使用默认白色背景")
            bg_color = (255, 255, 255)
    except:
        print(f"警告：背景颜色格式错误，使用默认白色背景")
        bg_color = (255, 255, 255)
    
    # 处理图片
    skew_and_rotate_image(args.input, args.output, args.skew, args.rotate, bg_color, args.perspective)

if __name__ == "__main__":
    main()