import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import json
from datetime import datetime
import argparse  # 新增：引入argparse模块

# 设置Matplotlib字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'Songti SC', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建Reader对象，指定语言
reader = easyocr.Reader(['ch_sim', 'en'])

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    metadata = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            result = reader.readtext(image_path)
            
            # 处理图像并保存标注结果
            output_image_path = os.path.join(output_dir, f"annotated_{filename}")
            image_metadata = annotate_image(image_path, output_image_path, result)
            image_metadata['filename'] = filename
            metadata.append(image_metadata)
    
    # 保存元数据为JSON文件
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    
    # 生成MD技术文档
    generate_md_document(metadata, output_dir)

def annotate_image(image_path, output_image_path, result):
    img = cv2.imread(image_path)
    spacer = 100
    image_metadata = {'detections': []}
    
    # 使用PIL库处理中文文本
    from PIL import Image, ImageDraw, ImageFont
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    # 使用系统默认字体，确保支持中文
    try:
        font = ImageFont.load_default()
    except:
        font = ImageFont.load_default(20)  # 如果默认字体加载失败，尝试设置大小

    # 使用宋体，使用上面的内容还是无法处理中文的        
    font = ImageFont.truetype("SimSun.ttf", 20)
    
    for detection in result: 
        top_left = tuple(map(int, detection[0][0]))  # 确保坐标为整数
        bottom_right = tuple(map(int, detection[0][2]))  # 确保坐标为整数
        text = detection[1]
        prob = detection[2]
        
        # 打印 top_left 和 bottom_right 的值进行排查
        print(f"top_left: {image_path} {top_left}, bottom_right: {bottom_right}")
        
        # 确保 bottom_right 的 y 坐标大于 top_left 的 y 坐标（TODO 出现这种异常情况：top_left: (451, 96), bottom_right: (641, 14)）
        if bottom_right[1] < top_left[1]:
            bottom_right = (bottom_right[0], top_left[1] + 1)
        
        # 使用PIL绘制矩形框
        draw.rectangle([top_left, bottom_right], outline=(0, 255, 0), width=3)
        
        # 使用PIL绘制中文文本
        draw.text((20, spacer), f'{text} ({prob:.2f})', font=font, fill=(0, 255, 0))
        spacer += 15
        
        # 记录元数据
        image_metadata['detections'].append({
            'text': text,
            'coordinates': [list(map(int, top_left)), list(map(int, bottom_right))],
            'confidence': prob
        })
    
    # 将PIL图像转换回OpenCV格式并保存
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path, img)
    return image_metadata

def generate_md_document(metadata, output_dir):
    md_path = os.path.join(output_dir, 'results.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# OCR 识别结果报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 概述\n")
        f.write("本报告展示了使用 EasyOCR 对图像进行文本识别的结果。每张图像的识别结果包括检测到的文本、其坐标位置以及置信度。\n\n")
        
        for item in metadata:
            f.write(f"## 文件: {item['filename']}\n")
            f.write(f"![{item['filename']}]({item['filename']})\n\n")  # 插入图片
            f.write("### 识别结果\n")
            f.write("| 文本 | 坐标 | 置信度 |\n")
            f.write("|------|------|--------|\n")
            for detection in item['detections']:
                f.write(f"| {detection['text']} | {detection['coordinates']} | {detection['confidence']:.2f} |\n")
            f.write("\n")
            f.write("### 说明\n")
            f.write("1. **文本**: 检测到的文本内容。\n")
            f.write("2. **坐标**: 文本在图像中的位置，以左上角和右下角的坐标表示。\n")
            f.write("3. **置信度**: 识别结果的置信度，范围从 0 到 1，值越高表示识别结果越可靠。\n\n")
        
        f.write("## 总结\n")
        f.write("通过 EasyOCR 的识别，我们能够从图像中提取出文本信息，并生成详细的报告。每张图像的识别结果都包含了文本的位置和置信度，便于进一步分析和处理。\n")

# 主程序
if __name__ == "__main__":
    # 新增：解析命令行参数
    parser = argparse.ArgumentParser(description="使用EasyOCR进行图像文本识别")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图像目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出结果目录")
    args = parser.parse_args()

    # 调用处理函数
    process_images(args.input_dir, args.output_dir)