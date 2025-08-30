import sys
from pathlib import Path
from unittest import result
import argparse  # 导入命令行参数处理模块
# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).resolve().parents[4]))

import cv2  # 导入OpenCV库，用于图像处理
import numpy as np  # 导入NumPy库，用于数值计算
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于图像可视化
import os  # 导入操作系统模块，用于文件和目录操作
from datetime import datetime  # 导入日期时间模块，用于生成时间戳
import logging  # 导入日志模块，用于记录程序运行信息
import torch  # 导入PyTorch库，用于深度学习模型
from PIL import Image  # 导入PIL库，用于图像处理
import torchvision.transforms as transforms  # 导入图像变换模块
import matplotlib  # 导入Matplotlib库
import json  # 导入JSON模块，用于处理JSON数据

# 导入自定义手写识别模型
from a_05_2_custom_handwriting_model_v2 import predict_image_v1

# 配置日志系统
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'Songti SC', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class DataObject:
    '''
    数据对象类，用于存储识别结果的内容、坐标和置信度
    '''
    def __init__(self, content, coordinates, confidence):
        '''
        初始化数据对象
        参数:
            content: 识别内容
            coordinates: 坐标元组(y_start, y_end, x_start, x_end)
            confidence: 置信度值
        '''
        self.content = str(content)  # 将识别内容转换为字符串格式
        self.coordinates = tuple(coordinates)  # 将坐标转换为元组格式(y_start, y_end, x_start, x_end)
        self.confidence = float(confidence)  # 将置信度值标准化为浮点数

    def to_dict(self):
        '''
        将数据对象转换为字典格式
        返回:
            包含内容、坐标、置信度和位置信息的字典
        '''
        return {
            'content': self.content,  # 识别内容
            'coordinates': self.coordinates,  # 原始坐标元组
            'confidence': self.confidence,  # 置信度值
            'position': {  # 位置信息
                'row': self.coordinates[0],  # 行位置（y起始坐标）
                'col': self.coordinates[2]   # 列位置（x起始坐标）
            }
        }

    def __repr__(self):
        '''
        对象的字符串表示
        返回:
            对象的格式化字符串表示
        '''
        return f"DataObject(content='{self.content}', confidence={self.confidence:.2f}, coords={self.coordinates})"

class ImageSegmenter:
    '''
    图像分割类，专门处理小学生手写的口算题图像分割。
    通过二值化、投影和切割等技术逐个提取数字和符号。
    '''

    def __init__(self, image_path):
        '''
        初始化图像分割器
        参数:
            image_path: 输入图像的路径
        '''
        self.image_path = image_path  # 存储图像路径
        self.gray_img = None  # 存储灰度图像
        self.binary_img = None  # 存储二值化图像

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

        print(f"水平投影计算完成，投影数组长度: {len(horizontal_projection)}")
        return horizontal_projection

    def vertical_projection(self, binary_img=None):
        '''
        计算图像的垂直投影
        参数:
            binary_img: 二值化图像，默认为类属性中的二值化图像
        返回:
            垂直投影数组
        示例:
            假设输入的二值化图像为：
            [[0, 255, 0],
             [255, 0, 255],
             [0, 255, 0]]
            则垂直投影结果为：[1, 2, 1]
        '''
        if binary_img is None:
            binary_img = self.binary_img

        h, w = binary_img.shape
        vertical_projection = np.zeros(w)

        for j in range(w):
            vertical_projection[j] = np.count_nonzero(binary_img[:, j])

        print(f"垂直投影计算完成，投影数组长度: {len(vertical_projection)}")
        return vertical_projection

    def cut_by_projection(self, projection, axis='horizontal', input_img=None):
        '''
        根据投影结果切割图像，并记录每个子图像的坐标
        参数:
            projection: 投影数组，表示图像在某个方向上的像素分布
            axis: 切割方向，'horizontal' 表示水平切割，'vertical' 表示垂直切割
            input_img: 输入图像，默认为类属性中的二值化图像
        返回:
            切割后的子图像列表及其坐标，每个子图像与其坐标以元组形式返回
        示例:
            假设投影数组为：[0, 2, 0, 3, 0]，axis='horizontal'，输入图像为 5x5 的二值化图像，
            则切割结果为两个子图像，分别对应第1行到第2行和第3行到第4行。
        '''
        sub_images = []  # 存储切割后的子图像
        coordinates = []  # 存储每个子图像的坐标
        start = 0  # 记录当前切割区域的起始位置
        in_object = False  # 标记是否处于一个切割区域内

        if input_img is None:
            input_img = self.binary_img

        print(f"开始根据{axis}投影切割图像")

        # 遍历投影数组，寻找切割点
        for i, value in enumerate(projection):
            if not in_object and value > 0:  # 进入一个切割区域
                start = i
                in_object = True
            elif in_object and value == 0:  # 离开一个切割区域
                if axis == 'horizontal':
                    # 水平切割，提取从 start 到 i 的行
                    sub_images.append(input_img[start:i, :])
                    # 记录坐标：(起始行, 结束行, 起始列, 结束列)
                    coordinates.append((start, i, 0, input_img.shape[1]))
                    print(f"水平切割：从 {start} 到 {i}")
                elif axis == 'vertical':
                    # 垂直切割，提取从 start 到 i 的列
                    sub_images.append(input_img[:, start:i])
                    # 记录坐标：(起始行, 结束行, 起始列, 结束列)
                    coordinates.append((0, input_img.shape[0], start, i))
                    print(f"垂直切割：从 {start} 到 {i}")
                in_object = False

        # 处理最后一个切割区域
        if in_object:
            if axis == 'horizontal':
                sub_images.append(input_img[start:, :])
                coordinates.append((start, input_img.shape[0], 0, input_img.shape[1]))
                print(f"水平切割：从 {start} 到末尾")
            elif axis == 'vertical':
                sub_images.append(input_img[:, start:])
                coordinates.append((0, input_img.shape[0], start, input_img.shape[1]))
                print(f"垂直切割：从 {start} 到末尾")

        print(f"{axis}切割完成，共得到 {len(sub_images)} 个子图像")
        return list(zip(sub_images, coordinates))  # 返回子图像及其坐标的列表

    def segment_image_v1(self):
        '''
        分割图像中的数字和符号，并记录每个分割块的坐标
        返回:
            分割后的子图像列表及其坐标，按照行列和等式分组形式组织
        '''
        logger.info("开始图像分割流程")
        self.preprocess_image()
        horizontal_projection = self.horizontal_projection()
        rows = self.cut_by_projection(horizontal_projection, axis='horizontal')

        all_segments = []

        for row_idx, (row, row_coords) in enumerate(rows):
            logger.info(f"处理第 {row_idx + 1} 行，坐标: {row_coords}")
            # 计算垂直投影并切割
            vertical_projection = self.vertical_projection(row)
            segments = self.cut_by_projection(vertical_projection, axis='vertical', input_img=row)
            adjusted_segments = [(img, (row_coords[0], row_coords[1], coords[2], coords[3])) for img, coords in segments]
            logger.info(f"第 {row_idx + 1} 行垂直切割完成，得到 {len(adjusted_segments)} 个分割块")

            # 分组等式
            grouped_segments = []
            current_group = []
            for segment_idx, segment in enumerate(adjusted_segments):
                img, coords = segment
                # 判断是否为等号
                is_equal_sign = self.is_equal_sign(img)
                logger.info(f"第 {row_idx + 1} 行第 {segment_idx + 1} 个分割块，是否为等号: {is_equal_sign}")
                if is_equal_sign:
                    # 将当前组添加到分组中
                    if current_group:
                        logger.info(f"发现等号，将当前组添加到分组中，当前组大小: {len(current_group)}")
                        grouped_segments.append(current_group)
                        current_group = []
                else:
                    current_group.append((img, coords))  # 确保每个 segment 都是 (img, coords) 元组
            # 添加最后一个分组
            if current_group:
                logger.info(f"添加最后一个分组，大小: {len(current_group)}")
                grouped_segments.append(current_group)

            all_segments.append(grouped_segments)
            logger.info(f"第 {row_idx + 1} 行分组完成，共 {len(grouped_segments)} 组")

        logger.info(f"图像分割完成，共得到 {len(all_segments)} 行分割结果")
        return all_segments

    def segment_image_v2(self, output_dir=None):
        '''
        分割图像中的数字和符号，并记录每个分割块的坐标
        参数:
            output_dir: 输出目录路径，如果为None则使用默认目录
        返回:
            分割后的子图像列表及其坐标，按照行列和等式分组形式组织
        '''
        logger.info("开始图像分割流程")  # 记录开始分割的日志
        self.preprocess_image()  # 预处理图像（灰度转换和二值化）
        horizontal_projection = self.horizontal_projection()  # 计算水平投影
        rows = self.cut_by_projection(horizontal_projection, axis='horizontal')  # 根据水平投影切割行

        all_segments = []  # 初始化分割结果列表

        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
        original_filename = get_filename_without_extension(self.image_path)  # 获取不含扩展名的文件名
        if output_dir is None:  # 如果未提供输出目录
            # 创建默认输出目录
            output_dir = os.path.join(os.getcwd(), f'{original_filename}_segments_v2_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

        for row_idx, (row, row_coords) in enumerate(rows):
            logger.info(f"处理第 {row_idx + 1} 行，坐标: {row_coords}")
            # 计算垂直投影并切割
            vertical_projection = self.vertical_projection(row)
            segments = self.cut_by_projection(vertical_projection, axis='vertical', input_img=row)
            adjusted_segments = [(img, (row_coords[0], row_coords[1], coords[2], coords[3])) for img, coords in segments]
            logger.info(f"第 {row_idx + 1} 行垂直切割完成，得到 {len(adjusted_segments)} 个分割块")

            # 使用图像膨胀的方法来分组
            grouped_segments = []
            kernel = np.ones((5, 5), np.uint8)  # 膨胀核大小
            row_img_b = cv2.dilate(row, kernel, iterations=9)  # 图像膨胀6次

            # 可视化膨胀前后的图像并保存
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(row, cmap='gray')
            plt.title('膨胀前图像')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(row_img_b, cmap='gray')
            plt.title('膨胀后图像')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f'row_{row_idx + 1}_dilation.png'))
            plt.close()

            # 计算膨胀后的垂直投影
            vertical_projection_dilated = self.vertical_projection(row_img_b)
            dilated_segments = self.cut_by_projection(vertical_projection_dilated, axis='vertical', input_img=row_img_b)

            # 可视化膨胀后的切割结果并保存
            plt.figure(figsize=(10, 5))
            for i, (segment, coords) in enumerate(dilated_segments):
                plt.subplot(1, len(dilated_segments), i + 1)
                plt.imshow(segment, cmap='gray')
                plt.title(f'膨胀后切割块 {i + 1}')
                plt.axis('off')
            plt.savefig(os.path.join(output_dir, f'row_{row_idx + 1}_dilated_segments.png'))
            plt.close()

            # 将未膨胀之前切分出来的小块丢到大块里面，组装成组
            grouped_segments = []
            # 添加调试日志输出坐标信息
            logger.debug(f'膨胀后大块数量: {len(dilated_segments)}')
            for d_idx, (_, d_coords) in enumerate(dilated_segments):
                logger.debug(f'大块[{d_idx}] 列范围: {d_coords[2]}-{d_coords[3]}')
            
            # 修正坐标匹配逻辑
            for dilated_segment, dilated_coords in dilated_segments:
                group = []
                logger.debug(f'处理膨胀大块 列范围: {dilated_coords[2]}-{dilated_coords[3]}')
                
                for segment_idx, (segment, coords) in enumerate(adjusted_segments):
                    # 修正判断条件：小块的列范围应在大块范围内
                    if (coords[2] >= dilated_coords[2]) and (coords[3] <= dilated_coords[3]):
                        logger.debug(f'小块[{segment_idx}] {coords[2]}-{coords[3]} 匹配当前大块')
                        group.append((segment, coords))
                    else:
                        logger.debug(f'小块[{segment_idx}] {coords[2]}-{coords[3]} 不匹配')
                
                if group:
                    logger.info(f'当前大块匹配到 {len(group)} 个小块')
                    grouped_segments.append(group)
                else:
                    logger.warning('当前大块未匹配到任何小块')

            # 可视化最终的分组结果并保存
            # 动态计算网格布局
            total_segments = sum(len(g) for g in grouped_segments)
            # n_cols = min(4, total_segments)  # 最大4列提高可读性
            n_cols = total_segments
            n_rows = (total_segments + n_cols - 1) // n_cols  # 向上取整
            
            logger.info(f"动态计算网格布局，行数: {n_rows}, 列数: {n_cols}")

            plt.figure(figsize=(15, 5 * n_rows))
            plot_index = 1

            try:
                for group_idx, group in enumerate(grouped_segments):
                    logger.info(f"处理第 {group_idx + 1} 组，共 {len(group)} 个小块")
                    
                    for segment_idx, (segment, _) in enumerate(group):
                        plt.subplot(n_rows, n_cols, plot_index)
                        plt.imshow(segment, cmap='gray')
                        plt.title(f'组{group_idx+1}-块{segment_idx+1}')
                        plt.axis('off')
                        plot_index += 1 
            except ValueError as e:
                logger.error(f'子图布局错误: {e}')
                plt.close()
                raise

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'row_{row_idx + 1}_final_groups.png'))
            plt.close()

            all_segments.append(grouped_segments)
            logger.info(f"第 {row_idx + 1} 行分组完成，共 {len(grouped_segments)} 组")

        logger.info(f"图像分割完成，共得到 {len(all_segments)} 行分割结果")
        return all_segments

    def is_equal_sign(self, img):
        '''
        判断图像是否为等号
        参数:
            img: 输入图像
        返回:
            bool: 是否为等号
        '''
        # 计算图像的水平投影
        h, w = img.shape
        horizontal_projection = np.zeros(h)
        for i in range(h):
            horizontal_projection[i] = np.count_nonzero(img[i, :])

        # 等号的特征是水平投影在中间部分有较高的值，且上下部分较低
        middle = h // 2
        upper = middle - h // 4
        lower = middle + h // 4

        upper_sum = np.sum(horizontal_projection[:upper])
        middle_sum = np.sum(horizontal_projection[upper:lower])
        lower_sum = np.sum(horizontal_projection[lower:])

        # 判断是否为等号
        return middle_sum > upper_sum and middle_sum > lower_sum

    def visualize_segments(self, segments, output_dir=None):
        '''
        可视化所有分割后的子图像，并将它们拼接到一起显示
        参数:
            segments: 分割后的子图像列表，按照行列和等式分组形式组织
            output_dir: 输出目录路径，如果为None则创建默认目录
        '''
        # 将嵌套列表展平
        flat_segments = []  # 初始化展平后的分割块列表
        for row in segments:  # 遍历每一行
            for group in row:  # 遍历每一组
                for segment in group:  # 遍历每个分割块
                    if isinstance(segment, tuple) and len(segment) > 0:  # 检查分割块是否有效
                        flat_segments.append(segment[0])  # 只取子图像部分
                    else:
                        logger.warning(f"警告: 忽略无效的 segment: {segment}")  # 记录警告日志

        num_segments = len(flat_segments)  # 计算分割块数量
        if num_segments == 0:  # 如果没有有效的分割块
            logger.warning("警告: 没有有效的子图像可供可视化")  # 记录警告日志
            return  # 退出函数

        # 计算网格布局的行列数
        rows = int(np.ceil(np.sqrt(num_segments)))  # 计算行数
        cols = int(np.ceil(num_segments / rows))  # 计算列数

        # 创建图形
        plt.figure(figsize=(cols * 2, rows * 2))  # 设置图形大小
        
        # 绘制每个分割块
        for i, segment in enumerate(flat_segments):  # 遍历每个分割块
            plt.subplot(rows, cols, i + 1)  # 创建子图
            plt.imshow(segment, cmap='gray')  # 显示图像
            plt.title(f'Segment {i+1}')  # 设置标题
            plt.axis('off')  # 关闭坐标轴

        plt.tight_layout()  # 调整布局
        
        # 添加保存图表的功能
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
        original_filename = get_filename_without_extension(self.image_path)  # 获取不含扩展名的文件名
        
        if output_dir is None:  # 如果未提供输出目录
            # 创建默认输出目录
            output_dir = os.path.join(os.getcwd(), f'{original_filename}_segments_{timestamp}')
            os.makedirs(output_dir, exist_ok=True)  # 创建目录
        
        # 构建输出路径
        output_path = os.path.join(output_dir, 'segments_visualization.png')
        plt.savefig(output_path)  # 保存图像
        logger.info(f"分割结果可视化已保存到: {output_path}")  # 记录日志
        
        plt.show()  # 显示图像

    def visualize_segments_by_row(self, segments, output_dir=None):
        '''
        按行和组为单位可视化分割后的图像
        参数:
            segments: 分割后的子图像列表，按照行列和等式分组形式组织
            output_dir: 输出目录，如果为None则保存在当前目录
        '''
        if output_dir is None:
            output_dir = os.getcwd()

        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = get_filename_without_extension(self.image_path)
        output_dir = os.path.join(output_dir, f'{original_filename}_segments_by_row_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)

        # 遍历每一行
        for row_idx, row in enumerate(segments):
            # 创建每行的图像
            plt.figure(figsize=(15, 5))
            plt.suptitle(f'Row {row_idx + 1}', fontsize=16)

            # 遍历每一组
            for group_idx, group in enumerate(row):
                # 创建每组的子图
                plt.subplot(1, len(row), group_idx + 1)
                plt.title(f'Group {group_idx + 1}')

                # 遍历组内的每个子图像
                for segment_idx, segment in enumerate(group):
                    # 检查 segment 是否为元组且包含图像和坐标
                    if isinstance(segment, tuple) and len(segment) == 2:
                        img, _ = segment
                        plt.imshow(img, cmap='gray')
                        plt.axis('off')
                    else:
                        print(f"警告: 忽略无效的 segment: {segment}")

            # 保存每行的图像
            output_path = os.path.join(output_dir, f'row_{row_idx + 1}.png')
            plt.savefig(output_path)
            plt.close()
            print(f"行 {row_idx + 1} 的可视化结果已保存到: {output_path}")

def get_filename_without_extension(file_path):
    '''
    从文件路径中提取文件名（不含扩展名）
    参数:
        file_path: 文件路径
    返回:
        文件名（不含扩展名）
    '''
    # 获取基本文件名（含扩展名）
    base_name = os.path.basename(file_path)  # 从路径中提取文件名部分
    # 分离文件名和扩展名
    file_name, _ = os.path.splitext(base_name)  # 将文件名和扩展名分开，忽略扩展名
    return file_name  # 返回不含扩展名的文件名

# 展示图片
def img_show_array(a):
    '''
    显示图像数组
    参数:
        a: 要显示的图像数组
    '''
    plt.imshow(a)  # 使用matplotlib显示图像数组
    plt.show()  # 显示图像
    
# 展示投影图， 输入参数arr是图片的二维数组，direction是x,y轴
def show_shadow(arr, direction='x', output_dir=None):
    a_max = max(arr)
    if direction == 'x':  # x轴方向的投影
        a_shadow = np.zeros((int(a_max), len(arr)), dtype=int)  # 将 a_max 转换为整数
        for i in range(0, len(arr)):
            if arr[i] == 0:
                continue
            for j in range(0, int(arr[i])):  # 将 arr[i] 转换为整数
                a_shadow[j][i] = 255
    elif direction == 'y':  # y轴方向的投影
        a_shadow = np.zeros((len(arr), int(a_max)), dtype=int)  # 将 a_max 转换为整数
        for i in range(0, len(arr)):
            if arr[i] == 0:
                continue
            for j in range(0, int(arr[i])):  # 将 arr[i] 转换为整数
                a_shadow[i][j] = 255

    img_show_array(a_shadow)

    # 保存投影图像
    if output_dir is not None:
        # 检查并创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'projection_{direction}_{timestamp}.png'
        output_path = os.path.join(output_dir, filename)
        
        plt.imshow(a_shadow)
        plt.savefig(output_path)
        plt.close()
        return output_path

def process_and_segment_image_v1(image_path):
    '''
    处理并分割图像，按照行列形式构建切割出来的块数据，并记录每个块的坐标
    参数:
        image_path: 输入图像的路径
    返回:
        分割后的子图像列表及其坐标，按照行列形式组织
    '''
    segmenter = ImageSegmenter(image_path)
        
    # 预处理图像
    segmenter.preprocess_image()
        
    # 计算水平投影并切割
    horizontal_projection = segmenter.horizontal_projection()
    rows = segmenter.cut_by_projection(horizontal_projection, axis='horizontal')
        
    all_segments = []

    for row, row_coords in rows:
        # 计算垂直投影并切割
        vertical_projection = segmenter.vertical_projection(row)
        segments = segmenter.cut_by_projection(vertical_projection, axis='vertical', input_img=row)
        adjusted_segments = [(img, (row_coords[0], row_coords[1], coords[2], coords[3])) for img, coords in segments]
        all_segments.append(adjusted_segments)

    print(f"图像分割完成，共得到 {len(all_segments)} 行分割结果")
        
    return all_segments


def process_and_segment_image_v2(image_path, output_dirs=None):
    '''
    处理并分割图像，生成结构化分割结果
    参数:
        image_path: 输入图像的路径
        output_dirs: 输出目录结构字典，包含images、visualizations、metadata等子目录
    返回:
        tuple: (output_base_dir, all_segments) 输出基础目录和结构化分割结果
    '''
    # 初始化图像分割器
    segmenter = ImageSegmenter(image_path)  # 创建图像分割器实例
    all_segments = segmenter.segment_image_v2()  # 执行图像分割
    
    # 生成唯一输出目录（如果未提供输出目录结构）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
    original_filename = get_filename_without_extension(image_path)  # 获取不含扩展名的文件名
    
    if output_dirs is None:
        # 如果未提供输出目录结构，创建默认输出目录
        output_base = os.path.join(os.getcwd(), f'{original_filename}_segments_{timestamp}')
        output_dirs = create_output_structure(output_base)  # 创建结构化输出目录
    else:
        # 使用提供的输出目录结构
        output_base = os.path.dirname(list(output_dirs.values())[0])  # 获取基础目录
    
    # 保存所有分割块到本地
    segmentation_map = {}  # 初始化分割映射字典
    for row_idx, row in enumerate(all_segments):  # 遍历每一行
        for group_idx, group in enumerate(row):  # 遍历每一组
            # 所有的行中是有分组的图片与对应的坐标内容的
            for segment_idx, (img, coords) in enumerate(group):  # 遍历组内每个分割块
                # 构建分割块图像的输出路径
                output_path = os.path.join(output_dirs['images'], f'row{row_idx+1}_group{group_idx+1}_seg{segment_idx+1}.png')
                cv2.imwrite(output_path, img)  # 保存分割块图像
                
                # 记录坐标映射关系
                segmentation_map[output_path] = {
                    'row': row_idx+1,  # 行索引
                    'group': group_idx+1,  # 组索引
                    'coordinates': coords  # 坐标信息
                }
    
    # 保存坐标元数据
    metadata_path = os.path.join(output_dirs['metadata'], 'segmentation_meta.json')
    with open(metadata_path, 'w') as f:
        json.dump(segmentation_map, f, indent=2)  # 将分割映射保存为JSON文件
    logger.info(f"分割元数据已保存到: {metadata_path}")  # 记录日志
        
    # 图像分割后的识别调用示例
    def recognize_segmented_character(imgs):
        """
        调用手写识别模型的预测接口
        参数:
            imgs: 图像列表
        返回:
            识别结果列表
        """
        return predict_image_v1(imgs)  # 调用预测函数
    
    # 遍历所有分割块进行预测
    recognition_results = []  # 初始化识别结果列表
    # 存储每个组的识别结果和坐标，用于后续统一处理表达式
    group_data = []  # 初始化组数据列表
    
    for row_idx, row in enumerate(all_segments):  # 遍历每一行
        for group_idx, group in enumerate(row):  # 遍历每一组
            # 提取组中的图像和坐标
            imgs = [img for img, _ in group]  # 提取图像列表
            coordinates = [coord for _, coord in group]  # 提取坐标列表
            
            logger.debug(f"坐标列表: {coordinates}")  # 记录调试日志
            
            # 识别图像内容
            content = recognize_segmented_character(imgs)  # 调用识别函数
            logger.info(f"第 {row_idx+1} 行第 {group_idx+1} 组识别结果: {content}")  # 记录日志
            
            # 存储当前组的数据，用于后续统一处理表达式
            group_data.append({  # 添加组数据
                'row_idx': row_idx,  # 行索引
                'group_idx': group_idx,  # 组索引
                'content': content,  # 识别内容
                'coordinates': coordinates,  # 坐标列表
                'group': group  # 原始组数据
            })
            
            # 可视化每个分割块及其预测结果
            plt.figure(figsize=(15, 5))  # 创建图形
            for segment_idx, (img, coords) in enumerate(group):  # 遍历组内每个分割块
                plt.subplot(1, len(group), segment_idx + 1)  # 创建子图
                plt.imshow(img, cmap='gray')  # 显示图像
                if segment_idx < len(content):  # 如果有对应的识别结果
                    ct = content[segment_idx]  # 获取识别结果
                    plt.title(f'预测: {ct["predicted_label"]}\n置信度: {ct["confidence"]:.2f}')  # 设置标题
                else:
                    plt.title(f'未识别')  # 设置标题
                plt.axis('off')  # 关闭坐标轴
            
            plt.suptitle(f'行 {row_idx + 1} - 组 {group_idx + 1}', fontsize=16)  # 设置总标题
            plt.tight_layout()  # 调整布局
            # 保存预测可视化结果
            vis_path = os.path.join(output_dirs['visualizations'], f'row{row_idx+1}_group{group_idx+1}_prediction.png')
            plt.savefig(vis_path)  # 保存图像
            plt.close()  # 关闭图形
            
            # 处理每个分割块的识别结果
            for segment_idx, (img, coords) in enumerate(group):  # 遍历组内每个分割块
                if segment_idx < len(content):  # 如果有对应的识别结果
                    ct = content[segment_idx]  # 获取识别结果
                    # 创建数据对象
                    data_obj = DataObject(
                        content=ct,  # 识别内容
                        coordinates=coords,  # 坐标信息
                        confidence=ct['confidence'],  # 置信度
                    )
                    
                    # 记录识别结果
                    recognition_results.append(data_obj)  # 添加到结果列表
                    
                    # 更新元数据
                    output_path = os.path.join(output_dirs['images'], f'row{row_idx+1}_group{group_idx+1}_seg{segment_idx+1}.png')
                    segmentation_map[output_path]['content'] = ct  # 添加识别内容
                    segmentation_map[output_path]['confidence'] = ct['confidence']  # 添加置信度
    
    # 统一处理所有表达式
    logger.info(f"开始统一处理所有表达式，共 {len(group_data)} 组数据")  # 记录日志
    
    # 读取原始图像一次，用于所有表达式的标注
    original_img = cv2.imread(image_path)  # 读取原始图像
    if original_img is None:  # 如果读取失败
        logger.error(f"无法读取原始图像: {image_path}")  # 记录错误日志
    else:  # 如果读取成功
        logger.info(f"成功读取原始图像，尺寸: {original_img.shape}")  # 记录日志
        
        # 处理所有表达式并在同一个图像上标注
        for data in group_data:  # 遍历每组数据
            row_idx = data['row_idx']  # 获取行索引
            group_idx = data['group_idx']  # 获取组索引
            content = data['content']  # 获取识别内容
            coordinates = data['coordinates']  # 获取坐标列表
            
            # 根据返回的数据构建算式表达式，并在原图上标注
            expression, expression_result, is_correct = build_and_excute_expression_on_image(content, coordinates, original_img)
            
            # 更新元数据
            output_path = os.path.join(output_dirs['images'], f'row{row_idx+1}_group{group_idx+1}_seg1.png')  # 使用第一个分割块的路径作为键
            segmentation_map[output_path]['expression'] = expression  # 添加表达式
            segmentation_map[output_path]['result'] = expression_result  # 添加计算结果
        
        # 所有表达式处理完毕后，保存标注后的图像
        annotated_path = os.path.join(output_dirs['visualizations'], f"annotated_{os.path.basename(image_path)}")
        cv2.imwrite(annotated_path, original_img)  # 保存标注后的图像
        logger.info(f"已保存包含所有标注的图像到: {annotated_path}")  # 记录日志
    
    # 保存更新后的元数据
    with open(metadata_path, 'w') as f:  # 打开元数据文件
        json.dump(segmentation_map, f, indent=2)  # 保存更新后的元数据
    
    return output_base, all_segments  # 返回输出基础目录和分割结果

def build_and_excute_expression_on_image(ct_data, coordinates, img):
    """
    构建可执行表达式并验证正确性，在提供的图像上绘制结果
    参数:
        ct_data: 识别结果字典（包含predicted_label）
        coordinates: 识别结果坐标列表
        img: 要绘制标注的图像（已加载的图像对象）
    返回:
        tuple: (完整表达式, 计算结果, 是否正确)
    """
    logger.debug("ct_data: %s", ct_data)  # 使用日志记录识别结果数据
    logger.debug("坐标列表: %s", coordinates)  # 使用日志记录坐标列表
    
    # 1. 构建原始表达式
    raw_expression = "".join([item['predicted_label'] for item in ct_data])  # 连接所有预测标签形成表达式
    logger.info(f"原始表达式: {raw_expression}")  # 记录原始表达式
    
    # 2. 符号替换（手写字符转计算机符号）
    expression = raw_expression.replace('x', '*').replace('÷', '/')  # 将乘除符号转换为Python可执行的符号
    logger.info(f"替换后的表达式: {expression}")  # 记录替换后的表达式
    
    # 3. 初始化 result 变量
    result = None  # 初始化计算结果为None
    is_correct = False  # 初始化正确性标志为False
    
    # 4. 分割等式（如果有等号）
    if '=' in expression:  # 如果表达式包含等号
        left, right = expression.split('=', 1)  # 分割等式左右两边
        try:
            # 计算左右两边
            calc_left = eval(left)  # 计算等式左边的值
            calc_right = eval(right)  # 计算等式右边的值
            is_correct = abs(calc_left - calc_right) < 1e-9  # 浮点容差，判断等式是否成立
            result = calc_left  # 使用左边的计算结果作为最终结果
        except Exception as e:
            logger.error(f"计算出错(1): {expression}, 错误: {e}")  # 记录计算错误
            calc_left = None  # 设置计算结果为None
            is_correct = False  # 设置正确性标志为False
    else:  # 如果表达式不包含等号
        try:
            result = eval(expression)  # 直接计算表达式的值
            is_correct = None  # 无对比基准，设置正确性标志为None
        except Exception as e:
            logger.error(f"计算出错(2): {expression}, 错误: {e}")  # 记录计算错误
            result = None  # 设置计算结果为None
            is_correct = False  # 设置正确性标志为False

    # 5. 在图像上绘制结果
    if img is None:  # 如果图像为None
        raise ValueError("无法读取图像")  # 抛出异常
    
    # 获取图像尺寸
    img_height, img_width = img.shape[:2]  # 获取图像高度和宽度
    logger.debug(f"图像尺寸: 宽={img_width}, 高={img_height}")  # 记录图像尺寸

    # 正确提取坐标信息
    # 坐标格式为 (y_start, y_end, x_start, x_end)
    if coordinates and len(coordinates) > 0:  # 如果有坐标信息
        # 使用第一个字符的起始位置作为基准
        first_coord = coordinates[0]  # 获取第一个坐标
        last_coord = coordinates[-1]  # 获取最后一个坐标
        
        # 提取坐标值
        y_start = first_coord[0]  # 第一个字符的y起始位置
        y_end = first_coord[1]    # 第一个字符的y结束位置
        x_start = first_coord[2]  # 第一个字符的x起始位置
        x_end = last_coord[3]     # 最后一个字符的x结束位置
        
        logger.debug(f"提取的坐标: y_start={y_start}, y_end={y_end}, x_start={x_start}, x_end={x_end}")  # 记录提取的坐标
        
        # 计算文本位置，确保在图像范围内
        text_x = max(10, x_start)  # 确保x坐标至少为10
        text_y = min(img_height - 20, y_end + 30)  # 在字符下方30像素，但不超出图像底部
        
        # 绘制原始表达式
        result_text = f"{raw_expression} = {result if result is not None else 'Error'}"  # 构建结果文本
        logger.debug(f"绘制文本: '{result_text}' 在位置 ({text_x}, {text_y})")  # 记录绘制文本信息
        
        # 使用OpenCV在图像上绘制文本
        cv2.putText(img, result_text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 绘制对错标识
        if is_correct is not None:  # 如果有正确性标志
            mark = "good" if is_correct else "wrong"  # 根据正确性设置标记文本
            mark_color = (0, 255, 0) if is_correct else (0, 0, 255)  # 根据正确性设置标记颜色（绿色或红色）
            mark_x = min(img_width - 50, text_x + 200)  # 确保不超出图像右侧
            # 在图像上绘制标记
            cv2.putText(img, mark, (mark_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, mark_color, 2)
    else:  # 如果没有坐标信息
        logger.warning("警告: 没有有效的坐标信息")  # 记录警告

    return expression, result, is_correct  # 返回表达式、计算结果和正确性标志
    

def create_output_structure(base_dir):
    '''
    创建结构化的输出目录
    参数:
        base_dir: 基础输出目录路径
    返回:
        包含各子目录路径的字典
    '''
    # 创建主目录
    os.makedirs(base_dir, exist_ok=True)
    
    # 创建子目录
    subdirs = {
        'images': os.path.join(base_dir, 'images'),  # 存储分割后的图像
        'visualizations': os.path.join(base_dir, 'visualizations'),  # 存储可视化结果
        'metadata': os.path.join(base_dir, 'metadata'),  # 存储元数据
        'debug': os.path.join(base_dir, 'debug')  # 存储调试信息
    }
    
    # 创建所有子目录
    for dir_path in subdirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return subdirs

def main():
    '''
    主函数，处理命令行参数并执行图像分割
    '''
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='手写算式图像分割与识别工具')
    
    # 添加命令行参数
    parser.add_argument('image_path', type=str, help='输入图像的路径（绝对路径或相对路径）')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出目录路径（默认为当前目录下的时间戳文件夹）')
    parser.add_argument('--version', '-v', type=int, choices=[1, 2], default=2, help='使用的分割算法版本（1或2，默认为2）')
    parser.add_argument('--visualize', '-vis', action='store_true', help='是否显示可视化结果')
    parser.add_argument('--debug', '-d', action='store_true', help='启用调试模式')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置日志级别
    if args.debug:  # 如果启用调试模式
        logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG
    
    # 处理图像路径（支持相对路径）
    image_path = args.image_path  # 获取图像路径
    if not os.path.isabs(image_path):  # 如果不是绝对路径
        # 将相对路径转换为绝对路径
        image_path = os.path.abspath(image_path)
    
    # 检查文件是否存在
    if not os.path.exists(image_path):  # 如果文件不存在
        logger.error(f"错误：图像文件不存在: {image_path}")  # 记录错误日志
        return  # 退出函数
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
    original_filename = get_filename_without_extension(image_path)  # 获取不含扩展名的文件名
    
    if args.output:  # 如果指定了输出目录
        # 使用用户指定的输出目录
        if not os.path.isabs(args.output):  # 如果不是绝对路径
            # 将相对路径转换为绝对路径
            output_base = os.path.abspath(args.output)
        else:
            output_base = args.output  # 使用原始路径
    else:  # 如果未指定输出目录
        # 使用默认输出目录
        output_base = os.path.join(os.getcwd(), f'{original_filename}_output_{timestamp}')
    
    # 创建结构化输出目录
    output_dirs = create_output_structure(output_base)  # 创建输出目录结构
    logger.info(f"创建输出目录: {output_base}")  # 记录日志
    
    # 根据版本选择分割算法
    logger.info(f"使用分割算法版本: {args.version}")  # 记录使用的算法版本
    if args.version == 1:  # 如果使用版本1
        # 使用版本1的分割算法
        segments = process_and_segment_image_v1(image_path)  # 处理并分割图像
        
        # 保存分割结果到结构化目录
        for row_idx, row in enumerate(segments):  # 遍历每一行
            for segment_idx, (img, coords) in enumerate(row):  # 遍历每个分割块
                # 构建输出路径
                output_path = os.path.join(output_dirs['images'], f'row{row_idx+1}_seg{segment_idx+1}.png')
                cv2.imwrite(output_path, img)  # 保存图像
        
        # 可视化分割结果
        if args.visualize:  # 如果需要可视化
            segmenter = ImageSegmenter(image_path)  # 创建分割器实例
            # 将可视化结果保存到可视化目录
            segmenter.visualize_segments(segments, output_dir=output_dirs['visualizations'])
    else:  # 如果使用版本2
        # 使用版本2的分割算法，并传入结构化输出目录
        output_base, segments = process_and_segment_image_v2(image_path, output_dirs)
    
    # 保存处理信息到元数据目录
    metadata_summary = {
        'image_path': image_path,
        'processing_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'algorithm_version': args.version,
        'output_directory': output_base
    }
    
    # 保存处理摘要
    summary_path = os.path.join(output_dirs['metadata'], 'processing_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(metadata_summary, f, indent=2)  # 保存处理摘要
    
    logger.info(f"处理完成，结果已保存到: {output_base}")  # 记录处理完成日志

if __name__ == "__main__":
    main()  # 调用主函数执行程序
                    
                    
                    
                    
                    