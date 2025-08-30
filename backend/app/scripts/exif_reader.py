import os
import exifread
from pathlib import Path
import json
import pyexiv2

def write_exif(image_path, exif_data):
    """写入EXIF信息"""
    try:
        with pyexiv2.Image(image_path) as img:
            # 获取现有的元数据
            metadata = img.read_exif()
            # 更新元数据
            for key, value in exif_data.items():
                metadata[key] = value
            # 写入更新后的元数据
            img.modify_exif(metadata)
        print(f"成功写入EXIF信息到 {image_path}")
    except Exception as e:
        print(f"写入EXIF信息时出错: {str(e)}")

def read_image_exif(image_path):
    """读取图片的EXIF信息"""
    print(f"\n========== 开始读取EXIF信息 ==========")
    print(f"输入参数 image_path: {image_path}")
    print(f"文件是否存在: {os.path.exists(image_path)}")
    print(f"文件大小: {os.path.getsize(image_path) if os.path.exists(image_path) else '文件不存在'} 字节")
    
    try:
        # 打开图片文件
        print("步骤1: 打开图片文件并读取EXIF信息")
        with open(image_path, 'rb') as f:
            # 读取EXIF信息
            print("调用exifread.process_file处理图片文件...")
            tags = exifread.process_file(f, details=False)
            print(f"exifread.process_file返回结果类型: {type(tags)}")
            print(f"返回的标签数量: {len(tags) if tags else 0}")
            
            if not tags:
                print(f"警告: 未在图片中找到EXIF信息: {image_path}")
                print("返回空字典 {}")
                return {}
            
            # 打印所有可用的EXIF信息
            print(f"\n图片 {Path(image_path).name} 的EXIF信息:")
            print("-" * 50)
            
            # 常见EXIF标签
            important_tags = [
                'EXIF DateTimeOriginal',  # 拍摄时间
                'Image Make',             # 相机制造商
                'Image Model',            # 相机型号
                'EXIF ExposureTime',      # 曝光时间
                'EXIF FNumber',           # 光圈值
                'EXIF ISOSpeedRatings',   # ISO感光度
                'EXIF FocalLength',       # 焦距
                'GPS GPSLatitude',        # GPS纬度
                'GPS GPSLongitude',       # GPS经度
                'GPS GPSLatitudeRef',     # GPS纬度参考
                'GPS GPSLongitudeRef',    # GPS经度参考
                'GPS GPSAltitude',        # GPS海拔
                'Image ImageDescription'  # 图片描述
            ]
            
            # 收集所有的tag信息，收集完成之后，将其返回给调用方
            print("\n步骤2: 收集EXIF标签信息")
            result = {}
            # 首先显示重要标签
            print("重要EXIF标签:")
            for tag in important_tags:
                if tag in tags:
                    tag_value = str(tags[tag])
                    print(f"  {tag}: {tag_value}")
                    result[tag] = tag_value
                else:
                    print(f"  {tag}: 不存在")
            
            # 显示其他标签
            print("\n其他EXIF信息:")
            other_tags_count = 0
            for tag, value in tags.items():
                if tag not in important_tags:
                    tag_value = str(value)
                    result[tag] = tag_value
                    other_tags_count += 1
                    if other_tags_count <= 10:  # 只打印前10个其他标签
                        print(f"  {tag}: {tag_value}")
            
            if other_tags_count > 10:
                print(f"  ... 还有 {other_tags_count - 10} 个其他标签 ...")
            
            print(f"\n步骤3: 返回结果")
            print(f"返回字典包含 {len(result)} 个键值对")
            print(f"返回字典类型: {type(result)}")
            
            # 检查GPS相关信息
            gps_keys = ['GPS GPSLatitude', 'GPS GPSLongitude', 'GPS GPSLatitudeRef', 'GPS GPSLongitudeRef', 'GPS GPSAltitude']
            print("\nGPS信息检查:")
            for key in gps_keys:
                if key in result:
                    print(f"  {key}: {result[key]}")
                else:
                    print(f"  {key}: 不存在")
                    
            return result
    except FileNotFoundError:
        print(f"错误：找不到图片文件 {image_path}")
        print("返回空字典 {}")
        return {}
    except Exception as e:
        import traceback
        print(f"处理图片时出错 {image_path}: {str(e)}")
        print(f"错误详情: {traceback.format_exc()}")
        print("返回空字典 {}")
        return {}

def main():
    # 获取用户输入的图片路径
    image_path = input("请输入图片文件的路径: ")
    
    if not os.path.exists(image_path):
        print("错误：指定的文件不存在！")
        return
    
    if not os.path.isfile(image_path):
        print("错误：指定的路径不是文件！")
        return
        
    # 检查文件扩展名
    valid_extensions = ('.jpg', '.jpeg', '.tiff', '.raw')
    if not image_path.lower().endswith(valid_extensions):
        print(f"错误：不支持的文件格式！请使用以下格式的图片文件: {', '.join(valid_extensions)}")
        return
    
    # 读取并显示EXIF信息
    read_image_exif(image_path)

# 读取某个目录下的所有图片文件，并且读取exif信息
def main_2():
    # 获取用户输入的图片路径
    image_path = input("请输入图片文件夹的路径: ")
    
    if not os.path.exists(image_path):
        print("错误：指定的文件夹不存在！")
        return
    
    if not os.path.isdir(image_path):
        print("错误：指定的路径不是文件夹！")
        return
        
    # 遍历图片文件夹下的所有图片文件
    for image_file in sorted(os.listdir(image_path)):    
        # 读取并显示EXIF信息(result是一个字典)
        result = read_image_exif(os.path.join(image_path, image_file))
        # 将result 写入到文件
        if result:
            with open('exif_info.txt', 'a') as f:
                f.write(json.dumps(result) + '\n')  # 使用json.dumps将字典转换为字符串，并添加换行符

'''
读取/Users/shhaofu/Code/cursor-projects/aka_music/frontend/public/gis/travel/zhejiang.json该文件，抽取出其所有的城市对应的经纬度信息,
将其写入到/Users/shhaofu/Code/cursor-projects/aka_music/frontend/public/gis/travel/img下对应名字的图片中
'''
def main_3():
    with open('/Users/shhaofu/Code/cursor-projects/aka_music/frontend/public/gis/travel/zhejiang.json', 'r') as f:
        data = json.load(f)
        for city in data['cities']:  # 根据JSON结构修改为'cities'
            city_name = city['name']
            lat = city['coordinates']['lat']  # 获取纬度
            lng = city['coordinates']['lng']  # 获取经度
            
            # 确定经纬度参考方向
            lat_ref = 'N' if lat >= 0 else 'S'
            lng_ref = 'E' if lng >= 0 else 'W'
            
            # 确保经纬度为正值
            lat = abs(lat)
            lng = abs(lng)
            
            # 将经纬度转换为度分秒格式 (pyexiv2需要这种格式)
            # 度分秒格式: (度, 分, 秒)
            lat_deg = int(lat)
            lat_min = int((lat - lat_deg) * 60)
            lat_sec = (lat - lat_deg - lat_min/60) * 3600
            
            lng_deg = int(lng)
            lng_min = int((lng - lng_deg) * 60)
            lng_sec = (lng - lng_deg - lng_min/60) * 3600
            
            # 将经纬度信息写入到图片的EXIF信息中
            image_path = os.path.join('/Users/shhaofu/Code/cursor-projects/aka_music/frontend/public/gis/travel/img', city_name + '.jpeg')
            try:
                # 创建EXIF数据字典
                exif_data = {
                    'Exif.GPSInfo.GPSLatitudeRef': lat_ref,
                    'Exif.GPSInfo.GPSLongitudeRef': lng_ref,
                    'Exif.GPSInfo.GPSLatitude': (lat_deg, lat_min, lat_sec),
                    'Exif.GPSInfo.GPSLongitude': (lng_deg, lng_min, lng_sec),
                    'Exif.Image.ImageDescription': f"经度: {lng}, 纬度: {lat}"
                }
                
                # 使用write_exif函数写入EXIF信息
                write_exif(image_path, exif_data)
                print(f"成功写入GPS经纬度信息到 {city_name}.jpeg")
            except Exception as e:
                print(f"写入GPS信息到 {city_name}.jpeg 时出错: {str(e)}")

if __name__ == '__main__':
    # main()
    main_2()
    # main_3()
    
'''
图片 IMG20250224165602.jpg 的EXIF信息:
--------------------------------------------------
Image Make: OPPO
Image Model: OPPO K9x 5G
EXIF ExposureTime: 4997/500000
EXIF FNumber: 17/10
EXIF ISOSpeedRatings: 652
EXIF FocalLength: 473/100

其他EXIF信息:
Image ImageLength: 4624
Image Orientation: Horizontal (normal)
GPS GPSAltitude: 0/0
GPS GPSAltitudeRef: 0
GPS GPSProcessingMethod: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... ]
GPS GPSVersionID: [0, 0, 0, 0]
Image GPSInfo: 1344
Image YResolution: 72
Image XResolution: 72
Image ImageWidth: 3468
Image Software: MediaTek Camera Application
Image ImageDescription: 
Image YCbCrPositioning: Co-sited
Image ExifOffset: 314
Image ResolutionUnit: Pixels/Inch
Thumbnail YResolution: 72
Thumbnail Orientation: Horizontal (normal)
Thumbnail Compression: JPEG (old-style)
Thumbnail JPEGInterchangeFormat: 2036
Thumbnail JPEGInterchangeFormatLength: 0
Thumbnail XResolution: 72
Thumbnail YCbCrPositioning: Co-sited
Thumbnail ResolutionUnit: Pixels/Inch
EXIF ExifVersion: 0220
EXIF ApertureValue: 153/100
EXIF ExposureBiasValue: 0
EXIF ExposureProgram: Unidentified
EXIF ColorSpace: sRGB
EXIF MaxApertureValue: 153/100
EXIF ExifImageLength: 4624
EXIF BrightnessValue: 9
EXIF FlashPixVersion: 0100
EXIF SubSecTimeOriginal: 331
EXIF WhiteBalance: Auto
Interoperability InteroperabilityIndex: R98
EXIF InteroperabilityOffset: 1470
EXIF RecommendedExposureIndex: 0
EXIF ExposureMode: Auto Exposure
EXIF Flash: Flash did not fire, compulsory flash mode
EXIF SubSecTime: 331
EXIF ExifImageWidth: 3468
EXIF ComponentsConfiguration: YCbCr
EXIF FocalLengthIn35mmFilm: 0
EXIF SubSecTimeDigitized: 331
EXIF DigitalZoomRatio: 1
EXIF ShutterSpeedValue: 6643/1000
EXIF MeteringMode: CenterWeightedAverage
EXIF SensitivityType: Unknown
EXIF OffsetTimeOriginal: +08:00
EXIF SceneCaptureType: Standard
EXIF LightSource: other light source
EXIF SensingMethod: 0
'''