from flask import Blueprint, request, jsonify
from app.utils.response import make_response
from app.services.face_recognition_service import FaceRecognitionService
from app.scripts.exif_reader import read_image_exif
import os
import logging
from werkzeug.utils import secure_filename

bp = Blueprint('face_recognition', __name__, url_prefix='/api/face')
face_service = FaceRecognitionService()

# 确保上传目录存在
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'uploads', 'faces')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@bp.route('/extract', methods=['POST', 'GET'])
def extract_face():
    """从上传的图片中提取人脸特征"""
    try:
        # # 检查是否有文件上传
        # if 'image' not in request.files:
        #     return make_response(code=400, message='未上传图片文件')
            
        # file = request.files['image']
        # if file.filename == '':
        #     return make_response(code=400, message='未选择文件')
            
        # # 保存上传的文件
        # filename = secure_filename(file.filename)
        # file_path = os.path.join(UPLOAD_FOLDER, filename)
        # file.save(file_path)
        
        # 步骤1: 获取并处理文件路径
        print("步骤1: 获取并处理文件路径")
        path = request.args.get('path')
        print(f"请求参数path: {path}")
        
        if path and path.startswith('/static/'):
            # 如果路径以/static/开头，则从项目根目录开始定位
            file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), path[1:])
            print(f"路径以/static/开头，处理后的文件路径: {file_path}")
        else:
            # 否则保持原有逻辑
            file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
            print(f"处理后的文件路径: {file_path}")
            
        print(f"文件是否存在: {os.path.exists(file_path)}")
        
        # 提取人脸特征
        face_info = face_service.extract_face_features(file_path)
        
        if not face_info:
            return make_response(code=404, message='未在图片中检测到人脸')
            
        return make_response(data={
            'image_path': face_info['image_path'],
            'face_path': face_info['face_path'],
            'face_location': face_info['face_location'],
            'metadata': face_info['metadata']
        })
        
    except Exception as e:
        logging.error(f'提取人脸特征失败: {str(e)}')
        return make_response(code=500, message=f'提取人脸特征失败: {str(e)}')

@bp.route('/search', methods=['POST', 'GET'])
def search_faces():
    """搜索相似人脸"""
    try:
        # 步骤1: 获取并处理文件路径
        print("步骤1: 获取并处理文件路径")
        path = request.args.get('path')
        print(f"请求参数path: {path}")
        
        if path and path.startswith('/static/'):
            # 如果路径以/static/开头，则从项目根目录开始定位
            file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), path[1:])
            print(f"路径以/static/开头，处理后的文件路径: {file_path}")
        else:
            # 否则保持原有逻辑
            file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
            print(f"处理后的文件路径: {file_path}")
            
        print(f"文件是否存在: {os.path.exists(file_path)}")
        
        # 搜索相似人脸
        similar_faces = face_service.search_similar_faces(file_path, limit=10)
        
        if not similar_faces:
            return make_response(code=404, message='未找到相似人脸')
            
        return make_response(data={
            'results': similar_faces
        })
        
    except Exception as e:
        logging.error(f'搜索相似人脸失败: {str(e)}')
        return make_response(code=500, message=f'搜索相似人脸失败: {str(e)}')

@bp.route('/associate', methods=['POST'])
def associate_face_with_user():
    """将人脸与用户关联"""
    try:
        data = request.get_json()
        if not data:
            return make_response(code=400, message='请求参数不能为空')
            
        face_id = data.get('face_id')
        user_id = data.get('user_id')
        
        if not face_id or not user_id:
            return make_response(code=400, message='face_id和user_id不能为空')
            
        # 关联人脸与用户
        success = face_service.associate_face_with_user(face_id, user_id)
        
        if not success:
            return make_response(code=404, message='关联失败，请检查face_id和user_id是否有效')
            
        return make_response(message='关联成功')
        
    except Exception as e:
        logging.error(f'关联人脸与用户失败: {str(e)}')
        return make_response(code=500, message=f'关联人脸与用户失败: {str(e)}')

@bp.route('/identify', methods=['POST'])
def identify_user_by_face():
    """通过人脸识别用户"""
    try:
        # 检查是否有文件上传
        if 'image' not in request.files:
            return make_response(code=400, message='未上传图片文件')
            
        file = request.files['image']
        if file.filename == '':
            return make_response(code=400, message='未选择文件')
            
        # 获取相似度阈值参数
        threshold = request.form.get('threshold', 0.8, type=float)
        
        # 保存上传的文件
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # 通过人脸识别用户
        user = face_service.get_user_by_face(file_path, similarity_threshold=threshold)
        
        if not user:
            return make_response(code=404, message='未识别到用户')
            
        return make_response(data={
            'user': user
        })
        
    except Exception as e:
        logging.error(f'人脸识别用户失败: {str(e)}')
        return make_response(code=500, message=f'人脸识别用户失败: {str(e)}')

@bp.route('/exif', methods=['POST', 'GET'])
def get_image_exif():
    """获取图片的EXIF信息"""
    try:
        print("========== 开始获取图片EXIF信息 ==========")
        '''
        # 检查是否有文件上传
        if 'image' not in request.files:
            return make_response(code=400, message='未上传图片文件')
            
        file = request.files['image']
        if file.filename == '':
            return make_response(code=400, message='未选择文件')
            
        # 保存上传的文件
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        '''
        
        # 步骤1: 获取并处理文件路径
        print("步骤1: 获取并处理文件路径")
        path = request.args.get('path')
        print(f"请求参数path: {path}")
        
        if path and path.startswith('/static/'):
            # 如果路径以/static/开头，则从项目根目录开始定位
            file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), path[1:])
            print(f"路径以/static/开头，处理后的文件路径: {file_path}")
        else:
            # 否则保持原有逻辑
            file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
            print(f"处理后的文件路径: {file_path}")
            
        print(f"文件是否存在: {os.path.exists(file_path)}")
        logging.info(f'获取图片的EXIF信息: {file_path}')
        
        # 提取EXIF信息
        try:
            # 步骤2: 加载图片
            print("\n步骤2: 加载图片")
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            try:
                img = Image.open(file_path)
                print(f"成功加载图片，图片模式: {img.mode}, 尺寸: {img.width}x{img.height}")
            except Exception as e:
                print(f"加载图片失败: {str(e)}")
                raise
                
            exif_data = {}
            
            # 步骤3: 获取PIL内置的EXIF数据
            print("\n步骤3: 获取PIL内置的EXIF数据")
            exif = img.getexif()
            print(f"PIL获取的EXIF数据: {exif}")
            # 打印EXIF信息的键值对
            if exif:
                for tag_id, value in exif.items():
                    logging.info(f'EXIF信息 - 键: {tag_id}, 值: {value}')
            else:
                logging.info('EXIF信息为空')
            '''
            if exif:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    # 处理二进制数据和复杂对象
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', 'ignore')
                        except:
                            value = str(value)
                    try:
                        exif_data[tag] = str(value)
                    except:
                        exif_data[tag] = "[无法序列化的数据]"
            '''
            
            # 步骤4: 使用exifread库读取EXIF信息
            print("\n步骤4: 使用exifread库读取EXIF信息")
            print(f"调用read_image_exif函数，参数: {file_path}")
            exif_data = read_image_exif(file_path)
            print(f"read_image_exif返回结果类型: {type(exif_data)}")
            if exif_data:
                print(f"read_image_exif返回的键值数量: {len(exif_data)}")
                print("部分EXIF键值示例:")
                # 打印所有EXIF数据的键值对
                for key in exif_data.keys():
                    print(f"  {key}: {exif_data[key]}")
            else:
                print("read_image_exif返回的结果为空")
            
            # 打印 exif_data 信息
            print("\n步骤5: 检查EXIF数据")
            print(f"EXIF Data类型: {type(exif_data)}")
            
            # 检查exif_data是否为None或空字典
            if exif_data is None:
                print("警告: exif_data为None，将其初始化为空字典")
                exif_data = {}
                logging.warning(f'未能从图片中提取EXIF信息: {file_path}')
            elif not exif_data:
                print("警告: exif_data为空字典")
                
            # 步骤6: 构建结构化的EXIF数据
            print("\n步骤6: 构建结构化的EXIF数据")
            print("开始构建structured_exif字典")
            
            # 打印关键EXIF字段的存在情况
            important_keys = ['Image Make', 'Image Model', 'Image Software', 'EXIF ExposureTime', 
                             'EXIF FNumber', 'EXIF ISOSpeedRatings', 'EXIF FocalLength', 
                             'Image Orientation', 'EXIF ColorSpace', 'EXIF DateTimeOriginal', 
                             'EXIF DateTimeDigitized']
            print("检查关键EXIF字段是否存在:")
            for key in important_keys:
                print(f"  {key}: {'存在' if key in exif_data else '不存在'}")
                
            structured_exif = {
                'camera': {
                    'make': exif_data.get('Image Make', '未知'),
                    'model': exif_data.get('Image Model', '未知'),
                    'software': exif_data.get('Image Software', '未知')
                },
                'exposure': {
                    'exposureTime': exif_data.get('EXIF ExposureTime', '未知'),
                    'fNumber': exif_data.get('EXIF FNumber', '未知'),
                    'iso': exif_data.get('EXIF ISOSpeedRatings', '未知'),
                    'focalLength': exif_data.get('EXIF FocalLength', '未知')
                },
                'image': {
                    'width': img.width,
                    'height': img.height,
                    'orientation': exif_data.get('Image Orientation', '未知'),
                    'colorSpace': exif_data.get('EXIF ColorSpace', '未知')
                },
                'datetime': {
                    'original': exif_data.get('EXIF DateTimeOriginal', '未知'),
                    'digitized': exif_data.get('EXIF DateTimeDigitized', '未知')
                },
                'gps': {},
                'other': {
                    'lightSource': exif_data.get('EXIF LightSource', '未知'),
                    'meteringMode': exif_data.get('EXIF MeteringMode', '未知'),
                    'whiteBalance': exif_data.get('EXIF WhiteBalance', '未知'),
                    'flash': exif_data.get('EXIF Flash', '未知')
                }
            }
            print("structured_exif字典构建完成")
            
            # 处理GPS信息
            # 从exif_info.txt可以看出，GPS信息可能以十进制度数格式存储，也可能以度分秒格式存储
            gps_lat = exif_data.get('GPS GPSLatitude')
            gps_lat_ref = exif_data.get('GPS GPSLatitudeRef', 'N')
            gps_lon = exif_data.get('GPS GPSLongitude')
            gps_lon_ref = exif_data.get('GPS GPSLongitudeRef', 'E')
            gps_alt = exif_data.get('GPS GPSAltitude')
            img_desc = exif_data.get('Image ImageDescription', '')
            
            # 首先尝试从ImageDescription中提取经纬度信息
            lat_dec = None
            lon_dec = None
            
            if img_desc and '经度:' in img_desc and '纬度:' in img_desc:
                try:
                    # 解析格式如"经度: 120.0863, 纬度: 30.893"的字符串
                    desc_parts = img_desc.split(',')
                    for part in desc_parts:
                        part = part.strip()
                        if '经度:' in part:
                            lon_dec = float(part.split('经度:')[1].strip())
                        elif '纬度:' in part:
                            lat_dec = float(part.split('纬度:')[1].strip())
                    
                    if lat_dec is not None and lon_dec is not None:
                        structured_exif['gps']['latitude'] = lat_dec
                        structured_exif['gps']['longitude'] = lon_dec
                        logging.info(f'从ImageDescription成功提取GPS信息: 纬度={lat_dec}, 经度={lon_dec}')
                except Exception as e:
                    logging.error(f'从ImageDescription解析GPS信息失败: {str(e)}')
            
            # 如果ImageDescription中没有提取到有效的经纬度，则尝试从GPS标签中提取
            if lat_dec is None and gps_lat:
                try:
                    # 首先检查是否为度分秒格式 [度, 分, 秒]
                    if gps_lat.startswith('[') and gps_lat.endswith(']'):
                        # 解析格式如"[30, 16, 912/25]"的字符串
                        lat_parts = gps_lat.strip('[]').split(', ')
                        if len(lat_parts) == 3:
                            degrees = float(lat_parts[0])
                            minutes = float(lat_parts[1])
                            # 处理分数形式，如"912/25"
                            seconds_part = lat_parts[2]
                            if '/' in seconds_part:
                                num, denom = seconds_part.split('/')
                                seconds = float(num) / float(denom)
                            else:
                                seconds = float(seconds_part)
                                
                            lat_dec = degrees + minutes/60 + seconds/3600
                    else:
                        # 尝试直接解析为十进制度数
                        lat_dec = float(gps_lat)
                    
                    # 应用南北半球参考
                    if lat_dec is not None and gps_lat_ref == 'S':
                        lat_dec = -lat_dec
                    
                    if lat_dec is not None:
                        structured_exif['gps']['latitude'] = lat_dec
                        logging.info(f'从GPS标签成功提取纬度信息: {lat_dec}')
                except Exception as e:
                    logging.error(f'解析GPS纬度信息失败: {str(e)}')
            
            # 处理经度信息
            if lon_dec is None and gps_lon:
                try:
                    # 首先检查是否为度分秒格式 [度, 分, 秒]
                    if gps_lon.startswith('[') and gps_lon.endswith(']'):
                        # 解析格式如"[120, 8, 186/25]"的字符串
                        lon_parts = gps_lon.strip('[]').split(', ')
                        if len(lon_parts) == 3:
                            degrees = float(lon_parts[0])
                            minutes = float(lon_parts[1])
                            # 处理分数形式，如"186/25"
                            seconds_part = lon_parts[2]
                            if '/' in seconds_part:
                                num, denom = seconds_part.split('/')
                                seconds = float(num) / float(denom)
                            else:
                                seconds = float(seconds_part)
                                
                            lon_dec = degrees + minutes/60 + seconds/3600
                    else:
                        # 尝试直接解析为十进制度数
                        lon_dec = float(gps_lon)
                    
                    # 应用东西半球参考
                    if lon_dec is not None and gps_lon_ref == 'W':
                        lon_dec = -lon_dec
                    
                    if lon_dec is not None:
                        structured_exif['gps']['longitude'] = lon_dec
                        logging.info(f'从GPS标签成功提取经度信息: {lon_dec}')
                except Exception as e:
                    logging.error(f'解析GPS经度信息失败: {str(e)}')
                    
            # 处理海拔信息
            if gps_alt:
                try:
                    if '/' in gps_alt:
                        num, denom = gps_alt.split('/')
                        if float(denom) != 0:
                            structured_exif['gps']['altitude'] = float(num) / float(denom)
                    else:
                        structured_exif['gps']['altitude'] = float(gps_alt)
                except Exception as e:
                    logging.error(f'解析GPS海拔信息失败: {str(e)}')
            
            return make_response(data={
                'exif': structured_exif,
                'raw_exif': exif_data
            })
            
        except Exception as e:
            logging.error(f'提取EXIF信息失败: {str(e)}')
            return make_response(code=400, message=f'提取EXIF信息失败: {str(e)}')
        
    except Exception as e:
        logging.error(f'获取图片EXIF信息失败: {str(e)}')
        return make_response(code=500, message=f'获取图片EXIF信息失败: {str(e)}')