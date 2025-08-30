import os
import numpy as np
import face_recognition
import torch
from PIL import Image
from io import BytesIO
import requests
import json
from datetime import datetime
from sqlalchemy import text
from typing import List, Dict, Any, Tuple, Optional

# 修改导入语句，使用相对导入
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app import create_app
from app.models.face_image import FaceImage, db
from app.models.user import User

import psycopg2

# 导入CLIP模型
from transformers import CLIPProcessor, CLIPModel

class FaceRecognitionService:
    def __init__(self):
        # 初始化CLIP模型
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)
        
    def extract_face_features(self, image_path: str, save_metadata: bool = True, save_to_db: bool = True) -> Optional[Dict[str, Any]]:
        """
        从图片中提取人脸特征并保存人脸图片到本地，可选择保存到数据库
        
        Args:
            image_path: 图片路径（本地路径或URL）
            save_metadata: 是否保存图片元数据
            save_to_db: 是否将特征向量保存到数据库
            
        Returns:
            包含人脸信息的字典或None（如果未检测到人脸）
        """
        print(f"[DEBUG] 开始提取人脸特征，图片路径: {image_path}, 保存元数据: {save_metadata}")
        try:
            # 加载图片
            print(f"[DEBUG] 开始加载图片: {image_path}")
            if image_path.startswith('http'):
                response = requests.get(image_path)
                img = Image.open(BytesIO(response.content))
                # 保存到本地临时文件
                local_path = f"app/static/uploads/faces/{os.path.basename(image_path)}"
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                img.save(local_path)
                image_path = local_path
                print(f"[DEBUG] 成功从URL加载图片并保存到本地: {local_path}")
            else:
                img = Image.open(image_path)
                print(f"[DEBUG] 成功从本地路径加载图片: {image_path}")
            
            # 转换为RGB模式（处理RGBA等格式）
            print(f"[DEBUG] 图片模式: {img.mode}, 尺寸: {img.size}")
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print(f"[DEBUG] 已将图片转换为RGB模式")
            
            # 使用face_recognition库检测人脸
            print(f"[DEBUG] 开始检测人脸")
            img_array = np.array(img)
            face_locations = face_recognition.face_locations(img_array)
            print(f"[DEBUG] 人脸检测结果: 检测到{len(face_locations)}个人脸")
            
            if not face_locations:
                print(f"[ERROR] 未在图片中检测到人脸: {image_path}")
                return None
            
            # 使用第一个检测到的人脸（可以扩展为处理多个人脸）
            face_location = face_locations[0]
            print(f"[DEBUG] 使用第一个检测到的人脸，位置: {face_location}")
            
            # 裁剪人脸区域
            top, right, bottom, left = face_location
            print(f"[DEBUG] 裁剪人脸区域: top={top}, right={right}, bottom={bottom}, left={left}")
            face_image = img.crop((left, top, right, bottom))
            print(f"[DEBUG] 裁剪后的人脸图像尺寸: {face_image.size}")
            
            # 使用CLIP模型提取特征向量
            print(f"[DEBUG] 开始使用CLIP模型提取特征向量，设备: {self.device}")
            inputs = self.clip_processor(images=face_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            print(f"[DEBUG] 特征向量提取完成，形状: {image_features.shape}")
            
            # 将特征向量转换为Python列表
            feature_vector = image_features[0].cpu().numpy().tolist()
            print(f"[DEBUG] 特征向量转换为列表完成，长度: {len(feature_vector)}")
            
            # 获取图片元数据（如EXIF信息）
            metadata = None
            if save_metadata:
                print(f"[DEBUG] 开始提取图片元数据")
                try:
                    from PIL.ExifTags import TAGS
                    exif_data = {}
                    exif = img._getexif()
                    if exif:
                        for tag_id, value in exif.items():
                            tag = TAGS.get(tag_id, tag_id)
                            # 处理二进制数据和复杂对象
                            if isinstance(value, bytes):
                                value = value.decode('utf-8', 'ignore')
                            try:
                                json.dumps({tag: str(value)})
                                exif_data[tag] = str(value)
                            except:
                                exif_data[tag] = "[无法序列化的数据]"
                    
                    metadata = {
                        "exif": exif_data,
                        "dimensions": {
                            "width": img.width,
                            "height": img.height
                        },
                        "format": img.format,
                        "mode": img.mode
                    }
                except Exception as e:
                    print(f"[ERROR] 提取元数据失败: {str(e)}")
            
            # 保存裁剪后的人脸图片到本地
            print(f"[DEBUG] 开始保存人脸图片到本地")
            # 生成唯一的文件名
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            face_filename = f"face_{timestamp}.jpg"
            # TODO 暂时就先存储到faces目录下，依赖执行python的路径，后续需要优化
            # face_dir = "faces"
            face_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "beauty", "人脸")
            os.makedirs(face_dir, exist_ok=True)
            face_path = os.path.join(face_dir, face_filename)
            
            # 保存人脸图片
            face_image.save(face_path)
            print(f"[DEBUG] 人脸图片已保存到: {face_path}")
            
            # 返回包含人脸信息的字典
            result = {
                "image_path": image_path[image_path.find('/static'):] if '/static' in image_path else image_path,
                "face_path": f"/static/beauty/人脸/{face_filename}",# face_path,
                "face_location": list(face_location),
                "feature_vector": feature_vector,
                "metadata": metadata
            }
            
            # 如果需要，将特征向量保存到数据库
            if save_to_db:
                # 尝试使用原生psycopg2连接保存
                try:
                    save_image_vector_to_pgvector(
                        image_path[image_path.find('/static'):] if '/static' in image_path else image_path,
                        result['face_path'],
                        list(face_location),
                        feature_vector,
                        metadata
                    )
                    print(f"[DEBUG] 使用原生psycopg2连接保存特征向量成功")
                except Exception as e2:
                    print(f"[ERROR] 使用原生psycopg2连接保存特征向量失败: {str(e2)}")
            
            print(f"[DEBUG] 返回人脸信息字典")
            return result
            
        except Exception as e:
            print(f"[ERROR] 处理图片时出错: {str(e)}")
            import traceback
            print(f"[ERROR] 错误详情: {traceback.format_exc()}")
            db.session.rollback()
            return None
    
    def search_similar_faces(self, query_image_path: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        搜索与给定图片中人脸相似的图片
        
        Args:
            query_image_path: 查询图片路径
            limit: 返回结果数量限制
            
        Returns:
            相似人脸图片列表，按相似度降序排序
        """
        # 提取查询图片的人脸特征，但不保存到数据库
        query_face = self.extract_face_features(query_image_path, save_metadata=False, save_to_db=False)
        
        if not query_face:
            return {}
        
        # 使用PostgreSQL的向量操作符进行相似度搜索
        # 注意：这需要在PostgreSQL中启用pgvector扩展
        try:
            # 将特征向量转换为PostgreSQL数组格式的字符串
            vector_str = str(query_face["feature_vector"]).replace('[', '{').replace(']', '}')
            
            # 使用余弦相似度进行搜索
            # 注意：这里使用原生SQL，因为SQLAlchemy对pgvector的支持有限
            sql = """
                SELECT 
                    id,
                    image_path,
                    face_path,
                    face_location,
                    image_metadata as metadata,
                    feature_vector <=> %s::vectors.vector AS cosine_distance,
                    1 - (feature_vector <=> %s::vectors.vector) AS cosine_similarity
                FROM face_images
                ORDER BY cosine_distance
                LIMIT 10
            """
            
            # 将特征向量转换为带单引号的字符串并添加类型转换
            # feature_str = ','.join(map(str, query_face['feature_vector']))
            # result = db.session.execute(sql, (f"'{feature_str}'", limit))
            
            feature_vector = query_face['feature_vector']
            vector_str = '[' + ','.join(str(x) for x in feature_vector) + ']'
            print(f"[DEBUG] 查询向量字符串: {vector_str}")
            
            # result = db.session.execute(sql, {'vector_str': vector_str, 'limit': limit})
            
            # 从Flask应用配置中获取PostgreSQL连接URI
            from flask import current_app
            postgres_uri = current_app.config.get('POSTGRES_DATABASE_URI')
            if not postgres_uri:
                print("[DEBUG] 未找到配置的数据库URI，使用默认连接参数")
                postgres_uri = 'postgresql://postgres:postgres468028475@localhost:5432/immich'
            print(f"[DEBUG] 使用数据库连接URI: {postgres_uri}")
            
            # 连接到PostgreSQL数据库
            print("[DEBUG] 尝试连接数据库...")
            conn = psycopg2.connect(postgres_uri)
            cursor = conn.cursor()
            print("[DEBUG] 数据库连接成功")
            
            # 打印调试信息
            print(f"[DEBUG] 完整SQL语句: {sql}")
            print(f"[DEBUG] 参数类型: vector_str类型={type(vector_str)}, 值={vector_str}")
            
            # 修复参数数量（需要两个vector参数和一个limit）
            # 使用mogrify获取真实SQL
            raw_sql = cursor.mogrify(sql, (vector_str, vector_str))
            
            # 对长向量参数进行截断处理
            truncated_vector = (vector_str[:50] + '...' + vector_str[-50:]) if len(vector_str) > 100 else vector_str
            
            print(f"[DEBUG] 完整SQL模板:\n{raw_sql.decode('utf-8')}")
            print(f"[DEBUG] 参数类型: {type(vector_str)}, 截断参数值: {truncated_vector}")
            
            # cursor.execute(sql, (vector_str, vector_str))
            cursor.execute(raw_sql)
            
            # 获取并打印查询结果行数
            print(f"[DEBUG] 成功获取 {cursor.rowcount} 条查询结果")
            
            # 获取查询结果
            rows = cursor.fetchall()
            if not rows:
                print("[DEBUG] 未找到匹配结果")
                return []
            
            # 构建结果列表
            similar_faces = []
            # 打印前5条查询结果
            print("\n[DEBUG] SQL查询结果：")
            for idx, row in enumerate(rows[:5]):
                print(f"记录 #{idx+1}:")
                print(f"  - ID: {row[0]}")
                print(f"  - 原始图片路径: {row[1]}")
                print(f"  - 人脸图片路径: {row[2]}")
                print(f"  - 人脸位置: {row[3]}")
                print(f"  - 元数据: {str(row[4])}")
                print(f"  - 余弦距离: {row[5]:.4f}")
                print(f"  - 余弦相似度: {row[6]:.4f}")
                print("-"*50)
            
            for row in rows:
                similar_faces.append({
                    "image_path": row[1],
                    "face_path": row[2],
                    "face_location": row[3],
                    "cosine_distance": float(row[5]),
                    "cosine_similarity": float(row[6])
                })
            
            return similar_faces
            
        except Exception as e:
            print(f"搜索相似人脸时出错: {str(e)}")
            return []
    
    def associate_face_with_user(self, face_image_id: int, user_id: int) -> bool:
        """
        将人脸图片与用户关联
        
        Args:
            face_image_id: 人脸图片ID
            user_id: 用户ID
            
        Returns:
            操作是否成功
        """
        try:
            face_img = FaceImage.query.get(face_image_id)
            user = User.query.get(user_id)
            
            if not face_img or not user:
                return False
            
            face_img.user_id = user_id
            db.session.commit()
            return True
            
        except Exception as e:
            print(f"关联人脸与用户时出错: {str(e)}")
            db.session.rollback()
            return False
    
    def get_user_by_face(self, image_path: str, similarity_threshold: float = 0.8) -> Optional[Dict[str, Any]]:
        """
        通过人脸识别获取用户信息
        
        Args:
            image_path: 图片路径
            similarity_threshold: 相似度阈值，超过此值才认为是同一个人
            
        Returns:
            用户信息字典或None（如果未找到匹配用户）
        """
        # 提取查询图片的人脸特征
        query_face = self.extract_face_features(image_path, save_metadata=False)
        
        if not query_face:
            return None
        
        # 搜索相似人脸
        similar_faces = self.search_similar_faces(image_path, limit=5)
        
        # 查找关联了用户且相似度超过阈值的人脸
        for face in similar_faces:
            if face["similarity"] >= similarity_threshold:
                face_img = FaceImage.query.get(face["id"])
                if face_img and face_img.user_id:
                    user = User.query.get(face_img.user_id)
                    if user:
                        return user.to_dict()
        
        return None
    
# 写一个简单的测试代码，调用extract_face_features函数，并验证返回值的类型和内容是否正确。
def test_extract_face_features():
    """
    测试extract_face_features函数
    """
    image_path = 'girl-4.jpg'
    face_service = FaceRecognitionService()
    
    # 提取人脸特征并自动保存到数据库
    print("\n[TEST] 开始提取人脸特征并保存到数据库...")
    result = face_service.extract_face_features(image_path, save_to_db=True)
    
    if not result:
        print("[TEST] 错误：未能提取人脸特征")
        return
    
    print("[TEST] 人脸特征提取成功：")
    print(f"  - 图片路径: {result['image_path']}")
    print(f"  - 人脸路径: {result['face_path']}")
    print(f"  - 人脸位置: {result['face_location']}")
    print(f"  - 特征向量长度: {len(result['feature_vector'])}")
    
    # 测试手动保存特征向量到pgvector数据库
    print("\n[TEST] 开始手动保存特征向量到pgvector数据库...")
    try:
        save_image_vector_to_pgvector(
            result['image_path'],
            result['face_path'],
            result['face_location'],
            result['feature_vector'],
            result['metadata']
        )
        print("[TEST] 特征向量手动保存成功")
    except Exception as e:
        print(f"[TEST] 错误：手动保存特征向量失败 - {str(e)}")
    
    # 测试相似人脸搜索
    print("\n[TEST] 开始测试相似人脸搜索...")
    try:
        similar_faces = face_service.search_similar_faces(image_path, limit=5)
        print(f"[TEST] 找到 {len(similar_faces)} 个相似人脸")
        
        # 打印相似人脸信息
        for i, face in enumerate(similar_faces):
            print(f"\n相似人脸 #{i+1}:")
            print(f"  - 图片路径: {face.get('image_path', 'N/A')}")
            print(f"  - 相似度: {face.get('similarity', 0):.4f}")
    except Exception as e:
        print(f"[TEST] 错误：相似人脸搜索失败 - {str(e)}")
    
    print("\n[TEST] 测试完成")


def save_image_vector_to_pgvector(image_path: str, face_path: str, face_location: List[int], feature_vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    将图片的向量保存到pgvector数据库中
    
    Args:
        image_path: 原始图片路径
        face_path: 人脸图片路径
        face_location: 人脸在原图中的位置 [top, right, bottom, left]
        feature_vector: 人脸特征向量
        metadata: 图片元数据
    """
    try:
        print("[DEBUG] 开始保存向量到pgvector数据库")
        
        # 从Flask应用配置中获取PostgreSQL连接URI
        from flask import current_app
        postgres_uri = current_app.config.get('POSTGRES_DATABASE_URI')
        if not postgres_uri:
            print("[DEBUG] 未找到配置的数据库URI，使用默认连接参数")
            postgres_uri = 'postgresql://postgres:postgres468028475@localhost:5432/immich'
        print(f"[DEBUG] 使用数据库连接URI: {postgres_uri}")
        
        # 连接到PostgreSQL数据库
        print("[DEBUG] 尝试连接数据库...")
        conn = psycopg2.connect(postgres_uri)
        cursor = conn.cursor()
        print("[DEBUG] 数据库连接成功")
        
        # 创建pgvector扩展
        print("[DEBUG] 尝试创建pgvector扩展...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("[DEBUG] pgvector扩展创建/确认完成")
        
        # 创建表
        print("[DEBUG] 尝试创建face_images表...")
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS face_images (
                id SERIAL PRIMARY KEY,
                image_path TEXT,
                face_path TEXT,
                face_location INTEGER[],
                feature_vector vector(512),
                image_metadata JSONB
            );
        """
        cursor.execute(create_table_sql)
        print("[DEBUG] face_images表创建/确认完成")
        
        # 验证表是否存在
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'face_images'
            );
        """)
        table_exists = cursor.fetchone()[0]
        print(f"[DEBUG] face_images表存在状态: {table_exists}")
        
        # 插入数据
        print("[DEBUG] 准备插入数据...")
        print(f"[DEBUG] image_path: {image_path}")
        print(f"[DEBUG] face_path: {face_path}")
        print(f"[DEBUG] face_location长度: {len(face_location)}")
        print(f"[DEBUG] feature_vector长度: {len(feature_vector)}")
        
        # 将feature_vector转换为PostgreSQL vector类型所需的格式
        # 使用SQL参数占位符，并在执行时传入字符串形式的向量
        insert_sql = """
            INSERT INTO face_images (image_path, face_path, face_location, feature_vector, image_metadata)
            VALUES (%s, %s, %s, %s, %s)
        """
        
        # 打印SQL语句前缀（调试用）
        print(f"[DEBUG] SQL语句前缀: {insert_sql[:1000]}...")
        
        # 将feature_vector转换为字符串格式，例如'[0.1,0.2,...]'
        vector_str = '[' + ','.join(str(x) for x in feature_vector) + ']'
        
        cursor.execute(insert_sql, (
            image_path, 
            face_path, 
            face_location, 
            vector_str,  # 传递字符串形式的向量
            json.dumps(metadata) if metadata else None
        ))
        
        # 验证插入是否成功
        cursor.execute("SELECT COUNT(*) FROM face_images;")
        row_count = cursor.fetchone()[0]
        print(f"[DEBUG] 当前表中总行数: {row_count}")
        
        conn.commit()
        print("[DEBUG] 数据已成功提交到数据库")
        
        conn.close()
        print("[DEBUG] 数据库连接已关闭")
        
    except Exception as e:
        print(f"[ERROR] 保存数据到pgvector数据库时出错: {str(e)}")
        print("[ERROR] 错误详细信息:")
        import traceback
        print(traceback.format_exc())
        raise  # 重新抛出异常，确保调用者知道操作失败
    
# main()
if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        test_extract_face_features()