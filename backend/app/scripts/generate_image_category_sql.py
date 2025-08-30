import json
import random
import colorsys
import hashlib

def generate_gradient_colors():
    """生成柔和的渐变色组合"""
    # 生成柔和的HSL颜色
    def generate_soft_color():
        hue = random.random()  # 随机色相
        saturation = random.uniform(0.3, 0.7)  # 适中的饱和度
        lightness = random.uniform(0.6, 0.8)  # 较高的亮度，使颜色更柔和
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
    
    color1 = generate_soft_color()
    color2 = generate_soft_color()
    return f'background: linear-gradient(45deg, {color1}, {color2})'

def process_directory(data, parent_id='id', level=2, order_counter=[0]):
    """递归处理目录结构并生成SQL语句"""
    sql_statements = []
    
    if isinstance(data, dict) and data.get('children'):
        for item in data['children']:
            if item.get('type') == 'directory':
                order_counter[0] += 1
                dir_name = item['dir_name']
                
                # 从children中随机选择一个图片的refer_id作为desc_image
                image_files = [child for child in item.get('children', []) 
                             if child.get('type') == 'file']
                
                # 随机选择一个图片作为desc_image
                random_image = random.choice(image_files) if image_files else None
                if random_image:
                    desc_image = f"'/static/category/{random_image['name']}'"
                else:
                    desc_image = "'/static/category/photography.jpg'"
                    
                if desc_image != "'/static/category/photography.jpg'":
                    # 将图片复制到static/category目录下
                    import shutil
                    import os
                    
                    # 确保目标目录存在
                    target_dir = 'backend/app/static/category'
                    os.makedirs(target_dir, exist_ok=True)
                    
                    # 获取原始图片路径和目标路径
                    source_path = os.path.join('backend/app/static/beauty', item['dir_name'], random_image['name'])
                    # target_path = os.path.join(target_dir, f"{random_image['refer_id']}.webp")
                    target_path = os.path.join(target_dir, f"{random_image['name']}")
                    
                    # 复制图片文件
                    shutil.copy(source_path, target_path)
                
                dir_name_md5_hash = hashlib.md5(str(dir_name).encode()).hexdigest()
                
                # 生成INSERT语句
                sql = f"""INSERT INTO category (name, parent_id, level, sort_order, background_style, desc_image, refer_id, prefix)\nSELECT \n    '{dir_name}', {parent_id}, {level}, {order_counter[0]}, '{generate_gradient_colors()}', {desc_image}, '{dir_name_md5_hash}', 'beaulist'\nFROM category WHERE name = '佳人';\n"""
                
                sql_statements.append(sql)
                
                # 递归处理子目录
                sql_statements.extend(process_directory(item, 
                                                     f"(SELECT id FROM category WHERE name = '{dir_name}')", 
                                                     level + 1, 
                                                     order_counter))
    
    return sql_statements

def main():
    # 读取image_mapping.json文件
    with open('backend/app/resource/image_mapping.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 生成SQL语句
    sql_statements = process_directory(data)
    
    # 将SQL语句写入文件
    with open('backend/app/scripts/category_init.sql', 'w', encoding='utf-8') as f:
        f.write('-- 初始化分类数据\n\n')
        f.write('-- 确保外键约束开启\nPRAGMA foreign_keys = ON;\n\n')
        for sql in sql_statements:
            f.write(sql + '\n')

if __name__ == '__main__':
    main()