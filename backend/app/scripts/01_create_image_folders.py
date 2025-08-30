import os
import sys
from pathlib import Path
import hashlib

'''
构建图集的子目录
写真
商务
家庭
团队
学生
医疗
城市
写真
农田
建筑
动物
机械
科技
家居

'''
def create_folders(base_path, folder_names):
    """根据字符串数组创建文件夹"""
    created_folders = []
    base_path = Path(base_path)
    
    # 确保基础目录存在
    try:
        base_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f'创建基础目录 {base_path} 时出错: {str(e)}')
        return created_folders
    
    for folder_name in folder_names:
        # 使用MD5对文件夹名称进行哈希
        folder_hash = hashlib.md5(folder_name.encode()).hexdigest()
        
        # 创建完整的文件夹路径
        folder_path = base_path / folder_name
        try:
            folder_path.mkdir(exist_ok=True)
            created_folders.append({
                'name': folder_name,
                'path': str(folder_path),
                'hash': folder_hash
            })
            print(f'已创建文件夹: {folder_path}')
        except Exception as e:
            print(f'创建文件夹 {folder_path} 时出错: {str(e)}')
    
    return created_folders

def generate_sql_script(folder_names):
    """生成SQLite初始化脚本"""
    sql_script = """"""
    
    # 生成INSERT语句
    for fname in folder_names:
        dir_name_md5_hash = hashlib.md5(str(fname).encode()).hexdigest()
        sql_script += f"""update category set refer_id = '{dir_name_md5_hash}' where name = '{fname}';
"""
    
    return sql_script

def main():
    # 获取脚本所在目录
    script_dir = Path(__file__).resolve().parent
    app_dir = script_dir.parent
    
    # 设置基础路径
    base_path = app_dir / 'static' / 'beauty'
    
    # 如果提供了命令行参数，使用命令行参数
    if len(sys.argv) >= 3:
        base_path = Path(sys.argv[1])
        folder_names = sys.argv[2:]
    else:
        # 默认文件夹列表
        folder_names = [
            '写真',
            '家庭',
            '商务',
            '团队',
            '学生',
            '医疗',
            '农田',
            '建筑',
            '动物',
            '机械',
            '科技',
            '家居'
        ]
    
    # 创建文件夹
    print('\n开始创建文件夹...')
    created_folders = create_folders(base_path, folder_names)
    
    if not created_folders:
        print('没有成功创建任何文件夹，程序退出')
        return
    
    # 生成SQL脚本
    print('\n生成SQL初始化脚本...')
    sql_script = generate_sql_script(folder_names)
    
    # 保存SQL脚本到models目录
    models_dir = app_dir / 'models'
    sql_file_path = models_dir / 'image_folders_init.sql'
    
    try:
        # 确保models目录存在
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # 写入SQL脚本
        sql_file_path.write_text(sql_script, encoding='utf-8')
        print(f'\nSQL初始化脚本已保存至: {sql_file_path}')
        
    except Exception as e:
        print(f'保存SQL脚本时出错: {str(e)}')

if __name__ == '__main__':
    main()