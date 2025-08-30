import os
from pathlib import Path

def rename_mp3_files(directory):
    """重命名指定目录下的所有MP3文件，添加递增数字前缀"""
    # 将输入路径转换为Path对象
    dir_path = Path(directory)
    
    # 获取所有MP3文件并排序
    mp3_files = sorted([f for f in dir_path.glob('*.mp3') if f.is_file()])
    
    # 用于记录重命名结果
    renamed_files = []
    
    # 遍历所有MP3文件进行重命名
    for index, file_path in enumerate(mp3_files, start=1):
        # 构建新文件名：数字序号 + 中划线（x下划线） + 原文件名
        new_name = f"{index}-{file_path.name}"
        new_path = file_path.parent / new_name
        
        try:
            # 重命名文件
            file_path.rename(new_path)
            renamed_files.append((file_path.name, new_name))
            print(f"已重命名: {file_path.name} -> {new_name}")
        except Exception as e:
            print(f"重命名失败 {file_path.name}: {str(e)}")
    
    return renamed_files

def main():
    # 设置要处理的目录路径
    directory = input("请输入要处理的目录路径: ")
    
    if not os.path.exists(directory):
        print("错误：指定的目录不存在！")
        return
    
    # 执行重命名操作
    renamed_files = rename_mp3_files(directory)
    
    # 打印重命名结果统计
    print(f"\n重命名完成！共处理 {len(renamed_files)} 个文件。")

if __name__ == '__main__':
    main()