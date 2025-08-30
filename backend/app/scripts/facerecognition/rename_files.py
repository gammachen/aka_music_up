import os

def rename_files_in_directory(directory):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 构造文件的完整路径
        old_file_path = os.path.join(directory, filename)
        
        # 检查是否是文件
        if os.path.isfile(old_file_path):
            # 将文件名中的空格替换为下划线
            new_filename = filename.replace(" ", "_")
            new_file_path = os.path.join(directory, new_filename)
            
            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    # 指定要处理的目录
    target_directory = "/Users/shhaofu/Code/cursor-projects/aka_music/backend/app/scripts/facerecognition/yolo_detected_faces_persons"
    
    # 调用函数进行文件重命名
    rename_files_in_directory(target_directory)