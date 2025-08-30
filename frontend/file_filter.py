import os

root_dir = "public"
output_file = "md_files.txt"

with open(output_file, "w") as f:
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".md"):
                # 构建相对路径
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, root_dir)
                # 转换为 POSIX 风格的路径
                posix_path = relative_path.replace(os.sep, "/")
                f.write(f"public/{posix_path}\n")

print(f"MD 文件列表已生成到 {output_file}")