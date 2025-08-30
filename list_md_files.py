#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

def main():
    # 检查命令行参数
    if len(sys.argv) != 2:
        sys.stderr.write("用法: python list_md_files.py <目录路径>\n")
        return 1
    
    # 获取目录路径
    directory_path = sys.argv[1]
    
    # 检查目录是否存在
    if not os.path.isdir(directory_path):
        sys.stderr.write(f"错误: 目录 '{directory_path}' 不存在\n")
        return 1
    
    # 获取目录中的所有.md文件
    md_files = [f for f in os.listdir(directory_path) if f.endswith('.md')]
    
    # 如果没有找到.md文件
    if not md_files:
        sys.stderr.write(f"警告: 在目录 '{directory_path}' 中未找到.md文件\n")
        return 0
    
    # 获取目录名
    dir_name = os.path.basename(directory_path)
    
    # 构建并输出格式化列表
    for file in sorted(md_files):
        # 提取文件名（不含扩展名）
        filename = os.path.splitext(file)[0]
        # 构建相对路径
        rel_path = f"{dir_name}/{filename}"
        # 输出格式化的列表项
        print(f"* [{filename}]({rel_path})")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())