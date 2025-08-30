#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

def build_md_list(directory_path):
    """
    从指定目录读取所有.md文件，并构建格式化的列表
    
    Args:
        directory_path: 目录路径
        
    Returns:
        格式化的列表字符串
    """
    # 检查目录是否存在
    if not os.path.isdir(directory_path):
        sys.stderr.write(f"错误: 目录 '{directory_path}' 不存在\n")
        return ""
    
    # 获取目录中的所有.md文件
    md_files = []
    for file in os.listdir(directory_path):
        if file.endswith('.md'):
            md_files.append(file)
    
    # 如果没有找到.md文件
    if not md_files:
        sys.stderr.write(f"警告: 在目录 '{directory_path}' 中未找到.md文件\n")
        return ""
    
    # 构建格式化列表
    formatted_list = []
    dir_name = os.path.basename(directory_path)
    
    for file in sorted(md_files):
        # 提取文件名（不含扩展名）
        filename = os.path.splitext(file)[0]
        # 构建相对路径
        rel_path = f"{dir_name}/{filename}"
        # 创建格式化的列表项
        list_item = f"* [{filename}]({rel_path})"
        formatted_list.append(list_item)
    
    # 返回格式化的列表字符串
    return "\n".join(formatted_list)

def main():
    # 检查命令行参数
    if len(sys.argv) != 2:
        sys.stderr.write("用法: python md_list_builder.py <目录路径>\n")
        sys.exit(1)
    
    # 获取目录路径
    directory_path = sys.argv[1]
    
    # 构建并打印格式化列表
    formatted_list = build_md_list(directory_path)
    if formatted_list:
        sys.stdout.write(formatted_list + "\n")

if __name__ == "__main__":
    main()