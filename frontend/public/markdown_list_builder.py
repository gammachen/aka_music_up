#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse

def build_markdown_list(directory_path):
    """
    从指定目录读取所有一级子目录和其中的.md文件，并按照指定格式构建层次化列表
    
    Args:
        directory_path: 目录路径
        
    Returns:
        格式化的列表字符串列表
    """
    # 检查目录是否存在
    if not os.path.isdir(directory_path):
        return []
    
    # 初始化结果列表，首先添加Home链接
    result = ["* [Home](/)"]
    
    # 获取目录中的所有子目录，排除隐藏目录和特定目录
    exclude_dirs = {'.git', '.vscode', '.history', 'node_modules', '__pycache__'}
    
    # 获取一级子目录列表
    subdirs = []
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if (os.path.isdir(item_path) 
            and not item.startswith('.') 
            and item not in exclude_dirs):
            subdirs.append(item)
    
    # 去重并排序子目录
    subdirs = sorted(list(set(subdirs)))
    
    # 处理每个子目录
    for subdir in subdirs:
        # 添加子目录链接（指向README）
        result.append(f"    * [{subdir}](/{subdir}/README)")
        
        # 获取子目录中的所有.md文件
        subdir_path = os.path.join(directory_path, subdir)
        md_files = []
        
        try:
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                if (os.path.isfile(file_path) 
                    and file.endswith('.md')):
                    # 提取文件名（不含扩展名）
                    filename = os.path.splitext(file)[0]
                    if filename != 'README':
                        md_files.append(filename)
        except (PermissionError, FileNotFoundError):
            # 如果无法访问子目录，则跳过
            continue
        
        # 去重并排序文件
        md_files = sorted(list(set(md_files)))
        
        # 处理子目录中的每个.md文件
        for filename in md_files:
            # 创建格式化的列表项，增加缩进
            result.append(f"        * [{filename}](/{subdir}/{filename})")
    
    return result

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='构建层次化Markdown文件列表')
    parser.add_argument('directory', help='要处理的目录路径')
    parser.add_argument('-o', '--output', help='输出文件路径（如果不指定则输出到标准输出）')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.isdir(args.directory):
        sys.stderr.write(f"错误: 目录 '{args.directory}' 不存在\n")
        return 1
    
    # 构建层次化Markdown列表
    md_list = build_markdown_list(args.directory)
    
    # 如果没有生成任何列表项
    if len(md_list) <= 1:  # 只有Home链接
        sys.stderr.write(f"警告: 在目录 '{args.directory}' 中未找到任何子目录或Markdown文件\n")
        return 0
    
    # 将结果写入文件或输出到标准输出
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_list))
    else:
        for item in md_list:
            print(item)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())