#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse

def build_markdown_list(directory_path):
    """
    从指定目录读取所有.md文件，并按照指定格式构建列表
    
    Args:
        directory_path: 目录路径
        
    Returns:
        格式化的列表字符串列表
    """
    # 检查目录是否存在
    if not os.path.isdir(directory_path):
        return []
    
    # 获取目录中的所有.md文件
    md_files = [f for f in os.listdir(directory_path) if f.endswith('.md')]
    
    # 如果没有找到.md文件
    if not md_files:
        return []
    
    # 获取目录名
    dir_name = os.path.basename(directory_path)
    
    # 构建格式化列表
    result = []
    for file in sorted(md_files):
        # 提取文件名（不含扩展名）
        filename = os.path.splitext(file)[0]
        # 构建相对路径
        rel_path = f"{dir_name}/{filename}"
        # 创建格式化的列表项
        list_item = f"* [{filename}]({rel_path})"
        result.append(list_item)
    
    return result

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='构建Markdown文件列表')
    parser.add_argument('directory', help='要处理的目录路径')
    parser.add_argument('-o', '--output', help='输出文件路径（如果不指定则输出到标准输出）')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.isdir(args.directory):
        sys.stderr.write(f"错误: 目录 '{args.directory}' 不存在\n")
        return 1
    
    # 构建Markdown列表
    md_list = build_markdown_list(args.directory)
    
    # 如果没有找到.md文件
    if not md_list:
        sys.stderr.write(f"警告: 在目录 '{args.directory}' 中未找到.md文件\n")
        return 0
    
    # 将结果写入文件或输出到标准输出
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_list))
    else:
        for item in md_list:
            sys.stdout.write(f"{item}\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())