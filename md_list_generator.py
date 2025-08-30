#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

def generate_md_list(directory_path, output_file=None):
    """
    从指定目录读取所有.md文件，并构建格式化的列表
    
    Args:
        directory_path: 目录路径
        output_file: 输出文件路径，如果为None则输出到标准输出
        
    Returns:
        成功返回True，失败返回False
    """
    try:
        # 检查目录是否存在
        if not os.path.isdir(directory_path):
            sys.stderr.write(f"错误: 目录 '{directory_path}' 不存在\n")
            return False
        
        # 获取目录中的所有.md文件
        md_files = [f for f in os.listdir(directory_path) if f.endswith('.md')]
        
        # 如果没有找到.md文件
        if not md_files:
            sys.stderr.write(f"警告: 在目录 '{directory_path}' 中未找到.md文件\n")
            return False
        
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
        
        # 将结果写入文件或输出到标准输出
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(result))
        else:
            for item in result:
                sys.stdout.write(item + '\n')
        
        return True
    except Exception as e:
        sys.stderr.write(f"错误: {str(e)}\n")
        return False

def main():
    # 检查命令行参数
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        sys.stderr.write("用法: python md_list_generator.py <目录路径> [输出文件路径]\n")
        return 1
    
    # 获取目录路径
    directory_path = sys.argv[1]
    
    # 获取输出文件路径（如果提供）
    output_file = sys.argv[2] if len(sys.argv) == 3 else None
    
    # 生成Markdown列表
    success = generate_md_list(directory_path, output_file)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())