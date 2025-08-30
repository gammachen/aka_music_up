import os
import json
from pathlib import Path
import hashlib

class ImageScanner:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.image_extensions = {
            '.jpg', '.jpeg', '.png', '.gif', '.bmp',
            '.webp', '.tiff', '.svg', '.ico'
        }
        self.result = {
            'dir_name': '',
            'children': []
        }
    
    def is_image_file(self, file_path):
        """检查文件是否为图片文件"""
        return file_path.suffix.lower() in self.image_extensions
    
    def scan_directory(self):
        """扫描目录及其子目录下的所有图片文件，构建树形结构"""
        self.result = self._scan_directory_recursive(self.root_dir)
    
    def _scan_directory_recursive(self, current_dir):
        """递归扫描目录，返回树形结构的数据"""
        # 使用 MD5 对目录路径进行哈希
        md5_hash = hashlib.md5(str(current_dir.name).encode()).hexdigest()
        
        result = {
            'dir_name': current_dir.name if current_dir != self.root_dir else '',
            'children': [],
            # 'refer_id': str(hash(str(current_dir.absolute())))
            'refer_id': md5_hash
        }
        
        try:
            # 获取当前目录下的所有文件和子目录
            items = list(current_dir.iterdir())
            
            # 分别处理文件和目录
            for item in sorted(items):
                if item.is_file():   # and self.is_image_file(item): # 不加是否图片的判断，可能有音视频等内容
                    # 如果是图片文件，添加到当前目录的children中
                    # 使用 MD5 对目录路径进行哈希
                    s_md5_hash = hashlib.md5(str(item.name).encode()).hexdigest()
                    result['children'].append({
                        'type': 'file',
                        'name': item.name,
                        # 'refer_id': str(hash(str(item.absolute())))
                        'refer_id': s_md5_hash
                    })
                elif item.is_dir():
                    # 如果是目录，递归处理
                    sub_dir_result = self._scan_directory_recursive(item)
                    if sub_dir_result['children']:  # 只添加包含图片的目录
                        result['children'].append({
                            'type': 'directory',
                            **sub_dir_result
                        })
        
        except PermissionError:
            print(f'警告：无法访问目录 {current_dir}')
        except Exception as e:
            print(f'警告：处理目录 {current_dir} 时发生错误：{str(e)}')
        
        return result
    
    def save_to_json(self, output_file):
        """将扫描结果保存为JSON文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.result, f, ensure_ascii=False, indent=2)

def main():
    # 获取用户输入的目录路径
    dir_path = input('请输入要扫描的目录路径: ')
    
    if not os.path.exists(dir_path):
        print('错误：指定的目录不存在！')
        return
    
    # 创建扫描器实例并执行扫描
    scanner = ImageScanner(dir_path)
    print('正在扫描目录...')
    scanner.scan_directory()
    
    # 生成输出文件路径到resource目录
    script_dir = Path(__file__).parent.parent
    resource_dir = script_dir / 'resource'
    output_file = resource_dir / 'image_mapping.json'
    scanner.save_to_json(output_file)
    
    # 计算统计信息
    def count_items(node):
        total_dirs = 0
        total_files = 0
        if 'type' in node and node['type'] == 'directory':
            total_dirs += 1
        elif 'type' in node and node['type'] == 'file':
            total_files += 1
        
        if 'children' in node:
            for child in node['children']:
                dirs, files = count_items(child)
                total_dirs += dirs
                total_files += files
        
        return total_dirs, total_files
    
    total_dirs, total_images = count_items(scanner.result)
    print(f'\n扫描完成！')
    print(f'发现 {total_dirs} 个包含图片的目录')
    print(f'共计 {total_images} 个图片文件')
    print(f'扫描结果已保存至: {output_file}')

if __name__ == '__main__':
    main()