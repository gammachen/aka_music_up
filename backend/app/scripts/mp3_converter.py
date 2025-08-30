import os
import re
import json
import time
import random
from pathlib import Path
import ffmpeg

class MP3Converter:
    def __init__(self, source_dir):
        self.source_dir = Path(source_dir)
        self.mapping = {}
    
    def get_number_prefix(self, filename):
        """从文件名中提取数字前缀"""
        match = re.match(r'^(\d+)[.-_]', filename)
        return match.group(1) if match else None
    
    def create_output_dir(self, number_prefix):
        """创建输出目录"""
        output_dir = self.source_dir / str(number_prefix)
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    def generate_output_filename(self):
        """生成基于时间戳和随机数的输出文件名"""
        timestamp = int(time.time())
        random_num = random.randint(1000, 9999)
        return f"{timestamp}_{random_num}"
    
    def convert_mp3_to_m3u8(self, input_path, output_dir):
        """将MP3文件转换为M3U8格式"""
        output_filename = self.generate_output_filename()
        output_path = output_dir / f"{output_filename}.m3u8"
        
        try:
            # 使用ffmpeg转换MP3为M3U8
            stream = ffmpeg.input(str(input_path))
            stream = ffmpeg.output(stream, str(output_path), format='hls', hls_time=10)
            ffmpeg.run(stream, overwrite_output=True)
            
            return output_path
        except ffmpeg.Error as e:
            print(f"转换失败: {input_path}\n{e.stderr}")
            return None
    
    def process_files(self):
        """处理所有MP3文件"""
        for file_path in self.source_dir.glob('*.mp3'):
            if not file_path.is_file():
                continue
                
            filename = file_path.name
            number_prefix = self.get_number_prefix(filename)
            
            if not number_prefix:
                print(f"跳过文件 {filename} - 未找到数字前缀")
                continue
            
            output_dir = self.create_output_dir(number_prefix)
            m3u8_path = self.convert_mp3_to_m3u8(file_path, output_dir)
            
            if m3u8_path:
                # 将路径转换为相对路径
                relative_path = str(m3u8_path.relative_to(self.source_dir))
                song_name = file_path.stem  # 文件名（不含扩展名）
                self.mapping[song_name] = f"/{relative_path}"
    
    def save_mapping(self):
        """保存映射关系到JSON文件"""
        output_file = self.source_dir / 'song_mapping.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.mapping, f, ensure_ascii=False, indent=2)

def main():
    # 这里设置源目录路径
    # source_dir = "/Volumes/toshiba/A/04.经典粤语歌曲(200首)"
    
    # converter = MP3Converter(source_dir)
    # converter.process_files()
    # converter.save_mapping()
    # print("转换完成！映射关系已保存到song_mapping.json")
    
    # source_dir = "/Users/shhaofu/Code/cursor-projects/aka_music/backend/app/static/videos/06.一人一首成名曲(160首)"
    
    # converter = MP3Converter(source_dir)
    # converter.process_files()
    # converter.save_mapping()
    # print("转换完成！映射关系已保存到song_mapping.json")
    
    # source_dir = "/Users/shhaofu/Code/cursor-projects/aka_music/backend/app/static/videos/16.8090后回忆录歌曲(200首)"
    
    # converter = MP3Converter(source_dir)
    # converter.process_files()
    # converter.save_mapping()
    # print("转换完成！映射关系已保存到song_mapping.json")
    
    source_dir = "/Users/shhaofu/Code/cursor-projects/aka_music/backend/app/static/videos/03.经典流行环绕歌曲(200首)"
    
    converter = MP3Converter(source_dir)
    converter.process_files()
    converter.save_mapping()
    print("转换完成！映射关系已保存到song_mapping.json")
    
    

if __name__ == '__main__':
    main()