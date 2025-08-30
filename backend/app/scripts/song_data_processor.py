import json
import random
import re
from pathlib import Path
from datetime import datetime

class SongDataProcessor:
    def __init__(self):
        self.cover_images = [
            '/static/def/a1.png',
            '/static/def/a2.png',
            '/static/def/a3.png',
            '/static/def/a4.png',
            '/static/def/a5.png',
            '/static/def/a6.png',
            '/static/def/a7.png',
            '/static/def/a8.png'
        ]

    def format_plays(self, number):
        """将数字格式化为千位符格式"""
        return f"{number/1000:.1f}K"

    def parse_song_info(self, key):
        """从歌曲key中解析出id、title和artist信息"""
        # 匹配开头的数字
        id_match = re.match(r'^(\d+)', key)
        if not id_match:
            return None

        song_id = id_match.group(1)
        
        # 移除开头的数字和分隔符
        remaining = re.sub(r'^\d+[.-]\s*', '', key)
        
        # 分割歌手和歌名
        parts = remaining.split('-', 1)
        if len(parts) < 2:
            return None

        artist = parts[0].strip()
        title = f"{key}[FLAC/MP3-320K]"

        return {
            'id': int(song_id),
            'title': title,
            'artist': artist
        }

    def process_song_mapping(self, mapping_file):
        """处理歌曲映射文件，生成前端所需的数据格式"""
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                song_mapping = json.load(f)
        except Exception as e:
            print(f"读取文件失败: {e}")
            return []

        # 从文件名中提取数字编号
        file_name = Path(mapping_file).name
        file_number = file_name.split('_')[0]

        songs_data = []
        for key, value in song_mapping.items():
            song_info = self.parse_song_info(key)
            if not song_info:
                continue

            # 构建URL
            url = f"static/videos/{file_number}{value}"

            # 生成随机播放次数
            plays = random.randint(10000, 20000)

            songs_data.append({
                'id': song_info['id'],
                'title': song_info['title'],
                'artist': song_info['artist'],
                'coverUrl': random.choice(self.cover_images),
                'plays': self.format_plays(plays),
                'url': url
            })

        return songs_data

def main():
    processor = SongDataProcessor()
    mapping_file = Path(__file__).parent.parent / 'resource' / '4_song_mapping.json'
    songs_data = processor.process_song_mapping(str(mapping_file))
    
    # 生成带日期前缀的输出文件名
    current_date = datetime.now().strftime('%Y%m%d')
    output_filename = f"{current_date}_processed_songs.json"
    output_file = Path(__file__).parent.parent / 'resource' / output_filename
    
    # 将结果保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(songs_data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，共处理 {len(songs_data)} 首歌曲")

if __name__ == '__main__':
    main()