from pathlib import Path
import sys


# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

def main():
    # 调用uitls中的pdf_utils.py中的convert_mobi_file_to_pdf函数
    from app.utils.pdf_utils import mobi_to_pdf
    # 定义相关目录路径
    mobi_to_pdf("/Volumes/toshiba/《杀戮都市》高清漫画[mobi]/[奥浩哉][殺戮都市GANTZ]第01卷.mobi", "/Users/shhaofu/Code/cursor-projects/aka_music/backend/app/static/comic/杀戮都市")

if __name__ == '__main__':
    main()