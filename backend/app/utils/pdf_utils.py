import os
import fitz  # PyMuPDF

def convert_pdf_to_images(pdf_path, output_dir):
    """将PDF文件转换为图片
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录路径
        
    Returns:
        None
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开PDF文件
    pdf_document = fitz.open(pdf_path)
    
    # 遍历每一页
    for page_number in range(pdf_document.page_count):
        # 获取页面
        page = pdf_document[page_number]
        
        # 将页面转换为图片
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x缩放以提高质量
        
        # 构建输出图片路径
        image_path = os.path.join(output_dir, f'page_{page_number + 1:03d}.jpg')
        
        # 保存图片
        pix.save(image_path)
    
    # 关闭PDF文件
    pdf_document.close()
    
import subprocess
import os

def mobi_to_pdf(mobi_path, output_dir):
    # 获取文件名（不含扩展名）
    filename = os.path.splitext(os.path.basename(mobi_path))[0]
    pdf_path = os.path.join(output_dir, f"{filename}.pdf")

    # 调用 Calibre 命令行工具
    command = [
        "ebook-convert",  # 或替换为完整路径，如 "/Applications/calibre.app/Contents/MacOS/ebook-convert"
        mobi_path,
        pdf_path,
        "--output-profile", "tablet"  # 可选：优化 PDF 排版
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"转换成功: {pdf_path}")
    except subprocess.CalledProcessError as e:
        print(f"转换失败: {e}")

# 示例调用
# mobi_to_pdf("/Volumes/toshiba/《银魂》漫画 77卷全[mobi]/[Vol.moe][銀魂Gintama]第01卷.mobi", "/Users/shhaofu/Code/cursor-projects/aka_music/backend/app/static/comic/银魂")

if __name__ == "__main__":
    # mobi_to_pdf("/Volumes/toshiba/《银魂》漫画 77卷全[mobi]/[Vol.moe][銀魂Gintama]第01卷.mobi", "/Users/shhaofu/Code/cursor-projects/aka_music/backend/app/static/comic/银魂")
    convert_pdf_to_images("/Users/shhaofu/Code/cursor-projects/aka_music/frontend/public/ml/DeepLearning_all.pdf", "/Users/shhaofu/Code/cursor-projects/aka_music/frontend/public/ml/deeplearning_all_book_pics")
    '''
    ffmpeg -framerate 1 -f image2 -i "concat:page_047.jpg|page_048.jpg|page_049.jpg|page_050.jpg|page_051.jpg|page_052.jpg|page_053.jpg" -vf "scale=iw:-2,format=yuv420p" -c:v libx264 -movflags +faststart output.mp4
    ffmpeg -framerate 1 -pattern_type glob -i "concat:page_047.jpg|page_048.jpg|page_049.jpg|page_050.jpg|page_051.jpg|page_052.jpg|page_053.jpg" -vf "scale=iw:-2,format=yuv420p"  -c:v libx264 -movflags +faststart output.mp4
    ffmpeg -framerate 1 -i "concat:page_047.jpg|page_048.jpg|page_049.jpg|page_050.jpg|page_051.jpg|page_052.jpg|page_053.jpg" -vf "scale=iw:-2,format=yuv420p" -c:v libx264 -movflags +faststart output.mp4
    ffmpeg -framerate 1 -i "concat:page_048.jpg" -vf "scale=iw:-2,format=yuv420p" -c:v libx264 -movflags +faststart output.mp4
    ffmpeg -framerate 1 -i "concat:page_048.jpg|page_048.jpg|page_049.jpg|page_050.jpg|page_051.jpg|page_052.jpg|page_053.jpg" -vf "scale=iw:-2,format=yuv420p" -c:v libx264 -movflags +faststart output.mp4
    
    ffmpeg -f image2 -framerate 1 -i "concat:page_049.jpg|page_048.jpg|page_049.jpg|page_050.jpg|page_051.jpg|page_052.jpg|page_053.jpg" \
       -r 1 -vf "scale=iw:-2,format=yuv420p" \
       -c:v libx264 -movflags +faststart output.mp4
    
    1002  ffmpeg -framerate 15 -pattern_type glob -i "./*.png" -vf "scale=iw:-2,format=yuv420p"  -c:v libx264 -movflags +faststart output.mp4
    1003  open output.mp4
    1004  ffmpeg -framerate 1000 -pattern_type glob -i "./*.png" -vf "scale=iw:-2,format=yuv420p"  -c:v libx264 -movflags +faststart output.mp4
    1005  open output.mp4
    1006  ffmpeg -framerate 1 -pattern_type glob -i "./*.jpg" -vf "scale=iw:-2,format=yuv420p"  -c:v libx264 -movflags +faststart output.mp4   
    ffmpeg  -framerate 1 -f image2 -i "concat:page_048.jpg|page_048.jpg" \
       -r 1 -c:v libx264 -movflags +faststart output.mp4
       
    ffmpeg \
        -framerate 1 -i page_047.jpg \
        -framerate 1 -i page_048.jpg \
        -framerate 1 -i page_049.jpg \
        -framerate 1 -i page_050.jpg \
        -framerate 1 -i page_051.jpg \
        -framerate 1 -i page_052.jpg \
        -framerate 1 -i page_053.jpg \
       -filter_complex "[0][1][2][3][4][5][6]concat=n=7:v=1:a=0" \
       -r 1 -c:v libx264 -movflags +faststart output.mp4 
       
    ffmpeg \
  -framerate 1 -i page_047.jpg \
  -framerate 1 -i page_048.jpg \
  -framerate 1 -i page_049.jpg \
  -framerate 1 -i page_050.jpg \
  -framerate 1 -i page_051.jpg \
  -framerate 1 -i page_052.jpg \
  -framerate 1 -i page_053.jpg \
  -filter_complex \
    "[0:v][1:v][2:v][3:v][4:v][5:v][6:v]concat=n=7:v=1:a=0[outv]" \
  -map "[outv]" \
  -r 1 \
  -c:v libx264 \
  -movflags +faststart \
  output.mp4
  
  ffmpeg \
  -framerate 1 -i page_047.jpg \
  -framerate 1 -i page_048.jpg \
  -framerate 1 -i page_049.jpg \
  -framerate 1 -i page_050.jpg \
  -framerate 1 -i page_051.jpg \
  -framerate 1 -i page_052.jpg \
  -framerate 1 -i page_053.jpg \
  -filter_complex \
    "[0][1][2][3][4][5][6]concat=n=7:v=1:a=0[outv]" \
  -map "[outv]" \
  -r 1 \
  -c:v libx264 \
  -movflags +faststart \
  output.mp4
  
       
    ffmpeg -framerate 1 -start_number 47 -i "page_%03d.jpg" \
       -vframes 7 -r 1 -vf "scale=iw:-2,format=yuv420p" \
       -c:v libx264 -movflags +faststart output.mp4
    
    
    # 测试前3张
ffmpeg -framerate 1 -i page_047.jpg -i page_048.jpg -i page_049.jpg \
  -filter_complex "[0:v][1:v][2:v]concat=n=3:v=1:a=0" -r 1 test.mp4

# 测试后4张
ffmpeg -f image2 -framerate 1 -i page_050.jpg -f image2 -framerate 1 -i page_051.jpg -f image2 -framerate 1 -i page_052.jpg -f image2 -framerate 1 -i page_053.jpg \
  -filter_complex "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0" -r 1 test2.mp4   
  
  ffmpeg -f image2 -framerate 1 -i page_047.jpg -framerate 1 -i page_048.jpg -framerate 1 -i page_049.jpg -framerate 1 -i page_050.jpg -f image2 -framerate 1 -i page_051.jpg -f image2 -framerate 1 -i page_052.jpg -f image2 -framerate 1 -i page_053.jpg \
  -filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v][6:v]concat=n=7:v=1:a=0" -r 1 test2.mp4
  
ffmpeg -f image2 -framerate 1 -i page_050.jpg -f image2 -framerate 1 -i page_051.jpg -f image2 -framerate 1 -i page_052.jpg -f image2 -framerate 1 -i page_053.jpg \
  -filter_complex "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0" -vf "framestep=2,fps=1,scale=iw:-2,format=yuv420p" -c:v libx264 -movflags +faststart test2.mp4  

ffmpeg -f image2 -framerate 1 -i page_050.jpg -f image2 -framerate 1 -i page_051.jpg -f image2 -framerate 1 -i page_052.jpg -f image2 -framerate 1 -i page_053.jpg \
-vf "framestep=4,fps=1,scale=iw:-2,format=yuv420p" -c:v libx264 -movflags +faststart test2.mp4  

  ffmpeg -f image2 -framerate 1 -i page_047.jpg -framerate 1 -i page_048.jpg -framerate 1 -i page_049.jpg -framerate 1 -i page_050.jpg -f image2 -framerate 1 -i page_051.jpg -f image2 -framerate 1 -i page_052.jpg -f image2 -framerate 1 -i page_053.jpg \
  -filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v][6:v]concat=n=7:v=1:a=0,fps=25,scale=iw:-2,format=yuv420p[outv]" -map "[outv]" -r 1 test2.mp4

  
ffmpeg -framerate 1 -f image2 -i "concat:page_047.jpg|page_048.jpg|page_049.jpg|page_050.jpg|page_051.jpg|page_052.jpg|page_053.jpg" \
       -vf "framestep=2,fps=1,scale=iw:-2,format=yuv420p" \
       -c:v libx264 -movflags +faststart output.mp4  
       
ffmpeg \
  -framerate 0.5 -i page_050.jpg \
  -framerate 0.5 -i page_051.jpg \
  -framerate 0.5 -i page_052.jpg \
  -framerate 0.5 -i page_053.jpg \
  -filter_complex \
    "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0, \
     fps=25,scale=iw:-2,format=yuv420p[outv]" \
  -map "[outv]" \
  -c:v libx264 \
  -movflags +faststart \
  test2.mp4    
  
  
    ffmpeg -f image2 -framerate 1 -i page_047.jpg -framerate 1 -i page_048.jpg -framerate 1 -i page_049.jpg -framerate 1 -i page_050.jpg -f image2 -framerate 1 -i page_051.jpg -f image2 -framerate 1 -i page_052.jpg -f image2 -framerate 1 -i page_053.jpg \
  -filter_complex "[0:v][1:v][2:v][3:v][4:v][5:v][6:v]concat=n=7:v=1:a=0" -r 1 test2.mp4   
    '''

       