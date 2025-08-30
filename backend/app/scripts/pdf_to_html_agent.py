import openai
import os
import fitz  # PyMuPDF
import re
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置ollama的baseUrl
openai.api_base = "http://localhost:11434/v1"  # 假设ollama服务运行在本地11434端口
openai.api_key = "ollama"  # 使用ollama时，api_key可以设置为任意值

def extract_text_from_pdf(pdf_path):
    """从PDF文件中提取文本内容
    
    Args:
        pdf_path: PDF文件路径
        
    Returns:
        提取的文本内容，按页面分组
    """
    try:
        # 打开PDF文件
        pdf_document = fitz.open(pdf_path)
        
        # 存储所有页面的文本
        all_pages = []
        
        # 遍历每一页
        for page_number in range(pdf_document.page_count):
            # 获取页面
            page = pdf_document[page_number]
            
            # 提取文本
            text = page.get_text()
            
            # 添加到结果列表
            all_pages.append(text)
        
        # 关闭PDF文件
        pdf_document.close()
        
        return all_pages
    except Exception as e:
        logger.error(f"提取PDF文本失败: {str(e)}")
        return []

def split_into_chapters(pages):
    """将PDF页面文本分割成章节
    
    Args:
        pages: 页面文本列表
        
    Returns:
        章节列表，每个章节包含标题和内容
    """
    # 合并所有页面文本
    full_text = "\n".join(pages)
    
    # 尝试识别章节标题的模式
    # 这里使用简单的正则表达式，可能需要根据实际PDF格式调整
    chapter_pattern = re.compile(r'(?:\n|^)(第[一二三四五六七八九十百千万零\d]+[章节篇].*?)(?=\n|$)')
    
    # 查找所有章节标题
    chapter_matches = list(chapter_pattern.finditer(full_text))
    
    # 如果没有找到章节，尝试其他模式
    if not chapter_matches:
        # 尝试其他常见的章节标题格式
        chapter_pattern = re.compile(r'(?:\n|^)([\d]+\.\s*.*?)(?=\n|$)')
        chapter_matches = list(chapter_pattern.finditer(full_text))
    
    # 如果仍然没有找到章节，将整个文档作为一个章节
    if not chapter_matches:
        return [{"title": "文档内容", "content": full_text}]
    
    # 提取章节
    chapters = []
    for i, match in enumerate(chapter_matches):
        title = match.group(1).strip()
        start_pos = match.start()
        
        # 确定章节内容的结束位置
        if i < len(chapter_matches) - 1:
            end_pos = chapter_matches[i + 1].start()
        else:
            end_pos = len(full_text)
        
        # 提取章节内容
        content = full_text[start_pos:end_pos].strip()
        
        chapters.append({"title": title, "content": content})
    
    return chapters

def call_ollama_generate_html(chapter, model="qwen2", temperature=0.7, max_tokens=4000):
    """调用大模型将章节内容转换为HTML
    
    Args:
        chapter: 章节内容，包含标题和正文
        model: 使用的模型名称
        temperature: 温度参数
        max_tokens: 最大生成token数
        
    Returns:
        生成的HTML代码
    """
    # 构建提示词
    prompt_template = """
    请将以下文本内容转换为HTML格式。要求：
    1. 保持原文的段落结构
    2. 使用适当的HTML标签（如h1, h2, p等）
    3. 添加适当的CSS样式，使页面整洁美观
    4. 不要添加任何不在原文中的内容
    5. 只返回HTML代码，不要包含任何解释
    
    章节标题：${title}
    
    章节内容：
    ${content}
    """
    
    formatted_prompt = prompt_template.replace("${title}", chapter["title"]).replace("${content}", chapter["content"])
    
    try:
        # 使用新版本的OpenAI API
        client = openai.OpenAI(
            api_key=openai.api_key,
            base_url=openai.api_base
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"调用大模型生成HTML失败: {str(e)}")
        # 如果调用失败，返回简单的HTML
        return f"<div><h2>{chapter['title']}</h2><pre>{chapter['content']}</pre></div>"

def generate_navigation_html(chapters):
    """生成左侧导航栏的HTML代码
    
    Args:
        chapters: 章节列表
        
    Returns:
        导航栏HTML代码
    """
    nav_html = "<div class='navigation'>\n"
    nav_html += "<h3>目录</h3>\n"
    nav_html += "<ul>\n"
    
    for i, chapter in enumerate(chapters):
        chapter_id = f"chapter-{i+1}"
        nav_html += f"<li><a href='#{chapter_id}'>{chapter['title']}</a></li>\n"
    
    nav_html += "</ul>\n"
    nav_html += "</div>\n"
    
    return nav_html

def generate_complete_html(pdf_name, chapters, chapter_htmls):
    """生成完整的HTML页面
    
    Args:
        pdf_name: PDF文件名
        chapters: 章节列表
        chapter_htmls: 章节HTML内容列表
        
    Returns:
        完整的HTML页面代码
    """
    # 生成导航栏
    nav_html = generate_navigation_html(chapters)
    
    # 生成主内容区
    content_html = "<div class='content'>\n"
    
    for i, html_content in enumerate(chapter_htmls):
        chapter_id = f"chapter-{i+1}"
        content_html += f"<div id='{chapter_id}' class='chapter'>\n"
        content_html += html_content
        content_html += "\n</div>\n"
    
    content_html += "</div>\n"
    
    # 组合完整HTML
    complete_html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{pdf_name}</title>
        <style>
            body {{  
                margin: 0;
                padding: 0;
                font-family: 'Microsoft YaHei', Arial, sans-serif;
                display: flex;
                min-height: 100vh;
            }}
            .navigation {{  
                width: 250px;
                background-color: #f5f5f5;
                padding: 20px;
                position: fixed;
                height: 100vh;
                overflow-y: auto;
                box-shadow: 2px 0 5px rgba(0,0,0,0.1);
            }}
            .navigation h3 {{  
                margin-top: 0;
                padding-bottom: 10px;
                border-bottom: 1px solid #ddd;
            }}
            .navigation ul {{  
                list-style-type: none;
                padding: 0;
            }}
            .navigation li {{  
                margin-bottom: 8px;
            }}
            .navigation a {{  
                text-decoration: none;
                color: #333;
                display: block;
                padding: 5px 0;
                transition: color 0.3s;
            }}
            .navigation a:hover {{  
                color: #1e88e5;
            }}
            .content {{  
                flex: 1;
                padding: 30px;
                margin-left: 250px;
                max-width: 800px;
            }}
            .chapter {{  
                margin-bottom: 40px;
                line-height: 1.6;
            }}
            h1, h2, h3, h4, h5, h6 {{  
                color: #333;
            }}
            p {{  
                margin-bottom: 16px;
            }}
            @media (max-width: 768px) {{  
                body {{  
                    flex-direction: column;
                }}
                .navigation {{  
                    width: 100%;
                    height: auto;
                    position: relative;
                }}
                .content {{  
                    margin-left: 0;
                }}
            }}
        </style>
    </head>
    <body>
        {nav_html}
        {content_html}
    </body>
    </html>
    """
    
    return complete_html

def process_pdf_to_html(pdf_dir='../resource/rtc', output_dir='../static/rtc_html'):
    """处理目录下的所有PDF文件，转换为HTML
    
    Args:
        pdf_dir: PDF文件目录
        output_dir: 输出HTML文件目录
        
    Returns:
        生成的HTML文件路径列表
    """
    # 确保输入目录存在
    if not os.path.exists(pdf_dir):
        logger.error(f"PDF目录不存在: {pdf_dir}")
        # 尝试创建目录
        try:
            os.makedirs(pdf_dir)
            logger.info(f"已创建PDF目录: {pdf_dir}")
        except Exception as e:
            logger.error(f"创建PDF目录失败: {str(e)}")
            return []
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储生成的HTML文件路径
    html_files = []
    
    # 遍历目录中的所有PDF文件
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            pdf_name = os.path.splitext(filename)[0]
            
            logger.info(f"处理PDF文件: {pdf_path}")
            
            # 提取PDF文本
            pages = extract_text_from_pdf(pdf_path)
            
            if not pages:
                logger.warning(f"PDF文件为空或提取失败: {pdf_path}")
                continue
            
            # 分割章节
            chapters = split_into_chapters(pages)
            
            logger.info(f"识别到{len(chapters)}个章节")
            
            # 为每个章节生成HTML
            chapter_htmls = []
            for chapter in chapters:
                logger.info(f"处理章节: {chapter['title']}")
                html_content = call_ollama_generate_html(chapter)
                chapter_htmls.append(html_content)
            
            # 生成完整HTML
            complete_html = generate_complete_html(pdf_name, chapters, chapter_htmls)
            
            # 保存HTML文件
            output_path = os.path.join(output_dir, f"{pdf_name}.html")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(complete_html)
            
            logger.info(f"HTML文件已保存: {output_path}")
            html_files.append(output_path)
    
    return html_files

# 主函数
if __name__ == "__main__":
    # 设置相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    # 构建PDF目录和输出目录的绝对路径
    pdf_dir = os.path.join(app_dir, 'backend', 'app', 'scripts', 'resource', 'rtc')
    logger.info(f"PDF目录: {pdf_dir}")
    output_dir = os.path.join(app_dir, 'backend', 'app', 'scripts', 'resource', 'rtc')
    logger.info(f"输出HTML目录: {output_dir}")
    
    # 处理PDF文件
    html_files = process_pdf_to_html(pdf_dir, output_dir)
    
    if html_files:
        logger.info(f"成功生成{len(html_files)}个HTML文件")
        for html_file in html_files:
            logger.info(f"- {html_file}")
    else:
        logger.warning("未生成任何HTML文件")