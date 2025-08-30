import openai
import os

# 配置ollama的baseUrl
openai.api_base = "http://localhost:11434/v1"  # 假设ollama服务运行在本地11434端口
openai.api_key = "ollama"  # 使用ollama时，api_key可以设置为任意值

def read_file_with_encoding(file_path, encodings=('utf-8', 'gb18030')):
    """尝试用不同的编码读取文件内容"""
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"无法用{encodings}中的编码读取文件: {file_path}")

def process_vovels(directory='../resource/novel'):
    """读取目录下的所有.txt文件，并分段提交给大模型"""
    responses = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            content = read_file_with_encoding(file_path)
            # 兼容不同操作系统下的换行符，包括\n和\r\n
            paragraphs = [p.strip() for p in content.splitlines() if p.strip()]
            for paragraph in paragraphs:
                response = call_ollama_process_novels(paragraph)
                
                print(f"{paragraph}: {response}")
                
                responses.append(response)
    return responses

def call_ollama_process_novels(prompt: str, model: str = "qwen2", temperature: float = 0.7, max_tokens: int = 1000):
    # 构建提取信息的prompt
    prompt_template = """
    文本：${pcontent} 
    请提取：
    - 人物关系：[xxx, xxx, event/action]
    - 时间：744年
    - 地点：洛阳
    """
    formatted_prompt = prompt_template.replace("${pcontent}", prompt)

    # 使用新版本的OpenAI API
    client = openai.OpenAI(
        api_key=openai.api_key,  # 使用配置的api_key
        base_url=openai.api_base  # 使用配置的baseUrl
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": formatted_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content



# 示例：处理sources目录下的文本文件
if __name__ == "__main__":
    responses = process_vovels()
    for response in responses:
        print(response)
    
    