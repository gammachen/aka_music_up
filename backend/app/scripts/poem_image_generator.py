import os
import json
import requests
import openai
import logging
import time
from typing import Dict, List, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置ollama的baseUrl
openai.api_base = "http://localhost:11434/v1"  # 假设ollama服务运行在本地11434端口
openai.api_key = "ollama"  # 使用ollama时，api_key可以设置为任意值

def call_ollama(prompt: str, model: str = "qwen2", temperature: float = 0.7, max_tokens: int = 1000):
    """调用ollama模型生成文本
    
    Args:
        prompt: 提示词
        model: 模型名称，默认为"qwen2"
        temperature: 温度参数，控制生成文本的随机性
        max_tokens: 生成文本的最大token数
        
    Returns:
        生成的文本内容
    """
    logger.info(f"Calling Ollama model: {model} with prompt length: {len(prompt)}")
    
    # 使用新版本的OpenAI API
    try:
        client = openai.OpenAI(
            api_key=openai.api_key,  # 使用配置的api_key
            base_url=openai.api_base  # 使用配置的baseUrl
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info(f"Ollama response received")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Ollama API调用失败: {str(e)}")
        raise

def generate_image_prompt(poem_content: str) -> str:
    """根据诗歌内容生成文生图提示词
    
    Args:
        poem_content: 诗歌内容
        
    Returns:
        生成的文生图提示词
    """
    prompt_template = """
    请根据以下古诗词的内容，生成一个用于AI文生图的详细提示词（prompt）。
    提示词应该：
    1. 捕捉诗歌的核心意境和场景
    2. 包含具体的视觉元素（如景物、人物、色彩、光线等）
    3. 适合用于生成高质量的艺术图像
    4. 使用英文表达，因为大多数AI绘画模型对英文提示词效果更好
    5. 长度控制在100-150个英文单词之间
    
    诗歌内容：
    {poem_content}
    
    请直接输出英文提示词，不要包含任何解释或其他内容。
    """
    
    formatted_prompt = prompt_template.format(poem_content=poem_content)
    return call_ollama(formatted_prompt)

def generate_image_with_siliconflow(prompt: str, api_key: str) -> str:
    """使用siliconflow API生成图片
    
    Args:
        prompt: 文生图提示词
        api_key: siliconflow API密钥
        
    Returns:
        生成的图片URL
    """
    url = "https://api.siliconflow.cn/v1/images/generations"
    
    payload = {
        "model": "Kwai-Kolors/Kolors",
        "prompt": prompt,
        "negative_prompt": "",
        "image_size": "1024x1024",
        "batch_size": 1,
        "seed": 4999999999,
        "num_inference_steps": 20,
        "guidance_scale": 7.5
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        logger.info(f"Calling siliconflow API with prompt: {prompt[:50]}...")
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # 如果请求失败，抛出异常
        
        result = response.json()
        logger.info(f"Image generated successfully")
        
        # 返回生成的图片URL
        if "images" in result and len(result["images"]) > 0 and "url" in result["images"][0]:
            return result["images"][0]["url"]
        else:
            logger.error(f"No image URL found in response: {result}")
            return ""
    except Exception as e:
        logger.error(f"siliconflow API调用失败: {str(e)}")
        return ""

def process_poem_for_image(poem: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """处理单首诗歌，生成图片并更新诗歌数据
    
    Args:
        poem: 诗歌数据
        api_key: siliconflow API密钥
        
    Returns:
        更新后的诗歌数据
    """
    # 构建完整的诗歌内容
    title = poem.get("title", "")
    paragraphs = poem.get("paragraphs", [])
    poem_content = title + "\n" + "\n".join(paragraphs)
    
    # 生成文生图提示词
    logger.info(f"Generating image prompt for poem: {title}")
    image_prompt = generate_image_prompt(poem_content)
    
    # 使用siliconflow API生成图片
    logger.info(f"Generating image with prompt: {image_prompt[:50]}...")
    image_url = generate_image_with_siliconflow(image_prompt, api_key)
    
    # 更新诗歌数据
    updated_poem = poem.copy()
    updated_poem["image_prompt"] = image_prompt
    updated_poem["image_url"] = image_url
    
    # 下载图片并保存到本地
    local_img_url = ""
    if image_url:
        try:
            # 获取项目根目录
            script_dir = os.path.abspath(__file__) # 错误的路径：os.path.dirname(os.path.abspath(__file__))
            backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
            frontend_dir = os.path.join(os.path.dirname(backend_dir), "frontend")
            poem_img_dir = os.path.join(frontend_dir, "public", "img", "poem")
            
            # 确保目标目录存在
            os.makedirs(poem_img_dir, exist_ok=True)
            
            # 生成唯一的文件名
            poem_id = poem.get("id", "")
            filename = f"{poem_id}_{title.replace(' ', '_')}_{int(time.time())}.jpg"
            filename = ''.join(c for c in filename if c.isalnum() or c in ['_', '.'])  # 确保文件名合法
            file_path = os.path.join(poem_img_dir, filename)
            
            # 下载图片
            logger.info(f"Downloading image from {image_url} to {file_path}")
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            
            # 保存图片到本地
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # 设置相对路径的URL
            local_img_url = f"/img/poem/{filename}"
            logger.info(f"Image saved locally at {local_img_url}")
        except Exception as e:
            logger.error(f"保存图片到本地失败: {str(e)}")
    
    # 添加本地图片URL
    updated_poem["localImgUrl"] = local_img_url
    
    return updated_poem

def process_poems_with_images(input_json_path: str, output_json_path: str, api_key: str, limit: int = None):
    """处理诗歌集合，为每首诗歌生成图片
    
    Args:
        input_json_path: 输入JSON文件路径
        output_json_path: 输出JSON文件路径
        api_key: siliconflow API密钥
        limit: 处理的诗歌数量限制，默认为None表示处理所有诗歌
    """
    # 加载诗歌数据
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            poems = json.load(f)
    except Exception as e:
        logger.error(f"加载诗歌数据失败: {str(e)}")
        return
    
    logger.info(f"Loaded {len(poems)} poems from {input_json_path}")
    
    # 限制处理的诗歌数量
    if limit is not None and limit < len(poems):
        poems = poems[:limit]
        logger.info(f"Limited to processing {limit} poems")
    
    # 处理每首诗歌
    updated_poems = []
    for i, poem in enumerate(poems):
        logger.info(f"Processing poem {i+1}/{len(poems)}: {poem.get('title', '')}")
        updated_poem = process_poem_for_image(poem, api_key)
        updated_poems.append(updated_poem)
        
        # 每处理5首诗歌保存一次中间结果
        if (i + 1) % 5 == 0:
            checkpoint_path = output_json_path.replace('.json', f'_checkpoint_{i+1}.json')
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(updated_poems, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # 保存最终结果
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(updated_poems, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Processed {len(updated_poems)} poems and saved to {output_json_path}")

'''
根据诗歌生成对应的图片（AI生图，调用了硅基的免费图片生成接口，前面还调用了本地ollama qwen2来生成诗歌对应的图片的提示词prompt功能）
'''
def main():
    # 设置参数
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resource_dir = os.path.join(script_dir, 'resource')
    
    poet_name = "李白"
    input_json_path = os.path.join(resource_dir, f'{poet_name}_enriched.json')
    output_json_path = os.path.join(resource_dir, f'{poet_name}_enriched_with_images.json')
    
    # 替换为你的siliconflow API密钥
    api_key = "sk-mwvkqkfpwifyltixozycqesclwpbcfvasnokiololciutddu"
    
    # 处理诗歌生成图片
    process_poems_with_images(input_json_path, output_json_path, api_key) # , limit=5)  # 限制处理5首诗歌，可以根据需要调整

if __name__ == "__main__":
    main()