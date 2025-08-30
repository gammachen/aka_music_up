import json
import os
import requests
import time
import random
from typing import Dict, List, Any, Optional, Tuple

# 定义文件路径处理函数
def get_resource_paths():
    """获取资源文件路径"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resource_dir = os.path.join(script_dir, 'resource')
    return {
        'resource_dir': resource_dir,
        'quan_tang_shi_path': os.path.join(resource_dir, 'quan_tang_shi.json'),
    }


def filter_poems_by_author(poems_data: List[Dict], author_name: str) -> List[Dict]:
    """根据作者名筛选诗歌
    
    Args:
        poems_data: 诗歌数据列表
        author_name: 作者名称
        
    Returns:
        符合条件的诗歌列表
    """
    return [poem for poem in poems_data if poem.get('author') == author_name]


def save_poems_to_file(poems: List[Dict], file_path: str) -> None:
    """将诗歌数据保存到文件
    
    Args:
        poems: 诗歌数据列表
        file_path: 保存的文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(poems, f, ensure_ascii=False, indent=2)
    print(f'已将诗歌保存到 {file_path}')


def load_poems_from_file(file_path: str) -> List[Dict]:
    """从文件加载诗歌数据
    
    Args:
        file_path: 文件路径
        
    Returns:
        诗歌数据列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_poems_by_author(author_name: str) -> str:
    """处理指定作者的诗歌并保存
    
    Args:
        author_name: 作者名称
        
    Returns:
        保存的文件路径
    """
    paths = get_resource_paths()
    output_path = os.path.join(paths['resource_dir'], f'{author_name}.json')
    
    # 读取全唐诗数据
    quan_tang_shi_data = load_poems_from_file(paths['quan_tang_shi_path'])
    
    # 筛选作者的诗歌
    author_poems = filter_poems_by_author(quan_tang_shi_data, author_name)
    
    # 输出统计信息
    print(f'全唐诗共有 {len(quan_tang_shi_data)} 首诗')
    print(f'{author_name}的诗共有 {len(author_poems)} 首')
    
    # 保存作者的诗歌
    save_poems_to_file(author_poems, output_path)
    
    return output_path


class LLMAI:
    """AI接口封装"""
    
    def __init__(self, api_key: str, base_url: str):
        """初始化智谱AI接口
        ZhipuAI(api_key="d4d54cde38f38f0f591af45d7ec7910a.0gxd5TQTmRuSnVca", 
                              base_url="https://open.bigmodel.cn/api/paas/v4/")
                              
        OpenAI(api_key="sk-mwvkqkfpwifyltixozycqesclwpbcfvasnokiololciutddu", 
        #                      base_url="https://api.siliconflow.cn/v1")
                              
        Args:
            api_key: 智谱AI的API密钥
        """
        self.api_key = api_key
        self.base_url = base_url
        self.last_request_time = 0  # 上次请求时间
        self.min_request_interval = 2.0  # 最小请求间隔（秒）
        self.max_retries = 5  # 最大重试次数
        self.base_wait_time = 2.0  # 基础等待时间（秒）
        
        # 初始化OpenAI客户端
        from openai import OpenAI
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
    def _wait_for_rate_limit(self) -> None:
        """等待请求间隔，避免触发频率限制"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_request_interval:
            wait_time = self.min_request_interval - elapsed
            print(f"等待 {wait_time:.2f} 秒以避免频率限制...")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def analyze_poem(self, poem: Dict, model: str = "glm-4") -> Dict:
        """分析诗歌，获取创作背景信息
        
        Args:
            poem: 诗歌数据
            model: AI模型名称，默认为"glm-4"
            
        Returns:
            包含创作背景的字典
        """
        # 构建提示词
        title = poem.get('title', '无题')
        author = poem.get('author', '未知')
        content = '\n'.join(poem.get('paragraphs', []))
        tags = ', '.join(poem.get('tags', []))
        
        prompt = f"""请根据以下唐诗的信息，推断这首诗可能的创作时间、地点、关联人物和创作场景。
        如果无法确定，请根据诗人生平和诗歌内容做合理推测。
        
        诗题：{title}
        作者：{author}
        内容：
        {content}
        标签：{tags}
        
        请以JSON格式返回，包含以下字段：
        1. creation_year: 创作年份（数字，如701）
        2. creation_place: 创作地点
        3. related_people: 相关人物数组
        4. creation_scene: 创作场景描述
        5. historical_background: 历史背景
        
        仅返回JSON格式数据，不要有其他文字。
        """
        
        # 使用OpenAI风格的客户端接口发送请求
        for attempt in range(self.max_retries):
            try:
                # 等待请求间隔，避免触发频率限制
                self._wait_for_rate_limit()
                
                # 发送请求
                response = self.client.chat.completions.create(
                    model=model,  
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    top_p=0.8,
                    max_tokens=800,
                    stream=False
                )
                
                # 获取响应内容
                content = response.choices[0].message.content
                
                # 尝试解析JSON
                try:
                    background_info = json.loads(content)
                    return background_info
                except json.JSONDecodeError:
                    # 如果返回的不是有效JSON，尝试提取JSON部分
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        try:
                            return json.loads(json_match.group(0))
                        except json.JSONDecodeError:
                            return {
                                "creation_year": "未知",
                                "creation_place": "未知",
                                "related_people": []
                            }
                    else:
                        return {
                            "creation_year": "未知",
                            "creation_place": "未知",
                            "related_people": []
                        }
                        
            except Exception as e:
                if "429" in str(e):  # Too Many Requests
                    # 计算退避时间（指数退避策略）
                    wait_time = self.base_wait_time * (2 ** attempt) + random.uniform(0, 1)
                    print(f"遇到频率限制 (429)，第 {attempt+1}/{self.max_retries} 次重试，等待 {wait_time:.2f} 秒...")
                    time.sleep(wait_time)
                else:
                    print(f"请求错误: {e}")
                    # 简单重试
                    wait_time = self.base_wait_time * (1.5 ** attempt)
                    print(f"第 {attempt+1}/{self.max_retries} 次重试，等待 {wait_time:.2f} 秒...")
                    time.sleep(wait_time)
                    
                # 最后一次尝试失败
                if attempt == self.max_retries - 1:
                    print(f"达到最大重试次数 ({self.max_retries})，放弃请求")
                    return {
                        "creation_year": "未知",
                        "creation_place": "未知",
                        "related_people": []
                    }


def enrich_poems_with_ai(poems_file_path: str, api_key: str, base_url: str, model: str = "glm-4", output_file_path: Optional[str] = None, checkpoint_file: Optional[str] = None) -> str:
    """使用AI丰富诗歌信息
    
    Args:
        poems_file_path: 诗歌文件路径
        api_key: AI的API密钥
        base_url: AI的API基础URL
        model: AI模型名称，默认为"glm-4"
        output_file_path: 输出文件路径，默认为None（在原文件名基础上添加_enriched后缀）
        checkpoint_file: 检查点文件路径，用于保存进度，默认为None
        
    Returns:
        输出文件路径
    """
    # 加载诗歌数据
    poems = load_poems_from_file(poems_file_path)
    
    # 初始化AI客户端
    ai_client = LLMAI(api_key, base_url)
    
    # 设置默认输出路径
    if output_file_path is None:
        file_name, file_ext = os.path.splitext(poems_file_path)
        output_file_path = f"{file_name}_enriched{file_ext}"
    
    # 设置默认检查点文件路径
    if checkpoint_file is None:
        checkpoint_file = f"{os.path.splitext(output_file_path)[0]}_checkpoint.json"
    
    # 尝试从检查点恢复
    start_index = 0
    enriched_poems = []
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                enriched_poems = checkpoint_data.get('enriched_poems', [])
                start_index = checkpoint_data.get('next_index', 0)
                print(f"从检查点恢复，已处理 {start_index} 首诗")
        except Exception as e:
            print(f"读取检查点文件失败: {e}，将从头开始处理")
            start_index = 0
            enriched_poems = []
    
    # 处理每首诗
    try:
        for i in range(start_index, len(poems)):
            poem = poems[i]
            print(f"正在处理第 {i+1}/{len(poems)} 首诗: {poem.get('title', '无题')}")
            
            try:
                # 获取AI分析结果
                background_info = ai_client.analyze_poem(poem, model=model)
                
                # 合并原始诗歌数据和AI分析结果
                enriched_poem = {**poem, **background_info}
                enriched_poems.append(enriched_poem)
                
                # 每处理5首诗保存一次检查点
                if (i + 1) % 5 == 0 or i == len(poems) - 1:
                    # 保存检查点
                    checkpoint_data = {
                        'enriched_poems': enriched_poems,
                        'next_index': i + 1
                    }
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                    print(f"已保存检查点，处理进度: {i+1}/{len(poems)}")
                    
                    # 同时保存当前结果到输出文件
                    save_poems_to_file(enriched_poems, output_file_path)
                    print(f"已保存当前结果到 {output_file_path}")
            
            except Exception as e:
                print(f"处理诗歌时出错: {e}")
                print("保存检查点并继续处理下一首诗...")
                # 保存检查点
                checkpoint_data = {
                    'enriched_poems': enriched_poems,
                    'next_index': i
                }
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    
    except KeyboardInterrupt:
        print("\n用户中断处理，保存当前进度...")
        # 保存检查点
        checkpoint_data = {
            'enriched_poems': enriched_poems,
            'next_index': i
        }
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        # 保存当前结果
        save_poems_to_file(enriched_poems, output_file_path)
    
    # 处理完成，保存最终结果
    save_poems_to_file(enriched_poems, output_file_path)
    
    # 处理完成后删除检查点文件
    # if os.path.exists(checkpoint_file):
    #     try:
    #         os.remove(checkpoint_file)
    #         print(f"处理完成，已删除检查点文件 {checkpoint_file}")
    #     except Exception as e:
    #         print(f"删除检查点文件失败: {e}")
    
    return output_file_path


def generate_timeline_html(enriched_poems: List[Dict], output_html_path: str, poet_name: str, poet_intro: str) -> None:
    """根据丰富后的诗歌数据生成时间线HTML
    
    Args:
        enriched_poems: 丰富后的诗歌数据列表
        output_html_path: 输出HTML文件路径
        poet_name: 诗人名称
        poet_intro: 诗人简介
    """
    # 按创作年份排序
    def get_year_value(poem):
        year = poem.get('creation_year', '未知')
        # 尝试将年份转换为整数
        if isinstance(year, int):
            return year
        try:
            return int(year)
        except (ValueError, TypeError):
            # 如果无法转换为整数，返回一个较大的值，确保排在最后
            return 9999
    
    sorted_poems = sorted(enriched_poems, key=get_year_value)
    
    # 生成时间节点HTML
    timeline_nodes = []
    for i, poem in enumerate(sorted_poems):
        title = poem.get('title', '无题')
        year = poem.get('creation_year', '未知')
        place = poem.get('creation_place', '未知')
        scene = poem.get('creation_scene', '未知')
        background = poem.get('historical_background', '')
        content = '\n'.join(poem.get('paragraphs', []))
        
        # 为偶数和奇数节点设置不同的样式
        style_delay = 0.2 * (i + 1)
        even_odd_class = 'even' if i % 2 == 0 else 'odd'
        
        # 先处理换行符，然后再放入f-string中
        formatted_content = content.replace('\n', '<br>')
        node_html = f'''
        <div class="timeline-node {even_odd_class}" style="animation-delay: {style_delay}s">
            <div class="timeline-card">
                <div class="card-front">
                    <h3>{year}年 · {title}</h3>
                    <p>创作地点：{place}</p>
                    <div class="poem-content">{formatted_content}</div>
                </div>
                <div class="card-back">
                    <h4>创作背景</h4>
                    <p>{scene}</p>
                    <h4>历史背景</h4>
                    <p>{background}</p>
                </div>
            </div>
        </div>
        '''
        timeline_nodes.append(node_html)
    
    # 读取HTML模板并替换内容
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, '..', '..', '..', 'frontend', 'public', 'li_bai_timeline.html')
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
        
        # 替换模板中的内容
        html_content = template.replace('<h1>诗仙李白</h1>', f'<h1>{poet_name}</h1>')
        html_content = html_content.replace('<p>唐代最杰出的浪漫主义诗人，字太白，号青莲居士...</p>', f'<p>{poet_intro}</p>')
        
        # 替换时间节点
        timeline_nodes_html = '\n'.join(timeline_nodes)
        start_marker = '<!-- 时间节点 -->'
        end_marker = '<!-- 更多时间节点... -->'
        
        start_index = html_content.find(start_marker) + len(start_marker)
        end_index = html_content.find(end_marker)
        
        if start_index > 0 and end_index > 0:
            html_content = html_content[:start_index] + '\n' + timeline_nodes_html + '\n        ' + html_content[end_index:]
        
        # 保存生成的HTML
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"已生成时间线HTML: {output_html_path}")
        
    except Exception as e:
        print(f"生成HTML时出错: {e}")


def analyze_poet_poems(poet_name: str, input_json_path: str, output_json_path: str, output_html_path: str, poet_intro: str, api_key: str, base_url: str, model: str = "glm-4"):
    """完整的诗人诗歌分析流程
    
    Args:
        poet_name: 诗人名称
        input_json_path: 输入JSON文件路径
        output_json_path: 输出JSON文件路径
        output_html_path: 输出HTML文件路径
        poet_intro: 诗人简介
        api_key: AI的API密钥
        base_url: AI的API基础URL
        model: AI模型名称，默认为"glm-4"
    """
    # 1. 提取诗人的诗歌
    quan_tang_shi_data = load_poems_from_file(input_json_path)
    poems = filter_poems_by_author(quan_tang_shi_data, poet_name)
    
    if not poems:
        print(f"未找到{poet_name}的诗歌")
        return
    
    # 2. 保存原始诗歌数据
    save_poems_to_file(poems, output_json_path)
    
    # 3. 使用AI丰富诗歌信息
    print(f"开始使用AI丰富{poet_name}的诗歌信息...")
    enriched_file_path = enrich_poems_with_ai(output_json_path, api_key, base_url, model=model)
    
    # 4. 加载丰富后的诗歌数据
    enriched_poems = load_poems_from_file(enriched_file_path)
    
    # 5. 生成时间线HTML
    generate_timeline_html(enriched_poems, output_html_path, poet_name, poet_intro)
    
    print(f"{poet_name}诗歌分析完成！")


def main():
    # 示例：处理李白的诗歌
    paths = get_resource_paths()
    
    # 设置参数
    poet_name = "李白"
    input_json_path = paths['quan_tang_shi_path']
    output_json_path = os.path.join(paths['resource_dir'], f'{poet_name}.json')
    output_html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'frontend', 'public', f'{poet_name}_timeline.html')
    poet_intro = "唐代最杰出的浪漫主义诗人，字太白，号青莲居士，被后人尊称为\"诗仙\"。李白的诗歌题材广泛、内容丰富、想象奇特、意境独特，艺术成就极高，对后世诗歌创作产生了深远影响。"
    
    # 使用智谱AI丰富诗歌信息（需要提供API密钥）
    # 注意：实际使用时需要替换为真实的API密钥
    api_key = "d4d54cde38f38f0f591af45d7ec7910a.0gxd5TQTmRuSnVca"  # 替换为实际的智谱AI API密钥
    base_url = "https://open.bigmodel.cn/api/paas/v4"  # 智谱AI的API地址
    model = "glm-4-flash"  # 智谱AI的模型名称
    
    # self.client = ZhipuAI(api_key="d4d54cde38f38f0f591af45d7ec7910a.0gxd5TQTmRuSnVca", 
    # base_url="https://open.bigmodel.cn/api/paas/v4/")
        
    # self.client = OpenAI(api_key="sk-ba5922006bc8408ca5ccaf59ca022b9b", 
    #                       base_url="https://api.deepseek.com")
    
    api_key="sk-ba5922006bc8408ca5ccaf59ca022b9b"
    base_url="https://api.deepseek.com"
    model = "deepseek-r1"  # deepseek AI的模型名称 "deepseek-v3" # aliyun for deepseek-r1
    
    # self.client = OpenAI(api_key="sk-b7e65b050fc9402690a2119f56bfe5fb", 
    #                       base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    api_key="sk-b7e65b050fc9402690a2119f56bfe5fb"
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    model = "qwen-long"  # 
    
    # self.client = OpenAI(api_key="sk-mwvkqkfpwifyltixozycqesclwpbcfvasnokiololciutddu", 
    #                      base_url="https://api.siliconflow.cn/v1")
    
    api_key="sk-mwvkqkfpwifyltixozycqesclwpbcfvasnokiololciutddu"
    base_url="https://api.siliconflow.cn/v1"
    model = "deepseek-ai/DeepSeek-R1"
    
    # 执行完整分析流程
    analyze_poet_poems(poet_name, input_json_path, output_json_path, output_html_path, poet_intro, api_key, base_url, model)


def main_2():
    paths = get_resource_paths()
    
    # 设置参数
    poet_name = "李白"
    
    input_json_path = os.path.join(paths['resource_dir'], f'{poet_name}_enriched.json')
    
    output_html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'frontend', 'public', f'{poet_name}_timeline.html')
    poet_intro = "唐代最杰出的浪漫主义诗人，字太白，号青莲居士，被后人尊称为\"诗仙\"。李白的诗歌题材广泛、内容丰富、想象奇特、意境独特，艺术成就极高，对后世诗歌创作产生了深远影响。"
    
    enriched_poems = load_poems_from_file(input_json_path)
    generate_timeline_html(enriched_poems, output_html_path, poet_name, poet_intro)

if __name__ == "__main__":
    main()
    # main_2()