# novel_analyzer.py
import re
import json
import requests
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm
import hashlib
import redis
import openai

# modelscope download --model Qwen/Qwen2.5-0.5B-Instruct
# modelscope download --model 'Qwen/Qwen2.5-0.5B-Instruct' --local_dir '/Users/shhaofu/Downloads/Qwen2.5-0.5B-Instruct'
# modelscope download --model 'Qwen/Qwen2.5-0.5B-Instruct' --local_dir '/Users/shhaofu/Downloads/Qwen2.5-0.5B-Instruct'
# modelscope download --model 'Qwen/Qwen2.5-0.5B-Instruct'
# modelscope download --model Qwen/Qwen2.5-0.5B-Instruct README.md --local_dir ./dir

# ================= 配置模块 =================
class Config:
    OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"  # Ollama服务地址
    QWEN2_PATH = "/Users/shhaofu/Downloads/Qwen2.5-0.5B-Instruct"  # QWen2模型本地路径 qwen2-7b
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    CONTEXT_WINDOW = 3  # 上下文窗口大小
    CHUNK_SIZE = 2000  # 文本块大小（字符数）

import logging
import time
from functools import wraps

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d',
    handlers=[
        logging.FileHandler('novel_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 函数执行时间装饰器
def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.debug(f"开始执行 {func.__name__} 函数，参数: {args}, {kwargs}")
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"函数 {func.__name__} 执行完成，耗时: {execution_time:.4f}秒")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"函数 {func.__name__} 执行失败，耗时: {execution_time:.4f}秒，错误: {str(e)}")
            raise
    return wrapper

# ================= 模型调用模块 =================
class ModelHandler:    
    # 配置ollama的baseUrl
    openai.api_base = "http://localhost:11434/v1"  # 假设ollama服务运行在本地11434端口
    openai.api_key = "ollama"  # 使用ollama时，api_key可以设置为任意值
    
    @staticmethod
    @log_execution_time
    def call_ollama(prompt: str, model: str = "qwen2", temperature: float = 0.7, max_tokens: int = 1000):
        logger.info(f"Calling Ollama model: {model} with prompt length: {len(prompt)}, temperature: {temperature}, max_tokens: {max_tokens}")
        logger.debug(f"Prompt content: {prompt[:200]}...")
        formatted_prompt = prompt

        # 使用新版本的OpenAI API
        try:
            client = openai.OpenAI(
                api_key=openai.api_key,  # 使用配置的api_key
                base_url=openai.api_base  # 使用配置的baseUrl
            )
            logger.debug(f"OpenAI client initialized with base_url: {openai.api_base}")
            
            start_time = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            api_time = time.time() - start_time
            logger.info(f"Ollama API call completed in {api_time:.2f}秒")
            logger.info(f"Ollama response received: {response.choices[0].message.content[:100]}...")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Ollama API调用失败: {str(e)}")
            raise

    @staticmethod
    @log_execution_time
    def call_ollama_rest(prompt: str, model: str = "qwen2") -> str:
        logger.info(f"Calling Ollama REST API with model: {model}, prompt length: {len(prompt)}")
        logger.debug(f"REST API endpoint: {Config.OLLAMA_ENDPOINT}")
        logger.debug(f"Prompt content: {prompt[:200]}...")
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        try:
            start_time = time.time()
            response = requests.post(Config.OLLAMA_ENDPOINT, json=payload)
            api_time = time.time() - start_time
            logger.info(f"Ollama REST API call completed in {api_time:.2f}秒")
            
            response.raise_for_status()
            response_json = response.json()
            logger.debug(f"Response status code: {response.status_code}, content length: {len(response_json.get('response', ''))}")
            logger.info(f"Ollama REST API response received: {response_json.get('response', '')[:100]}...")
            return response_json.get("response", "")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama REST API网络请求错误: {str(e)}")
            return ""
        except ValueError as e:
            logger.error(f"Ollama REST API JSON解析错误: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"Ollama REST API未知错误: {str(e)}")
            return ""

    @staticmethod
    @log_execution_time
    def call_qwen2(prompt: str) -> str:
        logger.info(f"Calling Qwen2 model with prompt length: {len(prompt)}")
        logger.debug(f"Qwen2 model path: {Config.QWEN2_PATH}")
        logger.debug(f"Prompt content: {prompt[:200]}...")
        
        try:
            # 假设已通过transformers加载模型
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            start_time = time.time()
            logger.debug(f"开始加载Qwen2模型和tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(Config.QWEN2_PATH)
            model = AutoModelForCausalLM.from_pretrained(Config.QWEN2_PATH)
            model_load_time = time.time() - start_time
            logger.info(f"Qwen2模型加载完成，耗时: {model_load_time:.2f}秒")
            
            logger.debug(f"开始tokenize输入文本")
            inputs = tokenizer(prompt, return_tensors="pt")
            logger.debug(f"输入tokens数量: {len(inputs['input_ids'][0])}")
            
            logger.debug(f"开始生成回复")
            generation_start = time.time()
            outputs = model.generate(**inputs, max_new_tokens=500)
            generation_time = time.time() - generation_start
            logger.info(f"Qwen2生成完成，耗时: {generation_time:.2f}秒")
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Qwen2 response received: {result[:100]}...")
            logger.debug(f"生成tokens数量: {len(outputs[0])}")
            return result
        except Exception as e:
            logger.error(f"Qwen2模型调用失败: {str(e)}")
            logger.exception("Qwen2调用异常详情")
            raise

# ================= 文本处理模块 =================
class TextProcessor:
    @staticmethod
    @log_execution_time
    def split_chapters(text: str) -> List[str]:
        """按章节分割文本"""
        logger.info(f"开始分割章节，文本总长度: {len(text)}")
        try:
            chapters = re.split(r'\n第[一二三四五六七八九十]+章\s', text)
            result = [chap.strip() for chap in chapters if chap.strip()]
            logger.info(f"章节分割完成，共找到 {len(result)} 个章节")
            logger.debug(f"章节长度分布: {[len(chap) for chap in result][:5]}...")
            return result
        except Exception as e:
            logger.error(f"章节分割失败: {str(e)}")
            raise
    
    @staticmethod
    @log_execution_time
    def semantic_chunking(text: str) -> List[str]:
        """基于标点的语义分块"""
        logger.info(f"开始语义分块，文本长度: {len(text)}")
        chunks = []
        current_chunk = []
        current_len = 0
        
        try:
            # 按句子分割
            start_time = time.time()
            sentences = re.split(r'(?<=[。！？；])', text)
            split_time = time.time() - start_time
            logger.debug(f"句子分割完成，耗时: {split_time:.4f}秒，共 {len(sentences)} 个句子")
            
            valid_sentences = 0
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                valid_sentences += 1
                sent_len = len(sent)
                if current_len + sent_len > Config.CHUNK_SIZE:
                    chunks.append("".join(current_chunk))
                    current_chunk = [sent]
                    current_len = sent_len
                else:
                    current_chunk.append(sent)
                    current_len += sent_len
            if current_chunk:
                chunks.append("".join(current_chunk))
                
            logger.info(f"语义分块完成，有效句子数: {valid_sentences}，生成 {len(chunks)} 个文本块")
            logger.debug(f"文本块长度分布: {[len(chunk) for chunk in chunks][:5]}...")
            return chunks
        except Exception as e:
            logger.error(f"语义分块失败: {str(e)}")
            raise

# ================= 知识管理模块 =================
class KnowledgeManager:
    def __init__(self):
        self.entities = defaultdict(dict)
        self.events = []
        self.relations = []
        self.redis = redis.Redis(
            host=Config.REDIS_HOST, 
            port=Config.REDIS_PORT,
            decode_responses=True
        )
    
    @log_execution_time
    def _get_cache_key(self, text: str) -> str:
        """生成文本块的唯一缓存键"""
        logger.debug(f"生成缓存键，文本长度: {len(text)}")
        hash_key = hashlib.md5(text.encode()).hexdigest()
        logger.debug(f"生成的缓存键: {hash_key}")
        return hash_key
    
    @log_execution_time
    def get_cached_result(self, text: str) -> dict:
        """获取缓存结果"""
        logger.info(f"尝试获取缓存结果，文本长度: {len(text)}")
        key = self._get_cache_key(text)
        try:
            start_time = time.time()
            cached = self.redis.get(key)
            redis_time = time.time() - start_time
            logger.debug(f"Redis查询耗时: {redis_time:.4f}秒")
            
            if cached:
                logger.info(f"找到缓存结果，键: {key[:8]}...，大小: {len(cached)}字节")
                try:
                    result = json.loads(cached)
                    logger.debug(f"缓存结果解析成功: {str(result)[:100]}...")
                    return result
                except json.JSONDecodeError as e:
                    logger.error(f"缓存结果JSON解析失败: {str(e)}")
                    return None
            else:
                logger.info(f"未找到缓存结果，键: {key[:8]}...")
                return None
        except Exception as e:
            logger.error(f"Redis缓存查询失败: {str(e)}")
            return None
    
    @log_execution_time
    def cache_result(self, text: str, result: dict) -> None:
        """缓存处理结果"""
        logger.info(f"缓存处理结果，文本长度: {len(text)}")
        key = self._get_cache_key(text)
        try:
            json_str = json.dumps(result)
            logger.debug(f"序列化结果大小: {len(json_str)}字节")
            
            start_time = time.time()
            self.redis.setex(key, 3600, json_str)
            redis_time = time.time() - start_time
            logger.info(f"结果已缓存，键: {key[:8]}...，过期时间: 3600秒，Redis写入耗时: {redis_time:.4f}秒")
        except Exception as e:
            logger.error(f"缓存结果失败: {str(e)}")
            logger.exception("缓存异常详情")
    
    @log_execution_time
    def merge_entities(self, new_data: dict) -> None:
        """实体合并与冲突解决"""
        logger.info(f"开始合并实体数据")
        if not isinstance(new_data, dict):
            logger.error(f"无效的数据格式，期望字典类型，实际为: {type(new_data)}")
            return
            
        logger.debug(f"新数据包含字段: {list(new_data.keys())}")
        
        # 人物合并
        characters = new_data.get("characters", [])
        logger.info(f"处理 {len(characters)} 个人物数据")
        merged_count = 0
        new_count = 0
        invalid_count = 0
        
        for char in characters:
            # 检查char是否为字典类型且包含name字段
            if not isinstance(char, dict) or "name" not in char:
                invalid_count += 1
                logger.warning(f"发现无效的人物数据: {char}")
                continue
                
            char_name = char["name"]
            logger.debug(f"处理人物: {char_name}")
            existing = self.entities["characters"].get(char_name)
            
            if existing:
                logger.debug(f"发现已存在的人物记录: {char_name}，进行合并")
                # 记录合并前的字段
                before_fields = set(existing.keys())
                # 冲突解决策略：保留最新信息
                existing.update(char)
                # 记录合并后的字段
                after_fields = set(existing.keys())
                new_fields = after_fields - before_fields
                if new_fields:
                    logger.debug(f"人物 {char_name} 新增字段: {new_fields}")
                merged_count += 1
            else:
                logger.debug(f"添加新人物: {char_name}")
                self.entities["characters"][char_name] = char
                new_count += 1
        
        # 事件合并
        events = new_data.get("events", [])
        logger.info(f"合并 {len(events)} 个事件数据")
        self.events.extend(events)
        
        # 关系合并
        relations = new_data.get("relations", [])
        logger.info(f"合并 {len(relations)} 个关系数据")
        self.relations.extend(relations)
        
        logger.info(f"实体合并完成: {merged_count}个已存在人物更新, {new_count}个新人物添加, {invalid_count}个无效人物数据被忽略")
        logger.debug(f"当前知识库状态: {len(self.entities['characters'])}个人物, {len(self.events)}个事件, {len(self.relations)}个关系")
    
    @log_execution_time
    def to_knowledge_graph(self) -> dict:
        """生成知识图谱结构"""
        logger.info(f"开始生成知识图谱结构")
        logger.debug(f"当前知识库状态: {len(self.entities['characters'])}个人物, {len(self.events)}个事件, {len(self.relations)}个关系")
        
        try:
            # 转换实体字典为列表
            characters = list(self.entities["characters"].values())
            logger.debug(f"人物实体转换完成，共 {len(characters)} 个")
            
            result = {
                "characters": characters,
                "events": self.events,
                "relations": self.relations
            }
            
            logger.info(f"知识图谱结构生成完成，包含 {len(result['characters'])} 个人物, {len(result['events'])} 个事件, {len(result['relations'])} 个关系")
            return result
        except Exception as e:
            logger.error(f"生成知识图谱结构失败: {str(e)}")
            logger.exception("知识图谱生成异常详情")
            raise

# ================= 核心处理模块 =================
class NovelAnalyzer:
    EXTRACTION_PROMPT = """请从以下文本中提取：
1. 新人物：[姓名][身份][外貌]
2. 事件：[时间][参与人物][动作][影响]
3. 关系变化：[类型][人物1][人物2]
4. 场景：[地点][时间]

已知信息：
{context}

当前文本：
{text}

用JSON格式返回，必须包含以下字段：
{{
  "characters": [  // 数组，每个元素是一个人物对象
    {{"name": "人物姓名", "identity": "身份", "appearance": "外貌描述"}}  
  ],
  "events": [  // 数组，每个元素是一个事件对象
    {{"time": "事件时间", "participants": ["参与人物1", "参与人物2"], "action": "动作", "impact": "影响"}}
  ],
  "relations": [  // 数组，每个元素是一个关系对象
    {{"type": "关系类型", "from": "人物1", "to": "人物2"}}
  ],
  "scenes": [  // 数组，每个元素是一个场景对象
    {{"location": "地点", "time": "时间"}}
  ]
}}

如果无法提取某个字段的信息，请将该字段设置为空数组，例如 "characters": []。
如果完全无法提取任何信息，请返回：{{"characters": [], "events": [], "relations": [], "scenes": []}}
不要使用``json ```包含返回结果，否则无法解析。
"""
    
    def __init__(self):
        self.processor = TextProcessor()
        self.models = ModelHandler()
        self.knowledge = KnowledgeManager()
        self.context_window = []
    
    @log_execution_time
    def _get_context(self) -> str:
        """获取当前上下文"""
        logger.debug(f"获取上下文，当前上下文窗口大小: {len(self.context_window)}")
        context_size = min(Config.CONTEXT_WINDOW, len(self.context_window))
        context = "\n".join(self.context_window[-context_size:])
        logger.debug(f"返回上下文，长度: {len(context)}字符")
        return context
    
    @log_execution_time
    def analyze_chapter(self, chapter_text: str) -> None:
        logger.info(f"开始分析章节，文本长度: {len(chapter_text)}字符")
        start_time = time.time()
        
        try:
            # 分块处理
            chunks = self.processor.semantic_chunking(chapter_text)
            logger.info(f"章节已分割为 {len(chunks)} 个文本块")
            
            processed_chunks = 0
            cached_chunks = 0
            failed_chunks = 0
            
            for i, chunk in enumerate(tqdm(chunks, desc="Processing Chunks")):
                chunk_start_time = time.time()
                logger.info(f"处理文本块 {i+1}/{len(chunks)}，长度: {len(chunk)}字符")
                
                try:
                    # 检查缓存
                    logger.debug(f"检查文本块缓存状态")
                    cached = self.knowledge.get_cached_result(chunk)
                    if cached:
                        logger.info(f"使用缓存结果，跳过模型调用")
                        self.knowledge.merge_entities(cached)
                        cached_chunks += 1
                        continue
                    
                    # 构建提示词
                    logger.debug(f"构建提示词，获取上下文")
                    context = self._get_context()
                    prompt = self.EXTRACTION_PROMPT.format(
                        context=context,
                        text=chunk
                    )
                    logger.debug(f"提示词构建完成，长度: {len(prompt)}字符")
                    
                    # 模型调用
                    model_start_time = time.time()
                    if len(chunk) > 1000:
                        logger.info(f"文本块较长({len(chunk)}字符)，使用Qwen2模型")
                        result_str = self.models.call_ollama(prompt)
                    else:
                        logger.info(f"文本块较短({len(chunk)}字符)，使用Ollama模型")
                        result_str = self.models.call_ollama(prompt)
                except Exception as e:
                    logger.error(f"处理文本块 {i+1}/{len(chunks)} 时发生错误: {str(e)}")
                    logger.exception("文本块处理异常详情")
                    failed_chunks += 1
                    continue
                
                try:
                    # 检查缓存
                    logger.debug(f"检查文本块缓存状态")
                    cached = self.knowledge.get_cached_result(chunk)
                    if cached:
                        logger.info(f"使用缓存结果，跳过模型调用")
                        self.knowledge.merge_entities(cached)
                        cached_chunks += 1
                        chunk_time = time.time() - chunk_start_time
                        logger.debug(f"文本块处理完成(使用缓存)，耗时: {chunk_time:.4f}秒")
                        continue
                    
                    # 构建提示词
                    logger.debug(f"构建提示词，获取上下文")
                    context = self._get_context()
                    prompt = self.EXTRACTION_PROMPT.format(
                        context=context,
                        text=chunk
                    )
                    logger.debug(f"提示词构建完成，长度: {len(prompt)}字符")
                    
                    # 模型调用
                    model_start_time = time.time()
                    if len(chunk) > 1000:
                        logger.info(f"文本块较长({len(chunk)}字符)，使用Qwen2模型")
                        result_str = self.models.call_ollama(prompt)
                    else:
                        logger.info(f"文本块较短({len(chunk)}字符)，使用Ollama模型")
                        result_str = self.models.call_ollama(prompt)
                    model_time = time.time() - model_start_time
                    logger.info(f"模型调用完成，耗时: {model_time:.2f}秒，返回结果长度: {len(result_str)}字符")
                    
                    # 解析结果
                    parse_start_time = time.time()
                    logger.debug(f"开始解析模型返回结果")
                    try:
                        # 添加容错处理，去掉可能的``json``包含返回结果，否则无法解析
                        cleaned_str = result_str.strip()
                        logger.debug(f"清理前的结果: {cleaned_str[:50]}...")
                        
                        # 移除开头的``json和结尾的```
                        import re
                        cleaned_str = re.sub(r'^\s*```json\s*', '', cleaned_str)
                        cleaned_str = re.sub(r'\s*```\s*$', '', cleaned_str)
                        # 移除其他可能的```
                        cleaned_str = re.sub(r'^\s*```\w*\s*', '', cleaned_str)
                        logger.debug(f"清理后的结果: {cleaned_str[:50]}...")
                        
                        result = json.loads(cleaned_str)
                        logger.debug(f"JSON解析成功，结果类型: {type(result)}")
                        
                        # 确保结果包含所有必要的字段
                        if not isinstance(result, dict):
                            logger.warning(f"警告：模型返回的不是字典: {result}")
                            result = {"characters": [], "events": [], "relations": [], "scenes": []}
                        
                        # 确保所有必要字段都存在
                        for field in ["characters", "events", "relations", "scenes"]:
                            if field not in result or not isinstance(result[field], list):
                                logger.warning(f"警告：字段 {field} 不存在或不是列表，已自动创建")
                                result[field] = []
                        
                        logger.info(f"成功解析模型响应: {str(result)[:100]}...")
                        logger.debug(f"解析结果统计: {len(result.get('characters', []))}个人物, {len(result.get('events', []))}个事件, {len(result.get('relations', []))}个关系, {len(result.get('scenes', []))}个场景")
                        
                        # 合并实体并缓存结果
                        merge_start_time = time.time()
                        self.knowledge.merge_entities(result)
                        merge_time = time.time() - merge_start_time
                        logger.debug(f"实体合并完成，耗时: {merge_time:.4f}秒")
                        
                        cache_start_time = time.time()
                        self.knowledge.cache_result(chunk, result)
                        cache_time = time.time() - cache_start_time
                        logger.debug(f"结果缓存完成，耗时: {cache_time:.4f}秒")
                        
                        self.context_window.append(chunk)
                        processed_chunks += 1
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON解析失败: {str(e)}")
                        logger.error(f"原始结果: {result_str[:200]}...")
                        result = {"characters": [], "events": [], "relations": [], "scenes": []}
                        self.context_window.append(chunk)
                        failed_chunks += 1
                    
                    parse_time = time.time() - parse_start_time
                    logger.debug(f"结果解析处理完成，耗时: {parse_time:.4f}秒")
                    
                except Exception as e:
                    logger.error(f"处理文本块时发生异常: {str(e)}")
                    logger.exception("文本块处理异常详情")
                    failed_chunks += 1
                
                chunk_time = time.time() - chunk_start_time
                logger.info(f"文本块 {i+1}/{len(chunks)} 处理完成，总耗时: {chunk_time:.2f}秒")
            
            chapter_time = time.time() - start_time
            logger.info(f"章节分析完成，总耗时: {chapter_time:.2f}秒，处理 {len(chunks)} 个文本块，其中 {processed_chunks} 个成功，{cached_chunks} 个使用缓存，{failed_chunks} 个失败")
        
        except Exception as e:
            logger.error(f"章节分析失败: {str(e)}")
            logger.exception("章节分析异常详情")
            raise
    
    @log_execution_time
    def analyze_novel(self, novel_path: str) -> dict:
        logger.info(f"开始小说分析，文件路径: {novel_path}")
        total_start_time = time.time()
        
        try:
            # 读取文件
            logger.debug(f"开始读取小说文件")
            file_start_time = time.time()
            try:
                with open(novel_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                file_time = time.time() - file_start_time
                logger.info(f"文件读取成功，大小: {len(text)}字符，耗时: {file_time:.2f}秒")
            except UnicodeDecodeError:
                logger.warning(f"UTF-8编码读取失败，尝试使用其他编码")
                with open(novel_path, 'r', encoding='gb18030') as f:
                    text = f.read()
                file_time = time.time() - file_start_time
                logger.info(f"文件读取成功(gb18030编码)，大小: {len(text)}字符，耗时: {file_time:.2f}秒")
            except Exception as e:
                logger.error(f"文件读取失败: {str(e)}")
                raise
            
            # 分割章节
            logger.debug(f"开始分割章节")
            chapters = self.processor.split_chapters(text)
            logger.info(f"找到 {len(chapters)} 个章节，平均章节长度: {sum(len(c) for c in chapters)/len(chapters):.2f}字符")
            
            # 处理每个章节
            for i, chap in enumerate(chapters):
                logger.info(f"开始处理第 {i+1}/{len(chapters)} 章，长度: {len(chap)}字符")
                chapter_start_time = time.time()
                try:
                    self.analyze_chapter(chap)
                    chapter_time = time.time() - chapter_start_time
                    logger.info(f"第 {i+1}/{len(chapters)} 章处理完成，耗时: {chapter_time:.2f}秒")
                except Exception as e:
                    logger.error(f"处理第 {i+1} 章时发生错误: {str(e)}")
                    logger.exception(f"章节处理异常详情")
            
            # 生成知识图谱
            logger.info(f"所有章节处理完成，开始生成知识图谱")
            kg_start_time = time.time()
            result = self.knowledge.to_knowledge_graph()
            kg_time = time.time() - kg_start_time
            
            # 统计结果
            char_count = len(result.get('characters', []))
            event_count = len(result.get('events', []))
            relation_count = len(result.get('relations', []))
            
            total_time = time.time() - total_start_time
            logger.info(f"小说分析完成，总耗时: {total_time:.2f}秒，提取了 {char_count} 个人物, {event_count} 个事件, {relation_count} 个关系")
            return result
            
        except Exception as e:
            logger.error(f"小说分析过程中发生错误: {str(e)}")
            logger.exception("小说分析异常详情")
            raise

# ================= 可视化模块 =================
class Visualizer:
    @staticmethod
    @log_execution_time
    def generate_html(knowledge: dict, output_path: str) -> None:
        """生成可视化HTML报告"""
        logger.info(f"开始生成可视化HTML报告，输出路径: {output_path}")
        
        try:
            # 验证知识图谱数据
            if not isinstance(knowledge, dict):
                logger.error(f"无效的知识图谱数据类型: {type(knowledge)}")
                raise ValueError("知识图谱必须是字典类型")
                
            characters = knowledge.get('characters', [])
            relations = knowledge.get('relations', [])
            logger.info(f"知识图谱包含 {len(characters)} 个人物和 {len(relations)} 个关系")
            
            # 生成节点数据
            start_time = time.time()
            try:
                nodes = [
                    {'name': c['name'], 'category': '人物'} 
                    for c in characters
                ]
                logger.debug(f"节点数据生成完成，共 {len(nodes)} 个节点")
            except Exception as e:
                logger.error(f"生成节点数据失败: {str(e)}")
                nodes = []
            
            # 生成连接数据
            try:
                links = [
                    {'source': r['from'], 'target': r['to'], 'name': r['type']}
                    for r in relations
                ]
                logger.debug(f"连接数据生成完成，共 {len(links)} 个连接")
            except Exception as e:
                logger.error(f"生成连接数据失败: {str(e)}")
                links = []
            
            data_time = time.time() - start_time
            logger.debug(f"数据准备完成，耗时: {data_time:.4f}秒")
            
            # 生成HTML模板
            html_template = f"""
            <html>
            <head>
                <title>小说知识图谱</title>
                <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.2/dist/echarts.min.js"></script>
            </head>
            <body>
                <div id="main" style="width: 1200px;height:800px;"></div>
                <script>
                    var chart = echarts.init(document.getElementById('main'));
                    var nodes = {json.dumps(nodes)};
                    
                    var links = {json.dumps(links)};
                    
                    var option = {{
                        title: {{ text: '知识图谱可视化' }},
                        tooltip: {{}},
                        legend: {{ data: ['人物', '事件'] }},
                        series: [{{
                            type: 'graph',
                            layout: 'force',
                            data: nodes,
                            links: links,
                            force: {{ repulsion: 100 }}
                        }}]
                    }};
                    chart.setOption(option);
                </script>
            </body>
            </html>
            """
            
            # 写入文件
            logger.debug(f"开始写入HTML文件: {output_path}")
            write_start_time = time.time()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_template)
            write_time = time.time() - write_start_time
            
            logger.info(f"可视化HTML报告生成完成，文件大小: {len(html_template)}字节，写入耗时: {write_time:.4f}秒")
        except Exception as e:
            logger.error(f"生成可视化HTML报告失败: {str(e)}")
            logger.exception("可视化生成异常详情")
            raise

def main():
    program_start_time = time.time()
    logger.info("========== 小说分析器启动 ==========")
    logger.info(f"日志级别: {logging.getLevelName(logger.level)}")
    logger.info(f"配置信息: Ollama端点={Config.OLLAMA_ENDPOINT}, Redis={Config.REDIS_HOST}:{Config.REDIS_PORT}")
    
    try:
        # 初始化分析器
        logger.info("初始化小说分析器")
        analyzer = NovelAnalyzer()
        
        # 设置小说路径
        novel_path = "/Users/shhaofu/Code/cursor-projects/aka_music/backend/app/resource/novel/tian_long_ba_bu.txt"
        logger.info(f"设置小说路径: {novel_path}")
        
        # 执行分析
        logger.info("开始执行小说分析")
        analysis_start_time = time.time()
        knowledge = analyzer.analyze_novel(novel_path)
        analysis_time = time.time() - analysis_start_time
        logger.info(f"小说分析完成，耗时: {analysis_time:.2f}秒")
        
        # 保存结果
        logger.info("开始保存知识图谱到JSON文件")
        save_start_time = time.time()
        output_file = "knowledge.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(knowledge, f, ensure_ascii=False, indent=2)
            save_time = time.time() - save_start_time
            logger.info(f"知识图谱已保存到 {output_file}，耗时: {save_time:.2f}秒")
        except Exception as e:
            logger.error(f"保存知识图谱失败: {str(e)}")
            logger.exception("保存异常详情")
        
        # 生成可视化
        logger.info("开始生成可视化HTML")
        vis_start_time = time.time()
        vis_file = "visualization.html"
        try:
            Visualizer.generate_html(knowledge, vis_file)
            vis_time = time.time() - vis_start_time
            logger.info(f"可视化HTML已生成到 {vis_file}，耗时: {vis_time:.2f}秒")
        except Exception as e:
            logger.error(f"生成可视化失败: {str(e)}")
            logger.exception("可视化异常详情")
        
        # 程序完成
        program_time = time.time() - program_start_time
        logger.info(f"========== 小说分析器完成 ==========，总耗时: {program_time:.2f}秒")
        
    except Exception as e:
        logger.error(f"程序执行过程中发生错误: {str(e)}")
        logger.exception("程序异常详情")
        program_time = time.time() - program_start_time
        logger.info(f"========== 小说分析器异常退出 ==========，运行时间: {program_time:.2f}秒")

@log_execution_time
def generate_relations_json_for_echarts(knowledge_path="/Users/shhaofu/Code/cursor-projects/aka_music/knowledge.json"):
    """从knowledge.json中读取数据，构造适合ECharts展示的数据结构
    
    返回:
        dict: 包含nodes和links两个键的字典，分别对应ECharts图表所需的节点和连接数据
    """
    logger.info(f"开始从{knowledge_path}读取知识图谱数据")
    
    try:
        # 读取knowledge.json文件
        with open(knowledge_path, 'r', encoding='utf-8') as f:
            knowledge = json.load(f)
        
        logger.info(f"成功读取知识图谱数据，开始构造ECharts数据结构")
        
        # 验证知识图谱数据
        if not isinstance(knowledge, dict):
            logger.error(f"无效的知识图谱数据类型: {type(knowledge)}")
            raise ValueError("知识图谱必须是字典类型")
        
        characters = knowledge.get('characters', [])
        relations = knowledge.get('relations', [])
        logger.info(f"知识图谱包含 {len(characters)} 个人物和 {len(relations)} 个关系")
        
        # 构造节点数据
        start_time = time.time()
        try:
            nodes = [
                {'name': c.get('name', ''), 'category': c.get('identity', '人物') or '人物', 'value': c.get('appearance', '')}
                for c in characters
                if c.get('name')
            ]
            logger.debug(f"节点数据生成完成，共 {len(nodes)} 个节点")
        except Exception as e:
            logger.error(f"生成节点数据失败: {str(e)}")
            nodes = []
        
        # 构造连接数据
        try:
            links = [
                {"source": r.get('from', ''), "target": r.get('to', ''), "name": r.get('type', '')}
                for r in relations
                if r.get('from') and r.get('to') and r.get('type')
            ]
            logger.debug(f"连接数据生成完成，共 {len(links)} 个连接")
        except Exception as e:
            logger.error(f"生成连接数据失败: {str(e)}")
            links = []
        
        data_time = time.time() - start_time
        logger.debug(f"数据准备完成，耗时: {data_time:.4f}秒")
        
        # 返回完整的图表数据结构
        result = {
            'nodes': nodes,
            'links': links
        }
        
        logger.info(f"ECharts数据构造完成，共 {len(nodes)} 个节点和 {len(links)} 个关系")
        return result
    except Exception as e:
        logger.error(f"生成ECharts数据失败: {str(e)}")
        logger.exception("ECharts数据生成异常详情")
        return {'nodes': [], 'links': []}
    
    # 示例输出格式：
    '''
    {
        "nodes": [
            {"name": "段誉", "category": "大理王子", "value": "年轻男子"},
            {"name": "王语嫣", "category": "人物", "value": "美丽女子"},
            ...
        ],
        "links": [
            {"source": "段誉", "target": "钟灵", "name": "喜欢"},
            {"source": "段誉", "target": "木婉清", "name": "喜欢"},
            {"source": "段誉", "target": "王语嫣", "name": "爱慕"},
            {"source": "段正淳", "target": "段誉", "name": "父子"},
            {"source": "刀白凤", "target": "段誉", "name": "母子"},
            {"source": "段正明", "target": "段誉", "name": "叔侄"},
            {"source": "无量剑", "target": "辛双清", "name": "门派关系"},
            {"source": "无量剑", "target": "左子穆", "name": "门派关系"},
            {"source": "无量剑", "target": "龚光杰", "name": "门派关系"},
            {"source": "无量剑", "target": "干光豪", "name": "门派关系"}
        ]
    }
    '''

    
    

# ================= 执行入口 =================
if __name__ == "__main__":
    main()