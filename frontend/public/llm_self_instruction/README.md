以下是对基于**Self-Instruct方法**的数据构造框架的详细说明，包含实施步骤和核心代码示例：

---

### **Self-Instruct框架概述**
Self-Instruct是一种利用预训练语言模型（LLM）自动生成指令遵循数据的方法，核心流程包括：
1. **指令生成**：LLM基于种子指令生成新指令
2. **输入-输出生成**：为生成的指令创建输入和输出对
3. **过滤与后处理**：清洗低质量数据
4. **迭代优化**：将生成数据加入种子池循环优化

---

### **实施步骤详解**
#### **步骤1：准备种子指令**
```python
seed_instructions = [
    {"instruction": "翻译成法语", "input": "Hello world", "output": "Bonjour le monde"},
    {"instruction": "分类情感", "input": "这部电影太棒了", "output": "积极"},
    # 初始种子数据（5-10条）
]
```

#### **步骤2：指令生成（Prompt模板）**
```python
prompt_template = """
你是一个指令生成器。请基于以下示例生成{num_prompts}条新的、多样化的任务指令：
{seed_examples}

新指令要求：
1. 避免重复示例中的指令
2. 覆盖不同领域（写作、翻译、编码等）
3. 使用自然语言描述任务

生成的指令列表：
"""
```

#### **步骤3：输入-输出生成**
```python
def generate_instance(instruction, model):
    prompt = f"""
    根据指令生成输入和输出：
    指令：{instruction}
    输入：<在此生成任务输入>
    输出：<在此生成任务输出>
    
    要求：
    1. 如果任务不需要输入，填写"无"
    2. 输出必须直接完成任务
    """
    return model.generate(prompt)
```

#### **步骤4：数据过滤规则**
```python
def is_valid_data(instance):
    # 规则1：指令长度检测
    if len(instance["instruction"]) < 5: 
        return False
    
    # 规则2：输出相关性检测
    if instance["output"].lower() in ["n/a", "我不知道", ""]:
        return False
    
    # 规则3：关键词黑名单过滤
    blacklist = ["色情", "暴力", "仇恨言论"]
    if any(word in instance["instruction"] for word in blacklist):
        return False
    
    return True
```

#### **步骤5：迭代优化流程**
```python
def self_instruct_bootstrap(seed_data, model, iterations=3):
    current_pool = seed_data.copy()
    
    for _ in range(iterations):
        # 1. 指令生成
        new_instructions = generate_instructions(current_pool, model)
        
        # 2. 实例化
        new_data = []
        for inst in new_instructions:
            instance = generate_instance(inst, model)
            if is_valid_data(instance):
                new_data.append(instance)
        
        # 3. 加入数据池（控制重复）
        current_pool += deduplicate(new_data, current_pool)
    
    return current_pool
```

---

### **完整代码框架**
```python
import openai
import json
from tqdm import tqdm

# 配置OpenAI API
openai.api_key = "YOUR_API_KEY"
MODEL_ENGINE = "gpt-3.5-turbo"

def call_llm(prompt):
    response = openai.ChatCompletion.create(
        model=MODEL_ENGINE,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=256
    )
    return response.choices[0].message.content.strip()

def generate_instructions(seed_data, num_to_generate=10):
    seed_examples = "\n".join([d["instruction"] for d in seed_data[:3]])
    prompt = prompt_template.format(
        num_prompts=num_to_generate,
        seed_examples=seed_examples
    )
    response = call_llm(prompt)
    return [line.split(". ", 1)[1] for line in response.split("\n") if ". " in line]

def generate_instance(instruction):
    prompt = f"指令：{instruction}\n生成符合要求的输入和输出：\n输入："
    response = call_llm(prompt)
    
    # 解析输入输出
    if "输出：" in response:
        input_part, output_part = response.split("输出：", 1)
        return {
            "instruction": instruction,
            "input": input_part.replace("输入：", "").strip(),
            "output": output_part.strip()
        }
    return None

# 主流程
seed_data = [...]  # 加载种子数据
final_data = seed_data.copy()

for _ in range(3):  # 3轮迭代
    new_instructions = generate_instructions(final_data)
    for inst in tqdm(new_instructions):
        instance = generate_instance(inst)
        if instance and is_valid_data(instance):
            final_data.append(instance)
    
    # 保存检查点
    with open(f"self_instruct_iter_{_}.json", "w") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

print(f"生成完成！共{len(final_data)}条数据")
```

---

### **关键优化技术**
1. **多样性控制**
   ```python
   # 在提示词中强调多样性要求
   prompt += "特别注意：生成至少三种不同任务类型（如创意写作、信息提取、逻辑推理）"
   ```

2. **语义去重**
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
   
   def semantic_deduplicate(new_data, pool, threshold=0.8):
       pool_embeddings = model.encode([d["instruction"] for d in pool])
       new_embeddings = model.encode([d["instruction"] for d in new_data])
       # 余弦相似度过滤...
   ```

3. **质量评分模型**
   ```python
   # 训练一个二分类器评估数据质量
   # 特征包括：指令长度、输出长度、关键词匹配度等
   ```

---

### **生成数据示例**
```json
[
  {
    "instruction": "用Python实现快速排序",
    "input": "无",
    "output": "def quicksort(arr): ..."
  },
  {
    "instruction": "将以下科技新闻总结为100字摘要",
    "input": "OpenAI发布新一代语言模型...",
    "output": "摘要内容..."
  }
]
```

---

### **注意事项**
1. **成本控制**：使用`gpt-3.5-turbo`比`gpt-4`成本低10倍
2. **安全过滤**：必须添加内容安全过滤器
3. **人工审核**：建议对10%生成数据进行人工校验
4. **领域适配**：可通过修改种子指令控制生成方向

> 实际论文中，使用175个种子指令经过5轮迭代生成52K指令数据，在Alpaca等模型中验证有效提升指令遵循能力。
>
## Codes

```python
import json
import os
from typing import List, Dict, Any

class Config:
    """配置类，负责加载和管理配置"""
    
    def __init__(self, config_path: str = None):
        # 默认配置
        self.api_key = "YOUR_API_KEY"
        self.base_url = "https://api.openai.com/v1"
        self.model = "gpt-3.5-turbo"
        self.temperature = 0.7
        self.max_tokens = 256
        self.retry_count = 3
        self.retry_delay = 5
        self.num_seed_examples = 3
        self.min_instruction_length = 5
        self.min_output_length = 5
        self.blacklist_keywords = ["色情", "暴力", "仇恨言论", "歧视", "政治", "宗教"]
        
        # 如果提供了配置文件路径，则加载配置
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 更新配置
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            print(f"成功加载配置: {config_path}")
        except Exception as e:
            print(f"加载配置失败: {e}，使用默认配置")
    
    def save_config(self, config_path: str) -> None:
        """保存配置到文件
        
        Args:
            config_path: 配置文件保存路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # 获取所有非内置属性
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_dict[key] = value
        
        # 保存到文件
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        print(f"配置已保存到: {config_path}")
    
    def __str__(self) -> str:
        """返回配置的字符串表示"""
        config_str = "配置信息:\n"
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_str += f"  {key}: {value}\n"
        return config_str
```

```python
filter.py
from typing import Dict, List, Any
import re

class DataFilter:
    """数据过滤器，负责清洗低质量数据"""
    
    def __init__(self, config):
        self.config = config
        self.blacklist = config.blacklist_keywords
        self.min_instruction_length = config.min_instruction_length
        self.min_output_length = config.min_output_length
        self.invalid_outputs = ["n/a", "我不知道", "不知道", "无法回答", "无法提供", "抱歉", ""]
    
    def is_valid(self, instance: Dict[str, str]) -> bool:
        """检查数据实例是否有效
        
        Args:
            instance: 包含指令、输入和输出的字典
            
        Returns:
            数据是否有效
        """
        # 规则1：指令长度检测
        if len(instance["instruction"]) < self.min_instruction_length:
            return False
        
        # 规则2：输出长度检测
        if len(instance["output"]) < self.min_output_length:
            return False
        
        # 规则3：输出相关性检测
        if instance["output"].lower() in self.invalid_outputs:
            return False
        
        # 规则4：关键词黑名单过滤
        if self._contains_blacklist_keywords(instance["instruction"]):
            return False
        
        # 规则5：输出中不应包含抱歉、歉意等表达
        if self._contains_apology(instance["output"]):
            return False
        
        return True
    
    def _contains_blacklist_keywords(self, text: str) -> bool:
        """检查文本是否包含黑名单关键词"""
        return any(word in text.lower() for word in self.blacklist)
    
    def _contains_apology(self, text: str) -> bool:
        """检查输出是否包含道歉或拒绝回答的表达"""
        apology_patterns = [
            r"抱歉", r"对不起", r"很遗憾", r"无法回答", r"无法提供",
            r"sorry", r"apologize", r"cannot answer", r"can't provide"
        ]
        return any(re.search(pattern, text.lower()) for pattern in apology_patterns)
    
    def filter_batch(self, instances: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """批量过滤数据
        
        Args:
            instances: 数据实例列表
            
        Returns:
            过滤后的数据实例列表
        """
        return [inst for inst in instances if self.is_valid(inst)]
```

```python
generator.py
import json
import random
from typing import List, Dict, Any

from .llm import LLMClient
from .utils import deduplicate_instructions

class InstructionGenerator:
    """指令生成器，负责基于种子指令生成新的指令"""
    
    def __init__(self, config):
        self.config = config
        self.llm_client = LLMClient(config)
        self.prompt_template = """
你是一个指令生成器。请基于以下示例生成{num_prompts}条新的、多样化的任务指令：
{seed_examples}

新指令要求：
1. 避免重复示例中的指令
2. 覆盖不同领域（写作、翻译、编码等）
3. 使用自然语言描述任务
4. 每条指令应该是独立的，不要有编号
5. 特别注意：生成至少三种不同任务类型（如创意写作、信息提取、逻辑推理）

生成的指令列表：
"""
    
    def generate(self, seed_data: List[Dict[str, Any]], num_to_generate: int = 10) -> List[str]:
        """生成新指令
        
        Args:
            seed_data: 种子数据列表
            num_to_generate: 要生成的指令数量
            
        Returns:
            生成的新指令列表
        """
        # 随机选择种子示例
        sample_size = min(self.config.num_seed_examples, len(seed_data))
        seed_samples = random.sample(seed_data, sample_size)
        
        # 提取指令部分
        seed_examples = "\n".join([f"- {d['instruction']}" for d in seed_samples])
        
        # 构建提示词
        prompt = self.prompt_template.format(
            num_prompts=num_to_generate,
            seed_examples=seed_examples
        )
        
        # 调用LLM生成
        response = self.llm_client.generate(prompt)
        
        # 解析响应获取指令列表
        instructions = self._parse_instructions(response)
        
        # 去重
        existing_instructions = [d["instruction"] for d in seed_data]
        unique_instructions = deduplicate_instructions(instructions, existing_instructions)
        
        return unique_instructions
    
    def _parse_instructions(self, response: str) -> List[str]:
        """解析LLM响应，提取指令列表"""
        instructions = []
        for line in response.strip().split("\n"):
            line = line.strip()
            # 跳过空行
            if not line:
                continue
            # 移除行首的序号和符号
            if line.startswith(("-", "*", "•")):
                line = line[1:].strip()
            elif any(line.startswith(f"{i}.") for i in range(1, 100)):
                line = line.split(".", 1)[1].strip()
            
            if line and len(line) > 5:  # 简单过滤太短的指令
                instructions.append(line)
        
        return instructions


class InstanceGenerator:
    """实例生成器，负责为指令生成输入-输出对"""
    
    def __init__(self, config):
        self.config = config
        self.llm_client = LLMClient(config)
        self.prompt_template = """
根据指令生成输入和输出：
指令：{instruction}

要求：
1. 如果任务不需要输入，填写"无"
2. 输出必须直接完成任务
3. 输入应该是真实、多样化的
4. 输出应该是高质量、有帮助的

请按以下格式回复：
输入：<在此生成任务输入>
输出：<在此生成任务输出>
"""
    
    def generate(self, instruction: str) -> Dict[str, str]:
        """为指令生成输入-输出对
        
        Args:
            instruction: 指令文本
            
        Returns:
            包含指令、输入和输出的字典，如果生成失败则返回None
        """
        # 构建提示词
        prompt = self.prompt_template.format(instruction=instruction)
        
        # 调用LLM生成
        response = self.llm_client.generate(prompt)
        
        # 解析响应
        try:
            instance = self._parse_instance(instruction, response)
            return instance
        except Exception as e:
            print(f"解析实例失败: {e}")
            return None
    
    def _parse_instance(self, instruction: str, response: str) -> Dict[str, str]:
        """解析LLM响应，提取输入和输出"""
        input_text = ""
        output_text = ""
        
        # 解析输入和输出
        if "输入：" in response and "输出：" in response:
            parts = response.split("输出：")
            output_text = parts[1].strip()
            input_text = parts[0].replace("输入：", "").strip()
        else:
            # 尝试其他可能的格式
            if "Input:" in response and "Output:" in response:
                parts = response.split("Output:")
                output_text = parts[1].strip()
                input_text = parts[0].replace("Input:", "").strip()
        
        # 验证解析结果
        if not output_text:
            raise ValueError("无法解析输出")
        
        return {
            "instruction": instruction,
            "input": input_text if input_text else "无",
            "output": output_text
        }
```

```python
llm.py

import time
from typing import Dict, Any, Optional

import openai
from openai import OpenAI

class LLMClient:
    """LLM客户端，负责与语言模型API交互"""

    def __init__(self, config):
        self.config = config
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.retry_count = config.retry_count
        self.retry_delay = config.retry_delay

        # 初始化API客户端
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本
        
        Args:
            prompt: 提示词
            **kwargs: 其他参数，会覆盖默认配置
            
        Returns:
            生成的文本
        """
        params = {
            "model": kwargs.get("model", self.model),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        # 重试机制
        for attempt in range(self.retry_count):
            try:
                return self._call_api(prompt, params)
            except Exception as e:
                if attempt < self.retry_count - 1:
                    print(f"API调用失败: {e}，{self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    raise e

    def _call_api(self, prompt: str, params: Dict[str, Any]) -> str:
        """调用OpenAI API"""
        response = self.client.chat.completions.create(
            model=params["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"]
        )
        return response.choices[0].message.content.strip()

    def batch_generate(self, prompts: list, **kwargs) -> list:
        """批量生成文本
        
        Args:
            prompts: 提示词列表
            **kwargs: 其他参数
            
        Returns:
            生成的文本列表
        """
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results
```

```python
utils.py

import json
import logging
import os
from typing import List, Dict, Any

def setup_logger():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('self_instruct.log')
        ]
    )
    return logging.getLogger('self_instruct')

def load_json(file_path: str) -> List[Dict[str, Any]]:
    """加载JSON文件
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        JSON数据
    """
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: List[Dict[str, Any]], file_path: str) -> None:
    """保存JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def deduplicate_instructions(new_instructions: List[str], existing_instructions: List[str]) -> List[str]:
    """指令去重
    
    Args:
        new_instructions: 新生成的指令列表
        existing_instructions: 已有的指令列表
        
    Returns:
        去重后的新指令列表
    """
    # 简单的文本匹配去重
    unique_instructions = []
    existing_lower = [inst.lower() for inst in existing_instructions]
    
    for inst in new_instructions:
        if inst.lower() not in existing_lower:
            unique_instructions.append(inst)
            existing_lower.append(inst.lower())
    
    return unique_instructions

def semantic_deduplicate(new_data: List[Dict[str, Any]], pool: List[Dict[str, Any]], threshold: float = 0.8):
    """语义去重
    
    Args:
        new_data: 新数据
        pool: 已有数据池
        threshold: 相似度阈值
        
    Returns:
        去重后的新数据
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        # 加载模型
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # 提取指令
        pool_instructions = [d["instruction"] for d in pool]
        new_instructions = [d["instruction"] for d in new_data]
        
        # 计算嵌入
        pool_embeddings = model.encode(pool_instructions)
        new_embeddings = model.encode(new_instructions)
        
        # 计算相似度并过滤
        unique_data = []
        for i, new_item in enumerate(new_data):
            # 计算与池中所有指令的相似度
            similarities = np.dot(new_embeddings[i], pool_embeddings.T) / \
                          (np.linalg.norm(new_embeddings[i]) * np.linalg.norm(pool_embeddings, axis=1))
            
            # 如果最大相似度低于阈值，则认为是唯一的
            if np.max(similarities) < threshold:
                unique_data.append(new_item)
        
        return unique_data
    except ImportError:
        print("警告: sentence-transformers未安装，使用简单文本匹配去重")
        return deduplicate_instructions_dict(new_data, pool)

def deduplicate_instructions_dict(new_data: List[Dict[str, Any]], pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """基于字典的指令去重"""
    unique_data = []
    pool_instructions = [d["instruction"].lower() for d in pool]
    
    for item in new_data:
        if item["instruction"].lower() not in pool_instructions:
            unique_data.append(item)
            pool_instructions.append(item["instruction"].lower())
    
    return unique_data

```

```python
default.json
{
  "api_key": "ollama",
  "base_url": "http://localhost:11434/v1",
  "model": "qwen2:latest",
  "temperature": 0.7,
  "max_tokens": 256,
  "retry_count": 3,
  "retry_delay": 5,
  "num_seed_examples": 3,
  "min_instruction_length": 5,
  "min_output_length": 5,
  "blacklist_keywords": ["色情", "暴力", "仇恨言论", "歧视", "政治", "宗教"]
}
```

```python
data/seed_instructions.json

[
  {
    "instruction": "翻译成法语",
    "input": "Hello world",
    "output": "Bonjour le monde"
  },
  {
    "instruction": "分类情感",
    "input": "这部电影太棒了",
    "output": "积极"
  },
  {
    "instruction": "总结以下文本",
    "input": "人工智能（AI）是计算机科学的一个分支，它致力于创造能够模拟人类智能的机器。这包括学习、推理、问题解决、感知和语言理解等能力。AI的发展可以追溯到20世纪50年代，但近年来由于计算能力的提升、大数据的可用性和算法的改进，AI取得了显著进展。",
    "output": "人工智能是模拟人类智能的计算机科学分支，包括学习、推理等能力，近年因计算能力提升和大数据可用性而发展迅速。"
  },
  {
    "instruction": "用Python实现快速排序算法",
    "input": "无",
    "output": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)"
  },
  {
    "instruction": "解释量子计算的基本原理",
    "input": "无",
    "output": "量子计算利用量子力学原理进行信息处理。传统计算机使用比特（0或1），而量子计算机使用量子比特，可以同时处于多个状态（叠加态）。这使得量子计算机能够并行处理大量可能性，理论上可以解决传统计算机难以处理的特定问题，如大数分解和复杂系统模拟。"
  },
  {
    "instruction": "创作一首关于春天的短诗",
    "input": "无",
    "output": "春风轻抚大地醒，\n百花争艳竞相迎。\n绿意盎然山与水，\n鸟语花香满心情。"
  },
  {
    "instruction": "将以下文本翻译成英文",
    "input": "人工智能正在改变我们的生活方式和工作方式。",
    "output": "Artificial intelligence is changing the way we live and work."
  },
  {
    "instruction": "解释以下代码的功能",
    "input": "function fibonacci(n) {\n  if (n <= 1) return n;\n  return fibonacci(n-1) + fibonacci(n-2);\n}",
    "output": "这是一个计算斐波那契数列的递归函数。斐波那契数列是一个数列，其中每个数是前两个数的和。这个函数接受一个参数n，如果n小于等于1，则直接返回n；否则，返回fibonacci(n-1)和fibonacci(n-2)的和。"
  },
  {
    "instruction": "提供三个减少碳足迹的方法",
    "input": "无",
    "output": "1. 减少肉类消费，增加植物性食品的比例，因为畜牧业是温室气体排放的主要来源之一。\n2. 选择公共交通、自行车或步行代替私家车出行，减少交通工具产生的碳排放。\n3. 提高能源效率，使用节能电器，并尽可能使用可再生能源，如太阳能或风能。"
  },
  {
    "instruction": "分析以下句子的语法结构",
    "input": "尽管天气不好，他们还是决定去远足。",
    "output": "这是一个复合句，由两个分句组成：\n1. 从属分句：'尽管天气不好'，是一个让步状语从句，由连词'尽管'引导。\n2. 主句：'他们还是决定去远足'。\n'尽管'表示让步关系，意味着即使天气不好（这通常会阻止远足），他们仍然决定去远足。"
  }
]
```
