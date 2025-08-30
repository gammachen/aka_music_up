import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# 创建临时目录用于模型权重卸载
offload_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_offload")
os.makedirs(offload_folder, exist_ok=True)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("FreedomIntelligence/HuatuoGPT2-7B", use_fast=True, trust_remote_code=True)

# 修改模型加载参数，添加offload_folder并使用4位量化减少内存占用
model = AutoModelForCausalLM.from_pretrained(
    "FreedomIntelligence/HuatuoGPT2-7B", 
    device_map={"":"cpu"},  # 强制使用CPU
    offload_folder=offload_folder,
    low_cpu_mem_usage=True,  # 低内存使用模式
    torch_dtype=torch.float16,  # 使用float16替代bfloat16，兼容性更好
    trust_remote_code=True
)

# 准备对话消息
messages = []
messages.append({"role": "user", "content": "肚子疼怎么办？"})

# 调用模型生成回复
response = model.HuatuoChat(tokenizer, messages)
print(response)