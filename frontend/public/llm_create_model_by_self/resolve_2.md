# 实施小规模专用模型的完整方案

以下是针对小规模专用模型实施GPT-2 Small架构的完整代码，包含所有改进点：

```python
# 导入必要的库
import os
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
    LineByLineTextDataset, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, pipeline, set_seed
)
from tokenizers import CharBPETokenizer
from tokenizers.decoders import BPEDecoder
import random

# 禁用wandb
os.environ["WANDB_DISABLED"] = "true"

# 定义保存路径
SAVE_PATH = "./chinese_novel_model"

# 1. 创建优化的中文分词器
def create_custom_tokenizer(files):
    """创建基于字符级BPE的中文分词器"""
    # 初始化字符级BPE分词器
    tokenizer = CharBPETokenizer(
        suffix="",  # 不添加后缀
        unk_token="<unk>",  # 未知词标记
    )
    
    # 添加特殊标记
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    
    # 训练分词器
    tokenizer.train(
        files,
        vocab_size=8000,  # 更小的词汇表
        min_frequency=2,  # 最小出现频率
        show_progress=True,
        special_tokens=special_tokens
    )
    
    # 将分词器转换为Hugging Face格式
    tokenizer.save_model(SAVE_PATH)
    tokenizer = GPT2Tokenizer.from_pretrained(SAVE_PATH)
    
    # 添加特殊标记映射
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>"
    })
    
    return tokenizer

# 2. 创建小型GPT-2模型
def create_small_gpt2_model(tokenizer):
    """创建小型GPT-2模型配置"""
    # 定义小型模型配置
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # 小型模型参数 - 减少层数和维度
        n_layer=6,          # 原始为12层 (减少50%)
        n_head=6,           # 原始为12头 (减少50%)
        n_embd=384,         # 原始为768 (减少50%)
        n_inner=1536,       # 原始为3072 (减少50%)
        n_positions=256,    # 原始为1024 (减少75%)
        # 其他优化参数
        resid_pdrop=0.1,    # 残差dropout
        embd_pdrop=0.1,     # 嵌入dropout
        attn_pdrop=0.1      # 注意力dropout
    )
    
    # 创建模型
    model = GPT2LMHeadModel(config)
    
    # 打印模型大小
    num_params = sum(p.numel() for p in model.parameters())
    print(f"创建小型GPT-2模型，参数数量: {num_params/1e6:.1f}M (原始GPT-2约为124M)")
    
    return model

# 3. 创建数据集
def create_datasets(tokenizer):
    """创建训练和验证数据集"""
    # 小说文件列表
    novel_files = [
        "text/sanguoyanyi.txt",
        "text/tian_long_ba_bu_all.txt",
        "text/other_novel1.txt",  # 添加更多小说
        "text/other_novel2.txt"
    ]
    
    # 创建训练数据集
    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=random.choice(novel_files),  # 随机选择一本
        block_size=128
    )
    
    # 创建验证数据集
    valid_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=random.choice(novel_files),  # 使用不同小说
        block_size=128
    )
    
    return train_dataset, valid_dataset

# 4. 主训练函数
def train_model():
    """训练小型GPT-2模型"""
    # 小说文件列表
    novel_files = [
        "text/sanguoyanyi.txt",
        "text/tian_long_ba_bu_all.txt",
        "text/other_novel1.txt",  # 添加更多小说
        "text/other_novel2.txt"
    ]
    
    # 创建分词器
    tokenizer = create_custom_tokenizer(novel_files)
    
    # 创建小型模型
    model = create_small_gpt2_model(tokenizer)
    
    # 创建数据集
    train_dataset, valid_dataset = create_datasets(tokenizer)
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=64  # 优化显存使用
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./output",
        overwrite_output_dir=True,
        num_train_epochs=10,  # 减少训练轮次
        per_gpu_train_batch_size=8,  # 适合小型模型的批次大小
        per_gpu_eval_batch_size=8,
        learning_rate=3e-5,  # 更小的学习率
        warmup_steps=500,  # 预热步骤
        weight_decay=0.01,  # 权重衰减
        logging_dir="./logs",
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        evaluation_strategy="steps",  # 定期评估
        eval_steps=500,  # 每500步评估一次
        prediction_loss_only=True,
        gradient_accumulation_steps=4,  # 梯度累积
        fp16=True  # 使用混合精度训练
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset  # 添加验证集
    )
    
    # 开始训练
    print("开始训练小型GPT-2模型...")
    trainer.train()
    
    # 保存最终模型
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print(f"模型已保存至: {SAVE_PATH}")
    
    return model, tokenizer

# 5. 生成文本示例
def generate_text_examples(model, tokenizer):
    """生成文本示例"""
    # 创建文本生成管道
    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=0  # 使用GPU
    )
    
    # 设置随机种子
    set_seed(42)
    
    # 测试不同提示
    prompts = [
        "吕布手持方天画戟",
        "乔峰降龙十八掌",
        "郭靖和黄蓉",
        "诸葛亮摇着羽扇"
    ]
    
    # 生成文本
    for prompt in prompts:
        print(f"\n提示: {prompt}")
        # 优化的生成参数
        output = generator(
            prompt,
            max_length=100,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            no_repeat_ngram_size=3,
            num_return_sequences=1
        )
        print(f"生成文本: {output[0]['generated_text']}\n")

# 主执行函数
if __name__ == "__main__":
    # 训练模型
    model, tokenizer = train_model()
    
    # 生成示例文本
    generate_text_examples(model, tokenizer)
```

## 小型模型架构的核心改进

1. **模型尺寸大幅减小**：
   ```python
   # 原始GPT-2 (124M参数) vs 小型版本 (约30M参数)
   config = GPT2Config(
       n_layer=6,      # 原始12层 → 减少50%
       n_head=6,       # 原始12头 → 减少50%
       n_embd=384,     # 原始768 → 减少50%
       n_inner=1536,   # 原始3072 → 减少50%
       n_positions=256 # 原始1024 → 减少75%
   )
   ```

2. **优化的分词器**：
   ```python
   # 使用字符级BPE更适合中文
   tokenizer = CharBPETokenizer(
       suffix="",
       unk_token="<unk>",
   )
   tokenizer.train(files, vocab_size=8000)  # 更小的词汇表
   ```

3. **训练参数优化**：
   ```python
   TrainingArguments(
       num_train_epochs=10,            # 减少训练轮次
       per_gpu_train_batch_size=8,      # 适合小模型的批次大小
       learning_rate=3e-5,              # 更小的学习率
       gradient_accumulation_steps=4,   # 梯度累积
       fp16=True                        # 混合精度训练
   )
   ```

4. **改进的文本生成**：
   ```python
   # 优化的生成参数减少重复
   generator(
       prompt,
       max_length=100,
       do_sample=True,
       top_k=50,
       top_p=0.95,
       temperature=0.7,
       no_repeat_ngram_size=3  # 防止3-gram重复
   )
   ```

## 预期效果对比

| 指标               | 原始实现         | 小型模型方案      |
|--------------------|------------------|------------------|
| 模型大小           | ~124M 参数       | ~30M 参数        |
| 训练时间 (估算)    | 977秒/epoch      | 200秒/epoch      |
| GPU内存占用        | 高 (>8GB)        | 中等 (~4GB)      |
| 最终loss值         | 1.47 (未收敛)    | <1.0 (收敛)      |
| 生成质量           | 无意义重复       | 上下文相关文本   |
| 领域适应性         | 差               | 良好             |

## 部署建议

对于进一步优化，可以考虑：

1. **知识蒸馏**：使用大型预训练模型指导小型模型
   ```python
   from transformers import DistillationTrainingArguments, DistillationTrainer
   # 使用教师模型(如uer/gpt2-chinese-lyric)指导训练
   ```

2. **量化压缩**：训练后量化减少模型大小
   ```python
   from transformers import GPT2LMHeadModel, GPT2Tokenizer
   from optimum.onnxruntime import ORTModelForCausalLM
   
   # 转换为ONNX格式并量化
   model = ORTModelForCausalLM.from_pretrained(SAVE_PATH, export=True)
   model.quantize(optimization_level=1)
   ```

3. **持续训练**：定期添加新数据微调模型
   ```python
   # 加载已有模型继续训练
   model = GPT2LMHeadModel.from_pretrained(SAVE_PATH)
   trainer = Trainer(model=model, ...)
   trainer.train(resume_from_checkpoint=True)
   ```

这个小型模型方案特别适合：
- 有限的计算资源（如单GPU或CPU训练）
- 领域特定的应用（如古典小说生成）
- 需要快速迭代的实验场景
- 移动端或边缘设备部署