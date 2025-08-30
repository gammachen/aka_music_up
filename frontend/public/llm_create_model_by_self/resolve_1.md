### 核心问题分析

1. **数据量严重不足（根本问题）**
   - 仅使用两部小说（约200-300万字）训练GPT-2这种规模的模型
   - 对比参考：
     - 原始GPT-2训练数据：40GB文本（约100亿词）
     - 中文GPT模型最小需求：通常需要10GB+中文语料
   - 后果：模型无法学习基本语言规律，只能记住局部模式

2. **分词器设计缺陷（关键问题）**
   ```python
   tokenizer = Tokenizer(BPE(unk_token="<unk>"))
   tokenizer.pre_tokenizer = ByteLevel()  # 字节级处理对中文不友好
   ```
   - **字节级BPE对中文效率低下**：
     - 中文需要词/字级别处理，字节级拆分导致无效子词
     - 例如："吕布" 被拆分为6个字节(0xE5, 0x90, 0x95, 0xE5, 0xB8, 0x83)
   - **词汇表过大**：
     - `vocab_size=50000` 远超实际需求
     - 两部小说唯一汉字不足5000个
   - **缺少中文优化**：
     - 未添加中文专用token（如[CLS], [SEP]）
     - 未处理中文标点分割

3. **训练过程问题**
   - **Loss值异常**：
     ```log
     {'loss': 1.4775, ... 'epoch': 19.23}  # 最终loss仍高达1.47
     ```
     - 正常中文模型loss应降至1.0以下
     - 表明模型未收敛
   - **梯度爆炸**：
     ```log
     'grad_norm': 10.862028121948242  # 持续>10表明梯度爆炸
     ```
   - **无验证集**：
     - 无法检测过拟合/欠拟合
     - 无法保存最佳模型

### 次要但重要的问题

4. **模型架构不匹配**
   ```python
   model = GPT2LMHeadModel(config)  # 从头训练
   ```
   - GPT-2默认配置(12层)对小数据过大
   - 未利用中文预训练权重

5. **训练参数不当**
   ```python
   TrainingArguments(
       num_train_epochs=20,  # 对过小数据集过多
       per_gpu_train_batch_size=16,  # 未考虑显存限制
       learning_rate=5e-5  # 默认LR对小数据过大
   )
   ```

6. **数据处理缺陷**
   ```python
   dataset = dataset + dataset2  # 简单合并
   ```
   - 未打乱数据顺序
   - 未处理文本连贯性（行间断裂）

### 生成结果分析
```text
"吕布字回 诸葛亮 丁奉雪破 孔明 姜维新野新野..."
```
- **重复模式**："新野"重复4次，"空空"重复20+次
- **角色混淆**：吕布关联诸葛亮/姜维
- **无意义组合**："丁奉雪破"
- **早期停止**：大量填充符("取取取")

### 解决方案（按优先级）

1. **数据层面**
   ```python
   # 增加数据源
   files = [
       "text/sanguo.txt", 
       "text/tianlong.txt",
       "text/hongloumeng.txt",        # 新增红楼梦
       "text/jinyong_all.txt",         # 金庸全集
       "text/wiki_zh.txt"              # 中文维基
   ]
   ```

2. **分词器优化**
   ```python
   # 改用字级分词
   from tokenizers import CharBPETokenizer
   
   tokenizer = CharBPETokenizer()  # 字符级BPE
   tokenizer.train(files, vocab_size=8000)  # 更小词汇量
   ```

3. **迁移学习**
   ```python
   # 使用预训练中文模型
   model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-lyric")
   tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-lyric")
   ```

4. **训练优化**
   ```python
   TrainingArguments(
       num_train_epochs=5,  # 减少轮次
       per_gpu_train_batch_size=4,  # 减小批次
       learning_rate=3e-5,  # 更低学习率
       gradient_accumulation_steps=8,  # 梯度累积
       evaluation_strategy="steps",  # 添加验证
       eval_steps=500
   )
   ```

5. **生成参数调整**
   ```python
   generator("吕布", 
            max_length=100,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            no_repeat_ngram_size=2)  # 防止重复
   ```

### 优化后训练建议

```python
# 完整改进方案
from transformers import BertTokenizer, GPT2LMHeadModel

# 1. 使用预训练中文模型
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-lyric")
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-lyric")

# 2. 扩展数据集
files = [f"text/{name}.txt" for name in 
         ["sanguo", "tianlong", "honglou", "shediao", "wiki_zh"]]

# 3. 配置动态填充
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=64  # 显存优化
)

# 4. 添加验证集
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=5,
    per_gpu_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=3e-5,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_steps=2000,
    logging_steps=100,
    prediction_loss_only=True
)

# 5. 训练时监控
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,  # 必须添加
    data_collator=data_collator
)
```

### 预期改进效果
| 指标         | 优化前         | 优化后预期     |
|--------------|---------------|--------------|
| Loss值       | 1.47 (不收敛)  | <1.0 (收敛)   |
| 生成连贯性   | 无意义重复     | 基本通顺      |
| 上下文关联   | 角色混淆       | 基本准确      |
| 训练效率     | 977秒/epoch   | 300秒/epoch  |

优先考虑实施数据扩充和预训练模型迁移，这两项改进将带来最显著的性能提升。对于小规模专用模型，可考虑将GPT-2替换为更小的架构（如GPT-2 Small）。