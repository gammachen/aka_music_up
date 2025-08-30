# 自定义大模型构建技术方案文档

从零开始构建一个自定义的大语言模型。该模型专门针对天龙八部文本进行训练，能够生成与天龙八部风格相似的文本内容。

## 1. 概述

本方案使用 Hugging Face Transformers 库构建一个基于 GPT-2 架构的自定义语言模型。整个流程包括：
- 自定义分词器训练
- 模型架构配置
- 训练数据准备
- 模型训练
- 模型测试与推理

## 2. 技术栈

- **核心框架**: Hugging Face Transformers
- **分词器库**: Hugging Face Tokenizers
- **深度学习框架**: PyTorch
- **模型架构**: GPT-2
- **训练数据**: 《天龙八部》中文文本

## 3. 实现细节

### 3.1 分词器构建

使用 BPE (Byte Pair Encoding) 算法训练自定义分词器：

```python
# 初始化分词器，使用 BPE 算法
tokenizer = Tokenizer(BPE(unk_token="<unk>"))

# 设置文本规范化器
tokenizer.normalizer = Sequence([NFKC()])

# 设置预分词器为字节级
tokenizer.pre_tokenizer = ByteLevel()

# 设置解码器
tokenizer.decoder = ByteLevelDecoder()
```

关键配置参数：
- 词汇表大小: 50,000
- 特殊标记: ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
- 训练文件: text/sanguoyanyi.txt

### 3.2 模型架构配置

使用 GPT-2 配置创建模型：

```python
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

model = GPT2LMHeadModel(config)
```

### 3.3 训练数据处理

使用 LineByLineTextDataset 处理训练数据：

```python
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./text/sanguoyanyi.txt",
    block_size=32,
)
```

数据整理器配置：
```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False,  # 不使用掩码语言模型
    mlm_probability=0.15
)
```

### 3.4 训练参数配置

```python
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_gpu_train_batch_size=16,
    save_steps=2000,
    save_total_limit=2,
)
```

### 3.5 模型训练

```python
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
model.save_pretrained(SAVE_PATH)
```

### 3.6 模型测试

```python
generator = pipeline('text-generation', model=SAVE_PATH)
set_seed(13)
txt = generator("吕布", max_length=10)
print(txt)
```

## 4. 优化建议

### 4.1 数据增强
- 增加更多古典文学作品进行混合训练
- 对文本进行清洗和预处理，去除噪声

### 4.2 模型改进
- 调整模型超参数，如层数、隐藏单元数等
- 尝试更大的预训练模型架构

### 4.3 训练优化
- 添加学习率调度器
- 使用早停机制防止过拟合
- 增加验证集监控训练过程

### 4.4 性能提升
- 使用混合精度训练加速训练过程
- 利用多GPU并行训练

## 5. 部署建议

1. 将训练好的模型导出为标准格式
2. 构建 API 服务供外部调用
3. 添加模型版本管理和监控机制
4. 考虑模型压缩和加速技术以提高推理速度

## Code

```python
# 导入os模块，用于操作系统相关功能，主要用于设置环境变量
import os
# 禁用 wandb（Weights & Biases）实验跟踪工具，避免训练过程中的网络连接和初始化超时问题
# 通过设置环境变量WANDB_DISABLED为"true"来完全禁用wandb功能
os.environ["WANDB_DISABLED"] = "true"

# 从Hugging Face transformers库导入核心组件
# pipeline: 用于快速创建各种NLP任务的推理管道，支持文本生成、分类等任务
# set_seed: 用于设置PyTorch、NumPy等库的随机种子，确保训练和推理结果的可重现性
from transformers import pipeline, set_seed
# GPT2Config: GPT-2模型的配置类，用于定义模型架构参数（层数、隐藏层大小、注意力头数等）
# GPT2LMHeadModel: 带有语言建模头的GPT-2模型，用于文本生成任务
# GPT2Tokenizer: GPT-2专用的分词器，用于将文本转换为模型可理解的token序列
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
# LineByLineTextDataset: 按行读取文本文件的数据集类，每行作为一个训练样本
from transformers import LineByLineTextDataset
# DataCollatorForLanguageModeling: 语言建模任务的数据整理器，负责批次数据的padding和掩码处理
from transformers import DataCollatorForLanguageModeling
# Trainer: Hugging Face提供的高级训练器，封装了完整的训练循环
# TrainingArguments: 训练参数配置类，包含学习率、批次大小、训练轮数等超参数
from transformers import Trainer, TrainingArguments
# 从tokenizers库导入自定义分词器相关组件，用于构建专门针对中文古典文学的分词器
# Tokenizer: 分词器的基础类，提供分词器的核心功能框架
from tokenizers import Tokenizer
# BPE: 字节对编码（Byte Pair Encoding）算法实现，通过统计字符对频率来学习子词分割
from tokenizers.models import BPE
# BpeTrainer: BPE算法的训练器，负责在给定语料上训练BPE模型，学习最优的词汇表
from tokenizers.trainers import BpeTrainer
# NFKC: Unicode标准化形式KC（兼容性分解后重组），用于统一不同编码的相同字符
# Sequence: 用于将多个文本处理步骤组合成序列，按顺序执行
from tokenizers.normalizers import NFKC, Sequence
# ByteLevel: 字节级预分词器，将文本转换为字节序列，能处理任何Unicode字符
from tokenizers.pre_tokenizers import ByteLevel
# ByteLevelDecoder: 字节级解码器，将token序列解码回原始文本，与ByteLevel预分词器配对使用
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
# GPT2TokenizerFast: 基于Rust实现的快速GPT-2分词器，性能优于Python版本
from transformers import GPT2TokenizerFast
# 导入wandb实验跟踪库（虽然通过环境变量被禁用，但保留导入以防后续需要）
import wandb

# 定义模型和分词器的保存路径常量
# 训练完成后，模型权重、配置文件、分词器等所有相关文件都将保存在此目录下
SAVE_PATH = "./sanguo"

# 创建基于BPE算法的自定义分词器实例
# BPE(unk_token="<unk>"): 初始化字节对编码模型，指定未知词（不在词汇表中的词）的标记为<unk>
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
# 配置文本标准化器：使用NFKC（兼容性分解后重组）统一字符编码
# 这对中文文本特别重要，因为同一个字符可能有多种Unicode编码形式
tokenizer.normalizer = Sequence([NFKC()])  # 统一字符格式，确保编码一致性
# 配置字节级预分词器：将文本转换为字节序列进行处理
# 这种方式能够处理任何Unicode字符，包括中文、标点符号和特殊字符
tokenizer.pre_tokenizer = ByteLevel()      # 字节级预分词，支持全字符集
# 配置字节级解码器：将分词后的token序列还原为可读文本
# 与ByteLevel预分词器配对使用，确保编码解码的一致性
tokenizer.decoder = ByteLevelDecoder()     # 字节级解码器，还原文本

# 定义模型训练和推理过程中需要的特殊标记列表
# <s>: 序列开始标记（Beginning of Sequence），标识文本的开始
# <pad>: 填充标记（Padding），用于批处理时将不同长度的序列对齐到相同长度
# </s>: 序列结束标记（End of Sequence），标识文本的结束
# <unk>: 未知词标记（Unknown），表示词汇表中不存在的词汇
# <mask>: 掩码标记（Mask），用于掩码语言建模任务（虽然本项目使用因果语言建模）
special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

# 创建BPE（字节对编码）训练器实例，用于在给定语料上学习最优的子词分割策略
trainer = BpeTrainer(
    # vocab_size=50000: 设置最终词汇表的大小为50000个token
    # 这个大小需要平衡模型性能和计算效率，太小会导致过多未知词，太大会增加计算开销
    vocab_size=50000, 
    # show_progress=True: 在训练过程中显示进度条，便于监控训练状态
    show_progress=True,
    # initial_alphabet: 设置初始字符集为字节级字母表（0-255的所有字节值）
    # 这确保分词器能够处理任何可能的字符，包括中文、英文、标点符号等
    initial_alphabet=ByteLevel.alphabet(),  # 字节级字符集，支持全Unicode字符
    # special_tokens: 将之前定义的特殊标记添加到词汇表中，确保它们不会被进一步分割
    special_tokens=special_tokens
)

# 定义用于训练分词器的中文古典文学语料文件列表
# "text/sanguoyanyi.txt": 三国演义全文，提供古代汉语和历史小说的语言特征
# "text/tian_long_ba_bu_all.txt": 天龙八部全文，提供武侠小说的语言特征和现代中文表达
files = ["text/sanguoyanyi.txt", "text/tian_long_ba_bu_all.txt"]
# 在指定的文本文件上训练BPE分词器
# 训练过程会统计字符对的出现频率，学习最优的子词分割策略，构建适合中文文本的词汇表
tokenizer.train(files, trainer)

# 将自定义训练的分词器包装为GPT2TokenizerFast格式
# GPT2TokenizerFast是基于Rust实现的高性能分词器，与Transformers库完全兼容
# tokenizer_object参数指定使用之前训练好的自定义分词器作为底层实现
newtokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
# 将包装后的分词器保存到指定目录
# 保存内容包括：词汇表文件、分词器配置、特殊标记映射等
# 后续可以直接从此路径加载分词器，无需重新训练
newtokenizer.save_pretrained(SAVE_PATH)  # 保存到./sanguo目录

# 从本地保存路径重新加载GPT-2分词器
# 使用GPT2Tokenizer.from_pretrained()方法加载之前保存的自定义分词器
tokenizer = GPT2Tokenizer.from_pretrained(SAVE_PATH)
# 显式添加特殊标记到分词器中，确保模型能正确识别和处理这些特殊情况
# eos_token="</s>": 序列结束标记，用于标识文本生成的结束点
# bos_token="<s>": 序列开始标记，用于标识文本生成的起始点
# unk_token="<unk>": 未知词标记，用于处理词汇表外的词汇
# pad_token="<pad>": 填充标记，用于批处理时对齐不同长度的序列
# mask_token="<mask>": 掩码标记，用于掩码语言建模（本项目中备用）
tokenizer.add_special_tokens({"eos_token": "</s>", "bos_token": "<s>",
                             "unk_token": "<unk>", "pad_token": "<pad>", "mask_token": "<mask>"})

# 创建GPT-2模型的配置对象，定义模型的架构参数
config = GPT2Config(
    # vocab_size: 设置模型词汇表大小，必须与分词器的词汇表大小保持一致
    # 这决定了模型输出层的维度和嵌入层的大小
    vocab_size=tokenizer.vocab_size,
    # bos_token_id: 设置序列开始标记的数字ID，用于文本生成时的起始标记
    bos_token_id=tokenizer.bos_token_id,
    # eos_token_id: 设置序列结束标记的数字ID，用于判断文本生成的结束条件
    eos_token_id=tokenizer.eos_token_id
    # 其他参数使用GPT2Config的默认值：
    # - n_positions=1024 (最大序列长度)
    # - n_embd=768 (嵌入维度)
    # - n_layer=12 (Transformer层数)
    # - n_head=12 (注意力头数)
)

# 根据配置创建GPT-2语言模型实例
# GPT2LMHeadModel包含完整的GPT-2架构：嵌入层、多层Transformer块、语言建模头
# 模型权重随机初始化，需要通过训练来学习语言模式
model = GPT2LMHeadModel(config)

# 创建第一个训练数据集：三国演义文本数据集
# LineByLineTextDataset会按行读取文本文件，每行作为一个独立的训练样本
dataset = LineByLineTextDataset(
    # tokenizer: 指定使用之前配置的自定义分词器进行文本tokenization
    tokenizer=tokenizer,
    # file_path: 指定三国演义文本文件的路径，包含完整的古典小说内容
    file_path="./text/sanguoyanyi.txt",
    # block_size=128: 设置每个训练样本的最大token长度为128
    # 相比原来的32，128能更好地捕捉中文古典小说的语言模式和上下文信息
    # 这个长度适合中文的句子结构，能包含完整的语义单元
    block_size=128,
)

# 创建第二个训练数据集：天龙八部文本数据集
# 添加第二个数据集可以增加训练语料的多样性，提高模型的泛化能力
dataset2 = LineByLineTextDataset(
    # tokenizer: 使用相同的分词器确保两个数据集的token化方式一致
    tokenizer=tokenizer,
    # file_path: 指定天龙八部文本文件路径，提供武侠小说的语言风格
    file_path="./text/tian_long_ba_bu_all.txt",
    # block_size=128: 与第一个数据集保持相同的序列长度，确保训练的一致性
    # 128个token足以包含天龙八部中的完整对话和描述片段
    block_size=128,
)

# 合并两个数据集，创建包含多样化中文文学内容的综合训练数据集
# 通过+操作符将两个LineByLineTextDataset对象合并，增加训练数据的丰富性
# 合并后的数据集包含古典历史小说和现代武侠小说的语言特征
dataset = dataset + dataset2

# 创建语言建模任务的数据整理器（Data Collator）
# 负责将单个样本组织成训练批次，处理padding、attention mask等
data_collator = DataCollatorForLanguageModeling(
    # tokenizer: 指定用于处理文本的分词器，用于padding和特殊标记处理
    tokenizer=tokenizer, 
    # mlm=False: 设置为False表示使用因果语言建模（Causal Language Modeling）
    # 因果语言建模是GPT系列模型的标准训练方式，模型只能看到当前位置之前的token
    mlm=False, 
    # mlm_probability=0.15: 掩码语言建模的掩码概率（在CLM模式下不生效）
    # 保留此参数是为了代码的完整性，实际训练中不会使用
    mlm_probability=0.15
)

# 配置模型训练的超参数和训练策略
# TrainingArguments包含了训练过程中的所有重要参数设置
training_args = TrainingArguments(
    # output_dir: 指定训练输出目录，用于保存模型检查点、日志和其他训练产物
    output_dir="./output",
    # overwrite_output_dir=True: 允许覆盖已存在的输出目录，避免路径冲突错误
    overwrite_output_dir=True,
    # num_train_epochs=20: 设置完整的训练轮数为20个epoch
    # 每个epoch表示模型完整遍历一次所有训练数据
    num_train_epochs=20,
    # per_gpu_train_batch_size=16: 设置每个GPU的批次大小为16个样本
    # 实际的全局批次大小 = per_gpu_train_batch_size × GPU数量
    per_gpu_train_batch_size=16,
    # save_steps=2000: 每训练2000步保存一次模型检查点
    # 这样可以在训练中断时从最近的检查点恢复训练
    save_steps=2000,
    # save_total_limit=2: 最多保留2个检查点文件，自动删除较旧的检查点
    # 这有助于节省磁盘空间，避免检查点文件过多
    save_total_limit=2,
    # 其他参数使用默认值：learning_rate=5e-5, weight_decay=0.0, warmup_steps=0等
)

# 创建Hugging Face Trainer实例，整合所有训练组件
# Trainer封装了完整的训练循环，包括前向传播、反向传播、优化器更新等
trainer = Trainer(
    # model: 指定要训练的GPT-2模型实例，包含随机初始化的权重
    model=model,
    # args: 传入训练参数配置，控制训练过程的各个方面
    args=training_args,
    # data_collator: 指定数据整理器，负责批次数据的组织和预处理
    data_collator=data_collator,
    # train_dataset: 指定训练数据集，包含合并后的三国演义和天龙八部文本
    train_dataset=dataset,
    # 注意：这里没有设置eval_dataset，表示只进行训练而不进行验证
)

# 启动模型训练过程
# trainer.train()会执行完整的训练循环，包括：
# 1. 数据加载和批次处理
# 2. 前向传播计算损失
# 3. 反向传播计算梯度
# 4. 优化器更新模型权重
# 5. 定期保存检查点
trainer.train()
# 训练完成后，将最终的模型权重和配置保存到指定目录
# 保存内容包括：模型权重文件、配置文件、生成配置等
# 这样可以在后续直接加载训练好的模型进行推理
model.save_pretrained(SAVE_PATH)

# 测试阶段：验证训练好的模型的文本生成能力
# 创建文本生成管道（pipeline），自动加载保存的模型和分词器
# pipeline会处理文本预处理、模型推理、后处理等完整流程
generator = pipeline('text-generation', model=SAVE_PATH)
# 设置随机种子为13，确保每次运行生成相同的结果，便于调试和比较
set_seed(13)
# 使用"吕布"作为起始提示词（prompt）进行文本生成
# max_length=30: 设置生成文本的最大长度为30个token（包括输入的提示词）
# 选择"吕布"是因为这是三国演义中的重要人物，能测试模型对古典文学的理解
txt = generator("吕布", max_length=30)
# 打印生成的文本结果，查看模型是否学会了中文古典文学的语言风格

# test model
generator = pipeline('text-generation', model=SAVE_PATH)
set_seed(13)
txt = generator("桃园三结义都有谁", max_length=100)
print(txt)

txt = generator("曹操字", max_length=100)
print(txt)

txt = generator("张飞的个人信息", max_length=100)
print(txt)

txt = generator("关羽温酒斩的是谁", max_length=100)
print(txt)

```

## 训练结果

```python

{'loss': 5.1553, 'grad_norm': 4.956688404083252, 'learning_rate': 4.520192307692308e-05, 'epoch': 1.92}                                                
{'loss': 4.3689, 'grad_norm': 5.231606960296631, 'learning_rate': 4.039423076923077e-05, 'epoch': 3.85}                                                
{'loss': 3.8915, 'grad_norm': 7.602330684661865, 'learning_rate': 3.558653846153847e-05, 'epoch': 5.77}                                                
{'loss': 3.4587, 'grad_norm': 8.42416000366211, 'learning_rate': 3.077884615384615e-05, 'epoch': 7.69}
{'loss': 3.0382, 'grad_norm': 11.204177856445312, 'learning_rate': 2.5971153846153847e-05, 'epoch': 9.62}                                              
{'loss': 2.616, 'grad_norm': 11.977376937866211, 'learning_rate': 2.116346153846154e-05, 'epoch': 11.54}                                               
{'loss': 2.2337, 'grad_norm': 13.870461463928223, 'learning_rate': 1.6355769230769233e-05, 'epoch': 13.46}                                             
{'loss': 1.9052, 'grad_norm': 11.351497650146484, 'learning_rate': 1.1548076923076924e-05, 'epoch': 15.38} 
{'loss': 1.6619, 'grad_norm': 11.588981628417969, 'learning_rate': 6.740384615384615e-06, 'epoch': 17.31}                                              
{'loss': 1.4775, 'grad_norm': 10.862028121948242, 'learning_rate': 1.932692307692308e-06, 'epoch': 19.23}                                              
{'train_runtime': 977.8944, 'train_samples_per_second': 84.938, 'train_steps_per_second': 5.318, 'train_loss': 2.919726286668044, 'epoch': 20.0} 
```

## 6. 总结

该技术方案提供了一套完整的自定义大模型构建流程，从数据预处理到模型训练和部署。通过针对特定领域的文本进行训练，可以获得在该领域表现优异的语言模型。后续可根据实际需求对模型进行进一步优化和调整。

### 效果

非常差

```plaintext
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
[{'generated_text': '桃园三结义都有谁神神回 姜维曹操曹操，诛诛诛诛回，神神神神。至今回，遂争锋回回，因生心心显神神神成心显显显已生心心心主难听取取取取取取取取取取取取取取取取取取，皆吴、张、赵云、马岱听、吕翔、赵云、张、赵云、吕翔'}]
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
[{'generated_text': '曹操字回 诸葛亮 丁奉雪破 孔明 姜维新野新野新野新野难 姜维 姜维。至今主难难处，谁空空空空空空空空空空空空空空空空空空空空空空空空空空空空空空空空空空空空空空空空空。神空空空空空空空空空空空空空空空空空空空空空空。神。神。神神神神'}]
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
[{'generated_text': '张飞的个人信息、郭汜回 姜维曹操曹操。云长等三 孔明、吕翔等先神神回，张飞断后。云、吕翔劝二人用计、吕翔等闻��遂遂遂遂遂遂遂遂遂遂欲同。。。孔明曰：“刘备等闻闻袁绍破了。”孔明曰：“我四十刘备遣“我显军师刘备遣刘备刘备刘备等闻丞相遂遂遂遂遂遂遂皆刘备�'}]
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
[{'generated_text': '关羽温酒斩的是谁败走 汉 姜维，又见荆州新野新野。曹操回，请 孔明，遂激臣。两阵对圆，左右使使屯兵屯兵方回，遂遂带剑入宫，遂入朝。操曰，遂皆刘备拜：“我：“我东吴已刘备刘备拜后主又上表，遂皆刘备拜后主问曰，遂入朝：“我显'}]
```

