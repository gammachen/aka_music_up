# 博物馆领域微调指令生成工具

这个工具集用于基于博物馆领域结构化知识生成高质量的微调指令，可用于训练大型语言模型，使其在博物馆领域具有更专业的知识和回答能力。

## 功能特点

- 基于结构化博物馆知识自动生成多样化的微调指令
- 支持不同类型的指令生成（问答型、解释型、比较型、推理型、创意型）
- 可控制指令难度级别（简单、中等、困难）
- 支持针对特定数据部分或主题生成指令
- 支持多种微调数据格式（OpenAI、Alpaca、LLaMA、Qwen等）
- 提供完整的数据处理流程，从知识库到微调数据集

## 文件说明

- `museum_knowledge_more.json`: 博物馆领域结构化知识数据
- `museum_finetune_generator.py`: 核心微调指令生成模块
- `generate_museum_finetune.py`: 命令行工具，用于生成微调指令
- `prepare_finetune_dataset.py`: 将生成的指令转换为不同格式的微调数据集

## 安装依赖

```bash
pip install openai tqdm python-dotenv
```

## 使用方法

### 1. 设置环境变量

```bash
export OPENAI_API_KEY="your-api-key"
# 可选：如果使用非默认API端点
export OPENAI_BASE_URL="https://your-api-endpoint"
```

### 2. 查看数据结构

```bash
python generate_museum_finetune.py --list-sections --data museum_knowledge_more.json
```

### 3. 生成微调指令

#### 基本用法

```bash
python generate_museum_finetune.py --output museum_finetune_instructions.json
```

#### 指定数据部分和难度

```bash
python generate_museum_finetune.py \
  --section museum_basic_info \
  --difficulty 中等 \
  --count 3 \
  --output museum_basic_info_instructions.json
```

#### 指定主题

```bash
python generate_museum_finetune.py \
  --section museum_basic_info \
  --topic opening_hours \
  --output opening_hours_instructions.json
```

### 4. 准备微调数据集

```bash
python prepare_finetune_dataset.py \
  --input museum_finetune_instructions.json \
  --format openai \
  --split 0.1
```

支持的格式：
- `openai`: OpenAI Chat格式
- `alpaca`: Alpaca指令格式
- `llama`: LLaMA指令格式
- `qwen`: 通义千问格式

## 工作流程

1. **知识准备**：使用结构化的博物馆领域知识（JSON格式）
2. **指令生成**：基于知识生成多样化的指令-回复对
3. **数据转换**：将指令转换为适合不同模型的微调数据格式
4. **模型微调**：使用生成的数据集进行模型微调

## 自定义与扩展

### 添加新的数据格式

在`prepare_finetune_dataset.py`中添加新的转换函数，并更新`format_converters`字典。

### 调整生成策略

修改`museum_finetune_generator.py`中的`build_finetune_prompt`函数，可以调整提示词结构和生成要求。

## 示例输出

生成的微调指令示例：

```json
{
  "finetune_instructions": [
    {
      "instruction": "请介绍博物馆的开放时间。",
      "response": "博物馆的常规开放时间是：工作日09:00-17:00，周末09:00-18:00。每周一和春节初一闭馆。特殊节假日会有调整，如国庆节期间为08:30-19:00，中秋节为09:00-17:30。最后入场时间为闭馆前30分钟。",
      "type": "问答型",
      "difficulty": "简单",
      "knowledge_source": ["museum_basic_info_opening_hours"]
    },
    {
      "instruction": "比较博物馆普通参观和会员参观的区别与优势。",
      "response": "普通参观和会员参观的主要区别在于：普通参观需要每次购票，成人票价50元，学生票25元；而会员制度年费200元，提供无限次参观、专属导览和商店折扣等特权。对于经常参观博物馆的访客，会员制度更为经济实惠，且能享受更丰富的博物馆体验和专属服务。",
      "type": "比较型",
      "difficulty": "中等",
      "knowledge_source": ["museum_basic_info_ticket_policy"]
    }
  ]
}
```

## 注意事项

- 如果未设置`OPENAI_API_KEY`环境变量，工具将返回模拟数据
- 生成高质量指令需要使用功能强大的模型，推荐使用GPT-4或同等能力的模型
- 生成的指令质量取决于输入的知识数据质量和完整性

## 许可证

MIT