## 更细化的场景与代码

```shell
bash /Users/shhaofu/Code/cursor-projects/p-llm-word2vec/run_docsify.sh
```

以下是三个主流中文预训练Word2Vec模型的使用示例，包含下载链接和基础代码示例：

---

### 1. **腾讯AI Lab中文词向量（Tencent AI Lab Embedding）**
**特点**：覆盖800万中文词汇，维度200/100，适合通用场景  
**下载地址**：  
- 官方下载页：[Tencent AI Lab Embedding](https://ai.tencent.com/ailab/nlp/zh/embedding.html)  
- 直接下载链接（200维）：[Tencent_AILab_ChineseEmbedding.tar.gz](https://ai.tencent.com/ailab/nlp/zh/data/Tencent_AILab_ChineseEmbedding.tar.gz)  

**使用示例**：
```python
from gensim.models import KeyedVectors

# 加载模型（解压后约16GB，需足够内存）
model = KeyedVectors.load_word2vec_format('Tencent_AILab_ChineseEmbedding.txt', binary=False)

# 示例操作
print("词向量维度:", model.vector_size)  # 输出: 200
print("相似词:", model.most_similar("人工智能", topn=3))
# 输出示例: [('AI', 0.78), ('机器学习', 0.75), ('深度学习', 0.72)]

print("类比推理:", model.most_similar(positive=['女人', '国王'], negative=['男人']))
# 可能输出: [('女王', 0.65), ...]
```

---

### 2. **中文维基百科预训练词向量**
**特点**：基于维基百科语料，适合学术或通用文本  
**下载地址**：  
- GitHub项目：[Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)  
- 直接下载（300维，Word2Vec格式）：[wiki_zh.word2vec.gz](https://pan.baidu.com/s/1kK0eevZruY4kDLk44p5Hug?pwd=6g3a)  

**使用示例**：
```python
import gensim.downloader as api

# 下载并加载模型（首次运行会自动下载）
model = api.load("text8")  # 此处需替换为实际路径
# 实际使用时需先解压.gz文件，然后:
model = KeyedVectors.load_word2vec_format('wiki_zh.word2vec', binary=False)

# 示例操作
print("'中国'的向量:", model['中国'][:5])  # 打印前5维
print("相近词:", model.most_similar("北京", topn=2))
# 输出示例: [('上海', 0.82), ('天津', 0.79)]
```

---

### 3. **搜狗新闻预训练词向量**
**特点**：基于搜狗新闻语料，适合新闻相关任务  
**下载地址**：  
- 同项目[Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)中的`sogounews`开头的文件  
- 百度网盘链接：[sogou_news.word2vec.gz](https://pan.baidu.com/s/1kK0eevZruY4kDLk44p5Hug?pwd=6g3a)  

**使用示例**：
```python
from gensim.models import KeyedVectors

# 加载模型（需先解压）
model = KeyedVectors.load_word2vec_format('sogou_news.word2vec', binary=False)

# 新闻相关示例
print("'疫情'的相似词:", model.most_similar("疫情", topn=3))
# 输出示例: [('防控', 0.88), ('肺炎', 0.85), ('新冠病毒', 0.83)]
```

---

### 通用注意事项：
1. **内存要求**：腾讯词向量需16GB+内存，其他模型约2-8GB
2. **二进制格式**：若文件为`.bin`，使用`binary=True`
3. **OOV处理**：预训练模型无法处理新词，可考虑用FastText替代
4. **简化加载**：对于大文件，可用以下代码限制词汇量：
   ```python
   model = KeyedVectors.load_word2vec_format('path.txt', binary=False, limit=500000)
   ```
