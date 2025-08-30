你指出的问题非常关键：**IK 分词器的分词结果无法直接拆分成 `city` 和 `shop_name` 的字段值**。这是因为 IK 的分词粒度是基于语义和规则的，无法通过简单的分词列表推断出字段边界（例如 `["杭州", "市", "纯味", "斑鱼府"]` 无法自动组合为 `city="杭州市"` 和 `shop_name="纯味斑鱼府"`）。

---

### **问题的根本原因**
- **IK 分词器的输出是语义切分**，但**无法区分字段边界**。  
  例如，`"杭州市纯味斑鱼府"` 可能被切分为 `["杭州", "市", "纯味", "斑鱼府"]`，但你无法知道哪个部分属于 `city`，哪个属于 `shop_name`。

- **手动拆分逻辑不可靠**：  
  如果直接拼接 `["杭州", "市"]` 为 `city="杭州市"`，但实际用户可能输入的是 `杭州市西湖区纯味斑鱼府`，此时 `["杭州", "市", "西湖", "区", ...]` 也无法正确拆分。

---

### **解决方案**
#### **1. 放弃手动拆分，改为 `multi_match` 查询**
直接使用 Elasticsearch 的 `multi_match` 查询，让 IK 分词器自动处理分词，并在多个字段中匹配：
```python
def search_shops(query):
    es_query = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["city", "shop_name"],
                "analyzer": "ik_max_word"
            }
        }
    }

    results = es.search(index="shops", body=es_query)
    return format_results(results)
```
- **优点**：无需手动拆分字段，完全依赖 Elasticsearch 的分词和匹配能力。
- **缺点**：可能匹配到无关字段（如 `shop_name` 中包含 `杭州市` 的店铺），但可以通过后续排序优化。

---

#### **2. 使用 `bool` 查询 + `match_phrase` 精确匹配**
如果希望严格匹配 `city` 和 `shop_name` 的字段，但又不依赖手动拆分，可以尝试以下方案：
```python
def search_shops(query):
    es_query = {
        "query": {
            "bool": {
                "should": [
                    # 1. 在 city.keyword 中精确匹配 "杭州市"
                    {"term": {"city.keyword": "杭州市"}},
                    # 2. 在 shop_name 中匹配 "纯味斑鱼府"（假设分词正确）
                    {"match_phrase": {"shop_name": "纯味斑鱼府"}}
                ],
                "minimum_should_match": 1
            }
        }
    }

    results = es.search(index="shops", body=es_query)
    return format_results(results)
```
- **前提条件**：`city.keyword` 是 `keyword` 类型（未分词），确保 `term` 查询能精确匹配。
- **适用场景**：用户输入明确包含城市名和店铺名（如 `杭州市纯味斑鱼府`）。

---

#### **3. 优化 IK 分词器的词典**
如果 IK 分词器无法正确切分城市名（如 `杭州市` 被拆分为 `杭州` + `市`），可以通过 **自定义词典** 强制保留完整城市名：
```bash
# 在 IK 分词器的 custom.dic 中添加
杭州市
纯味斑鱼府
```
- **优点**：确保 `analyze` API 返回 `["杭州市", "纯味斑鱼府"]`。
- **缺点**：需要维护词典，且无法处理所有可能的城市名。

---

#### **4. 混合方案：分词后结合规则拆分**
如果必须手动拆分字段，可以结合分词结果 + 规则逻辑：
```python
def analyze_query(query):
    response = es.indices.analyze(
        body={"analyzer": "ik_max_word", "text": query}
    )
    tokens = [token["token"] for token in response["tokens"]]
    return tokens

def search_shops(query):
    tokens = analyze_query(query)  # e.g., ["杭州", "市", "纯味", "斑鱼府"]
    
    # 规则 1: 城市名通常以 "市" 结尾
    city_candidate = None
    for i, token in enumerate(tokens):
        if token == "市":
            city_candidate = "".join(tokens[:i+1])
            break
    
    # 规则 2: 剩余部分作为 shop_name
    shop_name_candidate = "".join(tokens[len(city_candidate.split()):]) if city_candidate else query
    
    es_query = {
        "query": {
            "bool": {
                "must": [
                    {"match_phrase": {"city.keyword": city_candidate}},
                    {"match_phrase": {"shop_name": shop_name_candidate}}
                ]
            }
        }
    }

    results = es.search(index="shops", body=es_query)
    return format_results(results)
```
- **优点**：通过规则逻辑粗略拆分字段。
- **缺点**：规则可能无法覆盖所有情况（如 `上海市静安区` 会被误判为 `city=上海市静安区`）。

---

### **最终建议**
| **方案** | **适用场景** | **推荐程度** |
|----------|--------------|--------------|
| **`multi_match` 查询** | 用户输入模糊（如 `杭州市纯味斑鱼府`），无需严格字段拆分 | ✅ 推荐 |
| **`bool` + `match_phrase`** | 用户输入明确包含城市名和店铺名 | ✅ 推荐 |
| **自定义 IK 词典** | 需要强制保留完整城市名 | ⚠️ 需要维护词典 |
| **规则拆分 + `match_phrase`** | 极端场景（如必须手动拆分字段） | ⚠️ 复杂度高 |

---

### **总结**
你的核心问题在于 **IK 分词器的输出无法直接映射到结构化字段**。最合理的解决方案是：
1. **放弃手动拆分**，直接使用 `multi_match` 或 `bool` 查询。
2. **通过 Elasticsearch 的查询能力**（如 `match_phrase`、`term`）实现精准匹配。
3. **必要时优化 IK 分词器的词典**，但不要过度依赖手动拆分逻辑。