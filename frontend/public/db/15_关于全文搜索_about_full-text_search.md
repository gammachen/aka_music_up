# 全文搜索技术详解

## 一、全文搜索基本概念

全文搜索（Full-text Search）是一种通过扫描文档中的所有词语来找到匹配查询条件的文档的技术。与简单的字符串匹配不同，全文搜索能够：

- 理解文本的语义结构
- 处理词形变化（如单复数、时态）
- 忽略停用词（如"the"、"and"等）
- 支持相关性排序
- 处理同义词和近义词

全文搜索在各类应用中广泛使用，从简单的网站内容搜索到复杂的企业级搜索引擎，都依赖于这项技术。

## 二、传统数据库模式匹配的局限性

### 1. 性能问题

传统的SQL数据库使用`LIKE`或`REGEXP`进行文本匹配时存在严重的性能瓶颈：

```sql
EXPLAIN 
SELECT * FROM articles 
WHERE content LIKE '%database%';
```

**执行计划特征**：
- type: ALL（全表扫描）
- rows: 120,000（扫描大量行）
- filtered: 100.00（无法过滤）
- Extra: Using where（在服务器层面过滤）

这种查询方式会导致：
- 无法利用索引
- CPU使用率高（78%）
- 内存消耗大（1.2GB）
- 响应时间长（850ms/120k行）

### 2. 精确度问题

模式匹配在词边界识别上也存在问题：

**测试数据**：
```sql
INSERT INTO search_test(content) VALUES 
('Save your money'),
('Lonely programmer'),
('One plus one equals two');
```

**问题查询**：
```sql
SELECT * FROM search_test 
WHERE content REGEXP '\\bone\\b';
```

**实际匹配结果**：
- 'Lonely programmer'（错误匹配到"one"中的"on"）
- 'One plus one...'（正确匹配）

### 3. 改进尝试

**MySQL改进方案**：
```sql
SELECT * FROM search_test 
WHERE content REGEXP '[[:<:]]one[[:>:]]';
```

虽然可以通过复杂的正则表达式改进匹配精度，但这进一步降低了性能，且仍无法解决索引问题。

## 三、全文索引技术原理

### 1. 倒排索引结构

全文搜索的核心是倒排索引（Inverted Index）：

- **正向索引**：文档ID → 文档内容
- **倒排索引**：词语 → 包含该词语的文档ID列表

倒排索引包含：
- 词典（Dictionary）：所有不重复词语的集合
- 倒排表（Posting List）：每个词语对应的文档ID列表
- 位置信息（Position）：词语在文档中的位置

### 2. 分词技术

分词（Tokenization）是将文本拆分为单词或词语的过程：

- **基于规则的分词**：使用空格、标点等作为分隔符
- **基于统计的分词**：使用词频和共现概率
- **基于词典的分词**：使用预定义词典
- **混合分词**：结合多种方法

### 3. 文本处理流程

1. **文本分析**：将原始文本转换为标记（tokens）
2. **标准化**：转换为小写，删除标点
3. **词干提取**：将单词还原为词根形式（如"running" → "run"）
4. **停用词过滤**：移除常见但无意义的词
5. **同义词扩展**：添加同义词
6. **索引构建**：创建倒排索引

## 四、主流全文搜索解决方案对比

### 1. 数据库内置全文索引

**MySQL FULLTEXT索引**：
```sql
ALTER TABLE articles ADD FULLTEXT(title, body);
SELECT * FROM articles WHERE MATCH(title, body) AGAINST('search term');
```

**PostgreSQL全文搜索**：
```sql
SELECT * FROM articles WHERE to_tsvector('english', body) @@ to_tsquery('english', 'search & term');
```

**优点**：
- 集成在数据库中，无需额外服务
- 适合中小规模应用
- 与事务和ACID特性兼容

**缺点**：
- 功能相对有限
- 扩展性不如专业搜索引擎
- 性能在大规模数据下不佳

### 2. 专业搜索引擎

**Elasticsearch配置示例**：
```json
PUT /articles
{
  "mappings": {
    "properties": {
      "content": {
        "type": "text",
        "analyzer": "english"
      }
    }
  }
}
```

**Solr配置示例**：
```xml
<field name="content" type="text_general" indexed="true" stored="true" />
```

**优点**：
- 高性能（响应时间35ms vs LIKE的850ms）
- 低资源消耗（CPU 12% vs 78%，内存280MB vs 1.2GB）
- 丰富的分词器和分析器
- 支持复杂查询语法和相关性排序
- 高扩展性和分布式架构

**缺点**：
- 需要额外维护服务
- 与数据库同步需要额外工作
- 学习曲线较陡

## 五、实际应用案例与优化建议

### 1. 混合搜索策略

在实际应用中，可以结合多种搜索策略：

- 简单查询使用数据库索引
- 复杂文本搜索使用专业搜索引擎
- 使用缓存减少搜索压力

### 2. 索引优化技巧

- **选择合适的分词器**：根据语言和业务场景
- **自定义同义词**：增强搜索相关性
- **字段权重**：为不同字段设置不同权重
- **高亮显示**：突出显示匹配的文本片段

### 3. 性能调优

- **索引分片**：合理设置分片数量
- **预热缓存**：提前加载常用数据
- **批量操作**：使用批量API减少网络开销
- **定期重建索引**：优化索引结构

## 六、技术方案对比总结

| 维度       | LIKE/REGEXP          | 数据库全文索引      | 专业搜索引擎(ES/Solr) |
|------------|----------------------|-------------------|----------------------|
| 响应时间   | 850ms（120k行）      | 200-300ms         | 35ms                 |
| CPU占用    | 78%                 | 40-50%            | 12%                  |
| 内存消耗   | 1.2GB               | 500-700MB         | 280MB                |
| 精度控制   | 需复杂正则          | 基础分词支持       | 丰富的分词器和过滤器  |
| 扩展性     | 差                  | 中等              | 优秀                 |
| 部署复杂度 | 低                  | 低                | 高                   |
| 适用场景   | 小数据量简单匹配     | 中等数据量基础搜索  | 大数据量复杂搜索需求  |

## 七、相关技术文档

1. 索引失效原理（参见13_about_index.md#无法使用索引的查询场景）
2. Elasticsearch索引设计（参见26_about_elasticsearch_index.md）
3. SQL注入防御方案（参见11_about_sql_injection.md）

---

通过合理选择和配置全文搜索技术，可以显著提高应用的搜索性能和用户体验。在选择技术方案时，应综合考虑数据规模、查询复杂度、性能需求和维护成本等因素。