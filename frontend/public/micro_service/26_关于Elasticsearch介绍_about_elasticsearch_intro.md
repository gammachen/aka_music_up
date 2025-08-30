# Elasticsearch 详细指南

## 1. 什么是 Elasticsearch

Elasticsearch 是一个开源的分布式搜索和分析引擎，基于 Apache Lucene 构建。它提供了一个分布式、多租户的全文搜索引擎，具有 HTTP Web 接口和无模式 JSON 文档。

### 1.1 核心特性
- **分布式搜索**：自动分片和复制，支持水平扩展
- **实时分析**：近实时搜索和分析能力
- **全文搜索**：强大的全文搜索功能
- **多租户**：支持多索引和多类型
- **RESTful API**：简单易用的 HTTP 接口
- **高可用性**：自动故障转移和恢复

### 1.2 应用场景
- 企业搜索
- 日志分析
- 安全分析
- 业务分析
- 应用性能监控
- 地理空间分析

## 2. 关键概念

### 2.1 基本概念
- **索引（Index）**：类似数据库中的数据库
- **类型（Type）**：类似数据库中的表（7.x版本后已废弃）
- **文档（Document）**：类似数据库中的行
- **字段（Field）**：类似数据库中的列
- **映射（Mapping）**：类似数据库中的表结构

### 2.2 集群概念
- **节点（Node）**：运行中的 Elasticsearch 实例
- **集群（Cluster）**：一个或多个节点的集合
- **分片（Shard）**：索引的子集，用于分布式存储
- **副本（Replica）**：分片的备份，用于高可用

### 2.3 数据概念
- **倒排索引**：用于快速全文搜索的数据结构
- **分析器（Analyzer）**：用于文本分析和处理
- **查询（Query）**：用于搜索和过滤数据
- **聚合（Aggregation）**：用于数据分析和统计

## 3. 基本架构

### 3.1 集群架构
```
+------------------+     +------------------+     +------------------+
|     Node 1       |     |     Node 2       |     |     Node 3       |
| +--------------+ |     | +--------------+ |     | +--------------+ |
| |  Master Node | |     | |  Data Node   | |     | |  Data Node   | |
| +--------------+ |     | +--------------+ |     | +--------------+ |
| |  Data Node   | |     | |  Ingest Node | |     | |  Ingest Node | |
| +--------------+ |     | +--------------+ |     | +--------------+ |
+------------------+     +------------------+     +------------------+
```

### 3.2 节点类型
- **主节点（Master Node）**：负责集群管理
- **数据节点（Data Node）**：存储数据
- **协调节点（Coordinating Node）**：处理请求
- **摄取节点（Ingest Node）**：处理数据预处理

### 3.3 数据流
1. 客户端发送请求到协调节点
2. 协调节点将请求路由到相关分片
3. 数据节点处理请求并返回结果
4. 协调节点聚合结果并返回给客户端

## 4. API 使用方法

### 4.1 索引操作

#### 4.1.1 创建索引
```bash
# 创建索引
PUT /my_index
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1
  },
  "mappings": {
    "properties": {
      "title": {
        "type": "text",
        "analyzer": "standard"
      },
      "content": {
        "type": "text",
        "analyzer": "standard"
      },
      "date": {
        "type": "date"
      }
    }
  }
}
```

#### 4.1.2 删除索引
```bash
# 删除索引
DELETE /my_index
```

### 4.2 文档操作

#### 4.2.1 添加文档
```bash
# 添加文档
POST /my_index/_doc
{
  "title": "Elasticsearch 入门",
  "content": "这是一个关于 Elasticsearch 的入门教程",
  "date": "2023-01-01"
}
```

#### 4.2.2 更新文档
```bash
# 更新文档
POST /my_index/_update/1
{
  "doc": {
    "content": "更新后的内容"
  }
}
```

#### 4.2.3 删除文档
```bash
# 删除文档
DELETE /my_index/_doc/1
```

### 4.3 搜索操作

#### 4.3.1 简单搜索
```bash
# 简单搜索
GET /my_index/_search
{
  "query": {
    "match": {
      "title": "入门"
    }
  }
}
```

#### 4.3.2 复杂搜索
```bash
# 复杂搜索
GET /my_index/_search
{
  "query": {
    "bool": {
      "must": [
        { "match": { "title": "入门" } },
        { "range": { "date": { "gte": "2023-01-01" } } }
      ]
    }
  },
  "sort": [
    { "date": "desc" }
  ],
  "from": 0,
  "size": 10
}
```

### 4.4 聚合操作

#### 4.4.1 基本聚合
```bash
# 基本聚合
GET /my_index/_search
{
  "aggs": {
    "avg_date": {
      "avg": {
        "field": "date"
      }
    }
  }
}
```

#### 4.4.2 复杂聚合
```bash
# 复杂聚合
GET /my_index/_search
{
  "aggs": {
    "group_by_date": {
      "date_histogram": {
        "field": "date",
        "calendar_interval": "month"
      },
      "aggs": {
        "avg_content_length": {
          "avg": {
            "script": "doc['content'].value.length()"
          }
        }
      }
    }
  }
}
```

## 5. 最佳实践

### 5.1 索引设计
- 合理设置分片数量
- 使用别名管理索引
- 定期优化索引
- 设置合理的映射

### 5.2 查询优化
- 使用过滤器缓存
- 避免深度分页
- 使用批量操作
- 优化查询语句

### 5.3 集群管理
- 监控集群健康状态
- 定期备份数据
- 合理配置节点角色
- 设置适当的副本数

### 5.4 性能优化
- 使用合适的硬件配置
- 优化 JVM 设置
- 合理设置刷新间隔
- 使用索引模板

## 6. 常见问题

### 6.1 性能问题
- 查询响应慢
- 索引速度慢
- 内存使用高
- CPU 使用率高

### 6.2 可用性问题
- 节点离线
- 分片未分配
- 集群状态异常
- 数据不一致

### 6.3 数据问题
- 数据丢失
- 数据重复
- 数据不一致
- 索引损坏