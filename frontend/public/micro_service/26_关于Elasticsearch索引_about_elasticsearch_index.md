# Elasticsearch 索引详细指南

## 1. 索引概述

### 1.1 什么是索引
索引是Elasticsearch中存储、索引和搜索文档的基本单位。它类似于关系型数据库中的表，但提供了更强大的搜索和分析能力。

### 1.2 索引的核心特性
- **分布式存储**：数据被分片存储在多个节点上
- **实时搜索**：支持近实时的数据索引和搜索
- **高可用性**：通过副本机制确保数据安全
- **灵活映射**：支持动态和显式字段映射
- **全文搜索**：内置强大的全文搜索能力

## 2. 索引操作

### 2.1 创建索引
```json
PUT /my_index
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "refresh_interval": "1s"
  },
  "mappings": {
    "properties": {
      "title": {
        "type": "text",
        "analyzer": "standard"
      },
      "content": {
        "type": "text",
        "analyzer": "ik_max_word"
      },
      "create_time": {
        "type": "date"
      },
      "author": {
        "type": "keyword"
      }
    }
  }
}
```

### 2.2 索引设置
```json
PUT /my_index/_settings
{
  "index": {
    "refresh_interval": "5s",
    "number_of_replicas": 2,
    "max_result_window": 10000
  }
}
```

### 2.3 索引别名
```json
POST /_aliases
{
  "actions": [
    {
      "add": {
        "index": "my_index",
        "alias": "my_alias"
      }
    }
  ]
}
```

## 3. 索引映射

### 3.1 字段类型
- **核心类型**
  - text：全文搜索字段
  - keyword：精确值字段
  - date：日期时间
  - long：长整型
  - integer：整型
  - double：双精度浮点型
  - boolean：布尔值
  - ip：IP地址

- **复杂类型**
  - object：对象类型
  - nested：嵌套类型
  - array：数组类型

### 3.2 映射示例
```json
PUT /products
{
  "mappings": {
    "properties": {
      "name": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      },
      "price": {
        "type": "double"
      },
      "tags": {
        "type": "keyword"
      },
      "description": {
        "type": "text",
        "analyzer": "ik_max_word"
      },
      "specifications": {
        "type": "nested",
        "properties": {
          "key": {
            "type": "keyword"
          },
          "value": {
            "type": "text"
          }
        }
      }
    }
  }
}
```

## 4. 索引优化

### 4.1 分片策略
```json
PUT /large_index
{
  "settings": {
    "number_of_shards": 5,
    "number_of_replicas": 1,
    "routing": {
      "allocation": {
        "total_shards_per_node": 3
      }
    }
  }
}
```

### 4.2 索引模板
```json
PUT /_index_template/my_template
{
  "index_patterns": ["logs-*"],
  "template": {
    "settings": {
      "number_of_shards": 3,
      "number_of_replicas": 1
    },
    "mappings": {
      "properties": {
        "timestamp": {
          "type": "date"
        },
        "level": {
          "type": "keyword"
        },
        "message": {
          "type": "text"
        }
      }
    }
  }
}
```

### 4.3 索引生命周期管理
```json
PUT /_ilm/policy/my_policy
{
  "policy": {
    "phases": {
      "hot": {
        "actions": {
          "rollover": {
            "max_size": "50gb",
            "max_age": "30d"
          }
        }
      },
      "warm": {
        "min_age": "7d",
        "actions": {
          "forcemerge": {
            "max_num_segments": 1
          }
        }
      },
      "delete": {
        "min_age": "90d",
        "actions": {
          "delete": {}
        }
      }
    }
  }
}
```

## 5. 索引监控

### 5.1 索引统计
```json
GET /my_index/_stats
{
  "docs": {
    "count": 1000,
    "deleted": 0
  },
  "store": {
    "size_in_bytes": 1048576
  },
  "indexing": {
    "index_total": 1000,
    "index_time_in_millis": 1000
  }
}
```

### 5.2 索引健康状态
```json
GET /_cluster/health/my_index
{
  "status": "green",
  "number_of_shards": 3,
  "number_of_replicas": 1,
  "active_shards": 6,
  "unassigned_shards": 0
}
```

## 6. 索引维护

### 6.1 索引优化
```json
POST /my_index/_forcemerge
{
  "max_num_segments": 1
}
```

### 6.2 索引清理
```json
DELETE /my_index
```

### 6.3 索引重建
```json
POST /_reindex
{
  "source": {
    "index": "old_index"
  },
  "dest": {
    "index": "new_index"
  }
}
```

## 7. 最佳实践

### 7.1 索引设计
- 合理设置分片数量
- 使用合适的字段类型
- 配置适当的副本数
- 设置合理的刷新间隔

### 7.2 性能优化
- 使用索引模板
- 实施生命周期管理
- 定期优化索引
- 监控索引状态

### 7.3 高可用性
- 配置足够的副本
- 使用索引别名
- 实施备份策略
- 设置故障转移机制

### 7.4 索引创建最佳实践

#### 7.4.1 分片策略
- **分片数量计算**：分片数 = 数据总大小 / 单分片理想大小（通常为20-40GB）
- **避免过度分片**：每个节点的分片数应控制在20-25个以内
- **考虑未来增长**：预留30%-50%的增长空间，但不要过度预留
- **单节点分片限制**：设置`cluster.routing.allocation.total_shards_per_node`防止单节点分片过多

#### 7.4.2 字段映射优化
- **显式定义映射**：避免依赖动态映射，特别是对于生产环境
- **字段类型精确选择**：
  - 使用`keyword`而非`text`存储不需要分词的字段
  - 使用`integer`/`short`而非`long`存储范围较小的数值
  - 对日期使用标准格式并指定`format`
- **控制字段数量**：避免过多字段（尤其是`text`类型），考虑使用嵌套对象
- **禁用不需要的特性**：
  ```json
  "_source": { "enabled": false },  // 如果不需要返回原始文档
  "_all": { "enabled": false },     // 7.0+默认禁用
  "index.codec": "best_compression" // 优化存储空间
  ```

#### 7.4.3 分析器选择
- **语言相关分析器**：根据文档语言选择合适的分析器（如中文使用`ik_smart`/`ik_max_word`）
- **自定义分析器**：针对特定业务场景定制分析链
  ```json
  "analysis": {
    "analyzer": {
      "my_custom_analyzer": {
        "type": "custom",
        "tokenizer": "standard",
        "filter": ["lowercase", "asciifolding", "my_synonym"]
      }
    },
    "filter": {
      "my_synonym": {
        "type": "synonym",
        "synonyms_path": "synonyms.txt"
      }
    }
  }
  ```
- **多字段策略**：同一内容使用不同分析器索引
  ```json
  "title": {
    "type": "text",
    "analyzer": "standard",
    "fields": {
      "keyword": { "type": "keyword" },
      "ik": { "type": "text", "analyzer": "ik_smart" }
    }
  }
  ```

#### 7.4.4 索引设置调优
- **刷新间隔**：批量导入时增大`refresh_interval`（如"30s"或"-1"）
- **事务日志**：批量导入时调整`index.translog.durability`为"async"
- **段合并策略**：
  ```json
  "index.merge.policy.max_merged_segment": "5gb",
  "index.merge.policy.segments_per_tier": 4
  ```
- **批量写入优化**：
  ```json
  "index.number_of_replicas": 0,  // 导入时先设为0，完成后再增加
  "index.routing.allocation.total_shards_per_node": 2
  ```

#### 7.4.5 别名管理
- **零停机索引切换**：使用别名实现无缝切换
  ```json
  POST /_aliases
  {
    "actions": [
      { "remove": { "index": "old_index", "alias": "current_index" }},
      { "add": { "index": "new_index", "alias": "current_index" }}
    ]
  }
  ```
- **读写分离**：为同一索引创建读写分离的别名
  ```json
  POST /_aliases
  {
    "actions": [
      { "add": { "index": "my_index", "alias": "my_index_read" }},
      { "add": { 
          "index": "my_index", 
          "alias": "my_index_write",
          "is_write_index": true 
        }
      }
    ]
  }
  ```
- **时间序列索引**：结合ILM和别名管理时间序列数据
  ```json
  PUT /_template/logs_template
  {
    "index_patterns": ["logs-*"],
    "settings": {
      "index.lifecycle.name": "logs_policy",
      "index.lifecycle.rollover_alias": "logs"
    }
  }
  ```

#### 7.4.6 索引划分策略
- **按业务类型划分索引**：
  - 将不同类型的数据存储在不同的索引中，便于管理和扩展
  - 例如，商品信息存储在 `products` 索引，用户行为日志存储在 `user_logs` 索引
  - 优势：便于针对不同业务数据设置不同的分片策略和映射配置
  ```json
  PUT /products
  {
    "settings": {
      "number_of_shards": 3,
      "number_of_replicas": 1
    },
    "mappings": {
      "properties": {
        "name": { "type": "text" },
        "price": { "type": "double" }
      }
    }
  }
  
  PUT /user_logs
  {
    "settings": {
      "number_of_shards": 5,
      "number_of_replicas": 1
    },
    "mappings": {
      "properties": {
        "user_id": { "type": "keyword" },
        "action": { "type": "keyword" },
        "timestamp": { "type": "date" }
      }
    }
  }
  ```

- **按时间划分索引**：
  - 对于时间序列数据（如日志、订单记录等），按天、周或月创建索引
  - 例如，`logs-2023-10-01`、`orders-2023-10`
  - 优势：便于数据生命周期管理，可以轻松删除或归档旧数据
  ```json
  # 使用索引模板自动创建时间序列索引
  PUT /_index_template/daily_logs
  {
    "index_patterns": ["logs-*"],
    "template": {
      "settings": {
        "number_of_shards": 2,
        "number_of_replicas": 1
      },
      "mappings": {
        "properties": {
          "timestamp": { "type": "date" },
          "message": { "type": "text" },
          "level": { "type": "keyword" }
        }
      }
    }
  }
  ```

- **使用索引模板**：
  - 定义索引模板，在创建新索引时自动应用预定义的设置和映射
  - 例如，为所有商品索引定义相同的字段类型和分析器
  - 优势：确保索引配置一致性，减少重复工作和配置错误
  ```json
  PUT /_index_template/product_template
  {
    "index_patterns": ["product*", "item*"],
    "priority": 1,
    "template": {
      "settings": {
        "number_of_shards": 3,
        "analysis": {
          "analyzer": {
            "product_analyzer": {
              "type": "custom",
              "tokenizer": "standard",
              "filter": ["lowercase", "asciifolding"]
            }
          }
        }
      },
      "mappings": {
        "properties": {
          "name": { 
            "type": "text",
            "analyzer": "product_analyzer",
            "fields": {
              "keyword": { "type": "keyword" }
            }
          },
          "category": { "type": "keyword" },
          "created_at": { "type": "date" }
        }
      }
    }
  }
  ```

- **使用别名简化查询**：
  - 创建索引别名，简化客户端调用逻辑，避免硬编码索引名称
  - 例如，`products_alias` 可以指向多个实际索引
  - 优势：实现零停机索引重建，支持索引轮换和数据迁移
  ```json
  # 创建指向多个索引的别名
  POST /_aliases
  {
    "actions": [
      { "add": { "index": "products-2023", "alias": "products_current" }},
      { "add": { "index": "products-2022", "alias": "products_archive" }},
      { "add": { "index": "products-*", "alias": "products_all" }}
    ]
  }
  
  # 使用别名进行查询
  GET /products_current/_search
  {
    "query": { "match_all": {} }
  }
  ```

## 8. 常见问题

### 8.1 索引性能问题
- 分片数量过多
- 刷新间隔过短
- 字段映射不合理
- 查询性能瓶颈

### 8.2 索引管理问题
- 索引膨胀
- 磁盘空间不足
- 副本分配不均
- 节点负载过高

### 8.3 解决方案
- 优化分片策略
- 调整刷新间隔
- 实施索引生命周期管理
- 监控和告警机制
