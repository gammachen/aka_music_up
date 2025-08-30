# 基于Elasticsearch的电影搜索系统技术方案

## 1. 系统概述

### 1.1 系统目标
构建一个高性能、多维度、支持全文搜索的电影信息检索系统，提供电影名称、演员、导演、评分内容和标签等多维度的搜索能力。

### 1.2 核心功能
- 电影信息索引和更新
- 多维度搜索（名称、演员、导演、评分内容、标签）
- 搜索结果高亮显示
- 分页查询
- 模糊匹配

## 2. 技术架构

### 2.1 系统组件
- **Elasticsearch**：核心搜索引擎
- **Python Flask**：后端服务框架
- **MySQL**：原始数据存储
- **Elasticsearch Python Client**：ES客户端

### 2.2 数据流程
```
MySQL -> Python数据处理 -> Elasticsearch索引 -> 搜索服务 -> 前端展示
```

## 3. 数据模型设计

### 3.1 电影数据结构
```json
{
  "movie_id": "唯一标识",
  "name": "电影名称",
  "actors": "演员列表",
  "director": "导演",
  "ratings": [
    {
      "rating_id": "评分ID",
      "rating_detail": "评分内容"
    }
  ],
  "tags": [
    {
      "tag_id": "标签ID",
      "tag": "标签内容"
    }
  ]
}
```

### 3.2 索引映射
```json
{
  "mappings": {
    "properties": {
      "name": { "type": "text" },
      "actors": { "type": "text" },
      "director": { "type": "text" },
      "ratings": {
        "properties": {
          "rating_id": { "type": "keyword" },
          "rating_detail": { "type": "text" }
        }
      },
      "tags": {
        "properties": {
          "tag_id": { "type": "keyword" },
          "tag": { "type": "text" }
        }
      }
    }
  }
}
```

## 4. 核心功能实现

### 4.1 数据初始化
```python
def load_movies_to_es():
    """将电影数据从MySQL加载到Elasticsearch"""
    with app.app_context():
        movies = Movie.query.all()
        for movie in movies:
            movie_data = movie.to_dict()
            movie_data['ratings'] = []
            movie_data['tags'] = []
            
            # 加载评分数据
            ratings = Rating.query.filter_by(movie_id=movie.movie_id).all()
            for rating in ratings:
                rating_data = {
                    'rating_id': rating.rating_id,
                    'rating_detail': rating.rating_detail.content if rating.rating_detail else ''
                }
                movie_data['ratings'].append(rating_data)
            
            # 加载标签数据
            tags = Tag.query.filter_by(movie_id=movie.movie_id).all()
            for tag in tags:
                tag_data = {
                    'tag_id': tag.tag_id,
                    'tag': tag.tag
                }
                movie_data['tags'].append(tag_data)
                
            es.index(index=INDEX_NAME, id=movie.movie_id, body=movie_data)
```

### 4.2 多维度搜索
```python
def search_movies(query, page=1, page_size=10):
    """多维度搜索电影数据"""
    from_ = (page - 1) * page_size
    search_body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": [
                    "name^2",          # 名称权重最高
                    "actors^1.5",      # 演员次之
                    "director^1.5",    # 导演次之
                    "ratings.rating_detail^1.5",  # 评分内容
                    "tags.tag^1.5"     # 标签
                ],
                "fuzziness": "AUTO"    # 自动模糊匹配
            }
        },
        "highlight": {
            "fields": {
                "name": {},
                "actors": {},
                "director": {},
                "ratings.rating_detail": {},
                "tags.tag": {}
            }
        },
        "from": from_,
        "size": page_size
    }
    response = es.search(index=INDEX_NAME, body=search_body)
    return response['hits']['hits'], total_pages
```

### 4.3 单维度搜索
```python
def search_movies_by_name(query):
    """按电影名称搜索"""
    search_body = {
        "query": {
            "match": {
                "name": query
            }
        }
    }
    return es.search(index=INDEX_NAME, body=search_body)['hits']['hits']

def search_movies_by_actor(query):
    """按演员搜索"""
    search_body = {
        "query": {
            "match": {
                "actors": query
            }
        }
    }
    return es.search(index=INDEX_NAME, body=search_body)['hits']['hits']
```

## 5. 性能优化

### 5.1 索引优化
- 使用合适的数据类型
- 设置合理的分片数
- 配置适当的副本数

### 5.2 查询优化
- 使用字段权重
- 实现模糊匹配
- 优化分页查询

### 5.3 缓存策略
- 使用Elasticsearch的查询缓存
- 实现热点数据缓存
- 配置适当的刷新间隔

## 6. 高可用设计

### 6.1 集群部署
- 多节点部署
- 主从复制
- 数据分片

### 6.2 容错机制
- 自动故障转移
- 数据备份恢复
- 监控告警

## 7. 监控和维护

### 7.1 性能监控
- 查询响应时间
- 索引速度
- 系统资源使用

### 7.2 数据维护
- 定期数据同步
- 索引优化
- 数据清理

## 8. 最佳实践

### 8.1 开发建议
- 使用批量操作
- 合理设置超时时间
- 实现重试机制

### 8.2 运维建议
- 定期备份数据
- 监控系统健康
- 及时处理告警

## 9. 总结

本方案基于Elasticsearch构建了一个高性能的电影搜索系统，实现了多维度搜索、结果高亮、分页查询等功能。通过合理的索引设计、查询优化和高可用部署，确保了系统的性能和可靠性。系统可以方便地扩展新的搜索维度和功能，满足不断变化的业务需求。