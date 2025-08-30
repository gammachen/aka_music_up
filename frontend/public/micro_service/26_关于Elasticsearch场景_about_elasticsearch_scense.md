# Elasticsearch在电商系统中的应用场景

在电商系统中，Elasticsearch可以用于多种场景。以下是常见的应用场景及其实现方式，每个场景都附带了初始化脚本，方便验证查询示例：

```shell
curl -u elastic:mypassword https://localhost:9200 --insecure
```

### 1. **商品搜索**
   - **功能**：用户可以通过关键词搜索商品。
   - **实现方式**：
     - 使用 Elasticsearch 的全文搜索功能，支持模糊匹配、同义词扩展等。
     - 配置 `title`、`description` 等字段为可搜索字段。
     - 示例查询：
       ```json
       {
         "query": {
           "multi_match": {
             "query": "手机",
             "fields": ["title", "description"]
           }
         }
       }
       ```
   - **初始化脚本**：
     - 创建索引并定义映射：
       ```bash
       # 创建商品索引
       curl --insecure -u elastic:mypassword -X PUT "https://localhost:9200/products" -H "Content-Type:application/json" -d'
       {
         "settings": {
           "analysis": {
             "analyzer": {
               "text_analyzer": {
                 "type": "custom",
                 "tokenizer": "standard",
                 "filter": ["lowercase", "asciifolding"]
               }
             }
           }
         },
         "mappings": {
           "properties": {
             "id": { "type": "keyword" },
             "title": { 
               "type": "text", 
               "analyzer": "text_analyzer",
               "fields": {
                 "keyword": { "type": "keyword" }
               }
             },
             "description": { 
               "type": "text", 
               "analyzer": "text_analyzer" 
             },
             "price": { "type": "float" },
             "category": { 
               "type": "text",
               "fields": {
                 "keyword": { "type": "keyword" }
               }
             },
             "brand": { 
               "type": "text",
               "fields": {
                 "keyword": { "type": "keyword" }
               }
             },
             "in_stock": { "type": "boolean" },
             "stock_count": { "type": "integer" },
             "tags": { "type": "keyword" },
             "sales": { "type": "integer" },
             "rating": { "type": "float" },
             "created_at": { "type": "date" }
           }
         }
       }'
       ```
     
     - 添加示例数据：
       ```bash
       # 添加示例商品数据
       curl --insecure -u elastic:mypassword -X POST "https://localhost:9200/products/_bulk" -H "Content-Type: application/json" -d'
{"index":{"_id":"1"}}
{"id":"1","title":"iPhone 13 Pro","description":"Apple iPhone 13 Pro 智能手机，搭载A15仿生芯片","price":7999.00,"category":"电子产品","brand":"Apple","in_stock":true,"stock_count":100,"tags":["手机","苹果","智能手机"],"sales":500,"rating":4.8,"created_at":"2023-01-01"}
{"index":{"_id":"2"}}
{"id":"2","title":"Samsung Galaxy S21","description":"三星Galaxy S21 5G智能手机，高通骁龙888处理器","price":5999.00,"category":"电子产品","brand":"Samsung","in_stock":true,"stock_count":85,"tags":["手机","三星","智能手机","5G"],"sales":350,"rating":4.6,"created_at":"2023-01-15"}
{"index":{"_id":"3"}}
{"id":"3","title":"小米12 Pro","description":"小米12 Pro 5G智能手机，搭载骁龙8 Gen 1处理器","price":4999.00,"category":"电子产品","brand":"小米","in_stock":true,"stock_count":120,"tags":["手机","小米","智能手机","5G"],"sales":420,"rating":4.5,"created_at":"2023-02-01"}
{"index":{"_id":"4"}}
{"id":"4","title":"华为MateBook X Pro","description":"华为MateBook X Pro 2022款笔记本电脑","price":9999.00,"category":"电子产品","brand":"华为","in_stock":true,"stock_count":50,"tags":["笔记本","华为","电脑"],"sales":150,"rating":4.7,"created_at":"2023-02-15"}
{"index":{"_id":"5"}}
{"id":"5","title":"Apple MacBook Pro","description":"Apple MacBook Pro 搭载M2芯片的专业笔记本电脑","price":12999.00,"category":"电子产品","brand":"Apple","in_stock":true,"stock_count":30,"tags":["笔记本","苹果","电脑"],"sales":200,"rating":4.9,"created_at":"2023-03-01"}
'
        ```
     
     - 验证搜索功能：
       ```bash
       # 执行搜索查询
      curl --insecure -u elastic:mypassword -X GET "https://localhost:9200/products/_search" -H "Content-Type: application/json" -d'
       {
         "query": {
           "multi_match": {
             "query": "手机",
             "fields": ["title", "description"]
           }
         }
       }'
       ```

     - 结果：
        ```bash
        {
            "took": 464,
            "timed_out": false,
            "_shards": {
                "total": 1,
                "successful": 1,
                "skipped": 0,
                "failed": 0
            },
            "hits": {
                "total": {
                    "value": 3,
                    "relation": "eq"
                },
                "max_score": 1.1007943,
                "hits": [{
                    "_index": "products",
                    "_id": "1",
                    "_score": 1.1007943,
                    "_source": {
                        "id": "1",
                        "title": "iPhone 13 Pro",
                        "description": "Apple iPhone 13 Pro 智能手机，搭载A15仿生芯片",
                        "price": 7999.00,
                        "category": "电子产品",
                        "brand": "Apple",
                        "in_stock": true,
                        "stock_count": 100,
                        "tags": ["手机", "苹果", "智能手机"],
                        "sales": 500,
                        "rating": 4.8,
                        "created_at": "2023-01-01"
                    }
                }, {
                    "_index": "products",
                    "_id": "2",
                    "_score": 1.0455089,
                    "_source": {
                        "id": "2",
                        "title": "Samsung Galaxy S21",
                        "description": "三星Galaxy S21 5G智能手机，高通骁龙888处理器",
                        "price": 5999.00,
                        "category": "电子产品",
                        "brand": "Samsung",
                        "in_stock": true,
                        "stock_count": 85,
                        "tags": ["手机", "三星", "智能手机", "5G"],
                        "sales": 350,
                        "rating": 4.6,
                        "created_at": "2023-01-15"
                    }
                }, {
                    "_index": "products",
                    "_id": "3",
                    "_score": 0.9955112,
                    "_source": {
                        "id": "3",
                        "title": "小米12 Pro",
                        "description": "小米12 Pro 5G智能手机，搭载骁龙8 Gen 1处理器",
                        "price": 4999.00,
                        "category": "电子产品",
                        "brand": "小米",
                        "in_stock": true,
                        "stock_count": 120,
                        "tags": ["手机", "小米", "智能手机", "5G"],
                        "sales": 420,
                        "rating": 4.5,
                        "created_at": "2023-02-01"
                    }
                }]
            }
        }
        ```
### 2. **过滤与筛选**
   - **功能**：用户可以根据价格范围、品牌、类别等条件筛选商品。
   - **实现方式**：
     - 使用 Elasticsearch 的 `filter` 查询，结合布尔查询（`bool query`）。
     - 示例查询：
       ```json
       {
         "query": {
           "bool": {
             "must": [
               {"match": {"category": "电子产品"}}
             ],
             "filter": [
               {"range": {"price": {"gte": 1000, "lte": 5000}}}
             ]
           }
         }
       }
       ```
    - **初始化脚本**：
      - 使用第一个场景中已创建的索引和数据，无需重新创建。
      - 验证过滤与筛选功能：
        ```bash
        # 执行过滤筛选查询
       curl --insecure -u elastic:mypassword -X GET "https://localhost:9200/products/_search" -H "Content-Type: application/json" -d'
        {
          "query": {
            "bool": {
              "must": [
                {"match": {"category": "电子产品"}}
              ],
              "filter": [
                {"range": {"price": {"gte": 1000, "lte": 5000}}}
              ]
            }
          }
        }'
        ```

        ```bash
        {
            "took": 104,
            "timed_out": false,
            "_shards": {
                "total": 1,
                "successful": 1,
                "skipped": 0,
                "failed": 0
            },
            "hits": {
                "total": {
                    "value": 1,
                    "relation": "eq"
                },
                "max_score": 0.34804547,
                "hits": [{
                    "_index": "products",
                    "_id": "3",
                    "_score": 0.34804547,
                    "_source": {
                        "id": "3",
                        "title": "小米12 Pro",
                        "description": "小米12 Pro 5G智能手机，搭载骁龙8 Gen 1处理器",
                        "price": 4999.00,
                        "category": "电子产品",
                        "brand": "小米",
                        "in_stock": true,
                        "stock_count": 120,
                        "tags": ["手机", "小米", "智能手机", "5G"],
                        "sales": 420,
                        "rating": 4.5,
                        "created_at": "2023-02-01"
                    }
                }]
            }
        }
        ```
      
      - 添加更多筛选条件示例：
        ```bash
        # 按品牌和价格范围筛选
       curl --insecure -u elastic:mypassword -X GET "https://localhost:9200/products/_search" -H "Content-Type: application/json" -d'
        {
          "query": {
            "bool": {
              "must": [
                {"match": {"category": "电子产品"}}
              ],
              "filter": [
                {"range": {"price": {"gte": 3000, "lte": 8000}}},
                {"term": {"brand.keyword": "Apple"}}
              ]
            }
          }
        }'
        ```

### 3. **排序**
   - **功能**：用户可以选择按销量、价格、评价等排序。
   - **实现方式**：
     - 使用 Elasticsearch 的 `sort` 功能。
     - 示例查询：
       ```json
       {
         "query": {
           "match_all": {}
         },
         "sort": [
           {"sales": {"order": "desc"}},
           {"price": {"order": "asc"}}
         ]
       }
       ```
     - **初始化脚本**：
       - 使用第一个场景中已创建的索引和数据，无需重新创建。
       - 验证排序功能：
         ```bash
         # 执行排序查询 - 按销量降序和价格升序
        curl --insecure -u elastic:mypassword -X GET "https://localhost:9200/products/_search" -H "Content-Type: application/json" -d'
         {
           "query": {
             "match_all": {}
           },
           "sort": [
             {"sales": {"order": "desc"}},
             {"price": {"order": "asc"}}
           ]
         }'
         ```
       
       - 其他排序示例：
         ```bash
         # 按评分降序排序
        curl --insecure -u elastic:mypassword -X GET "https://localhost:9200/products/_search" -H "Content-Type: application/json" -d'
         {
           "query": {
             "match_all": {}
           },
           "sort": [
             {"rating": {"order": "desc"}}
           ]
         }'
         ```

### 4. **推荐系统**
   - **功能**：根据用户的浏览历史或购买记录，推荐相关商品。
   - **实现方式**：
     - 结合 Elasticsearch 的相似度计算功能（如 `more_like_this` 查询）。
       - `more_like_this` 查询通过以下方式计算文档相似度:
         1. 从源文档中提取特征词条(terms)
         2. 为每个词条计算TF-IDF得分
         3. 选择得分最高的词条作为查询条件
         4. 可配置参数包括:
           - `min_term_freq`: 词条在源文档中最小出现频率
           - `max_query_terms`: 最多选择多少个词条
           - `min_doc_freq`: 词条最少在多少文档中出现
           - `max_doc_freq`: 词条最多在多少文档中出现
           - `min_word_length`: 词条最小长度
           - `max_word_length`: 词条最大长度
           - `stop_words`: 停用词列表
         5. 支持多字段匹配,可为不同字段设置权重
         6. 可以基于多个源文档计算相似度
     - 示例查询：
       ```json
       {
         "query": {
           "more_like_this": {
             "fields": ["title", "description"],
             "like": [
               {"_id": "product_id_123"}
             ],
             "min_term_freq": 1,
             "min_doc_freq": 1
           }
         }
       }
       ```
      - **初始化脚本**：
        - 使用第一个场景中已创建的索引和数据，无需重新创建。
        - 验证推荐系统功能：
          ```bash
          # 执行相似商品推荐查询
          # 这个查询使用more_like_this API来计算文档相似度
          # 相似度计算过程:
          # 1. 从源文档(id=1的iPhone)提取特征词条(如"iPhone","手机","智能手机"等)
          # 2. 为每个词条计算TF-IDF得分:
          #    - TF(词频):词条在文档中出现的次数
          #    - IDF(逆文档频率):衡量词条的区分度,出现在越少文档中的词条IDF越高
          # 3. 选择得分最高的词条作为查询条件
          # 4. 对其他文档计算相似度得分,得分由匹配词条的TF-IDF累加得到
          curl --insecure -u elastic:mypassword -X GET "https://localhost:9200/products/_search" -H "Content-Type: application/json" -d'
          {
            "query": {
              "more_like_this": {
                "fields": ["title", "description"],  # 在这两个字段中寻找相似内容
                "like": [
                  {"_id": "1"}  # 源文档ID,即iPhone 13 Pro
                ],
                "min_term_freq": 1,    # 词条在源文档中最少出现次数
                "min_doc_freq": 1,     # 词条最少在多少文档中出现
                "minimum_should_match": "30%",  # 匹配词条的最小百分比
                "boost_terms": 2.0,    # 提升词条权重
                "include": false       # 不包含源文档在结果中
              }
            },
            "explain": true  # 返回相似度计算的详细解释
          }'
          
          # 在返回结果中:
          # 1. _score字段表示相似度得分,分值越高表示越相似
          # 2. 可以通过explain=true参数查看详细的得分计算过程
          # 3. 结果会按相似度得分降序排序
          # 4. 相似商品可能包含:
          #    - 同品牌的其他产品(如其他Apple设备)
          #    - 同类别的竞品(如其他品牌手机)
          #    - 相关配件或周边产品
          ```
        
       ```bash
        {
            "took": 226,
            "timed_out": false,
            "_shards": {
                "total": 1,
                "successful": 1,
                "skipped": 0,
                "failed": 0
            },
            "hits": {
                "total": {
                    "value": 2,
                    "relation": "eq"
                },
                "max_score": 4.280247,
                "hits": [{
                    "_index": "products",
                    "_id": "5",
                    "_score": 4.280247,
                    "_source": {
                        "id": "5",
                        "title": "Apple MacBook Pro",
                        "description": "Apple MacBook Pro 搭载M2芯片的专业笔记本电脑",
                        "price": 12999.00,
                        "category": "电子产品",
                        "brand": "Apple",
                        "in_stock": true,
                        "stock_count": 30,
                        "tags": ["笔记本", "苹果", "电脑"],
                        "sales": 200,
                        "rating": 4.9,
                        "created_at": "2023-03-01"
                    }
                }, {
                    "_index": "products",
                    "_id": "3",
                    "_score": 3.527378,
                    "_source": {
                        "id": "3",
                        "title": "小米12 Pro",
                        "description": "小米12 Pro 5G智能手机，搭载骁龙8 Gen 1处理器",
                        "price": 4999.00,
                        "category": "电子产品",
                        "brand": "小米",
                        "in_stock": true,
                        "stock_count": 120,
                        "tags": ["手机", "小米", "智能手机", "5G"],
                        "sales": 420,
                        "rating": 4.5,
                        "created_at": "2023-02-01"
                    }
                }]
            }
        }
        ```

        - 基于用户历史记录的推荐示例：
          ```bash
          # 添加用户浏览历史索引
          curl --insecure -u elastic:mypassword -X PUT "https://localhost:9200/user_history" -H "Content-Type: application/json" -d'
          {
            "mappings": {
              "properties": {
                "user_id": { "type": "keyword" },
                "product_id": { "type": "keyword" },
                "view_time": { "type": "date" },
                "product_category": { "type": "keyword" },
                "product_tags": { "type": "keyword" }
              }
            }
          }'
          ```
          ```bash
          # 添加用户浏览历史数据
          
          curl --insecure -u elastic:mypassword -X POST "https://localhost:9200/user_history/_bulk" -H       "Content-Type: application/json" -d'
{"index":{}}
{"user_id":"user_001","product_id":"1","view_time":"2023-05-01T10:30:00","product_category":"电子产品","product_tags":["手机","苹果"]}
{"index":{}}
{"user_id":"user_001","product_id":"3","view_time":"2023-05-01T11:15:00","product_category":"电子产品","product_tags":["手机","小米"]}
'
          ```
          ```bash
          {"errors":false,"took":1006,"items":[{"index":{"_index":"user_history","_id":"NG_VG5YBzVTfEcZ4GmU0","_version":1,"result":"created","_shards":{"total":2,"successful":1,"failed":0},"_seq_no":0,"_primary_term":1,"status":201}},{"index":{"_index":"user_history","_id":"NW_VG5YBzVTfEcZ4GmU1","_version":1,"result":"created","_shards":{"total":2,"successful":1,"failed":0},"_seq_no":1,"_primary_term":1,"status":201}}]}
          ```

          ```bash
          # 基于用户历史查询相似商品 - 两步骤推荐流程
          
          # 步骤1: 分析用户历史数据，获取用户偏好
          curl --insecure -u elastic:mypassword -X GET "https://localhost:9200/user_history/_search" -H "Content-Type: application/json" -d'
          {
            "size": 0,
            "query": {
              "term": {"user_id": "user_001"}
            },
            "aggs": {
              "user_categories": {
                "terms": {"field": "product_category"}
              },
              "user_tags": {
                "terms": {"field": "product_tags", "size": 10}
              }
            }
          }'
          
          # 注意：如果上述查询出现字段数据错误，可以使用以下替代方案：
          # 1. 确保字段已定义为keyword类型（如上面的映射所示）
          # 2. 或者，如果字段是text类型，可以通过以下方式启用fielddata：
          curl --insecure -u elastic:mypassword -X PUT "https://localhost:9200/user_history/_mapping" -H "Content-Type: application/json" -d'
          {
            "properties": {
              "product_category": { 
                "type": "text",
                "fielddata": true
              },
              "product_tags": {
                "type": "text",
                "fielddata": true
              }
            }
          }'

          {"error":{"root_cause":[{"type":"illegal_argument_exception","reason":"Fielddata is disabled on [product_category] in [user_history]. Text fields are not optimised for operations that require per-document field data like aggregations and sorting, so these operations are disabled by default. Please use a keyword field instead. Alternatively, set fielddata=true on [product_category] in order to load field data by uninverting the inverted index. Note that this can use significant memory."}],"type":"search_phase_execution_exception","reason":"all shards failed","phase":"query","grouped":true,"failed_shards":[{"shard":0,"index":"user_history","node":"j7LRBNhTRxa2B6njS7DqMw","reason":{"type":"illegal_argument_exception","reason":"Fielddata is disabled on [product_category] in [user_history]. Text fields are not optimised for operations that require per-document field data like aggregations and sorting, so these operations are disabled by default. Please use a keyword field instead. Alternatively, set fielddata=true on [product_category] in order to load field data by uninverting the inverted index. Note that this can use significant memory."}}],"caused_by":{"type":"illegal_argument_exception","reason":"Fielddata is disabled on [product_category] in [user_history]. Text fields are not optimised for operations that require per-document field data like aggregations and sorting, so these operations are disabled by default. Please use a keyword field instead. Alternatively, set fielddata=true on [product_category] in order to load field data by uninverting the inverted index. Note that this can use significant memory.","caused_by":{"type":"illegal_argument_exception","reason":"Fielddata is disabled on [product_category] in [user_history]. Text fields are not optimised for operations that require per-document field data like aggregations and sorting, so these operations are disabled by default. Please use a keyword field instead. Alternatively, set fielddata=true on [product_category] in order to load field data by uninverting the inverted index. Note that this can use significant memory."}}},"status":400}
          
          # 步骤2: 基于用户偏好构建个性化推荐查询
          curl --insecure -u elastic:mypassword -X GET "https://localhost:9200/products/_search" -H "Content-Type: application/json" -d'
          {
            "query": {
              "function_score": {
                "query": {
                  "bool": {
                    "must": [
                      {"term": {"category.keyword": "电子产品"}}
                    ],
                    "should": [
                      {"terms": {"tags": ["手机", "苹果", "小米"]}},
                      {"terms": {"brand.keyword": ["Apple", "小米"]}}
                    ],
                    "must_not": [
                      {"ids": {"values": ["1", "3"]}}
                    ]
                  }
                },
                "functions": [
                  {
                    "filter": {"terms": {"tags": ["手机"]}},
                    "weight": 3
                  },
                  {
                    "filter": {"terms": {"brand.keyword": ["Apple", "小米"]}},
                    "weight": 2
                  },
                  {
                    "field_value_factor": {
                      "field": "rating",
                      "factor": 1.2,
                      "modifier": "log1p"
                    }
                  },
                  {
                    "field_value_factor": {
                      "field": "sales",
                      "factor": 0.001,
                      "modifier": "log1p"
                    }
                  }
                ],
                "score_mode": "sum",
                "boost_mode": "multiply"
              }
            },
            "explain": true
          }'
          
          # 说明:
          # 1. 第一步使用聚合查询分析用户历史数据，获取用户偏好的类别和标签
          # 2. 第二步使用function_score查询构建个性化推荐:
          #    - 基本查询条件确保商品类别匹配用户偏好
          #    - should条件增加匹配用户偏好标签和品牌的商品得分
          #    - must_not排除用户已浏览的商品
          #    - functions部分为不同因素设置权重:
          #      * 匹配用户偏好标签的商品权重为3
          #      * 匹配用户偏好品牌的商品权重为2
          #      * 商品评分越高，得分越高(使用log1p避免极值影响)
          #      * 销量越高，得分略微提升
          
          {"took":284,"timed_out":false,"_shards":{"total":1,"successful":1,"skipped":0,"failed":0},"hits":{"total":{"value":3,"relation":"eq"},"max_score":6.0873313,"hits":[{"_shard":"[products][0]","_node":"j7LRBNhTRxa2B6njS7DqMw","_index":"products","_id":"5","_score":6.0873313,"_source":{"id":"5","title":"Apple MacBook Pro","description":"Apple MacBook Pro 搭载M2芯片的专业笔记本电脑","price":12999.00,"category":"电子产品","brand":"Apple","in_stock":true,"stock_count":30,"tags":["笔记本","苹果","电脑"],"sales":200,"rating":4.9,"created_at":"2023-03-01"},"_explanation":{"value":6.0873313,"description":"function score, product of:","details":[{"value":2.0870113,"description":"sum of:","details":[{"value":0.08701137,"description":"weight(category.keyword:电子产品 in 4) [PerFieldSimilarity], result of:","details":[{"value":0.08701137,"description":"score(freq=1.0), computed as boost * idf * tf from:","details":[{"value":2.2,"description":"boost","details":[]},{"value":0.087011375,"description":"idf, computed as log(1 + (N - n + 0.5) / (n + 0.5)) from:","details":[{"value":5,"description":"n, number of documents containing term","details":[]},{"value":5,"description":"N, total number of documents with field","details":[]}]},{"value":0.45454544,"description":"tf, computed as freq / (freq + k1 * (1 - b + b * dl / avgdl)) from:","details":[{"value":1.0,"description":"freq, occurrences of term within document","details":[]},{"value":1.2,"description":"k1, term saturation parameter","details":[]},{"value":0.75,"description":"b, length normalization parameter","details":[]},{"value":1.0,"description":"dl, length of field","details":[]},{"value":1.0,"description":"avgdl, average length of field","details":[]}]}]}]},{"value":1.0,"description":"tags:(小米 手机 苹果)","details":[]},{"value":1.0,"description":"brand.keyword:(Apple 小米)","details":[]}]},{"value":2.9167697,"description":"min of:","details":[{"value":2.9167697,"description":"function score, score mode [sum]","details":[{"value":2.0,"description":"function score, product of:","details":[{"value":1.0,"description":"match filter: brand.keyword:(Apple 小米)","details":[]},{"value":2.0,"description":"product of:","details":[{"value":1.0,"description":"constant score 1.0 - no function provided","details":[]},{"value":2.0,"description":"weight","details":[]}]}]},{"value":0.8375885,"description":"field value function: log1p(doc['rating'].value * factor=1.2)","details":[]},{"value":0.07918125,"description":"field value function: log1p(doc['sales'].value * factor=0.001)","details":[]}]},{"value":3.4028235E38,"description":"maxBoost","details":[]}]}]}},{"_shard":"[products][0]","_node":"j7LRBNhTRxa2B6njS7DqMw","_index":"products","_id":"2","_score":4.2878046,"_source":{"id":"2","title":"Samsung Galaxy S21","description":"三星Galaxy S21 5G智能手机，高通骁龙888处理器","price":5999.00,"category":"电子产品","brand":"Samsung","in_stock":true,"stock_count":85,"tags":["手机","三星","智能手机","5G"],"sales":350,"rating":4.6,"created_at":"2023-01-15"},"_explanation":{"value":4.2878046,"description":"function score, product of:","details":[{"value":1.0870113,"description":"sum of:","details":[{"value":0.08701137,"description":"weight(category.keyword:电子产品 in 1) [PerFieldSimilarity], result of:","details":[{"value":0.08701137,"description":"score(freq=1.0), computed as boost * idf * tf from:","details":[{"value":2.2,"description":"boost","details":[]},{"value":0.087011375,"description":"idf, computed as log(1 + (N - n + 0.5) / (n + 0.5)) from:","details":[{"value":5,"description":"n, number of documents containing term","details":[]},{"value":5,"description":"N, total number of documents with field","details":[]}]},{"value":0.45454544,"description":"tf, computed as freq / (freq + k1 * (1 - b + b * dl / avgdl)) from:","details":[{"value":1.0,"description":"freq, occurrences of term within document","details":[]},{"value":1.2,"description":"k1, term saturation parameter","details":[]},{"value":0.75,"description":"b, length normalization parameter","details":[]},{"value":1.0,"description":"dl, length of field","details":[]},{"value":1.0,"description":"avgdl, average length of field","details":[]}]}]}]},{"value":1.0,"description":"tags:(小米 手机 苹果)","details":[]}]},{"value":3.9445813,"description":"min of:","details":[{"value":3.9445813,"description":"function score, score mode [sum]","details":[{"value":3.0,"description":"function score, product of:","details":[{"value":1.0,"description":"match filter: tags:(手机)","details":[]},{"value":3.0,"description":"product of:","details":[{"value":1.0,"description":"constant score 1.0 - no function provided","details":[]},{"value":3.0,"description":"weight","details":[]}]}]},{"value":0.8142476,"description":"field value function: log1p(doc['rating'].value * factor=1.2)","details":[]},{"value":0.13033378,"description":"field value function: log1p(doc['sales'].value * factor=0.001)","details":[]}]},{"value":3.4028235E38,"description":"maxBoost","details":[]}]}]}},{"_shard":"[products][0]","_node":"j7LRBNhTRxa2B6njS7DqMw","_index":"products","_id":"4","_score":0.07681937,"_source":{"id":"4","title":"华为MateBook X Pro","description":"华为MateBook X Pro 2022款笔记本电脑","price":9999.00,"category":"电子产品","brand":"华为","in_stock":true,"stock_count":50,"tags":["笔记本","华为","电脑"],"sales":150,"rating":4.7,"created_at":"2023-02-15"},"_explanation":{"value":0.07681937,"description":"function score, product of:","details":[{"value":0.08701137,"description":"sum of:","details":[{"value":0.08701137,"description":"weight(category.keyword:电子产品 in 3) [PerFieldSimilarity], result of:","details":[{"value":0.08701137,"description":"score(freq=1.0), computed as boost * idf * tf from:","details":[{"value":2.2,"description":"boost","details":[]},{"value":0.087011375,"description":"idf, computed as log(1 + (N - n + 0.5) / (n + 0.5)) from:","details":[{"value":5,"description":"n, number of documents containing term","details":[]},{"value":5,"description":"N, total number of documents with field","details":[]}]},{"value":0.45454544,"description":"tf, computed as freq / (freq + k1 * (1 - b + b * dl / avgdl)) from:","details":[{"value":1.0,"description":"freq, occurrences of term within document","details":[]},{"value":1.2,"description":"k1, term saturation parameter","details":[]},{"value":0.75,"description":"b, length normalization parameter","details":[]},{"value":1.0,"description":"dl, length of field","details":[]},{"value":1.0,"description":"avgdl, average length of field","details":[]}]}]}]}]},{"value":0.8828659,"description":"min of:","details":[{"value":0.8828659,"description":"function score, score mode [sum]","details":[{"value":0.82216805,"description":"field value function: log1p(doc['rating'].value * factor=1.2)","details":[]},{"value":0.060697842,"description":"field value function: log1p(doc['sales'].value * factor=0.001)","details":[]}]},{"value":3.4028235E38,"description":"maxBoost","details":[]}]}]}}]}}
          
          ```

### 5. **实时库存查询**
   - **功能**：实时显示商品的库存状态。
   - **实现方式**：
     - 将库存信息同步到 Elasticsearch 中，并通过 `filter` 查询快速检索。
     - 示例查询：
       ```json
       {
         "query": {
           "bool": {
             "must": [
               {"match": {"in_stock": true}}
             ]
           }
         }
       }
       ```
     - **初始化脚本**：
       - 使用第一个场景中已创建的索引和数据，无需重新创建。
       - 验证实时库存查询功能：
         ```bash
         # 查询有库存的商品
        curl --insecure -u elastic:mypassword -X GET "https://localhost:9200/products/_search" -H "Content-Type: application/json" -d'
         {
           "query": {
             "bool": {
               "must": [
                 {"match": {"in_stock": true}}
               ]
             }
           }
         }'
         ```
       
       - 更新库存状态示例：
         ```bash
         # 更新商品库存状态
        curl --insecure -u elastic:mypassword -X POST "https://localhost:9200/products/_update/2" -H "Content-Type: application/json" -d'
         {
           "doc": {
             "in_stock": false,
             "stock_count": 0
           }
         }'
         
         # 查询缺货商品
        curl --insecure -u elastic:mypassword -X GET "https://localhost:9200/products/_search" -H "Content-Type: application/json" -d'
         {
           "query": {
             "bool": {
               "must": [
                 {"match": {"in_stock": false}}
               ]
             }
           }
         }'
         ```
       
       - 库存阈值查询示例：
         ```bash
         # 查询库存低于指定阈值的商品
        curl --insecure -u elastic:mypassword -X GET "https://localhost:9200/products/_search" -H "Content-Type: application/json" -d'
         {
           "query": {
             "bool": {
               "must": [
                 {"match": {"in_stock": true}}
               ],
               "filter": [
                 {"range": {"stock_count": {"lt": 50}}}
               ]
             }
           }
         }'
         ```

### 6. **数据分析与报表**
   - **功能**：分析商品销售趋势、用户行为等。
   - **实现方式**：
     - 使用 Elasticsearch 的聚合功能（`aggregations`）生成统计报表。
     - 示例查询：
       ```json
       {
         "size": 0,
         "aggs": {
           "sales_by_category": {
             "terms": {
               "field": "category.keyword"
             },
             "aggs": {
               "total_sales": {
                 "sum": {
                   "field": "sales"
                 }
               }
             }
           }
         }
       }
       ```
     - **初始化脚本**：
       - 使用第一个场景中已创建的索引和数据，无需重新创建。
       - 验证数据分析与报表功能：
         ```bash
         # 按类别统计销售总量
        curl --insecure -u elastic:mypassword -X GET "https://localhost:9200/products/_search" -H "Content-Type: application/json" -d'
         {
           "size": 0,
           "aggs": {
             "sales_by_category": {
               "terms": {
                 "field": "category.keyword"
               },
               "aggs": {
                 "total_sales": {
                   "sum": {
                     "field": "sales"
                   }
                 }
               }
             }
           }
         }'
         ```
       
       - 品牌销售分析示例：
         ```bash
         # 按品牌统计平均评分和销量
        curl --insecure -u elastic:mypassword -X GET "https://localhost:9200/products/_search" -H "Content-Type: application/json" -d'
         {
           "size": 0,
           "aggs": {
             "brands": {
               "terms": {
                 "field": "brand.keyword",
                 "size": 10
               },
               "aggs": {
                 "avg_rating": {
                   "avg": {
                     "field": "rating"
                   }
                 },
                 "total_sales": {
                   "sum": {
                     "field": "sales"
                   }
                 }
               }
             }
           }
         }'
         ```
       
       - 价格区间分析示例：
         ```bash
         # 按价格区间统计商品数量
        curl --insecure -u elastic:mypassword -X GET "https://localhost:9200/products/_search" -H "Content-Type: application/json" -d'
         {
           "size": 0,
           "aggs": {
             "price_ranges": {
               "range": {
                 "field": "price",
                 "ranges": [
                   { "to": 5000 },
                   { "from": 5000, "to": 10000 },
                   { "from": 10000 }
                 ]
               }
             }
           }
         }'
         ```

### 7. **拼写检查与建议**
   - **功能**：当用户输入错误的关键词时，提供拼写建议。
   - **实现方式**：
     - 使用 Elasticsearch 的 `suggest` API。
     - 示例查询：
       ```json
       {
         "suggest": {
           "text": "iphon",
           "suggestion": {
             "term": {
               "field": "title"
             }
           }
         }
       }
       ```
     - **初始化脚本**：
       - 使用第一个场景中已创建的索引和数据，但需要更新索引设置以支持拼写检查：
         ```bash
         # 关闭索引以更新设置
        curl --insecure -u elastic:mypassword -X POST "https://localhost:9200/products/_close"
         
         # 更新索引设置，添加拼写检查所需的配置
        curl --insecure -u elastic:mypassword -u elastic:mypassword -X PUT "https://localhost:9200/products/_settings" -H "Content-Type: application/json" -d'
         {
           "analysis": {
             "analyzer": {
               "suggest_analyzer": {
                 "type": "standard",
                 "max_token_length": 5
               }
             }
           }
         }'
         
         # 重新打开索引
        curl --insecure -u elastic:mypassword -X POST "https://localhost:9200/products/_open"
         ```
       
       - 验证拼写检查功能：
         ```bash
         # 执行拼写检查查询
        curl --insecure -u elastic:mypassword -X GET "https://localhost:9200/products/_search" -H "Content-Type: application/json" -d'
         {
           "suggest": {
             "text": "iphon",
             "product_suggestions": {
               "term": {
                 "field": "title"
               }
             }
           }
         }'
         ```
       
       - 使用更复杂的拼写检查示例：
         ```bash
         # 使用phrase suggester进行更智能的拼写检查
        curl --insecure -u elastic:mypassword -X GET "https://localhost:9200/products/_search" -H "Content-Type: application/json" -d'
         {
           "suggest": {
             "text": "ipone pro",
             "phrase_suggestion": {
               "phrase": {
                 "field": "title",
                 "size": 3,
                 "gram_size": 2,
                 "direct_generator": [{
                   "field": "title",
                   "suggest_mode": "always"
                 }],
                 "highlight": {
                   "pre_tag": "<em>",
                   "post_tag": "</em>"
                 }
               }
             }
           }
         }'
         ```

### 8. **多语言支持**
   - **功能**：支持多语言的商品搜索。
   - **实现方式**：
     - 配置 Elasticsearch 的分词器（analyzer），支持不同语言的文本处理。
     - 示例配置：
       ```json
       {
         "settings": {
           "analysis": {
             "analyzer": {
               "zh_analyzer": {
                 "type": "custom",
                 "tokenizer": "ik_max_word"
               }
             }
           }
         },
         "mappings": {
           "properties": {
             "title": {
               "type": "text",
               "analyzer": "zh_analyzer"
             }
           }
         }
       }
       ```
     - **初始化脚本**：
       - 创建多语言支持的索引：
         ```bash
         # 创建多语言商品索引
        curl --insecure -u elastic:mypassword -u elastic:mypassword -X PUT "https://localhost:9200/multilingual_products" -H "Content-Type: application/json" -d'
         {
           "settings": {
             "analysis": {
               "analyzer": {
                 "english_analyzer": {
                   "type": "english"
                 },
                 "chinese_analyzer": {
                   "type": "standard",
                   "tokenizer": "standard",
                   "filter": ["lowercase"]
                 }
               }
             }
           },
           "mappings": {
             "properties": {
               "id": { "type": "keyword" },
               "title_en": { 
                 "type": "text", 
                 "analyzer": "english_analyzer",
                 "fields": {
                   "keyword": { "type": "keyword" }
                 }
               },
               "title_zh": { 
                 "type": "text", 
                 "analyzer": "chinese_analyzer",
                 "fields": {
                   "keyword": { "type": "keyword" }
                 }
               },
               "description_en": { 
                 "type": "text", 
                 "analyzer": "english_analyzer" 
               },
               "description_zh": { 
                 "type": "text", 
                 "analyzer": "chinese_analyzer" 
               },
               "price": { "type": "float" },
               "category": { 
                 "type": "keyword"
               },
               "tags_en": { "type": "keyword" },
               "tags_zh": { "type": "keyword" }
             }
           }
         }'
         ```
       
       - 添加多语言示例数据：
         ```bash
         # 添加多语言商品数据
        curl --insecure -u elastic:mypassword -X POST "https://localhost:9200/multilingual_products/_bulk" -H "Content-Type: application/json" -d'
         {"index":{"_id":"1"}}
         {"id":"1","title_en":"iPhone 13 Pro","title_zh":"苹果 iPhone 13 Pro","description_en":"Apple iPhone 13 Pro smartphone with A15 Bionic chip","description_zh":"苹果 iPhone 13 Pro 智能手机，搭载A15仿生芯片","price":7999.00,"category":"electronics","tags_en":["phone","apple","smartphone"],"tags_zh":["手机","苹果","智能手机"]}
         {"index":{"_id":"2"}}
         {"id":"2","title_en":"Samsung Galaxy S21","title_zh":"三星 Galaxy S21","description_en":"Samsung Galaxy S21 5G smartphone with Snapdragon 888","description_zh":"三星Galaxy S21 5G智能手机，高通骁龙888处理器","price":5999.00,"category":"electronics","tags_en":["phone","samsung","smartphone","5G"],"tags_zh":["手机","三星","智能手机","5G"]}
         {"index":{"_id":"3"}}
         {"id":"3","title_en":"Xiaomi 12 Pro","title_zh":"小米12 Pro","description_en":"Xiaomi 12 Pro 5G smartphone with Snapdragon 8 Gen 1","description_zh":"小米12 Pro 5G智能手机，搭载骁龙8 Gen 1处理器","price":4999.00,"category":"electronics","tags_en":["phone","xiaomi","smartphone","5G"],"tags_zh":["手机","小米","智能手机","5G"]}
         '
         ```
       
       - 验证英文搜索功能：
         ```bash
         # 英文搜索查询
        curl --insecure -u elastic:mypassword -X GET "https://localhost:9200/multilingual_products/_search" -H "Content-Type: application/json" -d'
         {
           "query": {
             "multi_match": {
               "query": "smartphone",
               "fields": ["title_en", "description_en"]
             }
           }
         }'
         ```
       
       - 验证中文搜索功能：
         ```bash
         # 中文搜索查询
        curl --insecure -u elastic:mypassword -X GET "https://localhost:9200/multilingual_products/_search" -H "Content-Type: application/json" -d'
         {
           "query": {
             "multi_match": {
               "query": "智能手机",
               "fields": ["title_zh", "description_zh"]
             }
           }
         }'
         ```

### 9. **标签云**
   - **功能**：展示热门搜索词或商品标签。
   - **实现方式**：
     - 使用 Elasticsearch 的词频统计功能。
     - 示例查询：
       ```json
       {
         "size": 0,
         "aggs": {
           "top_tags": {
             "terms": {
               "field": "tags.keyword",
               "size": 10
             }
           }
         }
       }
       ```
     - **初始化脚本**：
       - 使用第一个场景中已创建的索引和数据，无需重新创建。
       - 验证标签云功能：
         ```bash
         # 获取热门标签
        curl --insecure -u elastic:mypassword -X GET "https://localhost:9200/products/_search" -H "Content-Type: application/json" -d'
         {
           "size": 0,
           "aggs": {
             "top_tags": {
               "terms": {
                 "field": "tags",
                 "size": 10
               }
             }
           }
         }'

         {
            "took": 30,
            "timed_out": false,
            "_shards": {
                "total": 1,
                "successful": 1,
                "skipped": 0,
                "failed": 0
            },
            "hits": {
                "total": {
                    "value": 5,
                    "relation": "eq"
                },
                "max_score": null,
                "hits": []
            },
            "aggregations": {
                "top_tags": {
                    "doc_count_error_upper_bound": 0,
                    "sum_other_doc_count": 0,
                    "buckets": [{
                        "key": "手机",
                        "doc_count": 3
                    }, {
                        "key": "智能手机",
                        "doc_count": 3
                    }, {
                        "key": "5G",
                        "doc_count": 2
                    }, {
                        "key": "电脑",
                        "doc_count": 2
                    }, {
                        "key": "笔记本",
                        "doc_count": 2
                    }, {
                        "key": "苹果",
                        "doc_count": 2
                    }, {
                        "key": "三星",
                        "doc_count": 1
                    }, {
                        "key": "华为",
                        "doc_count": 1
                    }, {
                        "key": "小米",
                        "doc_count": 1
                    }]
                }
            }
        }
         ```
       
       - 创建搜索历史索引：
         ```bash
         # 创建搜索历史索引
        curl --insecure -u elastic:mypassword -u elastic:mypassword -X PUT "https://localhost:9200/search_history" -H "Content-Type: application/json" -d'
         {
           "mappings": {
             "properties": {
               "user_id": { "type": "keyword" },
               "search_term": { "type": "keyword" },
               "search_time": { "type": "date" },
               "result_count": { "type": "integer" },
               "clicked_products": { "type": "keyword" }
             }
           }
         }'
         ```
       
       - 添加搜索历史数据：
         ```bash
         # 添加搜索历史数据
        curl --insecure -u elastic:mypassword -X POST "https://localhost:9200/search_history/_bulk" -H "Content-Type: application/json" -d'
         {"index":{}}
         {"user_id":"user_001","search_term":"手机","search_time":"2023-05-01T10:00:00","result_count":3,"clicked_products":["1"]}
         {"index":{}}
         {"user_id":"user_002","search_term":"手机","search_time":"2023-05-01T10:05:00","result_count":3,"clicked_products":["2"]}
         {"index":{}}
         {"user_id":"user_003","search_term":"笔记本","search_time":"2023-05-01T10:10:00","result_count":2,"clicked_products":["4"]}
         {"index":{}}
         {"user_id":"user_004","search_term":"手机","search_time":"2023-05-01T10:15:00","result_count":3,"clicked_products":["3"]}
         {"index":{}}
         {"user_id":"user_005","search_term":"笔记本","search_time":"2023-05-01T10:20:00","result_count":2,"clicked_products":["5"]}
         '
         ```
       
       - 获取热门搜索词：
         ```bash
         # 获取热门搜索词
        curl --insecure -u elastic:mypassword -X GET "https://localhost:9200/search_history/_search" -H "Content-Type: application/json" -d'
         {
           "size": 0,
           "aggs": {
             "popular_searches": {
               "terms": {
                 "field": "search_term",
                 "size": 5
               }
             }
           }
         }'
         ```

### 10. **个性化搜索**
   - **功能**：根据用户的偏好调整搜索结果。
   - **实现方式**：
     - 结合 Elasticsearch 的 `function_score` 查询，为不同的条件设置权重。
     - 示例查询：
       ```json
       {
         "query": {
           "function_score": {
             "query": {
               "match": {"title": "手机"}
             },
             "functions": [
               {
                 "filter": {"term": {"user_preferences": "科技"}},
                 "weight": 2
               }
             ],
             "score_mode": "multiply"
           }
         }
       }
       ```
     - **初始化脚本**：
       - 创建用户偏好索引：
         ```bash
         # 创建用户偏好索引
        curl --insecure -u elastic:mypassword -u elastic:mypassword -X PUT "https://localhost:9200/user_preferences" -H "Content-Type: application/json" -d'
         {
           "mappings": {
             "properties": {
               "user_id": { "type": "keyword" },
               "preferred_categories": { "type": "keyword" },
               "preferred_brands": { "type": "keyword" },
               "price_sensitivity": { "type": "keyword" },
               "preferred_tags": { "type": "keyword" },
               "last_updated": { "type": "date" }
             }
           }
         }'
         ```
       
       - 添加用户偏好数据：
         ```bash
         # 添加用户偏好数据
        curl --insecure -u elastic:mypassword -X POST "https://localhost:9200/user_preferences/_bulk" -H "Content-Type: application/json" -d'
         {"index":{"_id":"user_001"}}
         {"user_id":"user_001","preferred_categories":["电子产品"],"preferred_brands":["Apple","小米"],"price_sensitivity":"high","preferred_tags":["手机","智能手机"],"last_updated":"2023-05-01"}
         {"index":{"_id":"user_002"}}
         {"user_id":"user_002","preferred_categories":["电子产品"],"preferred_brands":["Samsung","华为"],"price_sensitivity":"medium","preferred_tags":["笔记本","电脑"],"last_updated":"2023-05-01"}
         '
         ```
       
       - 基于用户偏好的个性化搜索示例：
         ```bash
         # 基于用户偏好的个性化搜索
        curl --insecure -u elastic:mypassword -X GET "https://localhost:9200/products/_search" -H "Content-Type: application/json" -d'
         {
           "query": {
             "function_score": {
               "query": {
                 "match": {"category": "电子产品"}
               },
               "functions": [
                 {
                   "filter": {"terms": {"brand.keyword": ["Apple", "小米"]}},
                   "weight": 3
                 },
                 {
                   "filter": {"terms": {"tags": ["手机", "智能手机"]}},
                   "weight": 2
                 }
               ],
               "score_mode": "sum",
               "boost_mode": "multiply"
             }
           }
         }'
         ```
       
       - 结合用户搜索词和偏好的查询示例：
         ```bash
         # 结合用户搜索词和偏好的查询
        curl --insecure -u elastic:mypassword -X GET "https://localhost:9200/products/_search" -H "Content-Type: application/json" -d'
         {
           "query": {
             "function_score": {
               "query": {
                 "multi_match": {
                   "query": "手机",
                   "fields": ["title", "description"]
                 }
               },
               "functions": [
                 {
                   "filter": {"terms": {"brand.keyword": ["Apple", "小米"]}},
                   "weight": 2.5
                 },
                 {
                   "filter": {"range": {"rating": {"gte": 4.5}}},
                   "weight": 1.5
                 }
               ],
               "score_mode": "sum",
               "boost_mode": "multiply"
             }
           },
           "sort": [
             "_score"
           ]
         }'
         ```

### 总结
Elasticsearch 在电商系统中的应用非常广泛，涵盖了从商品搜索、过滤、排序到数据分析等多个方面。通过灵活使用其查询和聚合功能，可以显著提升用户体验和运营效率。