# ELK Stack 日志收集方案详解

## 1. ELK Stack 基本原理

### 1.1 ELK Stack 概述

ELK Stack 是由 Elasticsearch、Logstash 和 Kibana 三个开源项目组成的日志管理平台。随着 Beats 的加入，现在也被称为 Elastic Stack。

- **Elasticsearch**：分布式搜索和分析引擎，基于 Lucene 构建
- **Logstash**：服务器端数据处理管道，能够同时从多个源接收数据
- **Kibana**：数据可视化和探索工具，用于 Elasticsearch 数据的可视化
- **Beats**：轻量级数据采集器，作为代理安装在服务器上

### 1.2 工作原理

#### 1.2.1 数据流向

1. **数据收集**：Beats 或 Logstash 从各种源（文件、系统、网络等）收集日志数据
2. **数据处理**：Logstash 对收集的数据进行过滤、转换和丰富
3. **数据存储**：处理后的数据被发送到 Elasticsearch 进行索引和存储
4. **数据可视化**：Kibana 连接到 Elasticsearch，提供搜索和可视化界面

#### 1.2.2 各组件详细工作原理

**Elasticsearch 工作原理**：
- 基于分布式的 RESTful 搜索和分析引擎
- 使用倒排索引结构，支持快速全文搜索
- 数据以 JSON 文档形式存储，按索引组织
- 支持水平扩展，通过分片机制实现
- 提供高可用性，通过副本机制实现

**Logstash 工作原理**：
- 采用管道架构：输入 → 过滤器 → 输出
- 输入插件从各种源收集数据
- 过滤器插件处理和转换数据
- 输出插件将数据发送到目标存储
- 支持多种数据格式和协议

**Kibana 工作原理**：
- 通过 REST API 与 Elasticsearch 交互
- 提供基于浏览器的界面，用于搜索、查看和交互
- 支持多种可视化类型：图表、表格、地图等
- 提供仪表板功能，组合多个可视化

**Beats 工作原理**：
- 轻量级数据采集器，资源占用少
- 专注于单一数据收集任务
- 直接发送数据到 Elasticsearch 或通过 Logstash 处理
- 常见类型：Filebeat（日志文件）、Metricbeat（指标）、Packetbeat（网络数据）等

## 2. ELK Stack 架构设计

### 2.1 基础架构

#### 2.1.1 单节点架构

适用于开发环境或小型应用场景：

```
微服务 → Filebeat → Logstash → Elasticsearch → Kibana
```

#### 2.1.2 分布式架构

适用于生产环境或大型应用场景：

```
微服务集群 → Filebeat集群 → Logstash集群 → Elasticsearch集群 → Kibana集群
```

### 2.2 高级架构模式

#### 2.2.1 直接架构

对于简单场景，可以跳过 Logstash，直接从 Beats 到 Elasticsearch：

```
微服务 → Filebeat → Elasticsearch → Kibana
```

#### 2.2.2 缓冲架构

引入消息队列作为缓冲，提高系统稳定性：

```
微服务 → Filebeat → Kafka → Logstash → Elasticsearch → Kibana
```

#### 2.2.3 多集群架构

用于跨数据中心或地理位置分散的场景：

```
数据中心A：微服务 → Filebeat → Logstash → Elasticsearch A
数据中心B：微服务 → Filebeat → Logstash → Elasticsearch B
                                              ↓
                                            Kibana
```

### 2.3 Elasticsearch 集群架构

#### 2.3.1 节点类型

- **主节点（Master Node）**：负责集群管理和元数据操作
- **数据节点（Data Node）**：存储数据和执行数据相关操作
- **客户端节点（Client Node）**：处理请求路由和负载均衡
- **摄取节点（Ingest Node）**：预处理文档，执行转换

#### 2.3.2 索引设计

- **分片（Shard）**：将索引分成多个部分，分布在不同节点
- **副本（Replica）**：分片的复制，提供高可用和读取性能
- **索引生命周期管理（ILM）**：自动管理索引，包括滚动、收缩和删除

### 2.4 Logstash 架构

#### 2.4.1 管道配置

```
input {
  beats {
    port => 5044
  }
}

filter {
  grok {
    match => { "message" => "%{COMBINEDAPACHELOG}" }
  }
  date {
    match => [ "timestamp" , "dd/MMM/yyyy:HH:mm:ss Z" ]
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "logstash-%{+YYYY.MM.dd}"
  }
}
```

#### 2.4.2 性能优化

- **多管道配置**：根据数据类型分离管道
- **持久化队列**：防止数据丢失
- **批处理**：提高吞吐量

## 3. ELK Stack 部署指南

### 3.1 Docker Compose 部署

#### 3.1.1 docker-compose.yml

```yaml
version: '3'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.14.0
    container_name: elasticsearch
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ulimits:
      memlock:
        soft: -1
        hard: -1
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data
    ports:
      - 9200:9200
    networks:
      - elk

  logstash:
    image: docker.elastic.co/logstash/logstash:7.14.0
    container_name: logstash
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline
      - ./logstash/config/logstash.yml:/usr/share/logstash/config/logstash.yml
    ports:
      - 5044:5044
      - 9600:9600
    environment:
      LS_JAVA_OPTS: "-Xmx256m -Xms256m"
    networks:
      - elk
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:7.14.0
    container_name: kibana
    ports:
      - 5601:5601
    environment:
      ELASTICSEARCH_URL: http://elasticsearch:9200
      ELASTICSEARCH_HOSTS: http://elasticsearch:9200
    networks:
      - elk
    depends_on:
      - elasticsearch

networks:
  elk:
    driver: bridge

volumes:
  elasticsearch-data:
```

#### 3.1.2 Logstash 配置

**logstash.yml**：

```yaml
http.host: "0.0.0.0"
xpack.monitoring.elasticsearch.hosts: [ "http://elasticsearch:9200" ]
```

**pipeline.conf**：

```
input {
  beats {
    port => 5044
  }
}

filter {
  if [fileset][module] == "nginx" {
    if [fileset][name] == "access" {
      grok {
        match => { "message" => "%{COMBINEDAPACHELOG}" }
      }
    }
    date {
      match => [ "timestamp" , "dd/MMM/yyyy:HH:mm:ss Z" ]
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    manage_template => false
    index => "%{[@metadata][beat]}-%{[@metadata][version]}-%{+YYYY.MM.dd}"
  }
}
```

#### 3.1.3 Filebeat 配置

**filebeat.yml**：

```yaml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/nginx/access.log
  fields:
    service: nginx
    type: access
  fields_under_root: true

output.logstash:
  hosts: ["logstash:5044"]
```

### 3.2 Kubernetes 部署

#### 3.2.1 使用 Helm 部署

```bash
# 添加 Elastic Helm 仓库
helm repo add elastic https://helm.elastic.co

# 部署 Elasticsearch
helm install elasticsearch elastic/elasticsearch --version 7.14.0 \
  --set replicas=3 \
  --set minimumMasterNodes=2

# 部署 Kibana
helm install kibana elastic/kibana --version 7.14.0 \
  --set elasticsearchHosts=http://elasticsearch-master:9200

# 部署 Logstash
helm install logstash elastic/logstash --version 7.14.0 \
  --set logstashPipeline.logstash.conf="$(cat pipeline.conf)"

# 部署 Filebeat
helm install filebeat elastic/filebeat --version 7.14.0
```

#### 3.2.2 使用 Kubernetes 清单文件

**elasticsearch-statefulset.yaml**：

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: elasticsearch
spec:
  serviceName: elasticsearch
  replicas: 3
  selector:
    matchLabels:
      app: elasticsearch
  template:
    metadata:
      labels:
        app: elasticsearch
    spec:
      containers:
      - name: elasticsearch
        image: docker.elastic.co/elasticsearch/elasticsearch:7.14.0
        env:
        - name: cluster.name
          value: elasticsearch-cluster
        - name: node.name
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: discovery.seed_hosts
          value: "elasticsearch-0.elasticsearch,elasticsearch-1.elasticsearch,elasticsearch-2.elasticsearch"
        - name: cluster.initial_master_nodes
          value: "elasticsearch-0,elasticsearch-1,elasticsearch-2"
        - name: ES_JAVA_OPTS
          value: "-Xms512m -Xmx512m"
        ports:
        - containerPort: 9200
          name: http
        - containerPort: 9300
          name: transport
        volumeMounts:
        - name: data
          mountPath: /usr/share/elasticsearch/data
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 10Gi
```

### 3.3 部署后配置

#### 3.3.1 Elasticsearch 索引模板

```json
PUT _template/logs
{
  "index_patterns": ["logstash-*"],
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "index.lifecycle.name": "logs-policy",
    "index.lifecycle.rollover_alias": "logs"
  },
  "mappings": {
    "properties": {
      "@timestamp": { "type": "date" },
      "message": { "type": "text" },
      "service": { "type": "keyword" },
      "level": { "type": "keyword" }
    }
  }
}
```

#### 3.3.2 索引生命周期管理

```json
PUT _ilm/policy/logs-policy
{
  "policy": {
    "phases": {
      "hot": {
        "actions": {
          "rollover": {
            "max_age": "1d",
            "max_size": "50gb"
          },
          "set_priority": {
            "priority": 100
          }
        }
      },
      "warm": {
        "min_age": "2d",
        "actions": {
          "shrink": {
            "number_of_shards": 1
          },
          "forcemerge": {
            "max_num_segments": 1
          },
          "set_priority": {
            "priority": 50
          }
        }
      },
      "cold": {
        "min_age": "7d",
        "actions": {
          "set_priority": {
            "priority": 0
          }
        }
      },
      "delete": {
        "min_age": "30d",
        "actions": {
          "delete": {}
        }
      }
    }
  }
}
```

## 4. ELK Stack 实际应用案例

### 4.1 微服务日志收集案例

#### 4.1.1 场景描述

一个电子商务平台，包含以下微服务：
- 用户服务（Java Spring Boot）
- 商品服务（Node.js）
- 订单服务（Python Flask）
- API 网关（Nginx）

#### 4.1.2 日志收集架构

```
微服务 → Filebeat → Kafka → Logstash → Elasticsearch → Kibana
```

#### 4.1.3 实现步骤

1. **配置各微服务的日志格式**：
   - 统一使用 JSON 格式
   - 包含关键字段：timestamp, service_name, trace_id, span_id, level, message

2. **Filebeat 配置**：

```yaml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/user-service/*.log
  json.keys_under_root: true
  fields:
    service: user-service
  fields_under_root: true

- type: log
  enabled: true
  paths:
    - /var/log/product-service/*.log
  json.keys_under_root: true
  fields:
    service: product-service
  fields_under_root: true

- type: log
  enabled: true
  paths:
    - /var/log/order-service/*.log
  json.keys_under_root: true
  fields:
    service: order-service
  fields_under_root: true

- type: log
  enabled: true
  paths:
    - /var/log/nginx/access.log
  fields:
    service: api-gateway
    type: access
  fields_under_root: true

output.kafka:
  hosts: ["kafka:9092"]
  topic: "logs"
  partition.round_robin:
    reachable_only: false
  required_acks: 1
  compression: gzip
  max_message_bytes: 1000000
```

3. **Logstash 配置**：

```
input {
  kafka {
    bootstrap_servers => "kafka:9092"
    topics => ["logs"]
    consumer_threads => 4
    codec => json
  }
}

filter {
  if [service] == "api-gateway" and [type] == "access" {
    grok {
      match => { "message" => "%{COMBINEDAPACHELOG}" }
    }
    date {
      match => [ "timestamp" , "dd/MMM/yyyy:HH:mm:ss Z" ]
    }
  }
  
  # 添加地理位置信息
  if [clientip] {
    geoip {
      source => "clientip"
      target => "geoip"
    }
  }
  
  # 丰富日志信息
  mutate {
    add_field => { "environment" => "production" }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "%{[service]}-%{+YYYY.MM.dd}"
    manage_template => false
  }
}
```

4. **Kibana 仪表板**：
   - 服务健康状态仪表板
   - 错误日志分析仪表板
   - 用户行为分析仪表板
   - API 性能监控仪表板

### 4.2 容器化环境日志收集案例

#### 4.2.1 场景描述

基于 Kubernetes 的微服务平台，需要收集：
- 容器日志
- 节点系统日志
- Kubernetes 事件日志

#### 4.2.2 日志收集架构

```
Kubernetes Pod → Filebeat DaemonSet → Elasticsearch → Kibana
```

#### 4.2.3 实现步骤

1. **部署 Filebeat DaemonSet**：

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: filebeat
  namespace: logging
spec:
  selector:
    matchLabels:
      app: filebeat
  template:
    metadata:
      labels:
        app: filebeat
    spec:
      serviceAccountName: filebeat
      containers:
      - name: filebeat
        image: docker.elastic.co/beats/filebeat:7.14.0
        args: [
          "-c", "/etc/filebeat.yml",
          "-e",
        ]
        volumeMounts:
        - name: config
          mountPath: /etc/filebeat.yml
          subPath: filebeat.yml
        - name: data
          mountPath: /usr/share/filebeat/data
        - name: varlibdockercontainers
          mountPath: /var/lib/docker/containers
          readOnly: true
        - name: varlog
          mountPath: /var/log
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: filebeat-config
      - name: data
        hostPath:
          path: /var/lib/filebeat-data
      - name: varlibdockercontainers
        hostPath:
          path: /var/lib/docker/containers
      - name: varlog
        hostPath:
          path: /var/log
```

2. **Filebeat ConfigMap**：

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: filebeat-config
  namespace: logging
data:
  filebeat.yml: |
    filebeat.inputs:
    - type: container
      paths:
        - /var/lib/docker/containers/*/*.log
      processors:
        - add_kubernetes_metadata:
            host: ${NODE_NAME}
            matchers:
            - logs_path:
                logs_path: "/var/lib/docker/containers/{data.kubernetes.container.id}/*.log"
    
    filebeat.modules:
    - module: system
      syslog:
        enabled: true
      auth:
        enabled: true
    
    processors:
      - add_cloud_metadata: ~
      - add_host_metadata: ~
    
    output.elasticsearch:
      hosts: ['${ELASTICSEARCH_HOST:elasticsearch}:${ELASTICSEARCH_PORT:9200}']
      username: ${ELASTICSEARCH_USERNAME}
      password: ${ELASTICSEARCH_PASSWORD}
      index: "filebeat-%{[agent.version]}-%{+yyyy.MM.dd}"
```

3. **配置 Elasticsearch 和 Kibana**：
   - 部署 Elasticsearch StatefulSet
   - 部署 Kibana Deployment
   - 配置索引模式和仪表板

4. **创建 Kibana 仪表板**：
   - Kubernetes 节点监控仪表板
   - 容器日志分析仪表板
   - 系统日志监控仪表板

### 4.3 性能优化案例

#### 4.3.1 场景描述

大型金融机构，日志量超过 1TB/天，需要优化 ELK Stack 性能。

#### 4.3.2 优化策略

1. **Elasticsearch 集群优化**：
   - 增加数据节点数量（15 个节点）
   - 配置专用主节点和协调节点
   - 优化 JVM 堆大小（32GB）
   - 使用 SSD 存储
   - 优化分片策略（每个索引 5 个主分片，1 个副本）

2. **Logstash 优化**：
   - 部署多个 Logstash 实例（10 个）
   - 使用持久化队列
   - 优化批处理大小（1000 条/批）
   - 增加工作线程数（8 个）

3. **引入缓冲层**：
   - 部署 Kafka 集群（10 个 broker）
   - 配置适当的主题分区数（50 个）

4. **数据生命周期管理**：
   - 热数据保留 3 天
   - 温数据保留 15 天
   - 冷数据保留 60 天
   - 超过 60 天的数据归档到对象存储

#### 4.3.3 实施结果

- 日志处理延迟从 15 分钟降至 30 秒
- 查询响应时间从 10 秒降至 2 秒
- 存储成本降低 40%
- 系统稳定性显著提高，无数据丢失

## 5. 常见问题与解决方案

### 5.1 性能问题

#### 5.1.1 Elasticsearch 集群性能下降

**症状**：
- 查询响应时间增加
- 索引速度变慢
- 集群状态变为黄色或红色

**解决方案**：
- 检查 JVM 堆使用情况，确保不超过可用内存的 50%
- 优化分片数量，避免过多小分片
- 使用 Force Merge API 合并分片
- 使用 Index Lifecycle Management 管理索引
- 检查并优化查询语句

#### 5.1.2 Logstash 处理延迟

**症状**：
- 日志处理延迟增加
- Logstash 队列积压

**解决方案**：
- 增加 Logstash 实例数量
- 优化过滤器配置，减少复杂处理
- 使用持久化队列
- 调整批处理大小和超时设置
- 引入 Kafka 作为缓冲层

### 5.2 数据问题

#### 5.2.1 日志数据丢失

**症状**：
- 部分日志未出现在 Elasticsearch 中
- Logstash 报告数据丢失警告

**解决方案**：
- 配置 Filebeat 保证传输可靠性
- 使用 Logstash 持久化队列
- 配置 Kafka 作为缓冲层
- 增加 Elasticsearch 副本数量
- 实施监控和告警机制

#### 5.2.2 索引损坏

**症状**：
- 查询特定索引返回错误
- Elasticsearch 报告索引损坏

**解决方案**：
- 使用 _cat/indices API 检查索引状态
- 关闭并重新打开索引
- 从快照恢复索引
- 如果无法恢复，创建新索引并重新索引数据

### 5.3 部署问题

#### 5.3.1 Docker 环境内存问题

**症状**：
- Elasticsearch 容器频繁重启
- 日志中出现内存相关错误

**解决方案**：
- 调整 Docker 容器内存限制
- 配置适当的 JVM 堆大小
- 使用 `ES_JAVA_OPTS="-Xms512m -Xmx512m"` 设置内存
- 确保系统有足够的虚拟内存

#### 5.3.2 Kubernetes 部署问题

**症状**：
- Pod 无法调度或频繁重启
- 持久卷挂载失败

**解决方案**：
- 使用 StatefulSet 部署 Elasticsearch
- 配置适当的资源请求和限制
- 使用 PersistentVolumeClaim 确保数据持久性
- 配置 Pod 反亲和性，避免单点故障

## 6. 总结与最佳实践

### 6.1 ELK Stack 优势

- **强大的搜索能力**：基于 Elasticsearch 的全文搜索
- **灵活的数据处理**：Logstash 支持多种输入、过滤和输出
- **丰富的可视化**：Kibana 提供多种可视化和仪表板
- **可扩展性**：支持水平扩展，适应不同规模需求
- **活跃的社区**：持续更新和改进

### 6.2 最佳实践

#### 6.2.1 架构设计

- 根据日志量选择合适的架构模式
- 考虑引入消息队列作为缓冲层
- 规划合理的集群规模和节点类型
- 实施高可用性设计

#### 6.2.2 性能优化

- 优化 Elasticsearch 索引设计
- 配置合适的 JVM 堆大小
- 使用 SSD 存储提高性能
- 实施索引生命周期管理
- 优化 Logstash 批处理和并行度

#### 6.2.3 运维管理

- 实施监控和告警机制
- 定期备份 Elasticsearch 数据
- 制定容量规划和扩展策略
- 保持版本更新，应用安全补