# ELK Stack 日志收集方案详解

## 1. ELK Stack 基本原理

### 1.1 ELK Stack 概述

ELK Stack 是由 Elasticsearch、Logstash 和 Kibana 三个开源项目组成的日志管理平台。随着 Beats 的加入，现在也被称为 Elastic Stack。

- **Elasticsearch**：分布式搜索和分析引擎，基于 Lucene 构建，提供实时搜索和分析功能
- **Logstash**：服务器端数据处理管道，能够同时从多个源接收数据，进行转换，然后发送到存储系统
- **Kibana**：数据可视化和探索工具，用于 Elasticsearch 数据的可视化和管理
- **Beats**：轻量级数据采集器，作为代理安装在服务器上，用于收集各种类型的数据

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
- 使用 Lucene 作为核心搜索库
- 支持实时分析和复杂查询

**Logstash 工作原理**：
- 采用管道架构：输入 → 过滤器 → 输出
- 输入插件从各种源收集数据
- 过滤器插件处理和转换数据
- 输出插件将数据发送到目标存储
- 支持多种数据格式和协议
- 提供丰富的插件生态系统
- 支持动态配置和重载

**Kibana 工作原理**：
- 通过 REST API 与 Elasticsearch 交互
- 提供基于浏览器的界面，用于搜索、查看和交互
- 支持多种可视化类型：图表、表格、地图等
- 提供仪表板功能，组合多个可视化
- 支持高级分析功能，如机器学习和异常检测
- 提供安全和用户管理功能

**Beats 工作原理**：
- 轻量级数据采集器，资源占用少
- 专注于单一数据收集任务
- 直接发送数据到 Elasticsearch 或通过 Logstash 处理
- 常见类型：
  - **Filebeat**：收集日志文件
  - **Metricbeat**：收集系统和服务指标
  - **Packetbeat**：收集网络数据
  - **Winlogbeat**：收集 Windows 事件日志
  - **Auditbeat**：收集审计数据
  - **Heartbeat**：监控服务可用性

### 1.3 与其他日志系统的比较

| 特性 | ELK Stack | Fluentd + ES + Kibana | Loki + Promtail + Grafana |
|------|-----------|-----------------------|---------------------------|
| 全文搜索能力 | 强大 | 强大 | 有限 |
| 资源消耗 | 较高 | 中等 | 低 |
| 查询语言 | Elasticsearch DSL | Elasticsearch DSL | LogQL |
| 部署复杂度 | 高 | 中等 | 低 |
| 可扩展性 | 高 | 高 | 高 |
| 社区支持 | 强大 | 强大 | 快速成长 |
| 学习曲线 | 陡峭 | 中等 | 平缓 |
| 适用场景 | 复杂查询和分析 | 云原生环境 | 资源受限环境 |

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
  - 管理索引创建和删除
  - 分配分片
  - 跟踪集群节点
  - 更新集群状态

- **数据节点（Data Node）**：存储数据和执行数据相关操作
  - 存储索引分片
  - 执行 CRUD 操作
  - 执行搜索和聚合

- **客户端节点（Client Node）**：处理请求路由和负载均衡
  - 转发请求到适当的节点
  - 减轻主节点和数据节点的负担

- **摄取节点（Ingest Node）**：预处理文档，执行转换
  - 在索引前执行数据转换
  - 类似于轻量级的 Logstash

- **协调节点（Coordinating Node）**：仅协调请求
  - 分发搜索请求
  - 合并结果

#### 2.3.2 索引设计

- **分片（Shard）**：将索引分成多个部分，分布在不同节点
  - 主分片：索引的原始分片
  - 副本分片：主分片的复制
  - 分片大小建议：每个分片 20-40GB

- **副本（Replica）**：分片的复制，提供高可用和读取性能
  - 提高搜索性能
  - 提供故障恢复
  - 副本数量取决于可用性需求

- **索引生命周期管理（ILM）**：自动管理索引，包括滚动、收缩和删除
  - 热-温-冷-冻结架构
  - 自动滚动索引
  - 自动删除旧数据

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
  geoip {
    source => "clientip"
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
  ```
  # pipelines.yml
  - pipeline.id: apache
    path.config: "/etc/logstash/conf.d/apache.conf"
  - pipeline.id: nginx
    path.config: "/etc/logstash/conf.d/nginx.conf"
  ```

- **持久化队列**：防止数据丢失
  ```
  queue.type: persisted
  queue.max_bytes: 1gb
  ```

- **批处理**：提高吞吐量
  ```
  batch_size: 125
  batch_delay: 50
  ```

- **工作线程**：并行处理
  ```
  pipeline.workers: 4
  pipeline.batch.size: 250
  ```

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
      
  filebeat:
    image: docker.elastic.co/beats/filebeat:7.14.0
    container_name: filebeat
    volumes:
      - ./filebeat/filebeat.yml:/usr/share/filebeat/filebeat.yml:ro
      - /var/log:/var/log:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
    networks:
      - elk
    depends_on:
      - elasticsearch
      - logstash

networks:
  elk:

volumes:
  elasticsearch-data:
```

#### 3.1.2 Elasticsearch 配置

```yaml
# elasticsearch.yml
cluster.name: "docker-cluster"
network.host: 0.0.0.0
discovery.type: single-node

# 安全设置（生产环境）
xpack.security.enabled: true
xpack.license.self_generated.type: basic
xpack.security.transport.ssl.enabled: true
```

#### 3.1.3 Logstash 配置

```yaml
# logstash.yml
http.host: "0.0.0.0"
xpack.monitoring.elasticsearch.hosts: [ "http://elasticsearch:9200" ]

# 性能设置
pipeline.workers: 2
pipeline.batch.size: 125
pipeline.batch.delay: 50
queue.type: persisted
```

#### 3.1.4 Filebeat 配置

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/*.log
    - /var/log/messages*

filebeat.config.modules:
  path: ${path.config}/modules.d/*.yml
  reload.enabled: false

setup.dashboards.enabled: true
setup.kibana:
  host: "kibana:5601"

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  indices:
    - index: "filebeat-%{[agent.version]}-%{+yyyy.MM.dd}"

# 或者输出到 Logstash
# output.logstash:
#   hosts: ["logstash:5044"]
```

### 3.2 Kubernetes 部署

#### 3.2.1 使用 Helm 部署

```bash
# 添加 Elastic Helm 仓库
helm repo add elastic https://helm.elastic.co
helm repo update

# 创建命名空间
kubectl create namespace elk

# 部署 Elasticsearch
helm install elasticsearch elastic/elasticsearch \
  --namespace elk \
  --set replicas=3 \
  --set minimumMasterNodes=2

# 部署 Kibana
helm install kibana elastic/kibana \
  --namespace elk \
  --set elasticsearchHosts=http://elasticsearch-master:9200

# 部署 Logstash
helm install logstash elastic/logstash \
  --namespace elk \
  --set logstashPipeline.logstash.conf="input { tcp { port => 5044 } } output { elasticsearch { hosts => ['elasticsearch-master:9200'] } }"

# 部署 Filebeat
helm install filebeat elastic/filebeat \
  --namespace elk
```

#### 3.2.2 使用 Operator 部署

```bash
# 安装 ECK 操作符
kubectl apply -f https://download.elastic.co/downloads/eck/1.7.1/all-in-one.yaml

# 部署 Elasticsearch 集群
cat <<EOF | kubectl apply -f -
apiVersion: elasticsearch.k8s.elastic.co/v1
kind: Elasticsearch
metadata:
  name: elasticsearch
  namespace: elk
spec:
  version: 7.14.0
  nodeSets:
  - name: default
    count: 3
    config:
      node.master: true
      node.data: true
      node.ingest: true
    volumeClaimTemplates:
    - metadata:
        name: elasticsearch-data
      spec:
        accessModes:
        - ReadWriteOnce
        resources:
          requests:
            storage: 100Gi
EOF

# 部署 Kibana
cat <<EOF | kubectl apply -f -
apiVersion: kibana.k8s.elastic.co/v1
kind: Kibana
metadata:
  name: kibana
  namespace: elk
spec:
  version: 7.14.0
  count: 1
  elasticsearchRef:
    name: elasticsearch
EOF
```

### 3.3 云服务部署

#### 3.3.1 Elastic Cloud

1. 注册 Elastic Cloud 账户：https://cloud.elastic.co/
2. 创建部署：
   - 选择区域和云提供商（AWS、GCP、Azure）
   - 选择部署模板（日志监控、安全分析等）
   - 配置集群大小和硬件规格
   - 设置安全选项
3. 配置 Beats 或 Logstash 连接到云端 Elasticsearch：

```yaml
# filebeat.yml 连接到 Elastic Cloud
output.elasticsearch:
  cloud.id: "deployment-name:xxxxxxxxxxxx"
  cloud.auth: "elastic:password"
```

#### 3.3.2 AWS Elasticsearch Service

1. 登录 AWS 控制台，导航到 Amazon Elasticsearch Service
2. 创建域：
   - 选择部署类型（生产或开发）
   - 选择 Elasticsearch 版本
   - 配置实例类型和数量
   - 配置存储和网络设置
3. 配置 Logstash 输出到 AWS Elasticsearch：

```
output {
  elasticsearch {
    hosts => ["https://vpc-domain-name.region.es.amazonaws.com:443"]
    aws_access_key_id => "your-access-key"
    aws_secret_access_key => "your-secret-key"
    region => "us-west-1"
  }
}
```

## 4. ELK Stack 实际应用案例

### 4.1 微服务日志收集案例

#### 4.1.1 场景描述

一个电子商务平台，包含多个微服务：用户服务、产品服务、订单服务、支付服务等。需要集中收集和分析所有服务的日志，以便快速定位问题和监控系统健康状况。

#### 4.1.2 架构设计

```
用户服务 → Filebeat → Kafka → Logstash → Elasticsearch → Kibana
产品服务 → Filebeat ↗
订单服务 → Filebeat ↗
支付服务 → Filebeat ↗
```

#### 4.1.3 实现步骤

1. **配置 Filebeat 收集微服务日志**：

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/user-service/*.log
  fields:
    service: user-service
    environment: production
  fields_under_root: true
  json.keys_under_root: true
  json.message_key: message
  json.add_error_key: true

- type: log
  enabled: true
  paths:
    - /var/log/order-service/*.log
  fields:
    service: order-service
    environment: production
  fields_under_root: true
  json.keys_under_root: true
  json.message_key: message
  json.add_error_key: true

output.kafka:
  hosts: ["kafka1:9092", "kafka2:9092", "kafka3:9092"]
  topic: "logs"
  partition.round_robin:
    reachable_only: true
  required_acks: 1
  compression: gzip
  max_message_bytes: 1000000
```

2. **配置 Logstash 处理日志**：

```
# logstash.conf
input {
  kafka {
    bootstrap_servers => "kafka1:9092,kafka2:9092,kafka3:9092"
    topics => ["logs"]
    group_id => "logstash"
    codec => "json"
    auto_offset_reset => "latest"
  }
}

filter {
  if [service] == "user-service" {
    mutate {
      add_field => { "[@metadata][index]" => "user-service" }
    }
  } else if [service] == "order-service" {
    mutate {
      add_field => { "[@metadata][index]" => "order-service" }
    }
  } else {
    mutate {
      add_field => { "[@metadata][index]" => "other-services" }
    }
  }
  
  date {
    match => [ "timestamp", "ISO8601" ]
    target => "@timestamp"
  }
  
  if [level] == "ERROR" or [level] == "FATAL" {
    mutate {
      add_tag => [ "error" ]
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "%{[@metadata][index]}-%{+YYYY.MM.dd}"
    user => "elastic"
    password => "changeme"
  }
}
```

3. **创建 Kibana 仪表板**：

- 服务健康状况仪表板：显示各服务的日志数量、错误率和响应时间
- 错误分析仪表板：聚焦于错误日志，按服务和错误类型分类
- 用户行为仪表板：分析用户操作日志，展示用户活动模式
- 性能监控仪表板：监控关键性能指标，如响应时间和吞吐量

4. **设置告警**：

```
# Elasticsearch Watcher 告警配置
{
  "trigger": {
    "schedule": {
      "interval": "5m"
    }
  },
  "input": {
    "search": {
      "request": {
        "indices": ["*-service-*"],
        "body": {
          "query": {
            "bool": {
              "must": [
                { "match": { "level": "ERROR" } },
                { "range": { "@timestamp": { "gte": "now-5m" } } }
              ]
            }
          },
          "aggs": {
            "service_errors": {
              "terms": {
                "field": "service.keyword",
                "size": 10
              }
            }
          },
          "size": 0
        }
      }
    }
  },
  "condition": {
    "compare": {
      "ctx.payload.hits.total": {
        "gt": 10
      }
    }
  },
  "actions": {
    "email_admin": {
      "email": {
        "to": "admin@example.com",
        "subject": "High error rate detected",
        "body": "More than 10 errors in the last 5 minutes. Services affected: {{ctx.payload.aggregations.service_errors.buckets}}"
      }
    }
  }
}
```

### 4.2 安全日志分析案例

#### 4.2.1 场景描述

一个金融机构需要收集和分析各种安全日志，包括防火墙日志、入侵检测系统日志、身份验证日志等，以检测和响应安全威胁。

#### 4.2.2 架构设计

```
防火墙 → Filebeat → Logstash → Elasticsearch → Kibana
IDS/IPS → Filebeat ↗
服务器 → Auditbeat ↗
网络设备 → Packetbeat ↗
```

#### 4.2.3 实现步骤

1. **配置各种 Beats 收集安全日志**：

```yaml
# filebeat.yml (防火墙日志)
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/firewall/*.log
  fields:
    source: firewall
    type: security
  fields_under_root: true

# auditbeat.yml
auditbeat.modules:
- module: auditd
  audit_rules: |
    -w /etc/passwd -p wa -k identity
    -w /etc/group -p wa -k identity
    -a always,exit -F arch=b64 -S execve -k exec

- module: file_integrity
  paths:
  - /bin
  - /usr/bin
  - /sbin
  - /usr/sbin
  - /etc
```

2. **配置 Logstash 处理安全日志**：

```
# logstash-security.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [source] == "firewall" {
    grok {
      match => { "message" => "%{SYSLOGTIMESTAMP:timestamp} %{HOSTNAME:firewall_host} %{WORD:action} %{IP:src_ip}:%{NUMBER:src_port} -> %{IP:dest_ip}:%{NUMBER:dest_port}" }
    }
  }
  
  if [event.module] == "auditd" {
    # 处理审计日志
  }
  
  # 添加地理位置信息
  if [src_ip] {
    geoip {
      source => "src_ip"
      target => "src_geo"
    }
  }
  
  # 威胁情报丰富
  if [src_ip] {
    translate {
      field => "src_ip"
      destination => "threat_intel"
      dictionary_path => "/etc/logstash/threat_intel.yml"
      fallback => "unknown"
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "security-%{+YYYY.MM.dd}"
    user => "elastic"
    password => "changeme"
  }
}
```

3. **创建安全分析仪表板**：

- 安全概览仪表板：显示安全事件总数、按类型分布和趋势
- 网络流量分析：展示可疑网络连接和地理位置分布
- 用户活动监控：跟踪用户登录和权限变更
- 威胁检测：基于规则的异常检测和告警

4. **设置安全告警和响应**：

```
# Elasticsearch Watcher 安全告警
{
  "trigger": {
    "schedule": {
      "interval": "1m"
    }
  },
  "input": {
    "search": {
      "request": {
        "indices": ["security-*"],
        "body": {
          "query": {
            "bool": {
              "should": [
                { "match": { "action": "blocked" } },
                { "match": { "threat_intel": "malicious" } },
                { "range": { "failed_attempts": { "gt": 5 } } }
              ],
              "minimum_should_match": 1,
              "filter": {
                "range": { "@timestamp": { "gte": "now-2m" } }
              }
            }
          }
        }
      }
    }
  },
  "condition": {
    "compare": {
      "ctx.payload.hits.total": {
        "gt": 0
      }
    }
  },
  "actions": {
    "slack_notification": {
      "slack": {
        "message": {
          "from": "Security Monitoring",
          "to": ["#security-alerts"],
          "text": "Potential security threat detected. Check the security dashboard."
        }
      }
    },
    "create_incident": {
      "webhook": {
        "method": "POST",
        "url": "https://incident-management-system/api/incidents",
        "body": "{ \"title\": \"Security Alert\", \"description\": \"Potential security threat detected\", \"severity\": \"high\" }"
      }
    }
  }
}
```

### 4.3 容器化环境日志收集案例

#### 4.3.1 场景描述

一个基于 Kubernetes 的容器化环境，运行多个微服务应用。需要收集容器日志、Kubernetes 事件和节点指标，以监控应用性能和排查问题。

#### 4.3.2 架构设计

```
Kubernetes 节点 → Filebeat DaemonSet → Logstash StatefulSet → Elasticsearch StatefulSet → Kibana Deployment
```

#### 4.3.3 实现步骤

1. **部署 Elasticsearch 和 Kibana**：

```yaml
# elasticsearch.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: elasticsearch
  namespace: logging
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
          value: k8s-logs
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
          storage: 100Gi
```

2. **部署 Filebeat DaemonSet**：

```yaml
# filebeat-kubernetes.yaml
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
      terminationGracePeriodSeconds: 30
      containers:
      - name: filebeat
        image: docker.elastic.co/beats/filebeat:7.14.0
        args: [
          "-c", "/etc/filebeat.yml",
          "-e",
        ]
        env:
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        securityContext:
          runAsUser: 0
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
          defaultMode: 0600
      - name: varlibdockercontainers
        hostPath:
          path: /var/lib/docker/containers
      - name: varlog
        hostPath:
          path: /var/log
      - name: data
        hostPath:
          path: /var/lib/filebeat-data
          type: DirectoryOrCreate
```

3. **配置 Filebeat 收集容器日志**：

```yaml
# filebeat-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: filebeat-config
  namespace: logging
data:
  filebeat.yml: |
    filebeat.autodiscover:
      providers:
        - type: kubernetes
          node: ${NODE_NAME}
          hints.enabled: true
          hints.default_config:
            type: container
            paths:
              - /var/log/containers/*${data.kubernetes.container.id}.log

    processors:
      - add_cloud_metadata:
      - add_host_metadata:
      - add_kubernetes_metadata:
          host: ${NODE_NAME}
          matchers:
          - logs_path:
              logs_path: "/var/log/containers/"

    output.elasticsearch:
      hosts: ['elasticsearch:9200']
      index: "filebeat-%{[agent.version]}-%{+yyyy.MM.dd}"
```

4. **创建 Kubernetes 监控仪表板**：

- 容器日志仪表板：按命名空间、Pod 和容器分类显示日志
- Kubernetes 事件仪表板：显示集群事件，如 Pod 创建、删除和错误
- 节点监控仪表板：显示节点资源使用情况和健康状态
- 应用性能仪表板：显示应用响应时间、错误率和请求量

## 5. 常见问题与解决方案

### 5.1 性能问题

#### 5.1.1 Elasticsearch 性能优化

**问题**：Elasticsearch 查询响应慢，集群负载高。

**解决方案**：
- 优化索引设计：合理设置分片数量（每个分片 20-40GB）
- 使用索引生命周期管理（ILM）：热-温-冷-冻结架构
- 调整 JVM 堆大小：设置为可用内存的 50%，但不超过 32GB
- 使用 SSD 存储：提高 I/O 性能
- 优化查询：使用过滤器代替查询，避免通配符查询
- 使用索引别名和索引模板：简化索引管理
- 配置缓存：字段数据缓存、查询缓存和请求缓存

#### 5.1.2 Logstash 性能优化

**问题**：Logstash 处理能力不足，导致日志处理延迟。

**解决方案**：
- 使用多个 Logstash 实例：水平扩展
- 配置多管道：根据数据类型分离管道
- 优化过滤器：减少复杂正则表达式的使用
- 使用持久化队列：防止数据丢失
- 调整批处理设置：增加批处理大小
- 增加工作线程数：充分利用多核 CPU
- 使用 Kafka 作为缓冲：解耦数据收集和处理

#### 5.1.3 Beats 性能优化

**问题**：Beats 占用过多资源或丢失日志。

**解决方案**：
- 调整收集频率：根据需求设置合适的周期
- 配置批处理：增加批处理大小和超时时间
- 启用压缩：减少网络传输量
- 使用多播：减少网络负载
- 配置队列设置：防止数据丢失
- 调整并发设置：优化资源使用

### 5.2 存储问题

#### 5.2.1 存储空间快速增长

**问题**：Elasticsearch 存储空间快速耗尽。

**解决方案**：
- 实施索引生命周期管理：自动删除旧索引
- 配置索引滚动：限制单个索引大小
- 使用压缩：启用索引压缩
- 优化映射：减少不必要的字段存储
- 使用冷热架构：将旧数据移至低成本存储
- 实施日志采样：对高频日志进行采样
- 使用快照和恢复：定期备份和清理数据

#### 5.2.2 索引碎片化

**问题**：大量小索引导致性能下降。

**解决方案**：
- 使用索引模板：标准化索引设置
- 配置索引别名：简化索引管理
- 实施索引滚动策略：控制索引大小和数量
- 定期执行索引合并：减少碎片
- 使用 Curator 工具：自动化索引管理

### 5.3 高可用性问题

#### 5.3.1 Elasticsearch 集群稳定性

**问题**：Elasticsearch 集群不稳定，节点频繁离线。

**解决方案**：
- 配置适当的主节点数量：通常为 3 个
- 设置 `discovery.zen.minimum_master_nodes`：防止脑裂
- 使用专用主节点：分离主节点和数据节点角色
- 配置合理的超时设置：避免过早判断节点离线
- 监控集群健康状态：及时发现问题
- 实施滚动重启策略：减少维护影响
- 使用跨区域部署：提高容灾能力

#### 5.3.2 数据丢失问题

**问题**：系统故障导致日志数据丢失。

**解决方案**：
- 配置 Elasticsearch 副本：至少一个副本
- 使用持久化队列：在 Logstash 中启用
- 实施缓冲机制：使用 Kafka 或 Redis 作为缓冲
- 配置 Beats 重试机制：确保数据传输可靠
- 定期备份：使用快照功能
- 实施数据验证：监控数据完整性

## 6. 总结与最佳实践

### 6.1 ELK Stack 优势

- **强大的搜索能力**：基于 Elasticsearch 的全文搜索和分析
- **灵活的数据处理**：Logstash 提供丰富的数据处理能力
- **直观的可视化**：Kibana 提供丰富的可视化和仪表板功能
- **丰富的生态系统**：大量插件和集成选项
- **可扩展性**：支持从单节点到大型集群的扩展
- **多功能性**：不仅限于日志收集，还支持指标监控、安全分析等

### 6.2 最佳实践

#### 6.2.1 架构设计最佳实践

- **分层架构**：收集层、处理层、存储层、可视化层
- **缓冲机制**：使用消息队列解耦各层
- **冗余设计**：关键组件配置高可用
- **水平扩展**：根据需求扩展各层组件
- **资源隔离**：不同类型的数据使用不同的索引

#### 6.2.2 日志收集最佳实践

- **结构化日志**：使用 JSON 等结构化格式
- **标准化字段**：统一字段命名和格式
- **添加元数据**：环境、服务名称、版本等
- **日志分级**：合理使用日志级别
- **日志采样**：对高频日志进行采样
- **安全考虑**：保护敏感信息

#### 6.2.3 Elasticsearch 最佳实践

- **索引设计**：按时间和数据类型分索引
- **映射优化**：明确字段类型，禁用不需要的字段
- **分片策略**：每个分片 20-40GB，副本数根据可用性需求设置
- **资源配置**：JVM 堆大小设置为可用内存的 50%，但不超过 32GB
- **监控与维护**：定期监控集群健康状态，执行必要的维护

#### 6.2.4 Logstash 最佳实践

- **管道设计**：根据数据类型分离管道
- **过滤器优化**：减少复杂正则表达式的使用
- **批处理配置**：根据数据量和资源调整批处理设置
- **错误处理**：配置死信队列处理失败事件
- **监控性能**：监控处理延迟和吞吐量

#### 6.2.5 Kibana 最佳实践

- **仪表板设计**：从概览到详细信息的层次结构
- **可视化选择**：根据数据类型选择合适的可视化
- **保存搜索**：保存常用搜索查询
- **用户访问控制**：根据角色设置适当的权限
- **告警配置**：设置关键指标的告警

### 6.3 未来发展趋势

- **机器学习集成**：异常检测和预测分析
- **APM 集成**：应用性能监控与日志关联
- **安全分析增强**：SIEM 功能增强
- **可观测性平台**：日志、指标和追踪的统一
- **云原生支持**：更好的容器和 Kubernetes 集成
- **自动化运维**：自动化索引管理和集群维护

## 7. 参考资源

- [Elasticsearch 官方文档](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [Logstash 官方文档](https://www.elastic.co/guide/en/logstash/current/index.html)
- [Kibana 官方文档](https://www.elastic.co/guide/en/kibana/current/index.html)
- [Beats 官方文档](https://www.elastic.co/guide/en/beats/libbeat/current/index.html)
- [Elastic Stack 最佳实践](https://www.elastic.co/guide/en/elasticsearch/reference/current/setup.html)
- [Elasticsearch: The Definitive Guide](https://www.elastic.co/guide/en/elasticsearch/guide/current/index.html)
- [Elastic 博客](https://www.elastic.co/blog/)
- [Elastic 社区](https://discuss.elastic.co/)