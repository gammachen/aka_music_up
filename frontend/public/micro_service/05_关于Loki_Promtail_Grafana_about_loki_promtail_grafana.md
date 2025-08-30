# Loki + Promtail + Grafana 日志收集方案详解

## 1. Loki + Promtail + Grafana 基本原理

### 1.1 方案概述

Loki + Promtail + Grafana 是由 Grafana Labs 开发的云原生日志聚合系统，专为 Kubernetes 环境设计，但也适用于其他环境。该方案以高效、低成本和易于操作为设计理念。

- **Loki**：水平可扩展、高可用的多租户日志聚合系统，受 Prometheus 启发
- **Promtail**：专为 Loki 设计的日志收集代理
- **Grafana**：数据可视化和分析平台，支持多种数据源

### 1.2 工作原理

#### 1.2.1 数据流向

1. **数据收集**：Promtail 从各种源（文件、系统日志、容器日志等）收集日志数据
2. **标签添加**：Promtail 为日志添加标签（如应用名称、环境等）
3. **数据压缩**：Promtail 压缩日志数据并发送到 Loki
4. **数据存储**：Loki 根据标签索引日志，并将日志内容存储在对象存储中
5. **数据查询**：Grafana 通过 LogQL 查询语言从 Loki 检索和可视化日志

#### 1.2.2 各组件详细工作原理

**Loki 工作原理**：
- 采用类似 Prometheus 的标签索引系统
- 只索引元数据（标签），而非全文
- 将日志内容压缩存储在对象存储中
- 使用 "chunks" 存储压缩的日志数据
- 提供 LogQL 查询语言，类似于 Prometheus PromQL
- 支持多租户模式

**Promtail 工作原理**：
- 发现并跟踪日志文件
- 为日志添加标签（如 Kubernetes 元数据）
- 将日志发送到 Loki 实例
- 支持多种目标发现机制
- 提供基本的日志处理功能

**Grafana 工作原理**：
- 通过 Loki 数据源插件连接到 Loki
- 提供 Explore 界面进行日志查询
- 支持 LogQL 查询语言
- 允许创建日志可视化和仪表板
- 支持与 Prometheus 指标集成

### 1.3 与其他日志系统的比较

| 特性 | Loki + Promtail + Grafana | ELK Stack | Fluentd + ES + Kibana |
|------|---------------------------|-----------|----------------------|
| 资源消耗 | 非常低 | 较高 | 中等 |
| 存储效率 | 高（只索引元数据） | 低（全文索引） | 低（全文索引） |
| 查询能力 | 基本（LogQL） | 强大（Elasticsearch DSL） | 强大（Elasticsearch DSL） |
| 部署复杂度 | 低 | 高 | 中等 |
| 可扩展性 | 高 | 高 | 高 |
| 多租户支持 | 原生支持 | 需要额外配置 | 需要额外配置 |
| 云原生集成 | 优秀 | 良好 | 优秀 |
| 成本 | 低 | 高 | 中等 |
| 学习曲线 | 平缓 | 陡峭 | 中等 |

## 2. Loki + Promtail + Grafana 架构设计

### 2.1 基础架构

#### 2.1.1 单节点架构

适用于开发环境或小型应用场景：

```
微服务 → Promtail → Loki → Grafana
```

#### 2.1.2 分布式架构

适用于生产环境或大型应用场景：

```
微服务集群 → Promtail → Loki 分布式集群 → Grafana
```

### 2.2 Loki 组件架构

Loki 可以部署为单体模式或微服务模式：

#### 2.2.1 单体模式

所有组件在单个进程中运行，适合小型部署：

```
Promtail → Loki (单体) → Grafana
```

#### 2.2.2 微服务模式

各组件独立运行，适合大型部署：

```
Promtail → Loki 分布式集群 → Grafana
```

Loki 微服务模式包含以下组件：

- **Distributor**：接收来自客户端的日志流
- **Ingester**：将日志数据写入长期存储
- **Querier**：处理 LogQL 查询请求
- **Query Frontend**：查询负载均衡和缓存
- **Ruler**：评估告警规则
- **Compactor**：压缩和删除旧数据

### 2.3 存储架构

Loki 使用两种类型的存储：

#### 2.3.1 索引存储

存储日志流的标签和元数据，支持以下后端：

- **Cassandra**
- **BoltDB**
- **DynamoDB**
- **Bigtable**

#### 2.3.2 块存储

存储压缩的日志内容，支持以下后端：

- **本地文件系统**
- **S3 兼容对象存储**
- **Google Cloud Storage**
- **Azure Blob Storage**

### 2.4 多租户架构

Loki 原生支持多租户，每个租户的数据完全隔离：

```
租户 A 的 Promtail → Loki (X-Scope-OrgID: A) → 租户 A 的 Grafana
租户 B 的 Promtail → Loki (X-Scope-OrgID: B) → 租户 B 的 Grafana
```

## 3. Loki + Promtail + Grafana 部署指南

### 3.1 Docker Compose 部署

#### 3.1.1 docker-compose.yml

```yaml
version: '3'
services:
  loki:
    image: grafana/loki:2.4.0
    container_name: loki
    ports:
      - "3100:3100"
    volumes:
      - ./loki-config.yaml:/etc/loki/local-config.yaml
      - loki-data:/loki
    command: -config.file=/etc/loki/local-config.yaml
    networks:
      - loki

  promtail:
    image: grafana/promtail:2.4.0
    container_name: promtail
    volumes:
      - ./promtail-config.yaml:/etc/promtail/config.yaml
      - /var/log:/var/log
    command: -config.file=/etc/promtail/config.yaml
    networks:
      - loki
    depends_on:
      - loki

  grafana:
    image: grafana/grafana:8.3.0
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana-datasources.yaml:/etc/grafana/provisioning/datasources/datasources.yaml
    networks:
      - loki
    depends_on:
      - loki

networks:
  loki:

volumes:
  loki-data:
  grafana-data:
```

#### 3.1.2 Loki 配置

**loki-config.yaml**：

```yaml
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
    final_sleep: 0s
  chunk_idle_period: 5m
  chunk_retain_period: 30s

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    cache_ttl: 24h
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

compactor:
  working_directory: /loki/compactor
  shared_store: filesystem

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: false
  retention_period: 0s

ruler:
  storage:
    type: local
    local:
      directory: /loki/rules
  rule_path: /loki/rules-temp
  alertmanager_url: http://localhost:9093
  ring:
    kvstore:
      store: inmemory
  enable_api: true
```

#### 3.1.3 Promtail 配置

**promtail-config.yaml**：

```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: system
    static_configs:
      - targets:
          - localhost
        labels:
          job: varlogs
          __path__: /var/log/*log

  - job_name: containers
    static_configs:
      - targets:
          - localhost
        labels:
          job: containerlogs
          __path__: /var/lib/docker/containers/*/*log

    pipeline_stages:
      - json:
          expressions:
            stream: stream
            attrs: attrs
            tag: attrs.tag
      - labels:
          stream:
          tag:
```

#### 3.1.4 Grafana 数据源配置

**grafana-datasources.yaml**：

```yaml
apiVersion: 1

datasources:
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    version: 1
    editable: false
    isDefault: true
```

### 3.2 Kubernetes 部署

#### 3.2.1 使用 Helm 部署

```bash
# 添加 Grafana Helm 仓库
helm repo add grafana https://grafana.github.io/helm-charts

# 部署 Loki Stack (包含 Loki, Promtail, Grafana)
helm install loki-stack grafana/loki-stack \
  --set grafana.enabled=true \
  --set prometheus.enabled=false \
  --set loki.persistence.enabled=true \
  --set loki.persistence.size=10Gi \
  --namespace monitoring \
  --create-namespace
```

#### 3.2.2 使用 Kubernetes 清单文件

**loki-configmap.yaml**：

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: loki-config
  namespace: monitoring
data:
  loki.yaml: |
    auth_enabled: false

    server:
      http_listen_port: 3100

    ingester:
      lifecycler:
        address: 127.0.0.1
        ring:
          kvstore:
            store: inmemory
          replication_factor: 1
        final_sleep: 0s
      chunk_idle_period: 5m
      chunk_retain_period: 30s

    schema_config:
      configs:
        - from: 2020-10-24
          store: boltdb-shipper
          object_store: filesystem
          schema: v11
          index:
            prefix: index_
            period: 24h

    storage_config:
      boltdb_shipper:
        active_index_directory: /data/loki/boltdb-shipper-active
        cache_location: /data/loki/boltdb-shipper-cache
        cache_ttl: 24h
        shared_store: filesystem
      filesystem:
        directory: /data/loki/chunks

    limits_config:
      enforce_metric_name: false
      reject_old_samples: true
      reject_old_samples_max_age: 168h
```

**loki-deployment.yaml**：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: loki
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: loki
  template:
    metadata:
      labels:
        app: loki
    spec:
      containers:
        - name: loki
          image: grafana/loki:2.4.0
          args:
            - -config.file=/etc/loki/loki.yaml
          ports:
            - name: http
              containerPort: 3100
          volumeMounts:
            - name: config
              mountPath: /etc/loki
            - name: data
              mountPath: /data/loki
          resources:
            limits:
              memory: 512Mi
            requests:
              cpu: 100m
              memory: 256Mi
      volumes:
        - name: config
          configMap:
            name: loki-config
        - name: data
          persistentVolumeClaim:
            claimName: loki-data
---
apiVersion: v1
kind: Service
metadata:
  name: loki
  namespace: monitoring
spec:
  ports:
    - port: 3100
      protocol: TCP
      name: http
  selector:
    app: loki
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: loki-data
  namespace: monitoring
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

**promtail-configmap.yaml**：

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: promtail-config
  namespace: monitoring
data:
  promtail.yaml: |
    server:
      http_listen_port: 9080
      grpc_listen_port: 0

    positions:
      filename: /tmp/positions.yaml

    clients:
      - url: http://loki:3100/loki/api/v1/push

    scrape_configs:
      - job_name: kubernetes-pods
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_node_name]
            target_label: __host__
          - action: labelmap
            regex: __meta_kubernetes_pod_label_(.+)
          - action: replace
            replacement: $1
            separator: /
            source_labels:
              - __meta_kubernetes_namespace
              - __meta_kubernetes_pod_name
            target_label: job
          - action: replace
            source_labels: [__meta_kubernetes_namespace]
            target_label: namespace
          - action: replace
            source_labels: [__meta_kubernetes_pod_name]
            target_label: pod
          - action: replace
            source_labels: [__meta_kubernetes_pod_container_name]
            target_label: container
          - replacement: /var/log/pods/*$1/*.log
            separator: /
            source_labels:
              - __meta_kubernetes_namespace
              - __meta_kubernetes_pod_name
              - __meta_kubernetes_pod_container_name
            target_label: __path__
```

**promtail-daemonset.yaml**：

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: promtail
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: promtail
  template:
    metadata:
      labels:
        app: promtail
    spec:
      serviceAccount: promtail
      containers:
        - name: promtail
          image: grafana/promtail:2.4.0
          args:
            - -config.file=/etc/promtail/promtail.yaml
          volumeMounts:
            - name: config
              mountPath: /etc/promtail
            - name: varlog
              mountPath: /var/log
            - name: varlibdockercontainers
              mountPath: /var/lib/docker/containers
              readOnly: true
            - name: pods
              mountPath: /var/log/pods
              readOnly: true
          resources:
            limits:
              memory: 256Mi
            requests:
              cpu: 50m
              memory: 128Mi
      volumes:
        - name: config
          configMap:
            name: promtail-config
        - name: varlog
          hostPath:
            path: /var/log
        - name: varlibdockercontainers
          hostPath:
            path: /var/lib/docker/containers
        - name: pods
          hostPath:
            path: /var/log/pods
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: promtail
  namespace: monitoring
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: promtail
rules:
  - apiGroups: [""]
    resources:
      - nodes
      - nodes/proxy
      - services
      - endpoints
      - pods
    verbs: ["get", "watch", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: promtail
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: promtail
subjects:
  - kind: ServiceAccount
    name: promtail
    namespace: monitoring
```

### 3.3 高可用部署

#### 3.3.1 Loki 微服务模式配置

**loki-microservices-config.yaml**：

```yaml
auth_enabled: false

server:
  http_listen_port: 3100

common:
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 3
  ring:
    kvstore:
      store: memberlist

memberlist:
  join_members:
    - loki-memberlist

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    cache_ttl: 24h
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

ruler:
  alertmanager_url: http://alertmanager:9093

distributor:
  ring:
    kvstore:
      store: memberlist

ingester:
  lifecycler:
    ring:
      kvstore:
        store: memberlist
      replication_factor: 3
    final_sleep: 0s
  chunk_idle_period: 5m
  chunk_retain_period: 30s

querier:
  engine:
    timeout: 3m

query_range:
  results_cache:
    cache:
      embedded_cache:
        enabled: true
        max_size_mb: 100

compactor:
  working_directory: /loki/compactor
  shared_store: filesystem
  compaction_interval: 5m
```

## 4. Loki + Promtail + Grafana 实际应用案例

### 4.1 Kubernetes 微服务日志收集案例

#### 4.1.1 场景描述

一个基于 Kubernetes 的微服务平台，包含以下服务：
- 前端服务（React）
- API 服务（Node.js）
- 数据服务（Python）
- 认证服务（Java）

#### 4.1.2 日志收集架构

```
Kubernetes Pod → Promtail DaemonSet → Loki → Grafana
```

#### 4.1.3 实现步骤

1. **部署 Loki Stack**：

```bash
helm install loki-stack grafana/loki-stack \
  --set grafana.enabled=true \
  --set loki.persistence.enabled=true \
  --set loki.persistence.size=10Gi \
  --namespace monitoring \
  --create-namespace
```

2. **配置应用日志格式**：

为每个微服务配置结构化日志输出（JSON 格式），包含以下字段：
- timestamp
- level
- message
- service
- trace_id
- span_id

3. **创建 Grafana 仪表板**：

```
# 查询所有错误日志
{namespace="default"} |= "error" | json | level="error"

# 按服务分组查询响应时间
{namespace="default"} | json | response_time > 0 | unwrap response_time | by (service)

# 查询特定 trace_id 的所有日志
{namespace="default"} | json | trace_id="abc123"
```

4. **设置告警规则**：

```yaml
groups:
  - name: loki_alerts
    rules:
      - alert: HighErrorRate
        expr: sum(count_over_time({namespace="default"} |= "error"[5m])) > 10
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: High error rate detected
          description: More than 10 errors in 5 minutes
```

### 4.2 多环境日志集中管理案例

#### 4.2.1 场景描述

一个企业应用，需要集中管理开发、测试和生产环境的日志。

#### 4.2.2 日志收集架构

```
开发环境：应用 → Promtail (tenant=dev) → Loki
测试环境：应用 → Promtail (tenant=test) → Loki
生产环境：应用 → Promtail (tenant=prod) → Loki
                                      ↓
                                    Grafana
```

#### 4.2.3 实现步骤

1. **配置 Loki 多租户**：

```yaml
auth_enabled: true

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
    final_sleep: 0s
  chunk_idle_period: 5m
  chunk_retain_period: 30s

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    cache_ttl: 24h
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h
  max_global_streams_per_user: 10000
  ingestion_rate_mb: 10
  ingestion_burst_size_mb: 20

frontend:
  compress_responses: true

ruler:
  storage:
    type: local
    local:
      directory: /loki/rules
  rule_path: /loki/rules-temp
  alertmanager_url: http://localhost:9093
  ring:
    kvstore:
      store: inmemory
  enable_api: true
```

2. **配置开发环境 Promtail**：

```yaml
server:
  http_listen_port: 9080

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push
    tenant_id: dev

scrape_configs:
  - job_name: system
    static_configs:
      - targets:
          - localhost
        labels:
          job: varlogs
          env: dev
          __path__: /var/log/*log
```

3. **配置 Grafana 数据源**：

```yaml
apiVersion: 1

datasources:
  - name: Loki-Dev
    type: loki
    access: proxy
    url: http://loki:3100
    version: 1
    jsonData:
      maxLines: 1000
    secureJsonData:
      httpHeaderValue1: "dev"
    editable: false

  - name: Loki-Test
    type: loki
    access: proxy
    url: http://loki:3100
    version: 1
    jsonData:
      maxLines: 1000
    secureJsonData:
      httpHeaderValue1: "test"
    editable: false

  - name: Loki-Prod
    type: loki
    access: proxy
    url: http://loki:3100
    version: 1
    jsonData:
      maxLines: 1000
    secureJsonData:
      httpHeaderValue1: "prod"
    editable: false
```

4. **创建环境特定的仪表板**：
   - 开发环境监控仪表板
   - 测试环境监控仪表板
   - 生产环境监控仪表板
   - 跨环境比较仪表板

### 4.3 与 Prometheus 集成的可观测性案例

#### 4.3.1 场景描述

一个云原