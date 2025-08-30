以下是使用 Docker 快速搭建 Elasticsearch（ES）的详细流程与指令，涵盖单节点和集群模式的部署方案：

---

### 一、单节点 Elasticsearch 部署
适用于开发、测试环境。

#### 1. 拉取官方镜像
```bash
docker pull docker.elastic.co/elasticsearch/elasticsearch:8.12.0
```

#### 2. 创建数据持久化目录
```bash
mkdir -p ~/elasticsearch/data
chmod 777 ~/elasticsearch/data  # 确保容器有写入权限
```

#### 3. 启动容器
```bash
docker run -d \
  --name elasticsearch \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \  # 单节点模式
  -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" \  # JVM 内存限制
  -v ~/elasticsearch/data:/usr/share/elasticsearch/data \
  docker.elastic.co/elasticsearch/elasticsearch:8.12.0
```

#### 4. 验证是否运行成功
```bash
curl http://localhost:9200
# 返回类似以下 JSON 即成功：
# {
#   "name" : "xxxxxxxx",
#   "cluster_name" : "docker-cluster",
#   "cluster_uuid" : "xxxxxxxx",
#   "version" : { ... },
#   "tagline" : "You Know, for Search"
# }
```

---

### 二、Elasticsearch 集群部署
适用于生产环境（3 节点示例）。

#### 1. 创建 Docker 网络
```bash
docker network create es-net
```

#### 2. 创建配置文件目录
```bash
mkdir -p ~/elasticsearch/{config1,config2,config3}
```

#### 3. 配置每个节点的 `elasticsearch.yml`
**节点1配置（`~/elasticsearch/config1/elasticsearch.yml`）**：
```yaml
cluster.name: my-es-cluster
node.name: node-1
network.host: 0.0.0.0
discovery.seed_hosts: ["node1", "node2", "node3"]
cluster.initial_master_nodes: ["node-1", "node-2", "node-3"]
```

**节点2、节点3配置类似，仅需修改 `node.name`**。

#### 4. 启动 3 个节点
**节点1**：
```bash
docker run -d \
  --name es-node1 \
  --network es-net \
  -p 9200:9200 -p 9300:9300 \
  -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" \
  -v ~/elasticsearch/config1/elasticsearch.yml:/usr/share/elasticsearch/config/elasticsearch.yml \
  -v ~/elasticsearch/data1:/usr/share/elasticsearch/data \
  docker.elastic.co/elasticsearch/elasticsearch:8.12.0
```

**节点2**：
```bash
docker run -d \
  --name es-node2 \
  --network es-net \
  -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" \
  -v ~/elasticsearch/config2/elasticsearch.yml:/usr/share/elasticsearch/config/elasticsearch.yml \
  -v ~/elasticsearch/data2:/usr/share/elasticsearch/data \
  docker.elastic.co/elasticsearch/elasticsearch:8.12.0
```

**节点3**：
```bash
docker run -d \
  --name es-node3 \
  --network es-net \
  -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" \
  -v ~/elasticsearch/config3/elasticsearch.yml:/usr/share/elasticsearch/config/elasticsearch.yml \
  -v ~/elasticsearch/data3:/usr/share/elasticsearch/data \
  docker.elastic.co/elasticsearch/elasticsearch:8.12.0
```

#### 5. 验证集群状态
```bash
curl http://localhost:9200/_cluster/health?pretty
# 返回中 "status" 为 "green" 表示集群健康
```

---

### 三、安全配置（启用 HTTPS 和认证）
适用于生产环境安全加固。

#### 1. 生成证书（首次运行自动生成）
```bash
docker exec -it es-node1 \
  bin/elasticsearch-certutil cert -out config/elastic-certificates.p12 -pass ""
```

#### 2. 修改配置文件（所有节点）
在 `elasticsearch.yml` 中添加：
```yaml
xpack.security.enabled: true
xpack.security.transport.ssl.enabled: true
xpack.security.transport.ssl.verification_mode: certificate
xpack.security.transport.ssl.keystore.path: elastic-certificates.p12
xpack.security.transport.ssl.truststore.path: elastic-certificates.p12
```

#### 3. 设置用户密码
```bash
docker exec -it es-node1 \
  bin/elasticsearch-setup-passwords auto
# 记录生成的密码（如 elastic 用户）
```

#### 4. 访问时携带认证
```bash
curl -u elastic:your_password http://localhost:9200
```

---

### 四、常见问题解决

#### 1. 启动失败：内存不足
**错误**：`Exit code 137`  
**解决**：  
- 调整 `ES_JAVA_OPTS`（如 `-Xms512m -Xmx512m`）  
- 增加宿主机内存或限制容器内存：`--memory 4g`

#### 2. 权限问题
**错误**：`Permission denied`  
**解决**：  
```bash
chmod 777 ~/elasticsearch/data  # 或使用正确的用户权限
```

#### 3. 节点无法加入集群
**错误**：`master not discovered`  
**解决**：  
- 检查 `discovery.seed_hosts` 配置  
- 确保所有节点在同一 Docker 网络（`es-net`）  
- 开放端口 `9300`（节点间通信）

---

### 五、附：常用命令

- **查看日志**：
  ```bash
  docker logs -f es-node1
  ```

- **进入容器**：
  ```bash
  docker exec -it es-node1 /bin/bash
  ```

- **删除容器和数据**：
  ```bash
  docker rm -f es-node1 es-node2 es-node3
  rm -rf ~/elasticsearch/*
  ```

---

通过以上步骤，您可以在 Docker 中快速部署单节点或集群模式的 Elasticsearch，并配置基础安全功能。生产环境中建议至少部署 3 个节点，并启用 HTTPS 和密码认证。


