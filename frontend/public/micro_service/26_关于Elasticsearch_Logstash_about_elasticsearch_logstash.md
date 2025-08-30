以下是使用 Docker 快速搭建 Logstash 并验证其功能的详细步骤，涵盖配置、运行和功能测试：

---

### **1. 准备工作**
#### (1) 安装 Docker
确保已安装 Docker 及 Docker Compose：
```bash
# 安装 Docker（以 Ubuntu 为例）
sudo apt-get update && sudo apt-get install docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
```

#### (2) 创建测试目录
```bash
mkdir ~/logstash-test && cd ~/logstash-test
```

---

### **2. 创建 Logstash 配置文件**
#### (1) 编写基础配置文件 `logstash.conf`
```bash
cat << EOF > logstash.conf
input {
  stdin {}  # 从标准输入读取数据
}

output {
  stdout {  # 输出到标准输出（控制台）
    codec => rubydebug  # 格式化输出
  }
}
EOF
```

#### (2) 配置文件说明
- **input**: 使用 `stdin` 插件从命令行输入数据。
- **output**: 使用 `stdout` 插件将处理后的数据输出到控制台。

---

### **3. 启动 Logstash 容器**
#### (1) 运行临时容器
```bash
docker run -it --rm \
  -v $(pwd)/logstash.conf:/usr/share/logstash/pipeline/logstash.conf \
  docker.elastic.co/logstash/logstash:8.10.0
```

#### (2) 参数解释
- `-it`: 以交互模式运行容器（方便输入数据）。
- `--rm`: 容器退出后自动删除。
- `-v`: 挂载配置文件到容器内的 Logstash 配置目录。

---

### **4. 验证基础功能**
#### (1) 输入测试数据
容器启动后，直接在控制台输入任意文本：
```text
Hello, Logstash!
This is a test.
```

#### (2) 观察输出结果
Logstash 会实时处理输入数据并输出格式化结果：
```json
{
    "message" => "Hello, Logstash!",
    "@version" => "1",
    "host" => "a1b2c3d4e5f6",
    "@timestamp" => 2023-10-05T08:00:00.000Z
}
{
    "message" => "This is a test.",
    "@version" => "1",
    "host" => "a1b2c3d4e5f6",
    "@timestamp" => 2023-10-05T08:00:01.000Z
}
```

---

### **5. 进阶测试：处理文件并输出到 Elasticsearch**
#### (1) 更新配置文件 `logstash.conf`
```bash
cat << EOF > logstash.conf
input {
  file {  # 从文件读取数据
    path => "/usr/share/logstash/test.log"  # 容器内文件路径
    start_position => "beginning"
  }
}

filter {  # 添加过滤器（示例：解析 JSON）
  json {
    source => "message"
  }
}

output {
  stdout { codec => rubydebug }

  elasticsearch {  # 输出到 Elasticsearch
    hosts => ["http://host.docker.internal:9200"]  # 宿主机 ES 地址
    index => "logstash-test-%{+YYYY.MM.dd}"
  }
}
EOF
```

#### (2) 创建测试日志文件
```bash
echo '{"user": "Alice", "action": "login"}' > test.log
echo '{"user": "Bob", "action": "logout"}' >> test.log
```

#### (3) 启动容器（绑定文件）
```bash
docker run -it --rm \
  -v $(pwd)/logstash.conf:/usr/share/logstash/pipeline/logstash.conf \
  -v $(pwd)/test.log:/usr/share/logstash/test.log \
  --add-host=host.docker.internal:host-gateway \  # 允许容器访问宿主机服务
  docker.elastic.co/logstash/logstash:8.10.0
```

#### (4) 验证 Elasticsearch 数据
在宿主机上执行：
```bash
curl -XGET "http://localhost:9200/logstash-test-*/_search?pretty"
```
输出应包含 `Alice` 和 `Bob` 的记录。

---

### **6. 使用 Docker Compose 编排**
#### (1) 创建 `docker-compose.yml`
```bash
cat << EOF > docker-compose.yml
version: '3'
services:
  logstash:
    image: docker.elastic.co/logstash/logstash:8.10.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
      - ./test.log:/usr/share/logstash/test.log
    network_mode: host  # 直接使用宿主机网络
EOF
```

#### (2) 启动服务
```bash
docker-compose up
```

---

### **7. 关键问题排查**
#### (1) 查看 Logstash 日志
```bash
docker logs <container_id>
```

#### (2) 检查配置文件语法
```bash
docker run --rm -it \
  -v $(pwd)/logstash.conf:/usr/share/logstash/pipeline/logstash.conf \
  docker.elastic.co/logstash/logstash:8.10.0 \
  -t  # 测试配置语法
```

---

### **总结**
- **快速验证**: 使用 `stdin/stdout` 插件可在 1 分钟内测试 Logstash 流程。
- **文件处理**: 通过挂载文件实现日志采集。
- **生产级配置**: 结合 Elasticsearch 和过滤器实现完整数据处理流水线。
- **调试技巧**: 利用 `rubydebug` 格式和日志排查问题。

```bash
[2025-04-10T03:13:16,456][INFO ][logstash.outputs.elasticsearch][main] Failed to perform request {:message=>"Connect to 127.0.0.1:9200 [/127.0.0.1] failed: Connection refused", :exception=>Manticore::SocketException, :cause=>#<Java::OrgApacheHttpConn::HttpHostConnectException: Connect to 127.0.0.1:9200 [/127.0.0.1] failed: Connection refused>}
[2025-04-10T03:13:16,459][WARN ][logstash.outputs.elasticsearch][main] Attempted to resurrect connection to dead ES instance, but got an error {:url=>"https://elastic:xxxxxx@127.0.0.1:9200/", :exception=>LogStash::Outputs::ElasticSearch::HttpClient::Pool::HostUnreachableError, :message=>"Elasticsearch Unreachable: [https://127.0.0.1:9200/][Manticore::SocketException] Connect to 127.0.0.1:9200 [/127.0.0.1] failed: Connection refused"}
[2025-04-10T03:13:21,483][INFO ][logstash.outputs.elasticsearch][main] Failed to perform request {:message=>"Connect to 127.0.0.1:9200 [/127.0.0.1] failed: Connection refused", :exception=>Manticore::SocketException, :cause=>#<Java::OrgApacheHttpConn::HttpHostConnectException: Connect to 127.0.0.1:9200 [/127.0.0.1] failed: Connection refused>}
```


