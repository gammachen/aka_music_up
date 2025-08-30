---

### **解决方案：在 Elasticsearch 8+ 中配置 Logstash 安全连接**

Elasticsearch 8 及以上版本默认启用了 **HTTPS** 和 **安全认证**，因此 Logstash 需要以下配置才能连接：

---

### **步骤 1：获取 Elasticsearch 的 CA 证书**
Elasticsearch 首次启动时会自动生成 CA 证书，需要将证书文件挂载到 Logstash 容器。

#### (1) **查找证书路径**
- Elasticsearch 的默认证书路径：
  ```bash
  /usr/share/elasticsearch/config/certs/http_ca.crt
  ```
- 如果 Elasticsearch 是通过 Docker 运行的，需将证书文件从容器复制到宿主机：
  ```bash
  docker cp <es_container_id>:/usr/share/elasticsearch/config/certs/http_ca.crt ./http_ca.crt
  ```

#### (2) **保存证书到测试目录**
将 `http_ca.crt` 复制到 Logstash 测试目录：
```bash
cp http_ca.crt ~/logstash-test/
```

---

### **步骤 2：修改 Logstash 配置文件**
更新 `logstash.conf` 的 `output` 部分，启用 HTTPS 和身份验证。

#### **完整配置示例**：
```bash
cat << EOF > logstash.conf
input {
  file {
    path => "/usr/share/logstash/test.log"
    start_position => "beginning"
  }
}

filter {
  json {
    source => "message"
  }
}

output {
  stdout { codec => rubydebug }

  elasticsearch {
    hosts => ["https://host.docker.internal:9200"]  # 使用 HTTPS
    index => "logstash-test-%{+YYYY.MM.dd}"
    user => "elastic"                             # 默认内置用户
    password => "your_elastic_password"           # 替换为实际密码
    ssl => true
    cacert => "/usr/share/logstash/http_ca.crt"   # CA 证书容器内路径
  }
}
EOF
```

---

### **步骤 3：启动 Logstash 容器（绑定证书）**
挂载证书文件并启动容器：
```bash
docker run -it --rm \
  -v $(pwd)/logstash.conf:/usr/share/logstash/pipeline/logstash.conf \
  -v $(pwd)/test.log:/usr/share/logstash/test.log \
  -v $(pwd)/http_ca.crt:/usr/share/logstash/http_ca.crt \
  --add-host=host.docker.internal:host-gateway \
  docker.elastic.co/logstash/logstash:8.10.0
```

---

### **步骤 4：验证连接**
#### (1) **检查 Logstash 日志**
如果出现以下日志，表示连接成功：
```text
[INFO ][logstash.outputs.elasticsearch] Elasticsearch pool URLs updated {:changes=>{:removed=>[], :added=>[https://elastic:xxxxxx@host.docker.internal:9200/]}}
```

#### (2) **查询 Elasticsearch 数据**
在宿主机执行以下命令（需配置身份验证）：
```bash
curl -k -u elastic:your_password "https://localhost:9200/logstash-test-*/_search?pretty"
```

---

### **关键配置说明**
| 参数                | 说明                                                                 |
|---------------------|--------------------------------------------------------------------|
| `hosts => "https"`  | 必须使用 `https` 协议，端口默认为 `9200`                              |
| `user`/`password`   | 使用 Elasticsearch 内置用户（如 `elastic`）或自定义用户                  |
| `ssl => true`       | 强制启用 SSL/TLS 加密                                                 |
| `cacert`            | CA 证书路径（容器内路径）                                              |

---

### **常见问题排查**
#### **1. 证书路径错误**
- **错误日志**：
  ```text
  [ERROR][logstash.outputs.elasticsearch] Failed to install template {:message=>"PKIX path building failed"}
  ```
- **解决方案**：检查 `cacert` 路径是否挂载正确。

#### **2. 密码错误**
- **错误日志**：
  ```text
  [ERROR][logstash.outputs.elasticsearch] Encountered a retryable error. Will Retry with exponential backoff
  ```
- **解决方案**：确认 `user` 和 `password` 与 Elasticsearch 用户一致。

#### **3. 网络不通**
- **错误日志**：
  ```text
  [WARN ][logstash.outputs.elasticsearch] Attempted to resurrect connection to dead ES instance
  ```
- **解决方案**：确保 `host.docker.internal` 能解析到宿主机 IP。

---

### **高级配置：使用 API 密钥**
如果不想使用用户名密码，可以创建 API 密钥：
```bash
# 在 Elasticsearch 中生成 API 密钥
curl -k -u elastic:your_password -X POST "https://localhost:9200/_security/api_key" -H 'Content-Type: application/json' -d'
{
  "name": "logstash-api-key"
}'
```
在 Logstash 配置中替换 `user`/`password`：
```ruby
elasticsearch {
  ...
  api_key => "base64-encoded-api-key"  # 例如 "VnVhW...:dFlTYk..."
}
```

---

通过以上步骤，Logstash 即可安全连接 Elasticsearch 8+ 集群。