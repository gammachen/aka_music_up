以下是使用 Docker 安装和测试 Redis 的详细流程，结合了知识库中的最佳实践和常见配置：

---

### **1. 安装 Docker**
#### **步骤**：
1. **下载并安装 Docker**：
   - 访问 [Docker 官网](https://www.docker.com/) 下载 Docker Desktop。
   - 安装完成后，在终端输入以下命令验证安装：
     ```bash
     docker --version
     ```
     成功输出示例：
     ```
     Docker version 24.x.x, build xxxxx
     ```

---

### **2. 拉取 Redis 镜像**
#### **步骤**：
1. **拉取 Redis 镜像**（指定版本可避免兼容性问题）：
   ```bash
   docker pull redis:7.0.5  # 可替换为其他版本，如 redis:latest
   ```
2. **验证镜像**：
   ```bash
   docker images
   ```
   输出应包含 `redis` 镜像。

---

### **3. 创建 Redis 容器并配置**
#### **基础安装（无持久化）**：
```bash
docker run -d \
  --name redis-server \
  -p 6379:6379 \
  --restart unless-stopped \
  redis:7.0.5
```
- `-d`：后台运行容器。
- `--name redis-server`：容器名称。
- `-p 6379:6379`：映射宿主机 6379 端口到容器的 6379 端口。
- `--restart unless-stopped`：除非手动停止，否则容器自动重启。

---

### **4. 配置持久化（数据不丢失）**
#### **步骤**：
1. **创建持久化目录**：
   ```bash
   mkdir -p ~/docker/redis/{data,conf}
   ```
2. **创建 Redis 配置文件 `redis.conf`**：
   ```bash
   echo "appendonly yes" >> ~/docker/redis/conf/redis.conf
   ```
   或手动编辑 `redis.conf` 添加以下内容：
   ```conf
   # 开启 AOF 持久化
   appendonly yes
   # 设置密码（可选）
   requirepass your_password
   # 允许远程访问（可选）
   bind 0.0.0.0
   protected-mode no
   ```

3. **运行带持久化的容器**：
   ```bash
   docker run -d \
     --name redis-server \
     -p 6379:6379 \
     --restart unless-stopped \
     -v ~/docker/redis/data:/data \          # 持久化数据目录
     -v ~/docker/redis/conf/redis.conf:/etc/redis/redis.conf \  # 挂载配置文件
     redis:7.0.5 redis-server /etc/redis/redis.conf

     docker run -d \
     --name redis-server \
     -p 6379:6379 \
     --restart unless-stopped \
     -v ~/docker/redis/data:/data \
     -v ~/docker/redis/conf/redis.conf:/etc/redis/redis.conf \
     redis:latest redis-server /etc/redis/redis.conf

     docker run --rm -it --link redis-server:redis redis:latest redis-benchmark -h redis -p 6379 -c 50 -n 10000
   ```

---

### **5. 验证 Redis 容器状态**
#### **步骤**：
1. **查看运行中的容器**：
   ```bash
   docker ps
   ```
   应看到 `redis-server` 容器处于 `Up` 状态。

2. **进入容器测试 Redis**：
   ```bash
   docker exec -it redis-server redis-cli
   ```
   - 如果设置了密码，执行 `AUTH your_password`。
   - 测试键值对：
     ```bash
     SET test_key "Hello Redis"
     GET test_key
     ```
     成功返回 `Hello Redis`。

---

### **6. Redis 测试**
#### **(1) 基础测试**
```bash
# 连接 Redis
docker exec -it redis-server redis-cli -h localhost -p 6379
# 设置并获取键值
SET mykey "Hello Docker Redis"
GET mykey
```

#### **(2) 性能测试（使用 redis-benchmark）**
1. **启动性能测试容器**：
   ```bash
   docker run --rm -it --link redis-server:redis redis:7.0.5 redis-benchmark -h redis -p 6379 -c 50 -n 10000
   ```
   - `-c 50`：并发连接数 50。
   - `-n 10000`：执行 10,000 次请求。

2. **输出示例**：
   ```
   ====== SET ======
   10000 requests completed in 0.13 seconds
   50 parallel clients
   3 bytes payload
   keep alive: 1
   --- SET ---
   requests_per_second: 76923.08
   ```

---

### **7. 常见问题与解决**
#### **(1) 端口冲突**
- **问题**：端口 `6379` 被占用。
- **解决**：
  ```bash
  # 修改端口映射
  docker run ... -p 6380:6379 ...
  ```

#### **(2) 配置文件错误**
- **问题**：配置文件路径错误或权限问题。
- **解决**：
  ```bash
  # 确保配置文件路径正确且可读
  chmod 644 ~/docker/redis/conf/redis.conf
  ```

#### **(3) 持久化数据丢失**
- **问题**：容器删除后数据丢失。
- **解决**：
  ```bash
  # 挂载数据目录（如步骤4）
  -v ~/docker/redis/data:/data
  ```

---

### **8. 完整流程总结**
| **步骤**               | **命令/操作**                                                                 |
|------------------------|------------------------------------------------------------------------------|
| 安装 Docker            | 下载并安装 Docker Desktop                                                   |
| 拉取 Redis 镜像        | `docker pull redis:7.0.5`                                                    |
| 创建持久化目录         | `mkdir -p ~/docker/redis/{data,conf}`                                        |
| 配置 Redis（可选）     | 编辑 `redis.conf` 开启持久化、设置密码等                                     |
| 运行 Redis 容器        | `docker run -d -p 6379:6379 -v ... redis:7.0.5 redis-server /etc/redis/redis.conf` |
| 测试连接               | `docker exec -it redis-server redis-cli`                                     |
| 性能测试               | `docker run --rm redis redis-benchmark -h redis -p 6379`                     |

---

### **9. 扩展：使用 Docker Compose**
#### **步骤**：
1. **创建 `docker-compose.yml`**：
   ```yaml
   version: '3'
   services:
     redis:
       image: redis:7.0.5
       container_name: redis-server
       ports:
         - "6379:6379"
       volumes:
         - ~/docker/redis/data:/data
         - ~/docker/redis/conf/redis.conf:/etc/redis/redis.conf
       command: redis-server /etc/redis/redis.conf
       restart: unless-stopped
   ```

2. **启动服务**：
   ```bash
   docker-compose up -d
   ```

---

### **10. 完整知识库参考**
- **持久化配置**：参考知识库条目[1]、[3]、[7]。
- **性能测试**：参考知识库条目[9]、[10]、[12]。
- **常见问题**：参考知识库条目[8]（目录命名错误）。

通过以上步骤，你可以快速在 Docker 中搭建并测试 Redis，满足开发或生产环境的需求。