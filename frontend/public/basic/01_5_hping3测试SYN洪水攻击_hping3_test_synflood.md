以下是使用Docker快速构建Nginx服务并映射到本机8001端口的完整步骤：

### 1. 拉取官方Nginx镜像
```bash
docker pull nginx:latest
```

### 2. 创建并运行容器
```bash
docker run -d --name my-nginx -p 8001:80 nginx
```

### 3. 验证服务
```bash
curl http://localhost:8001
# 或在浏览器访问 http://localhost:8001
```

### 4. 测试SYN洪泛攻击
```bash
sudo hping3 -S -p 8001 --flood --rand-source 127.0.0.1
```

### 5. 监控攻击效果（新终端）
```bash
watch -n 1 "netstat -ant | grep SYN_RECV | wc -l"
```

### 关键说明：
1. **端口映射**：`-p 8001:80` 将容器内80端口映射到主机8001
2. **攻击目标**：使用`127.0.0.1`或你的本机IP
3. **防御观察**：默认情况下现代Docker网络会过滤部分异常流量

### 完整docker-compose方案（推荐）：
```yaml
version: '3'
services:
  nginx:
    image: nginx
    ports:
      - "8001:80"
```

启动：
```bash
docker-compose up -d
```

### 攻击效果验证：
- 成功时：应看到SYN_RECV连接数上升
- 无效果时：检查是否启用了SYN Cookie（`sysctl net.ipv4.tcp_syncookies`）

### 清理命令：
```bash
docker stop my-nginx && docker rm my-nginx
# 或
docker-compose down
```

这种方法比Python的http.server更适合测试SYN攻击，因为Nginx有完整的TCP协议栈实现。

