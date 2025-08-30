根据您的要求，以下是基于Docker搭建Nginx反向代理来负载均衡两个RTMP服务器的详细步骤：

### 步骤 1：创建Nginx配置文件
创建名为`nginx.conf`的配置文件，内容如下：

```nginx
user  nginx;
worker_processes  auto;

error_log  /var/log/nginx/error.log warn;
pid        /var/run/nginx.pid;

events {
    worker_connections  1024;
}

stream {
    upstream rtmp_servers {
        server 192.168.31.109:1935 weight=3;
        server 192.168.31.109:1936 weight=2;
    }

    server {
        listen 1934;
        proxy_pass rtmp_servers;
    }
}
```

**说明**：
- 使用`stream`模块进行四层TCP负载均衡。
- 监听`1934`端口，将请求转发到后端的1935和1936端口，权重分别为3和2。

### 步骤 2：启动Nginx反向代理容器
执行以下Docker命令：

```bash
docker run -d --name nginx-rtmp-proxy \
  -p 1934:1934 \
  -v /path/to/nginx.conf:/etc/nginx/nginx.conf \
  nginx:latest
```

**参数解释**：
- `-p 1934:1934`：将容器内的1934端口映射到宿主机。
- `-v`：挂载自定义的Nginx配置文件。

### 步骤 3：验证容器运行状态
检查容器日志以确保无错误：

```bash
docker logs nginx-rtmp-proxy
```

### 步骤 4：测试RTMP流代理
使用RTMP客户端推流到Nginx代理的1934端口：

```bash
# 推流示例
ffmpeg -re -i input.mp4 -c:v libx264 -c:a aac -f flv rtmp://192.168.31.109:1934/live/streamkey
```

观察后端RTMP服务器是否接收到流。

### 注意事项
1. **网络访问**：确保Nginx容器能访问宿主机的1935和1936端口。若RTMP服务运行在宿主机，需允许容器访问宿主机网络。
2. **权重分配**：四层负载均衡基于连接而非请求，权重决定新连接分配到后端服务器的比例。
3. **真实IP**：后端服务器看到的客户端IP将是Nginx容器的IP。如需真实IP，需配置`proxy_protocol`，但需RTMP服务器支持。

### 配置优化（可选）
如需应用层代理或流转发，可使用Nginx的RTMP模块配置：

```nginx
rtmp {
    server {
        listen 1935;
        application live {
            live on;
            # 推送到两个后端服务器（复制流）
            push rtmp://192.168.31.109:1935/live;
            push rtmp://192.168.31.109:1936/live;
        }
    }
}
```

此配置会将同一流转发到两个后端服务器，适用于流冗余而非负载均衡。需使用支持RTMP模块的镜像（如`tiangolo/nginx-rtmp`）：

```bash
docker run -d --name nginx-rtmp \
  -p 1935:1935 \
  -v /path/to/nginx.conf:/etc/nginx/nginx.conf \
  tiangolo/nginx-rtmp
```

### 总结
通过上述步骤，您已成功部署了一个Nginx反向代理，实现RTMP流的四层负载均衡。客户端连接到`1934`端口，请求将按权重分发至后端的1935和1936端口。根据实际需求选择四层或应用层代理方案。

