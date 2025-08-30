**SRS分布式部署** 
 
**业务端如何使用FFmpeg进行流分发**

**客户端如何建设索引服务器**：

---

## **一、业务端使用FFmpeg进行流分发**
### **1. 基础推流到SRS Origin节点**
业务端（如推流服务器）通过FFmpeg将流推送到SRS的 **Origin（源站）节点**，Origin节点负责将流分发到各个 **Edge（边缘）节点**。

#### **步骤与命令示例**
1. **推流命令**（将流推送到Origin节点）：
   ```bash
   ffmpeg -re -i input.mp4 \
     -c:v libx264 -c:a aac \
     -f flv rtmp://origin_server_ip:1935/live/stream_key
   ```
   - `-re`：以原始流速率推流（避免缓冲）。
   - `-c:v libx264`：视频编码格式。
   - `-f flv`：输出格式为FLV。

2. **Origin节点配置**（需支持分发到Edge节点）：
   ```conf
   # 在Origin节点的srs.conf中配置Edge节点列表
   vhost __defaultVhost__ {
       live {
           enable true;
       }
       http_remux {
           enabled true;
           remux rtmp;
           remux hls;
       }
       edge {
           enabled false;  # Origin节点不作为Edge
       }
       cluster {
           enabled true;
           nodes {
               edge1 edge_server1_ip:1935;  # Edge节点1地址
               edge2 edge_server2_ip:1935;  # Edge节点2地址
           }
       }
   }
   ```

3. **Edge节点配置**（从Origin拉流并分发）：
   ```conf
   # 在Edge节点的srs.conf中配置Origin地址
   vhost __defaultVhost__ {
       live {
           enable true;
       }
       cluster {
           enabled true;
           origin origin_server_ip:1935;  # 指向Origin节点
       }
   }
   ```

---

### **2. 多流分发与负载均衡**
- **自动分发**：Origin节点会自动将流推送到所有Edge节点。
- **手动分发**：通过FFmpeg同时推送到多个Edge节点（适用于简单场景）：
  ```bash
  ffmpeg -re -i input.mp4 \
    -c copy -f flv rtmp://edge1_ip/live/stream_key \
    -c copy -f flv rtmp://edge2_ip/live/stream_key
  ```

---

## **二、客户端索引服务器的建设**
### **1. 索引服务器的作用**
索引服务器用于 **集中管理流媒体资源的元数据**，提供以下功能：
- **流列表查询**：获取所有可用流的名称、状态、地址等信息。
- **负载均衡**：根据客户端位置返回最近的Edge节点地址。
- **权限控制**：验证用户权限，过滤可访问的流。

### **2. 构建步骤**
#### **步骤1：调用SRS的REST API获取流信息**
SRS提供HTTP API接口查询流状态，索引服务器需定期调用这些接口并缓存数据。

##### **关键API示例**
1. **获取所有流列表**：
   ```bash
   curl http://srs_server_ip:1985/api/v1/streams
   ```
   返回示例：
   ```json
   {
     "code": 0,
     "streams": [
       {
         "name": "stream_key",
         "vhost": "live.example.com",
         "app": "live",
         "clients": 50,
         "video": { "codec": "H264", "width": 1920 }
       }
     ]
   }
   ```

2. **获取单个流详情**：
   ```bash
   curl "http://srs_server_ip:1985/api/v1/stream?name=stream_key"
   ```

#### **步骤2：构建索引服务逻辑**
1. **数据聚合**：
   - 定期轮询所有SRS节点（Origin和Edge）的API，收集流信息。
   - 将数据存储到数据库（如MySQL、Redis）或缓存中。

2. **服务端代码示例（Python）**：
   ```python
   import requests
   from flask import Flask, jsonify

   app = Flask(__name__)

   @app.route('/api/streams', methods=['GET'])
   def get_streams():
       # 轮询所有SRS节点
       streams = []
       srs_nodes = ["http://origin_ip:1985", "http://edge1_ip:1985"]
       for node in srs_nodes:
           try:
               res = requests.get(f"{node}/api/v1/streams")
               streams.extend(res.json().get("streams", []))
           except:
               pass  # 跳过不可达节点
       return jsonify(streams)

   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=8000)
   ```

#### **步骤3：客户端调用索引服务**
客户端通过索引服务器的API获取流信息，并选择最优节点拉流。

##### **客户端流程示例**：
1. **请求索引服务**：
   ```bash
   curl http://index_server_ip:8000/api/streams
   ```

2. **解析响应并选择Edge节点**：
   - 根据客户端IP地理位置，选择延迟最低的Edge节点。
   - 返回拉流地址（如 `rtmp://edge1_ip/live/stream_key`）。

---

## **三、分布式部署的扩展建议**
### **1. 负载均衡与调度**
- **DNS调度**：通过DNS解析将客户端路由到最近的Edge节点（如AWS Route 53的延迟路由）。
- **动态权重**：根据Edge节点负载动态调整流量分配。

### **2. 实时性保障**
- **WebSocket推送**：索引服务器通过WebSocket实时推送流状态变化（如流断开或新增流）。
- **缓存策略**：设置流元数据的TTL（如10秒），避免频繁轮询。

### **3. 容灾与高可用**
- **多Region部署**：在不同地域部署Origin和Edge节点，结合Forward集群实现跨区域容灾。
- **自动故障转移**：当主Origin节点故障时，索引服务器切换到备用节点。

---

## **四、完整部署架构示例**
```
+-------------------+
|  推流服务器       |
|  (FFmpeg)         |
|  → 推流到Origin   |
+-------▲----------+
        │
+-------▼----------+
|  Origin节点       |
|  (SRS)           |
|  → 分发到Edge1   |
|  → 分发到Edge2   |
+-------▲----------+
        │
+-------▼----------+
|  索引服务器       |
|  (HTTP API聚合)   |
|  ← 客户端查询     |
+-------▲----------+
        │
+-------▼----------+
|  Edge节点1        |
|  (SRS)           |
|  ← 客户端拉流     |
+-------------------+
        │
+-------▼----------+
|  Edge节点2        |
|  (SRS)           |
|  ← 客户端拉流     |
+-------------------+
```

---

通过以上步骤，可以实现 **业务端通过FFmpeg推流到SRS集群**，并借助 **索引服务器** 管理流资源，最终实现分布式流媒体服务的高效分发和客户端访问。如果需要更具体的代码示例或配置细节，请进一步说明！