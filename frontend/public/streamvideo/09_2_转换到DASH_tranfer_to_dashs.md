---

**使用 nginx-rtmp 模块原生不支持直接生成 MPEG-DASH 流**，但可以通过以下方案实现 RTMP 到 DASH 的转换：

---

### 一、核心架构方案
```mermaid
graph LR
    A[主播端] --RTMP推流--> B[nginx-rtmp服务器]
    B --转码+切片--> C[FFmpeg处理]
    C --生成DASH分片/MPD--> D[Web服务器]
    D --HTTP分发--> E[客户端播放DASH]
```

---

### 二、详细实施步骤

#### **1. 安装必要组件**
```bash
# 安装带RTMP模块的Nginx
sudo apt install libnginx-mod-rtmp

# 安装FFmpeg（需支持libx264和dash封装）
sudo apt install ffmpeg
```

#### **2. 配置 nginx-rtmp**
在 `/etc/nginx/nginx.conf` 的 `rtmp` 块中添加：
```nginx
rtmp {
    server {
        listen 1935;
        chunk_size 4096;

        application live {
            live on;
            record off;
            
            # 关键配置：将RTMP流转发给FFmpeg处理
            exec_push ffmpeg -i rtmp://localhost:1935/live/$name 
              -c:v libx264 -preset veryfast -profile:v baseline -level 3.0 
              -c:a aac -b:a 128k 
              -f dash -window_size 5 -extra_window_size 3 
              -remove_at_exit 1 
              /var/www/dash/$name/manifest.mpd;
        }
    }
}
```

#### **3. 配置HTTP服务器（托管DASH文件）**
在Nginx的HTTP块中添加：
```nginx
server {
    listen 80;
    server_name your_domain.com;

    location /dash {
        root /var/www;
        add_header Cache-Control no-cache; # 禁用缓存确保实时性
        add_header Access-Control-Allow-Origin *; # CORS支持

        # 正确设置MIME类型
        types {
            application/dash+xml mpd;
            video/mp4 mp4;
            audio/mp4 m4a;
        }
    }
}
```

#### **4. 创建DASH输出目录**
```bash
sudo mkdir -p /var/www/dash
sudo chown -R www-data:www-data /var/www/dash
```

#### **5. 重启服务**
```bash
sudo systemctl restart nginx
```

---

### 三、FFmpeg参数详解
| **参数**                  | **作用**                                                                 |
|---------------------------|-------------------------------------------------------------------------|
| `-c:v libx264`            | 视频编码为H.264（兼容大部分DASH播放器）                                   |
| `-preset veryfast`        | 编码速度优化，降低延迟                                                   |
| `-profile:v baseline`     | 兼容低端设备的H.264配置                                                  |
| `-f dash`                 | 输出格式为DASH                                                           |
| `-window_size 5`          | 保留5个分片在清单文件中（滚动更新）                                       |
| `-remove_at_exit 1`       | 流结束时自动删除分片文件（适合直播）                                      |
| `/var/www/dash/$name/manifest.mpd` | 输出路径，`$name`对应推流URL中的流名称（如`rtmp://server/live/stream1`） |

---

### 四、客户端播放示例
使用 **dash.js** 播放生成的DASH流：
```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.dashjs.org/latest/dash.all.min.js"></script>
</head>
<body>
    <video id="player" controls></video>
    <script>
        const url = "http://your_domain.com/dash/stream1/manifest.mpd";
        const player = dashjs.MediaPlayer().create();
        player.initialize(document.querySelector("#player"), url, true);
    </script>
</body>
</html>
```

---

### 五、性能优化建议

#### **1. 多码率自适应**
修改FFmpeg命令生成多分辨率流：
```bash
exec_push ffmpeg -i rtmp://localhost:1935/live/$name 
  # 高清流
  -map 0:v:0 -c:v:0 libx264 -b:v:0 2500k -s:v:0 1280x720 
  # 标清流
  -map 0:v:0 -c:v:1 libx264 -b:v:1 1000k -s:v:1 640x360 
  # 音频流
  -map 0:a:0 -c:a aac -b:a 128k 
  -f dash -adaptation_sets "id=0,streams=v id=1,streams=a" 
  /var/www/dash/$name/manifest.mpd;
```

#### **2. 分片时长优化**
通过 `-seg_duration` 控制分片长度（默认4秒）：
```bash
-f dash -seg_duration 2 -window_size 10  # 2秒分片，保留10个分片
```

#### **3. 低延迟配置**
启用低延迟DASH参数：
```bash
-f dash -ldash 1 -streaming 1 -use_timeline 0 -utc_timing_url "https://time.akamai.com/?iso"
```

---

### 六、故障排查

#### **1. 检查DASH文件生成**
```bash
ls /var/www/dash/stream1/
# 应看到 manifest.mpd 和 chunk-*.m4s 分片文件
```

#### **2. 查看FFmpeg日志**
在nginx配置中添加 `exec_kill_signal term;` 并监控日志：
```bash
tail -f /var/log/nginx/error.log
```

#### **3. 验证MIME类型**
```bash
curl -I http://your_domain.com/dash/stream1/manifest.mpd
# 应返回 Content-Type: application/dash+xml
```

---

### 七、方案局限性
- **延迟较高**：RTMP→DASH转码+切片通常带来 **6-15秒延迟**
- **资源消耗**：实时转码需要较高CPU
- **无原生ABR**：需手动配置多码率

---

### 八、替代方案推荐
| **方案**                | **优点**                  | **缺点**                     |
|-------------------------|--------------------------|-----------------------------|
| **Wowza Streaming Engine** | 原生支持DASH/低延迟     | 商业收费                    |
| **GStreamer + SRS**     | 开源灵活                | 配置复杂                    |
| **AWS MediaLive**       | 全托管云服务            | 成本高                      |

---

通过上述配置，可以将现有的nginx-rtmp服务器扩展为支持MPEG-DASH的直播系统。尽管需要额外转码步骤，但此方案成本低且完全开源可控。