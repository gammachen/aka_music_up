以下是将视频转换为DASH流并使用dash.js播放的详细实施方案：

---

### 一、准备工作
1. 安装FFmpeg（需支持libx264和aac编码）
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg

   # MacOS
   brew install ffmpeg
   ```

2. 准备测试视频文件（如：input.mp4）

---

### 二、使用FFmpeg生成DASH流
#### 基础命令（单码率）
```bash
ffmpeg -i input.mp4 \
    -c:v libx264 -b:v 3M -g 60 -keyint_min 60 \
    -c:a aac -b:a 128k \
    -f dash \
    -seg_duration 4 \
    -use_template 1 \
    -use_timeline 1 \
    -init_seg_name init-\$RepresentationID\$.mp4 \
    -media_seg_name chunk-\$RepresentationID\$-\$Number%05d\$.mp4 \
    output.mpd
```

#### 多码率自适应流示例
```bash
ffmpeg -i input.mp4 \
    -map 0:v:0 -map 0:a:0 \
    -c:v:0 libx264 -b:v:0 800k -g 60 -keyint_min 60 \
    -c:v:1 libx264 -b:v:1 1500k -g 60 -keyint_min 60 \
    -c:a aac -b:a 128k \
    -f dash \
    -seg_duration 4 \
    -adaptation_sets "id=0,streams=v id=1,streams=a" \
    -use_template 1 \
    -use_timeline 1 \
    output.mpd
```

#### 参数说明：
- `-seg_duration`：每个分片时长（秒）
- `-g` 和 `-keyint_min`：设置GOP长度需等于分片时长的帧数
- `-use_template`：启用MPD模板化（减少文件体积）
- `-adaptation_sets`：分组视频和音频流

---

### 三、部署DASH文件
1. 生成以下文件：
   ```
   output.mpd          # 清单文件
   init-stream0.mp4    # 视频初始化段
   chunk-stream0-*.mp4 # 视频分片
   init-stream1.mp4    # 音频初始化段
   chunk-stream1-*.mp4 # 音频分片
   ```

2. 通过Web服务器部署（需配置正确MIME类型）：
   ```nginx
   location / {
       add_header Access-Control-Allow-Origin *;
       types {
           application/dash+xml mpd;
           video/mp4 mp4;
       }
   }
   ```

---

### 四、使用dash.js播放
#### 基础HTML示例
```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.dashjs.org/latest/dash.all.min.js"></script>
</head>
<body>
    <video id="videoPlayer" controls width="640" height="360"></video>
    
    <script>
        const url = "http://your-domain.com/path/output.mpd";
        const videoPlayer = document.getElementById("videoPlayer");
        const player = dashjs.MediaPlayer().create();
        
        player.initialize(videoPlayer, url, true);
        player.updateSettings({
            streaming: {
                buffer: {
                    bufferTimeAtTopQuality: 30
                }
            }
        });
    </script>
</body>
</html>
```

---

### 五、验证与调试
1. 使用浏览器开发者工具检查：
   - 确认MPD文件正确加载
   - 检查分片文件（chunk-*）是否按需加载

2. 使用`ffprobe`验证分片：
   ```bash
   ffprobe chunk-stream0-00001.mp4
   ```

---

### 六、常见问题解决
1. **跨域问题**：
   - 配置服务器返回`Access-Control-Allow-Origin: *`
   - 使用本地开发服务器（如Python快速启动）：
     ```bash
     python3 -m http.server 8000
     ```

2. **分片对齐问题**：
   - 确保GOP长度等于分片时长的帧数
   - 例如：30fps视频，4秒分片需要`-g 120`

3. **音频同步问题**：
   - 使用`-af "aresample=async=1"`参数

---

### 七、高级配置（可选）
1. 加密内容（DRM）：
   ```bash
   -encryption_scheme cenc-aes-ctr \
   -encryption_key <KEY> \
   -encryption_kid <KID>
   ```

2. 添加字幕：
   ```bash
   -map 0:s:0 -c:s webvtt
   ```

---

通过以上步骤即可实现完整的DASH流媒体服务。建议使用最新版FFmpeg（4.0+）和dash.js（4.0+）以获得最佳兼容性。