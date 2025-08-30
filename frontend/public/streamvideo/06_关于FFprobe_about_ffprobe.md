以下是 `ffprobe` 的 **详细用法与典型场景**，涵盖视频分析、元数据提取、流媒体诊断等常见需求：

---

### 一、基础用法：查看文件基本信息
#### 1. 显示 **容器格式与流信息**
```bash
ffprobe input.mp4
```
**输出示例**：
```yaml
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'input.mp4':
  Duration: 00:05:23.15, start: 0.000000, bitrate: 1500 kb/s
  Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p, 1280x720 [SAR 1:1 DAR 16:9], 1200 kb/s, 30 fps, 30 tbr, 15360 tbn, 60 tbc (default)
  Stream #0:1(eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 192 kb/s (default)
```

**场景**：  
- 快速了解视频的编码格式、分辨率、时长、码率、帧率、音频参数等核心信息。

---

### 二、高级用法：精准提取元数据
#### 1. 获取 **视频流详细信息**
```bash
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,bit_rate,r_frame_rate,duration -of default=noprint_wrappers=1 input.mp4
```
**输出**：
```bash
codec_name=h264
width=1280
height=720
bit_rate=1200000
r_frame_rate=30/1
duration=323.150000
```

**参数解析**：
- `-v error`：隐藏非错误信息
- `-select_streams v:0`：选择第一个视频流
- `-show_entries stream=...`：指定要显示的字段
- `-of default=noprint_wrappers=1`：简化输出格式

---

#### 2. 获取 **音频流参数**
```bash
ffprobe -v error -select_streams a:0 -show_entries stream=codec_name,sample_rate,channels,bit_rate -of default=noprint_wrappers=1 input.mp4
```
**输出**：
```bash
codec_name=aac
sample_rate=44100
channels=2
bit_rate=192000
```

**场景**：  
- 自动化脚本中提取音视频参数用于转码配置
- 验证文件是否符合特定规格（如平台上传要求）

---

### 三、深度分析：帧级别信息
#### 1. 统计 **关键帧（I帧）间隔**
```bash
ffprobe -v error -select_streams v:0 -show_frames -of csv input.mp4 | grep -n "pict_type=I"
```
**输出**：
```csv
50,0,I,,,,,,,,,,,,,,
150,0,I,,,,,,,,,,,,,,
250,0,I,,,,,,,,,,,,,,
```
**场景**：  
- 分析 GOP（画面组）结构，优化直播流的关键帧间隔
- 排查视频跳转时的卡顿问题

---

#### 2. 导出 **所有帧的详细信息（JSON格式）**
```bash
ffprobe -v error -show_frames -of json input.mp4 > frames.json
```
**输出片段**：
```json
{
  "frames": [
    {
      "media_type": "video",
      "key_frame": 1,
      "pkt_pts": 0,
      "pkt_dts": 0,
      "pict_type": "I",
      "width": 1280,
      "height": 720
    },
    ...
  ]
}
```

**场景**：  
- 开发视频分析工具时获取结构化数据
- 检测视频中的异常帧（如分辨率突变）

---

### 四、流媒体诊断
#### 1. 检查 **实时流可用性**
```bash
ffprobe -timeout 5000000 -i rtmp://example.com/live/stream
```
**参数**：
- `-timeout`：设置超时时间（微秒）

**场景**：  
- 验证 RTMP/HLS 流是否可访问
- 监控直播流状态

---

#### 2. 获取 **HLS 分片信息**
```bash
ffprobe -v error -show_packets -of json playlist.m3u8
```
**输出**：显示每个 TS 分片的时长、大小、偏移量等。

**场景**：  
- 分析 HLS 流的切片合理性
- 排查卡顿或加载缓慢问题

---

### 五、实用技巧
#### 1. **批量分析文件夹内所有视频**
```bash
for file in *.mp4; do
  ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1 "$file"
done
```
**输出**：输出每个视频的时长。

---

#### 2. **结合 jq 处理 JSON 输出**
```bash
ffprobe -v error -show_streams -of json input.mp4 | jq '.streams[0] | {codec: .codec_name, width: .width, height: .height}'
```
**输出**：
```json
{
  "codec": "h264",
  "width": 1280,
  "height": 720
}
```

---

### 六、完整参数速查表
| **参数**                | **作用**                                 |
|-------------------------|-----------------------------------------|
| `-show_format`          | 显示容器格式信息（时长、码率、格式）     |
| `-show_streams`         | 显示所有流的详细信息                     |
| `-show_frames`          | 显示帧级别的详细信息                     |
| `-show_packets`         | 显示数据包信息（时间戳、大小）           |
| `-select_streams v:0`   | 选择第一个视频流                         |
| `-of json/xml/csv`      | 指定输出格式（默认文本）                 |
| `-hide_banner`          | 隐藏 FFmpeg 版本信息                     |

---

### 七、典型场景总结
1. **转码前分析**  
   - 获取原始视频参数（编码格式、分辨率、帧率）以确定转码策略

2. **自动化脚本集成**  
   - 提取元数据生成报告或配置其他工具

3. **流媒体故障排查**  
   - 检查流是否可访问、分析分片结构

4. **视频质量评估**  
   - 统计关键帧间隔、检测异常帧

5. **多媒体资产管理**  
   - 批量提取视频属性建立数据库

---

掌握 `ffprobe` 的灵活用法，您将能高效完成视频分析、转码优化和流媒体调试等任务。

