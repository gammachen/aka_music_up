要查看视频文件（如 MP4）的编码格式是 H.264 还是 H.265，可以通过以下方法快速识别：

---

### 方法 1：使用 **FFprobe**（命令行工具，跨平台）
1. **安装 FFmpeg/FFprobe**  
   - Windows/Mac/Linux：从 [FFmpeg 官网](https://ffmpeg.org/download.html) 下载并安装。
   - macOS（Homebrew）：`brew install ffmpeg`
   - Ubuntu/Debian：`sudo apt install ffmpeg`

2. **执行命令**  
   在终端中输入：
   ```bash
   ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 input.mp4
   ```
   - **输出结果**：
     - `h264` → H.264 编码
     - `hevc` → H.265（HEVC）编码

---

### 方法 2：使用 **MediaInfo**（图形化工具，推荐新手）
1. **下载安装**  
   访问 [MediaInfo 官网](https://mediaarea.net/en/MediaInfo) 下载对应系统版本。

2. **查看编码格式**  
   - 右键点击视频文件 → 选择「MediaInfo」打开。
   - 在 **Video** 部分查找编码格式：
     - `AVC` 或 `H.264` → H.264
     - `HEVC` 或 `H.265` → H.265

---

### 方法 3：使用 **VLC 播放器**
1. 用 VLC 打开视频文件。
2. 点击菜单栏：**Tools → Codec Information**。
3. 在 **Codec** 部分查看视频编码：
   - `H264` 或 `AVC1` → H.264
   - `H265` 或 `HEVC` → H.265

---

### 方法 4：Windows 文件属性（部分文件支持）
1. 右键点击视频文件 → **属性 → 详细信息**。
2. 在 **视频** 部分查找：
   - `视频编码` 字段显示 `AVC` 或 `H.264` → H.264
   - 显示 `HEVC` 或 `H.265` → H.265  
   （注意：部分文件可能无此信息）

---

### 扩展知识：编码格式的常见标识
| **编码格式** | **标准名称**       | **常见标识符**               |
|--------------|--------------------|-----------------------------|
| H.264        | AVC (Advanced Video Coding) | `h264`, `avc1`, `x264` |
| H.265        | HEVC (High Efficiency Video Coding) | `hevc`, `hvc1` |

---

### 总结
- **推荐工具**：  
  - 技术用户 → `ffprobe`（快速精准）  
  - 普通用户 → `MediaInfo`（图形化界面）  
- **典型场景**：  
  - 检查视频是否符合平台上传要求（如 YouTube 推荐 H.264）。  
  - 确认转码后的文件编码是否正确。  
  - 解决播放兼容性问题（旧设备可能不支持 H.265）。