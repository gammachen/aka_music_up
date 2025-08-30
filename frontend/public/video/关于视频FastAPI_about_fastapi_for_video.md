以下是使用 **FastAPI** 快速搭建一个 Web 页面访问 MP4 文件的完整步骤，包含静态文件服务、视频播放界面和 API 端点：

---

### **一、环境准备**
1. **安装 FastAPI 和 uvicorn**：
   ```bash
   pip install fastapi uvicorn
   ```

2. **创建项目目录结构**：
   ```
   project/
   ├── main.py          # FastAPI 主文件
   ├── static/          # 存放 MP4 文件的目录
   │   └── videos/      # 存放 MP4 文件的子目录（如 video1.mp4, video2.mp4）
   └── templates/       # 存放 HTML 模板
       └── index.html   # 视频播放页面
   ```

---

### **二、编写 HTML 播放页面**
在 `templates/index.html` 中创建一个简单的视频播放界面：
```html
<!DOCTYPE html>
<html>
<head>
    <title>Video Player</title>
</head>
<body>
    <h1>视频播放器</h1>
    <video id="videoPlayer" width="640" height="360" controls>
        <source src="/static/videos/video1.mp4" type="video/mp4">
        您的浏览器不支持视频播放。
    </video>
    <br>
    <select id="videoSelect" onchange="loadVideo()">
        <option value="video1.mp4">视频1</option>
        <option value="video2.mp4">视频2</option>
        <!-- 动态生成选项（需后端支持） -->
    </select>
    <script>
        function loadVideo() {
            const selectedVideo = document.getElementById('videoSelect').value;
            document.getElementById('videoPlayer').src = `/static/videos/${selectedVideo}`;
        }
    </script>
</body>
</html>
```

---

### **三、编写 FastAPI 代码**
在 `main.py` 中配置静态文件服务和路由：
```python
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os

app = FastAPI()

# 挂载静态文件目录（存放 MP4 文件）
app.mount("/static", StaticFiles(directory="static"), name="static")

# 渲染视频播放页面
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>视频列表</title>
        </head>
        <body>
            <h1>可播放的视频列表</h1>
            <ul>
                <!-- 动态生成视频链接 -->
                <li><a href="/play/video1.mp4">视频1</a></li>
                <li><a href="/play/video2.mp4">视频2</a></li>
            </ul>
            <p>直接访问视频： <a href="/static/videos/video1.mp4">video1.mp4</a></p>
        </body>
    </html>
    """

# 动态视频播放页面（示例）
@app.get("/play/{video_name}", response_class=HTMLResponse)
async def play_video(video_name: str):
    return f"""
    <html>
        <body>
            <video width="640" height="360" controls>
                <source src="/static/videos/{video_name}" type="video/mp4">
                您的浏览器不支持视频播放。
            </video>
        </body>
    </html>
    """

# 可选：列出所有视频文件（API 端点）
@app.get("/api/videos")
async def list_videos():
    video_dir = Path("static/videos")
    videos = [f.name for f in video_dir.glob("*.mp4")]
    return {"videos": videos}
```

---

### **四、运行服务**
1. **启动 FastAPI 服务**：
   ```bash
   uvicorn main:app --reload
   ```

2. **访问页面**：
   - 浏览器访问：`http://localhost:8000`（主页面，显示视频列表）。
   - 直接访问视频：`http://localhost:8000/static/videos/video1.mp4`。
   - 使用动态播放页面：`http://localhost:8000/play/video1.mp4`。

---

### **五、功能扩展（可选）**
#### **1. 自动列出视频文件**
修改 `main.py`，动态生成视频链接：
```python
@app.get("/", response_class=HTMLResponse)
async def read_root():
    video_dir = Path("static/videos")
    videos = [f.name for f in video_dir.glob("*.mp4")]
    video_links = "\n".join([f'<li><a href="/play/{video}">{video}</a></li>' for video in videos])
    return f"""
    <html>
        <body>
            <h1>可播放的视频列表</h1>
            <ul>
                {video_links}
            </ul>
        </body>
    </html>
    """
```

#### **2. 文件上传功能**
添加上传接口（需 HTML 表单支持）：
```python
from fastapi import File, UploadFile

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    save_path = f"static/videos/{file.filename}"
    with open(save_path, "wb") as buffer:
        buffer.write(await file.read())
    return {"filename": file.filename, "status": "上传成功"}
```

在 `index.html` 中添加上传表单：
```html
<form action="/upload/" enctype="multipart/form-data" method="post">
    <input type="file" name="file">
    <input type="submit">
</form>
```

---

### **六、注意事项**
1. **文件路径安全**：
   - 使用 `pathlib.Path` 验证文件路径，防止路径遍历攻击。
   - 确保 `static/videos` 目录存在且有写入权限。

2. **性能优化**：
   - 大文件建议使用流式传输（`StreamingResponse`）。
   ```python
   from fastapi.responses import StreamingResponse

   @app.get("/stream/{video_name}")
   async def stream_video(video_name: str):
       file_path = f"static/videos/{video_name}"
       return StreamingResponse(
           iter_file(file_path),
           media_type="video/mp4",
           headers={"Content-Disposition": f"attachment; filename={video_name}"}
       )

   def iter_file(file_path):
       with open(file_path, "rb") as f:
           chunk_size = 1024 * 1024  # 1MB
           while True:
               chunk = f.read(chunk_size)
               if not chunk:
                   break
               yield chunk
   ```

3. **浏览器兼容性**：
   - 确保视频格式为 `MP4`（H.264 编码），兼容性最佳。

---

### **七、完整示例**
1. **目录结构**：
   ```
   project/
   ├── main.py
   ├── static/
   │   └── videos/
   │       ├── video1.mp4
   │       └── video2.mp4
   └── templates/
       └── index.html
   ```

2. **启动后效果**：
   - 访问 `http://localhost:8000`，显示视频列表。
   - 点击链接直接播放视频。
   - 通过 `/api/videos` 获取视频列表的 JSON 数据。

通过以上步骤，你可以快速搭建一个支持 MP4 文件访问和播放的 Web 服务。