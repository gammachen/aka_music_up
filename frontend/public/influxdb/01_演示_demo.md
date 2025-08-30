分步骤设计一个使用 **InfluxDB** 存储博物馆展厅人流数据的完整方案，包括数据模型设计、Docker部署、Python模拟数据输入与读取，以及简单的数据展示页面。

---

### **场景背景**
博物馆需要实时监控各展厅的人流量，以便优化参观路线、管理人流高峰。通过摄像头或传感器收集数据，使用 **InfluxDB** 存储时间序列数据，并通过 Python 模拟数据输入与读取，最终展示实时人流趋势。

---

### **1. 数据模型设计**
#### **InfluxDB 数据模型**
InfluxDB 的数据模型包含以下核心概念：
- **Measurement**：类似表名，例如 `exhibition_traffic`。
- **Tags**：元数据（不可变），用于快速过滤和分组，例如 `exhibition_id`（展厅ID）、`location`（区域）。
- **Fields**：数值型数据（可变），例如 `current_people`（当前人数）、`temperature`（温度）。
- **Timestamp**：时间戳（自动或手动添加）。

#### **示例**

![museum_monitor.gif](museum_monitor.gif)

#### **示例数据结构**
```text
exhibition_traffic,exhibition_id=exhibition_001,location=gallery_A current_people=50,temperature=22.5 1702301400000000000
exhibition_traffic,exhibition_id=exhibition_002,location=gallery_B current_people=30,temperature=23.0 1702301400000000000
```

---

### **2. 使用 Docker 搭建 InfluxDB**
#### **步骤 1：拉取 InfluxDB 镜像**
```bash
docker pull influxdb:latest
```

#### **步骤 2：运行 InfluxDB 容器**
```bash
docker run -d \
  --name influxdb \
  -p 8086:8086 \
  -v influxdb_data:/var/lib/influxdb \
  influxdb:latest
```
- `-v influxdb_data:/var/lib/influxdb`：挂载数据卷，持久化存储数据。
- 访问 InfluxDB UI：`http://localhost:8086`（默认用户名/密码：`admin`/`admin`）。

#### **步骤 3：创建数据库和用户**
进入 InfluxDB UI 或使用命令行：
```bash
docker exec -it influxdb influx
> CREATE DATABASE exhibition_db
> CREATE USER "exhibition_user" WITH PASSWORD 'exhibition_pass' WITH ALL PRIVILEGES

TODO:
这里使用了通过UI创建用户，console下没有成功（401，权限问题）
```

---

### **3. Python 模拟数据输入**
#### **安装依赖**
```bash
pip install influxdb-client pandas matplotlib flask
```

#### **数据生成脚本 (`generate_data.py`)**
```python
from flask import Flask, render_template, jsonify
from influxdb_client import InfluxDBClient, QueryApi
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# 连接 InfluxDB
client = InfluxDBClient(
    url="http://localhost:8086",
    token="iXKclTXCS-sTHoP2QgUt--UtVbJHog9bs5zjYgD16uNozmD36cDvuwUH_tWQ_F9CAfhh6QmiryV31hSPtuD_3g==",
    org="csd"
)
bucket = "csd"
query_api = client.query_api()

@app.route('/')
def index():
    # 查询最近10分钟的实时人流数据
    query = f'''
    from(bucket: "csd")
      |> range(start: -1m)
      |> filter(fn: (r) => r._measurement == "exhibition_traffic")
    '''
    tables = query_api.query(query)
    
    data = []
    for table in tables:
        for record in table.records:
            data.append({
                "exhibition_id": record["exhibition_id"],
                "location": record["location"],
                "current_people": record.get_value(),
                "time": record.get_time()
            })
    
    return render_template('index.html', data=data)

@app.route('/api/traffic_data')
def get_traffic_data():
    # 查询最近1分钟的实时人流数据
    query = f'''
    from(bucket: "csd")
      |> range(start: -1m)
      |> filter(fn: (r) => r._measurement == "exhibition_traffic")
    '''
    tables = query_api.query(query)
    
    data = []
    for table in tables:
        for record in table.records:
            data.append({
                "exhibition_id": record["exhibition_id"],
                "location": record["location"],
                "current_people": record.get_value(),
                "time": record.get_time().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)

```

#### **运行脚本**
```bash
python generate_data.py
```

---

### **4. 数据读取与展示**
#### **创建 Flask 简单 Web 页面 (`app.py`)**
```python
<!DOCTYPE html>
<html>
<head>
    <title>博物馆人流监控</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- 引入html2canvas和gif.js库 -->
    <script src="https://html2canvas.hertzen.com/dist/html2canvas.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
            padding: 20px;
        }
        .header {
            background-color: #343a40;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
        }
        .data-card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
            transition: transform 0.3s;
        }
        .data-card:hover {
            transform: translateY(-5px);
        }
        .data-table {
            width: 100%;
            border-collapse: collapse;
        }
        .data-table th {
            background-color: #007bff;
            color: white;
            padding: 12px;
            text-align: left;
        }
        .data-table td {
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
        }
        .data-table tr:hover {
            background-color: #f1f1f1;
        }
        .refresh-info {
            text-align: center;
            color: #6c757d;
            margin-top: 10px;
            font-size: 0.9rem;
        }
        .people-count {
            font-weight: bold;
            color: #007bff;
        }
        .crowd-indicator {
            display: inline-block;
            margin-left: 5px;
        }
        .crowd-indicator i {
            color: #007bff;
            margin-right: 2px;
        }
        .icon-column {
            margin-right: 5px;
            color: #555;
        }
        .location-badge {
            background-color: #28a745;
            color: white;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
        }
        .last-updated {
            font-style: italic;
            color: #6c757d;
            font-size: 0.8rem;
        }
        .export-btn {
            background-color: #6c757d;
            color: white;
            margin-left: 10px;
        }
        .export-btn:hover {
            background-color: #5a6268;
            color: white;
        }
        .gif-progress {
            display: none;
            margin-top: 10px;
            text-align: center;
        }
        .progress {
            height: 10px;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>博物馆实时人流监控</h1>
            <p>实时展示各展厅的人流量数据</p>
        </div>
        
        <div class="data-card">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h2>实时人流数据</h2>
                <div>
                    <button id="refresh-btn" class="btn btn-primary">手动刷新</button>
                    <button id="export-gif-btn" class="btn export-btn"><i class="fas fa-camera"></i> 导出GIF</button>
                </div>
            </div>
            <div class="table-responsive">
                <table class="data-table" id="traffic-table">
                    <thead>
                        <tr>
                            <th><i class="fas fa-landmark icon-column"></i> 展厅ID</th>
                            <th><i class="fas fa-map-marker-alt icon-column"></i> 位置</th>
                            <th><i class="fas fa-users icon-column"></i> 当前人数</th>
                            <th><i class="far fa-clock icon-column"></i> 时间</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for item in data %}
                        <tr>
                            <td><i class="fas fa-landmark icon-column"></i> {{ item.exhibition_id }}</td>
                            <td><i class="fas fa-map-marker-alt icon-column"></i> <span class="location-badge">{{ item.location }}</span></td>
                            <td>
                                <i class="fas fa-users icon-column"></i> <span class="people-count">{{ item.current_people }}</span> 人
                                <span class="crowd-indicator">
                                    {% set crowd_percentage = (item.current_people / 200) * 100 %}
                                    {% if crowd_percentage > 80 %}
                                        <i class="fas fa-user"></i><i class="fas fa-user"></i><i class="fas fa-user"></i><i class="fas fa-user"></i><i class="fas fa-user"></i>
                                    {% elif crowd_percentage > 60 %}
                                        <i class="fas fa-user"></i><i class="fas fa-user"></i><i class="fas fa-user"></i><i class="fas fa-user"></i>
                                    {% elif crowd_percentage > 40 %}
                                        <i class="fas fa-user"></i><i class="fas fa-user"></i><i class="fas fa-user"></i>
                                    {% elif crowd_percentage > 20 %}
                                        <i class="fas fa-user"></i><i class="fas fa-user"></i>
                                    {% else %}
                                        <i class="fas fa-user"></i>
                                    {% endif %}
                                </span>
                            </td>
                            <td><i class="far fa-clock icon-column"></i> <span class="last-updated">{{ item.time }}</span></td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            <div class="refresh-info" id="last-update">上次更新时间: <span id="update-time"></span></div>
            <div class="gif-progress" id="gif-progress">
                <p>正在生成GIF，请稍候... <span id="gif-progress-text">0%</span></p>
                <div class="progress">
                    <div id="gif-progress-bar" class="progress-bar" role="progressbar" style="width: 0%" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // 初始化页面时更新时间
        document.addEventListener('DOMContentLoaded', function() {
            updateLastRefreshTime();
            // 每1秒自动刷新一次数据
            setInterval(fetchTrafficData, 1000);
        });

        // 手动刷新按钮
        document.getElementById('refresh-btn').addEventListener('click', fetchTrafficData);

        // 获取最新数据
        function fetchTrafficData() {
            fetch('/api/traffic_data')
                .then(response => response.json())
                .then(data => {
                    updateTableData(data);
                    updateLastRefreshTime();
                })
                .catch(error => console.error('获取数据失败:', error));
        }

        // 更新表格数据
        function updateTableData(data) {
            const tableBody = document.querySelector('#traffic-table tbody');
            tableBody.innerHTML = '';
            
            data.forEach(item => {
                const row = document.createElement('tr');
                // 计算拥挤度图标
                const crowdPercentage = (item.current_people / 200) * 100;
                let crowdIcons = '';
                
                if (crowdPercentage > 80) {
                    crowdIcons = '<i class="fas fa-user"></i><i class="fas fa-user"></i><i class="fas fa-user"></i><i class="fas fa-user"></i><i class="fas fa-user"></i>';
                } else if (crowdPercentage > 60) {
                    crowdIcons = '<i class="fas fa-user"></i><i class="fas fa-user"></i><i class="fas fa-user"></i><i class="fas fa-user"></i>';
                } else if (crowdPercentage > 40) {
                    crowdIcons = '<i class="fas fa-user"></i><i class="fas fa-user"></i><i class="fas fa-user"></i>';
                } else if (crowdPercentage > 20) {
                    crowdIcons = '<i class="fas fa-user"></i><i class="fas fa-user"></i>';
                } else {
                    crowdIcons = '<i class="fas fa-user"></i>';
                }
                
                row.innerHTML = `
                    <td><i class="fas fa-landmark icon-column"></i> ${item.exhibition_id}</td>
                    <td><i class="fas fa-map-marker-alt icon-column"></i> <span class="location-badge">${item.location}</span></td>
                    <td>
                        <i class="fas fa-users icon-column"></i> <span class="people-count">${item.current_people}</span> 人
                        <span class="crowd-indicator">${crowdIcons}</span>
                    </td>
                    <td><i class="far fa-clock icon-column"></i> <span class="last-updated">${item.time}</span></td>
                `;
                tableBody.appendChild(row);
            });
        }

        // 更新最后刷新时间
        function updateLastRefreshTime() {
            const now = new Date();
            const timeString = now.toLocaleTimeString();
            document.getElementById('update-time').textContent = timeString;
        }

        // GIF导出功能
        document.getElementById('export-gif-btn').addEventListener('click', exportGif);

        function exportGif() {
            // 显示进度条
            const progressDiv = document.getElementById('gif-progress');
            const progressBar = document.getElementById('gif-progress-bar');
            const progressText = document.getElementById('gif-progress-text');
            progressDiv.style.display = 'block';
            progressBar.style.width = '0%';
            progressText.textContent = '0%';

            // 获取需要捕获的元素
            const container = document.querySelector('.container');
            
            // 创建GIF，设置正确的尺寸
            const gif = new GIF({
                workers: 2,
                quality: 5,  // 降低quality以减小文件大小
                width: container.offsetWidth,
                height: container.offsetHeight,
                workerScript: '/static/js/gif.worker.js'
            });

            // 进度更新
            gif.on('progress', function(p) {
                const percent = Math.round(p * 100);
                progressBar.style.width = percent + '%';
                progressBar.setAttribute('aria-valuenow', percent);
                progressText.textContent = percent + '%';
            });

            // GIF生成完成
            gif.on('finished', function(blob) {
                // 创建下载链接
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = '博物馆人流监控_' + new Date().toISOString().replace(/[:.]/g, '-') + '.gif';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);

                // 隐藏进度条
                progressDiv.style.display = 'none';
            });

            // 捕获10秒的画面，每秒1帧
            let framesCount = 0;
            const totalFrames = 10;
            const captureInterval = setInterval(function() {
                // 使用更完整的html2canvas配置
                html2canvas(container, {
                    logging: false,
                    allowTaint: true,
                    useCORS: true,
                    scale: 1,  // 确保使用1:1的比例
                    width: container.offsetWidth,
                    height: container.offsetHeight,
                    scrollX: window.scrollX,
                    scrollY: window.scrollY,
                    windowWidth: document.documentElement.offsetWidth,
                    windowHeight: document.documentElement.offsetHeight,
                    foreignObjectRendering: false,  // 禁用foreignObject渲染以提高兼容性
                    removeContainer: true  // 移除临时创建的容器
                }).then(canvas => {
                    gif.addFrame(canvas, {delay: 1000, copy: true});
                    framesCount++;
                    
                    if (framesCount >= totalFrames) {
                        clearInterval(captureInterval);
                        gif.render();
                    }
                });
            }, 1000);
        }
    </script>
</body>
</html>
```

#### **运行 Flask 应用**
```bash
python app.py
```
访问 `http://localhost:5000`，即可看到实时数据展示。

---

### **5. 完整流程总结**
1. **部署 InfluxDB**：通过 Docker 快速搭建数据库。
2. **数据模拟**：使用 Python 脚本生成模拟人流数据。
3. **数据查询与展示**：通过 Flask 实现简单 Web 界面，展示实时数据。

---

### **扩展建议**
- **可视化增强**：使用 `Plotly` 或 `Matplotlib` 生成动态图表。
- **告警功能**：当某展厅人数超过阈值时触发告警。
- **持久化存储**：通过 InfluxDB 的 `Continuous Queries` 或 `Retention Policies` 管理历史数据。

通过以上步骤，你可以快速搭建一个博物馆人流监控系统，实时追踪各展厅的人流变化。