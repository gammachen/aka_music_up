以下是一个完整的解决方案，用于对收集到的 `SHOW PROCESSLIST` 日志进行可视化分析，包含脚本和操作步骤：

---

### 一、日志解析与结构化处理脚本
将原始日志转换为结构化数据（CSV格式），便于后续分析。

#### 1. 解析脚本 `parse_processlist.py`
```python
import re
import csv
from datetime import datetime

INPUT_LOG = "/var/log/mysql_processlist/processlist.log"
OUTPUT_CSV = "processlist_parsed.csv"

# 修正后的正则表达式（适配制表符分隔的原始输出）
PATTERN = re.compile(
    r"^(\d+)\t"          # Id
    r"(\S+)\t"           # User
    r"([^\t]+)\t"        # Host（允许包含冒号和端口）
    r"([^\t]*)\t"        # db（可能为NULL）
    r"(\w+)\t"           # Command
    r"(\d+)\t"           # Time
    r"([^\t]*)\t"        # State（可能为空）
    r"([^\t]*)"          # Info（可能为NULL或SQL语句）
)

def parse_log(input_file, output_csv):
    with open(input_file, 'r') as f_in, open(output_csv, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow([
            "timestamp", "id", "user", "host", "db", 
            "command", "time_sec", "state", "query"
        ])
        
        # 跳过标题行
        next(f_in)
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            # 正则匹配
            match = PATTERN.match(line)
            if match:
                # 提取字段（共8个）
                (
                    id_, user, host, db, command, 
                    time_sec, state, query
                ) = match.groups()
                
                # 处理 NULL 值
                db = db if db != "NULL" else None
                query = query if query != "NULL" else None
                
                # 写入CSV
                writer.writerow([
                    datetime.now().isoformat(),  # 假设日志时间即采集时间
                    id_, user, host, db,
                    command, time_sec, state, query
                ])
            else:
                print(f"无法解析行: {line}")

if __name__ == "__main__":
    parse_log(INPUT_LOG, OUTPUT_CSV)
```

#### 2. 运行脚本
```bash
python3 parse_processlist.py
```

---

### 二、数据可视化脚本
使用 Python 的 `pandas` 和 `plotly` 生成交互式图表。

#### 1. 安装依赖
```bash
pip3 install pandas plotly
```

#### 2. 可视化脚本 `visualize_processlist.py`
```python
import pandas as pd
import plotly.express as px
import plotly.subplots as sp

# 读取解析后的CSV数据
df = pd.read_csv("processlist_parsed.csv")

# 1. 按用户统计活跃连接数
user_counts = df['user'].value_counts().reset_index()
user_counts.columns = ['user', 'count']
fig1 = px.bar(user_counts, x='user', y='count', title="活跃连接数按用户分布")

# 2. 按状态统计查询分布
state_counts = df['state'].value_counts().reset_index()
state_counts.columns = ['state', 'count']
fig2 = px.pie(state_counts, values='count', names='state', title="查询状态分布")

# 3. 长查询时间序列（筛选 time_sec > 5s）
df['timestamp'] = pd.to_datetime(df['timestamp'])
long_queries = df[df['time_sec'] > 5]
fig3 = px.scatter(
    long_queries, x='timestamp', y='time_sec', 
    color='user', hover_data=['query'],
    title="长查询时间分布 (>5s)"
)

# 4. 按数据库统计查询类型
db_command = df.groupby(['db', 'command']).size().reset_index(name='count')
fig4 = px.sunburst(
    db_command, path=['db', 'command'], values='count',
    title="数据库与操作类型分布"
)

# 保存为HTML报告
with open("processlist_report.html", "w") as f:
    f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write(fig2.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write(fig3.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write(fig4.to_html(full_html=False, include_plotlyjs='cdn'))
```

#### 3. 运行脚本
```bash
python3 visualize_processlist.py
```

#### 4. 查看结果
打开生成的 `processlist_report.html`，浏览器中会显示交互式图表，支持以下分析：
- **用户活动分布**：识别高负载用户。
- **查询状态占比**：发现锁等待或复制状态异常。
- **长查询时间线**：定位性能瓶颈时段。
- **数据库操作全景**：观察各库的读写比例。

---

### 三、自动化集成方案
#### 1. 定时任务 + 邮件报告
```bash
# 每日生成报告并发送邮件（需配置邮件服务）
0 0 * * * /usr/bin/python3 /path/to/visualize_processlist.py && mutt -a processlist_report.html -s "MySQL Processlist Report" admin@example.com < /dev/null
```

#### 2. 集成到 Grafana
1. **数据存储**：将解析后的 CSV 导入到 InfluxDB 或 PostgreSQL。
   ```bash
   # 示例：使用 CSV 导入到 PostgreSQL
   psql -c "COPY processlist_data FROM '/path/to/processlist_parsed.csv' DELIMITER ',' CSV HEADER;"
   ```

2. **Grafana 仪表盘**：
   - 配置数据源连接到数据库。
   - 创建面板展示关键指标：
     - 实时活跃连接数
     - 长查询比例
     - 按状态的查询分布

   ![Grafana 示例面板](https://example.com/grafana-screenshot.png)

---

### 四、高级分析：结合 `pt-query-digest`
```bash
# 1. 生成慢查询摘要报告
pt-query-digest --type processlist /var/log/mysql_processlist/*.log > slow_queries_report.txt

# 2. 提取TOP 10耗时查询
cat slow_queries_report.txt | grep -A 10 "Overall summary"
```

---

### 五、注意事项
1. **日志轮转**：定期清理旧日志，避免磁盘占满。
2. **敏感信息过滤**：在解析时剔除含敏感信息的查询（如`password`）。
   ```python
   # 在 parse_processlist.py 中添加过滤逻辑
   if "password" in query.lower():
       continue
   ```
3. **性能优化**：处理大规模日志时使用分块读取（`pandas.read_csv(chunksize=1000)`）。

通过上述脚本和方案，您可以快速将 `SHOW PROCESSLIST` 日志转化为直观的可视化报告，并集成到现有监控系统中。