### 一、`SHOW PROCESSLIST` 的使用与数据收集
`SHOW PROCESSLIST` 是 MySQL 中查看实时线程状态的核心命令，可用于以下场景：
- **实时监控**：查看当前活跃连接和查询状态。
- **问题排查**：识别长时间运行的查询、锁等待、异常连接。
- **性能分析**：统计查询类型、并发量、资源占用。

#### 1. 基础用法
```sql
-- 查看全部线程（需要 PROCESS 权限）
SHOW FULL PROCESSLIST;

-- 通过 INFORMATION_SCHEMA 获取结构化数据（推荐）
SELECT * FROM INFORMATION_SCHEMA.PROCESSLIST;
```

#### 2. 数据收集脚本
编写脚本定期采集 `PROCESSLIST` 数据，保存到日志文件或数据库：
```bash
#!/bin/bash
# 采集脚本示例：collect_processlist.sh

MYSQL_USER="monitor_user"
MYSQL_PASS="secure_password"
OUTPUT_DIR="/var/log/mysql_processlist"

# 生成时间戳文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/processlist_${TIMESTAMP}.log"

# 执行查询并保存结果
mysql -u $MYSQL_USER -p$MYSQL_PASS -e "SELECT * FROM INFORMATION_SCHEMA.PROCESSLIST" > $OUTPUT_FILE

# 可选：压缩历史文件
find $OUTPUT_DIR -name "processlist_*.log" -mtime +7 -exec gzip {} \;
```

**配置定时任务（cron）**：
```bash
# 每分钟采集一次
* * * * * /path/to/collect_processlist.sh
```

#### 3. 关键字段解析
- `Id`：线程 ID，用于终止异常查询（`KILL {Id}`）。
- `User`：连接用户，识别非法访问。
- `Host`：客户端地址，定位问题来源。
- `db`：当前数据库。
- `Command`：线程状态（Query/Sleep/Locked 等）。
- `Time`：执行时间（秒），长耗时需警惕。
- `State`：操作状态（Sending data/Copying to tmp table 等）。
- `Info`：完整 SQL 语句（需 `SHOW FULL PROCESSLIST`）。

---

### 二、结合 `pt-query-digest` 进行深度分析
`pt-query-digest` 是 Percona Toolkit 的核心工具，用于分析查询模式。结合 `PROCESSLIST` 的用法：

#### 1. 实时分析正在运行的查询
```bash
# 将 PROCESSLIST 输出传递给 pt-query-digest
mysql -u $USER -p$PASS -e "SHOW FULL PROCESSLIST" \
| grep -v "Sleep" \  # 过滤空闲连接
| pt-query-digest --type processlist --report
```

**输出内容**：
- 查询指纹（规范化后的 SQL 模式）。
- 执行次数、总耗时、平均/最大延迟。
- 锁定时间、返回行数等统计指标。

#### 2. 分析历史采集数据
```bash
# 分析多个采集文件
pt-query-digest --type processlist /var/log/mysql_processlist/*.log
```

---

### 三、与其他监控系统集成方案
#### 1. Prometheus + Grafana
- **数据采集**：使用 `mysqld_exporter` 抓取 `PROCESSLIST` 数据。
- **指标暴露**：自定义 Exporter 解析 `PROCESSLIST` 字段（如长查询计数、锁等待线程数）。
- **仪表盘**：在 Grafana 中配置实时监控面板。

**示例 Exporter 片段（Python）**：
```python
from prometheus_client import start_http_server, Gauge
import mysql.connector

# 定义指标
long_queries = Gauge('mysql_long_running_queries', 'Number of queries running > 5s')

def collect_processlist():
    conn = mysql.connector.connect(user='monitor', password='pass', host='localhost')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM INFORMATION_SCHEMA.PROCESSLIST WHERE TIME > 5")
    count = cursor.fetchone()[0]
    long_queries.set(count)
    conn.close()

if __name__ == '__main__':
    start_http_server(8000)
    while True:
        collect_processlist()
        time.sleep(15)
```

#### 2. Zabbix
- **自定义监控项**：通过 Zabbix Agent 执行脚本采集 `PROCESSLIST` 数据。
- **触发器配置**：当长查询数超过阈值时告警。
- **模板导入**：使用社区模板（如 Percona Zabbix Templates）。

#### 3. ELK Stack（日志分析）
- **数据管道**：Filebeat 采集 `PROCESSLIST` 日志 → Logstash 解析 → Elasticsearch 存储。
- **可视化**：Kibana 仪表盘展示查询趋势、异常模式。

---

### 四、高级问题排查流程
1. **定位慢查询**：
   ```sql
   -- 直接筛选执行时间 > N 秒的查询
   SELECT * FROM INFORMATION_SCHEMA.PROCESSLIST WHERE TIME > 10;
   ```

2. **分析锁争用**：
   ```sql
   -- 查看锁等待线程
   SELECT * FROM INFORMATION_SCHEMA.INNODB_LOCK_WAITS;
   ```

3. **终止异常线程**：
   ```sql
   -- 根据 PROCESSLIST 中的 Id 终止线程
   KILL {Id};
   ```

---

### 五、注意事项
1. **权限控制**：监控账号仅需 `PROCESS` 和 `SELECT` 权限。
2. **性能影响**：高频采集可能对高负载实例产生压力，建议间隔 ≥ 10 秒。
3. **数据安全**：避免日志中记录敏感 SQL（如含密码的语句）。

通过以上方法，可系统性地将 `SHOW PROCESSLIST` 数据纳入监控体系，快速定位性能瓶颈与异常行为。