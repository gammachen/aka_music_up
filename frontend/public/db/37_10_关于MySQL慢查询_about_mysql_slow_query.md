获取 MySQL 的慢查询日志（Slow Query Log）可以帮助你分析和优化数据库性能。以下是详细步骤，包括启用慢查询日志、配置相关参数以及查看日志内容。

---

### **一、启用慢查询日志**

#### **1. 检查当前状态**
运行以下 SQL 命令，检查慢查询日志是否已启用：
```sql
SHOW VARIABLES LIKE 'slow_query_log';
```
- 如果值为 `OFF`，表示慢查询日志未启用。
- 如果值为 `ON`，表示已启用。

```sql
mysql> SHOW VARIABLES LIKE 'slow_query_log';
+----------------+-------+
| Variable_name  | Value |
+----------------+-------+
| slow_query_log | ON    |
+----------------+-------+
1 row in set (0.08 sec)

mysql> show profile;
Empty set, 1 warning (0.02 sec)

mysql> SHOW VARIABLES LIKE 'slow_query_log_file';
+---------------------+--------------------------------------+
| Variable_name       | Value                                |
+---------------------+--------------------------------------+
| slow_query_log_file | /var/lib/mysql/b5e399d81223-slow.log |
+---------------------+--------------------------------------+
1 row in set (0.04 sec)

mysql> SHOW VARIABLES LIKE 'long_query_time';
+-----------------+-----------+
| Variable_name   | Value     |
+-----------------+-----------+
| long_query_time | 10.000000 |
+-----------------+-----------+
1 row in set (0.01 sec)

mysql> SHOW VARIABLES LIKE 'log_queries_not_using_indexes';
+-------------------------------+-------+
| Variable_name                 | Value |
+-------------------------------+-------+
| log_queries_not_using_indexes | OFF   |
+-------------------------------+-------+
1 row in set (0.04 sec)

mysql> set global log_queries_not_using_indexes = 'ON'
    -> ;
Query OK, 0 rows affected (0.01 sec)

mysql> SHOW VARIABLES LIKE 'log_queries_not_using_indexes';
+-------------------------------+-------+
| Variable_name                 | Value |
+-------------------------------+-------+
| log_queries_not_using_indexes | ON    |
+-------------------------------+-------+
1 row in set (0.01 sec)
```

#### **2. 启用慢查询日志**
可以通过以下两种方式启用慢查询日志：

##### **(1) 动态启用（无需重启 MySQL）**
运行以下命令启用慢查询日志：
```sql
SET GLOBAL slow_query_log = 'ON';
```

##### **(2) 永久启用（修改配置文件）**
编辑 MySQL 配置文件（通常是 `/etc/mysql/my.cnf` 或 `/etc/my.cnf`），添加或修改以下内容：
```ini
[mysqld]
slow_query_log = 1
slow_query_log_file = /var/log/mysql/mysql-slow.log
long_query_time = 2
```
- **`slow_query_log`**：启用慢查询日志（`1` 表示启用，`0` 表示禁用）。
- **`slow_query_log_file`**：指定慢查询日志文件路径。
- **`long_query_time`**：定义慢查询的阈值时间（单位为秒，默认值为 10 秒）。

保存后，重启 MySQL 服务以应用更改：
```bash
sudo systemctl restart mysql
```

---

### **二、配置慢查询日志参数**

#### **1. 设置慢查询阈值**
慢查询的判定标准是执行时间超过 `long_query_time` 的查询。可以通过以下命令设置：
```sql
SET GLOBAL long_query_time = 2;
```
- 单位为秒，例如 `2` 表示记录所有执行时间超过 2 秒的查询。

#### **2. 记录未使用索引的查询（可选）**
如果你希望记录未使用索引的查询，可以启用以下参数：
```sql
SET GLOBAL log_queries_not_using_indexes = 'ON';
```
注意：这可能会导致日志文件迅速增大，请谨慎使用。

---

### **三、查看慢查询日志**

#### **1. 查看日志文件位置**
运行以下命令确认慢查询日志文件的位置：
```sql
SHOW VARIABLES LIKE 'slow_query_log_file';
```
输出示例：
```
+---------------------+----------------------------+
| Variable_name       | Value                      |
+---------------------+----------------------------+
| slow_query_log_file | /var/log/mysql/mysql-slow.log |
+---------------------+----------------------------+
```

#### **2. 查看日志内容**
可以直接使用 `cat` 或 `tail` 命令查看日志文件内容：
```bash
cat /var/log/mysql/mysql-slow.log
```
或者实时监控日志：
```bash
tail -f /var/log/mysql/mysql-slow.log
```

```sql
bash-5.1# mysql -u root -p -e 'show full processlist';
Enter password:
+-------+-----------------+--------------------+--------------+---------+-------+----------------------------+-----------------------+
| Id    | User            | Host               | db           | Command | Time  | State                      | Info                  |
+-------+-----------------+--------------------+--------------+---------+-------+----------------------------+-----------------------+
|     5 | event_scheduler | localhost          | NULL         | Daemon  | 90108 | Waiting on empty queue     | NULL                  |
| 13673 | root            | localhost          | pperformance | Sleep   |   283 |                            | NULL                  |
| 28521 | root            | 192.168.65.1:16475 | pperformance | Execute |     0 | waiting for handler commit | COMMIT                |
| 28522 | root            | 192.168.65.1:52020 | pperformance | Sleep   |     0 |                            | NULL                  |
| 28523 | root            | 192.168.65.1:32812 | pperformance | Sleep   |     0 |                            | NULL                  |
| 28524 | root            | 192.168.65.1:27145 | pperformance | Sleep   |     0 |                            | NULL                  |
| 29053 | root            | localhost          | pperformance | Query   |     7 | User sleep                 | DO SLEEP(100)  --  10 |
| 29148 | root            | localhost          | NULL         | Query   |     0 | init                       | show full processlist |
+-------+-----------------+--------------------+--------------+---------+-------+----------------------------+-----------------------+
bash-5.1# mysql -u root -p -e 'show full processlist';
Enter password:
+-------+-----------------+--------------------+--------------+---------+-------+----------------------------+--------------------------------------------------------+
| Id    | User            | Host               | db           | Command | Time  | State                      | Info                                                   |
+-------+-----------------+--------------------+--------------+---------+-------+----------------------------+--------------------------------------------------------+
|     5 | event_scheduler | localhost          | NULL         | Daemon  | 90146 | Waiting on empty queue     | NULL                                                   |
| 13673 | root            | localhost          | pperformance | Sleep   |   321 |                            | NULL                                                   |
| 28521 | root            | 192.168.65.1:16475 | pperformance | Execute |     0 | waiting for handler commit | COMMIT                                                 |
| 28522 | root            | 192.168.65.1:52020 | pperformance | Execute |     0 | Opening tables             | SELECT c FROM sbtest4 WHERE id=50271                   |
| 28523 | root            | 192.168.65.1:32812 | pperformance | Sleep   |     0 |                            | NULL                                                   |
| 28524 | root            | 192.168.65.1:27145 | pperformance | Execute |     0 | waiting for handler commit | COMMIT                                                 |
| 29053 | root            | localhost          | pperformance | Query   |    45 | User sleep                 | DO SLEEP(100)  --  10                                  |
| 29149 | root            | localhost          | pperformance | Query   |    12 | statistics                 | SELECT * FROM test_lock WHERE id = 1 FOR UPDATE  --  A |
| 29150 | root            | localhost          | NULL         | Query   |     0 | init                       | show full processlist                                  |
+-------+-----------------+--------------------+--------------+---------+-------+----------------------------+--------------------------------------------------------+
bash-5.1# mysql -u root -p -e 'show full processlist';
Enter password:
+-------+-----------------+--------------------+--------------+---------+-------+----------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Id    | User            | Host               | db           | Command | Time  | State                      | Info                                                                                                                                                          |
+-------+-----------------+--------------------+--------------+---------+-------+----------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+
|     5 | event_scheduler | localhost          | NULL         | Daemon  | 90226 | Waiting on empty queue     | NULL                                                                                                                                                          |
| 13673 | root            | localhost          | pperformance | Sleep   |   401 |                            | NULL                                                                                                                                                          |
| 28521 | root            | 192.168.65.1:16475 | pperformance | Execute |     0 | Sending to client          | SELECT c FROM sbtest8 WHERE id=46535                                                                                                                          |
| 28522 | root            | 192.168.65.1:52020 | pperformance | Execute |     0 | updating                   | UPDATE sbtest3 SET c='34791537160-42290030205-87555819142-67090482356-97100414450-11036318760-21515536453-70546391258-24480889141-48466009702' WHERE id=50341 |
| 28523 | root            | 192.168.65.1:32812 | pperformance | Execute |     0 | waiting for handler commit | COMMIT                                                                                                                                                        |
| 28524 | root            | 192.168.65.1:27145 | pperformance | Execute |     0 | updating                   | UPDATE sbtest8 SET k=k+1 WHERE id=50203                                                                                                                       |
| 29053 | root            | localhost          | pperformance | Sleep   |    25 |                            | NULL                                                                                                                                                          |
| 29149 | root            | localhost          | pperformance | Sleep   |    42 |                            | NULL                                                                                                                                                          |
| 29151 | root            | localhost          | NULL         | Query   |     0 | init                       | show full processlist                                                                                                                                         |
+-------+-----------------+--------------------+--------------+---------+-------+----------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

---

### **四、分析慢查询日志**

#### **1. 使用 `mysqldumpslow` 工具**
MySQL 提供了一个内置工具 `mysqldumpslow`，用于汇总和分析慢查询日志。例如：
```bash
mysqldumpslow -t 10 /var/log/mysql/mysql-slow.log
```
- **`-t 10`**：显示最慢的 10 条查询。

#### **2. 使用第三方工具**
一些第三方工具（如 `pt-query-digest`）可以更深入地分析慢查询日志：
```bash
pt-query-digest /var/log/mysql/mysql-slow.log
```

```bash
(base) shaofu@shaofu:~$ pt-query-digest b5e399d81223-slow.log

# 310ms user time, 20ms system time, 37.46M rss, 49.92M vsz
# Current date: Sat Apr 12 14:51:57 2025
# Hostname: shaofu
# Files: b5e399d81223-slow.log
# Overall: 360 total, 8 unique, 0.04 QPS, 0.25x concurrency ______________
# Time range: 2025-04-12T04:04:29 to 2025-04-12T06:41:32
# Attribute          total     min     max     avg     95%  stddev  median
# ============     ======= ======= ======= ======= ======= ======= =======
# Exec time          2324s   371us   1000s      6s     4ms     72s   568us
# Lock time            98s       0     50s   273ms    33us      3s     5us
# Rows sent        340.82k       0    1000  969.44  964.41  165.99  964.41
# Rows examine     340.83k       0    1000  969.47  964.41  165.83  964.41
# Query size         5.65k      12     157   16.06   14.52   10.46   14.52

# Profile
# Rank Query ID                        Response time   Calls R/Call   V/M
# ==== =============================== =============== ===== ======== ====
#    1 0x9BBE887EE79E579A73667642E5... 2115.0944 91.0%     4 528.7736 39...
#    2 0x6918D0A039B0E94B2F5B748157...  110.0041  4.7%     2  55.0021 73.63
# MISC 0xMISC                            98.6920  4.2%   354   0.2788   0.0 <6 ITEMS>

# Query 1: 0.00 QPS, 1.73x concurrency, ID 0x9BBE887EE79E579A73667642E516B563 at byte 812
# This item is included in the report because it matches --limit.
# Scores: V/M = 392.15
# Time range: 2025-04-12T04:04:29 to 2025-04-12T04:24:53
# Attribute    pct   total     min     max     avg     95%  stddev  median
# ============ === ======= ======= ======= ======= ======= ======= =======
# Count          1       4
# Exec time     91   2115s     10s   1000s    529s    964s    455s    964s
# Lock time      0       0       0       0       0       0       0       0
# Rows sent      0       0       0       0       0       0       0       0
# Rows examine   0       4       1       1       1       1       0       1
# Query size     0      54      12      14   13.50   13.83    0.82   13.83
# String:
# Databases    pperformance
# Hosts        localhost
# Users        root
# Query_time distribution
#   1us
#  10us
# 100us
#   1ms
#  10ms
# 100ms
#    1s
#  10s+  ################################################################
DO SLEEP(1000)\G

# Query 2: 0.01 QPS, 0.63x concurrency, ID 0x6918D0A039B0E94B2F5B7481578D6E90 at byte 75937
# This item is included in the report because it matches --limit.
# Scores: V/M = 73.63
# Time range: 2025-04-12T06:38:37 to 2025-04-12T06:41:32
# Attribute    pct   total     min     max     avg     95%  stddev  median
# ============ === ======= ======= ======= ======= ======= ======= =======
# Count          0       2
# Exec time      4    110s     10s    100s     55s    100s     64s     55s
# Lock time      0       0       0       0       0       0       0       0
# Rows sent      0       0       0       0       0       0       0       0
# Rows examine   0       2       1       1       1       1       0       1
# Query size     0      41      20      21   20.50      21    0.71   20.50
# String:
# Databases    pperformance
# Hosts        localhost
# Users        root
# Query_time distribution
#   1us
#  10us
# 100us
#   1ms
#  10ms
# 100ms
#    1s
#  10s+  ################################################################
DO SLEEP(100)  --  10\G
```


- 安装 `pt-query-digest`：
  ```bash
  sudo apt install percona-toolkit
  ```

---

### **五、注意事项**

1. **日志文件大小管理**：
   慢查询日志可能会快速增长，建议定期清理或轮换日志文件。可以使用 `logrotate` 管理日志文件。

2. **性能影响**：
   启用慢查询日志会对性能产生轻微影响，尤其是在高负载环境下。确保仅在需要时启用，并合理设置 `long_query_time`。

3. **权限问题**：
   确保 MySQL 有权限写入日志文件。如果日志文件路径不存在，可以手动创建并赋予权限：
   ```bash
   sudo touch /var/log/mysql/mysql-slow.log
   sudo chown mysql:mysql /var/log/mysql/mysql-slow.log
   ```

---

### **六、总结**

获取 MySQL 慢查询日志的核心步骤如下：
1. 启用慢查询日志（动态或通过配置文件）。
2. 配置相关参数（如 `long_query_time` 和 `log_queries_not_using_indexes`）。
3. 查看日志文件内容（直接查看或使用工具分析）。
4. 注意日志文件大小和性能影响。

如果有其他需求或遇到问题，请随时补充说明！