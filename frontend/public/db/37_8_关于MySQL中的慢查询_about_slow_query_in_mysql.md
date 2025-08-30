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