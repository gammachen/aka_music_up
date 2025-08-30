### MySQL慢查询日志分析案例

慢查询日志（Slow Query Log）是MySQL提供的一种工具，用于记录执行时间超过指定阈值的查询。通过分析慢查询日志，可以识别性能瓶颈并采取相应的优化措施。以下是一个详细的分析案例，展示如何从慢查询日志中发现问题并进行优化。

---

#### **1. 启用慢查询日志**

首先，需要启用慢查询日志并设置适当的阈值。假设我们将慢查询日志的阈值设置为1秒。

**配置步骤**：
1. **编辑MySQL配置文件**（通常是 `my.cnf` 或 `my.ini`）：
   ```ini
   [mysqld]
   slow_query_log = 1
   slow_query_log_file = /var/log/mysql/slow-query.log
   long_query_time = 1
   log_queries_not_using_indexes = 1
   ```

2. **重启MySQL服务**：
   ```bash
   sudo systemctl restart mysql
   ```

---

#### **2. 收集慢查询日志**

启用慢查询日志后，MySQL会将执行时间超过1秒的查询记录到指定的日志文件中。假设我们已经收集了一些慢查询日志。

**示例日志文件**：
```sql
# Time: 230928 12:34:56
# User@Host: root[root] @ localhost []
# Query_time: 1.234567  Lock_time: 0.000123  Rows_sent: 100  Rows_examined: 10000
SET timestamp=1695875696;
SELECT * FROM orders WHERE customer_id = 123;
```

---

#### **3. 分析慢查询日志**

使用 `pt-query-digest` 工具来分析慢查询日志，该工具可以提供详细的性能分析报告。

**安装 `pt-query-digest`**：
```bash
wget percona-release.key
sudo apt-key add percona-release.key
sudo sh -c 'echo "deb http://repo.percona.com/apt/$(lsb_release -sc) main" >> /etc/apt/sources.list.d/percona.list'
sudo apt-get update
sudo apt-get install percona-toolkit
```

**分析日志**：
```bash
pt-query-digest /var/log/mysql/slow-query.log > slow-query-analysis.txt
```

**示例分析报告**：
```plaintext
# 1.23s user time, 10ms system time, 1.23M rss, 2.46M vsz
# Current date: Fri Sep 29 09:00:00 2023
# Hostname: localhost
# Files: /var/log/mysql/slow-query.log
# Overall: 10 total, 5 unique, 0 QPS, 0x concurrency __________________________
# Time range: 2023-09-28 12:34:56 to 2023-09-28 12:35:56
# Attribute          total     min     max     avg     95%  stddev  median
# ============     ======= ======= ======= ======= ======= ======= =======
# Query_time         10.23     1.00     2.00     1.02     1.99    0.34     1.01
# Lock_time          0.000     0.000     0.000     0.000     0.000    0.000     0.000
# Rows_sent          100.00    10.00   100.00    50.00    90.00    25.00    50.00
# Rows_examined     10000.00  1000.00  10000.00  5000.00  9000.00  2500.00  5000.00
# Rows_affected         0.00     0.00     0.00     0.00     0.00     0.00     0.00
# Bytes_sent            0.00     0.00     0.00     0.00     0.00     0.00     0.00
# Tmp_tables            0.00     0.00     0.00     0.00     0.00     0.00     0.00
# Tmp_disk_tables       0.00     0.00     0.00     0.00     0.00     0.00     0.00
# Filesort             0.00     0.00     0.00     0.00     0.00     0.00     0.00
# Filesort_on_disk     0.00     0.00     0.00     0.00     0.00     0.00     0.00
# InnoDB_IO_r_ops      0.00     0.00     0.00     0.00     0.00     0.00     0.00
# InnoDB_IO_r_bytes    0.00     0.00     0.00     0.00     0.00     0.00     0.00
# InnoDB_IO_r_wait     0.00     0.00     0.00     0.00     0.00     0.00     0.00
# InnoDB_rec_lock_wait 0.00     0.00     0.00     0.00     0.00     0.00     0.00
# InnoDB_queue_wait    0.00     0.00     0.00     0.00     0.00     0.00     0.00
# InnoDB_pages_distinct 0.00     0.00     0.00     0.00     0.00     0.00     0.00

# Profile
# Rank Query ID           Response time Calls R/Call V/M   Item
# ==== ================== ============= ===== ====== ===== ================
#    1 0xABCDEF123456789A  5.0000 50.0%     1 5.0000  0.00 SELECT orders
#    2 0x123456789ABCDEF0  3.0000 30.0%     1 3.0000  0.00 SELECT customers
#    3 0x23456789ABCDEF01  2.0000 20.0%     1 2.0000  0.00 SELECT products
```

---

#### **4. 识别性能瓶颈**

从分析报告中，我们可以识别出执行时间最长的查询。

**示例查询**：
```sql
# Query_time: 1.234567  Lock_time: 0.000123  Rows_sent: 100  Rows_examined: 10000
SET timestamp=1695875696;
SELECT * FROM orders WHERE customer_id = 123;
```

**分析**：
- **Query_time**：1.234567秒，超过了1秒的阈值。
- **Lock_time**：0.000123秒，锁等待时间较短。
- **Rows_sent**：100行，返回的行数较少。
- **Rows_examined**：10000行，扫描的行数较多。

**问题**：
- 查询 `orders` 表时，扫描了10000行，但只返回了100行，说明查询效率较低。
- 可能是因为 `customer_id` 列没有索引，导致全表扫描。

---

#### **5. 优化查询**

**优化步骤**：

1. **添加索引**：
   - 为 `customer_id` 列添加索引，以加速查询。
   ```sql
   ALTER TABLE orders ADD INDEX idx_customer_id (customer_id);
   ```

2. **验证索引效果**：
   - 使用 `EXPLAIN` 分析查询，确认索引是否生效。
   ```sql
   EXPLAIN SELECT * FROM orders WHERE customer_id = 123;
   ```

**示例 `EXPLAIN` 输出**：
```plaintext
+----+-------------+--------+------------+-------+---------------+-----------------+---------+-------+------+----------+-------+
| id | select_type | table  | partitions | type  | possible_keys | key             | key_len | ref   | rows | filtered | Extra |
+----+-------------+--------+------------+-------+---------------+-----------------+---------+-------+------+----------+-------+
|  1 | SIMPLE      | orders | NULL       | ref   | idx_customer_id | idx_customer_id | 4       | const |   100 |   100.00 | NULL  |
+----+-------------+--------+------------+-------+---------------+-----------------+---------+-------+------+----------+-------+
```

**分析**：
- **type**：`ref`，表示使用了索引。
- **key**：`idx_customer_id`，表示使用了 `customer_id` 列的索引。
- **rows**：100，表示扫描的行数减少到100行。

---

#### **6. 验证优化效果**

重新运行基准测试，验证优化效果。

**示例基准测试**：
```bash
sysbench --test=oltp --db-driver=mysql --mysql-host=localhost --mysql-port=3306 --mysql-user=root --mysql-password=password --oltp-table-size=1000000 --oltp-tables-count=10 --max-time=3600 --max-requests=0 --num-threads=64 run
```

**分析优化前后的性能**：
- **优化前**：
  - **Query_time**：1.234567秒。
  - **Rows_examined**：10000行。
- **优化后**：
  - **Query_time**：0.012345秒。
  - **Rows_examined**：100行。

**结论**：
- 添加索引后，查询时间从1.23秒降低到0.01秒，性能显著提升。
- 扫描的行数从10000行减少到100行，查询效率提高。

---

#### **7. 处理其他慢查询**

**示例查询**：
```sql
# Query_time: 3.000000  Lock_time: 0.000000  Rows_sent: 10  Rows_examined: 5000
SET timestamp=1695875796;
SELECT * FROM customers WHERE last_name = 'Smith';
```

**分析**：
- **Query_time**：3.000000秒，超过了1秒的阈值。
- **Lock_time**：0.000000秒，锁等待时间较短。
- **Rows_sent**：10行，返回的行数较少。
- **Rows_examined**：5000行，扫描的行数较多。

**问题**：
- 查询 `customers` 表时，扫描了5000行，但只返回了10行，说明查询效率较低。
- 可能是因为 `last_name` 列没有索引，导致全表扫描。

**优化步骤**：

1. **添加索引**：
   - 为 `last_name` 列添加索引，以加速查询。
   ```sql
   ALTER TABLE customers ADD INDEX idx_last_name (last_name);
   ```

2. **验证索引效果**：
   - 使用 `EXPLAIN` 分析查询，确认索引是否生效。
   ```sql
   EXPLAIN SELECT * FROM customers WHERE last_name = 'Smith';
   ```

**示例 `EXPLAIN` 输出**：
```plaintext
+----+-------------+-----------+------------+-------+---------------+---------------+---------+-------+------+----------+-------+
| id | select_type | table     | partitions | type  | possible_keys | key           | key_len | ref   | rows | filtered | Extra |
+----+-------------+-----------+------------+-------+---------------+---------------+---------+-------+------+----------+-------+
|  1 | SIMPLE      | customers | NULL       | ref   | idx_last_name | idx_last_name | 152     | const |   10 |   100.00 | NULL  |
+----+-------------+-----------+------------+-------+---------------+---------------+---------+-------+------+----------+-------+
```

**分析**：
- **type**：`ref`，表示使用了索引。
- **key**：`idx_last_name`，表示使用了 `last_name` 列的索引。
- **rows**：10，表示扫描的行数减少到10行。

**验证优化效果**：
- 重新运行基准测试，验证优化效果。
- 使用 `sysbench` 或其他基准测试工具进行测试。

---

#### **8. 处理复杂的查询**

**示例查询**：
```sql
# Query_time: 2.000000  Lock_time: 0.000000  Rows_sent: 1000  Rows_examined: 100000
SET timestamp=1695875896;
SELECT * FROM orders WHERE order_date BETWEEN '2023-01-01' AND '2023-12-31' AND status = 'shipped';
```

**分析**：
- **Query_time**：2.000000秒，超过了1秒的阈值。
- **Lock_time**：0.000000秒，锁等待时间较短。
- **Rows_sent**：1000行，返回的行数较多。
- **Rows_examined**：100000行，扫描的行数较多。

**问题**：
- 查询 `orders` 表时，扫描了100000行，但只返回了1000行，说明查询效率较低。
- 可能是因为 `order_date` 和 `status` 列没有复合索引，导致全表扫描。

**优化步骤**：

1. **添加复合索引**：
   - 为 `order_date` 和 `status` 列添加复合索引，以加速查询。
   ```sql
   ALTER TABLE orders ADD INDEX idx_order_date_status (order_date, status);
   ```

2. **验证索引效果**：
   - 使用 `EXPLAIN` 分析查询，确认索引是否生效。
   ```sql
   EXPLAIN SELECT * FROM orders WHERE order_date BETWEEN '2023-01-01' AND '2023-12-31' AND status = 'shipped';
   ```

**示例 `EXPLAIN` 输出**：
```plaintext
+----+-------------+--------+------------+-------+----------------------------+----------------------------+---------+------+------+----------+-----------------------+
| id | select_type | table  | partitions | type  | possible_keys              | key                        | key_len | ref  | rows | filtered | Extra                 |
+----+-------------+--------+------------+-------+----------------------------+----------------------------+---------+------+------+----------+-----------------------+
|  1 | SIMPLE      | orders | NULL       | range | idx_order_date_status      | idx_order_date_status      | 9       | NULL | 1000 |   100.00 | Using index condition |
+----+-------------+--------+------------+-------+----------------------------+----------------------------+---------+------+------+----------+-----------------------+
```

**分析**：
- **type**：`range`，表示使用了索引范围扫描。
- **key**：`idx_order_date_status`，表示使用了复合索引。
- **rows**：1000，表示扫描的行数减少到1000行。

**验证优化效果**：
- 重新运行基准测试，验证优化效果。
- 使用 `sysbench` 或其他基准测试工具进行测试。

---

#### **9. 处理未使用索引的查询**

**示例查询**：
```sql
# Query_time: 1.500000  Lock_time: 0.000000  Rows_sent: 50  Rows_examined: 50000
SET timestamp=1695875996;
SELECT * FROM products WHERE price > 100;
```

**分析**：
- **Query_time**：1.500000秒，超过了1秒的阈值。
- **Lock_time**：0.000000秒，锁等待时间较短。
- **Rows_sent**：50行，返回的行数较少。
- **Rows_examined**：50000行，扫描的行数较多。

**问题**：
- 查询 `products` 表时，扫描了