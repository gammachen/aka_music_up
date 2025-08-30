## 索引误用场景深度分析

### 1. 未建索引导致全表扫描
**场景**：gold_transactions表按user_id查询交易记录
```sql
EXPLAIN SELECT * FROM gold_transactions WHERE user_id = 1001;
```
**结果分析**：
- type: ALL（全表扫描）
- rows: 500,000
- filtered: 10%

**优化方案**：
```sql
ALTER TABLE gold_transactions ADD INDEX idx_user(user_id);
```
**效果对比**：
| 指标       | 无索引 | 有索引 |
|------------|--------|--------|
| 查询耗时   | 1200ms | 15ms   |
| 扫描行数   | 50万   | 132    |

### 2. 复合索引冗余
**错误案例**：
```sql
CREATE INDEX idx_redundant ON gold_transactions (user_id, create_time, amount);
```
**性能影响**：
- 写入性能下降35%
- 索引体积增加40%

**压测数据**：
| 并发数 | TPS（有冗余索引） | TPS（优化后） |
|--------|-------------------|---------------|
| 50     | 1200              | 1850          |
| 100    | 850               | 1520          |

### 3. 隐式类型转换
**失效案例**：
```sql
SELECT * FROM gold_transactions WHERE transaction_no = 2023050001;
```
**索引失效特征**：
- transaction_no字段为varchar类型
- 执行计划显示"key": null

**解决方案**：
```sql
SELECT * FROM gold_transactions WHERE transaction_no = '2023050001';
```

## 索引成本收益分析框架
```
收益权重 = 查询频率 × 数据筛选率 × 业务优先级
成本权重 = 写操作频率 × 索引维护成本
决策系数 = (收益权重 - 成本权重) / 索引存储空间
```

## 无法使用索引的查询场景
### 1. 排序顺序不符
**场景**：gold_transactions表按(first_name, last_name)排序查询
```sql
EXPLAIN SELECT * FROM gold_transactions ORDER BY first_name, last_name;
```
**索引失效特征**：
- 现有索引为`(last_name, first_name)`
- Using filesort显示在Extra列

**优化方案**：
```sql
ALTER TABLE gold_transactions ADD INDEX idx_name_order (first_name, last_name);
```

### 2. 函数表达式处理
**场景**：按月统计交易记录
```sql
EXPLAIN SELECT * FROM gold_transactions WHERE MONTH(create_time) = 4;
```
**索引失效特征**：
- 即使有create_time索引
- type: ALL，rows显示全表扫描

**解决方案**：
```sql
-- 创建衍生列并建立索引
ALTER TABLE gold_transactions ADD COLUMN create_month TINYINT;
UPDATE gold_transactions SET create_month = MONTH(create_time);
CREATE INDEX idx_create_month ON gold_transactions(create_month);
```

### 3. OR条件分散
**复合查询**：
```sql
EXPLAIN SELECT * FROM gold_transactions 
WHERE user_id = 1001 OR amount > 1000;
```
**执行计划分析**：
- possible_keys显示idx_user,idx_amount
- key: NULL
- Extra显示Using union()

**优化方案**：
```sql
-- 使用UNION替代OR
SELECT * FROM gold_transactions WHERE user_id = 1001
UNION
SELECT * FROM gold_transactions WHERE amount > 1000;
```

### 4. 模糊匹配
**全模糊查询**：
```sql
EXPLAIN SELECT * FROM gold_transactions 
WHERE transaction_desc LIKE '%bonus%';
```
**索引失效特征**：
- 即使transaction_desc有索引
- type: ALL，rows显示全表扫描

**优化建议**：
- 使用反向索引方案（参见26_about_elasticsearch_index.md）
- 限制前缀模糊匹配：LIKE 'bonus%'

### 5. 范围查询导致索引失效
**场景**：按金额范围查询交易记录
```sql
EXPLAIN SELECT * FROM gold_transactions WHERE amount > 1000 AND amount < 5000;
```
**索引失效特征**：
- 即使amount字段有索引
- 范围查询后的其他索引列无法使用
- 执行计划显示Using index condition

**优化方案**：
```sql
-- 调整索引顺序，将范围查询字段放在最后
ALTER TABLE gold_transactions DROP INDEX idx_amount_user;
ALTER TABLE gold_transactions ADD INDEX idx_user_amount(user_id, amount);
```

**性能对比**：
| 查询类型                                | 扫描行数 | 耗时   |
|-----------------------------------------|----------|--------|
| 范围字段在前 (idx_amount_user)          | 25,000   | 320ms  |
| 范围字段在后 (idx_user_amount)          | 5,200    | 85ms   |

### 6. 最左匹配原则失效
**场景**：复合索引未按最左匹配原则使用
```sql
EXPLAIN SELECT * FROM gold_transactions WHERE create_time = '2023-04-01' AND amount > 1000;
```
**索引失效特征**：
- 现有索引为`(user_id, create_time, amount)`
- 查询未包含索引最左列user_id
- 执行计划显示key: NULL

**优化方案**：
```sql
-- 方案1：调整查询以包含最左列
SELECT * FROM gold_transactions WHERE user_id IN (SELECT id FROM users) AND create_time = '2023-04-01' AND amount > 1000;

-- 方案2：为特定查询创建新索引
ALTER TABLE gold_transactions ADD INDEX idx_time_amount(create_time, amount);
```

### 7. 高选择性索引问题
**场景**：低选择性字段放在复合索引前列
```sql
EXPLAIN SELECT * FROM gold_transactions WHERE status = 'completed' AND user_id = 1001;
```
**索引失效特征**：
- 现有索引为`(status, user_id)`
- status字段只有几个固定值，选择性很低
- 执行计划显示rows扫描数量大

**优化方案**：
```sql
-- 调整索引顺序，高选择性字段放前面
ALTER TABLE gold_transactions DROP INDEX idx_status_user;
ALTER TABLE gold_transactions ADD INDEX idx_user_status(user_id, status);
```

**性能对比**：
| 索引结构                | 扫描行数 | 耗时   |
|-------------------------|----------|--------|
| (status, user_id)       | 125,000  | 450ms  |
| (user_id, status)       | 132      | 12ms   |

## SQL执行计划(QEP)工具详解

### EXPLAIN命令基础
EXPLAIN是MySQL提供的分析SQL语句执行计划的强大工具，它能够展示MySQL优化器如何执行SQL语句。

**基本语法**：
```sql
EXPLAIN SELECT * FROM gold_transactions WHERE user_id = 1001;
```

**核心输出字段解析**：

| 字段名      | 含义                           | 关键值解释                                           |
|-------------|--------------------------------|------------------------------------------------------|
| id          | 查询标识符                     | 数字越大越先执行                                     |
| select_type | 查询类型                       | SIMPLE, PRIMARY, SUBQUERY, DERIVED, UNION等         |
| table       | 表名                           | 当前行访问的表                                       |
| partitions  | 匹配的分区                     | 分区表中使用的分区                                   |
| type        | 访问类型                       | system > const > eq_ref > ref > range > index > ALL |
| possible_keys | 可能使用的索引               | MySQL可能选择的索引列表                              |
| key         | 实际使用的索引                 | NULL表示未使用索引                                   |
| key_len     | 索引使用的字节数               | 索引字段的最大可能长度                               |
| ref         | 索引比较的列                   | 哪些列或常量被用于查找索引列上的值                   |
| rows        | 预计需要检查的行数             | 估计值，越小越好                                     |
| filtered    | 按表条件过滤的行百分比         | 值越大越好，100%表示没有行被过滤                     |
| Extra       | 附加信息                       | 包含不适合显示在其他列中的额外信息                   |

**访问类型(type)详解**：
- **system**：表只有一行记录，是const类型的特例
- **const**：通过索引一次就能找到，用于主键或唯一索引等值查询
- **eq_ref**：对于前表的每一行，在当前表中只能找到一条记录，常用于主键或唯一索引关联
- **ref**：非唯一索引等值查询，可能返回多条记录
- **range**：范围查询，如>, <, BETWEEN, IN等
- **index**：全索引扫描，比ALL快，但仍需扫描全表索引
- **ALL**：全表扫描，性能最差

**Extra字段关键值**：
- **Using where**：表示MySQL服务器将在存储引擎检索行后再进行过滤
- **Using index**：表示查询使用了覆盖索引，不需要回表
- **Using filesort**：表示MySQL需要额外的排序操作，无法利用索引完成排序
- **Using temporary**：表示MySQL需要创建临时表来存储结果
- **Using index condition**：表示使用了索引条件下推优化

### EXPLAIN ANALYZE深度分析
MySQL 8.0引入的EXPLAIN ANALYZE提供了更详细的执行信息，包括实际执行时间和成本。

**基本语法**：
```sql
EXPLAIN ANALYZE SELECT * FROM gold_transactions WHERE user_id = 1001;
```

**输出示例**：
```
-> Index lookup on gold_transactions using idx_user (user_id=1001) (cost=0.35 rows=132) (actual time=0.032..0.075 rows=132 loops=1)
```

**关键指标解读**：
- **cost**：优化器估计的成本
- **actual time**：实际执行时间(毫秒)，格式为start..end
- **rows**：实际返回的行数
- **loops**：执行的循环次数

### 索引使用效率诊断方法

**1. 索引选择性评估**
```sql
-- 计算字段的选择性
SELECT COUNT(DISTINCT column_name) / COUNT(*) AS selectivity 
FROM table_name;
```
选择性接近1的字段适合建立索引，接近0的字段不适合。

**2. 索引使用情况监控**
```sql
-- 查看索引使用频率
SHOW GLOBAL STATUS LIKE 'Handler_read%';
```
比较Handler_read_key（索引读取）和Handler_read_rnd_next（全表扫描）的比值。

**3. 慢查询日志分析**
```sql
-- 开启慢查询日志
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL long_query_time = 1; -- 设置阈值为1秒
```
使用pt-query-digest等工具分析慢查询日志，找出需要优化的SQL。

## 索引失效的根本原因分析

### 1. 优化器成本估算偏差
当MySQL优化器估计使用索引的成本高于全表扫描时，会选择不使用索引。常见原因：

- **统计信息不准确**：表数据分布不均匀
- **小表效应**：表数据量小，全表扫描可能比索引查找更快
- **缓存影响**：数据已在内存中，减少了I/O成本差异

**解决方案**：
```sql
-- 更新统计信息
ANALYZE TABLE gold_transactions;

-- 强制使用索引（谨慎使用）
SELECT * FROM gold_transactions FORCE INDEX(idx_user) WHERE user_id = 1001;
```

### 2. 数据类型不匹配的深层影响
类型转换不仅导致索引失效，还会增加CPU开销和内存使用。

**案例分析**：
```sql
-- 原始表结构
CREATE TABLE orders (
  id INT PRIMARY KEY,
  order_no VARCHAR(20),
  amount DECIMAL(10,2),
  INDEX idx_order_no(order_no)
);

-- 类型不匹配查询
EXPLAIN SELECT * FROM orders WHERE order_no = 12345;
```

**性能影响**：
1. 索引失效导致全表扫描
2. 每行数据都需要进行类型转换
3. 转换过程消耗CPU资源

**最佳实践**：
- 查询条件与索引列数据类型保持一致
- 使用参数化查询避免隐式转换
- 设计表时选择合适的数据类型

### 3. 复合索引设计的权衡决策
复合索引设计需要考虑查询模式、基数、选择性和写入性能的平衡。

**决策框架**：
1. **查询频率分析**：识别高频查询模式
2. **选择性评估**：高选择性字段放前面
3. **范围查询识别**：范围查询字段放最后
4. **写入影响评估**：索引数量与写入性能成反比

**案例研究**：用户交易查询优化

| 查询模式                                   | 频率  | 最优索引设计                |
|--------------------------------------------|-------|-----------------------------|
| WHERE user_id=? AND create_time=?          | 高    | (user_id, create_time)      |
| WHERE user_id=? AND status=?               | 高    | (user_id, status)           |
| WHERE create_time BETWEEN ? AND ?          | 中    | (create_time)               |
| WHERE amount>? ORDER BY create_time        | 低    | (create_time, amount)       |

## 技术体系关联
与Elasticsearch索引设计（参见26_about_elasticsearch_index.md）的共性原则：
1. 查询模式驱动设计
2. 字段类型严格匹配
3. 写性能与查询性能的平衡
4. 生命周期管理策略

差异对比：
| 维度       | MySQL           | Elasticsearch     |
|------------|-----------------|-------------------|
| 索引单位   | 表级别          | 索引即数据库      |
| 更新策略   | 同步更新        | 近实时更新        |
| 数据结构   | B+Tree          | 倒排索引          |
| 分片策略   | 自动分区        | 自定义分片        |