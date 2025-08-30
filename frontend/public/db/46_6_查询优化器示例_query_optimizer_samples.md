### MySQL 针对性查询优化详解

---

#### **1. 优化 `COUNT()` 查询**
**原理**：`COUNT()` 的性能取决于索引覆盖和存储引擎。  
**优化方法**：  
- **使用近似值**：当精确值非必需时，`SHOW TABLE STATUS` 可快速获取行数。  
- **二级索引优化**：`COUNT(column)` 使用更小的二级索引而非主键。  
- **冗余统计**：定期更新统计表存储行数，避免全表扫描。  

**示例**：  
```sql
-- 低效：全表扫描
SELECT COUNT(*) FROM users WHERE status = 'active';

-- 优化1：使用覆盖索引
ALTER TABLE users ADD INDEX idx_status (status);
SELECT COUNT(*) FROM users WHERE status = 'active';

-- 优化2：近似值（InnoDB）
SHOW TABLE STATUS LIKE 'users'; -- 查看 Rows 字段
```

---

#### **2. 优化关联查询**
**原理**：关联查询性能取决于索引匹配和表大小顺序。  
**优化方法**：  
- **小表驱动大表**：优先扫描小表，减少嵌套循环次数。  
- **覆盖索引**：确保关联字段和查询列在索引中。  
- **避免笛卡尔积**：明确关联条件，减少中间结果集。  

**示例**：  
```sql
-- 低效：大表驱动小表
SELECT * FROM large_table 
JOIN small_table ON large_table.id = small_table.large_id;

-- 优化1：强制小表驱动
SELECT /*+ STRAIGHT_JOIN */ * FROM small_table 
JOIN large_table ON small_table.large_id = large_table.id;

-- 优化2：覆盖索引
ALTER TABLE large_table ADD INDEX idx_id_cover (id, name);
```

---

#### **3. 优化子查询**
**原理**：子查询易导致临时表和多次扫描。  
**优化方法**：  
- **改写为 `JOIN`**：消除嵌套查询，减少执行步骤。  
- **使用 `EXISTS` 替代 `IN`**：`EXISTS` 在找到匹配后立即终止扫描。  
- **物化派生表**：将子查询结果存储为临时表。  

**示例**：  
```sql
-- 低效：IN 子查询
SELECT * FROM users 
WHERE id IN (SELECT user_id FROM orders);

-- 优化1：JOIN 改写
SELECT users.* FROM users 
JOIN orders ON users.id = orders.user_id;

-- 优化2：EXISTS
SELECT * FROM users u 
WHERE EXISTS (
  SELECT 1 FROM orders o WHERE o.user_id = u.id
);
```

---

#### **4. 优化 `GROUP BY` 和 `DISTINCT`**
**原理**：分组和去重依赖索引排序和中间结果集大小。  
**优化方法**：  
- **松散索引扫描**：利用索引顺序跳过不必要的分组。  
- **紧凑索引扫描**：通过覆盖索引减少回表。  
- **避免非索引列排序**：为 `GROUP BY` 和 `ORDER BY` 使用相同索引。  

**示例**：  
```sql
-- 低效：未使用索引
SELECT country, COUNT(*) FROM users GROUP BY country;

-- 优化1：松散索引扫描
ALTER TABLE users ADD INDEX idx_country (country);
SELECT country, COUNT(*) FROM users GROUP BY country;

-- 优化2：覆盖索引
ALTER TABLE users ADD INDEX idx_country_city (country, city);
SELECT country, city, COUNT(*) FROM users GROUP BY country, city;
```

---

#### **5. 优化 `LIMIT` 分页**
**原理**：大偏移量导致大量数据扫描和排序。  
**优化方法**：  
- **延迟关联**：先获取 ID，再回表查询。  
- **游标分页**：使用 `WHERE` 条件替代 `OFFSET`。  
- **覆盖索引**：避免回表，直接通过索引返回数据。  

**示例**：  
```sql
-- 低效：大偏移量
SELECT * FROM users ORDER BY id LIMIT 100000, 10;

-- 优化1：延迟关联
SELECT * FROM users 
JOIN (SELECT id FROM users ORDER BY id LIMIT 100000, 10) AS tmp 
USING (id);

-- 优化2：游标分页（基于上一页最大值）
SELECT * FROM users 
WHERE id > 100000 ORDER BY id LIMIT 10;
```

---

#### **6. 优化 `SQL_CALC_FOUND_ROWS`**
**原理**：`SQL_CALC_FOUND_ROWS` 导致全表扫描计算总数。  
**优化方法**：  
- **分拆查询**：单独执行 `COUNT(*)` 和分页查询。  
- **近似统计**：使用缓存或预计算的总数。  
- **业务妥协**：不显示精确总数，仅提供“下一页”功能。  

**示例**：  
```sql
-- 低效：强制计算总数
SELECT SQL_CALC_FOUND_ROWS * FROM users LIMIT 10;

-- 优化1：分拆查询
SELECT * FROM users LIMIT 10;
SELECT COUNT(*) FROM users;
```

---

#### **7. 优化 `UNION` 查询**
**原理**：`UNION` 默认去重并生成临时表，开销大。  
**优化方法**：  
- **优先 `UNION ALL`**：避免去重开销。  
- **分页下推**：在子查询中分页，减少临时表数据量。  
- **索引覆盖**：确保每个 `UNION` 子句使用索引。  

**示例**：  
```sql
-- 低效：UNION 去重
SELECT id FROM users 
UNION 
SELECT id FROM admins;

-- 优化1：UNION ALL
SELECT id FROM users 
UNION ALL 
SELECT id FROM admins;

-- 优化2：分页下推
(SELECT id FROM users LIMIT 10) 
UNION ALL 
(SELECT id FROM admins LIMIT 10);
```

---

#### **8. 静态查询分析**
**原理**：通过 `EXPLAIN` 和 `SHOW PROFILE` 定位瓶颈。  
**优化方法**：  
- **分析执行计划**：检查 `type`、`key`、`rows` 和 `Extra` 列。  
- **索引建议**：根据 `possible_keys` 添加缺失索引。  
- **性能剖析**：使用 `SHOW PROFILE` 查看各阶段耗时。  

**示例**：  
```sql
-- 步骤1：查看执行计划
EXPLAIN SELECT * FROM users WHERE country = 'US';

-- 步骤2：添加索引
ALTER TABLE users ADD INDEX idx_country (country);

-- 步骤3：性能剖析
SET profiling = 1;
SELECT * FROM users WHERE country = 'US';
SHOW PROFILES;
```

---

#### **9. 用户自定义变量**
**原理**：自定义变量可用于临时存储中间结果。  
**优化方法**：  
- **分页优化**：存储上一页最大值，加速翻页。  
- **动态排名**：计算行号或排名，避免子查询。  
- **会话级缓存**：缓存复杂计算的中间值。  

**示例**：  
```sql
-- 示例1：计算行号
SET @row_number = 0;
SELECT (@row_number:=@row_number + 1) AS row_no, name 
FROM users ORDER BY id;

-- 示例2：分页优化
SET @last_id = 0;
SELECT * FROM users 
WHERE id > @last_id ORDER BY id LIMIT 10;
```

---

### **总结**
通过针对性优化，可显著提升不同场景下的查询性能。核心原则包括 **索引覆盖**、**减少数据扫描**、**利用存储引擎特性** 和 **避免冗余计算**。实际优化中需结合 `EXPLAIN` 分析，权衡业务需求与执行效率。