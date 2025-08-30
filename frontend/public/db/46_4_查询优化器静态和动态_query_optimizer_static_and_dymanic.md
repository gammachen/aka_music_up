---

### MySQL 查询优化器的静态优化与动态优化详解

MySQL 的查询优化器通过 **静态优化**（编译时优化）和 **动态优化**（运行时优化）来生成高效的执行计划。以下从两种优化的类型、原理、示例展开说明：

---

### **一、静态优化（Static Optimizations）**
静态优化在查询 **解析阶段** 完成，不依赖实际数据值，通过逻辑分析和语法树改写优化查询结构。

#### **1. 常量表达式计算（Constant Expression Evaluation）**
- **原理**：在解析阶段直接计算常量表达式，避免运行时重复计算。  
- **示例**：
  ```sql
  SELECT * FROM users WHERE age > 10 + 20;
  -- 优化为：
  SELECT * FROM users WHERE age > 30;
  ```

#### **2. 无效条件消除（Dead Code Elimination）**
- **原理**：移除永远为真或假的条件，简化查询逻辑。  
- **示例**：
  ```sql
  SELECT * FROM users WHERE 1 = 1;
  -- 优化为：
  SELECT * FROM users;
  ```

#### **3. 外连接转内连接（Outer Join to Inner Join Conversion）**
- **原理**：当外连接查询条件隐式排除了 `NULL` 值时，转为更高效的内连接。  
- **示例**：
  ```sql
  SELECT * FROM users
  LEFT JOIN orders ON users.id = orders.user_id
  WHERE orders.amount > 100;
  -- 优化为内连接（WHERE 条件隐含 orders.user_id 必须存在）：
  SELECT * FROM users
  INNER JOIN orders ON users.id = orders.user_id
  WHERE orders.amount > 100;
  ```

#### **4. 子查询优化（Subquery Optimization）**
- **原理**：将子查询转换为 `JOIN` 或半连接（Semi-Join），减少嵌套循环开销。  
- **示例**：
  ```sql
  SELECT * FROM users
  WHERE id IN (SELECT user_id FROM orders);
  -- 优化为半连接：
  SELECT users.* FROM users
  SEMI JOIN orders ON users.id = orders.user_id;
  ```

#### **5. 条件合并与下推（Predicate Merging and Pushdown）**
- **原理**：合并重复条件，并将过滤条件下推到最底层表。  
- **示例**：
  ```sql
  SELECT * FROM users
  JOIN orders ON users.id = orders.user_id
  WHERE users.country = 'US' AND orders.status = 'paid';
  -- 条件 users.country='US' 下推到 users 表扫描阶段；
  -- 条件 orders.status='paid' 下推到 orders 表扫描阶段。
  ```

---

### **二、动态优化（Dynamic Optimizations）**
动态优化在查询 **执行阶段** 进行，基于表统计信息、索引选择性等实时数据特征调整执行计划。

#### **1. 索引选择（Index Selection）**
- **原理**：根据索引基数和数据分布选择最优索引。  
- **示例**：
  ```sql
  -- 表 users 有索引 (country) 和 (age)
  SELECT * FROM users WHERE country = 'CN' AND age > 25;
  -- 优化器选择 country 索引（假设 country='CN' 过滤后数据量更少）。
  ```

#### **2. 连接顺序调整（Join Reordering）**
- **原理**：基于表大小和过滤条件，动态调整多表连接的顺序。  
- **示例**：
  ```sql
  SELECT * FROM small_table
  JOIN large_table ON small_table.id = large_table.small_id;
  -- 优化器优先扫描 small_table，再关联 large_table，减少中间结果集。
  ```

#### **3. 访问方法选择（Access Method Adjustment）**
- **原理**：根据数据量选择全表扫描或索引扫描。  
- **示例**：
  ```sql
  -- 表 orders 有 100 行数据
  SELECT * FROM orders WHERE user_id = 100;
  -- 若 user_id 索引非高选择性，优化器可能选择全表扫描。
  ```

#### **4. 临时表与排序优化（Temporary Table and Sorting）**
- **原理**：根据中间结果集大小决定是否使用内存临时表或磁盘临时表。  
- **示例**：
  ```sql
  SELECT * FROM users ORDER BY last_login DESC;
  -- 若结果集较小，使用内存排序；若超过 `sort_buffer_size`，使用磁盘文件排序。
  ```

#### **5. 自适应索引（Adaptive Indexing，MySQL 8.0+）**
- **原理**：根据查询模式自动创建或删除索引（需企业版支持）。  
- **示例**：
  ```sql
  -- 高频查询：
  SELECT * FROM logs WHERE request_time > '2023-01-01';
  -- 优化器可能自动创建 (request_time) 索引。
  ```

---

### **三、静态优化 vs 动态优化对比**

| 优化类型     | 触发阶段       | 依赖数据 | 示例场景                         |
|--------------|----------------|----------|----------------------------------|
| **静态优化** | 查询解析阶段   | 无       | 常量计算、子查询转 JOIN、条件下推 |
| **动态优化** | 查询执行阶段   | 有       | 索引选择、连接顺序调整、临时表策略 |

---

### **四、优化器决策示例**

#### **场景 1：多条件查询**
```sql
SELECT * FROM products 
WHERE category = 'Electronics' 
  AND price > 1000 
  AND stock > 0;
```
- **静态优化**：  
  - 合并 `category`、`price`、`stock` 条件。  
- **动态优化**：  
  - 选择 `(category, price)` 复合索引（假设 `category='Electronics'` 过滤后数据量少）。  

#### **场景 2：多表关联**
```sql
SELECT * FROM orders o
JOIN customers c ON o.customer_id = c.id
JOIN products p ON o.product_id = p.id
WHERE c.country = 'US' AND p.price > 100;
```
- **静态优化**：  
  - 将 `c.country='US'` 下推到 `customers` 表扫描阶段。  
- **动态优化**：  
  - 根据 `customers` 和 `products` 表的大小，决定先关联哪个表。  

---

### **五、总结**
- **静态优化** 通过逻辑改写简化查询，减少执行阶段的计算量。  
- **动态优化** 基于实时数据特征选择最优执行路径，提升查询效率。  
- **实践建议**：  
  - 为高频查询设计合适的复合索引。  
  - 避免复杂子查询，优先使用 `JOIN`。  
  - 定期更新统计信息（`ANALYZE TABLE`），确保动态优化准确性。