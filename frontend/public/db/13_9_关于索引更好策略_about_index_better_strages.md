---

### **高性能索引策略详解**

---

#### **1. 独立的列**
**定义**：索引列必须是独立的，即不能是表达式的一部分，否则无法使用索引。  
**示例**：  
```sql
-- 无法使用 age 列的索引
SELECT * FROM users WHERE age + 1 = 20;

-- 优化为独立列查询
SELECT * FROM users WHERE age = 19;
```
**总结**：避免在查询条件中对索引列进行运算或函数操作。

---

#### **2. 前缀索引与索引选择性**
**定义**：  
- **前缀索引**：仅对字符串列的前几个字符建立索引，节省存储空间。  
- **索引选择性**：不同值的数量与总行数的比率，值越接近 1 越好（如唯一索引选择性为 1）。  

**示例**：  
```sql
-- 计算 email 列前 10 个字符的选择性
SELECT 
  COUNT(DISTINCT LEFT(email, 10)) / COUNT(*) AS selectivity 
FROM users;
-- 若选择性 > 0.9，可创建前缀索引
ALTER TABLE users ADD INDEX idx_email (email(10));
```
**总结**：长字符串列推荐使用前缀索引，需确保选择性足够高。

---

#### **3. 多列索引（复合索引）**
**定义**：在多个列上建立索引，遵循最左前缀原则。  
**示例**：  
```sql
-- 创建复合索引 (country, city, age)
ALTER TABLE users ADD INDEX idx_country_city_age (country, city, age);

-- 有效查询（使用索引）
SELECT * FROM users WHERE country = 'CN' AND city = 'Shanghai';
SELECT * FROM users WHERE country = 'CN' ORDER BY city;

-- 无效查询（未使用索引）
SELECT * FROM users WHERE city = 'Shanghai'; -- 缺少 country
SELECT * FROM users WHERE country = 'CN' AND age > 20; -- 缺少 city
```
**总结**：复合索引的列顺序需按查询频率和过滤能力排列，确保最左前缀匹配。

---

#### **4. 选择适合的索引顺序**
**规则**：  
- 高频查询条件列应放在最左侧。  
- 高选择性列优先。  

**示例**：  
```sql
-- 场景：按日期和用户ID查询（date 选择性更高）
ALTER TABLE orders ADD INDEX idx_date_user (order_date, user_id);

-- 场景：按用户ID和状态查询（user_id 选择性更高）
ALTER TABLE orders ADD INDEX idx_user_status (user_id, status);
```
**总结**：复合索引的列顺序需结合查询模式和选择性综合决策。

---

#### **5. 聚簇索引**
**定义**：InnoDB 中，表数据按主键顺序存储，主键索引即聚簇索引。  
**示例**：  
```sql
-- 使用自增 ID 作为主键（顺序插入，减少页分裂）
CREATE TABLE logs (
  id INT AUTO_INCREMENT PRIMARY KEY,
  content TEXT
) ENGINE=InnoDB;

-- 使用 UUID 作为主键（随机写入，性能下降）
CREATE TABLE logs (
  uuid CHAR(36) PRIMARY KEY,
  content TEXT
) ENGINE=InnoDB;
```
**总结**：聚簇索引影响数据物理存储，主键应避免随机值（如 UUID）。

---

#### **6. 覆盖索引**
**定义**：索引包含查询所需的所有字段，无需回表查询数据行。  
**示例**：  
```sql
-- 创建覆盖索引 (user_id, status)
ALTER TABLE orders ADD INDEX idx_user_status (user_id, status);

-- 查询命中覆盖索引
SELECT user_id, status FROM orders WHERE user_id = 100;
```
**总结**：优先设计覆盖索引，减少 I/O 开销。

---

#### **7. 使用索引扫描做排序**
**条件**：  
- `ORDER BY` 和 `WHERE` 中的列匹配索引顺序。  
- 排序方向一致（全升序或全降序）。  

**示例**：  
```sql
-- 索引 (age, name)
ALTER TABLE students ADD INDEX idx_age_name (age, name);

-- 有效排序（使用索引）
SELECT * FROM students WHERE age = 18 ORDER BY name;

-- 无效排序（未使用索引）
SELECT * FROM students ORDER BY age DESC, name ASC; -- 混合排序方向
```
**总结**：索引排序需满足最左前缀和方向一致性。

---

#### **8. 压缩索引**
**定义**：通过缩短索引长度减少存储空间，提升内存利用率。  
**示例**：  
```sql
-- 使用前缀索引压缩长字符串
ALTER TABLE products ADD INDEX idx_title (title(20));

-- InnoDB 页压缩（需 Barracuda 文件格式）
ALTER TABLE products COMPRESSION="zlib";
```
**总结**：对长字段使用前缀索引或启用页压缩，减少索引大小。

---

#### **9. 冗余和重复索引**
**定义**：  
- **冗余索引**：已有复合索引 (A, B)，再创建 (A)。  
- **重复索引**：同一列创建了多个相同顺序的索引。  

**示例**：  
```sql
-- 冗余索引示例
ALTER TABLE users ADD INDEX idx_country (country); -- 冗余，因已有 (country, city)
ALTER TABLE users ADD INDEX idx_country_city (country, city);

-- 重复索引示例
ALTER TABLE users ADD INDEX idx_email_1 (email);
ALTER TABLE users ADD INDEX idx_email_2 (email); -- 重复
```
**总结**：定期审查并删除冗余和重复索引。

---

#### **10. 未使用的索引**
**识别方法**：  
```sql
-- 查询未使用的索引（MySQL 8.0+）
SELECT * FROM sys.schema_unused_indexes;
```
**处理**：删除未使用的索引以减少维护开销。  

---

#### **11. 索引和锁**
**影响**：索引在写入时需加锁，可能引发竞争。  
**示例**：  
```sql
-- 高并发写入场景，索引过多导致锁争用
ALTER TABLE logs ADD INDEX idx_user (user_id); -- 插入时需更新索引树，增加行锁竞争
```
**优化**：  
- 减少不必要的索引。  
- 使用批量插入替代单条插入。  

---

### **总结：高性能索引设计原则**
| 策略                | 核心要点                                                                 |
|---------------------|--------------------------------------------------------------------------|
| **独立列**          | 避免对索引列进行运算或函数操作                                           |
| **前缀索引**        | 长字符串列使用前缀，确保高选择性                                         |
| **复合索引顺序**    | 高频条件列在前，高选择性列优先                                           |
| **聚簇索引**        | 主键使用自增或有序值，避免随机写入                                       |
| **覆盖索引**        | 包含查询所需字段，避免回表                                               |
| **索引排序**        | 确保 `ORDER BY` 列匹配索引顺序和方向                                     |
| **压缩与精简**      | 删除冗余索引，启用压缩技术                                               |
| **锁优化**          | 减少高并发写入场景的索引数量                                             |

通过合理设计索引，结合业务查询模式和数据库引擎特性，可显著提升查询性能并降低资源消耗。

