在 MySQL 中，虽然原生不支持直接创建自定义的哈希索引（仅 `MEMORY` 存储引擎有自动哈希索引），但可以通过新增一列存储哈希值并结合 B-Tree 索引模拟哈希索引的效果。以下是具体案例：

---

### **场景描述**
假设有一个用户表 `users`，需要高频根据 `email` 字段进行精确查询。由于 `email` 字段较长，直接对其创建 B-Tree 索引会占用较多空间，可以通过新增哈希值列优化查询效率。

---

### **实现步骤**

#### 1. 新增哈希值列
为 `users` 表添加一个用于存储 `email` 哈希值的新列（例如 `email_hash`）：
```sql
ALTER TABLE users ADD COLUMN email_hash CHAR(32) NOT NULL DEFAULT '';
```

#### 2. 填充哈希值
使用哈希函数（如 `MD5`）为已有数据填充哈希值：
```sql
UPDATE users SET email_hash = MD5(email);
```

#### 3. 创建 B-Tree 索引
在 `email_hash` 列上创建普通索引：
```sql
CREATE INDEX idx_email_hash ON users (email_hash);
```

#### 4. 查询时使用哈希值
查询时先计算目标 `email` 的哈希值，再通过哈希列快速定位记录：
```sql
SELECT * FROM users 
WHERE email_hash = MD5('user@example.com') 
  AND email = 'user@example.com'; -- 二次校验防止哈希冲突
```

---

### **维护哈希值**
为保证哈希值与原始数据一致，需在插入/更新数据时自动维护 `email_hash` 列。可以通过以下两种方式实现：

#### 1. 应用层维护
在业务代码中插入或更新数据时，显式计算哈希值：
```sql
INSERT INTO users (email, email_hash) 
VALUES ('user@example.com', MD5('user@example.com'));
```

#### 2. 使用触发器自动维护
创建 `BEFORE INSERT` 和 `BEFORE UPDATE` 触发器：
```sql
DELIMITER //
CREATE TRIGGER hash_email_insert 
BEFORE INSERT ON users 
FOR EACH ROW 
BEGIN
  SET NEW.email_hash = MD5(NEW.email);
END;
//

CREATE TRIGGER hash_email_update 
BEFORE UPDATE ON users 
FOR EACH ROW 
BEGIN
  IF NEW.email <> OLD.email THEN
    SET NEW.email_hash = MD5(NEW.email);
  END IF;
END;
//
DELIMITER ;
```

---

### **注意事项**
1. **哈希冲突**  
   哈希函数（如 `MD5`）存在理论上的冲突可能，查询时必须同时校验原始字段（如 `email`）。
   
2. **哈希函数选择**  
   - `MD5`：生成 32 位哈希值，冲突概率极低但非零。
   - `SHA1`：生成 40 位哈希值，冲突概率更低，但占用更多空间。
   - 自定义哈希算法需谨慎，可能降低散列均匀性。

3. **适用场景**  
   - 适合等值查询（`=`），不支持范围查询（`BETWEEN`、`>` 等）。
   - 对长文本（如 URL、地址）或复合条件的查询优化效果显著。

---

### **性能对比**
| 方法                | 索引大小 | 查询速度 | 适用场景               |
|---------------------|----------|----------|------------------------|
| 直接对 `email` 建索引 | 大       | 快       | 精确查询、范围查询     |
| 哈希列 + B-Tree      | 小       | 更快     | 仅精确查询             |

---

通过这种方法，可以在 MySQL 中实现类似哈希索引的效果，尤其适合对长字段进行高效等值查询的场景。

