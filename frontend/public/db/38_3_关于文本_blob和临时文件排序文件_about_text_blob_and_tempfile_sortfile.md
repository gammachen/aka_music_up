以下是对 MySQL 中使用 **TEXT/BLOB 类型**时触发磁盘临时表和文件排序的详细说明及优化建议：

---

### 一、核心概念说明
#### 1. 磁盘临时表（Disk-based Temporary Table）
当查询需要临时存储中间结果集且**超过内存限制**时，MySQL 会将内存临时表转换为磁盘临时表（默认使用 MyISAM 引擎）。

**触发场景**：
```sql
-- 包含 TEXT/BLOB 字段的 GROUP BY/ORDER BY
SELECT content, COUNT(*) 
FROM articles 
GROUP BY content; -- content 是 TEXT 类型

-- 大字段参与 JOIN 操作
SELECT a.id, b.file_data 
FROM table_a a
JOIN table_b b ON a.id = b.id; -- file_data 是 BLOB
```

#### 2. 文件排序（Filesort）
当排序操作**无法在内存中完成**时，MySQL 会使用文件排序算法，将数据分块存储到磁盘，最后合并结果。

**触发条件**：
- `ORDER BY` 或 `GROUP BY` 涉及 TEXT/BLOB 字段
- 排序数据量超过 `sort_buffer_size` 设定值

---

### 二、TEXT/BLOB 对性能的影响机制
#### ▶️ 内存计算限制规则
| 参数                | 默认值     | 作用                                 |
|--------------------|-----------|--------------------------------------|
| `max_heap_table_size` | 16MB     | 内存临时表最大尺寸阈值                |
| `tmp_table_size`      | 16MB     | 临时表内存分配上限                    |
| `sort_buffer_size`    | 256KB    | 每个排序线程使用的缓冲区大小          |

**计算规则**：
```bash
# 临时表内存占用估算公式
临时表大小 ≈ SUM(字段长度) × 行数

# 当 SUM(TEXT长度) × 行数 > tmp_table_size 时
# 强制转换为磁盘临时表
```

#### ▶️ TEXT/BLOB 的特殊性
1. **存储方式**：TEXT/BLOB 内容与行数据分离存储（即使启用 `innodb_file_per_table`）
2. **排序限制**：`max_sort_length` 参数控制参与排序的前 N 字节（默认 1024）
   ```sql
   SHOW VARIABLES LIKE 'max_sort_length'; -- 可修改为 2048
   ```

---

### 三、问题诊断方法
#### 1. 使用 EXPLAIN 识别风险
```sql
EXPLAIN 
SELECT * FROM logs 
ORDER BY error_message; -- error_message 是 TEXT 类型
```
**输出关键字段**：
- `Using temporary`：使用临时表
- `Using filesort`：使用文件排序
- `Extra` 字段会显示详细信息

#### 2. 监控临时表创建
```sql
SHOW STATUS LIKE 'Created_tmp%';
```
| 状态变量                  | 含义                         |
|--------------------------|-----------------------------|
| Created_tmp_tables        | 内存临时表创建次数           |
| Created_tmp_disk_tables   | 磁盘临时表创建次数           |

---

### 四、优化方案
#### 1. 查询层优化
```sql
-- 避免在大字段上直接排序
SELECT id, LEFT(content, 100) AS preview 
FROM articles 
ORDER BY preview;

-- 使用覆盖索引减少数据量
ALTER TABLE articles ADD INDEX (author_id, created_at);
SELECT author_id, created_at 
FROM articles 
ORDER BY created_at; -- 无需访问 TEXT 字段
```

#### 2. 参数调优
```ini
# my.cnf 调整（根据服务器内存合理配置）
tmp_table_size = 64M
max_heap_table_size = 64M
sort_buffer_size = 2M
max_sort_length = 2048  -- 控制参与排序的文本长度
```

#### 3. 表结构设计优化
```sql
-- 分离大字段到副表
CREATE TABLE main_table (
    id INT PRIMARY KEY,
    title VARCHAR(255),
    meta_data JSON
);

CREATE TABLE detail_table (
    id INT PRIMARY KEY,
    main_id INT,
    content TEXT,
    FOREIGN KEY (main_id) REFERENCES main_table(id)
);
```

---

### 五、不同存储引擎对比
| 行为                | InnoDB                          | MyISAM               |
|---------------------|---------------------------------|----------------------|
| 磁盘临时表格式       | 默认 InnoDB（8.0+）             | MyISAM               |
| 大字段存储           | 行外存储（off-page）            | 行内存储（可能分页） |
| 事务支持             | 支持                            | 不支持               |
| 文件排序效率         | 较高（8.0+ 改进排序算法）       | 较低                 |

---

### 六、典型案例分析
#### 场景：博客系统按文章内容排序
```sql
-- 原始问题查询
SELECT * FROM posts 
WHERE category = 'tech' 
ORDER BY content LIMIT 100;
```
**优化步骤**：
1. 创建摘要字段：
   ```sql
   ALTER TABLE posts 
     ADD COLUMN content_preview VARCHAR(500) 
     GENERATED ALWAYS AS (LEFT(content, 500)) VIRTUAL;
   ```
2. 使用摘要排序：
   ```sql
   SELECT id, title, content_preview 
   FROM posts 
   WHERE category = 'tech' 
   ORDER BY content_preview 
   LIMIT 100;
   ```

**优化效果**：
- 临时表内存占用从平均 200MB 降至 15MB
- 执行时间从 2.3 秒缩短至 0.4 秒

---

### 七、扩展知识：文件排序模式
1. **单路排序（Single-pass）**  
   - 将所需字段和排序键一起放入 `sort_buffer`  
   - 8.0+ 默认模式，减少磁盘访问次数

2. **双路排序（Two-pass）**  
   - 仅将排序键和行指针放入缓冲区  
   - 需二次回表读取数据，效率较低