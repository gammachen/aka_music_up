### 修改 MySQL 表结构的方法与底层原理

#### **一、修改表结构的方法**
MySQL 中修改表结构的核心方法是使用 **`ALTER TABLE`** 命令，具体操作包括：
1. **添加列**：`ALTER TABLE table_name ADD COLUMN column_name INT;`
2. **删除列**：`ALTER TABLE table_name DROP COLUMN column_name;`
3. **修改列定义**：
   - **`MODIFY COLUMN`**：修改列的数据类型或属性（不重命名）。
     ```sql
     ALTER TABLE table_name MODIFY COLUMN column_name VARCHAR(100) NOT NULL;
     ```
   - **`CHANGE COLUMN`**：重命名列或修改数据类型。
     ```sql
     ALTER TABLE table_name CHANGE COLUMN old_name new_name INT;
     ```
4. **重命名表**：`ALTER TABLE table_name RENAME TO new_table_name;`
5. **添加/删除索引**：
   ```sql
   ALTER TABLE table_name ADD INDEX index_name (column_name);
   ALTER TABLE table_name DROP INDEX index_name;
   ```

#### **二、底层操作原理**
MySQL 修改表结构的底层操作与存储引擎相关，常见情况如下：

1. **MyISAM 引擎**：
   - 执行大多数 `ALTER TABLE` 操作时，会**锁定表并重建整个表**，过程如下：
     - 创建临时表，复制数据到新结构。
     - 删除原表，重命名临时表为原表名。
   - **缺点**：长时间锁表，阻塞写入操作。

2. **InnoDB 引擎**：
   - **在线 DDL（Data Definition Language）**：支持部分操作无需锁表或短暂锁表。
     - **快速索引创建（FIC）**：添加二级索引时，直接修改元数据，无需复制数据。
     - **重建表操作**：修改列类型或删除列时，需重建表（复制数据到临时表）。
   - **在线 DDL 选项**：
     ```sql
     ALTER TABLE table_name 
       ADD COLUMN new_column INT, 
       ALGORITHM=INPLACE,  -- 尽量原地操作（不复制数据）
       LOCK=NONE;          -- 不锁定表
     ```

#### **三、`ALTER` 与 `MODIFY` 的区别**
1. **`ALTER TABLE`**：
   - 是一个**通用命令**，用于执行多种表结构修改操作（如添加列、重命名表）。
2. **`MODIFY COLUMN`**：
   - 是 `ALTER TABLE` 的**子命令**，专门用于修改列的属性（如数据类型、默认值），但不重命名列。
   - 若需重命名列，必须使用 **`CHANGE COLUMN`**。

**示例对比**：
- 使用 `MODIFY` 修改列类型：
  ```sql
  ALTER TABLE users MODIFY COLUMN age TINYINT UNSIGNED;
  ```
- 使用 `CHANGE` 重命名列并修改类型：
  ```sql
  ALTER TABLE users CHANGE COLUMN age user_age INT;
  ```

#### **四、推荐的操作模式**
为避免表结构修改对生产环境的影响，推荐以下方法：

1. **在线 DDL（InnoDB 专属）**：
   - 使用 `ALGORITHM=INPLACE` 和 `LOCK=NONE` 选项，减少锁表时间。
   - 示例：
     ```sql
     ALTER TABLE orders 
       ADD COLUMN discount DECIMAL(5,2), 
       ALGORITHM=INPLACE, 
       LOCK=NONE;
     ```

2. **使用工具辅助**：
   - **pt-online-schema-change**（Percona Toolkit）：
     - 通过创建影子表、同步数据和原子切换，实现无锁表结构变更。
     - 命令示例：
       ```bash
       pt-online-schema-change \
         --alter "ADD COLUMN email VARCHAR(255)" \
         D=database,t=users \
         --execute
       ```

3. **分阶段操作**：
   - 对大表分批次操作（如按主键范围分批更新），减少单次操作压力。

4. **低峰时段操作**：
   - 在业务低峰期执行 DDL，减少对用户的影响。

5. **测试环境验证**：
   - 在生产环境操作前，先在测试环境验证耗时和影响。

#### **五、典型场景与优化**
| **场景**               | **推荐方法**                     | **说明**                              |
|-------------------------|----------------------------------|---------------------------------------|
| **添加索引**           | 使用 `ALGORITHM=INPLACE`         | InnoDB 支持在线添加二级索引，不锁表。 |
| **修改列类型**         | 使用 pt-online-schema-change     | 避免锁表，适合大表。                  |
| **重命名列**           | 直接使用 `CHANGE COLUMN`         | 快速完成，锁表时间短。                |
| **删除大表列**         | 分阶段删除+低峰操作              | 减少单次操作压力。                    |

#### **六、总结**
- **底层操作**：多数 `ALTER` 操作在 InnoDB 中需重建表，但部分操作（如加索引）支持在线完成。
- **`ALTER` vs `MODIFY`**：`MODIFY` 是 `ALTER` 的子命令，用于修改列属性。
- **推荐模式**：优先使用在线 DDL 或工具（如 pt-online-schema-change），确保高可用性。

