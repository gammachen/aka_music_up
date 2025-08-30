---

### **8.4 配置内存使用**

---

#### ** 1 MySQL 可以使用多少内存？**
**核心内容**：  
MySQL 的内存使用由多个组件构成，包括全局缓存（如缓冲池）、连接级内存、线程缓存等。其总内存占用可通过以下公式估算：  
```text
总内存 = 全局缓存 + (每个连接内存 × 最大连接数) + 操作系统保留内存
```

**最佳实践**：  
1. **全局内存限制**：  
   - 通过 `innodb_buffer_pool_size`（InnoDB 缓冲池）和 `key_buffer_size`（MyISAM 键缓存）控制主要全局缓存。  
   - 推荐设置：  
     - `innodb_buffer_pool_size` 占物理内存的 **60%~80%**（例如 64GB 内存分配 40~50GB）。  
     - `key_buffer_size` 仅在 MyISAM 表较多时分配，建议不超过 **256MB**。  

2. **动态调整**：  
   - 监控 `SHOW GLOBAL STATUS` 和 `SHOW ENGINE INNODB STATUS`，根据实际负载调整内存分配。  
   - 使用工具如 `mysqltuner.pl` 或 `pt-mysql-summary` 分析内存使用情况。  

**示例配置**：  
```ini
# my.cnf 配置示例
[mysqld]
innodb_buffer_pool_size = 48G
key_buffer_size = 256M
max_connections = 200
```

---

#### ** 2 每个连接需要的内存**
**核心内容**：  
每个连接的内存消耗包括：  
- **会话级缓存**：排序缓冲区（`sort_buffer_size`）、连接缓冲区（`join_buffer_size`）。  
- **临时表**：内存临时表（`tmp_table_size`）。  
- **其他开销**：线程栈（`thread_stack`，默认 256KB）。

**最佳实践**：  
1. **优化连接级参数**：  
   - 限制 `max_connections` 避免内存耗尽（例如 200~500）。  
   - 降低 `sort_buffer_size` 和 `join_buffer_size`（默认 256KB，可设为 1~4MB）。  
   - 设置 `tmp_table_size` 和 `max_heap_table_size` 为 **16~64MB**，避免大临时表转为磁盘存储。  

2. **计算公式**：  
   ```text
   单连接内存 ≈ sort_buffer_size + join_buffer_size + read_buffer_size + thread_stack
   ```  
   若单连接需 4MB，`max_connections=500`，则总连接内存 ≈ 2GB。  

**示例配置**：  
```ini
[mysqld]
sort_buffer_size = 2M
join_buffer_size = 2M
tmp_table_size = 64M
max_heap_table_size = 64M
max_connections = 300
```

---

#### ** 3 为操作系统保留内存**
**核心内容**：  
操作系统需要内存用于文件缓存、内核进程等，若 MySQL 占用过多，可能导致系统不稳定或 OOM（内存溢出）。  

**最佳实践**：  
1. **内存保留规则**：  
   - 物理内存 ≤ 4GB：保留 **1~2GB**。  
   - 物理内存 > 4GB：保留 **10%~20%**（例如 64GB 内存保留 6~12GB）。  

2. **监控工具**：  
   - `free -h` 查看系统剩余内存。  
   - `vmstat 1` 监控 swap 使用情况（若频繁 swap，需减少 MySQL 内存分配）。  

**示例计算**：  
```text
总内存：64GB
MySQL 全局缓存：48GB
连接内存：2GB
操作系统保留：64 × 0.15 = 9.6GB
剩余内存：64 - 48 - 2 - 9.6 = 4.4GB（足够系统运行）
```

---

#### ** 4 为缓存分配内存**
**核心内容**：  
MySQL 的缓存包括查询缓存、表缓存、InnoDB 缓冲池等，需合理分配以提高性能。  

**最佳实践**：  
1. **查询缓存（Query Cache）**：  
   - **禁用场景**：高并发写入或表频繁更新时，查询缓存会引发锁竞争，建议关闭：  
     ```ini
     query_cache_type = 0
     query_cache_size = 0
     ```  
   - **启用场景**：仅读多写少的应用，分配 **64~256MB**。  

2. **表缓存（Table Cache）**：  
   - 控制 `table_open_cache`（默认 2000）和 `table_definition_cache`（默认 1400）。  
   - 监控 `Open_tables` 和 `Opened_tables`，若 `Opened_tables` 持续增长，需增大 `table_open_cache`。  

3. **InnoDB 缓冲池**：  
   - 详见 ** 5**。  

**示例配置**：  
```ini
[mysqld]
query_cache_type = 0
table_open_cache = 4096
table_definition_cache = 2048
```

---

#### ** 5 InnoDB 缓冲池 (Buffer Pool)**
**核心内容**：  
InnoDB 缓冲池用于缓存数据页和索引页，是提升读性能的核心组件。  

**最佳实践**：  
1. **容量设置**：  
   - 分配物理内存的 **60%~80%**（例如 64GB 内存分配 48GB）。  
   - 通过 `innodb_buffer_pool_size` 配置。  

2. **多实例优化**：  
   - 若缓冲池 > 16GB，启用多实例（`innodb_buffer_pool_instances=8`），减少锁竞争。  

3. **监控与调优**：  
   - 查看命中率：  
     ```sql
     SHOW STATUS LIKE 'Innodb_buffer_pool_read%';
     命中率 = 1 - (Innodb_buffer_pool_reads / Innodb_buffer_pool_read_requests)
     ```  
   - 目标命中率 > **99%**，若低于此值，需增大缓冲池。  

**示例配置**：  
```ini
[mysqld]
innodb_buffer_pool_size = 48G
innodb_buffer_pool_instances = 8
```

---

#### ** 6 MyISAM 键缓存 (Key Caches)**
**核心内容**：  
MyISAM 键缓存用于缓存索引块，提升索引查询速度。  

**最佳实践**：  
1. **容量设置**：  
   - 分配 `key_buffer_size`，建议不超过 **256MB**（除非 MyISAM 表占比极高）。  
   - 监控索引未命中率：  
     ```sql
     SHOW STATUS LIKE 'Key_read%';
     未命中率 = Key_reads / Key_read_requests
     ```  
   - 目标未命中率 < 1%。  

2. **多键缓存**：  
   - 为不同表分配独立键缓存（适用于混合引擎环境）：  
     ```sql
     CACHE INDEX table1, table2 IN key_cache1;
     SET GLOBAL key_cache1.key_buffer_size = 128M;
     ```  

**示例配置**：  
```ini
[mysqld]
key_buffer_size = 256M
```

---

#### ** 7 线程缓存**
**核心内容**：  
线程缓存（`thread_cache_size`）用于复用已创建的线程，减少频繁创建线程的开销。  

**最佳实践**：  
1. **设置规则**：  
   - 默认值：`thread_cache_size = 8 + (max_connections / 100)`。  
   - 高并发场景建议设为 **50~100**。  

2. **监控指标**：  
   ```sql
   SHOW STATUS LIKE 'Threads_created';
   ```  
   - 若 `Threads_created` 持续增长，需增大 `thread_cache_size`。  

**示例配置**：  
```ini
[mysqld]
thread_cache_size = 100
```

---

#### ** 8 表缓存 (Table Cache)**
**核心内容**：  
表缓存用于缓存表文件描述符，避免频繁开关表文件。  

**最佳实践**：  
1. **参数配置**：  
   - `table_open_cache`：控制打开表的数量（默认 2000），建议设为 `max_connections × 2`。  
   - `table_definition_cache`：缓存表定义（默认 1400），建议与 `table_open_cache` 一致。  

2. **避免竞争**：  
   - 若出现 `Too many open tables` 错误，需增大 `table_open_cache`。  

**示例配置**：  
```ini
[mysqld]
table_open_cache = 4096
table_definition_cache = 4096
```

---

#### ** 9 InnoDB 数据字典 (Data Dictionary)**
**核心内容**：  
InnoDB 数据字典存储表结构和元数据信息，内存占用较小但需合理管理。  

**最佳实践**：  
1. **内存占用**：  
   - 数据字典内存由 `innodb_additional_mem_pool_size` 控制（已弃用，8.0 后自动管理）。  
   - 现代版本无需手动配置，重点关注 `INFORMATION_SCHEMA` 查询效率。  

2. **元数据管理**：  
   - 避免频繁执行 `ALTER TABLE`，减少数据字典锁竞争。  
   - 定期清理无用表（`DROP TABLE`）以减少数据字典体积。  

**监控工具**：  
```sql
SHOW ENGINE INNODB STATUS;  -- 查看 SEMAPHORES 部分锁等待情况
```

---

### **总结**
通过合理配置全局缓存、连接级内存、操作系统保留内存等，可显著提升 MySQL 性能和稳定性。关键原则包括：  
1. **缓冲池最大化**：优先分配内存给 InnoDB 缓冲池。  
2. **连接与线程优化**：限制连接数并复用线程。  
3. **缓存平衡**：避免过度分配导致系统内存不足。  
4. **持续监控**：通过性能工具动态调整参数。  

根据实际负载和硬件资源灵活调整参数，结合压力测试验证配置合理性。