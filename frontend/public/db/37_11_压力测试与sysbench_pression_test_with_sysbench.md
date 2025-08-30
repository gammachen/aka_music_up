以下是基于您提供的三张表（A、B、test_lock）使用 sysbench 进行压测的详细步骤：

---

### **1. 环境准备**
#### 1.1 安装 sysbench
```bash
# Ubuntu/Debian
sudo apt-get install sysbench

# CentOS/RHEL
sudo yum install sysbench
```

#### 1.2 创建测试数据库
```sql
CREATE DATABASE sbtest;
```

---

### **2. 自定义 Lua 测试脚本**
由于默认的 sysbench OLTP 脚本不匹配您的表结构，需自定义脚本。  
创建 `custom_oltp.lua` 文件，内容如下：

```lua
#!/usr/bin/env sysbench

-- 定义表结构
sysbench.cmdline.options = {
    table_count = {"Number of tables", 3},  -- 对应 A/B/test_lock 三张表
    table_size = {"Number of rows per table", 10000},
    -- 添加标准参数映射
    tables = {"Alias for table_count", "table_count"},
    -- 添加其他命令行参数
    create_secondary = {"Whether to create secondary indexes", "off"},
    auto_inc = {"Whether to use auto_increment for primary keys", "off"}
 }
 
 function thread_init()
    drv = sysbench.sql.driver()
    con = drv:connect()
 end
 
 function thread_done()
    con:disconnect()
 end

-- 准备测试表
function prepare()
   local drv = sysbench.sql.driver()
   local con = drv:connect()
   local table_id = sysbench.opt.tables
   
   print("根据tables参数值 " .. table_id .. " 选择性初始化表")

   -- 根据tables参数值选择要初始化的表
   if table_id == 1 or table_id == 3 then
      -- 创建A表
      con:query([[CREATE TABLE IF NOT EXISTS sbtest.A (
         a1 INT NOT NULL PRIMARY KEY,
         a2 VARCHAR(100) NOT NULL,
         a3 INT NOT NULL
      )]]) 
      
      if table_id == 1 then
         -- 初始化A表数据
         print("正在初始化A表数据，共" .. sysbench.opt.table_size .. "行...")
         for i = 1, sysbench.opt.table_size do
            -- 生成随机值用于a3字段，确保部分数据可以与B表关联
            local a3_value = math.random(1, sysbench.opt.table_size)
            
            -- 构建插入语句
            local query = string.format(
               "INSERT INTO sbtest.A (a1, a2, a3) VALUES (%d, 'data-%d', %d)",
               i, i, a3_value
            )
            
            -- 执行插入
            con:query(query)
            
            -- 每1000行打印一次进度
            if i % 1000 == 0 then
               print(string.format("A表已插入 %d/%d 行数据", i, sysbench.opt.table_size))
            end
         end
      end
   end

   if table_id == 2 or table_id == 3 then
      -- 创建B表
      con:query([[CREATE TABLE IF NOT EXISTS sbtest.B (
         b1 INT NOT NULL PRIMARY KEY,
         b2 VARCHAR(100) NOT NULL,
         c2 INT NOT NULL
      )]]) 
      
      if table_id == 2 then
         -- 初始化B表数据
         print("正在初始化B表数据，共" .. sysbench.opt.table_size .. "行...")
         for i = 1, sysbench.opt.table_size do
            -- 生成随机值用于c2字段，确保部分数据可以与A表关联
            local c2_value = math.random(1, sysbench.opt.table_size)
            
            -- 构建插入语句
            local query = string.format(
               "INSERT INTO sbtest.B (b1, b2, c2) VALUES (%d, 'value-%d', %d)",
               i, i, c2_value
            )
            
            -- 执行插入
            con:query(query)
            
            -- 每1000行打印一次进度
            if i % 1000 == 0 then
               print(string.format("B表已插入 %d/%d 行数据", i, sysbench.opt.table_size))
            end
         end
      end
   end

   if table_id == 3 then
      -- 创建test_lock表（用于锁竞争测试）
      con:query([[CREATE TABLE IF NOT EXISTS sbtest.test_lock (
         id INT NOT NULL PRIMARY KEY,
         value INT NOT NULL
      )]]) 

      -- 初始化test_lock表
      con:query("INSERT INTO sbtest.test_lock VALUES (1, 0) ON DUPLICATE KEY UPDATE id=id")
      print("test_lock表初始化完成")
   end

   print("数据初始化完成！")
   con:disconnect()
end
 
-- 清理测试表
function cleanup()
   local drv = sysbench.sql.driver()
   local con = drv:connect()
   local table_id = sysbench.opt.tables
   
   print("根据tables参数值 " .. table_id .. " 选择性清理表")

   -- 根据tables参数值选择要清理的表
   if table_id == 1 or table_id == 3 then
      -- 删除A表
      print("正在删除A表...")
      con:query("DROP TABLE IF EXISTS sbtest.A")
      print("A表已删除")
   end

   if table_id == 2 or table_id == 3 then
      -- 删除B表
      print("正在删除B表...")
      con:query("DROP TABLE IF EXISTS sbtest.B")
      print("B表已删除")
   end

   if table_id == 3 then
      -- 删除test_lock表
      print("正在删除test_lock表...")
      con:query("DROP TABLE IF EXISTS sbtest.test_lock")
      print("test_lock表已删除")
   end

   print("数据清理完成！")
   con:disconnect()
end

function event()
    -- 随机操作：SELECT/UPDATE/INSERT/DELETE
    local choice = math.random(1, 4)
    local table_name = "sbtest.A"  -- 可扩展为随机选择表

    if choice == 1 then
        -- SELECT 操作（基于唯一索引）
        con:query(string.format("SELECT * FROM %s WHERE a1 = %d", table_name, math.random(1, sysbench.opt.table_size)))
    elseif choice == 2 then
        -- UPDATE 操作（更新非索引字段）
        con:query(string.format("UPDATE %s SET a2 = 'test%d' WHERE a1 = %d", table_name, math.random(1, 1000), math.random(1, sysbench.opt.table_size)))
    elseif choice == 3 then
        -- INSERT 操作（需处理唯一键冲突）
        local a1 = math.random(1, sysbench.opt.table_size * 2)  -- 50% 冲突概率
        con:query(string.format("INSERT INTO %s (a1, a2, a3) VALUES (%d, 'data%d', %d) ON DUPLICATE KEY UPDATE a3 = a3 + 1", table_name, a1, a1, a1))
    else
        -- DELETE 操作（删除随机行）
        con:query(string.format("DELETE FROM %s WHERE a1 = %d", table_name, math.random(1, sysbench.opt.table_size)))
    end

    -- 跨表 JOIN 操作（示例）
    if math.random(1, 10) == 1 then  -- 10% 概率执行 JOIN
        con:query("SELECT A.a1, B.b2 FROM sbtest.A JOIN sbtest.B ON A.a3 = B.c2 WHERE A.a1 = " .. math.random(1, sysbench.opt.table_size))
    end

    -- test_lock 表锁竞争模拟（可选）
    if math.random(1, 20) == 1 then  -- 5% 概率触发锁竞争
        con:query("BEGIN")
        con:query("SELECT * FROM sbtest.test_lock WHERE id = 1 FOR UPDATE")
        con:query("COMMIT")
    end
end
```

---

### **3. 准备测试数据**
#### 3.1 为每张表生成数据
```bash
# 生成 A 表数据
sysbench custom_oltp.lua \
--mysql-host=127.0.0.1 \
--mysql-port=3306 \
--mysql-user=root \
--mysql-password=123123csd \
--mysql-db=sbtest \
--tables=1 \
--table-size=10000 \
--create_secondary=off \
--auto_inc=off \
prepare

# 重复生成 B 表
sysbench custom_oltp.lua \
--mysql-host=127.0.0.1 \
--mysql-port=3306 \
--mysql-user=root \
--mysql-password=123123csd \
--mysql-db=sbtest \
--tables=2 \
--table-size=10000 \
--create_secondary=off \
--auto_inc=off \
prepare

# 生成 C 表数据
sysbench custom_oltp.lua \
--mysql-host=127.0.0.1 \
--mysql-port=3306 \
--mysql-user=root \
--mysql-password=123123csd \
--mysql-db=sbtest \
--tables=3 \
--table-size=10000 \
--create_secondary=off \
--auto_inc=off \
prepare
```

#### 3.2 手动处理唯一键冲突(备选，不用作)
由于 `a1` 和 `b1` 是唯一索引，需确保数据唯一性：
```sql
-- 示例：为 A 表插入数据
INSERT INTO sbtest.A (a1, a2, a3)
SELECT seq, CONCAT('a2-', seq), FLOOR(RAND() * 1000)
FROM seq_1_to_10000;
```

---

### **4. 运行压测**
```bash
sysbench custom_oltp.lua \
--mysql-host=127.0.0.1 \
--mysql-port=3306 \
--mysql-user=root \
--mysql-password=yourpassword \
--mysql-db=sbtest \
--tables=3 \          # 对应三张表
--table-size=10000 \
--threads=32 \        # 并发线程数
--time=600 \          # 测试时长（秒）
--report-interval=10 \ # 每10秒输出一次统计
--rate=100 \          # 目标每秒事务数
--rand-type=uniform \ # 均匀分布随机数
run

--- 实际运行的是这个，必须将注释与斜杆后面的空格去掉，否则指令会有异常
sysbench custom_oltp.lua \
--mysql-host=127.0.0.1 \
--mysql-port=3306 \
--mysql-user=root \
--mysql-password=123123csd \
--mysql-db=sbtest \
--tables=3 \
--table-size=10000 \
--threads=32 \
--time=600 \
--report-interval=10 \
--rate=100 \
--rand-type=uniform \
run
```

---

### **5. 监控与优化**
#### 5.1 实时监控指标
```bash
# 查看 MySQL 状态
mysqladmin -uroot -p ext | grep -E 'Queries|Threads_running|Innodb_row_lock%'

# 监控锁等待
SHOW ENGINE INNODB STATUS\G
```

#### 5.2 关键指标分析
- **TPS/QPS**：sysbench 最终报告中的 `transactions/sec` 和 `queries/sec`  
- **锁争用**：检查 `Innodb_row_lock_waits` 和 `Innodb_row_lock_time_avg`  
- **慢查询**：通过 `pt-query-digest` 分析慢日志  

---

### **6. 测试后清理**
```bash
sysbench custom_oltp.lua \
--mysql-host=127.0.0.1 \
--mysql-user=root \
--mysql-password=yourpassword \
--mysql-db=sbtest \
cleanup
```

---

### **扩展配置建议**
#### 针对不同表设计独立操作比例
在 Lua 脚本中分配权重：
```lua
function event()
   local table_choice = math.random(1, 3)
   if table_choice == 1 then
      -- 操作 A 表（40% 概率）
   elseif table_choice == 2 then
      -- 操作 B 表（40% 概率）
   else
      -- 操作 test_lock 表（20% 概率）
   end
end
```

#### 模拟真实负载
通过调整以下参数控制压力模型：
```bash
--oltp-read-only=off      # 读写混合
--oltp-point-selects=20   # 20% 点查询
--oltp-range-selects=10   # 10% 范围查询
--oltp-simple-updates=30  # 30% 简单更新
```

---

通过以上步骤，您可以基于现有表结构模拟复杂的并发场景，精准定位性能瓶颈。