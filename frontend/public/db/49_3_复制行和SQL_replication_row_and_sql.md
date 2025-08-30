---

### MySQL 主备复制原理详解

MySQL 主备复制（Master-Slave Replication）是一种通过将主库（Master）的数据变更同步到一个或多个备库（Slave）的机制，用于实现数据冗余、负载均衡和故障恢复。以下是其核心原理及关键组件的详细说明：

---

#### **一、复制的基本流程**
1. **主库记录变更**：  
   主库将数据变更操作（如 DML 和 DDL）写入 **二进制日志（Binary Log, Binlog）**。  
2. **备库拉取日志**：  
   备库通过 I/O 线程（`IO Thread`）从主库的 Binlog 中读取变更事件，并写入本地的 **中继日志（Relay Log）**。  
3. **备库应用变更**：  
   备库的 SQL 线程（`SQL Thread`）读取 Relay Log 中的事件，按顺序重放这些操作，完成数据同步。

---

#### **二、复制的核心组件**
| **组件**            | **作用**                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **二进制日志 (Binlog)** | 主库记录所有数据变更的日志文件，支持三种格式：`Statement`、`Row`、`Mixed`。 |
| **中继日志 (Relay Log)** | 备库从主库拉取的 Binlog 事件暂存文件，供 SQL 线程重放。                   |
| **IO Thread**        | 备库用于连接主库并拉取 Binlog 的线程。                                    |
| **SQL Thread**       | 备库用于解析并执行 Relay Log 中的事件的线程。                             |

---

#### **三、复制格式：基于语句 vs 基于行**
MySQL 支持三种 Binlog 格式，影响复制的行为和兼容性：

##### **1. 基于语句的复制（Statement-Based Replication, SBR）**
- **原理**：  
  记录实际执行的 SQL 语句（如 `INSERT INTO users VALUES (1, 'Alice')`）。  
- **优点**：  
  - 日志量小，节省存储和网络带宽。  
  - 可读性强，便于人工审计。  
- **缺点**：  
  - 非确定性操作（如 `RAND()`、`NOW()`）可能导致主备数据不一致。  
  - 依赖上下文（如存储过程、触发器）可能引发错误。  
- **配置**：  
  ```sql
  SET GLOBAL binlog_format = 'STATEMENT';
  ```

##### **2. 基于行的复制（Row-Based Replication, RBR）**
- **原理**：  
  记录每行数据的变更（如 `UPDATE users SET name='Bob' WHERE id=1` 转为变更前后的行数据）。  
- **优点**：  
  - 数据一致性高，避免非确定性操作的问题。  
  - 适合高并发和大事务场景。  
- **缺点**：  
  - 日志量大（尤其是全表更新）。  
  - 可读性差，难以直接解析。  
- **配置**：  
  ```sql
  SET GLOBAL binlog_format = 'ROW';
  ```

##### **3. 混合模式（Mixed）**
- **原理**：  
  默认使用 SBR，仅在可能引发不一致时自动切换为 RBR。  
- **优点**：  
  - 平衡日志量和数据一致性。  
- **配置**：  
  ```sql
  SET GLOBAL binlog_format = 'MIXED';
  ```

---

#### **四、复制相关文件**
| **文件**               | **作用**                                                                 |
|------------------------|-------------------------------------------------------------------------|
| **Binlog 文件**         | 主库生成，文件名如 `mysql-bin.000001`，记录所有数据变更。                 |
| **Relay Log 文件**      | 备库生成，文件名如 `relay-bin.000002`，临时存储从主库拉取的 Binlog 事件。 |
| **Master Info 文件**    | 备库保存主库连接信息（如主机名、用户名），默认位于 `master.info`。         |
| **Relay Log Info 文件** | 备库记录 Relay Log 应用位置，默认位于 `relay-log.info`。                   |

---

#### **五、复制过滤器**
复制过滤器允许备库仅同步特定的数据库或表，减少不必要的数据传输和存储。

##### **1. 主库过滤（不推荐）**
- **配置**：  
  通过 `binlog-do-db` 或 `binlog-ignore-db` 控制主库写入 Binlog 的数据库。  
- **缺点**：  
  主库过滤可能导致数据丢失，备库无法恢复未记录的变更。

##### **2. 备库过滤（推荐）**
- **配置**：  
  在备库配置 `replicate-do-db`、`replicate-ignore-db` 或 `replicate-wild-do-table`。  
- **示例**：  
  ```ini
  # my.cnf 配置
  [mysqld]
  replicate-do-db = orders
  replicate-ignore-db = logs
  replicate-wild-do-table = sales.%
  ```

---

#### **六、主备复制配置步骤**
1. **主库配置**：  
   ```ini
   # my.cnf
   [mysqld]
   server-id = 1
   log_bin = /var/lib/mysql/mysql-bin
   binlog_format = ROW
   ```

2. **创建复制用户**：  
   ```sql
   CREATE USER 'repl'@'%' IDENTIFIED BY 'password';
   GRANT REPLICATION SLAVE ON *.* TO 'repl'@'%';
   ```

3. **备库配置**：  
   ```ini
   # my.cnf
   [mysqld]
   server-id = 2
   relay_log = /var/lib/mysql/relay-bin
   read_only = ON
   ```

4. **启动复制**：  
   ```sql
   CHANGE MASTER TO
     MASTER_HOST = 'master_ip',
     MASTER_USER = 'repl',
     MASTER_PASSWORD = 'password',
     MASTER_LOG_FILE = 'mysql-bin.000001',
     MASTER_LOG_POS = 154;
   START SLAVE;
   ```

5. **验证状态**：  
   ```sql
   SHOW SLAVE STATUS\G
   -- 确保 Slave_IO_Running 和 Slave_SQL_Running 均为 Yes
   ```

---

#### **七、常见问题与解决方案**
1. **主备数据不一致**：  
   - 使用 `pt-table-checksum` 校验数据一致性，`pt-table-sync` 修复差异。  
2. **复制延迟**：  
   - 优化主库写入效率，减少大事务；备库使用并行复制（`slave_parallel_workers`）。  
3. **复制中断**：  
   - 跳过错误（`SET GLOBAL SQL_SLAVE_SKIP_COUNTER=1;`）或手动修复后重启复制。

---

### **总结**
- **复制方式选择**：  
  - OLTP 场景推荐 **Row-Based Replication**（高一致性）。  
  - 审计需求场景可考虑 **Mixed 模式**。  
- **过滤器配置**：  
  - 优先在备库过滤，避免主库数据丢失风险。  
- **监控与维护**：  
  - 定期检查 `SHOW SLAVE STATUS`，监控延迟和错误日志。  

通过合理配置主备复制，可以实现高可用性和数据冗余，是 MySQL 架构设计的核心组件之一。


