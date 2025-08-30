### MySQL数据库日常维护工作指南

#### **一、备份与恢复**
1. **备份策略**  
   - **全量备份**：每日一次，使用`mysqldump`或`Percona XtraBackup`。  
     ```bash
     # 使用mysqldump备份
     mysqldump -u root -p --single-transaction --all-databases > full_backup.sql
     
     # 使用XtraBackup进行物理备份
     xtrabackup --backup --target-dir=/backup/full
     ```
   - **增量备份**：每小时一次，基于全量备份（仅限XtraBackup）。  
     ```bash
     xtrabackup --backup --target-dir=/backup/incr --incremental-basedir=/backup/full
     ```
   - **备份验证**：定期恢复备份到测试环境，检查数据完整性。

2. **备份存储**  
   - 本地磁盘 + 异地云存储（如AWS S3、阿里云OSS）。  
   - 加密敏感数据，设置保留策略（如保留最近7天备份）。

---

#### **二、性能监控与优化**
1. **实时监控**  
   - 使用工具监控关键指标：  
     - **QPS/TPS**：`SHOW GLOBAL STATUS LIKE 'Questions'`。  
     - **连接数**：`SHOW STATUS LIKE 'Threads_connected'`。  
     - **锁等待**：`SHOW ENGINE INNODB STATUS`。  
   - 推荐工具：Prometheus + Grafana、Percona Monitoring and Management (PMM)。

2. **慢查询分析**  
   - 开启慢查询日志：  
     ```sql
     SET GLOBAL slow_query_log = ON;
     SET GLOBAL long_query_time = 2;  -- 超过2秒的查询记录
     ```
   - 使用`pt-query-digest`分析日志：  
     ```bash
     pt-query-digest /var/log/mysql/slow.log
     ```

3. **索引优化**  
   - 定期检查未使用的索引：  
     ```sql
     SELECT * FROM sys.schema_unused_indexes;
     ```
   - 添加缺失索引（使用`EXPLAIN`分析执行计划）。

---

#### **三、安全性维护**
1. **权限管理**  
   - 遵循最小权限原则，定期审查用户权限：  
     ```sql
     SHOW GRANTS FOR 'user'@'host';
     ```
   - 删除未使用的账户：  
     ```sql
     DROP USER 'old_user'@'%';
     ```

2. **密码策略**  
   - 强制密码复杂度（MySQL 8.0+）：  
     ```sql
     SET GLOBAL validate_password.policy = STRONG;
     ```
   - 定期更换密码（如每90天）。

3. **漏洞与补丁**  
   - 订阅MySQL安全公告，及时应用补丁。  
   - 使用`mysql_secure_installation`加固新实例。

---

#### **四、日志管理**
1. **日志类型与配置**  
   - **错误日志**：记录启动、运行错误。  
     ```ini
     [mysqld]
     log_error = /var/log/mysql/error.log
     ```
   - **慢查询日志**：记录执行缓慢的查询。  
   - **二进制日志**：用于复制和点-in-time恢复。  
     ```ini
     [mysqld]
     log_bin = /var/log/mysql/mysql-bin.log
     expire_logs_days = 7
     ```

2. **日志轮换**  
   - 使用`logrotate`工具自动切割日志：  
     ```conf
     /var/log/mysql/error.log {
         daily
         rotate 30
         missingok
         compress
         postrotate
             mysqladmin flush-logs
         endscript
     }
     ```

---

#### **五、存储与碎片管理**
1. **磁盘空间监控**  
   - 监控表空间使用：  
     ```sql
     SELECT table_schema, SUM(data_length+index_length)/1024/1024 AS total_mb 
     FROM information_schema.tables 
     GROUP BY table_schema;
     ```
   - 使用`df -h`跟踪磁盘分区使用率。

2. **表碎片整理**  
   - 优化InnoDB表（低峰期执行）：  
     ```sql
     OPTIMIZE TABLE orders;
     ```
   - 重建表：  
     ```sql
     ALTER TABLE users ENGINE=InnoDB;
     ```

---

#### **六、主从复制与高可用**
1. **复制状态监控**  
   - 检查复制延迟和错误：  
     ```sql
     SHOW REPLICA STATUS\G
     -- 关注Seconds_Behind_Source, Last_IO_Error
     ```
   - 自动故障切换：使用MHA（Master High Availability）或Orchestrator。

2. **定期一致性校验**  
   - 使用`pt-table-checksum`检查主从数据一致性：  
     ```bash
     pt-table-checksum --replicate=test.checksums u=root,p=password
     ```

---

#### **七、版本升级与变更管理**
1. **版本升级**  
   - 测试环境验证：先在非生产环境测试升级步骤。  
   - 滚动升级（集群环境）：逐台升级从库，最后升级主库。

2. **Schema变更**  
   - 使用`pt-online-schema-change`无锁修改表结构：  
     ```bash
     pt-online-schema-change --alter "ADD COLUMN email VARCHAR(255)" D=database,t=users --execute
     ```

---

#### **八、日常检查清单**
| **任务**                 | **频率**   | **操作命令/步骤**                     |
|--------------------------|------------|----------------------------------------|
| 检查备份是否成功         | 每日       | 验证备份文件存在且可恢复              |
| 监控慢查询               | 实时       | `pt-query-digest /var/log/mysql/slow.log` |
| 清理旧日志               | 每周       | `PURGE BINARY LOGS BEFORE '2023-10-01'` |
| 优化碎片化表             | 每月       | `OPTIMIZE TABLE large_table;`          |
| 审查用户权限             | 每季度     | `SHOW GRANTS FOR 'user'@'host';`       |

---

#### **九、紧急恢复计划**
1. **数据误删恢复**：  
   - 从备份恢复 + 应用二进制日志到指定时间点。  
     ```bash
     mysqlbinlog --start-position=123456 mysql-bin.000001 | mysql -u root -p
     ```
2. **主库故障**：  
   - 提升从库为主库，调整应用连接配置。  

---

通过系统化的维护流程，可确保MySQL数据库的稳定性、高性能与安全性。根据实际业务需求调整维护频率，并结合自动化工具减少人工操作。

