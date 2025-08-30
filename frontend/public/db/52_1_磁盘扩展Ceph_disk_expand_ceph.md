
针对自建机房 MySQL 数据库磁盘空间不足（200G 磁盘已用 150G）的扩容需求，需在最小成本、最短停机时间的前提下完成扩容。以下是分阶段方案：

---

### **一、紧急处理：临时释放空间（可选）**
若需争取更多操作时间，可先临时清理非关键数据：
```bash
# 清理 MySQL 二进制日志（确认已备份或不再需要）
PURGE BINARY LOGS BEFORE 'yyyy-mm-dd hh:mm:ss';

# 清理慢查询日志、通用日志（需在 my.cnf 中关闭相关日志后删除）
rm /var/lib/mysql/slow.log
```

---

### **二、最小成本扩容方案（推荐）**
#### **方案 1：物理替换磁盘（低成本，短停机）**
**适用场景**：服务器支持热插拔或可接受短暂停机更换磁盘。  
**步骤**：  
1. **准备新磁盘**：购买更大容量磁盘（如 500G），格式化为与原磁盘相同的文件系统（如 `ext4`）。  
2. **在线备份数据库**：  
   ```bash
   # 使用 Percona XtraBackup 进行热备份（无需停机）
   xtrabackup --backup --target-dir=/path/to/backup/
   ```
3. **停机切换磁盘**：  
   - 停止 MySQL 服务：`systemctl stop mysql`  
   - 替换旧磁盘为新磁盘，恢复备份：  
     ```bash
     xtrabackup --prepare --target-dir=/path/to/backup/
     xtrabackup --copy-back --target-dir=/path/to/backup/
     chown -R mysql:mysql /var/lib/mysql
     ```
   - 启动 MySQL：`systemctl start mysql`  
   **停机时间**：仅限磁盘替换和恢复操作（通常 10-30 分钟）。

#### **方案 2：挂载新磁盘并迁移数据（无需硬件替换）**
**适用场景**：服务器有空余磁盘插槽或支持挂载云盘。  
**步骤**：  
1. 挂载新磁盘（如 `/dev/sdb`）到临时目录（如 `/mnt/new_disk`）。  
2. 迁移 MySQL 数据目录：  
   ```bash
   rsync -av /var/lib/mysql/ /mnt/new_disk/mysql/
   ```
3. 修改 MySQL 配置指向新路径：  
   ```ini
   # /etc/my.cnf
   datadir = /mnt/new_disk/mysql
   ```
4. 重启 MySQL：`systemctl restart mysql`  
**停机时间**：仅限重启 MySQL（通常 1-5 分钟）。

---

### **三、长期规范化方案（避免未来扩容风险）**
#### **1. 启用 LVM 管理磁盘**
**步骤**：  
1. 将新磁盘初始化为 LVM 物理卷（PV）：  
   ```bash
   pvcreate /dev/sdb
   ```
2. 创建卷组（VG）和逻辑卷（LV）：  
   ```bash
   vgcreate mysql_vg /dev/sdb
   lvcreate -n mysql_lv -l 100%FREE mysql_vg
   mkfs.ext4 /dev/mysql_vg/mysql_lv
   mount /dev/mysql_vg/mysql_lv /var/lib/mysql
   ```
3. 迁移数据到 LVM 卷（参考方案 2 的 `rsync` 步骤）。  
**优势**：未来可直接添加新磁盘到 VG，在线扩容 LV，无需迁移数据。

#### **2. 部署 Ceph 存储（需额外硬件）**
**适用场景**：多节点环境，需高可用存储。  
**步骤**：  
1. 部署至少 3 个 Ceph 节点（OSD + Monitor）。  
2. 创建 RBD 块设备并挂载到 MySQL 服务器：  
   ```bash
   rbd create mysql-data --size 500G
   rbd map mysql-data
   mkfs.ext4 /dev/rbd/mysql-data
   mount /dev/rbd/mysql-data /var/lib/mysql
   ```
**优势**：支持动态扩容、快照和分布式高可用。  
**成本**：需至少 3 台服务器，适合长期规划。

---

### **四、关键注意事项**
1. **备份优先**：操作前务必全量备份（`mysqldump` + `xtrabackup`）。  
2. **测试验证**：在非生产环境模拟操作流程，避免兼容性问题。  
3. **监控与告警**：部署 Prometheus + Grafana 监控磁盘和数据库增长趋势。  
4. **数据归档**：定期清理历史数据或归档至低成本存储（如 MinIO）。

---

### **五、方案对比**
| 方案                | 成本   | 停机时间  | 复杂度 | 未来扩展性       |
|---------------------|--------|-----------|--------|------------------|
| 物理替换磁盘        | 低     | 短（10-30m） | 低     | 有限             |
| 挂载新磁盘迁移      | 低     | 短（1-5m）  | 中     | 中等             |
| LVM 管理            | 中     | 中等      | 中     | 高（支持在线扩容）|
| Ceph 存储           | 高     | 长（部署时间） | 高     | 极高（分布式扩展）|

---

### **总结**
- **短期应急**：选择 **方案 1（物理替换磁盘）** 或 **方案 2（挂载新磁盘）**，快速扩容且成本最低。  
- **长期规划**：迁移至 **LVM** 或 **Ceph**，前者适合单机扩展，后者适合多节点高可用场景。  
- **自建 Ceph 建议**：仅当有至少 3 台空闲服务器且需长期弹性存储时采用，否则优先选择 LVM。