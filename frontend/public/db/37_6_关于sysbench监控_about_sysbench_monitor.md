# 使用 sysbench 压测 MySQL 的最简单步骤

以下是使用 sysbench 对 MySQL 进行性能测试的最简流程：

## 1. 安装 sysbench

```bash
# Ubuntu/Debian
sudo apt-get install sysbench

# CentOS/RHEL
sudo yum install sysbench

# 或者从源码安装最新版
curl -s https://packagecloud.io/install/repositories/akopytov/sysbench/script.rpm.sh | sudo bash
sudo yum -y install sysbench
```

## 2. 准备测试数据

```bash
sysbench oltp_read_write \
--db-driver=mysql \
--mysql-host=127.0.0.1 \
--mysql-port=3306 \
--mysql-user=root \
--mysql-password=yourpassword \
--mysql-db=sbtest \
--tables=10 \
--table-size=10000 \
prepare
```

参数说明：
- `--tables=10`：创建10张测试表
- `--table-size=10000`：每张表插入10000行数据

## 3. 运行基准测试

```bash
sysbench oltp_read_write \
--db-driver=mysql \
--mysql-host=127.0.0.1 \
--mysql-port=3306 \
--mysql-user=root \
--mysql-password=yourpassword \
--mysql-db=sbtest \
--tables=10 \
--table-size=10000 \
--threads=4 \
--time=60 \
--report-interval=10 \
run
```

关键参数：
- `--threads=4`：使用4个并发线程
- `--time=60`：测试持续60秒
- `--report-interval=10`：每10秒报告一次中间结果

## 4. 清理测试数据

```bash
sysbench oltp_read_write \
--db-driver=mysql \
--mysql-host=127.0.0.1 \
--mysql-port=3306 \
--mysql-user=root \
--mysql-password=yourpassword \
--mysql-db=sbtest \
cleanup
```

## 常用测试模式

除了`oltp_read_write`，还可以使用以下测试模式：

1. **只读测试**：
   ```bash
   sysbench oltp_read_only ...
   ```

2. **只写测试**：
   ```bash
   sysbench oltp_write_only ...
   ```

3. **点查询测试**：
   ```bash
   sysbench oltp_point_select ...
   ```

4. **更新索引测试**：
   ```bash
   sysbench oltp_update_index ...
   ```

## 结果解读

测试结果中关键指标：
- **queries per second (qps)**：每秒查询数
- **transactions per second (tps)**：每秒事务数
- **latency (95th percentile)**：95%请求的延迟

通过调整`--threads`参数值，可以测试MySQL在不同并发下的性能表现。