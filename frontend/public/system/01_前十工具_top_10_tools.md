以下是一份 Linux 系统监控必备的 **20 个工具/指令** 及简明教程，覆盖 CPU、内存、磁盘、网络等核心指标监控：

---

### 一、系统概览类
1. **top**  
   **用途**：实时查看进程资源占用（CPU/内存）  
   **用法**：  
   ```bash
   top               # 默认动态刷新
   top -p [PID]      # 监控指定进程
   top -u [用户名]    # 按用户过滤进程
   ```

2. **htop** (需安装)  
   **用途**：增强版 top，支持颜色标记和鼠标操作  
   **安装**：`sudo apt install htop` 或 `yum install htop`  
   **用法**：  
   ```bash
   htop              # 交互式界面，按 F1 查看快捷键
   ```

3. **glances** (需安装)  
   **用途**：全系统监控仪表盘（含温度、磁盘IO等）  
   **安装**：`pip install glances`  
   **用法**：`glances`

---

### 二、内存监控
4. **vmstat**  
   **用途**：虚拟内存统计（含进程、内存分页、块IO）  
   **示例**：  
   ```bash
   vmstat 1          # 每秒刷新一次
   ```

5. **free**  
   **用途**：查看物理内存和交换分区使用量  
   **示例**：  
   ```bash
   free -h           # 以易读单位显示（GB/MB）
   ```

---

### 三、磁盘 I/O 监控
6. **iostat**  
   **用途**：磁盘 I/O 和 CPU 使用率统计  
   **示例**：  
   ```bash
   iostat -dx 1      # 每秒显示扩展磁盘统计
   ```

7. **iotop** (需安装)  
   **用途**：实时监控进程磁盘 I/O  
   **安装**：`sudo apt install iotop`  
   **用法**：`iotop -o`（显示活跃 I/O 进程）

---

### 四、网络监控
8. **netstat**  
   **用途**：网络连接和端口监听状态  
   **示例**：  
   ```bash
   netstat -tulnp    # 查看 TCP/UDP 监听端口及进程
   ```

9. **iftop** (需安装)  
   **用途**：实时网络带宽监控（按主机/IP 排序）  
   **安装**：`sudo apt install iftop`  
   **用法**：`iftop -i eth0`（指定网卡）

10. **tcpdump**  
    **用途**：抓取网络数据包（需 root 权限）  
    **示例**：  
    ```bash
    tcpdump -i eth0 port 80  # 捕获 eth0 网卡 80 端口流量
    ```

---

### 五、进程级分析
11. **pidstat** (sysstat 包)  
    **用途**：监控进程的 CPU、内存、IO 等  
    **安装**：`sudo apt install sysstat`  
    **示例**：  
    ```bash
    pidstat -d 1      # 每秒显示进程磁盘 I/O
    ```

12. **lsof**  
    **用途**：列出被进程打开的文件/网络连接  
    **示例**：  
    ```bash
    lsof -i :80       # 查看占用 80 端口的进程
    ```

---

### 六、高级工具
13. **sar** (sysstat 包)  
    **用途**：历史性能数据收集与分析  
    **示例**：  
    ```bash
    sar -u 1 3        # 每秒采样 CPU 使用率，共 3 次
    ```

14. **dstat**  
    **用途**：多功能资源统计工具（替代 vmstat/iostat）  
    **安装**：`sudo apt install dstat`  
    **示例**：  
    ```bash
    dstat -cmsdn       # 组合显示 CPU、内存、磁盘、网络
    ```

15. **nmon**  
    **用途**：交互式系统监控（支持导出 CSV）  
    **安装**：`sudo apt install nmon`  
    **用法**：`nmon` → 按 `c` (CPU)、`m` (内存) 切换视图

---

### 七、日志与追踪
16. **journalctl**  
    **用途**：查看 systemd 日志（支持按时间/服务过滤）  
    **示例**：  
    ```bash
    journalctl -u nginx --since "2025-04-12"
    ```

17. **strace**  
    **用途**：追踪进程的系统调用  
    **示例**：  
    ```bash
    strace -p [PID]    # 追踪运行中进程
    ```

---

### 八、综合工具包
18. **tsar** (腾讯云推荐)  
    **用途**：采集、存储和展示系统指标  
    **安装**：  
    ```bash
    git clone https://github.com/alibaba/tsar.git
    cd tsar && make && make install
    ```
    **用法**：`tsar --cpu --mem -i 1`（每秒采集 CPU/内存）

19. **Prometheus + Grafana**  
    **用途**：构建可视化监控系统（适合长期监控）  
    **步骤**：  
    1. 安装 Prometheus 采集数据  
    2. 使用 Node Exporter 收集主机指标  
    3. Grafana 配置仪表盘展示

---

### 九、压测工具
20. **stress**  
    **用途**：模拟 CPU/内存/磁盘负载  
    **示例**：  
    ```bash
    stress --cpu 4 --vm 2 --vm-bytes 512M  # 模拟 4 核 CPU 和 2 个 512MB 内存进程
    ```

---

### 使用建议
- **快速诊断**：组合使用 `top` + `iostat` + `iftop`  
- **历史分析**：依赖 `sar` 或 `tsar` 的日志  
- **可视化**：优先配置 `Prometheus` + `Grafana`