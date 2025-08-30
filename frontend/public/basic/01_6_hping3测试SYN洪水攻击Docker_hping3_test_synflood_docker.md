### **1. Docker 容器的网络隔离**
**问题核心**：  
Docker 容器有自己的网络栈，宿主机的 `netstat` 默认只能查看宿主机自身的网络连接，而无法直接看到容器内部的连接状态（如 `SYN_RECV`）。因此，需要进入容器内部查看连接状态。

#### **验证步骤**：
1. **进入 Nginx 容器**：
   ```bash
   docker exec -it <容器名或ID> /bin/bash
   ```
   例如：
   ```bash
   docker exec -it my-nginx /bin/bash
   ```

2. **在容器内检查 `SYN_RECV` 状态**：
   ```bash
   netstat -ant | grep SYN_RECV
   ```
   或使用更详细的命令（如 `ss`）：
   ```bash
   ss -ant | grep SYN-RECV
   ```

#### **预期结果**：  
如果容器确实收到 `SYN` 请求，你应该会看到类似以下输出：
```plaintext
tcp 0 1 0.0.0.0:80 LISTEN -
tcp SYN_RECV 0 1 192.168.31.109:80 192.168.31.96:12345 SYN_RECV
```

---

### **2. SYNCookies 机制生效**
**问题核心**：  
如果宿主机启用了 **SYN Cookies**（即使在 macOS 中通过其他机制实现），系统会立即响应 `SYN` 请求并发送 `SYN-ACK`，但不会将未完成的连接保留在队列中，因此 `netstat` 或 `ss` 无法看到 `SYN_RECV` 状态。

#### **验证步骤**：
1. **检查内核参数（macOS）**：
   ```bash
   sysctl net.inet.tcp.syncookies
   ```
   - 如果值为 `1`，表示已启用。
   - 如果值为 `0`，则需要进一步排查其他原因。

2. **临时禁用 SYNCookies（仅测试用途）**：
   ```bash
   sudo sysctl -w net.inet.tcp.syncookies=0
   ```
   然后重新运行 `hping3` 测试，并再次检查 `netstat`。

---

### **3. 防火墙或网络设备拦截**
**问题核心**：  
宿主机或网络设备（如路由器）可能拦截了 `SYN` 包或 `SYN-ACK` 响应，导致连接状态未被记录。

#### **验证步骤**：
1. **检查宿主机的防火墙规则**：
   - **macOS 防火墙**：
     ```bash
     sudo pfctl -s rules
     ```
   - **Linux 防火墙**（如果宿主机是 Linux）：
     ```bash
     sudo iptables -L -n -v
     ```

2. **临时关闭防火墙**：
   - **macOS**：
     ```bash
     sudo pfctl -d
     ```
   - **Linux**：
     ```bash
     sudo ufw disable
     ```

3. **重新运行测试**：
   ```bash
   sudo hping3 -S 192.168.31.109 -p 8001 -c 1000 -V
   ```

---

### **4. 端口映射配置问题**
**问题核心**：  
Docker 的端口映射可能未正确配置，导致 `SYN` 请求未到达容器的 `80` 端口。

#### **验证步骤**：
1. **检查容器的端口映射**：
   ```bash
   docker ps
   ```
   确保输出中包含：
   ```
   0.0.0.0:8001->80/tcp
   ```

2. **检查容器内的 Nginx 监听端口**：
   进入容器后运行：
   ```bash
   netstat -ant | grep :80
   ```
   应显示 `LISTEN` 状态：
   ```plaintext
   tcp 0 0 0.0.0.0:80 LISTEN
   ```

---

### **5. 使用 `ss` 替代 `netstat`**
**问题核心**：  
`netstat` 在某些系统上可能无法显示短暂的连接状态（如 `SYN_RECV`），而 `ss` 更适合查看实时连接。

#### **命令**：
```bash
ss -ant | grep SYN-RECV
```

---

### **6. 检查 hping3 的响应**
**问题核心**：  
从 `hping3` 的输出来看，宿主机确实响应了 `SYN-ACK`（`flags=SA`），说明连接可能已进入 `ESTABLISHED` 状态，而非 `SYN_RECV`。

#### **分析输出**：
你的 `hping3` 输出显示：
```plaintext
sport=8001 flags=SA seq=3 win=65535 rtt=4.2 ms
```
- `SA` 表示 `SYN-ACK`，说明宿主机已成功响应 `SYN` 请求。
- 如果 `SYN Cookies` 启用，系统会立即发送 `SYN-ACK`，但不会将连接保留在队列中，因此 `SYN_RECV` 状态不会持久存在。

---

### **总结与解决方案**
#### **步骤 1：进入容器检查连接状态**
```bash
docker exec -it my-nginx /bin/bash
netstat -ant | grep SYN_RECV
```
如果看到 `SYN_RECV`，则说明宿主机的 `netstat` 无法直接查看容器内的状态。

#### **步骤 2：检查 SYNCookies 配置**
```bash
sysctl net.inet.tcp.syncookies
```
如果启用，尝试临时禁用并重新测试。

#### **步骤 3：使用 `ss` 命令**
```bash
ss -ant | grep SYN-RECV
```

#### **步骤 4：验证端口映射和防火墙**
确保端口映射正确，防火墙未拦截流量。

---

### **最终结论**
1. **主要原因是 Docker 的网络隔离**：宿主机的 `netstat` 无法直接看到容器内的 `SYN_RECV` 状态，需进入容器内部查看。
2. **SYN Cookies 的影响**：即使看到 `SYN-ACK` 响应，SYN Cookies 可能导致 `SYN_RECV` 状态未被持久记录。
3. **解决方案**：  
   - 在容器内运行 `netstat` 或 `ss` 查看连接状态。  
   - 如果需要分析攻击流量，建议在容器内安装监控工具（如 `tcpdump`）。

如果仍有疑问，可以进一步提供以下信息：  
- 容器内的 `netstat` 或 `ss` 输出。  
- 宿主机的防火墙规则。  
- 容器的启动命令和 `docker ps` 的完整输出。