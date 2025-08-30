以下是关于 **Locust** 的详细介绍，结合知识库中的信息整理而成：

---

### **1. Locust 简介**
- **定义**：  
  Locust 是一个 **开源的负载测试工具**，用于模拟高并发用户访问系统（如网站、API、服务等），以测试系统的性能和承载能力。它通过 Python 脚本定义用户行为，支持分布式部署，可模拟数以万计的虚拟用户（Virtual Users）。
- **核心特点**：  
  - **基于事件驱动**：使用 Python 的 `gevent` 库实现协程，单机可支持数千个并发用户，资源占用低。
  - **轻量且灵活**：通过 Python 脚本定义测试场景，无需复杂的配置文件。
  - **分布式扩展**：支持多机协同运行，轻松实现百万级并发测试。
  - **实时监控界面**：提供 Web UI 实时显示测试数据（如响应时间、吞吐量、失败率等），并支持动态调整负载。

---

### **2. Locust 的核心功能与优势**
#### **2.1 核心功能**
- **模拟用户行为**：  
  通过定义 `TaskSet` 或 `HttpUser` 类，编写用户操作的 Python 任务（如登录、请求接口、表单提交等）。
- **分布式测试**：  
  支持主从（Master-Slave）模式，多台机器协同生成负载，突破单机性能限制。
- **实时监控**：  
  提供 Web 界面（默认端口 `8089`）展示测试结果，包括响应时间分布、每秒请求数（RPS）、失败率等指标。
- **自定义扩展**：  
  可通过插件机制扩展功能（如集成监控工具、自定义报告格式）。

#### **2.2 优势**
- **高并发能力**：  
  通过 `gevent` 协程技术，单机可支持数千并发用户，适合测试高负载场景。
- **易用性**：  
  - 使用 Python 编写脚本，开发效率高。
  - 提供直观的 Web 界面，方便实时监控和调试。
- **灵活性**：  
  - 支持 HTTP/HTTPS 协议测试，并可通过自定义客户端扩展到其他协议（如 TCP、MQTT 等）。
  - 可设置任务权重、执行顺序，模拟真实用户行为。
- **开源免费**：  
  开源社区活跃，文档和插件资源丰富。

---

### **3. 安装与配置**
#### **3.1 安装**
- **依赖环境**：  
  Python 3.6+（推荐 Python 3.8+）。
- **安装命令**：  
  ```bash
  pip install locust  # 安装最新稳定版
  # 或指定版本（如 v2.33.1）
  pip install locust==2.33.1
  ```

#### **3.2 版本注意事项**
- **版本差异**：  
  - **v1.0 之前**：使用 `HttpLocust` 和 `TaskSet` 类定义测试场景。
  - **v2.0 之后**：推荐使用 `HttpUser` 替代 `HttpLocust`，任务权重通过 `@task` 装饰器直接设置，`task_set` 被 `tasks` 属性取代。

---

### **4. 核心概念与代码示例**
#### **4.1 核心类与装饰器**
- **`HttpUser` 类**：  
  继承自 `User`，内置 `client` 属性（基于 `requests.Session`），支持 HTTP 请求（如 `get`、`post`）。
- **`@task` 装饰器**：  
  标记一个方法为测试任务，可设置权重（如 `@task(3)` 表示该任务执行概率是其他任务的 3 倍）。
- **`@seq_task` 装饰器**：  
  定义任务的执行顺序（如 `@seq_task(1)` 表示第一个执行的任务）。

#### **4.2 示例脚本**
```python
from locust import HttpUser, task, between
import random

class MyUser(HttpUser):
    wait_time = between(1, 3)  # 用户请求间隔时间（1-3秒）
    host = "http://example.com"  # 测试目标地址

    @task(2)  # 权重为2，执行概率更高
    def login(self):
        data = {"username": "user", "password": "pass"}
        self.client.post("/login", json=data)

    @task(1)
    def get_data(self):
        self.client.get("/api/data")

    # 设置任务顺序（可选）
    @task
    class MyTaskSet(TaskSet):
        @seq_task(1)
        def first_task(self):
            self.client.get("/page1")

        @seq_task(2)
        def second_task(self):
            self.client.post("/submit", json={"key": "value"})
```

---

### **5. 使用流程**
#### **5.1 编写测试脚本**
1. 定义用户行为（`HttpUser` 或 `User` 类）。
2. 使用 `@task` 装饰器定义具体任务。
3. 设置并发用户数、请求间隔、权重等参数。

#### **5.2 运行测试**
- **Web 界面模式**：  
  ```bash
  locust -f your_script.py --host=http://target.com
  ```
  - 访问 `http://localhost:8089`，输入用户数和孵化率（Hatch Rate，即每秒启动的用户数），点击“Start”。
  
- **命令行模式（无界面）**：  
  ```bash
  locust -f your_script.py --headless -u 100 -r 10 -t 60s
  # 参数说明：
  # -u: 总用户数（100）
  # -r: 每秒启动用户数（10）
  # -t: 测试时长（60秒）
  ```

#### **5.3 分布式测试**
1. 启动 Master 节点：  
   ```bash
   locust -f your_script.py --master
   ```
2. 启动 Slave 节点：  
   ```bash
   locust -f your_script.py --worker --master-host <master_ip>
   ```
3. 通过 Master 的 Web 界面统一控制负载。

---

### **6. 性能指标与分析**
- **关键指标**：  
  - **响应时间（Response Time）**：请求从发送到响应完成的时间。
  - **吞吐量（Throughput）**：单位时间内系统处理的请求数（如 RPS）。
  - **失败率（Failure Rate）**：请求失败的比例。
- **分析方法**：  
  - 通过 Web 界面的图表观察指标变化，识别性能瓶颈。
  - 结合日志分析具体失败请求的原因（如超时、HTTP 错误码）。

---

### **7. 与 JMeter 的对比**
| **特性**       | **Locust**                          | **JMeter**                          |
|----------------|-------------------------------------|-------------------------------------|
| **并发机制**   | 基于协程（gevent），资源占用低      | 基于线程/进程，高并发时资源消耗大    |
| **脚本语言**   | Python（灵活，适合复杂逻辑）        | 基于 XML 配置，学习成本较高          |
| **分布式支持** | 原生支持，配置简单                  | 需额外配置，复杂度较高               |
| **实时监控**   | 内置 Web UI，交互友好              | 需附加插件（如 JMeter Dashboard）    |

---

### **8. 典型应用场景**
- **Web 应用性能测试**：模拟高并发用户访问，验证服务器响应能力。
- **API 压力测试**：测试 RESTful API 在高负载下的稳定性。
- **系统瓶颈定位**：通过逐步增加负载，识别系统在 CPU、内存、网络等方面的瓶颈。
- **容量规划**：根据测试结果预估系统可承载的用户量。

---

### **9. 注意事项**
1. **版本兼容性**：  
   - 使用 `HttpUser` 替代已废弃的 `HttpLocust`（v2.0+）。
   - 任务权重需通过 `@task` 装饰器直接设置。
2. **资源限制**：  
   - 单机并发数受限于 CPU 和网络带宽，需合理分配 Slave 节点。
3. **登录态保持**：  
   - 使用 `self.client` 的 `Session` 功能，自动维护 Cookie 或 Token。
4. **异常处理**：  
   - 在任务函数中通过 `catch_response` 参数捕获异常，标记失败请求。

---

### **10. 扩展与插件**
- **常用插件**：  
  - **locust-plugins**：提供自定义报告（如 CSV、InfluxDB）、分布式增强等功能。
  - **locust-grafana**：集成 Grafana 可视化监控。
- **自定义扩展**：  
  - 通过 `events` 模块监听测试事件（如请求成功、失败、启动、停止），实现自定义日志或告警。

---

### **参考资料**
- **官方文档**：[https://docs.locust.io](https://docs.locust.io)
- **GitHub 仓库**：[https://github.com/locustio/locust](https://github.com/locustio/locust)

### 更常规的启动指令

```shell
locust --web-host 127.0.0.1
```

