好的！以下是一个使用 Flask 和 Python 实现简单熔断器（Circuit Breaker）的示例。我们将通过模拟一个不稳定的外部服务调用来演示熔断机制。

---

### 依赖库
1. **Flask**：用于创建 Web 服务端点。
2. **circuitbreaker**：一个轻量级的熔断器库（需安装）。

```bash
pip install flask circuitbreaker
```

---

### 代码实现

```python
from flask import Flask, jsonify, request
from circuitbreaker import circuit
import random
import time

app = Flask(__name__)

# 配置熔断器参数
breaker = circuit(
    fail_max=3,  # 允许的最大连续失败次数
    reset_timeout=5  # 熔断器打开后，等待多少秒后进入半开状态
)

# 模拟一个不稳定的外部服务（50% 的概率失败）
def unstable_service():
    if random.random() < 0.5:
        raise Exception("External service is down!")
    return "Service response successful"

# 使用熔断器装饰器包装外部服务调用
@breaker
def safe_call_service():
    return unstable_service()

@app.route('/api/call-service', methods=['GET'])
def call_service():
    try:
        # 调用被熔断器保护的服务
        result = safe_call_service()
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        # 如果熔断器打开，返回错误信息
        return jsonify({"status": "error", "message": str(e)}), 503  # 503 Service Unavailable

@app.route('/circuit-status', methods=['GET'])
def circuit_status():
    # 获取熔断器状态
    status = {
        "state": breaker.current_state,  # 当前状态（closed/open/half-open）
        "fail_count": breaker.fail_counter,  # 当前失败计数
        "time_remaining": breaker.time_remaining,  # 熔断剩余时间（秒）
    }
    return jsonify(status)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

---

### 代码说明

#### 1. 熔断器配置
- `fail_max=3`：允许连续失败 3 次后熔断器打开。
- `reset_timeout=5`：熔断器打开后，等待 5 秒后进入半开状态，尝试恢复。

#### 2. 不稳定的外部服务模拟
`unstable_service` 函数模拟了一个 50% 概率失败的外部服务，用于测试熔断逻辑。

#### 3. 熔断器装饰器
`@breaker` 装饰器将 `safe_call_service` 函数包装在熔断器中：
- 当连续失败次数超过 `fail_max`，熔断器进入 **打开（Open）** 状态，后续请求直接抛出异常。
- 在熔断器打开后，`reset_timeout` 秒后进入 **半开（Half-Open）** 状态，允许一次请求尝试恢复：
  - 如果成功，熔断器关闭（Closed）。
  - 如果再次失败，重新打开。

#### 4. 端点说明
- **`/api/call-service`**：调用被熔断器保护的服务，返回成功或错误信息。
- **`/circuit-status`**：获取熔断器当前状态（如 `closed`/`open`/`half-open`）。

---

### 测试步骤

1. **启动服务**：
   ```bash
   python app.py
   ```

2. **连续调用 `/api/call-service`**：
   使用 `curl` 或 Postman 发送多次请求：
   ```bash
   curl http://localhost:5000/api/call-service
   ```

   - **初始状态**：可能返回成功或错误。
   - **触发熔断**：连续失败 3 次后，后续请求返回 `503 Service Unavailable`。
   - **查看状态**：访问 `/circuit-status` 确认熔断器状态：
     ```json
     {
       "state": "OPEN",
       "fail_count": 3,
       "time_remaining": 4  # 剩余熔断时间
     }
     ```

3. **等待熔断恢复**：
   - 等待 `reset_timeout`（5 秒）后，熔断器进入 **半开** 状态。
   - 再次调用 `/api/call-service`：
     - 如果成功，熔断器关闭。
     - 如果失败，熔断器重新打开。

---

### 扩展建议
1. **集成更复杂的熔断逻辑**：
   - 使用 `resilience4j`（Java 的 Python 实现）或 `tenacity`（支持重试、熔断）。
   - 示例：
     ```python
     from tenacity import retry, wait_fixed, stop_after_attempt

     @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
     def call_service_with_retry():
         return unstable_service()
     ```

2. **日志与监控**：
   - 记录熔断事件，结合 Prometheus/Grafana 监控熔断状态。

3. **降级策略**：
   - 在熔断打开时，返回预设的默认值或缓存数据：
     ```python
     try:
         result = safe_call_service()
     except Exception:
         return jsonify({"status": "warning", "data": "Fallback data"})
     ```

---

### 完整代码总结
这个示例展示了如何用 Flask 和 `circuitbreaker` 库实现一个简单的熔断器。你可以通过调整 `fail_max` 和 `reset_timeout` 参数，或集成更复杂的库（如 `tenacity`）来满足实际需求。
