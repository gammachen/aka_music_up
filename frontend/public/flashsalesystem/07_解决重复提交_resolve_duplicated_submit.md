---

### **秒杀系统防重复下单方案及对比**

---

#### **一、重复下单的场景与风险**  
1. **用户误操作**：网络延迟导致多次点击提交按钮。  
2. **恶意刷单**：脚本或工具绕过前端限制高频提交订单。  
3. **系统重试**：客户端或服务端因超时自动重试请求。  

**风险**：  
- **超卖**：同一商品被重复下单，导致库存不足。  
- **数据混乱**：同一用户生成多个订单，需人工介入处理。  
- **资损风险**：恶意用户利用漏洞套取优惠或商品。  

---

### **二、防重复下单的多种方案**  

#### **1. 前端控制（快速拦截）**  
- **实现方式**：  
  - 提交按钮置灰：用户点击后禁用按钮，防止多次点击。  
  - 倒计时控制：提交后显示倒计时，限制用户重复操作。  
- **优点**：简单易实现，减少用户误操作。  
- **缺点**：无法阻止绕过前端的恶意请求（如直接调用API）。  
- **适用场景**：辅助手段，需结合后端方案使用。  

---

#### **2. Token 机制（服务端幂等性）**  
- **实现方式**：  
  1. 用户进入下单页时，服务端生成唯一 Token 并返回给前端。  
  2. 提交订单时携带 Token，服务端校验 Token 是否存在且未被使用。  
  3. 校验通过后删除 Token，防止重复使用。  
- **代码示例**（Redis + Token）：  
  ```java
  // 生成 Token
  String token = UUID.randomUUID().toString();
  redisTemplate.opsForValue().set("order:token:" + userId + ":" + skuId, token, 5, TimeUnit.MINUTES);
  
  // 校验 Token
  String storedToken = redisTemplate.opsForValue().get("order:token:" + userId + ":" + skuId);
  if (token.equals(storedToken)) {
      redisTemplate.delete("order:token:" + userId + ":" + skuId);
      // 处理订单
  } else {
      throw new RepeatOrderException("重复提交");
  }
  ```  
- **优点**：有效防止重复提交，支持分布式环境。  
- **缺点**：需维护 Token 状态，增加系统复杂度。  
- **适用场景**：高并发秒杀场景，需强一致性保障。  

---

#### **3. 数据库唯一索引（最终兜底）**  
- **实现方式**：  
  - 在订单表中设置 `用户ID + 商品ID + 活动ID` 的组合唯一索引。  
  - 插入订单时数据库自动拦截重复数据。  
- **优点**：绝对可靠，无需额外开发逻辑。  
- **缺点**：  
  - 高并发下可能因唯一索引冲突导致插入性能下降。  
  - 无法区分“重复提交”与“正常并发请求”（需结合其他方案）。  
- **适用场景**：最终一致性保障，作为兜底方案。  

---

#### **4. 分布式锁（控制并发）**  
- **实现方式**：  
  - 使用 Redis 或 ZooKeeper 对用户ID或订单请求加锁。  
  - 锁粒度：用户级别（`lock:user:123`）或商品级别（`lock:sku:1001`）。  
  ```java
  // Redis 分布式锁（SETNX）
  String lockKey = "lock:order:" + userId + ":" + skuId;
  Boolean locked = redisTemplate.opsForValue().setIfAbsent(lockKey, "1", 10, TimeUnit.SECONDS);
  if (locked) {
      try {
          // 处理订单
      } finally {
          redisTemplate.delete(lockKey);
      }
  } else {
      throw new RepeatOrderException("请勿重复提交");
  }
  ```  
- **优点**：精确控制并发，防止同一用户重复请求。  
- **缺点**：锁粒度需谨慎设计，过度加锁可能影响性能。  
- **适用场景**：需要强一致性的高频操作。  

---

#### **5. Redis 防重标记（快速去重）**  
- **实现方式**：  
  - 用户提交订单后，在 Redis 中记录标记（如 `order:user:123:sku:1001`），设置过期时间（如5分钟）。  
  - 后续请求直接检查标记是否存在。  
  ```java
  String key = "order:user:" + userId + ":sku:" + skuId;
  if (redisTemplate.opsForValue().setIfAbsent(key, "1", 5, TimeUnit.MINUTES)) {
      // 处理订单
  } else {
      throw new RepeatOrderException("请勿重复提交");
  }
  ```  
- **优点**：高性能，适合瞬时高并发场景。  
- **缺点**：Redis 宕机可能导致数据丢失（需持久化或补偿）。  
- **适用场景**：秒杀高峰期快速拦截重复请求。  

---

#### **6. 请求参数去重（请求指纹）**  
- **实现方式**：  
  - 对请求参数（用户ID + 商品ID + 时间戳）生成唯一哈希值。  
  - 在 Redis 或内存中记录哈希值，短时间内重复请求直接拦截。  
  ```java
  String requestHash = DigestUtils.md5Hex(userId + skuId + System.currentTimeMillis() / 1000);
  if (redisTemplate.opsForValue().setIfAbsent("req:" + requestHash, "1", 10, TimeUnit.SECONDS)) {
      // 处理订单
  } else {
      throw new RepeatOrderException("重复请求");
  }
  ```  
- **优点**：灵活，适用于复杂请求参数。  
- **缺点**：可能因时间戳精度或参数变化导致漏判。  
- **适用场景**：API 接口防重，需结合业务参数设计。  

---

### **三、方案对比与选型建议**  

| **方案**           | **性能** | **可靠性** | **复杂度** | **适用阶段**         |  
|--------------------|----------|------------|------------|----------------------|  
| 前端控制           | 高       | 低         | 低         | 用户交互层           |  
| Token 机制         | 中       | 高         | 中         | 服务端幂等性         |  
| 数据库唯一索引     | 低       | 极高       | 低         | 数据持久层（兜底）   |  
| 分布式锁           | 中       | 高         | 高         | 高并发控制           |  
| Redis 防重标记     | 高       | 中         | 低         | 瞬时拦截             |  
| 请求参数去重       | 高       | 中         | 中         | API 接口层           |  

---

#### **四、组合方案建议**  
1. **分层拦截策略**：  
   - **前端层**：按钮置灰 + 倒计时，减少用户误操作。  
   - **网关层**：请求参数去重 + 限流（如每秒1次）。  
   - **服务层**：Token 机制 + Redis 防重标记。  
   - **数据层**：数据库唯一索引兜底。  

2. **示例流程**：  
   ```plaintext
   用户提交 → 前端拦截 → 网关校验请求指纹 → 服务端验证 Token → Redis 防重标记 → 处理订单 → 数据库唯一索引
   ```  

3. **异常处理**：  
   - 数据库唯一索引冲突时，提示“订单已存在”。  
   - Redis 标记失效时，通过数据库索引最终拦截。  

---

### **五、总结**  
防重复下单需结合业务场景选择多层级方案：  
- **高性能拦截**：前端控制 + Redis 防重标记。  
- **强一致性**：Token 机制 + 数据库唯一索引。  
- **灵活适配**：请求参数去重 + 分布式锁。  

最终通过组合设计，在用户体验、系统性能和数据一致性之间取得平衡。


