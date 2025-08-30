---

### **深入分析“少卖”与数据不一致问题**

#### **1. 少卖的本质与典型场景**
**少卖**指系统实际可售库存被错误扣减，导致商品未售出但库存显示不足。其核心原因是 **缓存（Redis）与数据库（MySQL）数据不一致**，且缺乏回滚机制。以下是两种典型场景：

---

##### **场景1：订单创建失败导致Redis库存未回滚**
**流程步骤**：
1. **Redis预扣成功**：用户下单，Redis通过Lua脚本原子扣减库存（如从100→99）。
2. **生成订单失败**：后续业务逻辑失败（如优惠券校验异常、网络超时），订单未生成。
3. **库存未回滚**：系统未将Redis库存恢复为100，导致后续用户看到库存为99，但实际可售量应为100。

**结果**：
- **用户端**：显示库存99，但下单时可能因实际无订单生成而提示失败。
- **系统端**：Redis库存虚低，数据库库存未同步，实际少卖1件。

---

##### **场景2：多系统依赖不同数据源导致超卖**
**流程步骤**：
1. **主系统扣减Redis库存**：主秒杀系统扣减Redis库存至98。
2. **异步同步数据库失败**：数据库仍为100。
3. **其他系统读取数据库库存**：如营销系统直接查询数据库，认为库存充足，允许用户下单。
4. **超卖发生**：主系统实际可售库存仅98件，但其他系统基于数据库100件放行订单，导致超卖2件。

**结果**：
- **数据不一致性**：Redis（真实库存）与数据库（过时库存）差异。
- **业务风险**：超卖引发用户投诉或赔偿。

---

#### **2. 根本原因分析**

| **问题类型**       | **原因**                                                                 | **影响**                     |
|--------------------|--------------------------------------------------------------------------|------------------------------|
| **少卖**           | Redis扣减后业务失败，未回滚库存                                          | 可售库存虚减，收益损失       |
| **超卖**           | 多系统依赖不同数据源（Redis vs 数据库）                                  | 订单超量，需人工介入取消     |
| **数据不一致**     | 异步同步机制不完善（无重试/补偿）                                        | 运营决策失误，用户体验下降   |

---

#### **3. 解决方案设计**

##### **方案1：订单失败时回滚Redis库存**
- **实现逻辑**：
  1. **引入本地事务表**：在Redis预扣库存时，同步记录操作流水到本地任务表（含`预扣数量`、`状态`）。
  2. **订单生成失败触发回滚**：若后续流程失败，调用Redis Lua脚本回滚库存。
  ```lua
  -- Redis回滚库存Lua脚本
  local key = KEYS[1]    -- 库存Key
  local rollback_num = tonumber(ARGV[1])  -- 回滚数量
  redis.call("INCRBY", key, rollback_num)
  ```
- **流程时序图**：
  ```mermaid
  sequenceDiagram
    participant 用户 as 用户
    participant 订单服务 as 订单服务
    participant Redis as Redis
    participant 任务表 as 任务表

    用户 ->> 订单服务: 提交订单
    订单服务 ->> Redis: 预扣库存（Lua脚本）
    Redis -->> 订单服务: 扣减成功
    订单服务 ->> 任务表: 写入任务（状态=PENDING）
    订单服务 ->> 订单服务: 生成订单（失败）
    订单服务 ->> Redis: 回滚库存（Lua脚本）
    订单服务 ->> 任务表: 更新任务状态=ROLLBACK
  ```

---

##### **方案2：强制统一库存数据源**
- **实现逻辑**：
  1. **封装库存服务API**：所有系统必须通过统一接口查询或扣减库存，禁止直连数据库。
  2. **数据库只读从库**：运营后台或其他系统仅允许查询从库（数据延迟可控）。
  3. **缓存与数据库最终一致**：
     - 异步同步Worker保证Redis与数据库数据对齐。
     - 差异检测任务每小时运行，触发告警并自动修复。
     ```sql
     -- 校准脚本示例
     UPDATE db_sku_stock d
     JOIN redis_sku_stock r ON d.sku_id = r.sku_id
     SET d.stock = r.stock
     WHERE d.stock != r.stock;
     ```

---

##### **方案3：异步任务可靠性增强**
- **设计要点**：
  1. **任务表+重试机制**：所有异步操作（如库存同步）记录任务表，失败任务按策略重试。
  2. **消息队列ACK确认**：使用RocketMQ/Kafka的事务消息，确保异步任务必达。
  3. **人工干预兜底**：提供管理界面手动回滚或补偿库存。
- **任务表结构**：
  ```sql
  CREATE TABLE inventory_sync_task (
    task_id VARCHAR(64) PRIMARY KEY,
    sku_id INT NOT NULL,
    deduct_num INT NOT NULL,
    status ENUM('PENDING', 'SYNCED', 'FAILED'),
    retry_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  ```

---

#### **4. 关键代码示例**

##### **Redis预扣与回滚Lua脚本**
```lua
-- 预扣库存脚本：deduct_stock.lua
local key = KEYS[1]
local uuid = ARGV[1]
local deduct_num = tonumber(ARGV[2])

-- 防重校验
if redis.call("EXISTS", "dedup:"..uuid) == 1 then
    return 0
end

-- 扣减库存
local stock = tonumber(redis.call("GET", key))
if stock < deduct_num then
    return -1
end
redis.call("DECRBY", key, deduct_num)
redis.call("SET", "dedup:"..uuid, 1, "EX", 60)
return 1

-- 回滚库存脚本：rollback_stock.lua
local key = KEYS[1]
local rollback_num = tonumber(ARGV[1])
redis.call("INCRBY", key, rollback_num)
return 1
```

##### **订单服务回滚逻辑（Java伪代码）**
```java
public void createOrder(OrderRequest request) {
    // 1. Redis预扣库存
    boolean deductSuccess = redisClient.evalLua("deduct_stock.lua", skuKey, request.getUuid(), request.getNum());
    if (!deductSuccess) {
        throw new InventoryException("库存不足");
    }

    // 2. 写入本地任务表
    inventoryTaskDao.insertTask(request.getUuid(), skuId, request.getNum(), "PENDING");

    try {
        // 3. 生成订单（可能失败）
        orderService.generateOrder(request);
        // 4. 异步同步库存至数据库
        mqProducer.sendSyncStockMessage(skuId, request.getNum());
    } catch (Exception e) {
        // 5. 订单失败，回滚Redis库存
        redisClient.evalLua("rollback_stock.lua", skuKey, request.getNum());
        inventoryTaskDao.updateTaskStatus(request.getUuid(), "ROLLBACK");
        throw e;
    }
}
```

---

#### **5. 总结与建议**
- **少卖根因**：订单失败未回滚缓存 + 多系统数据源不一致。
- **解决关键**：  
  1. **事务性设计**：通过本地任务表+Redis Lua脚本实现“预扣-提交-回滚”闭环。  
  2. **数据源统一**：所有库存操作收敛至缓存，异步同步保障最终一致。  
  3. **监控对账**：定时任务检测缓存与数据库差异，自动修复或告警。  
- **架构演进**：  
  - 短期：通过任务表+重试减少少卖风险。  
  - 长期：引入分布式事务框架（如Seata）或切换至更强一致性存储（如TiDB）。

