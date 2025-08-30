以下是为金融系统设计的 **汇率因子详细设计方案**，包含表结构、精度控制、计算逻辑和最佳实践：

---

### 一、汇率表基础设计
#### 1. 表结构定义 (exchange_rate)
```sql
CREATE TABLE exchange_rate (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    source_currency CHAR(3) NOT NULL COMMENT '源货币代码 (ISO 4217)',
    target_currency CHAR(3) NOT NULL COMMENT '目标货币代码',
    rate DECIMAL(20, 12) NOT NULL COMMENT '汇率因子（精确到小数点后12位）',
    effective_start DATETIME NOT NULL COMMENT '生效开始时间',
    effective_end DATETIME DEFAULT NULL COMMENT '生效结束时间（NULL表示当前有效）',
    updated_by VARCHAR(64) NOT NULL COMMENT '最后修改人',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CHECK (source_currency <> target_currency),
    INDEX idx_currency_pair (source_currency, target_currency, effective_start)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

#### 2. 关键字段说明
| 字段               | 类型              | 约束/说明                                                                 |
|--------------------|-------------------|--------------------------------------------------------------------------|
| `source_currency`  | CHAR(3)           | 必须符合 ISO 4217 标准（如 USD/CNY）                                      |
| `target_currency`  | CHAR(3)           | 同上                                                                     |
| `rate`             | DECIMAL(20,12)    | 汇率因子精度 = 源货币单位对应目标货币的数量（1 USD = 6.456789012345 CNY） |
| `effective_start`  | DATETIME          | 精确到秒级，用于时段有效性判断                                            |
| `effective_end`    | DATETIME          | 设置为 NULL 时表示当前生效的汇率                                          |

---

### 二、核心业务规则
#### 1. 汇率更新逻辑
```sql
-- 当新汇率生效时，关闭旧汇率的有效期
UPDATE exchange_rate 
SET effective_end = NOW()
WHERE source_currency = 'USD' 
  AND target_currency = 'CNY' 
  AND effective_end IS NULL;

-- 插入新汇率记录
INSERT INTO exchange_rate 
(source_currency, target_currency, rate, effective_start, updated_by)
VALUES ('USD', 'CNY', 6.456789012345, NOW(), 'system');
```

#### 2. 有效性校验约束
- **时间重叠校验**：同一货币对在相同时间段内不允许存在重叠的生效记录
- **反向汇率校验**：若存储反向汇率（如 CNY→USD），需保证 `1/(USD→CNY_rate) ≈ CNY→USD_rate`（允许微小误差）

---

### 三、汇率计算逻辑
#### 1. 查询当前有效汇率
```sql
SELECT rate 
FROM exchange_rate
WHERE source_currency = 'USD' 
  AND target_currency = 'CNY'
  AND effective_start <= NOW()
  AND (effective_end IS NULL OR effective_end > NOW())
ORDER BY effective_start DESC
LIMIT 1;
```

#### 2. 金额转换计算（Java示例）
```java
BigDecimal sourceAmount = new BigDecimal("1000.00"); // 源金额（USD）
BigDecimal exchangeRate = new BigDecimal("6.456789012345"); // USD→CNY汇率

// 转换为目标货币（保留2位小数，银行家舍入法）
BigDecimal targetAmount = sourceAmount.multiply(exchangeRate)
                                     .setScale(2, RoundingMode.HALF_EVEN);
```

---

### 四、增强设计建议
#### 1. 历史汇率归档
```sql
-- 创建历史表（结构与主表相同）
CREATE TABLE exchange_rate_history LIKE exchange_rate;

-- 定期迁移过期数据
INSERT INTO exchange_rate_history 
SELECT * FROM exchange_rate 
WHERE effective_end < NOW() - INTERVAL 90 DAY;

DELETE FROM exchange_rate 
WHERE effective_end < NOW() - INTERVAL 90 DAY;
```

#### 2. 汇率审计追踪
```sql
-- 添加变更日志表
CREATE TABLE exchange_rate_audit (
    log_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    rate_id BIGINT UNSIGNED NOT NULL,
    old_rate DECIMAL(20,12),
    new_rate DECIMAL(20,12),
    changed_by VARCHAR(64),
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

### 五、数据维护策略
#### 1. 外部数据集成
```bash
# 示例：每日从央行API同步汇率
0 3 * * * curl -X GET "https://api.centralbank.com/rates?base=USD" | jq '.rates.CNY' > update_rate.sql
```

#### 2. 数据校验规则
```sql
-- 每日检查汇率波动幅度（防止异常值）
SELECT 
    source_currency, 
    target_currency,
    (MAX(rate) - MIN(rate)) / MIN(rate) AS volatility
FROM exchange_rate
WHERE effective_start >= CURDATE() - INTERVAL 1 DAY
GROUP BY source_currency, target_currency
HAVING volatility > 0.1; -- 超过10%波动触发告警
```

---

### 六、注意事项
1. **精度选择**：DECIMAL(20,12) 可覆盖绝大多数货币需求（如日元→科威特第纳尔换算）
2. **时区处理**：所有时间字段需统一使用 UTC 时区
3. **反向汇率**：建议存储正向汇率（如 USD→CNY），反向汇率通过计算获得（`1/rate`）
4. **法律合规**：保留至少 5 年的历史汇率记录以满足审计要求

> 依据知识库资料：参考《跨境金融便民手册》对汇率折算的规范性要求，结合海通证券《宏观因子与债券风险溢价》研究中汇率波动率计算方法。