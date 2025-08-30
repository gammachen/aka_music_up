# 金融领域数值处理规范

## 1. 浮点数精度陷阱与解决方案
### 1.1 IEEE 754标准精度问题
```sql
-- 浮点运算误差示例
/* MySQL浮点运算误差示例 */
SELECT CAST(0.1 AS FLOAT) + CAST(0.2 AS FLOAT);  -- 输出0.30000001192092896

-- 账户金额对比测试表
CREATE TABLE account_compare (
  id INT PRIMARY KEY,
  amount_float FLOAT(10,2),
  amount_decimal DECIMAL(10,2)
);

INSERT INTO account_compare VALUES 
(1, 3.22, 3.22),
(2, 2.08, 2.08),
(3, 0.1, 0.1);

/* 数值存储误差示例 */
SELECT 
  id,
  amount_float,
  amount_decimal 
FROM account_compare; 
-- 输出结果可能显示3.22存储为3.2199999...

/* 比较运算符失效示例 */
SELECT * FROM account_compare WHERE amount_float = 3.22;  -- 无结果返回
SELECT * FROM account_compare WHERE amount_decimal = 3.22;  -- 正确返回ID=1的记录

/* 聚合计算偏差示例 */
SELECT 
  SUM(amount_float) AS float_sum,
  SUM(amount_decimal) AS decimal_sum 
FROM account_compare;
-- 输出可能显示float_sum=5.3999999... decimal_sum=5.40
```

### 1.2 四舍五入问题
**银行家舍入法实现逻辑**：
1. 当精确值为中间值时
2. 优先舍入到最近的偶数位
3. 减少累计误差

## 2. NUMERIC/DECIMAL类型规范
### 2.1 账户金额字段设计
```sql
CREATE TABLE account (
  balance DECIMAL(19,4) CHECK(balance >= 0)
);
```

### 2.2 精度选择原则
| 业务场景       | 推荐精度 |
|----------------|----------|
| 法币交易       | DECIMAL(19,4) |
| 加密货币交易   | DECIMAL(38,18) |

## 3. 大额资金处理策略

### 3.3 MySQL资金计算验证案例
```sql
-- 资金累计误差测试（重复执行10次）
UPDATE account_compare 
SET amount_float = amount_float + 0.1,
    amount_decimal = amount_decimal + 0.1
WHERE id = 3;

/* 十次累加后查询结果 */
SELECT 
  amount_float,   -- 可能显示1.0999999...
  amount_decimal   -- 准确显示1.00
FROM account_compare 
WHERE id = 3;

-- 创建交易流水表
CREATE TABLE transaction (
  id INT AUTO_INCREMENT PRIMARY KEY,
  amount FLOAT(12,2),
  balance DECIMAL(16,2)
);

-- 模拟资金流水（每天0.01元利息）
INSERT INTO transaction (amount, balance)
SELECT 0.01, SUM(0.01) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
FROM generate_series(1,365);

/* 年度利息总额查询 */
SELECT 
  SUM(amount) AS float_total,   -- 可能显示3.6499999...
  SUM(balance) AS decimal_total -- 准确显示3.65
FROM transaction;
### 3.1 资金计算校验
```sql
-- 使用OVERFLOW CHECK防止计算溢出
CREATE FUNCTION safe_add(a DECIMAL, b DECIMAL) 
RETURNS DECIMAL
AS $$
BEGIN
  IF (a > 0 AND b > 0 AND a + b < a) THEN
    RAISE EXCEPTION 'Positive overflow';
  END IF;
  RETURN a + b;
END;
$$ LANGUAGE plpgsql;
```

### 3.2 审计日志规范
1. 金额字段必须记录原始值
2. 计算过程记录精度参数
3. 使用XID实现事务追溯