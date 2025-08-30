# 金币系统技术方案设计文档

## 1. 系统概述

### 1.1 设计目标
- 构建完整的虚拟货币体系
- 提供多样化的金币获取途径
- 实现合理的金币消费机制
- 确保系统安全性和可靠性

### 1.2 核心功能
- 金币获取
- 金币消费
- 金币流转
- 防作弊机制

## 2. 金币获取机制

### 2.1 基础获取
- **新用户注册**
  - 奖励金额：2金币
  - 触发时机：完成注册后自动发放
  - 限制条件：每个账号仅首次注册可获得

- **每日签到**
  - 基础奖励：1金币
  - 连续签到额外奖励：
    - 连续7天：额外2金币
    - 连续30天：额外10金币
  - 补签机制：不支持补签

### 2.2 互动获取
- **内容互动**
  - 发表优质主题：最高10金币/次（由管理员判定）
  - 获得主题回复打赏：按打赏者意愿给予
  - 隐藏内容查看费用分成：50%概率获得查看者支付的1金币

- **红包系统**
  - 发红包：
    - 用户可自定义红包总额和个数
    - 支持随机金额和固定金额两种模式
    - 红包有效期：24小时
  - 抢红包：
    - 所有用户均可参与
    - 每个红包每用户仅可抢一次
    - 根据红包类型随机或固定获得金币

## 3. 金币消费机制

### 3.1 内容消费
- **查看隐藏内容**
  - 消费金额：1金币/次
  - 特殊规则：
    - 同一主题仅首次查看收费
    - VIP用户免费查看
    - 自己发布的主题免费查看
    - 50%概率金币转给主题作者

- **打赏功能**
  - 每日免费打赏次数：10次
  - 超出次数后打赏：1金币/次
  - 单次打赏上限：100金币

### 3.2 VIP购买
- 1年VIP：1000金币
- 2年VIP：2000金币
- 3年VIP：2800金币

## 4. 系统安全设计

### 4.1 防作弊机制
- **操作频率限制**
  - 发帖间隔：2分钟
  - 回复间隔：30秒
  - 打赏间隔：10秒

- **内容质量控制**
  - 主题内容最少字数：100字
  - 回复内容最少字数：10字
  - 关键词过滤
  - 图片上传限制

- **账号限制**
  - IP注册限制：每IP每日最多注册3个账号
  - 设备注册限制：每设备每日最多注册2个账号
  - 异常行为监控：
    - 短时间大量获取金币
    - 可疑的打赏行为
    - 频繁的隐藏内容查看

## 5. 数据库设计

### 5.1 金币流水表
```sql
CREATE TABLE coin_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    amount INTEGER NOT NULL,  -- 正数表示收入，负数表示支出
    balance INTEGER NOT NULL, -- 交易后余额
    type VARCHAR(20) NOT NULL,  -- register/signin/view_hidden/reward/red_packet
    related_id INTEGER,  -- 关联的主题/红包ID
    description TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### 5.2 签到记录表
```sql
CREATE TABLE sign_in_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    sign_date DATE NOT NULL,
    continuous_days INTEGER NOT NULL DEFAULT 1,
    reward_coins INTEGER NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    UNIQUE (user_id, sign_date)
);
```

### 5.3 红包表
```sql
CREATE TABLE red_packets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    total_amount INTEGER NOT NULL,
    remaining_amount INTEGER NOT NULL,
    packet_count INTEGER NOT NULL,
    remaining_count INTEGER NOT NULL,
    packet_type VARCHAR(10) NOT NULL,  -- random/fixed
    expire_time DATETIME NOT NULL,
    status VARCHAR(10) NOT NULL DEFAULT 'active',  -- active/expired/finished
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### 5.4 红包领取记录表
```sql
CREATE TABLE red_packet_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    red_packet_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    amount INTEGER NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (red_packet_id) REFERENCES red_packets(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### 5.5 打赏记录表
```sql
CREATE TABLE reward_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_user_id INTEGER NOT NULL,
    to_user_id INTEGER NOT NULL,
    topic_id INTEGER,
    comment_id INTEGER,
    amount INTEGER NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (from_user_id) REFERENCES users(id),
    FOREIGN KEY (to_user_id) REFERENCES users(id),
    FOREIGN KEY (topic_id) REFERENCES topics(id),
    FOREIGN KEY (comment_id) REFERENCES topic_comments(id)
);
```

## 6. 接口设计

### 6.1 金币相关接口
```python
# 获取用户金币余额
GET /api/v1/coins/balance

# 获取金币交易记录
GET /api/v1/coins/transactions

# 每日签到
POST /api/v1/coins/sign-in

# 发红包
POST /api/v1/coins/red-packets

# 抢红包
POST /api/v1/coins/red-packets/{id}/grab

# 打赏
POST /api/v1/coins/reward
```

### 6.2 查看隐藏内容接口
```python
# 检查隐藏内容查看权限
GET /api/v1/topics/{id}/hidden-content/check

# 查看隐藏内容
POST /api/v1/topics/{id}/hidden-content/view
```

## 7. 定时任务

### 7.1 系统维护任务
- 每日零点重置打赏次数限制
- 每小时检查并关闭过期红包
- 每日统计异常账号行为
- 每周清理30天前的流水记录（保留汇总数据）

### 7.2 数据统计任务
- 每日统计金币发放总量
- 每日统计金币消费总量
- 每日统计用户活跃度
- 每月生成运营报表