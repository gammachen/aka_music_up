# 用户经验值系统技术方案 V2

## 1. 系统概述

### 1.1 设计目标
- 建立完整的用户经验值体系
- 通过经验值激励用户活跃度
- 实现用户等级成长体系
- 提供清晰的经验值变更记录
- 优化积分查询性能
- 实现积分规则的灵活配置

### 1.2 核心功能
- 经验值获取与消耗
- 用户等级划分
- 经验值流水记录
- 等级特权管理
- 积分规则配置管理

## 2. 数据库设计

### 2.1 用户表（users）
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) NOT NULL,
    current_score INTEGER NOT NULL DEFAULT 0,  -- 当前积分
    level INTEGER NOT NULL DEFAULT 1,         -- 当前等级
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

### 2.2 积分规则表（score_type_rules）
```sql
CREATE TABLE score_type_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type VARCHAR(20) NOT NULL UNIQUE,         -- 变更类型
    name VARCHAR(50) NOT NULL,                -- 规则名称
    score_change INTEGER NOT NULL,            -- 积分变更值
    description TEXT,                         -- 规则描述
    is_one_time BOOLEAN DEFAULT false,        -- 是否一次性
    daily_limit INTEGER,                      -- 每日限制次数
    status INTEGER DEFAULT 1,                 -- 规则状态：1-启用，0-禁用
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

### 2.3 经验值记录表（score_logs）
```sql
CREATE TABLE score_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,                 -- 用户ID
    score_change INTEGER NOT NULL,            -- 经验值变更量
    type VARCHAR(20) NOT NULL,                -- 变更类型
    current_score INTEGER NOT NULL,           -- 变更后的当前积分
    related_id INTEGER,                       -- 关联ID
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (type) REFERENCES score_type_rules(type)
);
```

### 2.4 索引设计
```sql
-- 用户表索引
CREATE INDEX idx_user_score ON users(current_score);
CREATE INDEX idx_user_level ON users(level);

-- 积分规则表索引
CREATE UNIQUE INDEX idx_rule_type ON score_type_rules(type);
CREATE INDEX idx_rule_status ON score_type_rules(status);

-- 经验值记录表索引
CREATE INDEX idx_score_user ON score_logs(user_id);
CREATE INDEX idx_score_type ON score_logs(type);
CREATE INDEX idx_score_time ON score_logs(created_at);
```

## 3. 接口设计

### 3.1 经验值变更接口
```python
def change_user_score(user_id: int, type: str, related_id: int = None) -> bool:
    """更新用户经验值
    
    Args:
        user_id: 用户ID
        type: 变更类型
        related_id: 关联ID（可选）
    
    Returns:
        bool: 更新是否成功
    """
    try:
        with db.transaction():
            # 1. 获取积分规则
            rule = db.get_one(
                "SELECT * FROM score_type_rules WHERE type = ? AND status = 1",
                (type,)
            )
            if not rule:
                raise ValueError(f"Invalid score type: {type}")
            
            # 2. 检查规则限制
            if rule["is_one_time"]:
                # 检查是否已经获得过一次性奖励
                exists = db.get_scalar(
                    "SELECT 1 FROM score_logs WHERE user_id = ? AND type = ?",
                    (user_id, type)
                )
                if exists:
                    return False
            
            if rule["daily_limit"]:
                # 检查今日是否达到限制
                today_count = db.get_scalar(
                    "SELECT COUNT(*) FROM score_logs "
                    "WHERE user_id = ? AND type = ? AND DATE(created_at) = DATE('now')",
                    (user_id, type)
                )
                if today_count >= rule["daily_limit"]:
                    return False
            
            # 3. 更新用户总经验值
            db.execute(
                "UPDATE users SET current_score = current_score + ? "
                "WHERE id = ?",
                (rule["score_change"], user_id)
            )
            
            # 4. 获取更新后的当前积分
            current_score = db.get_scalar(
                "SELECT current_score FROM users WHERE id = ?",
                (user_id,)
            )
            
            # 5. 记录经验值变更日志
            db.execute(
                "INSERT INTO score_logs "
                "(user_id, score_change, type, current_score, related_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, rule["score_change"], type, current_score, related_id)
            )
            
            # 6. 检查是否需要更新用户等级
            update_user_level(user_id)
            
        return True
    except Exception as e:
        log.error(f"更新用户经验值失败: {e}")
        return False
```

### 3.2 查询用户积分接口
```python
def get_user_score(user_id: int) -> dict:
    """获取用户积分信息
    
    Args:
        user_id: 用户ID
    
    Returns:
        dict: 用户积分信息
    """
    try:
        # 直接从users表获取当前积分和等级
        user = db.get_one(
            "SELECT current_score, level FROM users WHERE id = ?",
            (user_id,)
        )
        if not user:
            return None
        
        return {
            "current_score": user["current_score"],
            "level": user["level"]
        }
    except Exception as e:
        log.error(f"获取用户积分信息失败: {e}")
        return None
```

### 3.3 积分规则管理接口
```python
def create_score_rule(type: str, name: str, score_change: int, **kwargs) -> bool:
    """创建积分规则
    
    Args:
        type: 规则类型
        name: 规则名称
        score_change: 积分变更值
        **kwargs: 其他可选参数
    
    Returns:
        bool: 创建是否成功
    """
    try:
        with db.transaction():
            db.execute(
                "INSERT INTO score_type_rules "
                "(type, name, score_change, description, is_one_time, daily_limit) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (type, name, score_change,
                 kwargs.get("description"),
                 kwargs.get("is_one_time", False),
                 kwargs.get("daily_limit"))
            )
        return True
    except Exception as e:
        log.error(f"创建积分规则失败: {e}")
        return False

def update_score_rule(type: str, **kwargs) -> bool:
    """更新积分规则
    
    Args:
        type: 规则类型
        **kwargs: 需要更新的字段
    
    Returns:
        bool: 更新是否成功
    """
    try:
        fields = []
        values = []
        for key, value in kwargs.items():
            if key in ["name", "score_change", "description", "is_one_time", "daily_limit", "status"]:
                fields.append(f"{key} = ?")
                values.append(value)
        
        if not fields:
            return False
        
        values.append(type)
        with db.transaction():
            db.execute(
                f"UPDATE score_type_rules SET {', '.join(fields)} "
                "WHERE type = ?",
                tuple(values)
            )
        return True
    except Exception as e:
        log.error(f"更新积分规则失败: {e}")
        return False
```

## 4. 业务流程示例

### 4.1 初始化积分规则
```python
def init_score_rules():
    """初始化系统默认积分规则"""
    rules = [
        {
            "type": "post_topic",
            "name": "发布主题",
            "score_change": 10,
            "daily_limit": 10
        },
        {
            "type": "comment",
            "name": "发表评论",
            "score_change": 5,
            "daily_limit": 20
        },
        {
            "type": "receive_like",
            "name": "收到点赞",
            "score_change": 2,
            "daily_limit": 100
        },
        {
            "type": "profile_complete",
            "name": "完善资料",
            "score_change": 20,
            "is_one_time": True
        }
    ]
    
    for rule in rules:
        create_score_rule(**rule)
```

### 4.2 发布主题流程
```python
def create_topic(user_id: int, title: str, content: str) -> int:
    """发布主题"""
    try:
        with db.transaction():
            # 1. 创建主题
            topic_id = db.execute(
                "INSERT INTO topics (user_id, title, content) VALUES (?, ?, ?)",
                (user_id, title, content)
            )
            
            # 2. 增加用户经验值
            change_user_score(
                user_id=user_id,
                type="post_topic",
                related_id=topic_id
            )
            
            return topic_id
    except Exception as e:
        log.error(f"发布主题失败: {e}")
        return None
```

## 5. 注意事项

### 5.1 性能优化
- users表增加current_score字段，避免频繁统计
- 合理使用索引提升查询效率
- 积分规则配置化，便于管理和修改

### 5.2 数据一致性
- 使用事务确保积分变更的原子性
- 记录变更后的current_score，便于数据校验
- 定期对账用户积分和变更记录

### 5.3 安全控制
- 验证积分规则的合法性
- 控制积分获取的频率和上限
- 记录关键操作日志