# 用户经验值系统技术方案

## 1. 系统概述

### 1.1 设计目标
- 建立完整的用户经验值体系
- 通过经验值激励用户活跃度
- 实现用户等级成长体系
- 提供清晰的经验值变更记录

### 1.2 核心功能
- 经验值获取与消耗
- 用户等级划分
- 经验值流水记录
- 等级特权管理

## 2. 经验值规则设计

### 2.1 经验值获取规则

#### 2.1.1 基础行为奖励
- 发布主题：+10分
- 发表评论：+5分
- 主题被点赞：+2分
- 主题被收藏：+5分
- 每日登录：+2分
- 完善个人资料：+20分（一次性）

#### 2.1.2 特殊奖励
- 主题被设为精华：+50分
- 评论被设为最佳回复：+20分
- 主题被管理员推荐：+30分

### 2.2 经验值消耗规则
- 主题被删除：-10分
- 评论被删除：-5分
- 违规行为处罚：-50分

### 2.3 等级划分
```
Lv1：0-100分（新手）
Lv2：101-500分（进阶）
Lv3：501-1000分（老手）
Lv4：1001-2000分（专家）
Lv5：2001分以上（大师）
```

## 3. 数据库设计

### 3.1 经验值记录表（score_logs）详解

```sql
CREATE TABLE score_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,          -- 用户ID
    score_change INTEGER NOT NULL,      -- 经验值变更量（正数为增加，负数为减少）
    type VARCHAR(20) NOT NULL,          -- 变更类型
    related_id INTEGER,                 -- 关联ID
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,  -- 创建时间
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

字段说明：
- **score_change**：
  - 表示单次变更的经验值数量
  - 正数表示获得经验值
  - 负数表示扣除经验值
  - 不是累计值，而是每次变更的增量或减量

- **type**：变更类型，可选值：
  ```
  post_topic        -- 发布主题
  comment           -- 发表评论
  receive_like      -- 收到点赞
  receive_collect   -- 被收藏
  daily_login       -- 每日登录
  profile_complete  -- 完善资料
  topic_featured    -- 主题精华
  best_comment      -- 最佳回复
  topic_recommend   -- 主题推荐
  topic_delete      -- 主题删除
  comment_delete    -- 评论删除
  violation        -- 违规处罚
  ```

- **related_id**：关联ID说明
  - 关联到触发经验值变动的具体业务对象
  - 不同type对应不同关联对象：
    ```
    post_topic: 主题ID
    comment: 评论ID
    receive_like: 点赞记录ID
    receive_collect: 收藏记录ID
    topic_featured: 主题ID
    best_comment: 评论ID
    topic_recommend: 主题ID
    topic_delete: 主题ID
    comment_delete: 评论ID
    violation: 违规记录ID
    daily_login: null
    profile_complete: null
    ```

### 3.2 索引设计
```sql
-- 用户经验值记录索引
CREATE INDEX idx_score_user ON score_logs(user_id);
CREATE INDEX idx_score_time ON score_logs(created_at);
CREATE INDEX idx_score_type ON score_logs(type);
```

## 4. 接口设计

### 4.1 经验值变更接口
```python
def change_user_score(user_id: int, score_change: int, type: str, related_id: int = None) -> bool:
    """更新用户经验值
    
    Args:
        user_id: 用户ID
        score_change: 变更值（正数为增加，负数为减少）
        type: 变更类型
        related_id: 关联ID（可选）
    
    Returns:
        bool: 更新是否成功
    """
    try:
        with db.transaction():
            # 1. 记录经验值变更日志
            db.execute(
                "INSERT INTO score_logs (user_id, score_change, type, related_id) "
                "VALUES (?, ?, ?, ?)",
                (user_id, score_change, type, related_id)
            )
            
            # 2. 更新用户总经验值
            db.execute(
                "UPDATE users SET score = score + ? WHERE id = ?",
                (score_change, user_id)
            )
            
            # 3. 检查是否需要更新用户等级
            update_user_level(user_id)
            
        return True
    except Exception as e:
        log.error(f"更新用户经验值失败: {e}")
        return False
```

### 4.2 查询经验值记录接口
```python
def get_user_score_logs(user_id: int, page: int = 1, page_size: int = 20) -> dict:
    """获取用户经验值变更记录
    
    Args:
        user_id: 用户ID
        page: 页码
        page_size: 每页数量
    
    Returns:
        dict: 包含分页信息和记录列表
    """
    offset = (page - 1) * page_size
    
    # 获取总记录数
    total = db.get_scalar(
        "SELECT COUNT(*) FROM score_logs WHERE user_id = ?",
        (user_id,)
    )
    
    # 获取记录列表
    logs = db.query(
        "SELECT * FROM score_logs "
        "WHERE user_id = ? "
        "ORDER BY created_at DESC "
        "LIMIT ? OFFSET ?",
        (user_id, page_size, offset)
    )
    
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "logs": logs
    }
```

## 5. 业务流程示例

### 5.1 发布主题流程
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
                score_change=10,  # 发帖奖励10分
                type="post_topic",
                related_id=topic_id
            )
            
            return topic_id
    except Exception as e:
        log.error(f"发布主题失败: {e}")
        return None
```

### 5.2 主题被点赞流程
```python
def like_topic(user_id: int, topic_id: int) -> bool:
    """点赞主题"""
    try:
        with db.transaction():
            # 1. 创建点赞记录
            like_id = db.execute(
                "INSERT INTO user_likes (user_id, topic_id) VALUES (?, ?)",
                (user_id, topic_id)
            )
            
            # 2. 获取主题作者ID
            topic_author_id = db.get_scalar(
                "SELECT user_id FROM topics WHERE id = ?",
                (topic_id,)
            )
            
            # 3. 给主题作者增加经验值
            change_user_score(
                user_id=topic_author_id,
                score_change=2,  # 被点赞奖励2分
                type="receive_like",
                related_id=like_id
            )
            
            return True
    except Exception as e:
        log.error(f"点赞主题失败: {e}")
        return False
```

## 6. 注意事项

### 6.1 并发处理
- 使用事务确保经验值变更的原子性
- 避免用户经验值计算错误
- 防止重复记录经验值变更

### 6.2 性能优化
- 合理使用索引提升查询效率
- 定期清理历史记录
- 考虑使用缓存优化热点数据

### 6.3 安全控制
- 验证经验值变更的合法性
- 防止经验值刷分行为
- 记录关键操作日志