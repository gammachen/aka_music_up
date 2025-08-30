# 音乐站点数据库设计文档 V2

## 1. 数据库概述

### 1.1 数据库选型
- 数据库：SQLite2
- 特点：
  - 零配置，易于部署和维护
  - 单文件数据库，方便备份和迁移
  - 支持并发访问和事务处理

## 2. 数据库表设计

### 2.1 用户相关表

#### 2.1.1 用户表（users）
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    password_hash VARCHAR(128) NOT NULL,
    phone VARCHAR(20) UNIQUE,
    email VARCHAR(100) UNIQUE,
    avatar_path VARCHAR(200),
    bio TEXT,
    coins INTEGER NOT NULL DEFAULT 0,
    score INTEGER NOT NULL DEFAULT 0,
    vip_expire_time DATETIME,
    is_admin BOOLEAN NOT NULL DEFAULT 0,
    register_source VARCHAR(20) NOT NULL,
    last_login_time DATETIME,
    last_login_ip VARCHAR(50),
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_register_source CHECK (register_source IN ('password', 'wechat', 'qq', 'weibo'))
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_phone ON users(phone);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_vip ON users(vip_expire_time);
```

#### 2.1.2 第三方认证表（third_party_auths）
```sql
CREATE TABLE third_party_auths (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    platform VARCHAR(20) NOT NULL,
    open_id VARCHAR(100) NOT NULL,
    union_id VARCHAR(100),
    nickname VARCHAR(50),
    avatar_url VARCHAR(200),
    access_token VARCHAR(200) NOT NULL,
    refresh_token VARCHAR(200),
    token_expires_at DATETIME NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT unique_platform_openid UNIQUE (platform, open_id)
);

CREATE INDEX idx_third_party_user ON third_party_auths(user_id);
```

#### 2.1.3 用户登录日志表（user_login_logs）
```sql
CREATE TABLE user_login_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    login_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    login_ip VARCHAR(50) NOT NULL,
    login_type VARCHAR(20) NOT NULL,
    device_info TEXT,
    status VARCHAR(20) NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT check_login_type CHECK (login_type IN ('password', 'wechat', 'qq', 'weibo', 'token'))
);

CREATE INDEX idx_login_user ON user_login_logs(user_id);
CREATE INDEX idx_login_time ON user_login_logs(login_time);
```

### 2.2 主题相关表

#### 2.2.1 主题表（topics）
```sql
CREATE TABLE topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    title VARCHAR(200) NOT NULL,
    content TEXT NOT NULL,
    hidden_content TEXT,
    hidden_cost INTEGER NOT NULL DEFAULT 0,
    view_count INTEGER NOT NULL DEFAULT 0,
    like_count INTEGER NOT NULL DEFAULT 0,
    favorite_count INTEGER NOT NULL DEFAULT 0,
    reply_count INTEGER NOT NULL DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'normal',
    is_top BOOLEAN NOT NULL DEFAULT 0,
    is_essence BOOLEAN NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT check_status CHECK (status IN ('normal', 'hidden', 'deleted'))
);

CREATE INDEX idx_topics_user ON topics(user_id);
CREATE INDEX idx_topics_status ON topics(status);
CREATE INDEX idx_topics_created ON topics(created_at);
```

#### 2.2.2 主题点赞表（topic_likes）
```sql
CREATE TABLE topic_likes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    topic_id INTEGER NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE,
    CONSTRAINT unique_user_topic_like UNIQUE (user_id, topic_id)
);

CREATE INDEX idx_likes_user ON topic_likes(user_id);
CREATE INDEX idx_likes_topic ON topic_likes(topic_id);
```

#### 2.2.3 主题收藏表（topic_favorites）
```sql
CREATE TABLE topic_favorites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    topic_id INTEGER NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE,
    CONSTRAINT unique_user_topic_favorite UNIQUE (user_id, topic_id)
);

CREATE INDEX idx_favorites_user ON topic_favorites(user_id);
CREATE INDEX idx_favorites_topic ON topic_favorites(topic_id);
```

### 2.3 通知相关表

#### 2.3.1 通知表（notifications）
```sql
CREATE TABLE notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    type VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    related_id INTEGER,
    is_read BOOLEAN NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT check_type CHECK (type IN ('like', 'favorite', 'reply', 'system'))
);

CREATE INDEX idx_notifications_user ON notifications(user_id);
CREATE INDEX idx_notifications_read ON notifications(is_read);
```

### 2.4 验证码相关表

#### 2.4.1 验证码表（verification_codes）
```sql
CREATE TABLE verification_codes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    type VARCHAR(20) NOT NULL,
    target VARCHAR(100) NOT NULL,
    code VARCHAR(10) NOT NULL,
    expire_time DATETIME NOT NULL,
    is_used BOOLEAN NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT check_type CHECK (type IN ('register', 'login', 'reset_password'))
);

CREATE INDEX idx_verification_target ON verification_codes(target);
CREATE INDEX idx_verification_expire ON verification_codes(expire_time);
```

### 2.5 金币相关表

#### 2.5.1 金币交易表（coin_transactions）
```sql
CREATE TABLE coin_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    amount INTEGER NOT NULL,
    type VARCHAR(20) NOT NULL,
    related_id INTEGER,
    description TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT check_type CHECK (type IN ('recharge', 'consume', 'reward', 'refund'))
);

CREATE INDEX idx_transactions_user ON coin_transactions(user_id);
CREATE INDEX idx_transactions_time ON coin_transactions(created_at);
```

### 2.6 经验值相关表

#### 2.6.1 经验值记录表（score_logs）
```sql
CREATE TABLE score_type_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type VARCHAR(20) NOT NULL UNIQUE,
    name VARCHAR(50) NOT NULL,
    score_change INTEGER NOT NULL,
    description TEXT,
    is_one_time BOOLEAN NOT NULL DEFAULT 0,
    daily_limit INTEGER,
    status INTEGER NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX idx_rule_type ON score_type_rules(type);
CREATE INDEX idx_rule_status ON score_type_rules(status);
```

CREATE TABLE score_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    score_change INTEGER NOT NULL,
    type VARCHAR(20) NOT NULL,
    related_id INTEGER,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (type) REFERENCES score_type_rules(type),
    CONSTRAINT check_type CHECK (type IN (SELECT type FROM score_type_rules))
);

CREATE INDEX idx_score_user ON score_logs(user_id);
CREATE INDEX idx_score_time ON score_logs(created_at);
```

## 3. 数据库优化策略

### 3.1 索引优化
- 为常用查询字段创建索引
- 避免过多索引，影响写入性能
- 定期维护索引统计信息

### 3.2 查询优化
- 使用预编译语句
- 避免SELECT *
- 合理使用JOIN和子查询
- 适当使用索引提示

### 3.3 事务处理
- 合理设置事务隔离级别
- 避免长事务
- 使用事务保证数据一致性

### 3.4 数据维护
- 定期VACUUM优化存储空间
- 定期清理过期数据
- 定期备份数据库文件

### 3.5 并发控制
- 使用适当的锁粒度
- 避免死锁
- 合理设置连接池大小

## 4. 数据库监控

### 4.1 性能监控
- 慢查询日志
- 连接数监控
- 缓存命中率

### 4.2 空间监控
- 数据文件大小
- 索引空间占用
- 临时文件使用情况

### 4.3 错误监控
- 错误日志分析
- 死锁检测
- 异常连接处理