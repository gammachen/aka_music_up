# 音乐站点技术方案设计文档 V2

## 1. 技术架构概述

### 1.1 整体架构
- 采用前后端分离架构
- 前端：Vue3 + TypeScript + Ant Design Vue
- 后端：Python + Flask
- 数据库：SQLite2
- 文件存储：本地文件系统
- 缓存：Redis（可选，用于提升性能）

### 1.2 技术选型理由
- **前端**：
  - Vue3：
    - 更好的TypeScript支持
    - Composition API提供更灵活的代码组织方式
    - 更小的打包体积和更好的性能
  - TypeScript：提供类型安全，提高代码可维护性
  - Ant Design Vue：成熟的UI组件库，与Vue3完美适配
- **后端**：
  - Python：开发效率高，生态系统丰富
  - Flask：
    - 轻量级框架，易于学习和使用
    - 灵活性强，可根据需求选择合适的扩展
    - 适合中小型应用开发
- **数据库**：
  - SQLite2：
    - 零配置，易于部署和维护
    - 单文件数据库，方便备份和迁移
    - 支持并发访问和事务处理
- **文件存储**：
  - 本地文件系统：
    - 简化部署和维护
    - 降低运营成本
    - 便于数据备份和迁移

## 2. 系统模块设计

### 2.1 用户认证模块
- **认证机制**
  - Session-based认证：
    - 使用Flask-Login管理用户会话
    - Session有效期：2小时
    - 支持"记住我"功能（延长至30天）
  - 密码安全：
    - 使用Werkzeug.security进行密码加密
    - 采用PBKDF2算法，迭代次数10000
  - 验证码服务：
    - 图形验证码：使用Pillow生成
    - 短信验证码：第三方服务集成
    - 邮箱验证码：Flask-Mail发送
    - 验证码有效期：5分钟

### 2.2 用户系统
- **用户信息管理**
  - 基础信息：
    - 用户名（必填，唯一）
    - 密码（必填，加密存储）
    - 手机号（选填，唯一）
    - 邮箱（选填，唯一）
    - 头像（选填，本地存储）
  - 扩展信息：
    - 个人简介
    - 注册时间
    - 最后登录时间
    - IP地址记录

- **用户等级体系**
  - 经验值规则：
    - 发帖：+10分
    - 评论：+5分
    - 被点赞：+2分
    - 被收藏：+5分
  - 等级划分：
    - Lv1：0-100分
    - Lv2：101-500分
    - Lv3：501-1000分
    - Lv4：1001-2000分
    - Lv5：2001分以上

- **VIP特权系统**
  - 特权内容：
    - 免费查看隐藏内容
    - 发帖无需等待
    - 专属头像框
    - 评论免扣金币
  - 开通方式：
    - 1000金币/年
    - 2000金币/2年
    - 2800金币/3年
  - 自动续费机制
  - 到期提醒功能

### 2.3 文件管理模块
- **本地存储配置**
  - 基础路径：`/static/uploads/`
  - 子目录划分：
    - 头像：`/avatars/`
    - 主题图片：`/topics/`
    - 临时文件：`/temp/`
  - 访问URL规则：`/uploads/{type}/{filename}`

- **文件上传限制**
  - 图片格式：jpg、png、gif
  - 单文件大小：≤5MB
  - 头像尺寸：200x200px
  - 主题图片：最大1920x1080px

- **文件处理**
  - 图片压缩：使用Pillow
  - 水印添加：可选
  - 定时清理：删除未使用文件

## 3. 数据库设计

### 3.1 核心表结构
```sql
-- 用户表
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(20) NOT NULL UNIQUE,
    password_hash VARCHAR(128) NOT NULL,
    phone VARCHAR(20) UNIQUE,
    email VARCHAR(100) UNIQUE,
    avatar_path VARCHAR(255),
    bio TEXT,
    coins INTEGER NOT NULL DEFAULT 0,
    score INTEGER NOT NULL DEFAULT 0,
    vip_expire_time DATETIME,
    is_admin BOOLEAN NOT NULL DEFAULT 0,
    last_login_time DATETIME,
    last_login_ip VARCHAR(45),
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 用户登录记录表
CREATE TABLE user_login_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    login_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    login_ip VARCHAR(45) NOT NULL,
    login_type VARCHAR(20) NOT NULL, -- password, sms, email
    device_info TEXT,
    status VARCHAR(10) NOT NULL, -- success, failed
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 用户验证码表
CREATE TABLE verification_codes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    type VARCHAR(10) NOT NULL, -- sms, email
    target VARCHAR(100) NOT NULL, -- phone number or email
    code VARCHAR(10) NOT NULL,
    expire_time DATETIME NOT NULL,
    is_used BOOLEAN NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 用户金币变动记录表
CREATE TABLE coin_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    amount INTEGER NOT NULL, -- 正数表示收入，负数表示支出
    type VARCHAR(20) NOT NULL, -- recharge, reward, view_topic, etc
    related_id INTEGER, -- 关联的主题或评论ID
    description TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 用户经验值变动记录表
CREATE TABLE score_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    score_change INTEGER NOT NULL,
    type VARCHAR(20) NOT NULL, -- post_topic, comment, receive_like, etc
    related_id INTEGER,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### 3.2 索引设计
```sql
-- 用户表索引
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_phone ON users(phone);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_vip ON users(vip_expire_time);

-- 登录记录表索引
CREATE INDEX idx_login_user ON user_login_logs(user_id);
CREATE INDEX idx_login_time ON user_login_logs(login_time);

-- 验证码表索引
CREATE INDEX idx_verification_target ON verification_codes(target);
CREATE INDEX idx_verification_expire ON verification_codes(expire_time);

-- 金币交易表索引
CREATE INDEX idx_transactions_user ON coin_transactions(user_id);
CREATE INDEX idx_transactions_time ON coin_transactions(created_at);

-- 经验值记录表索引
CREATE INDEX idx_score_user ON score_logs(user_id);
CREATE INDEX idx_score_time ON score_logs(created_at);
```

### 3.3 数据库优化策略
- **查询优化**
  - 合理使用索引
  - 避免SELECT *
  - 使用适当的字段类型
  - 控制索引数量

- **并发处理**
  - 使用事务确保数据一致性
  - 适当的锁粒度
  - 死锁预防

- **数据维护**
  - 定期VACUUM
  - 日志清理策略
  - 备份计划

## 4. 安全设计

### 4.1 接口安全
- CSRF防护
- XSS防护
- SQL注入防护
- 请求频率限制

### 4.2 数据安全
- 敏感数据加密
- 定期备份
- 访问权限控制

### 4.3 运行监控
- 错误日志
- 性能监控
- 用户行为分析