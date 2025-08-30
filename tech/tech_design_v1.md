# 音乐站点技术方案设计文档

## 1. 技术架构概述

### 1.1 整体架构
- 采用前后端分离架构
- 前端：React + TypeScript + Ant Design
- 后端：Node.js + Express + TypeScript
- 数据库：MySQL + Redis
- 文件存储：阿里云OSS
- 消息队列：RabbitMQ（用于异步任务处理）

### 1.2 技术选型理由
- **前端**：
  - React：组件化开发，生态完善，社区活跃
  - TypeScript：提供类型安全，提高代码可维护性
  - Ant Design：成熟的UI组件库，提供丰富的组件
- **后端**：
  - Node.js：适合I/O密集型应用，前后端统一语言
  - Express：轻量级框架，易于扩展
  - TypeScript：类型安全，提高代码质量
- **数据库**：
  - MySQL：关系型数据库，适合存储结构化数据
  - Redis：缓存层，提高访问速度，支持计数器功能
- **文件存储**：
  - 阿里云OSS：可靠的云存储服务，适合存储用户上传的图片等文件
- **消息队列**：
  - RabbitMQ：处理异步任务，如发送通知、更新统计数据等

## 2. 系统模块设计

### 2.1 用户认证模块
- **JWT认证机制**
  - Access Token：有效期2小时
  - Refresh Token：有效期7天
  - Token存储：Redis
- **密码加密**
  - 使用bcrypt进行密码加密
  - Salt轮数：12轮
- **验证码服务**
  - 短信验证码：阿里云SMS
  - 邮箱验证码：NodeMailer
  - 验证码有效期：5分钟

### 2.2 用户系统
- **用户信息缓存**
  - 使用Redis缓存用户基本信息
  - 缓存时间：1小时
  - 采用缓存穿透防护策略
- **用户等级体系**
  - 经验值计算规则
  - VIP特权管理
  - 自动降级任务

### 2.3 内容管理系统
- **富文本编辑器**
  - 使用TinyMCE
  - 支持图片上传（OSS）
  - XSS防护
- **内容缓存策略**
  - 热门主题缓存
  - 首页数据缓存
  - 评论列表缓存

### 2.4 支付系统
- **支付集成**
  - 支付宝支付
  - 微信支付
- **金币系统**
  - 金币变动记录
  - 交易流水
  - 账户余额

## 3. 数据库设计

### 3.1 表结构设计
```sql
-- 用户表
CREATE TABLE users (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(20) UNIQUE NOT NULL,
    password VARCHAR(60) NOT NULL,
    phone VARCHAR(20) UNIQUE,
    email VARCHAR(100) UNIQUE,
    avatar_url VARCHAR(255),
    coins INT DEFAULT 0,
    score INT DEFAULT 0,
    vip_expire_time DATETIME,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- 主题表
CREATE TABLE topics (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    hidden_content TEXT,
    author_id BIGINT NOT NULL,
    category_id INT NOT NULL,
    likes_count INT DEFAULT 0,
    collections_count INT DEFAULT 0,
    reward_amount INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (author_id) REFERENCES users(id),
    FOREIGN KEY (category_id) REFERENCES categories(id)
);