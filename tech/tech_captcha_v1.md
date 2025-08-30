# 验证码模块技术方案设计文档

## 1. 系统概述

### 1.1 模块定位
- 提供统一的验证码服务
- 支持多种验证码类型：图形验证码、短信验证码、邮箱验证码
- 确保系统安全性，防止恶意攻击

### 1.2 功能特性
- 验证码生成和存储
- 验证码刷新和失效管理
- 验证码校验
- 防重放攻击
- 访问频率控制

## 2. 详细设计

### 2.1 数据模型设计

```sql
-- 验证码表（verification_codes）
CREATE TABLE verification_codes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code_id VARCHAR(32) NOT NULL,           -- 验证码唯一标识
    type VARCHAR(10) NOT NULL,              -- 验证码类型：image/sms/email
    code VARCHAR(10) NOT NULL,              -- 验证码内容
    target VARCHAR(100) NOT NULL,           -- 目标（手机号/邮箱/会话ID）
    user_id INTEGER,                        -- 关联用户ID（可选）
    scene VARCHAR(20) NOT NULL,             -- 使用场景（login/register/reset_pwd）
    expire_time DATETIME NOT NULL,          -- 过期时间
    is_used BOOLEAN NOT NULL DEFAULT 0,     -- 是否已使用
    is_valid BOOLEAN NOT NULL DEFAULT 1,    -- 是否有效（用于软删除）
    attempt_count INTEGER DEFAULT 0,        -- 尝试次数
    client_ip VARCHAR(45),                  -- 客户端IP
    user_agent TEXT,                        -- 用户代理信息
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 索引
CREATE INDEX idx_verification_code_id ON verification_codes(code_id);
CREATE INDEX idx_verification_target ON verification_codes(target, type);
CREATE INDEX idx_verification_expire ON verification_codes(expire_time);
```

### 2.2 缓存设计

使用Redis进行验证码缓存，提高访问性能：

```plaintext
# 验证码缓存键格式
verification:code:{code_id} -> {code_info}

# 验证码计数器键格式（用于限流）
verification:counter:{ip}:{type} -> count
verification:counter:{phone}:{type} -> count
```

### 2.3 核心流程

#### 2.3.1 验证码生成流程
1. 接收验证码请求，进行权限和频率检查
2. 生成验证码唯一标识code_id（UUID）
3. 根据类型生成验证码：
   - 图形验证码：使用Pillow生成
   - 短信验证码：6位数字
   - 邮箱验证码：6位数字字母组合
4. 将验证码信息同时写入数据库和Redis缓存
5. 对于之前的未使用验证码：
   - 将is_valid设置为false（软删除）
   - 从Redis缓存中删除

#### 2.3.2 验证码刷新流程
1. 检查上一次验证码生成时间，防止频繁刷新
2. 将旧验证码标记为无效（is_valid=false）
3. 生成新验证码
4. 更新数据库和缓存

#### 2.3.3 验证码校验流程
1. 根据code_id查询验证码信息
2. 校验有效性：
   - 是否存在且有效（is_valid=true）
   - 是否已过期
   - 是否已使用
   - 尝试次数是否超限
3. 校验成功：
   - 标记为已使用（is_used=true）
   - 从缓存中删除
4. 校验失败：
   - 增加尝试次数
   - 超过最大尝试次数则失效

### 2.4 安全措施

#### 2.4.1 频率限制
- 同一IP的验证码生成请求限制：
  - 图形验证码：60次/小时
  - 短信验证码：10次/小时
  - 邮箱验证码：10次/小时
- 同一手机号/邮箱的验证码发送限制：
  - 短信：5次/小时，20次/天
  - 邮箱：5次/小时，20次/天

#### 2.4.2 验证码有效期
- 图形验证码：5分钟
- 短信验证码：5分钟
- 邮箱验证码：15分钟

#### 2.4.3 防重放攻击
- 验证码一次性使用
- 使用code_id作为唯一标识
- 记录验证码使用历史

### 2.5 接口定义

#### 2.5.1 生成验证码
```python
POST /api/v1/verification/generate
Request:
{
    "type": "image|sms|email",
    "target": "手机号|邮箱|会话ID",
    "scene": "login|register|reset_pwd"
}

Response:
{
    "code": 0,
    "message": "success",
    "data": {
        "code_id": "uuid",
        "expire_time": "2024-02-15 12:00:00"
    }
}
```

#### 2.5.2 验证码校验
```python
POST /api/v1/verification/verify
Request:
{
    "code_id": "uuid",
    "code": "验证码内容"
}

Response:
{
    "code": 0,
    "message": "success",
    "data": {
        "is_valid": true
    }
}
```

## 3. 异常处理

### 3.1 错误码定义
```python
ERROR_CODE = {
    "4001": "验证码不存在",
    "4002": "验证码已过期",
    "4003": "验证码已使用",
    "4004": "验证码错误",
    "4005": "验证码尝试次数超限",
    "4006": "请求频率超限"
}
```

### 3.2 异常恢复机制
- 缓存异常：降级使用数据库
- 验证码生成失败：重试机制
- 短信发送失败：备用通道

## 4. 监控告警

### 4.1 监控指标
- 验证码生成成功率
- 验证码验证成功率
- 验证码发送延迟
- 异常验证码请求量

### 4.2 告警规则
- 验证码生成失败率 > 5%
- 验证码验证失败率 > 10%
- 短信发送失败率 > 5%
- 同一IP的高频请求