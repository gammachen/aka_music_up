# 用户认证API接口文档 V1

## 1. 第三方登录接口

### 1.1 获取授权URL

```http
GET /api/v1/auth/{platform}/url
```

**参数说明：**
- platform: 平台标识(wechat/qq)

**响应示例：**
```json
{
  "code": 0,
  "data": {
    "auth_url": "https://open.weixin.qq.com/connect/qrconnect?appid=xxx",
    "state": "random_state_xxx"
  }
}
```

### 1.2 授权回调处理

```http
GET /api/v1/auth/{platform}/callback
```

**参数说明：**
- platform: 平台标识
- code: 授权码
- state: 状态码

**响应示例：**
```json
{
  "code": 0,
  "data": {
    "token": "jwt_token_xxx",
    "user": {
      "id": 1,
      "username": "user_xxx",
      "avatar_path": "/uploads/avatars/xxx.jpg"
    }
  }
}
```

## 2. 用户信息接口

### 2.1 获取当前用户信息

```http
GET /api/v1/user/profile
```

**请求头：**
- Authorization: Bearer {token}

**响应示例：**
```json
{
  "code": 0,
  "data": {
    "id": 1,
    "username": "user_xxx",
    "avatar_path": "/uploads/avatars/xxx.jpg",
    "register_source": "wechat",
    "created_at": "2024-02-15T10:00:00Z"
  }
}
```

### 2.2 更新用户信息

```http
PUT /api/v1/user/profile
```

**请求头：**
- Authorization: Bearer {token}

**请求参数：**
```json
{
  "username": "new_username",
  "avatar": "base64_image_data"
}
```

**响应示例：**
```json
{
  "code": 0,
  "data": {
    "id": 1,
    "username": "new_username",
    "avatar_path": "/uploads/avatars/new_xxx.jpg"
  }
}
```

## 3. 错误码说明

| 错误码 | 说明 |
|--------|------|
| 0 | 成功 |
| 1001 | 参数错误 |
| 1002 | 未授权 |
| 2001 | 用户不存在 |
| 2002 | 用户名已存在 |
| 3001 | 第三方授权失败 |
| 3002 | Token过期 |
| 5001 | 系统错误 |

## 4. 安全规范

### 4.1 接口安全
1. 所有接口使用HTTPS
2. 敏感接口需要JWT认证
3. 实现接口访问频率限制

### 4.2 数据安全
1. 敏感数据传输加密
2. 文件上传类型限制
3. 防止SQL注入

### 4.3 异常处理
1. 统一错误响应格式
2. 详细的错误日志记录
3. 关键操作审计日志