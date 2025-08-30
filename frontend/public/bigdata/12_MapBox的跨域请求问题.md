当你在 `superset_config.py` 中配置了 CORS 但仍然遇到 `401 Unauthorized` 错误和跨域问题时，这通常涉及多个层面的配置。以下是综合解决方案：

### 完整解决步骤

#### 1. 增强 CORS 配置 (superset_config.py)
```python
# 核心 CORS 配置
ENABLE_CORS = True
CORS_OPTIONS = {
    "supports_credentials": True,
    "allow_headers": ["*"],
    "resources": ["*"],
    "origins": [
        "http://localhost:8000",  # 开发环境
        "http://your-frontend-domain.com"  # 生产环境
    ]
}

# 关键安全配置 (解决 401 问题)
AUTH_TYPE = AUTH_DB
AUTH_USER_REGISTRATION = True
AUTH_USER_REGISTRATION_ROLE = "Alpha"
PUBLIC_ROLE_LIKE = "Gamma"
WTF_CSRF_ENABLED = False  # 重要：禁用 CSRF 用于 API 请求

# Mapbox 配置(这个才是关键，只是它在注册时还需要填写信用卡信息，从metasearch获取了一个api key：)
MAPBOX_API_KEY = "pk.your_mapbox_key"  # 从 mapbox.com 获取
```

#### 2. 确保正确挂载配置文件 (Docker 命令)
```bash
docker run -d --name superset \
  -p 8088:8088 \
  -v $(pwd)/superset_config.py:/app/superset/superset_config.py \
  apache/superset
```

#### 3. 解决 401 Unauthorized 问题
```python
# 在 superset_config.py 中添加以下配置
from flask_appbuilder.security.manager import AUTH_DB

# 关闭不必要的认证检查
FEATURE_FLAGS = {
    "ENABLE_REACT_CRUD_VIEWS": True,
    "DASHBOARD_CROSS_FILTERS": True,
    "DISABLE_CRSF_PROTECTION": True  # 关键：禁用 CSRF
}

# 允许匿名访问特定资源 (谨慎使用)
PUBLIC_ROLE_LIKE_GAMMA = True
```

#### 4. 前端请求调整 (确保带凭证)
在 AJAX 请求中添加 `credentials: 'include'`：
```javascript
fetch("http://superset-host/api/v1/chart/", {
  method: "GET",
  credentials: "include"  // 确保发送 cookies
})
```

#### 5. 修改 Docker 启动参数
```bash
docker run -d \
  --name superset \
  -p 8088:8088 \
  -e "SUPERSET_ENV=production" \
  -e "FLASK_APP=superset.app:create_app()" \
  -e "FLASK_ENV=production" \
  -v $(pwd)/superset_config.py:/app/superset/superset_config.py \
  apache/superset \
  superset run -p 8088 --with-threads --reload --debugger --host=0.0.0.0
```

#### 6. 验证配置是否加载
进入 Docker 容器检查配置：
```bash
docker exec -it superset bash
python -c "from superset import app; print(app.config['CORS_OPTIONS'])"
```

### 关键注意事项

1. **CSRF 保护冲突**：
   - `401` 错误通常是由于 CSRF 保护导致的
   - 必须设置 `WTF_CSRF_ENABLED = False` 和 `DISABLE_CRSF_PROTECTION = True`

2. **生产环境安全**：
   ```python
   # 生产环境应严格限制源
   origins = [
       "https://your-production-domain.com",
       "https://admin.your-domain.com"
   ]
   
   # 启用安全 Cookies
   SESSION_COOKIE_SECURE = True
   SESSION_COOKIE_HTTPONLY = True
   SESSION_COOKIE_SAMESITE = 'Lax'
   ```

3. **浏览器缓存问题**：
   - 使用隐身模式测试
   - 清除 Superset 缓存：浏览器开发者工具 > Application > Clear storage

4. **网络层检查**：
   ```bash
   # 验证端口开放
   curl -I http://localhost:8088/api/v1/chart/
   
   # 检查响应头
   HTTP/1.1 200 OK
   Access-Control-Allow-Origin: http://localhost:8000
   Access-Control-Allow-Credentials: true
   ```

### 备用方案：Nginx 代理配置
```nginx
server {
    listen 80;
    server_name superset.your-domain.com;

    location / {
        proxy_pass http://superset:8088;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # 强制处理 CORS
        add_header 'Access-Control-Allow-Origin' '$http_origin' always;
        add_header 'Access-Control-Allow-Credentials' 'true' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' '*' always;
        
        # 处理预检请求
        if ($request_method = 'OPTIONS') {
            return 204;
        }
    }
}
```

### 调试建议
1. 检查响应头：
   ```bash
   curl -I -H "Origin: http://your-frontend" http://superset-host/api/v1/chart/
   ```
   确保包含：
   ```
   Access-Control-Allow-Origin: http://your-frontend
   Access-Control-Allow-Credentials: true
   ```

2. 查看 Superset 日志：
   ```bash
   docker logs superset --tail 100
   ```

3. 临时放宽安全设置测试：
   ```python
   # 临时配置
   CORS_OPTIONS = {"origins": ["*"]}
   WTF_CSRF_ENABLED = False
   ```

> 根据经验，90% 的此类问题是由 CSRF 保护和 CORS 配置冲突导致的。确保同时配置 `WTF_CSRF_ENABLED = False` 和 CORS 设置通常能解决问题。如果使用 HTTPS，请确保所有 URL 协议一致（全 HTTP 或全 HTTPS）。