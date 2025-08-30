---

### **技术方案：基于Keycloak实现SSO整合**

#### **一、架构设计**
```plaintext
                            +---------------------+
                            |     Keycloak        |
                            | (SSO服务提供者)     |
                            +----------+----------+
                                       ↑
                                       | 认证/授权
                                       ↓
+----------------+              +------+------+              +----------------+
|  应用A         |              |  反向代理   |              |  应用B         |
| (a.lc.ltd:5001)| ←HTTPS→      | (Nginx/Caddy)| ←HTTPS→      | (b.lc.ltd:5002)|
+----------------+              +-------------+              +----------------+
```

#### **二、技术选型**
- **SSO服务端**：[Keycloak](https://www.keycloak.org/)  
  - 开源、支持OAuth 2.0/OpenID Connect (OIDC)、提供用户联邦、多因素认证等功能。
- **客户端库**：[Flask-OIDC](https://flask-oidc.readthedocs.io/) 或 [Authlib](https://authlib.org/)  
  - 简化Flask应用的OIDC集成。
- **反向代理**：Nginx/Caddy  
  - 统一域名、HTTPS支持（解决跨域问题）。

---

### **三、详细实施步骤**

#### **步骤1：部署Keycloak**
1. **安装Keycloak**（Docker示例）：
   ```bash
   docker run -d --name keycloak \
     -p 8080:8080 \
     -e KEYCLOAK_ADMIN=admin \
     -e KEYCLOAK_ADMIN_PASSWORD=admin \
     quay.io/keycloak/keycloak:24.0.2 start-dev
   ```
2. **访问管理控制台**：  
   `http://localhost:8080/admin`（后续通过反向代理配置HTTPS）。

#### **步骤2：配置Keycloak**
1. **创建Realm**：  
   - 名称：`lc-sso`（根据业务命名）。
2. **创建Clients**：  
   - 应用A：`client-a`  
     - Valid Redirect URIs: `https://a.lc.ltd/*`
   - 应用B：`client-b`  
     - Valid Redirect URIs: `https://b.lc.ltd/*`
   - 启用`OIDC Compatibility Mode`。
3. **配置用户存储**：  
   - 同步现有用户数据（见**数据迁移**部分）。

#### **步骤3：配置反向代理（以Caddy为例）**
```Caddyfile
# Caddyfile
a.lc.ltd {
    reverse_proxy localhost:5001
    tls internal_certs
}

b.lc.ltd {
    reverse_proxy localhost:5002
    tls internal_certs
}

sso.lc.ltd {
    reverse_proxy localhost:8080
    tls internal_certs
}
```

#### **步骤4：改造Flask应用（以应用A为例）**
1. **安装依赖**：
   ```bash
   pip install flask-oidc
   ```
2. **配置OIDC**：
   ```python
   # app.py
   from flask import Flask, session, redirect
   from flask_oidc import OpenIDConnect

   app = Flask(__name__)
   app.secret_key = 'your-secret-key'

   # Keycloak配置
   app.config.update({
       'OIDC_CLIENT_SECRETS': {
           "web": {
               "issuer": "https://sso.lc.ltd/realms/lc-sso",
               "auth_uri": "https://sso.lc.ltd/realms/lc-sso/protocol/openid-connect/auth",
               "client_id": "client-a",
               "client_secret": "生成的应用A的密钥",
               "redirect_uris": ["https://a.lc.ltd/oidc/callback"],
           }
       }
   })
   oidc = OpenIDConnect(app)

   # 业务路由改造
   @app.route('/business')
   @oidc.require_login
   def business():
       user = oidc.user_getinfo(['email', 'preferred_username'])
       return f"Welcome {user['preferred_username']}!"

   # 登录路由替换
   @app.route('/login')
   def login():
       return oidc.redirect_to_auth_server()
   ```

#### **步骤5：数据迁移**
1. **导出现有用户数据**：
   ```bash
   # 从SQLite导出用户表（应用A/B）
   sqlite3 app_a.db "SELECT username, password_hash FROM users" > users.csv
   ```
2. **导入Keycloak**（脚本示例）：
   ```python
   import csv
   from keycloak import KeycloakAdmin

   admin = KeycloakAdmin(
       server_url='https://sso.lc.ltd',
       username='admin',
       password='admin',
       realm_name='lc-sso'
   )

   with open('users.csv') as f:
       reader = csv.reader(f)
       for row in reader:
           username, password_hash = row
           admin.create_user({
               "username": username,
               "credentials": [{
                   "type": "password",
                   "value": password_hash,
                   "temporary": False
               }],
               "enabled": True
           })
   ```

---

### **四、关键问题与解决方案**

#### **1. 跨域问题**
- **方案**：通过反向代理统一域名（`a.lc.ltd`/`b.lc.ltd`），Keycloak回调地址需严格匹配。
- **验证**：确保所有服务的`Access-Control-Allow-Origin`头正确配置。

#### **2. Session管理**
- **原系统改造**：移除Flask本地Session，改用OIDC的Token验证。
- **安全性**：启用Keycloak的`Refresh Token`机制，自动续期登录状态。

#### **3. 用户数据关联**
- **业务数据迁移**：  
  - 保留原数据库中的业务数据，通过`sub`（用户唯一标识）关联Keycloak用户。
  ```sql
  -- 原用户表增加字段
  ALTER TABLE orders ADD COLUMN user_sub VARCHAR(36);
  ```

---

### **五、安全增强措施**
1. **HTTPS强制启用**：  
   - 通过反向代理配置，禁止HTTP访问。
2. **密钥管理**：  
   - Keycloak客户端密钥使用环境变量或Vault存储。
3. **审计日志**：  
   - 启用Keycloak的日志记录，监控异常登录行为。

---

### **六、未来扩展**
1. **多因素认证（MFA）**：  
   - Keycloak支持TOTP、短信/邮件验证码。
2. **社会化登录**：  
   - 集成微信、GitHub等第三方登录。
3. **用户自助服务**：  
   - 启用Keycloak的密码重置、账户锁定功能。

---

### **总结：从“孤岛”到“联邦”**
通过Keycloak实现SSO后：  
- **用户体验**：一次登录，多系统通行。  
- **安全提升**：集中管控认证策略，避免密码泄露风险。  
- **维护成本**：用户管理统一化，降低多系统同步负担。  

**实施关键**：  
- 确保反向代理和HTTPS配置正确。  
- 彻底测试跨系统会话状态同步。  
- 提供用户迁移指南（如密码重置流程）。


