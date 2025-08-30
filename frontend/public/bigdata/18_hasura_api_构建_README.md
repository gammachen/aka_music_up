下面是一个使用 **Hasura** 与 **Postgres** 构建 API 服务的详细 Demo，包含环境搭建、数据建模和 API 测试步骤：

---

### 1. 环境准备
#### 使用 Docker 启动 Postgres 和 Hasura
创建 `docker-compose.yml` 文件：
```yaml
version: '3.8'

services:
  # Postgres 服务
  postgres:
    image: postgres:15
    container_name: postgres
    environment:
      POSTGRES_PASSWORD: root
      POSTGRES_DB: movie_database
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # Hasura GraphQL 引擎
  hasura:
    image: hasura/graphql-engine
    container_name: hasura
    ports:
      - "8080:8080"
    depends_on:
      - postgres
    environment:
      HASURA_GRAPHQL_DATABASE_URL: postgres://postgres:root@postgres:5432/movie_database
      HASURA_GRAPHQL_ENABLE_CONSOLE: "true"  # 启用控制台
      HASURA_GRAPHQL_DEV_MODE: "true"
    volumes:
      - hasura_metadata:/hasura-metadata

volumes:
  postgres_data:
  hasura_metadata:
```

#### 启动服务
```bash
docker-compose up -d
```

---

### 2. 初始化 Postgres 数据
进入 Postgres 容器创建表：
```bash
docker exec -it postgres psql -U postgres -d movie_database
```
执行 SQL：
```sql
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  email VARCHAR(100) UNIQUE NOT NULL
);

CREATE TABLE posts (
  id SERIAL PRIMARY KEY,
  title VARCHAR(255) NOT NULL,
  content TEXT,
  user_id INTEGER REFERENCES users(id)
);

INSERT INTO users (name, email) VALUES
('Alice', 'alice@example.com'),
('Bob', 'bob@example.com');

INSERT INTO posts (title, content, user_id) VALUES
('Hello', 'World!', 1),
('GraphQL', 'With Postgres', 2);
```

---

### 3. 配置 Hasura
1. 访问控制台：`http://localhost:8080`
2. **连接数据库**：
   - 进入 `Data` → `Connect Database`
   - 选择 Postgres，填写连接信息（自动从 `docker-compose.yml` 继承，直接确认即可）

3. **跟踪表**：
   - 进入 `Data` → `movie_database` → `Tables`
   - 点击 `Track` 按钮跟踪 `users` 和 `posts` 表

4. **设置关系**（可选）：
   - 在 `posts` 表中创建对象关系（Object Relationship）：
     - 名称：`author`
     - 引用：`user_id` → `users.id`

---

### 4. 使用 GraphQL API
#### 查询所有用户及其文章
```graphql
query GetUsersWithPosts {
  users {
    id
    name
    email
    posts {  # 自动解析关系
      id
      title
    }
  }
}
```

#### 创建新用户
```graphql
mutation CreateUser {
  insert_users_one(object: {
    name: "Charlie",
    email: "charlie@example.com"
  }) {
    id
    name
  }
}
```

#### 带条件的查询
```graphql
query GetPostsByUser {
  posts(where: {user_id: {_eq: 1}}) {
    title
    content
    author {  # 通过关系获取用户信息
      name
    }
  }
}
```

---

### 5. 测试 API
#### 使用 cURL 示例
```bash
# 查询用户
curl -X POST http://localhost:8080/v1/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "query { users { id name posts { title } } }"}'
```

#### 响应示例
```json
{
  "data": {
    "users": [
      {
        "id": 1,
        "name": "Alice",
        "posts": [
          {"title": "Hello"}
        ]
      },
      {
        "id": 2,
        "name": "Bob",
        "posts": [
          {"title": "GraphQL"}
        ]
      }
    ]
  }
}
```

---

### 6. 进阶配置
1. **权限控制**：
   - 在 `Settings` → `Roles` 中创建角色（如 `user`）
   - 为表配置行级权限（如限制用户只能访问自己的数据）

2. **事件触发器**：
   - 当插入新用户时自动发送欢迎邮件
   - 配置路径：`Events` → `Create Trigger`

3. **远程 Schema**：
   - 整合自定义 GraphQL 服务（如身份验证）

---

### 常见问题解决
1. **Hasura 连接 Postgres 失败**：
   - 检查数据库 URL 格式：`postgres://<user>:<password>@<host>:<port>/<dbname>`
   - 确保 Postgres 版本 ≥ 12

2. **控制台无法访问**：
   - 确认 Hasura 容器日志无报错：`docker logs hasura`

3. **关系未生效**：
   - 外键约束需在 Postgres 中预先创建
   - 在 Hasura 控制台重新跟踪表

> **注意**：Hasura 对 Postgres 支持最为完善，推荐生产环境使用。

通过这个 Demo，您已实现：
✅ 自动生成 GraphQL CRUD API  
✅ 数据库关系映射  
✅ 实时查询（Subscription）  
✅ 图形化管理界面