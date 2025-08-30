# 数据库主键设计详解

## 1. 主键的基本概念

主键是好的数据库设计的一部分。主键是数据库确保数据行在整张表中唯一性的保障，它是定位到一条记录并且确保不会重复存储的逻辑机制。主键也同时可以被外键引用来建立表与表之间的关系。

主键必须满足以下特性：
- **唯一性**：表中的每一行数据必须有一个唯一的标识符
- **非空性**：主键值不能为NULL
- **稳定性**：一旦创建，主键值不应频繁变动
- **最小性**：主键应该尽可能简单，避免使用过多的列

## 2. 主键的类型

### 2.1 自然主键

自然主键是从实体的属性中选择的，它们在业务领域中本身就具有唯一性。例如：
- 公民身份证号
- 产品编码
- ISBN书号

**优点**：
- 具有业务含义，便于理解
- 无需额外存储空间
- 可以直接用于业务逻辑

**缺点**：
- 可能随业务规则变化而变化
- 可能过长或格式复杂，影响性能
- 在某些情况下可能不保证绝对唯一（如电子邮件地址可能被重用）

### 2.2 代理主键（伪主键）

难点是选择哪一列作为主键。大多数表中的每个属性的值都有可能被很多行使用。例如一个人的姓和名就一定会在表中重复出现，即使电子邮件地址或者美国社保编号或者税单编号也不能保证绝对不会重复。

在这样的表中，需要引入一个对于表的域模型无意义的新列来存储一个伪值。这一列被用作这张表的主键，从而通过它来确定表中的一条记录，即便其他的列允许出现适当的重复项。这种类型的主键列我们通常称其为伪主键或者代理键。

代理主键的常见实现方式：

#### 2.2.1 自增ID

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) UNIQUE
);
```

**优点**：
- 简单易用，数据库自动生成
- 占用空间小，通常为整数类型
- 索引效率高

**缺点**：
- 在分布式系统中可能产生冲突
- 可能泄露业务信息（如用户数量）
- 在数据迁移或合并时可能产生冲突

#### 2.2.2 UUID/GUID

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(200) NOT NULL,
    content TEXT
);
```

**优点**：
- 全局唯一，适合分布式系统
- 不泄露业务信息
- 可以在应用层生成，减轻数据库负担

**缺点**：
- 占用空间大（通常16字节）
- 索引效率较低
- 不便于人工阅读和记忆

### 2.3 组合主键

一个组合键包含了多个不同的列。组合键的典型场景是在像 BugsProducts 这样的交叉表中。主键需要确保一个给定的 bug_id 和 product_id 的组合在整张表中只能出现一次，虽然同一个值可能在很多不同的配对中出现。

```sql
CREATE TABLE bug_products (
    bug_id INTEGER,
    product_id INTEGER,
    PRIMARY KEY (bug_id, product_id),
    FOREIGN KEY (bug_id) REFERENCES bugs(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);
```

**优点**：
- 直接表达多对多关系
- 无需额外的ID列
- 自然地强制唯一性约束

**缺点**：
- 外键引用复杂
- 索引开销较大
- 主键值变更影响较大

## 3. 主键选择策略

### 3.1 何时使用自然主键

- 当实体有明确的、稳定的、全局唯一的标识符
- 当该标识符已经是业务流程的一部分
- 当该标识符不会过长或过于复杂

### 3.2 何时使用代理主键

- 当没有明确的自然主键
- 当自然主键可能变化
- 当需要跨系统集成
- 当表需要频繁的连接操作

### 3.3 何时使用组合主键

- 在多对多关系的关联表中
- 当组合的列本身就代表了业务实体的唯一性
- 当不需要通过单一ID引用该记录

## 4. 主键设计的常见误区

### 4.1 过度使用代理主键

你可能会发现在一张表中定义了 id 这一列作为主键，仅仅因为这么做符合传统，然而可能又同时存在另一列从逻辑上来说更为自然的主键，这一列甚至也具有 UNIQUE 约束。

在某些情况下，使用自然主键可能更有意义，特别是当该列已经具有唯一性约束时。

### 4.2 在关联表中使用额外的ID

然而，当你使用 id 这一列作为主键，约束就不再是 bug_id 和 product_id 的组合必须唯一了。当你在 bug_id 和 product_id 这两列上应用了唯一性约束，id 这一列就会变成多余的。

在多对多关系的关联表中，通常组合主键就足够了，无需额外的ID列。

### 4.3 忽略主键的性能影响

主键的选择会直接影响：
- 索引大小和效率
- 连接操作的性能
- 数据插入的速度
- 存储空间的使用

## 5. 实际案例分析

### 5.1 用户表的主键设计

在我们的音乐社区平台中，用户表采用了自增ID作为主键：

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    password_hash VARCHAR(128) NOT NULL,
    phone VARCHAR(20) UNIQUE,
    email VARCHAR(100) UNIQUE,
    -- 其他字段
);
```

这种设计的理由：
- 用户ID需要在多个相关表中被引用
- 自增ID简单高效，便于索引
- 虽然username、phone和email都是唯一的，但它们可能会变更

### 5.2 用户登录日志表的主键设计

```sql
CREATE TABLE user_login_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    login_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    login_ip VARCHAR(45) NOT NULL,
    login_type VARCHAR(20) NOT NULL,
    -- 其他字段
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

这里使用自增ID而非复合主键的原因：
- 日志表通常不需要基于业务字段的唯一性约束
- 自增ID便于按时间顺序检索记录
- 简化了外键引用和索引设计

### 5.3 第三方认证表的主键设计

```sql
CREATE TABLE third_party_auths (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    platform VARCHAR(20) NOT NULL,
    open_id VARCHAR(100) NOT NULL,
    -- 其他字段
    FOREIGN KEY (user_id) REFERENCES users(id),
    UNIQUE (platform, open_id)
);
```

这里同时使用了代理主键和唯一性约束：
- 代理主键简化了外键引用
- 唯一性约束确保了平台和开放ID的组合不会重复
- 这种设计兼顾了性能和业务规则

## 6. 主键设计的最佳实践

1. **优先考虑业务需求**：主键设计应该服务于业务需求，而不是技术偏好

2. **保持简单**：除非有明确的理由，否则使用单列主键

3. **考虑长期影响**：主键一旦确定，变更成本很高

4. **注意性能影响**：主键会影响索引、连接和查询性能

5. **适当冗余**：有时适当的冗余（如在关联表中同时使用组合键和代理键）可以兼顾性能和灵活性

6. **考虑分布式场景**：在分布式系统中，UUID可能比自增ID更适合

7. **文档化决策**：记录主键选择的理由，便于后续维护和理解

主键设计是数据库设计中的基础环节，良好的主键设计可以提高数据完整性、查询效率和系统可维护性。在实际项目中，应根据具体业务场景和技术要求，选择最适合的主键策略。

