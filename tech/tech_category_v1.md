# 音乐站点分类树技术方案设计

## 1. 需求分析

- 实现二级分类树结构，支持主题分类管理
- 需要支持分类的增删改查操作
- 需要支持分类的排序展示
- 需要支持分类树的层级控制（最多二级）

## 2. 技术方案对比

### 2.1 邻接表（Adjacency List）

**优点：**
- 实现简单，直观易懂
- 插入、更新、删除操作效率高
- 使用外键约束可以保证数据完整性
- 适合固定层级的树结构（如二级分类）

**缺点：**
- 查询多级子节点需要多次递归（但在二级分类场景下影响较小）
- 在深层次树结构中性能较差（但不影响我们的二级分类场景）

**表设计：**
```sql
CREATE TABLE category (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(50) NOT NULL,
    parent_id INTEGER,
    level INTEGER DEFAULT 1,
    sort_order INTEGER DEFAULT 0,
    FOREIGN KEY (parent_id) REFERENCES category(id)
);
```

**典型SQL操作：**
```sql
-- 插入一级分类
INSERT INTO category (name, parent_id, level) VALUES ('音乐', NULL, 1);

-- 插入二级分类
INSERT INTO category (name, parent_id, level) VALUES ('流行音乐', 1, 2);

-- 查询完整分类树
WITH RECURSIVE category_tree AS (
    SELECT id, name, parent_id, level, sort_order, 0 as depth
    FROM category
    WHERE parent_id IS NULL
    UNION ALL
    SELECT c.id, c.name, c.parent_id, c.level, c.sort_order, ct.depth + 1
    FROM category c
    JOIN category_tree ct ON c.parent_id = ct.id
)
SELECT * FROM category_tree ORDER BY sort_order;

-- 更新分类
UPDATE category SET name = '流行音乐2023' WHERE id = 2;

-- 删除分类（包含子分类）
DELETE FROM category WHERE id IN (
    WITH RECURSIVE subcategories AS (
        SELECT id FROM category WHERE id = 1
        UNION ALL
        SELECT c.id FROM category c
        JOIN subcategories s ON c.parent_id = s.id
    )
    SELECT id FROM subcategories
);
```

### 2.2 枚举路径（Path Enumeration）

**优点：**
- 查询祖先和后代节点方便
- 可以直观地看到节点的完整路径

**缺点：**
- 数据存储冗余
- 不能保证引用完整性
- 路径更新维护复杂
- 不适合频繁变动的场景

**表设计：**
```sql
CREATE TABLE category (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(50) NOT NULL,
    path VARCHAR(255) NOT NULL,  -- 存储完整路径，如 /1/2/3/
    level INTEGER DEFAULT 1,
    sort_order INTEGER DEFAULT 0
);

-- 创建路径索引
CREATE INDEX idx_category_path ON category(path);
```

**典型SQL操作：**
```sql
-- 插入一级分类
INSERT INTO category (name, path, level) VALUES ('音乐', '/1/', 1);

-- 插入二级分类
INSERT INTO category (name, path, level) 
VALUES ('流行音乐', '/1/2/', 2);

-- 查询某个分类的所有子分类
SELECT * FROM category 
WHERE path LIKE '/1/%' AND id != 1;

-- 查询某个分类的所有父分类
SELECT * FROM category 
WHERE '/1/2/3/' LIKE path || '%' AND length(path) < length('/1/2/3/');

-- 更新分类（需要同时更新子分类的路径）
UPDATE category 
SET path = REPLACE(path, '/1/', '/99/') 
WHERE path LIKE '/1/%';
```

### 2.3 嵌套集（Nested Sets）

**优点：**
- 查询子树效率高
- 适合读多写少的场景

**缺点：**
- 插入和删除操作复杂且成本高
- 需要维护左右值
- 不能保证引用完整性
- 实现复杂，维护困难

**表设计：**
```sql
CREATE TABLE category (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(50) NOT NULL,
    lft INTEGER NOT NULL,  -- 左值
    rgt INTEGER NOT NULL,  -- 右值
    level INTEGER DEFAULT 1,
    sort_order INTEGER DEFAULT 0
);

-- 创建左右值索引
CREATE INDEX idx_category_nested ON category(lft, rgt);
```

**典型SQL操作：**
```sql
-- 插入根节点
INSERT INTO category (name, lft, rgt, level) VALUES ('音乐', 1, 2, 1);

-- 插入子节点（需要先更新其他节点的左右值）
UPDATE category SET rgt = rgt + 2 WHERE rgt >= 2;
UPDATE category SET lft = lft + 2 WHERE lft >= 2;
INSERT INTO category (name, lft, rgt, level) VALUES ('流行音乐', 2, 3, 2);

-- 查询某个节点的所有子节点
SELECT child.* 
FROM category AS node, category AS child 
WHERE child.lft BETWEEN node.lft AND node.rgt 
AND node.id = 1;

-- 查询某个节点的所有父节点
SELECT parent.* 
FROM category AS node, category AS parent 
WHERE node.lft BETWEEN parent.lft AND parent.rgt 
AND node.id = 2 
ORDER BY parent.lft;

-- 删除节点（包含子节点）
DELETE FROM category 
WHERE lft BETWEEN 2 AND 3;
UPDATE category SET 
    rgt = rgt - 2 
    WHERE rgt > 3;
UPDATE category SET 
    lft = lft - 2 
    WHERE lft > 3;
```

### 2.4 闭包表（Closure Table）

**优点：**
- 查询性能好
- 支持多棵树
- 保证数据完整性

**缺点：**
- 需要额外的关系表
- 存储空间占用大
- 对于简单的二级分类来说过于复杂

**表设计：**
```sql
-- 分类表
CREATE TABLE category (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(50) NOT NULL,
    level INTEGER DEFAULT 1,
    sort_order INTEGER DEFAULT 0
);

-- 分类关系表
CREATE TABLE category_closure (
    ancestor_id INTEGER,
    descendant_id INTEGER,
    depth INTEGER,
    PRIMARY KEY (ancestor_id, descendant_id),
    FOREIGN KEY (ancestor_id) REFERENCES category(id),
    FOREIGN KEY (descendant_id) REFERENCES category(id)
);
```

**典型SQL操作：**
```sql
-- 插入根节点
INSERT INTO category (name, level) VALUES ('音乐', 1);
INSERT INTO category_closure (ancestor_id, descendant_id, depth)
VALUES (1, 1, 0);

-- 插入子节点
INSERT INTO category (name, level) VALUES ('流行音乐', 2);
INSERT INTO category_closure (ancestor_id, descendant_id, depth)
SELECT ancestor_id, 2, depth + 1
FROM category_closure
WHERE descendant_id = 1
UNION ALL SELECT 2, 2, 0;

-- 查询某个节点的所有子节点
SELECT c.* 
FROM category c
JOIN category_closure cc ON c.id = cc.descendant_id
WHERE cc.ancestor_id = 1;

-- 查询某个节点的所有父节点
SELECT c.* 
FROM category c
JOIN category_closure cc ON c.id = cc.ancestor_id
WHERE cc.descendant_id = 2;

-- 删除节点（包含子节点）
DELETE FROM category_closure
WHERE descendant_id IN (
    SELECT descendant_id
    FROM category_closure
    WHERE ancestor_id = 1
);
DELETE FROM category
WHERE id IN (
    SELECT descendant_id
    FROM category_closure
    WHERE ancestor_id = 1
);
```

## 3. 方案选择

综合考虑以下因素：
1. 我们只需要支持二级分类
2. 分类数据变动频率适中
3. 需要良好的数据完整性
4. 实现和维护的复杂度要适中
5. 使用SQLite数据库

**最终选择：邻接表方案**

理由：
1. 邻接表方案实现简单直观，维护成本低
2. 对于二级分类场景，不存在深层递归查询的性能问题
3. 可以通过外键约束保证数据完整性
4. 支持通过level字段轻松控制分类层级
5. 适合中小规模的分类管理需求

## 4. 具体实现

### 4.1 数据库表设计

```sql
CREATE TABLE category (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(50) NOT NULL,
    parent_id INTEGER,
    level INTEGER DEFAULT 1,
    sort_order INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_id) REFERENCES category(id)
);
```

### 4.2 核心字段说明

- id：分类唯一标识
- name：分类名称
- parent_id：父分类ID，一级分类为null
- level：分类层级（1-一级分类，2-二级分类）
- sort_order：排序权重

### 4.3 关键功能实现

1. 创建分类：校验层级限制
2. 查询分类树：按层级和排序获取
3. 更新分类：维护层级关系
4. 删除分类：级联删除子分类

## 5. 性能优化

1. 合理使用索引
   - parent_id建立索引
   - level + sort_order组合索引

2. 缓存优化
   - 缓存完整分类树结构
   - 分类变更时更新缓存

## 6. 扩展性考虑

1. 预留分类属性字段
2. 支持分类状态管理
3. 预留分类权限控制