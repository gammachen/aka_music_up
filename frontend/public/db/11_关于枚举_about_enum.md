# 数据库列约束方案对比

## 实现方式

### 1. ENUM类型
```sql
-- MySQL示例
CREATE TABLE Project (
  status ENUM('draft', 'review', 'published') NOT NULL
);
```
**特点**：
- 修改需执行ALTER TABLE
- 无法存储额外元数据
- 类型安全但扩展性差
- 排序基于定义序数（非字母序）

### 2. CHECK约束
```sql
-- MySQL 8.0示例
CREATE TABLE Project (
  status VARCHAR(20) CHECK (status IN ('draft', 'review', 'published'))
);
```
**特点**：
- 修改需重建约束
- 支持复杂表达式
- 无额外存储开销

### 3. 检查表（推荐方案）
```sql
-- 状态字典表
CREATE TABLE ProjectStatus (
  code VARCHAR(20) PRIMARY KEY,
  display_name VARCHAR(50),
  color VARCHAR(7)
);

-- 主表引用
CREATE TABLE Project (
  status VARCHAR(20) REFERENCES ProjectStatus(code)
);
```
**特点**：
- 可扩展性强（支持元数据存储）
- 维护成本低（独立维护状态列表）
- 查询需要JOIN操作
- 通过display_name字段实现可控排序

## 综合对比
| 维度        | ENUM       | CHECK      | 检查表      |
|------------|------------|------------|------------|
| 维护成本     | 高         | 中         | 低         |
| 查询性能     | 最优       | 最优       | 需JOIN     |
| 扩展性       | 差         | 中         | 优         |
| 元数据支持   | 不支持     | 不支持     | 支持       |
| 排序行为     | 序数排序    | 字母序      | 可控排序    |

## 与现有方案统一
（参见tech_db_v1.md中的状态约束）
```sql
-- 现有订单状态约束（CHECK方式）
CREATE TABLE orders (
  status VARCHAR(20) CHECK(status IN ('pending', 'success', 'failed'))
);
```
## 排序示例
```sql
-- ENUM排序（按定义序数）
CREATE TABLE enum_sort_demo (
  status ENUM('published', 'draft', 'review') 
);

-- VARCHAR排序（按字母序）
CREATE TABLE varchar_sort_demo (
  status VARCHAR(20) CHECK(status IN('published', 'draft', 'review'))
);

/*
插入测试数据后：
ENUM排序结果：published -> draft -> review
VARCHAR排序结果：draft -> published -> review
*/
```

建议新项目采用检查表方案，现有系统保持CHECK约束以保持兼容性。