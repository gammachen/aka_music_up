# 数据库子查询优化技术详解

## 引言

子查询是SQL语言中强大而灵活的特性，允许在查询语句内部嵌套另一个查询语句。虽然子查询提供了表达复杂查询逻辑的便捷方式，但它们通常是查询执行过程中的性能瓶颈。本文将系统地介绍子查询的分类、优化技术及最佳实践，帮助开发者和数据库管理员编写高效的查询语句。

## 表1-1 子查询优化技术概览

| 优化技术 | 适用子查询类型 | 优化原理 | 性能提升 |
| ------- | ------------ | ------- | ------- |
| 子查询展开 | 非相关子查询 | 将子查询转换为等价的连接操作 | 显著 |
| 子查询合并 | 相关/非相关子查询 | 将子查询与主查询合并为单一查询块 | 显著 |
| 子查询物化 | 相关/非相关子查询 | 预先计算并缓存子查询结果 | 中等 |
| 谓词下推 | 相关子查询 | 将外层查询的条件下推到子查询 | 中等 |
| 子查询提升 | FROM子句中的子查询 | 将子查询提升到与主查询同级 | 中等 |

## 1. 子查询的分类与特点

子查询可以从多个维度进行分类，每种类型的子查询具有不同的特点和优化策略。

### 1.1 按位置分类

子查询可以出现在SQL语句的多个位置，不同位置的子查询具有不同的语义和限制：

- **目标列位置**：出现在SELECT子句中，必须是标量子查询（返回单一值）
  ```sql
  SELECT employee_name, (SELECT MAX(salary) FROM salaries WHERE employee_id = e.id) AS max_salary
  FROM employees e;
  ```

- **FROM子句位置**：作为派生表，不能是相关子查询
  ```sql
  SELECT d.department_name, avg_salary
  FROM departments d
  JOIN (SELECT department_id, AVG(salary) AS avg_salary FROM employees GROUP BY department_id) AS dept_avg
    ON d.department_id = dept_avg.department_id;
  ```

- **WHERE子句位置**：作为条件表达式的一部分，可以是相关或非相关子查询
  ```sql
  SELECT * FROM employees
  WHERE salary > (SELECT AVG(salary) FROM employees);
  ```

- **JOIN/ON子句位置**：类似于FROM和WHERE子句的组合
  ```sql
  SELECT e.employee_name, d.department_name
  FROM employees e
  JOIN departments d ON e.department_id = d.department_id AND d.location IN (SELECT location FROM premium_locations);
  ```

- **GROUP BY子句位置**：理论上可以使用，但实际应用意义有限

- **ORDER BY子句位置**：理论上可以使用，但实际应用意义有限

### 1.2 按相关性分类

根据子查询是否引用外层查询的表，子查询可分为：

- **相关子查询**：子查询中引用了外层查询的表，执行依赖于外层查询的值
  ```sql
  SELECT * FROM employees e
  WHERE salary > (SELECT AVG(salary) FROM employees WHERE department_id = e.department_id);
  ```

- **非相关子查询**：子查询独立于外层查询，可以单独执行
  ```sql
  SELECT * FROM employees
  WHERE department_id IN (SELECT department_id FROM departments WHERE location = 'Headquarters');
  ```

### 1.3 按谓词类型分类

根据子查询使用的谓词操作符，可分为：

- **[NOT] IN/ALL/ANY/SOME子查询**：检查值是否在子查询结果集中
  ```sql
  -- IN子查询
  SELECT * FROM employees WHERE department_id IN (SELECT department_id FROM departments WHERE budget > 1000000);
  
  -- ALL子查询
  SELECT * FROM employees WHERE salary > ALL (SELECT avg_salary FROM department_stats);
  
  -- ANY子查询
  SELECT * FROM employees WHERE salary > ANY (SELECT salary FROM employees WHERE department_id = 10);
  ```

- **[NOT] EXISTS子查询**：检查子查询结果集是否为空
  ```sql
  SELECT * FROM departments d
  WHERE EXISTS (SELECT 1 FROM employees e WHERE e.department_id = d.department_id AND e.salary > 100000);
  ```

- **其他子查询**：如比较操作符（=, >, <等）与标量子查询组合
  ```sql
  SELECT * FROM employees
  WHERE hire_date = (SELECT MIN(hire_date) FROM employees);
  ```

### 1.4 按复杂度分类

根据查询的构成复杂程度，子查询可分为：

- **SPJ子查询**：由选择、投影、连接操作组成
  ```sql
  SELECT * FROM employees
  WHERE department_id IN (SELECT department_id FROM departments WHERE location = 'Headquarters');
  ```

- **GROUP BY子查询**：包含分组和聚集操作
  ```sql
  SELECT * FROM departments
  WHERE department_id IN (SELECT department_id FROM employees GROUP BY department_id HAVING COUNT(*) > 10);
  ```

- **其他子查询**：包含更复杂的操作，如Top-N、LIMIT/OFFSET、集合操作等
  ```sql
  SELECT * FROM employees
  WHERE department_id IN (SELECT department_id FROM departments ORDER BY budget DESC LIMIT 3);
  ```

### 1.5 按结果集类型分类

根据子查询返回的结果集类型，可分为：

- **标量子查询**：返回单一值
  ```sql
  SELECT employee_name, (SELECT MAX(salary) FROM salaries WHERE employee_id = e.id) AS max_salary
  FROM employees e;
  ```

- **列子查询**：返回单一列多行
  ```sql
  SELECT * FROM employees
  WHERE department_id IN (SELECT department_id FROM departments WHERE location = 'Headquarters');
  ```

- **行子查询**：返回单一行多列
  ```sql
  SELECT * FROM employees
  WHERE (department_id, job_id) = (SELECT department_id, job_id FROM job_history WHERE employee_id = 101);
  ```

- **表子查询**：返回多行多列
  ```sql
  SELECT e.employee_name, d.department_name
  FROM employees e
  JOIN (SELECT department_id, department_name FROM departments WHERE active = 1) d
    ON e.department_id = d.department_id;
  ```

## 2. 子查询优化技术

### 2.0 子查询优化的思路

#### 2.0.1 为什么需要子查询优化

在数据库实现早期，查询优化器对子查询一般采用嵌套执行的方式，即对父查询中的每一行，都执行一次子查询，这样子查询会执行很多次。这种执行方式效率很低。而对子查询进行优化，可能带来几个数量级的查询效率提高。

子查询转变成为连接操作之后，会得到如下好处：
- 子查询不用执行很多次
- 优化器可以根据统计信息来选择不同的连接方法和不同的连接顺序
- 子查询中的连接条件、过滤条件分别变成了父查询的连接条件、过滤条件，优化器可以对这些条件进行下推，以提高执行效率

#### 2.0.2 主要的子查询优化技术

子查询优化技术的思路主要包括：

- **子查询合并（Subquery Coalescing）**：在某些条件下（语义等价：两个查询块产生同样的结果集），多个子查询能够合并成一个子查询。这样可以把多次表扫描、多次连接减少为单次表扫描和单次连接，例如：
  ```sql
  SELECT * FROM t1 WHERE a1<10 AND (
    EXISTS (SELECT a2 FROM t2 WHERE t2.a2<5 AND t2.b2=1) OR 
    EXISTS (SELECT a2 FROM t2 WHERE t2.a2<5 AND t2.b2=2)
  );
  ```
  可优化为：
  ```sql
  SELECT * FROM t1 WHERE a1<10 AND (
    EXISTS (SELECT a2 FROM t2 WHERE t2.a2<5 AND (t2.b2=1 OR t2.b2=2))
  );
  ```

- **子查询展开（Subquery Unnesting）**：又称子查询反嵌套或子查询上拉。把一些子查询置于外层的父查询中，作为连接关系与外层父查询并列，其实质是把某些子查询重写为等价的多表连接操作。例如：
  ```sql
  SELECT * FROM t1, (SELECT * FROM t2 WHERE t2.a2 >10) v_t2 
  WHERE t1.a1<10 AND v_t2.a2<20;
  ```
  可优化为：
  ```sql
  SELECT * FROM t1, t2 
  WHERE t1.a1<10 AND t2.a2<20 AND t2.a2 >10;
  ```

- **聚集子查询消除（Aggregate Subquery Elimination）**：聚集函数上推，将子查询转变为一个新的不包含聚集函数的子查询，并与父查询的部分或者全部表做左外连接。通常，一些系统支持的是标量聚集子查询消除，例如：
  ```sql
  SELECT * FROM t1 WHERE t1.a1>(SELECT avg(t2.a2) FROM t2);
  ```

- **其他技术**：利用窗口函数消除子查询的技术（Remove Subquery using Window functions，RSW）、子查询推进（Push Subquery）等技术也可用于子查询的优化。

#### 2.0.3 子查询展开的规则

子查询展开是一种最为常用的子查询优化技术。如果子查询中出现了聚集、GROUP BY、DISTINCT子句，则子查询只能单独求解，不可以上拉到上层。如果子查询只是一个简单格式（SPJ格式）的查询语句，则可以上拉到上层，这样往往能提高查询效率。

把子查询上拉到上层查询，前提是上拉（展开）后的结果不能带来多余的元组，所以子查询展开需要遵循如下规则：

- 如果上层查询的结果没有重复（即SELECT子句中包含主码），则可以展开其子查询，并且展开后的查询的SELECT子句前应加上DISTINCT标志。
- 如果上层查询的SELECT语句中有DISTINCT标志，则可以直接进行子查询展开。
- 如果内层查询结果没有重复元组，则可以展开。

子查询展开的具体步骤如下：
1. 将子查询和上层查询的FROM子句连接为同一个FROM子句，并且修改相应的运行参数。
2. 将子查询的谓词符号进行相应修改（如：IN修改为=ANY）。
3. 将子查询的WHERE条件作为一个整体与上层查询的WHERE条件合并，并用AND条件连接词连接，从而保证新生成的谓词与原谓词的上下文意思相同，且成为一个整体。

#### 2.0.4 最常见的子查询类型的优化

##### IN类型子查询的优化

IN类型有3种不同的格式：

格式一：
```
outer_expr [NOT] IN (SELECT inner_expr FROM ... WHERE subquery_where)
```

格式二：
```
outer_expr =ANY (SELECT inner_expr FROM ... WHERE subquery_where)
```

格式三：
```
(oe_1, ..., oe_N) [NOT] IN (SELECT ie_1, ..., ie_N FROM ... WHERE subquery_where)
```

IN类型子查询的优化可分为几种情况：

1. **情况一**：outer_expr和inner_expr均为非NULL值
   - 优化后的表达式（外部条件outer_expr下推到子查询中）：
     ```
     EXISTS (SELECT 1 FROM ... WHERE subquery_where AND outer_expr=inner_expr)
     ```
   - 子查询优化需要满足的条件：
     - outer_expr和inner_expr不能为NULL
     - 不需要从结果为FALSE的子查询中区分NULL

2. **情况二**：outer_expr是非NULL值（情况一的两个转换条件中至少有一个不满足时）
   - 优化后的表达式：
     ```
     EXISTS (SELECT 1 FROM ... WHERE subquery_where AND (outer_expr=inner_expr OR inner_expr IS NULL))
     ```

3. **情况三**：outer_expr为NULL值
   - 原先的表达式等价于：
     ```
     NULL IN (SELECT inner_expr FROM ... WHERE subquery_where)
     ```

需要注意的是：
- 谓词IN等价于=ANY
- 带有谓词IN的子查询，如果满足上述3种情况，可以做等价变换，把外层的条件下推到子查询中，变形为一个EXISTS类型的逻辑表达式判断；而子查询为EXISTS类型则可以被半连接算法实现优化。

### 2.1 子查询展开（Subquery Unnesting）

子查询展开是将子查询转换为等价的连接操作，这是最常见也是最有效的子查询优化技术之一。

#### 2.1.1 原理

子查询展开基于关系代数的等价变换规则，将嵌套的查询结构转换为扁平化的连接结构，使优化器能够有更多选择来确定最优的执行计划。

#### 2.1.2 适用场景

- **IN子查询**：可转换为内连接
- **EXISTS子查询**：可转换为半连接（Semi-join）
- **NOT EXISTS子查询**：可转换为反连接（Anti-join）
- **标量子查询**：可转换为左外连接

#### 2.1.3 优化示例

**优化前（IN子查询）：**
```sql
SELECT * FROM employees
WHERE department_id IN (SELECT department_id FROM departments WHERE location = 'Headquarters');
```

**优化后（转换为连接）：**
```sql
SELECT e.*
FROM employees e
JOIN departments d ON e.department_id = d.department_id
WHERE d.location = 'Headquarters';
```

**优化前（EXISTS子查询）：**
```sql
SELECT * FROM departments d
WHERE EXISTS (SELECT 1 FROM employees e WHERE e.department_id = d.department_id AND e.salary > 100000);
```

**优化后（转换为半连接）：**
```sql
SELECT DISTINCT d.*
FROM departments d
JOIN employees e ON d.department_id = e.department_id
WHERE e.salary > 100000;
```

### 2.2 子查询合并（Subquery Merging）

子查询合并是将子查询与主查询合并为单一查询块的优化技术。

#### 2.2.1 原理

当子查询与主查询之间存在简单的关系时，可以将它们合并为单一查询块，减少查询处理的复杂性。

#### 2.2.2 适用场景

- 简单的标量子查询
- WHERE子句中的简单条件子查询

#### 2.2.3 优化示例

**优化前：**
```sql
SELECT * FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);
```

**优化后：**
```sql
SELECT e.*
FROM employees e, (SELECT AVG(salary) AS avg_sal FROM employees) AS stats
WHERE e.salary > stats.avg_sal;
```

### 2.3 子查询物化（Subquery Materialization）

子查询物化是预先计算并缓存子查询结果的优化技术。

#### 2.3.1 原理

对于在外层查询中多次使用的子查询，或者相关子查询中重复执行的部分，可以预先计算并缓存结果，避免重复计算。

#### 2.3.2 适用场景

- 外层查询中多次使用的子查询
- 相关子查询中的不变部分

#### 2.3.3 优化示例

**优化前（相关子查询）：**
```sql
SELECT e.employee_name,
       (SELECT AVG(salary) FROM employees WHERE department_id = e.department_id) AS dept_avg_salary
FROM employees e;
```

**优化后（物化子查询）：**
```sql
WITH dept_avg AS (
  SELECT department_id, AVG(salary) AS avg_salary
  FROM employees
  GROUP BY department_id
)
SELECT e.employee_name, d.avg_salary AS dept_avg_salary
FROM employees e
LEFT JOIN dept_avg d ON e.department_id = d.department_id;
```

### 2.4 谓词下推（Predicate Pushdown）

谓词下推是将外层查询的条件下推到子查询中的优化技术。

#### 2.4.1 原理

通过将外层查询的条件下推到子查询中，可以在子查询阶段就过滤掉不满足条件的数据，减少中间结果集的大小。

#### 2.4.2 适用场景

- FROM子句中的子查询
- 相关子查询

#### 2.4.3 优化示例

**优化前：**
```sql
SELECT * FROM (
  SELECT e.*, d.department_name
  FROM employees e
  JOIN departments d ON e.department_id = d.department_id
) AS emp_dept
WHERE salary > 50000 AND department_name = 'Sales';
```

**优化后（谓词下推）：**
```sql
SELECT * FROM (
  SELECT e.*, d.department_name
  FROM employees e
  JOIN departments d ON e.department_id = d.department_id
  WHERE e.salary > 50000 AND d.department_name = 'Sales'
) AS emp_dept;
```

### 2.5 子查询提升（Subquery Lifting）

子查询提升是将FROM子句中的子查询提升到与主查询同级的优化技术。

#### 2.5.1 原理

通过将FROM子句中的子查询提升为与主查询同级的查询块，可以使优化器有更多选择来确定最优的连接顺序。

#### 2.5.2 适用场景

- FROM子句中的复杂子查询

#### 2.5.3 优化示例

**优化前：**
```sql
SELECT e.employee_name, d.department_name
FROM employees e
JOIN (
  SELECT department_id, department_name
  FROM departments
  WHERE budget > 1000000
) d ON e.department_id = d.department_id;
```

**优化后（子查询提升）：**
```sql
SELECT e.employee_name, d.department_name
FROM employees e
JOIN departments d ON e.department_id = d.department_id
WHERE d.budget > 1000000;
```

## 3. 子查询优化的执行计划分析

### 3.1 执行计划中的子查询处理

数据库优化器在处理子查询时，会根据查询的特点选择不同的执行策略：

- **嵌套循环执行**：对外层查询的每一行，执行一次子查询
- **一次性执行**：先执行子查询，将结果缓存，然后处理外层查询
- **连接转换执行**：将子查询转换为连接操作后执行

### 3.2 执行计划分析示例

以MySQL为例，使用EXPLAIN分析子查询的执行计划：

```sql
EXPLAIN
SELECT * FROM employees
WHERE department_id IN (SELECT department_id FROM departments WHERE location = 'Headquarters');
```

**优化前的执行计划（简化）：**
```
+----+-------------+------------+------+---------------+------+---------+------+------+-------------+
| id | select_type | table      | type | possible_keys | key  | key_len | ref  | rows | Extra       |
+----+-------------+------------+------+---------------+------+---------+------+------+-------------+
|  1 | PRIMARY     | employees  | ALL  | NULL          | NULL | NULL    | NULL | 1000 | Using where |
|  2 | SUBQUERY    | departments| ref  | location_idx  | location_idx | 8 | const | 10   | Using where |
+----+-------------+------------+------+---------------+------+---------+------+------+-------------+
```

**优化后的执行计划（转换为连接后）：**
```
+----+-------------+------------+------+---------------+------+---------+------+------+-------------+
| id | select_type | table      | type | possible_keys | key  | key_len | ref  | rows | Extra       |
+----+-------------+------------+------+---------------+------+---------+------+------+-------------+
|  1 | SIMPLE      | departments| ref  | location_idx  | location_idx | 8 | const | 10   | Using where |
|  1 | SIMPLE      | employees  | ref  | dept_id_idx   | dept_id_idx | 4 | dept.id | 100 | NULL        |
+----+-------------+------------+------+---------------+------+---------+------+------+-------------+
```

### 3.3 执行计划优化指标

评估子查询优化效果的关键指标：

- **扫描的行数**：优化后应显著减少
- **临时表使用**：优化后应减少或消除
- **执行时间**：优化后应显著缩短
- **内存使用**：优化后应减少

## 4. 不同数据库系统的子查询优化特点

### 4.1 MySQL子查询优化

- 5.6版本前子查询优化能力有限
- 5.6版本后引入了子查询物化
- 5.7版本后引入了半连接优化
- 8.0版本进一步增强了子查询优化能力

### 4.2 Oracle子查询优化

- 强大的子查询展开能力
- 支持子查询物化
- 支持子查询提升
- 支持复杂谓词下推

### 4.3 SQL Server子查询优化

- 支持子查询展开
- 支持相关子查询优化
- 支持子查询物化
- 查询提示可控制子查询优化策略

### 4.4 PostgreSQL子查询优化

- 强大的子查询展开能力
- 支持子查询物化
- 支持复杂表达式下推
- 支持通用表表达式（CTE）优化

## 5. 子查询优化最佳实践

### 5.1 子查询改写为连接

当可能时，手动将子查询改写为等价的连接操作：

```sql
-- 优化前
SELECT * FROM employees
WHERE department_id IN (SELECT department_id FROM departments WHERE location = 'Headquarters');

-- 优化后
SELECT e.*
FROM employees e
JOIN departments d ON e.department_id = d.department_id
WHERE d.location = 'Headquarters';
```

### 5.2 使用EXISTS代替IN

对于大型结果集，EXISTS通常比IN更高效：

```sql
-- 优化前
SELECT * FROM departments
WHERE department_id IN (SELECT department_id FROM employees WHERE salary > 100000);

-- 优化后
SELECT * FROM departments d
WHERE EXISTS (SELECT 1 FROM employees e WHERE e.department_id = d.department_id AND e.salary > 100000);
```

### 5.3 使用通用表表达式（CTE）

对于需要多次使用的子查询，使用CTE可以提高可读性和性能：

```sql
-- 优化前
SELECT d.department_name,
       (SELECT COUNT(*) FROM employees e WHERE e.department_id = d.department_id) AS emp_count,
       (SELECT AVG(salary) FROM employees e WHERE e.department_id = d.department_id) AS avg_salary
FROM departments d;

-- 优化后
WITH dept_stats AS (
  SELECT department_id, COUNT(*) AS emp_count, AVG(salary) AS avg_salary
  FROM employees
  GROUP BY department_id
)
SELECT d.department_name, ds.emp_count, ds.avg_salary
FROM departments d
LEFT JOIN dept_stats ds ON d.department_id = ds.department_id;
```

### 5.4 避免相关子查询

尽可能避免使用相关子查询，特别是在大型表上：

```sql
-- 优化前（相关子查询）
SELECT e.employee_name,
       (SELECT department_name FROM departments d WHERE d.department_id = e.department_id) AS dept_name
FROM employees e;

-- 优化后（连接）
SELECT e.employee_name, d.department_name AS dept_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.department_id;
```

### 5.5 合理使用索引

确保子查询中的连接列和过滤条件列有适当的索引：

```sql
-- 为子查询中的连接列和过滤条件列创建索引
CREATE INDEX idx_dept_location ON departments(location);
CREATE INDEX idx_emp_dept_id ON employees(department_id);
CREATE INDEX idx_emp_salary ON employees(salary);
```

## 6. 案例分析：子查询优化实战

### 6.1 案例1：复杂报表查询优化

**优化前：**
```sql
SELECT d.department_name,
       (SELECT COUNT(*) FROM employees e WHERE e.department_id = d.department_id) AS emp_count,
       (SELECT AVG(salary) FROM employees e WHERE e.department_id = d.department_id) AS avg_salary,
       (SELECT MAX(salary) FROM employees e WHERE e.department_id = d.department_id) AS max_salary,
       (SELECT MIN(salary) FROM employees e WHERE e.department_id = d.department_id) AS min_salary
FROM departments d
WHERE d.department_id IN (SELECT department_id FROM employees GROUP BY department_id HAVING COUNT(*) > 10);
```

**优化后：**
```sql
WITH dept_stats AS (
  SELECT department_id,
         COUNT(*) AS emp_count,
         AVG(salary) AS avg_salary,
         MAX(salary) AS max_salary,
         MIN(salary) AS min_salary
  FROM employees
  GROUP BY department_id
  HAVING COUNT(*) > 10
)
SELECT d.department_name, ds.emp_count, ds.avg_salary, ds.max_salary, ds.min_salary
FROM departments d
JOIN dept_stats ds ON d.department_id = ds.department_id;
```

### 6.2 案例2：多层嵌套子查询优化

**优化前：**
```sql
SELECT * FROM employees
WHERE department_id IN (
  SELECT department_id FROM departments
  WHERE location_id IN (
    SELECT location_id FROM locations
    WHERE country_id = 'US'
  )
);
```

**优化后：**
```sql
SELECT e.*
FROM employees e
JOIN departments d ON e.department_id = d.department_id
JOIN locations l ON d.location_id = l.location_id
WHERE l.country_id = 'US';
```

### 6.3 案例3：NOT EXISTS子查询优化

**优化前：**
```sql
SELECT * FROM departments d
WHERE NOT EXISTS (
  SELECT 1 FROM employees e
  WHERE e.department_id = d.department_id
);
```

**优化后：**
```sql
SELECT d.*
FROM departments d
LEFT JOIN employees e ON d.department_id = e.department_id
WHERE e.employee_id IS NULL;
```

## 总结

子查询是SQL中强大而灵活的特性，但不当使用可能导致性能问题。通过了解子查询的分类和优化技术，可以显著提高查询性能：

1. **子查询展开**：将子查询转换为等价的连接操作
2. **子查询合并**：将子查询与主查询合并为单一查询块
3. **子查询物化**：预先计算并缓存子查询结果
4. **谓词下推**：将外层查询的条件下推到子查询
5. **子查询提升**：将FROM子句中的子查询提升到与主查询同级

在实际应用中，应根据具体场景选择合适的优化策略，并通过执行计划分析验证优化效果。现代数据库系统的优化器通常能自动应用这些优化技术，但了解这些技术的原理和适用场景，有助于编写更高效的查询语句和进行手动优化。