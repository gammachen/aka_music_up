SQL技能、逻辑思维能力和问题解决能力。以下是一些核心且有难度的问题，涵盖了查询优化、复杂查询、数据建模和性能分析等方面：

---

### **一、SQL**

#### **1. 复杂查询与子查询**
- **问题**：
  ```sql
  -- 找出每个部门中工资最高的员工，并列出他们的姓名、部门和工资。
  SELECT e.name, e.department, e.salary
  FROM employees e
  WHERE e.salary = (
      SELECT MAX(salary)
      FROM employees
      WHERE department = e.department
  );
  ```
- **扩展**：
  - 使用 `JOIN` 替代子查询。
  - 处理多个员工具有相同最高工资的情况。

#### **2. 使用窗口函数**
- **问题**：
  ```sql
  -- 找出每个部门中工资排名前三的员工。
  SELECT name, department, salary, rank
  FROM (
      SELECT name, department, salary,
             DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rank
      FROM employees
  ) ranked
  WHERE rank <= 3;
  ```
- **扩展**：
  - 使用 `ROW_NUMBER()` 和 `RANK()` 的区别。
  - 处理并列排名的情况。

#### **3. 数据建模与规范化**
- **问题**：
  - 设计一个数据库模式来存储学生、课程和选课信息。
- **扩展**：
  - 解释范式（1NF、2NF、3NF）及其应用。
  - 处理多对多关系（如学生与课程）。

#### **4. 性能优化**
- **问题**：
  ```sql
  -- 查询每个客户的订单总数和总金额。
  SELECT c.customer_id, c.customer_name, COUNT(o.order_id) AS total_orders, SUM(o.amount) AS total_amount
  FROM customers c
  JOIN orders o ON c.customer_id = o.customer_id
  GROUP BY c.customer_id, c.customer_name;
  ```
- **扩展**：
  - 分析查询计划（使用 `EXPLAIN`）。
  - 优化索引（如在 `customer_id` 和 `order_id` 上创建索引）。
  - 处理大数据集的性能问题。

#### **5. 聚合与分组**
- **问题**：
  ```sql
  -- 找出每个季度销售额最高的产品。
  SELECT q.quarter, p.product_name, MAX(s.sales_amount) AS max_sales
  FROM sales s
  JOIN products p ON s.product_id = p.product_id
  JOIN quarters q ON s.sale_date BETWEEN q.start_date AND q.end_date
  GROUP BY q.quarter, p.product_name;
  ```
- **扩展**：
  - 使用 `GROUP BY` 和 `HAVING` 的区别。
  - 处理多个产品具有相同最高销售额的情况。

#### **6. 连接（Joins）**
- **问题**：
  ```sql
  -- 找出所有没有订单的客户。
  SELECT c.customer_id, c.customer_name
  FROM customers c
  LEFT JOIN orders o ON c.customer_id = o.customer_id
  WHERE o.order_id IS NULL;
  ```
- **扩展**：
  - 解释不同类型的连接（`INNER JOIN`, `LEFT JOIN`, `RIGHT JOIN`, `FULL OUTER JOIN`）。
  - 处理多个连接条件和复杂连接逻辑。

#### **7. 递归查询（CTE）**
- **问题**：
  ```sql
  -- 找出所有员工及其直接和间接上级。
  WITH RECURSIVE employee_hierarchy AS (
      SELECT employee_id, name, manager_id
      FROM employees
      WHERE employee_id = 1
      UNION ALL
      SELECT e.employee_id, e.name, e.manager_id
      FROM employees e
      INNER JOIN employee_hierarchy eh ON e.manager_id = eh.employee_id
  )
  SELECT * FROM employee_hierarchy;
  ```
- **扩展**：
  - 解释递归CTE的工作原理。
  - 处理递归深度限制和性能优化。

#### **8. 时间序列分析**
- **问题**：
  ```sql
  -- 找出每个客户的月度订单总数。
  SELECT c.customer_id, c.customer_name, DATE_FORMAT(o.sale_date, '%Y-%m') AS month, COUNT(o.order_id) AS total_orders
  FROM customers c
  JOIN orders o ON c.customer_id = o.customer_id
  GROUP BY c.customer_id, c.customer_name, DATE_FORMAT(o.sale_date, '%Y-%m')
  ORDER BY c.customer_id, month;
  ```
- **扩展**：
  - 使用 `DATE_TRUNC` 或 `EXTRACT` 进行时间序列分析。
  - 处理缺失月份的填充。

#### **9. 数据清洗与去重**
- **问题**：
  ```sql
  -- 删除重复的订单记录，保留最新的一条。
  DELETE o1
  FROM orders o1
  JOIN orders o2 ON o1.customer_id = o2.customer_id AND o1.product_id = o2.product_id
  WHERE o1.order_id > o2.order_id;
  ```
- **扩展**：
  - 使用 `ROW_NUMBER()` 进行去重。
  - 处理复杂的数据清洗逻辑。

#### **10. 动态SQL与存储过程**
- **问题**：
  - 编写一个存储过程，根据传入的部门ID返回该部门的所有员工及其工资。
- **扩展**：
  - 使用动态SQL生成查询。
  - 处理存储过程中的异常和错误处理。

#### **11. 数据完整性与约束**
- **问题**：
  - 设计一个数据库模式来存储学生、课程和选课信息，并确保数据完整性。
- **扩展**：
  - 使用外键约束（Foreign Keys）。
  - 处理唯一性约束（Unique Constraints）。
  - 使用触发器（Triggers）进行数据验证。

#### **12. 分页查询**
- **问题**：
  ```sql
  -- 实现分页查询，每页10条记录，查询第2页的数据。
  SELECT *
  FROM (
      SELECT *, ROW_NUMBER() OVER (ORDER BY order_date DESC) AS row_num
      FROM orders
  ) AS numbered_orders
  WHERE row_num BETWEEN 11 AND 20;
  ```
- **扩展**：
  - 使用 `LIMIT` 和 `OFFSET` 进行分页。
  - 处理大数据集的分页性能。

#### **13. 临时表与表变量**
- **问题**：
  - 使用临时表或表变量存储中间结果，并编写查询。
- **扩展**：
  - 解释临时表与表变量的区别。
  - 处理大数据集的临时表性能优化。

#### **14. 数据透视（Pivot）**
- **问题**：
  ```sql
  -- 将销售数据按产品和季度进行透视。
  SELECT product_id, 
         SUM(CASE WHEN quarter = 'Q1' THEN sales_amount ELSE 0 END) AS Q1_Sales,
         SUM(CASE WHEN quarter = 'Q2' THEN sales_amount ELSE 0 END) AS Q2_Sales,
         SUM(CASE WHEN quarter = 'Q3' THEN sales_amount ELSE 0 END) AS Q3_Sales,
         SUM(CASE WHEN quarter = 'Q4' THEN sales_amount ELSE 0 END) AS Q4_Sales
  FROM sales
  GROUP BY product_id;
  ```
- **扩展**：
  - 使用 `PIVOT` 操作符（如SQL Server）。
  - 处理动态列和复杂透视逻辑。

#### **15. 数据迁移与转换**
- **问题**：
  - 将一个旧表的数据迁移到新表，并处理数据转换逻辑。
- **扩展**：
  - 使用 `INSERT INTO ... SELECT` 进行数据迁移。
  - 处理数据类型转换和数据验证。

---

### **二、示例问题详解**

#### **1. 复杂查询与子查询**
- **问题**：
  ```sql
  -- 找出每个部门中工资最高的员工，并列出他们的姓名、部门和工资。
  SELECT e.name, e.department, e.salary
  FROM employees e
  WHERE e.salary = (
      SELECT MAX(salary)
      FROM employees
      WHERE department = e.department
  );
  ```
- **扩展**：
  - **使用 `JOIN` 替代子查询**：
    ```sql
    SELECT e.name, e.department, e.salary
    FROM employees e
    JOIN (
        SELECT department, MAX(salary) AS max_salary
        FROM employees
        GROUP BY department
    ) dept_max ON e.department = dept_max.department AND e.salary = dept_max.max_salary;
    ```
  - **处理多个员工具有相同最高工资的情况**：
    ```sql
    SELECT e.name, e.department, e.salary
    FROM employees e
    JOIN (
        SELECT department, MAX(salary) AS max_salary
        FROM employees
        GROUP BY department
    ) dept_max ON e.department = dept_max.department AND e.salary = dept_max.max_salary;
    ```

#### **2. 使用窗口函数**
- **问题**：
  ```sql
  -- 找出每个部门中工资排名前三的员工。
  SELECT name, department, salary, rank
  FROM (
      SELECT name, department, salary,
             DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rank
      FROM employees
  ) ranked
  WHERE rank <= 3;
  ```
- **扩展**：
  - **使用 `ROW_NUMBER()` 和 `RANK()` 的区别**：
    - `ROW_NUMBER()`：为每个分区内的行分配唯一的行号。
    - `RANK()`：为每个分区内的行分配排名，允许并列排名。
    - `DENSE_RANK()`：为每个分区内的行分配排名，不允许并列排名。
  - **处理并列排名的情况**：
    ```sql
    SELECT name, department, salary, rank
    FROM (
        SELECT name, department, salary,
               RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rank
        FROM employees
    ) ranked
    WHERE rank <= 3;
    ```

#### **3. 数据建模与规范化**
- **问题**：
  - 设计一个数据库模式来存储学生、课程和选课信息。
- **扩展**：
  - **范式（1NF、2NF、3NF）及其应用**：
    - **1NF**：确保表中的每一列都是原子的。
    - **2NF**：确保表中的每一列都完全依赖于主键。
    - **3NF**：确保表中的每一列都只依赖于主键，不依赖于其他非主键列。
  - **处理多对多关系（如学生与课程）**：
    ```sql
    CREATE TABLE students (
        student_id INT PRIMARY KEY,
        student_name VARCHAR(100)
    );

    CREATE TABLE courses (
        course_id INT PRIMARY KEY,
        course_name VARCHAR(100)
    );

    CREATE TABLE enrollments (
        enrollment_id INT PRIMARY KEY,
        student_id INT,
        course_id INT,
        enrollment_date DATE,
        FOREIGN KEY (student_id) REFERENCES students(student_id),
        FOREIGN KEY (course_id) REFERENCES courses(course_id)
    );
    ```

#### **4. 性能优化**
- **问题**：
  ```sql
  -- 查询每个客户的订单总数和总金额。
  SELECT c.customer_id, c.customer_name, COUNT(o.order_id) AS total_orders, SUM(o.amount) AS total_amount
  FROM customers c
  JOIN orders o ON c.customer_id = o.customer_id
  GROUP BY c.customer_id, c.customer_name;
  ```
- **扩展**：
  - **分析查询计划（使用 `EXPLAIN`）**：
    ```sql
    EXPLAIN SELECT c.customer_id, c.customer_name, COUNT(o.order_id) AS total_orders, SUM(o.amount) AS total_amount
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, c.customer_name;
    ```
  - **优化索引**：
    ```sql
    CREATE INDEX idx_customer_id ON orders(customer_id);
    CREATE INDEX idx_customer_name ON customers(customer_name);
    ```
  - **处理大数据集的性能问题**：
    - 使用分区表（Partitioning）。
    - 使用物化视图（Materialized Views）。

#### **5. 聚合与分组**
- **问题**：
  ```sql
  -- 找出每个季度销售额最高的产品。
  SELECT q.quarter, p.product_name, MAX(s.sales_amount) AS max_sales
  FROM sales s
  JOIN products p ON s.product_id = p.product_id
  JOIN quarters q ON s.sale_date BETWEEN q.start_date AND q.end_date
  GROUP BY q.quarter, p.product_name;
  ```
- **扩展**：
  - **使用 `GROUP BY` 和 `HAVING` 的区别**：
    - `GROUP BY`：用于分组。
    - `HAVING`：用于过滤分组后的结果。
  - **处理多个产品具有相同最高销售额的情况**：
    ```sql
    WITH ranked_sales AS (
        SELECT p.product_name, q.quarter, s.sales_amount,
               RANK() OVER (PARTITION BY q.quarter ORDER BY s.sales_amount DESC) AS rank
        FROM sales s
        JOIN products p ON s.product_id = p.product_id
        JOIN quarters q ON s.sale_date BETWEEN q.start_date AND q.end_date
    )
    SELECT product_name, quarter, sales_amount
    FROM ranked_sales
    WHERE rank = 1;
    ```

#### **6. 连接（Joins）**
- **问题**：
  ```sql
  -- 找出所有没有订单的客户。
  SELECT c.customer_id, c.customer_name
  FROM customers c
  LEFT JOIN orders o ON c.customer_id = o.customer_id
  WHERE o.order_id IS NULL;
  ```
- **扩展**：
  - **解释不同类型的连接**：
    - `INNER JOIN`：返回两个表中匹配的行。
    - `LEFT JOIN`：返回左表中的所有行，以及右表中匹配的行。
    - `RIGHT JOIN`：返回右表中的所有行，以及左表中匹配的行。
    - `FULL OUTER JOIN`：返回两个表中的所有行，不匹配的行用 `NULL` 填充。
  - **处理多个连接条件和复杂连接逻辑**：
    ```sql
    SELECT c.customer_id, c.customer_name, o.order_id, o.amount
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    LEFT JOIN order_details od ON o.order_id = od.order_id
    WHERE o.order_id IS NULL OR od.product_id IS NULL;
    ```

#### **7. 递归查询（CTE）**
- **问题**：
  ```sql
  -- 找出所有员工及其直接和间接上级。
  WITH RECURSIVE employee_hierarchy AS (
      SELECT employee_id, name, manager_id
      FROM employees
      WHERE employee_id = 1
      UNION ALL
      SELECT e.employee_id, e.name, e.manager_id
      FROM employees e
      INNER JOIN employee_hierarchy eh ON e.manager_id = eh.employee_id
  )
  SELECT * FROM employee_hierarchy;
  ```
- **扩展**：
  - **解释递归CTE的工作原理**：
    - **锚定部分**：初始查询，返回起始行。
    - **递归部分**：基于锚定部分的结果，递归地查询更多行。
  - **处理递归深度限制和性能优化**：
    - 使用 `MAXRECURSION` 限制递归深度（如SQL Server）。
    - 优化递归查询中的连接条件和索引。

#### **8. 时间序列分析**
- **问题**：
  ```sql
  -- 找出每个客户的月度订单总数。
  SELECT c.customer_id, c.customer_name, DATE_FORMAT(o.sale_date, '%Y-%m') AS month, COUNT(o.order_id) AS total_orders
  FROM customers c
  JOIN orders o ON c.customer_id = o.customer_id
  GROUP BY c.customer_id, c.customer_name, DATE_FORMAT(o.sale_date, '%Y-%m')
  ORDER BY c.customer_id, month;
  ```
- **扩展**：
  - **使用 `DATE_TRUNC` 或 `EXTRACT` 进行时间序列分析**：
    ```sql
    -- PostgreSQL 示例
    SELECT c.customer_id, c.customer_name, DATE_TRUNC('month', o.sale_date) AS month, COUNT(o.order_id) AS total_orders
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, c.customer_name, DATE_TRUNC('month', o.sale_date)
    ORDER BY c.customer_id, month;
    ```
  - **处理缺失月份的填充**：
    ```sql
    -- 使用生成月份序列并左连接
    WITH RECURSIVE months AS (
        SELECT DATE_TRUNC('month', CURRENT_DATE) AS month
        UNION ALL
        SELECT month - INTERVAL '1 month'
        FROM months
        WHERE month > DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '12 months'
    )
    SELECT m.month, c.customer_id, c.customer_name, COUNT(o.order_id) AS total_orders
    FROM months m
    LEFT JOIN customers c ON TRUE
    LEFT JOIN orders o ON c.customer_id = o.customer_id AND DATE_TRUNC('month', o.sale_date) = m.month
    GROUP BY m.month, c.customer_id, c.customer_name
    ORDER BY m.month, c.customer_id;
    ```

#### **9. 数据清洗与去重**
- **问题**：
  ```sql
  -- 删除重复的订单记录，保留最新的一条。
  DELETE o1
  FROM orders o1
  JOIN orders o2 ON o1.customer_id = o2.customer_id AND o1.product_id = o2.product_id
  WHERE o1.order_id > o2.order_id;
  ```
- **扩展**：
  - **使用 `ROW_NUMBER()` 进行去重**：
    ```sql
    WITH ranked_orders AS (
        SELECT order_id, customer_id, product_id, order_date,
               ROW_NUMBER() OVER (PARTITION BY customer_id, product_id