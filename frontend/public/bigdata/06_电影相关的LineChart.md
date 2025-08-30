折线图最适合展示随时间变化的趋势数据：

### 1. **月度电影新增趋势**
```sql
-- 每月新增电影数量变化趋势
SELECT 
    DATE_FORMAT(created_at, '%Y-%m') AS month,
    COUNT(*) AS new_movies
FROM movies
GROUP BY DATE_FORMAT(created_at, '%Y-%m')
ORDER BY month;
```
**折线图说明**：X轴=月份，Y轴=新增电影数量，展示电影库的月度增长趋势

---

### 2. **每日用户评分活动趋势**
```sql
-- 每日用户评分数量变化
SELECT 
    DATE(created_at) AS rating_date,
    COUNT(*) AS rating_count
FROM ratings
GROUP BY DATE(created_at)
ORDER BY rating_date;
```
**折线图说明**：X轴=日期，Y轴=评分数量，展示用户活跃度的日变化趋势

---

### 3. **年度电影平均评分变化**
```sql
-- 每年上映电影的平均豆瓣评分趋势
SELECT 
    year,
    AVG(CAST(douban_score AS DECIMAL(3,1))) AS avg_score,
    COUNT(*) AS movie_count
FROM movies
WHERE douban_score IS NOT NULL AND douban_score != ''
GROUP BY year
HAVING movie_count > 10  -- 确保有足够样本
ORDER BY year;
```
**折线图说明**：X轴=上映年份，Y轴=平均豆瓣评分，展示电影质量的年度变化

---

### 4. **演员出生年代分布**
```sql
-- 演员出生年代分布趋势
SELECT 
    FLOOR(YEAR(birth)/10)*10 AS birth_decade,
    COUNT(*) AS actor_count
FROM actors
WHERE birth IS NOT NULL
GROUP BY FLOOR(YEAR(birth)/10)*10
ORDER BY birth_decade;
```
**折线图说明**：X轴=出生年代（如1980、1990），Y轴=演员数量，展示演员年龄分布

---

### 5. **评论活动时间趋势**
```sql
-- 按小时统计的评论活动分布
SELECT 
    HOUR(comment_date) AS hour_of_day,
    COUNT(*) AS comment_count
FROM comments
GROUP BY HOUR(comment_date)
ORDER BY hour_of_day;
```
**折线图说明**：X轴=一天中的小时（0-23），Y轴=评论数量，展示用户评论的时段分布

---

### 6. **电影时长分布趋势**
```sql
-- 电影时长分布趋势（按10分钟分段）
SELECT 
    FLOOR(CAST(mins AS UNSIGNED)/10)*10 AS duration_bin,
    COUNT(*) AS movie_count
FROM movies
WHERE mins REGEXP '^[0-9]+$'  -- 确保是数字
GROUP BY FLOOR(CAST(mins AS UNSIGNED)/10)*10
ORDER BY duration_bin;
```
**折线图说明**：X轴=时长分段（如90-99分钟），Y轴=电影数量，展示电影时长分布

---

### 7. **用户评分时间序列**
```sql
-- 用户评分数量按周变化趋势
SELECT 
    DATE_FORMAT(created_at, '%Y-%u') AS week,
    COUNT(*) AS rating_count
FROM ratings
GROUP BY DATE_FORMAT(created_at, '%Y-%u')
ORDER BY week;
```
**折线图说明**：X轴=周数（如2023-01），Y轴=评分数量，展示每周评分活动趋势

### 折线图使用建议：
1. **时间序列**：优先选择1、2、5、7这类严格按时间排序的数据
2. **趋势分析**：适合展示3（质量变化）、7（用户行为变化）这类趋势性指标
3. **数据密度**：当数据点超过50个时，考虑使用滚动平均值平滑曲线
4. **多线对比**：可添加多个指标线（如同时展示新增电影和新增评论）
5. **异常检测**：折线图能清晰展示数据异常点（如某天评分量突增）
