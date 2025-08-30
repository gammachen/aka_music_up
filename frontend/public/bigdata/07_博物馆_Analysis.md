下面是为博物馆企业内部数据大屏设计的完整MySQL Schema及模拟数据脚本。该设计覆盖了之前提到的所有核心业务指标，包括观众服务、展览运营、设备管理、安全保障、人力资源、商业运营和数字业务等维度。

### 博物馆数据大屏 Schema 设计

```sql
-- 创建数据库
CREATE DATABASE IF NOT EXISTS museum_management;
USE museum_management;

-- 1. 观众服务模块
CREATE TABLE visitors (
    visitor_id INT AUTO_INCREMENT PRIMARY KEY,
    ticket_id VARCHAR(20) UNIQUE,
    entry_time DATETIME NOT NULL,
    exit_time DATETIME,
    visit_duration INT COMMENT '停留分钟数',
    reservation_channel ENUM('官网', '小程序', '第三方平台') NOT NULL,
    group_type ENUM('个人', '团体', '学生', 'VIP') DEFAULT '个人',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE reservations (
    reservation_id INT AUTO_INCREMENT PRIMARY KEY,
    visitor_id INT,
    reservation_date DATE NOT NULL,
    time_slot VARCHAR(20) NOT NULL COMMENT '如09:00-11:00',
    num_visitors INT NOT NULL,
    status ENUM('预约中', '已到馆', '已取消', '爽约') DEFAULT '预约中',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (visitor_id) REFERENCES visitors(visitor_id)
);

-- 2. 展览运营模块
CREATE TABLE exhibitions (
    exhibition_id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    type ENUM('常设展', '特展', '巡回展', '数字展') NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    location VARCHAR(100) NOT NULL,
    area DECIMAL(8,2) COMMENT '展览面积(m²)',
    investment DECIMAL(12,2) COMMENT '投资金额',
    visitor_capacity INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE exhibits (
    exhibit_id INT AUTO_INCREMENT PRIMARY KEY,
    exhibition_id INT,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    era VARCHAR(50) COMMENT '年代',
    grade ENUM('一级', '二级', '三级', '一般') NOT NULL,
    status ENUM('在展', '在库', '修复中', '外借'),
    current_location VARCHAR(100),
    last_check_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (exhibition_id) REFERENCES exhibitions(exhibition_id)
);

-- 3. 设备设施模块
CREATE TABLE facilities (
    facility_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type ENUM('安防', '多媒体', '电梯', '照明', '空调', '消防') NOT NULL,
    location VARCHAR(100) NOT NULL,
    install_date DATE NOT NULL,
    status ENUM('正常', '故障', '维护中', '停用') DEFAULT '正常',
    last_maintenance_date DATE,
    next_maintenance_date DATE
);

CREATE TABLE maintenance_records (
    record_id INT AUTO_INCREMENT PRIMARY KEY,
    facility_id INT NOT NULL,
    maintenance_date DATETIME NOT NULL,
    maintenance_type ENUM('日常维护', '故障维修', '预防性维护') NOT NULL,
    description TEXT,
    downtime_hours DECIMAL(5,2) COMMENT '停机时间',
    cost DECIMAL(10,2),
    technician VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (facility_id) REFERENCES facilities(facility_id)
);

-- 4. 安全保障模块
CREATE TABLE security_events (
    event_id INT AUTO_INCREMENT PRIMARY KEY,
    event_time DATETIME NOT NULL,
    location VARCHAR(100) NOT NULL,
    event_type ENUM('设备故障', '人员冲突', '文物异常', '消防警报', '其他') NOT NULL,
    severity ENUM('低', '中', '高', '紧急') NOT NULL,
    description TEXT,
    handler VARCHAR(100),
    status ENUM('待处理', '处理中', '已解决') DEFAULT '待处理',
    resolve_time DATETIME,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE environment_monitoring (
    monitor_id INT AUTO_INCREMENT PRIMARY KEY,
    exhibit_id INT,
    monitor_time DATETIME NOT NULL,
    temperature DECIMAL(5,2) COMMENT '温度(℃)',
    humidity DECIMAL(5,2) COMMENT '湿度(%)',
    lux DECIMAL(7,2) COMMENT '光照强度(lx)',
    uv_index DECIMAL(4,2) COMMENT '紫外线指数',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (exhibit_id) REFERENCES exhibits(exhibit_id)
);

-- 5. 人力资源模块
CREATE TABLE staff (
    staff_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    position ENUM('讲解员', '安保', '保洁', '管理员', '技术员', '策展人') NOT NULL,
    hire_date DATE NOT NULL,
    phone VARCHAR(20),
    status ENUM('在职', '休假', '离职') DEFAULT '在职',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE staff_schedules (
    schedule_id INT AUTO_INCREMENT PRIMARY KEY,
    staff_id INT NOT NULL,
    work_date DATE NOT NULL,
    shift ENUM('早班', '中班', '晚班') NOT NULL,
    area VARCHAR(100) NOT NULL,
    checkin_time DATETIME,
    checkout_time DATETIME,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (staff_id) REFERENCES staff(staff_id)
);

-- 6. 商业运营模块
CREATE TABLE retail_sales (
    sale_id INT AUTO_INCREMENT PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    sale_time DATETIME NOT NULL,
    quantity INT NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    channel ENUM('线下', '线上') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE inventory (
    item_id INT AUTO_INCREMENT PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    current_stock INT NOT NULL,
    min_stock INT NOT NULL,
    last_restock_date DATE,
    turnover_days INT COMMENT '周转天数',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. 数字业务模块
CREATE TABLE digital_activities (
    activity_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type ENUM('云展览', '数字藏品', '在线预约', '互动活动') NOT NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    participants INT DEFAULT 0,
    page_views INT DEFAULT 0,
    avg_duration DECIMAL(6,2) COMMENT '平均停留分钟数',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE device_usage (
    usage_id INT AUTO_INCREMENT PRIMARY KEY,
    device_type ENUM('导览机', 'AR设备', 'VR设备', '互动屏') NOT NULL,
    usage_date DATE NOT NULL,
    usage_count INT DEFAULT 0,
    avg_duration DECIMAL(6,2) COMMENT '平均使用分钟数',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 模拟数据初始化脚本

```sql
USE museum_management;

-- 插入观众数据
INSERT INTO visitors (ticket_id, entry_time, exit_time, visit_duration, reservation_channel, group_type) VALUES
('T20231001001', '2023-10-01 09:15:32', '2023-10-01 12:30:45', 195, '官网', '个人'),
('T20231001002', '2023-10-01 10:05:21', '2023-10-01 13:45:18', 220, '小程序', '团体'),
('T20231001003', '2023-10-01 11:30:15', '2023-10-01 14:20:30', 170, '第三方平台', '学生'),
('T20231002001', '2023-10-02 09:45:00', '2023-10-02 11:30:00', 105, '官网', '个人'),
('T20231002002', '2023-10-02 13:20:00', NULL, NULL, '小程序', 'VIP');

-- 插入预约数据
INSERT INTO reservations (visitor_id, reservation_date, time_slot, num_visitors, status) VALUES
(1, '2023-10-01', '09:00-11:00', 1, '已到馆'),
(2, '2023-10-01', '10:00-12:00', 15, '已到馆'),
(3, '2023-10-01', '11:00-13:00', 30, '已到馆'),
(4, '2023-10-02', '09:00-11:00', 1, '已到馆'),
(5, '2023-10-02', '13:00-15:00', 2, '已到馆'),
(NULL, '2023-10-03', '10:00-12:00', 20, '预约中'),
(NULL, '2023-10-04', '14:00-16:00', 5, '预约中');

-- 插入展览数据
INSERT INTO exhibitions (title, type, start_date, end_date, location, area, investment, visitor_capacity) VALUES
('青铜时代', '常设展', '2023-01-01', '2025-12-31', '一层东厅', 500.00, 500000.00, 100),
('敦煌艺术', '特展', '2023-09-01', '2023-12-31', '二层特展厅', 800.00, 1200000.00, 150),
('数字故宫', '数字展', '2023-07-15', '2024-07-14', '数字展厅', 300.00, 800000.00, 80);

-- 插入展品数据
INSERT INTO exhibits (exhibition_id, name, category, era, grade, status, current_location, last_check_date) VALUES
(1, '司母戊鼎', '青铜器', '商代', '一级', '在展', '一层东厅-展柜A01', '2023-09-28'),
(1, '四羊方尊', '青铜器', '商代', '一级', '在展', '一层东厅-展柜A02', '2023-09-28'),
(2, '敦煌壁画-飞天', '壁画', '唐代', '一级', '在展', '二层特展厅-展墙B01', '2023-09-30'),
(2, '千手观音像', '雕塑', '宋代', '二级', '修复中', '修复室', '2023-09-25'),
(3, '数字清明上河图', '数字展品', '现代', '一般', '在展', '数字展厅-互动墙', '2023-09-20');

-- 插入设备数据
INSERT INTO facilities (name, type, location, install_date, status, last_maintenance_date, next_maintenance_date) VALUES
('摄像头-东厅01', '安防', '一层东厅入口', '2020-05-10', '正常', '2023-09-15', '2023-12-15'),
('触摸屏-数字厅01', '多媒体', '数字展厅入口', '2022-03-20', '维护中', '2023-10-01', '2024-01-01'),
('电梯-主楼01', '电梯', '主楼中庭', '2019-11-05', '正常', '2023-09-20', '2023-12-20'),
('空调-特展厅01', '空调', '二层特展厅', '2021-07-12', '故障', '2023-09-25', '2023-12-25'),
('消防喷淋-西厅01', '消防', '一层西厅', '2020-02-18', '正常', '2023-09-18', '2023-12-18');

-- 插入维护记录
INSERT INTO maintenance_records (facility_id, maintenance_date, maintenance_type, description, downtime_hours, cost, technician) VALUES
(2, '2023-10-01 10:30:00', '故障维修', '触摸屏响应失灵', 3.5, 1200.00, '张工'),
(4, '2023-09-25 14:00:00', '日常维护', '空调滤网更换', 1.0, 300.00, '李工'),
(1, '2023-09-15 09:00:00', '预防性维护', '摄像头清洁校准', 0.5, 500.00, '王工');

-- 插入安全事件
INSERT INTO security_events (event_time, location, event_type, severity, description, handler, status, resolve_time) VALUES
('2023-10-01 11:20:00', '一层东厅', '人员冲突', '中', '游客争执', '安保张三', '已解决', '2023-10-01 11:35:00'),
('2023-10-02 14:45:00', '二层特展厅', '消防警报', '紧急', '烟雾探测器误报', '安保李四', '已解决', '2023-10-02 15:00:00'),
('2023-09-28 10:15:00', '修复室', '文物异常', '高', '千手观音像湿度异常', '技术员王五', '处理中', NULL);

-- 插入环境监测数据
INSERT INTO environment_monitoring (exhibit_id, monitor_time, temperature, humidity, lux, uv_index) VALUES
(1, '2023-10-01 09:00:00', 20.5, 45.0, 150.0, 0.1),
(1, '2023-10-01 12:00:00', 21.2, 44.5, 180.0, 0.2),
(1, '2023-10-01 15:00:00', 21.0, 45.5, 160.0, 0.1),
(3, '2023-10-01 10:30:00', 20.8, 48.0, 120.0, 0.0);

-- 插入员工数据
INSERT INTO staff (name, position, hire_date, phone, status) VALUES
('张讲解', '讲解员', '2020-05-01', '13500135000', '在职'),
('李保安', '安保', '2019-11-15', '13600136000', '在职'),
('王保洁', '保洁', '2021-02-10', '13700137000', '休假'),
('赵技术', '技术员', '2022-03-20', '13800138000', '在职');

-- 插入排班数据
INSERT INTO staff_schedules (staff_id, work_date, shift, area, checkin_time, checkout_time) VALUES
(1, '2023-10-01', '早班', '一层东厅', '2023-10-01 08:45:00', '2023-10-01 12:00:00'),
(2, '2023-10-01', '中班', '二层巡逻', '2023-10-01 12:30:00', '2023-10-01 17:00:00'),
(4, '2023-10-01', '早班', '全馆设备', '2023-10-01 08:50:00', '2023-10-01 17:30:00');

-- 插入销售数据
INSERT INTO retail_sales (product_name, category, sale_time, quantity, amount, channel) VALUES
('青铜器书签', '文具', '2023-10-01 10:30:00', 5, 100.00, '线下'),
('千里江山图丝巾', '服饰', '2023-10-01 15:20:00', 2, 360.00, '线下'),
('青花瓷U盘', '数码', '2023-10-01 11:15:00', 3, 240.00, '线上'),
('敦煌明信片', '文具', '2023-10-02 09:45:00', 10, 150.00, '线下');

-- 插入库存数据
INSERT INTO inventory (product_name, category, current_stock, min_stock, last_restock_date, turnover_days) VALUES
('青铜器书签', '文具', 150, 50, '2023-09-25', 30),
('千里江山图丝巾', '服饰', 80, 20, '2023-09-20', 45),
('青花瓷U盘', '数码', 200, 100, '2023-09-28', 60),
('敦煌明信片', '文具', 300, 100, '2023-09-15', 25);

-- 插入数字活动数据
INSERT INTO digital_activities (name, type, start_time, end_time, participants, page_views, avg_duration) VALUES
('青铜时代云展览', '云展览', '2023-09-01 00:00:00', NULL, 25000, 120000, 8.5),
('数字藏品发售-敦煌', '数字藏品', '2023-10-01 10:00:00', '2023-10-01 12:00:00', 5000, 15000, 3.2);

-- 插入设备使用数据
INSERT INTO device_usage (device_type, usage_date, usage_count, avg_duration) VALUES
('导览机', '2023-10-01', 120, 45.5),
('AR设备', '2023-10-01', 85, 12.3),
('VR设备', '2023-10-01', 65, 8.7),
('互动屏', '2023-10-01', 230, 3.2);
```

### 大屏数据查询示例

#### 1. 实时在馆人数统计
```sql
SELECT COUNT(*) AS current_visitors
FROM visitors
WHERE exit_time IS NULL;
```

#### 2. 当日展览参观热度
```sql
SELECT 
    e.title AS exhibition,
    COUNT(v.visitor_id) AS visitors,
    ROUND(AVG(v.visit_duration), 0) AS avg_duration
FROM visitors v
JOIN reservations r ON v.visitor_id = r.visitor_id
JOIN exhibits ex ON v.entry_time BETWEEN ex.last_check_date AND DATE_ADD(ex.last_check_date, INTERVAL 1 DAY)
JOIN exhibitions e ON ex.exhibition_id = e.exhibition_id
WHERE DATE(v.entry_time) = CURDATE()
GROUP BY e.exhibition_id
ORDER BY visitors DESC;
```

#### 3. 设备状态监控
```sql
SELECT 
    type,
    SUM(CASE WHEN status = '正常' THEN 1 ELSE 0 END) AS normal,
    SUM(CASE WHEN status = '故障' THEN 1 ELSE 0 END) AS fault,
    SUM(CASE WHEN status = '维护中' THEN 1 ELSE 0 END) AS maintenance,
    COUNT(*) AS total
FROM facilities
GROUP BY type;
```

#### 4. 环境监控预警
```sql
SELECT 
    ex.name AS exhibit,
    em.temperature,
    em.humidity,
    em.lux,
    CASE 
        WHEN humidity < 40 OR humidity > 60 THEN '湿度异常'
        WHEN lux > 200 THEN '光照过强'
        ELSE '正常'
    END AS status
FROM environment_monitoring em
JOIN exhibits ex ON em.exhibit_id = ex.exhibit_id
WHERE em.monitor_time = (
    SELECT MAX(monitor_time) 
    FROM environment_monitoring 
    WHERE exhibit_id = ex.exhibit_id
);
```

#### 5. 文创销售实时数据
```sql
SELECT 
    product_name,
    SUM(quantity) AS total_quantity,
    SUM(amount) AS total_amount
FROM retail_sales
WHERE DATE(sale_time) = CURDATE()
GROUP BY product_name
ORDER BY total_amount DESC
LIMIT 5;
```

### 大屏实现建议

1. **实时数据更新**：使用MySQL事件调度器定期刷新数据
   ```sql
   CREATE EVENT update_dashboard_data
   ON SCHEDULE EVERY 1 MINUTE
   DO
     -- 更新逻辑
   ```

2. **数据聚合**：创建物化视图存储预聚合数据
3. **缓存机制**：对实时性要求不高的数据使用Redis缓存
4. **API接口**：通过RESTful API为前端大屏提供数据
5. **数据可视化**：使用ECharts、D3.js或专业BI工具实现可视化

博物馆运营的核心指标，支持实时监控、历史分析和趋势预测，能够全面支撑博物馆数据大屏的展示需求。