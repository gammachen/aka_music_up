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