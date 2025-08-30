---

### **地理位置存储与查询优化文档**

---

#### **一、表结构设计与改造**

##### **1. 初始表结构**
```sql
CREATE TABLE locations (
    id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(30),
    lat FLOAT NOT NULL,  -- 纬度（范围：-90.0 到 90.0）
    lon FLOAT NOT NULL   -- 经度（范围：-180.0 到 180.0）
);
```
**问题**：  
- 直接使用 `FLOAT` 存储经纬度，虽然节省空间，但精度有限（建议使用 `DECIMAL(9,6)`）。  
- 复杂的地理距离计算（如 Haversine 公式）无法利用索引，导致全表扫描。

---

##### **2. 优化改造：添加分块字段与索引**
```sql
-- 新增分块字段（将经纬度离散化为整数区）
ALTER TABLE locations 
  ADD lat_floor INT NOT NULL DEFAULT 0,  -- 纬度区块（取 lat 的整数部分）
  ADD lon_floor INT NOT NULL DEFAULT 0,   -- 经度区块（取 lon 的整数部分）
  ADD KEY (lat_floor, lon_floor);         -- 复合索引加速区块筛选
```

**逻辑说明**：  
- **离散化分块**：将连续的经纬度划分为 1°×1° 的区块（通过 `floor(lat)` 和 `floor(lon)` 实现）。  
- **索引优化**：通过 `(lat_floor, lon_floor)` 索引快速定位目标区块，减少扫描数据量。

---

#### **二、数据预处理**
```sql
-- 填充分块字段（将经纬度转换为区块值）
UPDATE locations 
SET lat_floor = FLOOR(lat), 
    lon_floor = FLOOR(lon);
```

**示例数据**：  
| name                   | lat    | lon     | lat_floor | lon_floor |
|------------------------|--------|---------|-----------|-----------|
| Charlottesville, Virginia | 38.03 | -78.48  | 38        | -79       |
| Chicago, Illinois      | 42.85  | -87.65  | 42        | -88       |
| Washington, DC         | 38.89  | -77.04  | 38        | -78       |

---

#### **三、查询优化策略**

##### **1. 直接使用 Haversine 公式（未优化）**
```sql
SELECT * FROM locations 
WHERE 3979 * ACOS(
  COS(RADIANS(lat)) * COS(RADIANS(38.03)) * 
  COS(RADIANS(lon) - RADIANS(-78.48)) + 
  SIN(RADIANS(lat)) * SIN(RADIANS(38.03))
) <= 100;
```
**问题**：  
- 全表扫描，无法利用索引。  
- 复杂计算导致性能低下。

---

##### **2. 范围预筛选优化**
```sql
-- 计算经纬度范围（近似筛选）
SELECT * FROM locations
WHERE 
  lat BETWEEN 38.03 - DEGREES(0.0253) AND 38.03 + DEGREES(0.0253)
  AND lon BETWEEN -78.48 - DEGREES(0.0253) AND -78.48 + DEGREES(0.0253);
```
**公式解析**：  
- `DEGREES(0.0253) ≈ 1.45°`：假设 100 英里（160.9 km）对应的经纬度偏移量。  
- **目的**：快速筛选出目标点附近的候选区域，减少后续计算的规模。

---

##### **3. 分块索引优化**
```sql
-- 步骤1：计算分块范围
SELECT 
  FLOOR(38.03 - DEGREES(0.0253)) AS lat_lb,  -- 最小纬度区块
  CEILING(38.03 + DEGREES(0.0253)) AS lat_ub, -- 最大纬度区块
  FLOOR(-78.48 - DEGREES(0.0253)) AS lon_lb,  -- 最小经度区块
  CEILING(-78.48 + DEGREES(0.0253)) AS lon_ub; -- 最大经度区块

-- 示例结果：lat_lb=36, lat_ub=40, lon_lb=-80, lon_ub=-77

-- 步骤2：分块筛选 + 精确计算
SELECT * FROM locations 
WHERE 
  lat_floor IN (36,37,38,39,40) 
  AND lon_floor IN (-80,-79,-78,-77)  -- 利用索引快速筛选区块
  AND 3979 * ACOS(
    COS(RADIANS(lat)) * COS(RADIANS(38.03)) * 
    COS(RADIANS(lon) - RADIANS(-78.48)) + 
    SIN(RADIANS(lat)) * SIN(RADIANS(38.03))
  ) <= 100;  -- 精确计算距离
```

**优化逻辑**：  
1. **分块筛选**：  
   - 通过 `lat_floor` 和 `lon_floor` 索引快速定位到可能包含目标点的区块。  
   - 减少需要精确计算的数据量（例如从 100 万行减少到 1 万行）。  
2. **精确计算**：  
   - 对筛选后的数据应用 Haversine 公式，确保结果精确。

---

#### **四、公式解析与参数说明**

##### **1. Haversine 公式**
```sql
距离 = 地球半径 × ACOS(
  COS(RADIANS(lat)) * COS(RADIANS(target_lat)) * 
  COS(RADIANS(lon) - RADIANS(target_lon)) + 
  SIN(RADIANS(lat)) * SIN(RADIANS(target_lat))
)
```
- **参数**：  
  - `地球半径`：`3979` 英里（约 6371 公里）。  
  - `target_lat` 和 `target_lon`：目标点的经纬度。  
- **用途**：计算两个地理点之间的球面距离。

---

##### **2. 范围预筛选参数**
- **偏移量计算**：`DEGREES(0.0253) ≈ 1.45°`  
  - 假设 `100 英里 ≈ 160.9 公里`，对应地球表面的弧度为：  
    ```text
    弧度 = 距离 / 地球半径 = 100 / 3979 ≈ 0.0253
    ```
  - 转换为经纬度偏移量：`DEGREES(0.0253) ≈ 1.45°`。

---

#### **五、优化效果对比**

| **方法**               | **扫描行数** | **索引利用** | **性能** | **精度** |
|------------------------|--------------|--------------|----------|----------|
| 直接计算 Haversine     | 全表扫描     | 无           | 极差     | 精确     |
| 范围预筛选             | 部分扫描     | 可能         | 中等     | 近似     |
| 分块索引 + 精确计算    | 极少扫描     | 是           | 最优     | 精确     |

---

#### **六、潜在优化方向**

1. **提高分块精度**：  
   - 使用更小的分块（如 0.1°×0.1°），需权衡索引开销与筛选效率。  
   - 示例：`lat_floor = FLOOR(lat * 10)`，`lon_floor = FLOOR(lon * 10)`。

2. **空间索引（GIS）**：  
   - 使用 MySQL 的 `SPATIAL` 索引和 `ST_Distance_Sphere` 函数。  
   - 示例：  
     ```sql
     ALTER TABLE locations ADD pt POINT;
     UPDATE locations SET pt = POINT(lon, lat);
     CREATE SPATIAL INDEX idx_spatial ON locations(pt);
     SELECT * FROM locations 
     WHERE ST_Distance_Sphere(pt, POINT(-78.48, 38.03)) <= 100 * 1609.34;  -- 100英里转米
     ```

3. **数据分区**：  
   - 按 `lat_floor` 或 `lon_floor` 分区，进一步提升查询性能。

---

#### **七、总结**
通过分块索引和范围预筛选，将复杂的地理距离计算拆解为“快速筛选 + 精确计算”两步，显著减少需要处理的数据量，从而提升查询性能。此方法适用于对精度要求较高且数据量大的场景，但需根据实际业务调整分块粒度和计算公式参数。


```sql
CREATE TABLE locations (
    id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(30),
    lat FLOAT NOT NULL,
    lon FLOAT NOT NULL
);

INSERT INTO locations(name, lat, lon)
VALUES 
('Charlottesville, Virginia', 38.03, -78.48), 
('Chicago, Illinois', 42.85, -87.65), 
('Washington, DC', 38.89, -77.04)
;

SELECT * FROM locations WHERE 3979 * ACOS(COS(RADIANS(lat)) *COS(RADIANS(38.03)) * COS(RADIANS(lon) - RADIANS(-78.48)) +SIN(RADIANS(lat)) *SIN(RADIANS(38.03))) <= 100;


SELECT * FROM locations
WHERE Lat BETWEEN 38.03 - DEGREES(0.0253) AND 38.03 +DEGREES(0.0253)
AND lon BETWEEN -78.48 - DEGREES(0.0253) AND -78.48 + DEGREES(0.0253);

alter table locations add lat_floor int not null default 0,
add lon_floor int not null default 0,
add key (lat_floor, lon_floor);

update locations set lat_floor = floor(lat), lon_floor = floor(lon);

select floor(38.03 - degrees(0.0253)) as lat_lb,
    ceiling(38.03 + degrees(0.0253)) as lat_ub,
    floor(-78.48 - degrees(0.0253)) as lon_lb,
    ceiling(-78.48 + degrees(0.0253)) as lon_ub;

SELECT * FROM locations
where lat_floor in(36,37,38,39,40) and lon_floor in(-80,-79,-78,-77)
AND 3979 * ACOS(COS(RADIANS(lat)) * COS(RADIANS(38.03)) * COS(RADIANS(lon) - RADIANS(-78.48)) + SIN(RADIANS(lat)) * SIN(RADIANS(38.03))
) <= 100;
```