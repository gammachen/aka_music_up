# Redis核心数据结构与典型场景实战案例

## 在线案例

python ~/Code/cursor-projects/p-redis-web/app.py

http://127.0.0.1:5000/

---

![summary](summary.png)]

## 一、String类型：分布式锁

### 场景说明
实现分布式系统中的互斥资源访问控制，确保多节点操作的原子性。

![string](string.png)

```redis
# 获取锁（NX：不存在时设置，EX：过期时间30秒）
SET resource_lock "client1" NX EX 30

# 执行业务操作...

# 释放锁（Lua脚本保证原子性）
EVAL "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) else return 0 end" 1 resource_lock "client1"
```

---

## 二、Hash类型：电商商品详情存储

### 场景说明
存储商品的多维度属性，支持字段级更新操作。

![hash](hash.png)

```redis
# 存储商品信息
HSET product:1001 
    name "iPhone 15" 
    price 6999 
    stock 100 
    tags "5G,Smartphone"

# 获取单个字段
HGET product:1001 price  # 返回"6999"

# 扣减库存
HINCRBY product:1001 stock -1

# 获取全部字段
HGETALL product:1001
```

---

## 三、List类型：消息队列系统

### 场景说明
实现生产消费模型，处理订单通知等异步任务。

![list](list.png)

![list](list2.png)

```redis
# 生产者推送消息
LPUSH order_queue '{"order_id":10001, "user_id":2001, "amount":2999}'

# 消费者阻塞获取（超时5秒）
BRPOP order_queue 5
# 返回：1) "order_queue" 2) "{\"order_id\":10001,...}"
```

---

## 四、Set类型：社交标签系统

### 场景说明
管理用户兴趣标签，实现共同好友/兴趣发现。

![set](set.png)

```redis
# 添加用户标签
SADD user:2001:tags "科技" "数码" "编程"
SADD user:2002:tags "科技" "电影" "美食"

# 查找共同兴趣
SINTER user:2001:tags user:2002:tags
# 返回：1) "科技"

# 随机推荐标签
SRANDMEMBER user:2001:tags 2
```

---

## 五、ZSet类型：直播打赏排行榜

### 场景说明
实时更新和展示主播收到礼物的价值排名。

![zset](zset.png)

```redis
# 添加/更新主播分数
ZADD live_gift_rank 
    150000 "主播A" 
    98000 "主播B" 
    123000 "主播C"

# 获取TOP3主播
ZREVRANGE live_gift_rank 0 2 WITHSCORES
# 返回：1) "主播A" 2) "150000" 3) "主播C" 4) "123000"...

# 查询特定主播排名
ZREVRANK live_gift_rank "主播B"  # 返回2（第3名）
```

---

## 六、BitMap类型：用户签到系统

### 场景说明
记录用户每日签到状态，支持高效统计查询。

![bitmap](bitmap.png)

```redis
# 用户2023年10月签到记录（偏移量对应日期-1）
SETBIT sign:202310:2001 0 1  # 10月1日签到
SETBIT sign:202310:2001 2 1  # 10月3日签到

# 统计当月签到次数
BITCOUNT sign:202310:2001  # 返回2

# 检查某日是否签到
GETBIT sign:202310:2001 5  # 检查10月6日，返回0

# 连续签到天数（需配合BITPOS计算）
```

---

## 七、HyperLogLog：网站UV统计

### 场景说明
海量用户访问量统计，0.81%误差率的高效方案。

![hyperloglog](hyperloglog.png)

```redis
# 记录用户访问
PFADD 20231001:uv "192.168.1.1" "10.0.0.2" "172.16.0.3"

# 获取当日UV
PFCOUNT 20231001:uv  # 返回3

# 合并多日数据
PFMERGY 202310_uv 20231001:uv 20231002:uv
PFCOUNT 202310_uv
```

---

## 八、GEO类型：附近加油站查询

### 场景说明
基于地理位置的范围检索服务。

![geo](geo.png)

```redis
# 添加位置信息
GEOADD gas_stations 
    116.397469 39.908745 "加油站A" 
    116.405285 39.912506 "加油站B"

# 查询5公里范围内加油站
GEORADIUS gas_stations 116.400000 39.900000 5 km WITHDIST
# 返回：
# 1) 1) "加油站B" 2) "1.2345"
# 2) 1) "加油站A" 2) "2.3456"

# 查看位置原始坐标
ZRANGE gas_stations 0 -1 WITHSCORES
```

---

## 各数据结构对比总结

| 数据类型   | 优势场景                                | 性能特点                  | 存储优化技巧                |
|------------|---------------------------------------|--------------------------|---------------------------|
| String     | 原子计数器/分布式锁                   | O(1)读写                 | 使用整型存储减少内存占用    |
| Hash       | 对象属性存储                          | 字段级操作               | 控制field数量使用ziplist编码|
| List       | 消息队列/最新列表                     | 头尾操作高效             | 避免超大列表，分片存储      |
| Set        | 去重集合运算                          | 存在性检查O(1)          | 小集合使用intset编码        |
| ZSet       | 带权重排序场景                        | 范围查询O(logN)          | 合理设置跳表层数            |
| BitMap     | 布尔值海量存储                        | 位操作O(1)              | 按时间分片管理              |
| HyperLogLog| 大数据量去重统计                      | 固定12KB存储空间         | 合并分片数据提升精度         |
| GEO        | 地理位置服务                          | 半径查询O(logN)          | 使用Geohash优化存储         |

---

通过合理运用Redis的多数据结构特性，开发者可以高效解决各种复杂场景需求。建议根据业务特征选择最匹配的数据模型，同时关注内存优化与集群扩展方案，充分发挥Redis的高性能优势。