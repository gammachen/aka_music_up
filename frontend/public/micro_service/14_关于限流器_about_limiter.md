## 限流算法概览

### 1. 常见限流算法

#### 1.1 计数器算法（固定窗口）
**原理**：
- 在固定时间窗口内统计请求次数
- 超过阈值则拒绝请求
- 实现简单，但存在临界问题

**应用场景**：
1. **API访问控制**
   - 场景：限制用户每分钟的API调用次数
   - 案例：Twitter API限制每个用户每分钟最多发送300条推文
   - 实现：使用Redis的INCR和EXPIRE命令实现简单计数器

2. **短信发送限制**
   - 场景：防止短信轰炸
   - 案例：某电商平台限制每个手机号每小时最多发送3条验证码短信
   - 实现：使用Redis存储手机号和发送次数，设置1小时过期

3. **简单业务场景**
   - 场景：不需要精确控制的限流场景
   - 案例：限制每个IP每天最多访问网站1000次
   - 实现：使用内存计数器，每天零点重置

**实现示例**：
```java
// Java实现
public class CounterLimiter {
    private final int limit;
    private final long window;
    private final AtomicInteger counter;
    private long lastResetTime;

    public CounterLimiter(int limit, long window) {
        this.limit = limit;
        this.window = window;
        this.counter = new AtomicInteger(0);
        this.lastResetTime = System.currentTimeMillis();
    }

    public boolean tryAcquire() {
        long now = System.currentTimeMillis();
        if (now - lastResetTime > window) {
            counter.set(0);
            lastResetTime = now;
        }
        return counter.incrementAndGet() <= limit;
    }
}
```

#### 1.2 滑动窗口算法
**原理**：
- 将时间窗口细分为多个小窗口
- 统计最近N个小窗口的请求数
- 解决固定窗口的临界问题

**应用场景**：
1. **支付系统限流**
   - 场景：控制支付接口的调用频率
   - 案例：支付宝限制每个用户每分钟最多发起10笔支付请求
   - 实现：使用Redis的ZSET实现滑动窗口，记录最近1分钟内的请求时间戳

2. **秒杀系统**
   - 场景：控制商品抢购请求
   - 案例：某电商平台限制每个商品每秒最多处理1000个抢购请求
   - 实现：将1秒分为10个100ms的小窗口，统计最近1秒内的请求数

3. **API网关限流**
   - 场景：保护后端服务
   - 案例：某微服务架构中，网关限制每个服务每秒最多处理5000个请求
   - 实现：使用Redis+Lua脚本实现分布式滑动窗口

**实现示例**：
```java
// Java实现
public class SlidingWindowLimiter {
    private final int limit;
    private final int windowSize;
    private final Queue<Long> timestamps;

    public SlidingWindowLimiter(int limit, int windowSize) {
        this.limit = limit;
        this.windowSize = windowSize;
        this.timestamps = new LinkedList<>();
    }

    public synchronized boolean tryAcquire() {
        long now = System.currentTimeMillis();
        while (!timestamps.isEmpty() && now - timestamps.peek() > windowSize) {
            timestamps.poll();
        }
        if (timestamps.size() < limit) {
            timestamps.offer(now);
            return true;
        }
        return false;
    }
}
```

#### 1.3 令牌桶算法
**原理**：
- 以固定速率向桶中添加令牌
- 请求需要获取令牌才能通过
- 支持突发流量

**应用场景**：
1. **视频流媒体服务**
   - 场景：控制视频流传输速率
   - 案例：Netflix使用令牌桶算法控制视频码率，支持突发流量
   - 实现：根据网络状况动态调整令牌生成速率

2. **消息队列消费**
   - 场景：控制消息消费速率
   - 案例：Kafka消费者使用令牌桶控制消息拉取速率
   - 实现：根据消费者处理能力动态调整令牌生成速率

3. **爬虫控制**
   - 场景：控制爬虫访问频率
   - 案例：Google搜索引擎限制每个IP每秒最多发起10次请求，但允许短时间突发
   - 实现：使用Redis实现分布式令牌桶

**实现示例**：
```java
// Java实现
public class TokenBucketLimiter {
    private final int capacity;
    private final double refillRate;
    private double tokens;
    private long lastRefillTime;

    public TokenBucketLimiter(int capacity, double refillRate) {
        this.capacity = capacity;
        this.refillRate = refillRate;
        this.tokens = capacity;
        this.lastRefillTime = System.currentTimeMillis();
    }

    public synchronized boolean tryAcquire() {
        refill();
        if (tokens >= 1) {
            tokens -= 1;
            return true;
        }
        return false;
    }

    private void refill() {
        long now = System.currentTimeMillis();
        double timePassed = (now - lastRefillTime) / 1000.0;
        tokens = Math.min(capacity, tokens + timePassed * refillRate);
        lastRefillTime = now;
    }
}
```

#### 1.4 漏桶算法
**原理**：
- 请求以固定速率流出
- 超过桶容量则拒绝请求
- 平滑流量

**应用场景**：
1. **数据库访问控制**
   - 场景：保护数据库不被突发流量压垮
   - 案例：某电商平台使用漏桶算法控制数据库写入速率
   - 实现：使用消息队列作为漏桶，控制消费速率

2. **第三方API调用**
   - 场景：遵守第三方API的调用限制
   - 案例：调用微信支付API时，限制每秒最多发起100次请求
   - 实现：使用内存队列实现漏桶，固定速率处理请求

3. **日志处理系统**
   - 场景：控制日志写入速率
   - 案例：ELK日志系统使用漏桶算法控制日志索引速率
   - 实现：使用消息队列缓冲日志，固定速率消费

**实现示例**：
```java
// Java实现
public class LeakyBucketLimiter {
    private final int capacity;
    private final long leakInterval;
    private int water;
    private long lastLeakTime;

    public LeakyBucketLimiter(int capacity, long leakInterval) {
        this.capacity = capacity;
        this.leakInterval = leakInterval;
        this.water = 0;
        this.lastLeakTime = System.currentTimeMillis();
    }

    public synchronized boolean tryAcquire() {
        leak();
        if (water < capacity) {
            water++;
            return true;
        }
        return false;
    }

    private void leak() {
        long now = System.currentTimeMillis();
        long timePassed = now - lastLeakTime;
        int leaks = (int) (timePassed / leakInterval);
        if (leaks > 0) {
            water = Math.max(0, water - leaks);
            lastLeakTime = now;
        }
    }
}
```

### 2. 算法选择指南

#### 2.1 选择依据
1. **流量特征**
   - 突发流量：令牌桶算法
   - 平稳流量：漏桶算法
   - 简单计数：计数器算法
   - 精确控制：滑动窗口算法

2. **系统要求**
   - 低延迟：令牌桶算法
   - 平滑处理：漏桶算法
   - 简单实现：计数器算法
   - 精确统计：滑动窗口算法

3. **分布式需求**
   - 单机限流：内存实现
   - 分布式限流：Redis实现
   - 高并发场景：本地缓存+分布式限流

#### 2.2 实际案例
1. **电商系统限流方案**
   ```yaml
   # 多级限流配置
   rate-limit:
     # 网关层：滑动窗口算法
     gateway:
       type: sliding-window
       limit: 10000
       window: 1s
     # 服务层：令牌桶算法
     service:
       type: token-bucket
       rate: 1000
       burst: 2000
     # 接口层：计数器算法
     api:
       type: counter
       limit: 100
       window: 1m
   ```

2. **社交平台限流方案**
   ```java
   // 多维度限流实现
   @RateLimit(
     // 用户维度：滑动窗口
     user = @Limit(type = "sliding-window", limit = 100, window = "1m"),
     // IP维度：令牌桶
     ip = @Limit(type = "token-bucket", rate = 10, burst = 20),
     // 全局维度：漏桶
     global = @Limit(type = "leaky-bucket", rate = 1000)
   )
   public void postContent() {
     // 业务逻辑
   }
   ```

3. **金融系统限流方案**
   ```python
   # 多级限流实现
   class FinancialRateLimiter:
       def __init__(self):
           # 交易限流：漏桶算法
           self.trade_limiter = LeakyBucketLimiter(100, 1000)
           # 查询限流：令牌桶算法
           self.query_limiter = TokenBucketLimiter(1000, 2000)
           # 风控限流：滑动窗口算法
           self.risk_limiter = SlidingWindowLimiter(100, 60000)
   ```

### 3. Spring Cloud 限流实现

#### 3.1 使用 Spring Cloud Gateway
```yaml
# application.yml
spring:
  cloud:
    gateway:
      routes:
        - id: service-route
          uri: http://service
          predicates:
            - Path=/api/**
          filters:
            - name: RequestRateLimiter
              args:
                redis-rate-limiter.replenishRate: 10
                redis-rate-limiter.burstCapacity: 20
```

#### 3.2 使用 Sentinel
```java
// 注解方式
@SentinelResource(value = "resourceName", blockHandler = "handleBlock")
public String process() {
    // 业务逻辑
}

// 代码方式
public void process() {
    try (Entry entry = SphU.entry("resourceName")) {
        // 业务逻辑
    } catch (BlockException e) {
        // 限流处理
    }
}
```

### 4. Python 限流实现

#### 4.1 使用 Flask-Limiter
```python
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route("/api")
@limiter.limit("10 per minute")
def api():
    return "API Response"
```

#### 4.2 使用 Redis 实现分布式限流
```python
import redis
import time

class RedisRateLimiter:
    def __init__(self, redis_client, key, limit, window):
        self.redis = redis_client
        self.key = key
        self.limit = limit
        self.window = window

    def is_allowed(self):
        now = time.time()
        pipeline = self.redis.pipeline()
        pipeline.zremrangebyscore(self.key, 0, now - self.window)
        pipeline.zcard(self.key)
        pipeline.zadd(self.key, {str(now): now})
        pipeline.expire(self.key, self.window)
        _, current, _ = pipeline.execute()
        return current <= self.limit
```

### 5. 限流策略选择

#### 5.1 选择依据
- 系统特点：单机/分布式
- 流量特征：突发/平稳
- 性能要求：低延迟/高吞吐
- 资源限制：内存/CPU

#### 5.2 最佳实践
1. **多级限流**：
   - 网关层限流
   - 服务层限流
   - 接口层限流

2. **动态调整**：
   - 基于系统负载
   - 基于业务指标
   - 基于时间特征

3. **监控告警**：
   - 限流触发统计
   - 系统性能监控
   - 异常情况告警

### 6. 限流注意事项

1. **限流粒度**：
   - 全局限流
   - 用户限流
   - 接口限流
   - 资源限流

2. **限流处理**：
   - 直接拒绝
   - 排队等待
   - 降级处理
   - 返回缓存

3. **限流配置**：
   - 阈值设置
   - 时间窗口
   - 突发容量
   - 预热时间

4. **限流扩展**：
   - 分布式限流
   - 动态限流
   - 智能限流
   - 自适应限流

### 7. Redis+Lua实现分布式限流

#### 7.1 实现原理
1. **Redis数据结构选择**
   - 使用ZSET（有序集合）存储请求时间戳
   - 使用SCORE存储时间戳
   - 使用MEMBER存储请求标识

2. **Lua脚本优势**
   - 原子性操作
   - 减少网络往返
   - 高性能执行

#### 7.2 具体实现

##### 7.2.1 滑动窗口限流
```lua
-- rate_limiter.lua
-- KEYS[1]: 限流key
-- ARGV[1]: 时间窗口大小（毫秒）
-- ARGV[2]: 限流阈值
-- ARGV[3]: 当前时间戳（毫秒）

local key = KEYS[1]
local window = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

-- 移除时间窗口外的数据
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)

-- 获取当前窗口内的请求数
local current = redis.call('ZCARD', key)

if current < limit then
    -- 添加当前请求
    redis.call('ZADD', key, now, now)
    -- 设置key过期时间
    redis.call('PEXPIRE', key, window)
    return 1
else
    return 0
end
```

##### 7.2.2 Java调用示例
```java
public class RedisRateLimiter {
    private final JedisPool jedisPool;
    private final String script;
    
    public RedisRateLimiter(JedisPool jedisPool) {
        this.jedisPool = jedisPool;
        // 加载Lua脚本
        this.script = loadScript("rate_limiter.lua");
    }
    
    public boolean tryAcquire(String key, int limit, long window) {
        try (Jedis jedis = jedisPool.getResource()) {
            long now = System.currentTimeMillis();
            // 执行Lua脚本
            Object result = jedis.eval(script, 
                Collections.singletonList(key),
                Arrays.asList(
                    String.valueOf(window),
                    String.valueOf(limit),
                    String.valueOf(now)
                )
            );
            return (long)result == 1;
        }
    }
    
    private String loadScript(String scriptPath) {
        // 从文件加载Lua脚本
        try {
            return new String(Files.readAllBytes(Paths.get(scriptPath)));
        } catch (IOException e) {
            throw new RuntimeException("Failed to load script", e);
        }
    }
}
```

##### 7.2.3 Spring Boot集成示例
```java
@Configuration
public class RedisConfig {
    @Bean
    public RedisRateLimiter redisRateLimiter(RedisConnectionFactory factory) {
        return new RedisRateLimiter(factory);
    }
}

@RestController
public class ApiController {
    @Autowired
    private RedisRateLimiter rateLimiter;
    
    @GetMapping("/api")
    public ResponseEntity<String> api(@RequestHeader("X-User-ID") String userId) {
        String key = "rate_limit:api:" + userId;
        if (!rateLimiter.tryAcquire(key, 100, 60000)) {
            return ResponseEntity.status(429).body("Too Many Requests");
        }
        return ResponseEntity.ok("Success");
    }
}
```

#### 7.3 性能优化

1. **本地缓存优化**
```java
public class CachedRedisRateLimiter {
    private final RedisRateLimiter delegate;
    private final Cache<String, Boolean> localCache;
    
    public CachedRedisRateLimiter(RedisRateLimiter delegate) {
        this.delegate = delegate;
        this.localCache = Caffeine.newBuilder()
            .expireAfterWrite(1, TimeUnit.SECONDS)
            .build();
    }
    
    public boolean tryAcquire(String key, int limit, long window) {
        // 先从本地缓存检查
        Boolean cached = localCache.getIfPresent(key);
        if (cached != null) {
            return cached;
        }
        
        // 本地缓存未命中，查询Redis
        boolean result = delegate.tryAcquire(key, limit, window);
        localCache.put(key, result);
        return result;
    }
}
```

2. **批量处理优化**
```lua
-- batch_rate_limiter.lua
-- KEYS[1]: 限流key
-- ARGV[1]: 时间窗口大小
-- ARGV[2]: 限流阈值
-- ARGV[3]: 当前时间戳
-- ARGV[4]: 批量大小

local key = KEYS[1]
local window = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local batch = tonumber(ARGV[4])

redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local current = redis.call('ZCARD', key)

if current + batch <= limit then
    for i = 1, batch do
        redis.call('ZADD', key, now + i, now + i)
    end
    redis.call('PEXPIRE', key, window)
    return 1
else
    return 0
end
```

#### 7.4 监控与告警

1. **监控指标**
```java
public class MonitoredRedisRateLimiter {
    private final RedisRateLimiter delegate;
    private final MeterRegistry meterRegistry;
    
    public MonitoredRedisRateLimiter(RedisRateLimiter delegate, MeterRegistry meterRegistry) {
        this.delegate = delegate;
        this.meterRegistry = meterRegistry;
    }
    
    public boolean tryAcquire(String key, int limit, long window) {
        Timer.Sample sample = Timer.start(meterRegistry);
        boolean result = delegate.tryAcquire(key, limit, window);
        sample.stop(meterRegistry.timer("rate.limiter.latency", "key", key));
        
        meterRegistry.counter("rate.limiter.requests", "key", key, "result", String.valueOf(result)).increment();
        return result;
    }
}
```

2. **告警配置**
```yaml
# prometheus告警规则
groups:
- name: rate_limiter
  rules:
  - alert: HighRateLimitRejection
    expr: rate(rate_limiter_requests_total{result="false"}[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High rate limit rejection rate
      description: Rate limit rejection rate is above 10% for the last 5 minutes
```

#### 7.5 最佳实践

1. **Key设计**
```java
public class RateLimitKeyBuilder {
    public static String buildKey(String prefix, String resource, String dimension) {
        return String.format("%s:%s:%s", prefix, resource, dimension);
    }
    
    // 使用示例
    String userKey = buildKey("rate_limit", "api", "user:" + userId);
    String ipKey = buildKey("rate_limit", "api", "ip:" + ipAddress);
}
```

2. **动态配置**
```java
@Configuration
@ConfigurationProperties(prefix = "rate.limit")
public class RateLimitProperties {
    private Map<String, LimitConfig> configs = new HashMap<>();
    
    @Data
    public static class LimitConfig {
        private int limit;
        private long window;
    }
    
    public LimitConfig getConfig(String resource) {
        return configs.getOrDefault(resource, new LimitConfig(100, 60000));
    }
}
```

3. **降级策略**
```java
public class RateLimitFallback {
    public static Object fallback(String key, Throwable e) {
        // 记录降级日志
        log.warn("Rate limit fallback triggered for key: {}", key, e);
        
        // 返回降级响应
        return new ResponseEntity<>(
            new ErrorResponse("Service temporarily unavailable"),
            HttpStatus.SERVICE_UNAVAILABLE
        );
    }
}
```

### 8. 请求排队实现方案

#### 8.1 整体架构
```
[客户端] -> [API网关] -> [请求队列] -> [处理服务] -> [结果存储] -> [客户端]
```

#### 8.2 核心组件

1. **请求队列服务**
```java
public class RequestQueueService {
    private final RedisTemplate<String, String> redisTemplate;
    private final String queueKey;
    private final String processingKey;
    
    public RequestQueueService(RedisTemplate<String, String> redisTemplate) {
        this.redisTemplate = redisTemplate;
        this.queueKey = "request:queue";
        this.processingKey = "request:processing";
    }
    
    public QueuePosition enqueue(Request request) {
        // 生成唯一请求ID
        String requestId = UUID.randomUUID().toString();
        
        // 将请求信息序列化
        String requestJson = serializeRequest(request);
        
        // 将请求加入队列
        Long position = redisTemplate.opsForList().rightPush(queueKey, requestJson);
        
        // 返回排队位置信息
        return new QueuePosition(requestId, position);
    }
    
    public QueueStatus getQueueStatus(String requestId) {
        // 获取队列总长度
        Long total = redisTemplate.opsForList().size(queueKey);
        
        // 获取当前处理位置
        Long processingPosition = getProcessingPosition();
        
        // 计算等待位置
        Long waitPosition = total - processingPosition;
        
        return new QueueStatus(requestId, waitPosition, total);
    }
}
```

2. **处理服务**
```java
@Service
public class RequestProcessor {
    private final RequestQueueService queueService;
    private final ExecutorService executorService;
    
    @PostConstruct
    public void init() {
        // 启动处理线程
        executorService.submit(this::processRequests);
    }
    
    private void processRequests() {
        while (true) {
            try {
                // 从队列获取请求
                String requestJson = queueService.dequeue();
                if (requestJson != null) {
                    // 处理请求
                    processRequest(requestJson);
                } else {
                    // 队列为空，短暂休眠
                    Thread.sleep(100);
                }
            } catch (Exception e) {
                log.error("Error processing request", e);
            }
        }
    }
    
    private void processRequest(String requestJson) {
        // 反序列化请求
        Request request = deserializeRequest(requestJson);
        
        // 执行实际处理逻辑
        Result result = process(request);
        
        // 存储处理结果
        storeResult(request.getRequestId(), result);
    }
}
```

3. **结果存储服务**
```java
@Service
public class ResultStorageService {
    private final RedisTemplate<String, String> redisTemplate;
    
    public void storeResult(String requestId, Result result) {
        String resultKey = "result:" + requestId;
        String resultJson = serializeResult(result);
        
        // 存储结果，设置过期时间
        redisTemplate.opsForValue().set(resultKey, resultJson, 1, TimeUnit.HOURS);
    }
    
    public Result getResult(String requestId) {
        String resultKey = "result:" + requestId;
        String resultJson = redisTemplate.opsForValue().get(resultKey);
        
        if (resultJson != null) {
            return deserializeResult(resultJson);
        }
        return null;
    }
}
```

#### 8.3 客户端实现

1. **轮询获取结果**
```java
public class RequestClient {
    private final ResultStorageService resultStorage;
    private final RequestQueueService queueService;
    
    public Result submitRequest(Request request) {
        // 提交请求到队列
        QueuePosition position = queueService.enqueue(request);
        
        // 轮询获取结果
        while (true) {
            // 获取队列状态
            QueueStatus status = queueService.getQueueStatus(position.getRequestId());
            
            // 检查是否有结果
            Result result = resultStorage.getResult(position.getRequestId());
            if (result != null) {
                return result;
            }
            
            // 返回排队信息
            if (status.getWaitPosition() > 0) {
                return new QueueResponse(
                    "当前模型请求量过大，请求排队约 " + status.getWaitPosition() + " 位"
                );
            }
            
            // 短暂休眠后继续轮询
            Thread.sleep(1000);
        }
    }
}
```

2. **WebSocket实时通知**
```java
@RestController
public class QueueController {
    private final SimpMessagingTemplate messagingTemplate;
    
    @PostMapping("/submit")
    public QueuePosition submitRequest(@RequestBody Request request) {
        QueuePosition position = queueService.enqueue(request);
        
        // 启动异步任务监控队列状态
        monitorQueueStatus(position.getRequestId());
        
        return position;
    }
    
    private void monitorQueueStatus(String requestId) {
        CompletableFuture.runAsync(() -> {
            while (true) {
                QueueStatus status = queueService.getQueueStatus(requestId);
                
                // 通过WebSocket发送状态更新
                messagingTemplate.convertAndSendToUser(
                    requestId,
                    "/queue/status",
                    status
                );
                
                if (status.getWaitPosition() == 0) {
                    break;
                }
                
                Thread.sleep(1000);
            }
        });
    }
}
```

#### 8.4 数据模型

1. **请求对象**
```java
@Data
public class Request {
    private String requestId;
    private String userId;
    private Map<String, Object> parameters;
    private long timestamp;
}
```

2. **队列位置信息**
```java
@Data
public class QueuePosition {
    private String requestId;
    private Long position;
}
```

3. **队列状态信息**
```java
@Data
public class QueueStatus {
    private String requestId;
    private Long waitPosition;
    private Long totalPosition;
}
```

#### 8.5 性能优化

1. **批量处理**
```java
public class BatchProcessor {
    private static final int BATCH_SIZE = 10;
    
    public void processBatch() {
        List<String> requests = queueService.dequeueBatch(BATCH_SIZE);
        if (!requests.isEmpty()) {
            executorService.submit(() -> {
                for (String request : requests) {
                    processRequest(request);
                }
            });
        }
    }
}
```

2. **优先级队列**
```java
public class PriorityQueueService {
    private static final String HIGH_PRIORITY_QUEUE = "queue:high";
    private static final String NORMAL_PRIORITY_QUEUE = "queue:normal";
    
    public QueuePosition enqueue(Request request, Priority priority) {
        String queueKey = priority == Priority.HIGH ? 
            HIGH_PRIORITY_QUEUE : NORMAL_PRIORITY_QUEUE;
            
        return enqueueToQueue(queueKey, request);
    }
}
```

#### 8.6 监控告警

1. **队列监控**
```java
@Scheduled(fixedRate = 60000)
public void monitorQueue() {
    Long queueSize = redisTemplate.opsForList().size(queueKey);
    Long processingSize = redisTemplate.opsForList().size(processingKey);
    
    // 记录指标
    meterRegistry.gauge("queue.size", queueSize);
    meterRegistry.gauge("processing.size", processingSize);
    
    // 检查告警条件
    if (queueSize > threshold) {
        alertService.sendAlert("Queue size exceeds threshold: " + queueSize);
    }
}
```

2. **处理时间监控**
```java
public class ProcessingTimeMonitor {
    public void monitorProcessingTime(String requestId) {
        Timer.Sample sample = Timer.start(meterRegistry);
        
        try {
            processRequest(requestId);
        } finally {
            sample.stop(meterRegistry.timer("processing.time"));
        }
    }
}
```

#### 8.7 最佳实践

1. **队列清理**
```java
@Scheduled(fixedRate = 3600000)
public void cleanupQueue() {
    // 清理过期的请求
    redisTemplate.opsForList().trim(queueKey, 0, maxQueueSize);
    
    // 清理过期的结果
    Set<String> expiredResults = findExpiredResults();
    redisTemplate.delete(expiredResults);
}
```

2. **错误处理**
```java
public class ErrorHandler {
    public void handleError(Request request, Exception e) {
        // 记录错误日志
        log.error("Error processing request: " + request.getRequestId(), e);
        
        // 更新请求状态
        updateRequestStatus(request.getRequestId(), RequestStatus.FAILED);
        
        // 发送错误通知
        notifyError(request.getUserId(), e.getMessage());
    }
}
```

3. **限流控制**
```java
public class QueueRateLimiter {
    public boolean canEnqueue(String userId) {
        String key = "user:queue:" + userId;
        Long count = redisTemplate.opsForValue().increment(key);
        
        if (count == 1) {
            redisTemplate.expire(key, 1, TimeUnit.HOURS);
        }
        
        return count <= maxRequestsPerUser;
    }
}
```