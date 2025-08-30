# 使用Redis+Lua实现时间窗内某个接口的请求数限流

## 1. 基本原理

### 1.1 为什么选择Redis+Lua

在分布式系统中实现限流，需要考虑以下几个关键因素：

- **原子性**：限流操作必须是原子的，避免竞态条件
- **性能**：限流不应成为系统瓶颈
- **分布式一致性**：在多个节点间保持限流状态一致
- **可扩展性**：随着系统规模增长，限流机制能够水平扩展

Redis+Lua的组合提供了一个理想的解决方案：

- Redis提供高性能的内存数据存储
- Lua脚本在Redis中执行具有原子性
- 单个Lua脚本可以执行多个Redis命令，减少网络往返
- Redis集群支持水平扩展

### 1.2 限流算法选择

在Redis+Lua实现中，常用的限流算法有：

1. **固定窗口计数器**：简单但有临界问题
2. **滑动窗口计数器**：更精确的请求控制
3. **漏桶算法**：以固定速率处理请求
4. **令牌桶算法**：允许一定的突发流量

本文主要介绍**滑动窗口计数器**的实现，它能够在保持较高性能的同时提供精确的限流控制。

## 2. Redis数据结构选择

### 2.1 有序集合(ZSET)的优势

在实现滑动窗口限流时，Redis的有序集合(ZSET)是理想的数据结构：

- **排序能力**：ZSET根据score自动排序，可用于存储时间戳
- **范围操作**：支持按score范围删除元素，便于移除过期请求
- **计数功能**：ZCARD命令可快速获取集合大小，即当前窗口内的请求数
- **过期设置**：可以为整个集合设置过期时间，自动清理过期数据

### 2.2 数据结构设计

```
key格式: rate_limit:{resource}:{dimension}
例如: rate_limit:login_api:user_123

ZSET结构:
- score: 请求时间戳(毫秒)
- member: 请求标识(可以是时间戳或请求ID)
```

## 3. Lua脚本实现

### 3.1 基本滑动窗口限流脚本

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
    return 1  -- 允许请求
else
    return 0  -- 拒绝请求
end
```

### 3.2 脚本解析

1. **参数说明**：
   - `KEYS[1]`: 限流的唯一标识，如API名称+用户ID
   - `ARGV[1]`: 时间窗口大小，单位毫秒，如60000表示1分钟
   - `ARGV[2]`: 限流阈值，表示窗口期内允许的最大请求数
   - `ARGV[3]`: 当前时间戳，单位毫秒

2. **执行流程**：
   - 清理过期的请求记录
   - 获取当前窗口内的请求数
   - 判断是否超过限制
   - 如果未超过，记录当前请求并设置过期时间
   - 返回结果：1表示允许，0表示拒绝

### 3.3 高级滑动窗口限流脚本

```lua
-- advanced_rate_limiter.lua
-- KEYS[1]: 限流key
-- ARGV[1]: 时间窗口大小（毫秒）
-- ARGV[2]: 限流阈值
-- ARGV[3]: 当前时间戳（毫秒）
-- ARGV[4]: 请求标识（可选）

local key = KEYS[1]
local window = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requestId = ARGV[4] or now

-- 移除时间窗口外的数据
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)

-- 获取当前窗口内的请求数
local current = redis.call('ZCARD', key)

-- 返回结果包含更多信息
local result = {}
result[1] = current < limit and 1 or 0  -- 是否允许请求
result[2] = limit  -- 限流阈值
result[3] = current  -- 当前请求数

if result[1] == 1 then
    -- 添加当前请求
    redis.call('ZADD', key, now, requestId)
    -- 设置key过期时间
    redis.call('PEXPIRE', key, window)
    -- 获取剩余可用请求数
    result[4] = limit - current - 1
else
    -- 获取最早的请求时间
    local earliest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    if #earliest > 0 then
        -- 计算重试时间
        result[4] = math.ceil((tonumber(earliest[2]) + window - now) / 1000)
    else
        result[4] = 0
    end
end

return result
```

## 4. Java实现示例

### 4.1 基本实现

```java
public class RedisRateLimiter {
    private final StringRedisTemplate redisTemplate;
    private final String luaScript;
    private final DefaultRedisScript<Long> redisScript;
    
    public RedisRateLimiter(StringRedisTemplate redisTemplate) {
        this.redisTemplate = redisTemplate;
        this.luaScript = loadScriptFromClasspath("rate_limiter.lua");
        this.redisScript = new DefaultRedisScript<>(luaScript, Long.class);
    }
    
    public boolean tryAcquire(String key, int limit, long windowInMillis) {
        List<String> keys = Collections.singletonList(key);
        long now = System.currentTimeMillis();
        
        Long result = redisTemplate.execute(
            redisScript,
            keys,
            String.valueOf(windowInMillis),
            String.valueOf(limit),
            String.valueOf(now)
        );
        
        return result != null && result == 1L;
    }
    
    private String loadScriptFromClasspath(String scriptName) {
        try {
            Resource resource = new ClassPathResource(scriptName);
            return StreamUtils.copyToString(resource.getInputStream(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException("Cannot load script: " + scriptName, e);
        }
    }
}
```

### 4.2 Spring Boot集成

```java
@Configuration
public class RedisConfig {
    @Bean
    public StringRedisTemplate redisTemplate(RedisConnectionFactory connectionFactory) {
        return new StringRedisTemplate(connectionFactory);
    }
    
    @Bean
    public RedisRateLimiter redisRateLimiter(StringRedisTemplate redisTemplate) {
        return new RedisRateLimiter(redisTemplate);
    }
}

@Service
public class RateLimitService {
    private final RedisRateLimiter rateLimiter;
    
    @Autowired
    public RateLimitService(RedisRateLimiter rateLimiter) {
        this.rateLimiter = rateLimiter;
    }
    
    public boolean allowRequest(String resource, String userId, int limit, long windowInMillis) {
        String key = String.format("rate_limit:%s:%s", resource, userId);
        return rateLimiter.tryAcquire(key, limit, windowInMillis);
    }
}

@RestController
public class ApiController {
    private final RateLimitService rateLimitService;
    
    @Autowired
    public ApiController(RateLimitService rateLimitService) {
        this.rateLimitService = rateLimitService;
    }
    
    @GetMapping("/api/resource")
    public ResponseEntity<String> accessResource(
            @RequestHeader("X-User-ID") String userId) {
        
        boolean allowed = rateLimitService.allowRequest(
            "resource_api", userId, 100, 60000); // 每分钟100次
            
        if (!allowed) {
            return ResponseEntity
                .status(HttpStatus.TOO_MANY_REQUESTS)
                .body("请求频率超限，请稍后再试");
        }
        
        // 正常处理请求
        return ResponseEntity.ok("请求成功");
    }
}
```

## 5. 性能优化

### 5.1 批量处理

对于需要一次获取多个令牌的场景，可以优化Lua脚本：

```lua
-- batch_rate_limiter.lua
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

### 5.2 本地缓存优化

使用本地缓存减少Redis访问：

```java
public class CachedRedisRateLimiter {
    private final RedisRateLimiter delegate;
    private final LoadingCache<String, Boolean> localCache;
    
    public CachedRedisRateLimiter(RedisRateLimiter delegate) {
        this.delegate = delegate;
        this.localCache = Caffeine.newBuilder()
            .expireAfterWrite(1, TimeUnit.SECONDS)
            .build(key -> false); // 默认拒绝
    }
    
    public boolean tryAcquire(String key, int limit, long window) {
        // 先查本地缓存
        Boolean cached = localCache.getIfPresent(key);
        if (cached != null && !cached) {
            return false; // 本地缓存显示已被限流
        }
        
        // 查询Redis
        boolean result = delegate.tryAcquire(key, limit, window);
        if (!result) {
            // 缓存拒绝结果，减少Redis压力
            localCache.put(key, false);
        }
        return result;
    }
}
```

### 5.3 Redis集群优化

在高并发场景下，可以使用Redis集群提高性能：

1. **分片策略**：根据限流key进行哈希分片，将限流请求分散到不同Redis节点
2. **主从复制**：使用Redis主从架构提高可用性
3. **Redis Cluster**：使用Redis集群模式实现自动分片和故障转移

## 6. 最佳实践

### 6.1 Key设计

```java
public class RateLimitKeyBuilder {
    public static String buildKey(String prefix, String resource, String dimension) {
        return String.format("%s:%s:%s", prefix, resource, dimension);
    }
    
    // 使用示例
    // String userKey = buildKey("rate_limit", "login_api", "user:" + userId);
    // String ipKey = buildKey("rate_limit", "login_api", "ip:" + ipAddress);
}
```

### 6.2 动态配置

```java
@Configuration
@ConfigurationProperties(prefix = "rate.limit")
@RefreshScope // 支持配置中心动态刷新
public class RateLimitProperties {
    private Map<String, LimitConfig> configs = new HashMap<>();
    
    @Data
    public static class LimitConfig {
        private int limit = 100; // 默认限制
        private long window = 60000; // 默认1分钟
    }
    
    public LimitConfig getConfig(String resource) {
        return configs.getOrDefault(resource, 
            new LimitConfig()); // 返回默认配置
    }
}
```

### 6.3 降级策略

```java
public class RateLimitFallback {
    public static Object fallback(String key, Throwable e) {
        // 记录降级日志
        log.warn("Rate limit fallback triggered for key: {}", key, e);
        
        // Redis不可用时的策略
        if (e instanceof RedisConnectionFailureException) {
            // 临时允许请求通过，避免Redis故障导致服务不可用
            return true;
        }
        
        // 其他异常情况，默认拒绝
        return false;
    }
}
```

### 6.4 监控与告警

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
        boolean result = false;
        try {
            result = delegate.tryAcquire(key, limit, window);
            return result;
        } finally {
            sample.stop(Timer.builder("rate.limiter.execution")
                .tag("key", key)
                .tag("result", String.valueOf(result))
                .register(meterRegistry));
            
            // 记录限流计数
            if (!result) {
                meterRegistry.counter("rate.limiter.rejected", "key", key).increment();
            }
        }
    }
}
```

## 7. 与其他限流方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|-----|-----|-----|-------|
| **Redis+Lua** | 高性能、原子操作、分布式支持 | 依赖Redis可用性 | 大型分布式系统 |
| **Guava RateLimiter** | 轻量级、低延迟 | 仅单机有效 | 单体应用 |
| **Nginx限流** | 网关层限流、无代码侵入 | 配置复杂、不易动态调整 | API网关 |
| **Sentinel** | 功能丰富、熔断降级一体化 | 学习成本高 | 复杂微服务架构 |

## 8. 实际应用场景

### 8.1 登录接口防暴力破解

```java
@Service
public class LoginService {
    private final RedisRateLimiter rateLimiter;
    
    @Autowired
    public LoginService(RedisRateLimiter rateLimiter) {
        this.rateLimiter = rateLimiter;
    }
    
    public LoginResult login(String username, String password, String ip) {
        // IP维度限流：每分钟最多5次登录尝试
        String ipKey = "rate_limit:login:ip:" + ip;
        if (!rateLimiter.tryAcquire(ipKey, 5, 60000)) {
            return LoginResult.builder()
                .success(false)
                .message("登录尝试过于频繁，请稍后再试")
                .build();
        }
        
        // 用户名维度限流：每分钟最多3次登录尝试
        String usernameKey = "rate_limit:login:username:" + username;
        if (!rateLimiter.tryAcquire(usernameKey, 3, 60000)) {
            return LoginResult.builder()
                .success(false)
                .message("账号登录失败次数过多，请稍后再试")
                .build();
        }
        
        // 正常登录逻辑
        boolean authenticated = authenticateUser(username, password);
        if (!authenticated) {
            return LoginResult.builder()
                .success(false)
                .message("用户名或密码错误")
                .build();
        }
        
        return LoginResult.builder()
            .success(true)
            .message("登录成功")
            .build();
    }
}
```

### 8.2 短信验证码发送限制

```java
@Service
public class SmsService {
    private final RedisRateLimiter rateLimiter;
    
    @Autowired
    public SmsService(RedisRateLimiter rateLimiter) {
        this.rateLimiter = rateLimiter;
    }
    
    public boolean sendVerificationCode(String phone) {
        // 手机号维度：1分钟内最多发送1条
        String minuteKey = "rate_limit:sms:minute:" + phone;
        if (!rateLimiter.tryAcquire(minuteKey, 1, 60000)) {
            throw new BusinessException("发送过于频繁，请1分钟后再试");
        }
        
        // 手机号维度：1小时内最多发送5条
        String hourKey = "rate_limit:sms:hour:" + phone;
        if (!rateLimiter.tryAcquire(hourKey, 5, 3600000)) {
            throw new BusinessException("发送次数超限，请1小时后再试");
        }
        
        // 手机号维度：1天内最多发送10条
        String dayKey = "rate_limit:sms:day:" + phone;
        if (!rateLimiter.tryAcquire(dayKey, 10, 86400000)) {
            throw new BusinessException("今日发送次数已达上限，请明天再试");
        }
        
        // 发送短信验证码
        return sendSms(phone, generateCode());
    }
}
```

## 9. 总结

Redis+Lua实现的分布式限流方案具有以下优势：

1. **高性能**：Redis内存操作，响应时间通常在毫秒级
2. **原子性**：Lua脚本保证了限流逻辑的原子执行
3. **分布式支持**：适用于分布式系统，保证集群限流一致性
4. **灵活性**：可以根据不同维度（用户、IP、接口等）进行限流
5. **可扩展**：支持多种限流算法，可根据业务需求定制

在实际应用中，应根据系统规模、性能需求和业务场景选择合适的限流策略，并结合监控、告警和降级机制，构建完整的流量控制体系。