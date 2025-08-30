# 服务降级（Service Degradation）详解

## 1. 降级概述

### 1.1 什么是服务降级
服务降级是指当系统面临高负载、资源不足或部分服务故障时，主动牺牲次要功能，保证核心服务正常运行的一种应急处理机制。在以下情况下可能需要服务降级：
- 访问量剧增，系统负载过高
- 服务响应时间过长或服务不可用
- 非核心服务影响到核心流程的性能

降级的最终目的是保证核心服务可用，即使是有损的方式。需要注意，有些服务是无法降级的（如加入购物车、结算等核心交易环节）。

### 1.2 降级决策依据
系统降级需要基于以下指标进行决策：
- 系统吞吐量
- 服务响应时间
- 系统可用率
- 资源利用率（CPU、内存、网络）
- 错误率和失败次数

## 2. 降级分类

### 2.1 按自动化程度分类
- **自动降级**：系统根据预设的规则和阈值自动触发降级
- **人工降级**：运维人员通过降级开关手动触发降级

### 2.2 按功能类型分类
- **读服务降级**：降低读操作的一致性要求，使用缓存或默认值
- **写服务降级**：将同步写操作转为异步，或限制写操作的频率和范围

### 2.3 按系统层次分类
- **多级降级**：在不同层次（前端、接入层、应用层、数据层）实施不同程度的降级

## 3. 降级策略与场景

### 3.1 页面级降级
- **整页降级**：在特殊情况下（如大促活动）完全关闭某些非核心页面，以节省系统资源
- **页面片段降级**：关闭页面中非核心模块，如商品详情页中的商家信息区域
- **页面异步请求降级**：取消或简化页面中的异步加载内容，如推荐信息、配送信息等

### 3.2 服务级降级
- **功能降级**：暂时关闭非核心功能，如商品详情页中的相关分类、热销榜等
- **读降级**：当后端服务异常时，降级为只读缓存，适用于对一致性要求不高的场景
- **写降级**：将实时写入转为异步写入，如秒杀场景下的库存扣减

### 3.3 特殊场景降级
- **爬虫请求降级**：在高负载情况下，将爬虫流量导向静态页面或返回简化数据
- **风控降级**：在高风险业务中（如秒杀），根据用户风险等级进行差异化处理，直接拒绝高风险用户

## 4. 自动降级机制

自动降级是根据系统负载、资源使用情况、SLA等指标自动触发的降级机制。

### 4.1 超时降级
当访问的数据库/HTTP服务/远程调用响应时间过长，且该服务不属于核心服务时，可以设置超时阈值，超时后自动降级。

**场景示例**：商品详情页上的推荐内容/评价信息，如果获取超时可以暂时不展示，这不会对用户的购物流程产生重大影响。

**实施要点**：
- 为不同服务设置合理的超时阈值
- 配置适当的超时重试机制
- 与外部服务提供方明确定义服务响应的最大时间

**代码实现**：

1. **Java中使用CompletableFuture实现超时降级**：
```java
public RecommendationResult getRecommendationsWithTimeout(long productId) {
    try {
        CompletableFuture<RecommendationResult> future = CompletableFuture.supplyAsync(() -> {
            return recommendationService.getRecommendations(productId);
        });
        
        // 设置超时时间为200毫秒，超时后返回降级结果
        return future.completeOnTimeout(getDefaultRecommendations(), 200, TimeUnit.MILLISECONDS).get();
    } catch (Exception e) {
        log.warn("获取推荐信息超时，启用降级策略", e);
        return getDefaultRecommendations();
    }
}

private RecommendationResult getDefaultRecommendations() {
    // 返回默认的推荐结果，可以是预先准备好的热门商品或空结果
    return new RecommendationResult(Collections.emptyList(), "降级推荐");
}
```

2. **Spring Cloud中使用Feign的超时配置**：
```yaml
# application.yml
feign:
  client:
    config:
      recommendation-service:
        connectTimeout: 1000
        readTimeout: 200
        loggerLevel: basic
```

```java
@FeignClient(name = "recommendation-service", fallback = RecommendationServiceFallback.class)
public interface RecommendationServiceClient {
    @GetMapping("/recommendations/{productId}")
    RecommendationResult getRecommendations(@PathVariable("productId") long productId);
}

@Component
public class RecommendationServiceFallback implements RecommendationServiceClient {
    @Override
    public RecommendationResult getRecommendations(long productId) {
        // 降级逻辑
        return new RecommendationResult(Collections.emptyList(), "降级推荐");
    }
}
```

### 4.2 熔断降级
当依赖的不稳定API连续失败次数达到设定阈值时，自动触发熔断降级，暂时切断对该服务的调用。同时通过异步线程定期探测服务是否恢复，恢复后自动取消降级。

**场景示例**：调用外部机票服务API，当连续失败超过预设次数时触发熔断。

**代码实现**：

1. **使用Netflix Hystrix实现熔断降级**：
```java
public class FlightServiceCommand extends HystrixCommand<FlightSearchResult> {
    private final FlightService flightService;
    private final FlightSearchRequest request;
    
    public FlightServiceCommand(FlightService flightService, FlightSearchRequest request) {
        super(Setter.withGroupKey(HystrixCommandGroupKey.Factory.asKey("FlightGroup"))
            .andCommandKey(HystrixCommandKey.Factory.asKey("searchFlights"))
            .andCommandPropertiesDefaults(
                // 设置熔断器相关属性
                HystrixCommandProperties.Setter()
                    .withCircuitBreakerEnabled(true) // 启用熔断器
                    .withCircuitBreakerRequestVolumeThreshold(10) // 熔断器请求阈值
                    .withCircuitBreakerErrorThresholdPercentage(50) // 错误率阈值
                    .withCircuitBreakerSleepWindowInMilliseconds(5000) // 熔断恢复时间
                    .withExecutionTimeoutInMilliseconds(1000) // 执行超时时间
            ));
        this.flightService = flightService;
        this.request = request;
    }
    
    @Override
    protected FlightSearchResult run() throws Exception {
        return flightService.searchFlights(request);
    }
    
    @Override
    protected FlightSearchResult getFallback() {
        // 熔断或执行失败时的降级逻辑
        log.warn("Flight service degraded for request: {}", request);
        return new FlightSearchResult(Collections.emptyList(), "服务暂时不可用，请稍后再试");
    }
}
```

2. **使用Resilience4j实现熔断降级**：
```java
@Service
public class FlightServiceWithResilience4j {
    private final FlightService flightService;
    private final CircuitBreaker circuitBreaker;
    
    public FlightServiceWithResilience4j(FlightService flightService) {
        this.flightService = flightService;
        
        // 创建CircuitBreaker配置
        CircuitBreakerConfig config = CircuitBreakerConfig.custom()
            .failureRateThreshold(50) // 50%的失败率将触发熔断
            .waitDurationInOpenState(Duration.ofMillis(1000)) // 熔断后等待时间
            .permittedNumberOfCallsInHalfOpenState(2) // 半开状态允许的调用次数
            .slidingWindowSize(10) // 滑动窗口大小
            .build();
        
        // 创建CircuitBreaker
        this.circuitBreaker = CircuitBreaker.of("flightService", config);
    }
    
    public FlightSearchResult searchFlights(FlightSearchRequest request) {
        // 使用CircuitBreaker包装服务调用
        Supplier<FlightSearchResult> supplier = CircuitBreaker.decorateSupplier(
            circuitBreaker, 
            () -> flightService.searchFlights(request)
        );
        
        try {
            return Try.ofSupplier(supplier)
                .recover(e -> {
                    // 降级逻辑
                    log.warn("Flight service degraded: {}", e.getMessage());
                    return new FlightSearchResult(Collections.emptyList(), "服务暂时不可用，请稍后再试");
                })
                .get();
        } catch (Exception e) {
            return new FlightSearchResult(Collections.emptyList(), "服务暂时不可用，请稍后再试");
        }
    }
}
```

3. **Spring Cloud Circuit Breaker + Resilience4j配置**：
```yaml
# application.yml
resilience4j:
  circuitbreaker:
    configs:
      default:
        registerHealthIndicator: true
        slidingWindowSize: 10
        minimumNumberOfCalls: 5
        permittedNumberOfCallsInHalfOpenState: 3
        automaticTransitionFromOpenToHalfOpenEnabled: true
        waitDurationInOpenState: 5s
        failureRateThreshold: 50
        eventConsumerBufferSize: 10
    instances:
      flightService:
        baseConfig: default
```

```java
@Service
public class FlightService {
    private final CircuitBreakerFactory circuitBreakerFactory;
    private final FlightServiceClient flightServiceClient;
    
    public FlightService(CircuitBreakerFactory circuitBreakerFactory, FlightServiceClient flightServiceClient) {
        this.circuitBreakerFactory = circuitBreakerFactory;
        this.flightServiceClient = flightServiceClient;
    }
    
    public FlightSearchResult searchFlights(FlightSearchRequest request) {
        return circuitBreakerFactory.create("flightService")
            .run(
                () -> flightServiceClient.searchFlights(request),
                throwable -> getFallbackFlightResult(request, throwable)
            );
    }
    
    private FlightSearchResult getFallbackFlightResult(FlightSearchRequest request, Throwable t) {
        log.warn("Flight service call failed: {}", t.getMessage());
        return new FlightSearchResult(Collections.emptyList(), "服务暂时不可用，请稍后再试");
    }
}
```

### 4.3 故障降级
当远程服务完全不可用时（网络故障、DNS故障、HTTP服务返回错误状态码、RPC服务抛出异常等），直接触发降级。

**降级处理方案**：
- 使用默认值（如库存服务不可用时，返回默认现货）
- 提供兜底数据（如广告服务不可用时，返回预先准备的静态广告）
- 使用本地缓存（展示之前缓存的数据）

**代码实现**：

1. **使用本地缓存作为降级方案**：
```java
@Service
public class ProductInventoryService {
    private final InventoryClient inventoryClient;
    private final LoadingCache<Long, InventoryInfo> inventoryCache;
    
    public ProductInventoryService(InventoryClient inventoryClient) {
        this.inventoryClient = inventoryClient;
        
        // 初始化Guava缓存
        this.inventoryCache = CacheBuilder.newBuilder()
            .maximumSize(10000)
            .expireAfterWrite(10, TimeUnit.MINUTES)
            .build(new CacheLoader<Long, InventoryInfo>() {
                @Override
                public InventoryInfo load(Long productId) throws Exception {
                    return inventoryClient.getInventory(productId);
                }
            });
    }
    
    public InventoryInfo getInventory(long productId) {
        try {
            // 尝试调用远程服务
            return inventoryClient.getInventory(productId);
        } catch (Exception e) {
            log.warn("Inventory service unavailable, using cached data: {}", e.getMessage());
            
            // 故障降级：从缓存获取
            try {
                return inventoryCache.getUnchecked(productId);
            } catch (Exception cacheException) {
                // 缓存也失败时，返回默认值
                log.error("Cache retrieval failed: {}", cacheException.getMessage());
                return new InventoryInfo(productId, true, 100, "降级库存信息");
            }
        }
    }
}
```

2. **使用服务注册发现的故障转移**：
```java
@Configuration
public class RibbonConfig {
    @Bean
    public IRule ribbonRule() {
        // 配置优先选择非故障实例
        return new AvailabilityFilteringRule();
    }
}
```

```java
@Service
public class InventoryServiceWithFailover {
    private final RestTemplate restTemplate;
    private final InventoryFallbackService fallbackService;
    
    public InventoryServiceWithFailover(
            @LoadBalanced RestTemplate restTemplate,
            InventoryFallbackService fallbackService) {
        this.restTemplate = restTemplate;
        this.fallbackService = fallbackService;
    }
    
    public InventoryInfo getInventory(long productId) {
        try {
            return restTemplate.getForObject(
                "http://inventory-service/inventory/{productId}",
                InventoryInfo.class,
                productId
            );
        } catch (Exception e) {
            log.warn("Inventory service call failed, using fallback: {}", e.getMessage());
            return fallbackService.getDefaultInventory(productId);
        }
    }
}

@Service
public class InventoryFallbackService {
    private final InventoryLocalCacheRepository cacheRepository;
    
    public InventoryFallbackService(InventoryLocalCacheRepository cacheRepository) {
        this.cacheRepository = cacheRepository;
    }
    
    public InventoryInfo getDefaultInventory(long productId) {
        // 首先尝试从本地缓存读取
        Optional<InventoryInfo> cachedInfo = cacheRepository.findById(productId);
        if (cachedInfo.isPresent()) {
            InventoryInfo info = cachedInfo.get();
            // 标记为缓存数据
            info.setSource("local_cache");
            return info;
        }
        
        // 没有缓存则返回默认值
        return new InventoryInfo(productId, true, 99, "default");
    }
}
```

### 4.4 限流降级
在高并发场景（如秒杀、抢购）中，当请求量超过系统承载能力时，对后续请求进行降级处理。

**降级处理方案**：
- 引导用户到排队页面，稍后重试
- 返回"无货"或"活动太火爆"等友好提示
- 按照优先级处理部分请求，拒绝低优先级请求

**代码实现**：

1. **使用Google Guava RateLimiter实现限流降级**：
```java
@Service
public class FlashSaleService {
    private final RateLimiter rateLimiter;
    private final OrderService orderService;
    
    public FlashSaleService(OrderService orderService) {
        // 创建限流器，每秒允许100个请求
        this.rateLimiter = RateLimiter.create(100.0);
        this.orderService = orderService;
    }
    
    public PlaceOrderResult placeOrder(OrderRequest request) {
        // 尝试获取令牌，设置100毫秒超时
        boolean acquired = rateLimiter.tryAcquire(100, TimeUnit.MILLISECONDS);
        
        if (!acquired) {
            // 限流降级：返回排队提示
            log.info("Rate limited for order: {}", request.getOrderId());
            return new PlaceOrderResult(false, "活动太火爆了，请稍后再试", "RATE_LIMITED");
        }
        
        // 正常下单逻辑
        try {
            return orderService.createOrder(request);
        } catch (Exception e) {
            log.error("Order creation failed: {}", e.getMessage());
            return new PlaceOrderResult(false, "下单失败，请重试", "ERROR");
        }
    }
}
```

2. **使用Redis+Lua脚本实现分布式限流**：
```java
@Service
public class DistributedRateLimiter {
    private final StringRedisTemplate redisTemplate;
    
    // Lua脚本，实现令牌桶算法
    private final String luaScript = 
        "local key = KEYS[1] " +
        "local limit = tonumber(ARGV[1]) " +
        "local interval = tonumber(ARGV[2]) " +
        "local current = tonumber(redis.call('get', key) or '0') " +
        "if current + 1 > limit then " +
        "   return 0 " +
        "else " +
        "   redis.call('INCRBY', key, 1) " +
        "   redis.call('EXPIRE', key, interval) " +
        "   return 1 " +
        "end";
    
    public DistributedRateLimiter(StringRedisTemplate redisTemplate) {
        this.redisTemplate = redisTemplate;
    }
    
    public boolean tryAcquire(String key, int limit, int interval) {
        try {
            DefaultRedisScript<Long> redisScript = new DefaultRedisScript<>();
            redisScript.setScriptText(luaScript);
            redisScript.setResultType(Long.class);
            
            // 执行Lua脚本
            Long result = redisTemplate.execute(
                redisScript,
                Collections.singletonList(key),
                String.valueOf(limit),
                String.valueOf(interval)
            );
            
            return result != null && result == 1L;
        } catch (Exception e) {
            log.error("Rate limiter error", e);
            return false; // 出错时拒绝请求，保护系统
        }
    }
}

@RestController
@RequestMapping("/flash-sale")
public class FlashSaleController {
    private final DistributedRateLimiter rateLimiter;
    private final OrderService orderService;
    
    public FlashSaleController(DistributedRateLimiter rateLimiter, OrderService orderService) {
        this.rateLimiter = rateLimiter;
        this.orderService = orderService;
    }
    
    @PostMapping("/orders")
    public ResponseEntity<?> placeOrder(@RequestBody OrderRequest request) {
        // 获取用户ID或IP作为限流标识
        String limitKey = "rate_limit:flash_sale:" + request.getUserId();
        
        // 尝试获取令牌，限制每个用户每10秒最多1次请求
        boolean allowed = rateLimiter.tryAcquire(limitKey, 1, 10);
        
        if (!allowed) {
            // 返回限流响应
            Map<String, String> response = new HashMap<>();
            response.put("status", "TOO_MANY_REQUESTS");
            response.put("message", "抢购人数太多，请稍后再试");
            return ResponseEntity.status(HttpStatus.TOO_MANY_REQUESTS).body(response);
        }
        
        // 正常处理订单
        return ResponseEntity.ok(orderService.createOrder(request));
    }
}
```

3. **使用Sentinel实现复杂限流降级策略**：
```java
@Service
public class SeckillServiceWithSentinel {
    private final SeckillOrderService orderService;
    
    public SeckillServiceWithSentinel(SeckillOrderService orderService) {
        this.orderService = orderService;
        
        // 配置限流规则
        initFlowRules();
    }
    
    private void initFlowRules() {
        List<FlowRule> rules = new ArrayList<>();
        
        // 创建限流规则
        FlowRule rule = new FlowRule();
        rule.setResource("createSeckillOrder");
        rule.setGrade(RuleConstant.FLOW_GRADE_QPS); // 基于QPS限流
        rule.setCount(100); // 每秒允许100个请求
        
        // 限流行为：CONTROL_BEHAVIOR_DEFAULT 直接拒绝请求
        rule.setControlBehavior(RuleConstant.CONTROL_BEHAVIOR_DEFAULT);
        
        rules.add(rule);
        FlowRuleManager.loadRules(rules);
    }
    
    public SeckillResult createOrder(SeckillRequest request) {
        Entry entry = null;
        try {
            // 尝试通过Sentinel限流检查
            entry = SphU.entry("createSeckillOrder");
            
            // 通过限流检查，执行正常下单逻辑
            return orderService.createOrder(request);
        } catch (BlockException e) {
            // 被限流，执行降级逻辑
            log.warn("Request blocked by Sentinel: {}", request);
            return new SeckillResult(false, "当前抢购人数过多，请稍后再试", "RATE_LIMITED");
        } catch (Exception e) {
            // 其他异常
            log.error("Order creation failed", e);
            return new SeckillResult(false, "系统异常，请稍后再试", "ERROR");
        } finally {
            if (entry != null) {
                entry.exit();
            }
        }
    }
}
```

4. **使用Spring Cloud Gateway实现API网关层限流**：
```yaml
# application.yml
spring:
  cloud:
    gateway:
      routes:
        - id: flash_sale_route
          uri: lb://flash-sale-service
          predicates:
            - Path=/api/flash-sale/**
          filters:
            - name: RequestRateLimiter
              args:
                redis-rate-limiter.replenishRate: 10
                redis-rate-limiter.burstCapacity: 20
                redis-rate-limiter.requestedTokens: 1
                key-resolver: "#{@userKeyResolver}"
```

```java
@Configuration
public class GatewayConfig {
    @Bean
    public KeyResolver userKeyResolver() {
        // 基于用户ID限流
        return exchange -> {
            String userId = exchange.getRequest().getHeaders().getFirst("X-User-ID");
            if (userId != null) {
                return Mono.just(userId);
            }
            // 如果没有用户ID，则基于IP限流
            return Mono.just(Objects.requireNonNull(
                exchange.getRequest().getRemoteAddress()).getHostString());
        };
    }
    
    // 自定义限流结果返回
    @Bean
    public ServerCodecConfigurer serverCodecConfigurer() {
        return ServerCodecConfigurer.create();
    }
    
    @Bean
    public GlobalFilter customGlobalFilter() {
        return (exchange, chain) -> chain.filter(exchange)
            .onErrorResume(throwable -> {
                if (throwable instanceof ResponseStatusException) {
                    ResponseStatusException ex = (ResponseStatusException) throwable;
                    if (ex.getStatus() == HttpStatus.TOO_MANY_REQUESTS) {
                        // 自定义限流响应
                        byte[] bytes = "{\"status\":429,\"message\":\"请求过于频繁，请稍后再试\"}".getBytes();
                        DataBuffer buffer = exchange.getResponse().bufferFactory().wrap(bytes);
                        
                        exchange.getResponse().setStatusCode(HttpStatus.TOO_MANY_REQUESTS);
                        exchange.getResponse().getHeaders().setContentType(MediaType.APPLICATION_JSON);
                        return exchange.getResponse().writeWith(Mono.just(buffer));
                    }
                }
                return Mono.error(throwable);
            });
    }
}
```

## 5. 人工降级机制

### 5.1 开关降级
通过预设的功能开关，在系统出现异常时手动触发降级。开关可以存放在：
- 配置文件
- 数据库
- Redis/ZooKeeper等分布式配置中心

**实施方式**：
- 定期同步开关状态（如每秒同步一次）
- 根据开关值决定是否执行降级逻辑

### 5.2 人工降级场景
- **服务异常**：监控发现服务异常，需要临时下线
- **依赖故障**：依赖的数据库出现网络拥塞、故障或慢查询时
- **流量突增**：发现调用量突然增大，需要改变处理方式（同步转异步）
- **灰度发布**：新服务上线时设置降级开关，出现问题可快速回退
- **多机房切换**：机房故障时进行服务切换
- **功能屏蔽**：某功能出现数据问题，需要临时屏蔽

## 6. 读服务降级实践

读服务通常相对容易实现降级，常见的策略包括：

### 6.1 多级缓存降级
构建多级缓存体系，当后端服务出现问题时，可以逐级降级：
1. 本地缓存
2. 分布式缓存
3. 数据库

### 6.2 数据一致性降级
在读服务中，可以降低数据一致性要求：
- 从强一致性降级为最终一致性
- 允许读取稍微过期的数据
- 返回默认值或空值

## 7. 写服务降级实践

写服务在大多数场景下是不可降级的，但可以通过一些策略实现有限的降级：

### 7.1 库存更新降级方案
**方案1（常规）**：
- 扣减DB库存
- 扣减成功后，更新Redis中的库存

**方案2（DB优先）**：
- 先扣减Redis库存
- 同步扣减DB库存
- 如果DB扣减失败，则回滚Redis库存

**方案3（降级方案）**：
- 扣减Redis库存
- 正常情况下同步扣减DB库存
- 高负载情况下，发送消息队列，异步扣减DB库存

**方案4（进一步降级）**：
- 扣减Redis库存
- 高负载情况下，将扣减DB库存的消息写入本地，后续异步处理

### 7.2 其他写服务降级策略
- **评价服务降级**：将用户评价从同步写入改为异步写入
- **按比例开放**：限制部分用户使用写功能（如部分用户看不到评价按钮）
- **降级奖励发放**：将奖励发放从同步改为异步
- **下单降级**：大促期间将下单数据写入Redis，峰值过后再同步到DB

## 8. 多级降级实践

系统可以在不同层次实施降级，越靠近用户端的降级越能有效保护后端系统：

### 8.1 前端降级
- **页面JS降级开关**：通过前端JS控制功能的启用/禁用
- **UI降级**：简化界面，减少复杂渲染和动画

### 8.2 接入层降级
- **网关/代理降级**：在API网关层面进行请求过滤和降级
- **CDN降级**：动态内容降级为静态内容

### 8.3 应用层降级
- **功能开关**：通过配置中心控制业务功能的开启/关闭
- **服务隔离**：核心服务与非核心服务资源隔离

### 8.4 数据层降级
- **读写分离**：读流量降级到从库
- **分片策略调整**：调整数据分片策略，保障核心业务数据的处理


