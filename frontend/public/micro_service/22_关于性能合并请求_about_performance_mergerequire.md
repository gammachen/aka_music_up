## 关于请求合并的实现方案

在高并发系统中，请求合并（Request Merging）是一种重要的性能优化技术，通过将多个相同或相似的请求合并为一个，减少对下游服务的调用次数，从而降低系统负载、提高吞吐量并减少资源消耗。本文详细介绍请求合并的原理、实现方案及最佳实践。

## 1. 请求合并的核心原理

### 1.1 基本概念

请求合并是指在特定时间窗口内，将多个相同或相似的请求合并为一个请求发送给下游服务，然后将获得的结果分发给原始请求的调用方。

### 1.2 合并策略

1. **时间窗口合并**：在固定时间窗口内的请求进行合并
2. **计数窗口合并**：累积到特定数量的请求后进行合并
3. **混合策略**：结合时间窗口和计数窗口，满足任一条件时触发合并

### 1.3 适用场景

- 批量查询数据（如根据多个ID查询用户信息）
- 缓存加载（如多个线程同时加载相同的缓存项）
- API调用（如多个微服务同时调用同一个外部API）
- 数据库查询（合并多个相同的SQL查询）

## 2. 请求合并的实现方案

### 2.1 批量查询合并

#### 2.1.1 基本原理

将多个单条查询合并为一个批量查询，减少网络交互次数和连接消耗。

#### 2.1.2 Java实现示例

```java
import java.util.*;
import java.util.concurrent.*;

public class BatchQueryMerger<K, V> {
    private final int maxBatchSize;
    private final long maxWaitTimeMs;
    private final ExecutorService executor;
    private final BatchQueryFunction<Collection<K>, Map<K, V>> batchQueryFunction;

    // 存储待处理的请求
    private final ConcurrentHashMap<K, CompletableFuture<V>> pendingRequests = new ConcurrentHashMap<>();
    // 当前批次的keys
    private final Set<K> currentBatchKeys = Collections.synchronizedSet(new HashSet<>());
    // 批次锁
    private final Object batchLock = new Object();
    // 是否有批次正在处理
    private volatile boolean batchInProgress = false;

    /**
     * 构造函数
     * @param maxBatchSize 最大批次大小
     * @param maxWaitTimeMs 最大等待时间(毫秒)
     * @param batchQueryFunction 批量查询函数
     */
    public BatchQueryMerger(int maxBatchSize, long maxWaitTimeMs, 
                           BatchQueryFunction<Collection<K>, Map<K, V>> batchQueryFunction) {
        this.maxBatchSize = maxBatchSize;
        this.maxWaitTimeMs = maxWaitTimeMs;
        this.executor = Executors.newSingleThreadExecutor();
        this.batchQueryFunction = batchQueryFunction;
    }

    /**
     * 查询接口
     * @param key 查询键
     * @return 查询结果的Future
     */
    public CompletableFuture<V> query(K key) {
        // 检查是否已有相同的请求正在处理
        CompletableFuture<V> future = pendingRequests.get(key);
        if (future != null) {
            return future;
        }

        // 创建新的Future
        CompletableFuture<V> newFuture = new CompletableFuture<>();
        CompletableFuture<V> existingFuture = pendingRequests.putIfAbsent(key, newFuture);
        if (existingFuture != null) {
            return existingFuture;
        }

        // 添加到当前批次
        currentBatchKeys.add(key);

        // 检查是否需要触发批量查询
        synchronized (batchLock) {
            if (!batchInProgress && (currentBatchKeys.size() >= maxBatchSize || pendingRequests.size() == 1)) {
                triggerBatchQuery();
            }
        }

        // 如果是第一个请求，启动定时器
        if (pendingRequests.size() == 1) {
            scheduleTimeout();
        }

        return newFuture;
    }

    /**
     * 触发批量查询
     */
    private void triggerBatchQuery() {
        synchronized (batchLock) {
            if (currentBatchKeys.isEmpty() || batchInProgress) {
                return;
            }
            
            batchInProgress = true;
            final Set<K> batchKeys = new HashSet<>(currentBatchKeys);
            currentBatchKeys.clear();
            
            executor.submit(() -> {
                try {
                    // 执行批量查询
                    Map<K, V> results = batchQueryFunction.apply(batchKeys);
                    
                    // 分发结果
                    for (K key : batchKeys) {
                        CompletableFuture<V> future = pendingRequests.remove(key);
                        if (future != null) {
                            V result = results.get(key);
                            if (result != null) {
                                future.complete(result);
                            } else {
                                future.completeExceptionally(
                                    new NoSuchElementException("No result for key: " + key));
                            }
                        }
                    }
                } catch (Exception e) {
                    // 处理异常，将异常传播给所有相关的Future
                    for (K key : batchKeys) {
                        CompletableFuture<V> future = pendingRequests.remove(key);
                        if (future != null) {
                            future.completeExceptionally(e);
                        }
                    }
                } finally {
                    synchronized (batchLock) {
                        batchInProgress = false;
                        // 检查是否有新的批次需要处理
                        if (!currentBatchKeys.isEmpty()) {
                            triggerBatchQuery();
                        }
                    }
                }
            });
        }
    }

    /**
     * 设置超时触发器
     */
    private void scheduleTimeout() {
        CompletableFuture.delayedExecutor(maxWaitTimeMs, TimeUnit.MILLISECONDS)
            .execute(() -> {
                synchronized (batchLock) {
                    if (!currentBatchKeys.isEmpty() && !batchInProgress) {
                        triggerBatchQuery();
                    }
                }
            });
    }

    /**
     * 批量查询函数接口
     */
    @FunctionalInterface
    public interface BatchQueryFunction<I, O> {
        O apply(I input) throws Exception;
    }

    /**
     * 关闭资源
     */
    public void shutdown() {
        executor.shutdown();
    }
}
```

#### 2.1.3 使用示例

```java
public class UserServiceExample {
    private final BatchQueryMerger<Long, UserInfo> userBatchQueryMerger;
    
    public UserServiceExample(UserRepository userRepository) {
        // 创建批量查询合并器，最多50个请求，最长等待20ms
        this.userBatchQueryMerger = new BatchQueryMerger<>(
            50, 
            20, 
            ids -> userRepository.findByIdIn(new ArrayList<>(ids))
                .stream()
                .collect(Collectors.toMap(UserInfo::getId, user -> user))
        );
    }
    
    public CompletableFuture<UserInfo> getUserById(Long userId) {
        return userBatchQueryMerger.query(userId);
    }
}
```

### 2.2 本地缓存请求合并

#### 2.2.1 基本原理

使用本地缓存拦截并合并重复的请求，只有第一个请求实际执行远程调用，其他请求共享结果。

#### 2.2.2 Guava LoadingCache实现

```java
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class CacheMergeService<K, V> {
    private final LoadingCache<K, V> cache;
    private final ListeningExecutorService executorService;
    
    public CacheMergeService(DataLoader<K, V> dataLoader, int expireSeconds) {
        ExecutorService executor = Executors.newFixedThreadPool(10);
        this.executorService = MoreExecutors.listeningDecorator(executor);
        
        this.cache = CacheBuilder.newBuilder()
            .maximumSize(10000)
            .expireAfterWrite(expireSeconds, TimeUnit.SECONDS)
            .build(new CacheLoader<K, V>() {
                @Override
                public V load(K key) throws Exception {
                    return dataLoader.load(key);
                }
                
                @Override
                public ListenableFuture<V> reload(K key, V oldValue) {
                    return executorService.submit(() -> dataLoader.load(key));
                }
            });
    }
    
    public V get(K key) throws Exception {
        return cache.get(key);
    }
    
    public interface DataLoader<K, V> {
        V load(K key) throws Exception;
    }
    
    public void shutdown() {
        executorService.shutdown();
    }
}
```

### 2.3 分布式请求合并

#### 2.3.1 基本原理

在分布式环境中，使用消息队列或共享存储来合并来自不同节点的请求。

#### 2.3.2 Redis实现请求合并

```java
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

public class RedisRequestMerger<K, V> {
    private final JedisPool jedisPool;
    private final String queueKey;
    private final String resultPrefix;
    private final int batchSize;
    private final long pollIntervalMs;
    private final long resultExpirySeconds;
    private final ExecutorService executor;
    private final BatchProcessor<K, V> batchProcessor;
    private volatile boolean running = true;
    
    public RedisRequestMerger(JedisPool jedisPool, String queueKey, String resultPrefix,
                             int batchSize, long pollIntervalMs, long resultExpirySeconds,
                             BatchProcessor<K, V> batchProcessor) {
        this.jedisPool = jedisPool;
        this.queueKey = queueKey;
        this.resultPrefix = resultPrefix;
        this.batchSize = batchSize;
        this.pollIntervalMs = pollIntervalMs;
        this.resultExpirySeconds = resultExpirySeconds;
        this.batchProcessor = batchProcessor;
        this.executor = Executors.newSingleThreadExecutor();
        
        // 启动批处理线程
        startBatchProcessingThread();
    }
    
    public CompletableFuture<V> submitRequest(K key) {
        String requestId = UUID.randomUUID().toString();
        String resultKey = resultPrefix + requestId;
        CompletableFuture<V> future = new CompletableFuture<>();
        
        // 将请求提交到Redis队列
        try (Jedis jedis = jedisPool.getResource()) {
            Map<String, String> requestData = new HashMap<>();
            requestData.put("key", key.toString());
            requestData.put("resultKey", resultKey);
            
            String requestJson = new ObjectMapper().writeValueAsString(requestData);
            jedis.lpush(queueKey, requestJson);
        } catch (Exception e) {
            future.completeExceptionally(e);
            return future;
        }
        
        // 启动结果轮询
        pollForResult(resultKey, future);
        
        return future;
    }
    
    private void pollForResult(String resultKey, CompletableFuture<V> future) {
        ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();
        scheduler.scheduleWithFixedDelay(() -> {
            try (Jedis jedis = jedisPool.getResource()) {
                String resultJson = jedis.get(resultKey);
                if (resultJson != null) {
                    V result = new ObjectMapper().readValue(resultJson, 
                        new TypeReference<V>() {});
                    future.complete(result);
                    jedis.del(resultKey);
                    scheduler.shutdown();
                }
            } catch (Exception e) {
                future.completeExceptionally(e);
                scheduler.shutdown();
            }
        }, 0, 100, TimeUnit.MILLISECONDS);
        
        // 设置超时
        scheduler.schedule(() -> {
            if (!future.isDone()) {
                future.completeExceptionally(
                    new TimeoutException("Request timed out after 30 seconds"));
                scheduler.shutdown();
            }
        }, 30, TimeUnit.SECONDS);
    }
    
    private void startBatchProcessingThread() {
        executor.submit(() -> {
            while (running) {
                try {
                    processBatch();
                    Thread.sleep(pollIntervalMs);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }
    
    private void processBatch() throws Exception {
        List<Map<String, String>> requestBatch = new ArrayList<>();
        
        // 从Redis队列获取批量请求
        try (Jedis jedis = jedisPool.getResource()) {
            for (int i = 0; i < batchSize; i++) {
                String requestJson = jedis.rpop(queueKey);
                if (requestJson == null) {
                    break;
                }
                Map<String, String> requestData = new ObjectMapper().readValue(
                    requestJson, new TypeReference<Map<String, String>>() {});
                requestBatch.add(requestData);
            }
        }
        
        if (requestBatch.isEmpty()) {
            return;
        }
        
        // 提取所有键并执行批处理
        List<K> keys = requestBatch.stream()
            .map(req -> (K) req.get("key"))
            .collect(Collectors.toList());
        
        Map<K, V> results = batchProcessor.processBatch(keys);
        
        // 将结果存储到Redis
        try (Jedis jedis = jedisPool.getResource()) {
            for (Map<String, String> request : requestBatch) {
                String keyStr = request.get("key");
                String resultKey = request.get("resultKey");
                
                // 查找对应的结果
                V result = results.get(keyStr);
                if (result != null) {
                    String resultJson = new ObjectMapper().writeValueAsString(result);
                    jedis.setex(resultKey, resultExpirySeconds, resultJson);
                } else {
                    // 没有结果时设置空值
                    jedis.setex(resultKey, resultExpirySeconds, "null");
                }
            }
        }
    }
    
    public interface BatchProcessor<K, V> {
        Map<K, V> processBatch(List<K> keys) throws Exception;
    }
    
    public void shutdown() {
        running = false;
        executor.shutdown();
    }
}
```

### 2.4 框架层请求合并

#### 2.4.1 使用Hystrix RequestCollapser

Hystrix提供了请求合并的功能，可以在框架层面自动合并请求。

```java
import com.netflix.hystrix.HystrixCollapser;
import com.netflix.hystrix.HystrixCollapserKey;
import com.netflix.hystrix.HystrixCollapserProperties;
import com.netflix.hystrix.HystrixCommand;
import com.netflix.hystrix.HystrixCommandGroupKey;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

public class UserRequestCollapser extends HystrixCollapser<List<User>, User, Long> {
    private final Long userId;
    private final UserRepository userRepository;
    
    public UserRequestCollapser(Long userId, UserRepository userRepository) {
        super(HystrixCollapser.Setter.withCollapserKey(HystrixCollapserKey.Factory.asKey("userCollapser"))
                .andCollapserPropertiesDefaults(HystrixCollapserProperties.Setter()
                        .withTimerDelayInMilliseconds(10) // 10ms窗口
                        .withMaxRequestsInBatch(50))); // 最多50个请求一批
        this.userId = userId;
        this.userRepository = userRepository;
    }
    
    @Override
    public Long getRequestArgument() {
        return userId;
    }
    
    @Override
    protected HystrixCommand<List<User>> createCommand(Collection<CollapsedRequest<User, Long>> collapsedRequests) {
        List<Long> userIds = collapsedRequests.stream()
                .map(CollapsedRequest::getArgument)
                .collect(Collectors.toList());
        
        return new BatchCommand(userIds, userRepository);
    }
    
    @Override
    protected void mapResponseToRequests(List<User> batchResponse, 
                                        Collection<CollapsedRequest<User, Long>> collapsedRequests) {
        // 创建ID到User的映射
        Map<Long, User> userMap = batchResponse.stream()
                .collect(Collectors.toMap(User::getId, Function.identity()));
        
        // 为每个请求设置结果
        for (CollapsedRequest<User, Long> request : collapsedRequests) {
            User user = userMap.get(request.getArgument());
            if (user != null) {
                request.setResponse(user);
            } else {
                request.setException(new NotFoundException("User not found"));
            }
        }
    }
    
    private static class BatchCommand extends HystrixCommand<List<User>> {
        private final List<Long> userIds;
        private final UserRepository userRepository;
        
        public BatchCommand(List<Long> userIds, UserRepository userRepository) {
            super(HystrixCommandGroupKey.Factory.asKey("userGroup"));
            this.userIds = new ArrayList<>(userIds);
            this.userRepository = userRepository;
        }
        
        @Override
        protected List<User> run() {
            return userRepository.findAllByIds(userIds);
        }
    }
}
```

#### 2.4.2 使用示例

```java
// 同步调用
User user = new UserRequestCollapser(userId, userRepository).execute();

// 异步调用
Future<User> future = new UserRequestCollapser(userId, userRepository).queue();
```

## 3. 性能对比分析

假设单次请求耗时是10ms，数据库连接建立时间是5ms，以下是不同场景下的性能对比：

| 场景 | 无合并 | 请求合并 | 提升比例 |
|------|------|---------|---------|
| 100个相同请求（串行） | ~1500ms (100×15ms) | ~20ms | 75倍 |
| 100个相同请求（50并发） | ~45ms | ~20ms | 2.25倍 |
| 100个不同ID查询（串行） | ~1500ms | ~25ms | 60倍 |
| 100个不同ID查询（50并发） | ~45ms | ~25ms | 1.8倍 |

## 4. 请求合并的最佳实践

### 4.1 合并参数选择

1. **时间窗口大小**：通常为5-20ms，过大会增加请求延迟，过小则合并效果不佳
2. **批次大小**：根据下游服务能力选择，通常为20-100，避免过大造成单次处理压力
3. **超时设置**：设置合理的超时时间，防止请求长时间等待

### 4.2 适用场景选择

1. **高频、低延迟敏感场景**：合并窗口应短（5-10ms）
2. **批量数据处理**：可使用较大窗口（20-50ms）和较大批次
3. **缓存加载**：适合使用Guava LoadingCache等工具

### 4.3 注意事项

1. **请求分组**：按照请求类型、参数特征进行分组合并
2. **错误处理**：批处理中单个请求错误不应影响其他请求
3. **监控指标**：监控合并率、延迟增加情况和资源节约效果
4. **回退策略**：合并服务不可用时提供回退到非合并模式的能力

## 5. 总结

请求合并是高并发系统中一种重要的优化手段，通过减少重复请求、降低下游服务调用次数，有效提升系统性能和资源利用率。根据不同场景选择合适的实现方式和参数配置，可以在保证服务质量的同时显著提高系统吞吐量。