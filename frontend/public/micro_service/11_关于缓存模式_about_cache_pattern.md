# 缓存使用模式详解

## 1. 基本概念

### 1.1 核心术语

- **SoR (System-of-Record)**: 记录系统，或者可以叫做数据源，即实际存储原始数据的系统。
- **Cache**: 缓存，是SoR的快照数据，Cache的访问速度比SoR要快，放入Cache的目的是提升访问速度，减少回源到SoR的次数。
- **回源**: 即回到数据源头获取数据，Cache没有命中时，需要从SoR读取数据，这叫做回源。

## 2. 缓存模式

### 2.1 Cache-Aside模式

Cache-Aside即业务代码围绕着Cache写，是由业务代码直接维护缓存。这种模式适合使用AOP模式去实现。

#### 2.1.1 读场景实现

```java
@Service
public class ProductService {
    private final LoadingCache<String, Product> productCache;
    
    public ProductService() {
        this.productCache = CacheBuilder.newBuilder()
            .maximumSize(1000)  // 最大缓存数量
            .expireAfterWrite(10, TimeUnit.MINUTES)  // 写入后10分钟过期
            .build(new CacheLoader<String, Product>() {
                @Override
                public Product load(String key) throws Exception {
                    return loadFromSoR(key);  // 回源加载数据
                }
            });
    }
    
    public Product getProduct(String id) {
        try {
            // 1. 先从缓存中获取数据
            return productCache.get(id);
        } catch (ExecutionException e) {
            // 2. 如果缓存没有命中，则回源到SoR获取源数据
            return loadFromSoR(id);
        }
    }
    
    private Product loadFromSoR(String id) {
        // 从数据库或其他数据源加载数据
        return productRepository.findById(id).orElse(null);
    }
}
```

#### 2.1.2 写场景实现

```java
@Service
public class ProductService {
    private final Cache<String, Product> productCache;
    
    public ProductService() {
        this.productCache = CacheBuilder.newBuilder()
            .maximumSize(1000)
            .expireAfterWrite(10, TimeUnit.MINUTES)
            .build();
    }
    
    @Transactional
    public void updateProduct(Product product) {
        // 1. 先将数据写入SoR
        productRepository.save(product);
        
        // 2. 执行成功后立即同步写入缓存
        productCache.put(product.getId(), product);
    }
    
    @Transactional
    public void deleteProduct(String id) {
        // 1. 先将数据从SoR删除
        productRepository.deleteById(id);
        
        // 2. 失效缓存
        productCache.invalidate(id);
    }
}
```

### 2.2 Cache-As-SoR模式

Cache-As-SoR即把Cache看作为SoR，所有操作都是对Cache进行，然后Cache再委托给SoR进行真实的读/写。业务代码中只看到Cache的操作，看不到关于SoR相关的代码。

#### 2.2.1 Read-Through实现

```java
@Service
public class ProductService {
    private final LoadingCache<String, Product> productCache;
    
    public ProductService() {
        this.productCache = CacheBuilder.newBuilder()
            .maximumSize(1000)
            .expireAfterWrite(10, TimeUnit.MINUTES)
            .build(new CacheLoader<String, Product>() {
                @Override
                public Product load(String key) throws Exception {
                    return loadFromSoR(key);
                }
            });
    }
    
    public Product getProduct(String id) {
        try {
            // 直接通过Cache获取数据，Cache会自动处理回源
            return productCache.get(id);
        } catch (ExecutionException e) {
            throw new RuntimeException("Failed to load product", e);
        }
    }
}
```

#### 2.2.2 Write-Through实现

```java
@Service
public class ProductService {
    private final Cache<String, Product> productCache;
    
    public ProductService() {
        this.productCache = CacheBuilder.newBuilder()
            .maximumSize(1000)
            .expireAfterWrite(10, TimeUnit.MINUTES)
            .removalListener(new RemovalListener<String, Product>() {
                @Override
                public void onRemoval(RemovalNotification<String, Product> notification) {
                    if (notification.getCause() == RemovalCause.EXPLICIT) {
                        // 当缓存被显式移除时，同步到SoR
                        productRepository.save(notification.getValue());
                    }
                }
            })
            .build();
    }
    
    public void updateProduct(Product product) {
        // 直接更新缓存，缓存会通过RemovalListener同步到SoR
        productCache.put(product.getId(), product);
    }
}
```

#### 2.2.3 Write-Behind实现

```java
@Service
public class ProductService {
    private final Cache<String, Product> productCache;
    private final Queue<Product> writeQueue = new ConcurrentLinkedQueue<>();
    
    public ProductService() {
        this.productCache = CacheBuilder.newBuilder()
            .maximumSize(1000)
            .expireAfterWrite(10, TimeUnit.MINUTES)
            .removalListener(new RemovalListener<String, Product>() {
                @Override
                public void onRemoval(RemovalNotification<String, Product> notification) {
                    if (notification.getCause() == RemovalCause.EXPLICIT) {
                        // 将更新操作加入队列
                        writeQueue.offer(notification.getValue());
                    }
                }
            })
            .build();
        
        // 启动后台线程处理队列
        startWriteBehindThread();
    }
    
    private void startWriteBehindThread() {
        new Thread(() -> {
            while (true) {
                try {
                    Product product = writeQueue.poll();
                    if (product != null) {
                        // 批量写入SoR
                        productRepository.save(product);
                    } else {
                        Thread.sleep(1000);  // 队列为空时休眠
                    }
                } catch (Exception e) {
                    log.error("Write behind error", e);
                }
            }
        }).start();
    }
    
    public void updateProduct(Product product) {
        // 直接更新缓存，异步写入SoR
        productCache.put(product.getId(), product);
    }
}
```

## 3. 最佳实践

### 3.1 缓存策略选择

1. **Cache-Aside适用场景**:
   - 需要精确控制缓存行为的场景
   - 缓存逻辑与业务逻辑紧密相关的场景
   - 需要灵活处理缓存失效的场景

2. **Cache-As-SoR适用场景**:
   - 需要简化业务代码的场景
   - 缓存逻辑相对固定的场景
   - 需要统一管理缓存行为的场景

### 3.2 性能优化建议

1. **缓存配置优化**:
   ```java
   CacheBuilder.newBuilder()
       .maximumSize(1000)  // 设置最大缓存数量
       .expireAfterWrite(10, TimeUnit.MINUTES)  // 设置过期时间
       .expireAfterAccess(5, TimeUnit.MINUTES)  // 设置访问后过期时间
       .refreshAfterWrite(1, TimeUnit.MINUTES)  // 设置刷新时间
       .concurrencyLevel(4)  // 设置并发级别
       .recordStats()  // 开启统计
       .build();
   ```

2. **监控与统计**:
   ```java
   CacheStats stats = cache.stats();
   System.out.println("命中率: " + stats.hitRate());
   System.out.println("平均加载时间: " + stats.averageLoadPenalty());
   System.out.println("缓存项数量: " + cache.size());
   ```

### 3.3 异常处理

1. **缓存穿透防护**:
   ```java
   public Product getProduct(String id) {
       try {
           return productCache.get(id, () -> {
               Product product = loadFromSoR(id);
               if (product == null) {
                   // 缓存空值，防止缓存穿透
                   return new Product();
               }
               return product;
           });
       } catch (ExecutionException e) {
           throw new RuntimeException("Failed to load product", e);
       }
   }
   ```

2. **缓存雪崩防护**:
   ```java
   public ProductService() {
       this.productCache = CacheBuilder.newBuilder()
           .maximumSize(1000)
           .expireAfterWrite(10 + new Random().nextInt(5), TimeUnit.MINUTES)  // 随机过期时间
           .build();
   }
   ```

