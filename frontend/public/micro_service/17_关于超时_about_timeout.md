# 超时与重试机制

## 1. 概述

超时与重试是分布式系统中保障服务可靠性的基础机制。合理设置超时时间和重试策略可以：
- 防止资源无限期占用
- 快速失败并触发降级或容错机制
- 提高系统整体可用性
- 避免请求堆积导致的连锁故障

超时设置过短可能导致正常请求被误判为失败，过长则可能导致资源长时间占用。下面从各个层面详细探讨超时与重试机制的最佳实践。

## 2. 代理层超时与重试

常见的代理组件有Nginx、HAProxy、Envoy等，它们作为反向代理或负载均衡器，是实现超时控制的第一道防线。

### 2.1 Nginx配置

```nginx
# nginx.conf

http {
    # 与上游服务器建立连接的超时时间
    proxy_connect_timeout 5s;
    
    # 从上游服务器读取响应的超时时间
    proxy_read_timeout 10s;
    
    # 向上游服务器写入请求的超时时间
    proxy_send_timeout 10s;
    
    # 定义上游服务器组
    upstream backend_servers {
        # 最大失败次数与失败超时
        server backend1.example.com max_fails=3 fail_timeout=30s;
        server backend2.example.com max_fails=3 fail_timeout=30s;
        
        # 启用后端健康检查
        keepalive 32;
    }
    
    # 定义重试机制
    proxy_next_upstream error timeout invalid_header http_500 http_502 http_503 http_504;
    proxy_next_upstream_timeout 15s;
    proxy_next_upstream_tries 3;
    
    server {
        listen 80;
        
        location /api/ {
            proxy_pass http://backend_servers;
            
            # 此位置的特定超时设置（覆盖全局设置）
            proxy_read_timeout 20s;
        }
    }
}
```

### 2.2 HAProxy配置

```
# haproxy.cfg

global
    log 127.0.0.1 local0
    maxconn 4096
    
defaults
    log global
    mode http
    option httplog
    option dontlognull
    
    # 客户端超时设置
    timeout connect 5s
    timeout client 50s
    timeout server 50s
    
    # 重试设置
    retries 3
    
frontend http-in
    bind *:80
    default_backend servers
    
backend servers
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    
    # 服务器健康检查和超时设置
    server server1 10.0.0.1:8080 check inter 2000 rise 2 fall 3 maxconn 100
    server server2 10.0.0.2:8080 check inter 2000 rise 2 fall 3 maxconn 100
    
    # 如果后端失败，尝试重新连接
    option redispatch
```

### 2.3 Twemproxy (Redis代理)

```yaml
# nutcracker.yml

alpha:
  listen: 127.0.0.1:22121
  hash: fnv1a_64
  distribution: ketama
  auto_eject_hosts: true
  redis: true
  server_retry_timeout: 30000    # 30秒后重试被标记为失败的服务器
  server_failure_limit: 3        # 连续失败3次后标记服务器为失败
  timeout: 400                   # 与Redis通信的超时时间(毫秒)
  servers:
   - 127.0.0.1:6379:1
   - 127.0.0.1:6380:1
```

## 3. Web容器超时

Web容器是Java Web应用程序的运行环境，需要合理设置其连接超时参数。

### 3.1 Tomcat配置

```xml
<!-- server.xml -->
<Connector port="8080" protocol="HTTP/1.1"
           connectionTimeout="20000"           <!-- 连接超时: 20秒 -->
           redirectPort="8443"
           maxThreads="200"                   <!-- 最大线程数 -->
           acceptCount="100"                  <!-- 等待队列大小 -->
           connectionUploadTimeout="50000"    <!-- 上传超时: 50秒 -->
           disableUploadTimeout="false"
           keepAliveTimeout="60000"           <!-- 保持连接超时: 60秒 -->
           socketBuffer="65536" />            <!-- Socket缓冲区大小 -->
```

在Spring Boot应用中，可以通过application.properties配置：

```properties
# 连接超时时间(毫秒)
server.tomcat.connection-timeout=20000

# 最大线程数
server.tomcat.max-threads=200

# 接受队列大小
server.tomcat.accept-count=100

# 保持连接超时时间(毫秒)
server.tomcat.keep-alive-timeout=60000
```

### 3.2 Jetty配置

```xml
<!-- jetty.xml -->
<Configure id="Server" class="org.eclipse.jetty.server.Server">
    <Call name="addConnector">
        <Arg>
            <New class="org.eclipse.jetty.server.ServerConnector">
                <Arg><Ref refid="Server"/></Arg>
                <Set name="port">8080</Set>
                <Set name="idleTimeout">30000</Set>       <!-- 空闲超时: 30秒 -->
                <Set name="acceptQueueSize">100</Set>     <!-- 接受队列大小 -->
            </New>
        </Arg>
    </Call>
    
    <!-- 配置线程池 -->
    <Set name="threadPool">
        <New class="org.eclipse.jetty.util.thread.QueuedThreadPool">
            <Set name="minThreads">10</Set>
            <Set name="maxThreads">200</Set>
            <Set name="idleTimeout">60000</Set>
        </New>
    </Set>
</Configure>
```

在Spring Boot应用中：

```properties
# Jetty配置
server.jetty.threads.min=10
server.jetty.threads.max=200
server.jetty.threads.idle-timeout=60000
server.jetty.connection-idle-timeout=30000
```

## 4. 中间件客户端超时与重试

中间件客户端包括RPC框架、HTTP客户端和消息队列客户端等。

### 4.1 Spring RestTemplate

```java
@Configuration
public class RestTemplateConfig {
    
    @Bean
    public RestTemplate restTemplate() {
        // 创建请求工厂
        HttpComponentsClientHttpRequestFactory factory = new HttpComponentsClientHttpRequestFactory();
        // 连接超时：5秒
        factory.setConnectTimeout(5000);
        // 读取超时：15秒
        factory.setReadTimeout(15000);
        // 从连接池获取连接的超时：2秒
        factory.setConnectionRequestTimeout(2000);
        
        // 配置连接池
        PoolingHttpClientConnectionManager connectionManager = new PoolingHttpClientConnectionManager();
        connectionManager.setMaxTotal(200);
        connectionManager.setDefaultMaxPerRoute(20);
        
        HttpClient httpClient = HttpClientBuilder.create()
                .setConnectionManager(connectionManager)
                .setRetryHandler(new DefaultHttpRequestRetryHandler(3, true)) // 重试3次
                .build();
        
        factory.setHttpClient(httpClient);
        
        return new RestTemplate(factory);
    }
}
```

### 4.2 Dubbo超时与重试

```xml
<!-- dubbo消费者配置 -->
<dubbo:consumer timeout="5000" retries="2" />

<!-- 特定服务的配置 -->
<dubbo:reference id="userService" interface="com.example.UserService"
                 timeout="3000" retries="2" />
                 
<!-- 特定方法的配置 -->
<dubbo:reference id="userService" interface="com.example.UserService">
    <dubbo:method name="findUser" timeout="1000" retries="0" />
    <dubbo:method name="listUsers" timeout="5000" retries="3" />
</dubbo:reference>
```

或在Spring Boot中使用注解：

```java
@DubboReference(timeout = 3000, retries = 2)
private UserService userService;
```

### 4.3 Spring Cloud OpenFeign

```java
// Feign客户端定义
@FeignClient(name = "user-service", configuration = UserFeignConfig.class)
public interface UserFeignClient {
    @GetMapping("/users/{id}")
    UserDTO getUser(@PathVariable("id") Long id);
}

// Feign配置类
@Configuration
public class UserFeignConfig {
    @Bean
    public Request.Options requestOptions() {
        // 连接超时5秒，读取超时10秒
        return new Request.Options(5000, TimeUnit.MILLISECONDS, 
                                 10000, TimeUnit.MILLISECONDS, true);
    }
    
    @Bean
    public Retryer retryer() {
        // 重试间隔100ms，最大重试间隔1s，最多重试3次
        return new Retryer.Default(100, TimeUnit.SECONDS.toMillis(1), 3);
    }
}
```

在application.yml中全局配置：

```yaml
feign:
  client:
    config:
      default:
        connectTimeout: 5000
        readTimeout: 10000
        loggerLevel: full
        retryer: com.example.CustomRetryer
      user-service:  # 特定服务配置
        connectTimeout: 3000
        readTimeout: 5000
```

### 4.4 Apache HttpClient

```java
@Bean
public CloseableHttpClient httpClient() {
    RequestConfig requestConfig = RequestConfig.custom()
            .setConnectTimeout(5000)
            .setSocketTimeout(10000)
            .setConnectionRequestTimeout(2000)
            .build();
    
    // 配置重试策略
    HttpRequestRetryHandler retryHandler = (exception, executionCount, context) -> {
        if (executionCount > 3) {
            return false;
        }
        if (exception instanceof InterruptedIOException) {
            // 超时异常不重试
            return false;
        }
        if (exception instanceof UnknownHostException) {
            // 未知主机异常不重试
            return false;
        }
        if (exception instanceof ConnectTimeoutException) {
            // 连接超时可重试
            return true;
        }
        if (exception instanceof SSLException) {
            // SSL异常不重试
            return false;
        }
        
        HttpClientContext clientContext = HttpClientContext.adapt(context);
        HttpRequest request = clientContext.getRequest();
        // 幂等请求允许重试
        return !(request instanceof HttpEntityEnclosingRequest);
    };
    
    return HttpClients.custom()
            .setDefaultRequestConfig(requestConfig)
            .setRetryHandler(retryHandler)
            .setConnectionManager(poolingConnectionManager())
            .build();
}

@Bean
public PoolingHttpClientConnectionManager poolingConnectionManager() {
    PoolingHttpClientConnectionManager connectionManager = 
            new PoolingHttpClientConnectionManager();
    connectionManager.setMaxTotal(200);
    connectionManager.setDefaultMaxPerRoute(20);
    return connectionManager;
}
```

## 5. 数据库客户端超时

数据库客户端超时设置在应用程序与数据库之间的交互中至关重要。

### 5.1 JDBC超时配置

```java
@Configuration
public class DataSourceConfig {
    
    @Bean
    @ConfigurationProperties(prefix = "spring.datasource")
    public DataSource dataSource() {
        HikariConfig config = new HikariConfig();
        
        // 连接池参数
        config.setMaximumPoolSize(20);
        config.setMinimumIdle(5);
        config.setIdleTimeout(30000);      // 空闲连接超时：30秒
        config.setConnectionTimeout(30000); // 获取连接超时：30秒
        config.setMaxLifetime(1800000);    // 连接最大生命周期：30分钟
        
        // 连接参数
        config.addDataSourceProperty("connectTimeout", "10000"); // 建立连接超时：10秒
        config.addDataSourceProperty("socketTimeout", "60000");  // Socket超时：60秒
        
        return new HikariDataSource(config);
    }
    
    @Bean
    public JdbcTemplate jdbcTemplate(DataSource dataSource) {
        JdbcTemplate jdbcTemplate = new JdbcTemplate(dataSource);
        // 设置查询超时：30秒
        jdbcTemplate.setQueryTimeout(30);
        return jdbcTemplate;
    }
}
```

### 5.2 Spring Boot中的配置

```yaml
spring:
  datasource:
    hikari:
      connection-timeout: 30000  # 连接超时（毫秒）
      maximum-pool-size: 20      # 最大连接数
      minimum-idle: 5            # 最小空闲连接
      idle-timeout: 30000        # 空闲连接超时（毫秒）
      max-lifetime: 1800000      # 连接最大生命周期（毫秒）
    url: jdbc:mysql://localhost:3306/mydb?connectTimeout=10000&socketTimeout=60000
```

### 5.3 事务超时配置

```java
@Service
public class UserService {
    
    @Transactional(timeout = 30) // 事务超时时间：30秒
    public void createUser(User user) {
        // 业务逻辑
    }
}
```

或在XML配置中：

```xml
<tx:advice id="txAdvice" transaction-manager="transactionManager">
    <tx:attributes>
        <tx:method name="create*" timeout="30" />
        <tx:method name="update*" timeout="20" />
        <tx:method name="*" read-only="true" timeout="10" />
    </tx:attributes>
</tx:advice>
```

## 6. NoSQL客户端超时

NoSQL数据库如Redis、MongoDB等的客户端超时配置。

### 6.1 Redis客户端超时

#### 6.1.1 Spring Data Redis配置

```java
@Configuration
public class RedisConfig {
    
    @Bean
    public RedisConnectionFactory redisConnectionFactory() {
        // Lettuce连接工厂配置
        LettuceConnectionFactory factory = new LettuceConnectionFactory();
        
        // 超时配置
        factory.setTimeout(Duration.ofSeconds(5)); // 命令执行超时：5秒
        
        // Lettuce客户端配置
        LettuceClientConfiguration clientConfig = LettuceClientConfiguration.builder()
                .commandTimeout(Duration.ofSeconds(5))  // 命令超时
                .shutdownTimeout(Duration.ofSeconds(2)) // 关闭超时
                .build();
        
        factory.setClientConfiguration(clientConfig);
        return factory;
    }
}
```

#### 6.1.2 Jedis客户端配置

```java
@Bean
public JedisConnectionFactory redisConnectionFactory() {
    JedisPoolConfig poolConfig = new JedisPoolConfig();
    poolConfig.setMaxTotal(20);
    poolConfig.setMaxIdle(5);
    poolConfig.setMinIdle(1);
    poolConfig.setTestOnBorrow(true);
    poolConfig.setTestOnReturn(true);
    poolConfig.setTestWhileIdle(true);
    poolConfig.setMaxWaitMillis(10000); // 获取连接最大等待时间：10秒
    
    JedisConnectionFactory factory = new JedisConnectionFactory(poolConfig);
    factory.setHostName("localhost");
    factory.setPort(6379);
    factory.setTimeout(5000); // 操作超时：5秒
    
    return factory;
}
```

### 6.2 MongoDB客户端超时

```java
@Configuration
public class MongoConfig {
    
    @Bean
    public MongoClient mongoClient() {
        MongoClientOptions options = MongoClientOptions.builder()
                .connectTimeout(10000)          // 连接超时：10秒
                .socketTimeout(15000)           // Socket超时：15秒
                .serverSelectionTimeout(30000)  // 服务器选择超时：30秒
                .maxWaitTime(20000)             // 连接池最大等待时间：20秒
                .connectionsPerHost(20)         // 每主机最大连接数
                .build();
        
        return new MongoClient(new ServerAddress("localhost", 27017), options);
    }
}
```

Spring Boot配置：

```yaml
spring:
  data:
    mongodb:
      host: localhost
      port: 27017
      database: mydb
      connection-pool:
        max-size: 20
        min-size: 5
        max-wait-time: 20000 # 毫秒
      socket-timeout: 15000 # 毫秒
      connect-timeout: 10000 # 毫秒
      server-selection-timeout: 30000 # 毫秒
```

## 7. 业务超时

业务超时是应用程序内部对长时间运行的任务和流程的超时控制。

### 7.1 Java中的Future超时

```java
@Service
public class ProductService {
    
    @Autowired
    private ExecutorService executorService;
    
    public ProductResult getProductWithTimeout(Long productId) {
        try {
            Future<ProductResult> future = executorService.submit(() -> {
                // 复杂的产品查询逻辑
                return productRepository.findDetailedProduct(productId);
            });
            
            // 等待结果，最多等待500毫秒
            return future.get(500, TimeUnit.MILLISECONDS);
        } catch (TimeoutException e) {
            // 超时处理逻辑
            log.warn("Product query timed out for id: {}", productId);
            return ProductResult.fallback(productId);
        } catch (Exception e) {
            log.error("Error retrieving product", e);
            throw new ServiceException("Failed to retrieve product details", e);
        }
    }
}
```

### 7.2 CompletableFuture超时处理

```java
public CompletableFuture<OrderResult> processOrderWithTimeout(Order order) {
    CompletableFuture<OrderResult> future = CompletableFuture.supplyAsync(() -> {
        // 复杂的订单处理逻辑
        return orderProcessor.process(order);
    });
    
    // 设置超时处理
    return future.completeOnTimeout(
        OrderResult.timeout(order.getId()), // 超时后的默认值
        2, 
        TimeUnit.SECONDS  // 2秒超时
    ).exceptionally(ex -> {
        log.error("Order processing failed", ex);
        return OrderResult.error(order.getId(), ex.getMessage());
    });
}
```

### 7.3 定时任务超时控制

```java
@Service
public class OrderTimeoutService {
    
    @Autowired
    private OrderRepository orderRepository;
    
    // 每5分钟执行一次
    @Scheduled(fixedDelay = 300000)
    public void cancelTimeoutOrders() {
        // 查找所有创建时间超过30分钟且未支付的订单
        LocalDateTime cutoffTime = LocalDateTime.now().minusMinutes(30);
        
        List<Order> timeoutOrders = orderRepository.findByStatusAndCreateTimeBefore(
            OrderStatus.PENDING_PAYMENT, cutoffTime);
        
        for (Order order : timeoutOrders) {
            try {
                // 取消订单
                order.setStatus(OrderStatus.CANCELLED);
                order.setCancelReason("支付超时自动取消");
                orderRepository.save(order);
                
                // 发送通知
                notificationService.sendOrderCancelledNotification(order);
                
                // 释放库存
                inventoryService.releaseInventory(order);
                
                log.info("Order cancelled due to payment timeout: {}", order.getId());
            } catch (Exception e) {
                log.error("Failed to cancel timeout order: {}", order.getId(), e);
            }
        }
    }
}
```

## 8. 前端Ajax超时

前端Ajax请求的超时设置对用户体验至关重要。

### 8.1 原生XMLHttpRequest

```javascript
function fetchData(url, timeout = 10000) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        
        // 设置超时
        xhr.timeout = timeout;
        
        xhr.onload = function() {
            if (xhr.status >= 200 && xhr.status < 300) {
                resolve(JSON.parse(xhr.responseText));
            } else {
                reject(new Error(`HTTP error, status = ${xhr.status}`));
            }
        };
        
        xhr.onerror = function() {
            reject(new Error('Network error'));
        };
        
        xhr.ontimeout = function() {
            reject(new Error(`Request timed out after ${timeout}ms`));
        };
        
        xhr.open('GET', url);
        xhr.send();
    });
}

// 使用示例
fetchData('/api/products', 5000)
    .then(data => {
        console.log('Products:', data);
    })
    .catch(error => {
        if (error.message.includes('timed out')) {
            console.error('请求超时，请稍后重试');
            // 显示友好的超时提示
        } else {
            console.error('获取数据失败:', error);
        }
    });
```

### 8.2 使用Fetch API

```javascript
function fetchWithTimeout(url, options = {}, timeout = 10000) {
    return Promise.race([
        fetch(url, options),
        new Promise((_, reject) => 
            setTimeout(() => reject(new Error(`Request timed out after ${timeout}ms`)), timeout)
        )
    ]);
}

// 使用示例
fetchWithTimeout('/api/products', {
    method: 'GET',
    headers: {
        'Content-Type': 'application/json'
    }
}, 5000)
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Products:', data);
    })
    .catch(error => {
        if (error.message.includes('timed out')) {
            console.error('请求超时，请稍后重试');
            // 显示超时UI
        } else {
            console.error('获取数据失败:', error);
        }
    });
```

### 8.3 使用Axios

```javascript
import axios from 'axios';

// 创建axios实例
const api = axios.create({
    baseURL: '/api',
    timeout: 10000, // 全局超时设置: 10秒
    headers: {
        'Content-Type': 'application/json'
    }
});

// 请求拦截器
api.interceptors.request.use(config => {
    console.log(`Request to ${config.url} started`);
    return config;
}, error => {
    return Promise.reject(error);
});

// 响应拦截器
api.interceptors.response.use(response => {
    return response.data;
}, error => {
    if (error.code === 'ECONNABORTED' && error.message.includes('timeout')) {
        console.error('请求超时，请稍后重试');
        // 可以在这里实现自动重试逻辑
        return retryRequest(error.config);
    }
    return Promise.reject(error);
});

// 重试逻辑
function retryRequest(config, retries = 1, maxRetries = 3, retryDelay = 1000) {
    if (retries > maxRetries) {
        return Promise.reject(new Error(`Maximum retries (${maxRetries}) exceeded`));
    }
    
    console.log(`Retrying request to ${config.url}, attempt ${retries}/${maxRetries}`);
    
    // 延迟重试
    return new Promise(resolve => {
        setTimeout(() => {
            resolve(api(config).catch(error => {
                if (error.code === 'ECONNABORTED' && error.message.includes('timeout')) {
                    return retryRequest(config, retries + 1, maxRetries, retryDelay);
                }
                return Promise.reject(error);
            }));
        }, retryDelay);
    });
}

// 使用示例
api.get('/products')
    .then(data => {
        console.log('Products:', data);
    })
    .catch(error => {
        console.error('Error fetching products:', error.message);
    });

// 为特定请求设置不同的超时
api.post('/orders', orderData, { timeout: 20000 })
    .then(response => {
        console.log('Order created:', response);
    })
    .catch(error => {
        console.error('Error creating order:', error.message);
    });
```

## 9. 超时与重试的最佳实践

1. **超时时间梯度设计**
   - 外层组件超时时间 > 内层组件超时时间
   - 例如：Gateway > 服务调用 > 数据库查询

2. **重试策略选择**
   - 只对幂等操作设置自动重试
   - 使用退避策略（指数退避、随机抖动）
   - 有限次数重试，避免无限重试

3. **监控与告警**
   - 记录超时事件，设置告警阈值
   - 对频繁超时的服务进行重点监控

4. **熔断配合**
   - 超时重试与熔断结合使用
   - 当故障率达到阈值时停止重试，启用熔断

5. **超时调优流程**
   - 从业务场景分析预期响应时间
   - 设置略高于正常响应时间的超时值
   - 根据实际运行情况和监控数据调整