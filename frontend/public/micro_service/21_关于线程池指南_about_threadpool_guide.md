# 线程池详解

## 1. 线程池基础

### 1.1 线程池概述

线程池的目的类似于连接池，通过减少频繁创建和销毁线程来降低性能损耗。每个线程都需要一个内存栈，用于存储如局部变量、操作栈等信息，可以通过-Xss参数来调整每个线程栈大小(64位系统默认1024KB，可以根据实际情况调小，比如256KB)，通过调整该参数可以创建更多的线程，不过JVM不能无限制地创建线程，通过使用线程池可以限制创建的线程数，从而保护系统。线程池一般配合队列一起工作，使用线程池限制并发处理任务的数量。然后设置队列的大小，当任务超过队列大小时，通过一定的拒绝策略来处理，这样可以保护系统免受大流量而导致崩溃——只是部分拒绝服务，还是有一部分是可以正常服务的。

### 1.2 线程池工作原理

线程池一般有核心线程池大小和线程池最大大小配置，当线程池中的线程空闲一段时间时将会被回收，而核心线程池中的线程不会被回收。

线程池的工作流程如下：

1. 当有新任务提交时，如果线程池中运行的线程数小于核心线程数（corePoolSize），创建新线程来处理任务
2. 如果线程池中的线程数大于等于核心线程数，则将任务放入工作队列（workQueue）
3. 如果工作队列已满，且运行的线程数小于最大线程数（maximumPoolSize），则创建新线程来处理任务
4. 如果工作队列已满，且运行的线程数大于等于最大线程数，则执行拒绝策略

```java
public class ThreadPoolExecutorExample {
    public static void main(String[] args) {
        ThreadPoolExecutor executor = new ThreadPoolExecutor(
            2,                      // 核心线程数
            4,                      // 最大线程数
            60, TimeUnit.SECONDS,   // 空闲线程的保留时间
            new LinkedBlockingQueue<>(10), // 工作队列
            Executors.defaultThreadFactory(), // 线程工厂
            new ThreadPoolExecutor.CallerRunsPolicy() // 拒绝策略
        );
        
        // 提交任务
        for (int i = 0; i < 20; i++) {
            final int taskId = i;
            executor.submit(() -> {
                System.out.println("执行任务: " + taskId + " 线程: " + Thread.currentThread().getName());
                try {
                    // 模拟任务执行
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                return taskId;
            });
        }
        
        // 关闭线程池
        executor.shutdown();
    }
}
```

## 2. 线程池分类

### 2.1 Java内置线程池

Java提供了几种预定义的线程池，通过Executors工厂类可以创建：

#### 2.1.1 固定大小线程池 (FixedThreadPool)

```java
ExecutorService fixedThreadPool = Executors.newFixedThreadPool(nThreads);
```

特点：
- 核心线程数和最大线程数相同
- 使用无界队列 LinkedBlockingQueue
- 适用于处理CPU密集型任务

#### 2.1.2 缓存线程池 (CachedThreadPool)

```java
ExecutorService cachedThreadPool = Executors.newCachedThreadPool();
```

特点：
- 初始核心线程数为0，最大线程数为Integer.MAX_VALUE
- 使用SynchronousQueue，不存储任务，必须有线程来处理
- 空闲线程存活时间为60秒
- 适用于处理大量短期异步任务

#### 2.1.3 单线程执行器 (SingleThreadExecutor)

```java
ExecutorService singleThreadExecutor = Executors.newSingleThreadExecutor();
```

特点：
- 核心线程数和最大线程数都为1
- 使用无界队列 LinkedBlockingQueue
- 保证任务按提交顺序串行执行

#### 2.1.4 调度线程池 (ScheduledThreadPool)

```java
ScheduledExecutorService scheduledThreadPool = Executors.newScheduledThreadPool(corePoolSize);
```

特点：
- 核心线程数固定，最大线程数为Integer.MAX_VALUE
- 使用DelayedWorkQueue
- 支持定时或周期性任务执行

#### 2.1.5 工作窃取线程池 (WorkStealingPool)

```java
ExecutorService workStealingPool = Executors.newWorkStealingPool(parallelism);
```

特点：
- 基于ForkJoinPool实现
- 每个线程维护自己的双端队列
- 当一个线程空闲时，可以从其他线程队列"窃取"任务
- 适用于任务可以被拆分的场景，如并行计算

### 2.2 自定义线程池

创建自定义的线程池使用ThreadPoolExecutor类：

```java
ThreadPoolExecutor customThreadPool = new ThreadPoolExecutor(
    corePoolSize,          // 核心线程数
    maximumPoolSize,       // 最大线程数
    keepAliveTime,         // 空闲线程存活时间
    unit,                  // 时间单位
    workQueue,             // 工作队列
    threadFactory,         // 线程工厂
    handler                // 拒绝策略
);
```

## 3. 线程池关键参数

### 3.1 核心线程数与最大线程数

多少个线程合适呢？建议根据实际业务情况来压测决定，或者根据利特尔法则来算出一个合理的线程池大小，其定义是，在一个稳定的系统中，长时间观察到的平均用户数量L，等于长时间观察到的有效到达速率λ与平均每个用户在系统中花费的时间W的乘积，即L = λW。但实际情况是复杂的，如存在处理超时、网络抖动都会导致线程花费时间不一样。因此，还要考虑超时机制、线程隔离机制、快速失败机制等，来保护系统免遭大量请求或异常情况的冲击。

#### 3.1.1 CPU密集型任务

对于CPU密集型任务，线程数通常设置为CPU核心数+1：

```java
int cpuCores = Runtime.getRuntime().availableProcessors();
ThreadPoolExecutor cpuIntensivePool = new ThreadPoolExecutor(
    cpuCores + 1, cpuCores + 1, 0L, TimeUnit.MILLISECONDS,
    new LinkedBlockingQueue<>(1000)
);
```

#### 3.1.2 IO密集型任务

对于IO密集型任务，线程数可以设置为CPU核心数的2倍甚至更多：

```java
int cpuCores = Runtime.getRuntime().availableProcessors();
ThreadPoolExecutor ioIntensivePool = new ThreadPoolExecutor(
    cpuCores * 2, cpuCores * 2, 0L, TimeUnit.MILLISECONDS,
    new LinkedBlockingQueue<>(1000)
);
```

### 3.2 工作队列

线程池支持多种队列类型：

1. **ArrayBlockingQueue**: 有界队列，基于数组实现
   ```java
   BlockingQueue<Runnable> arrayQueue = new ArrayBlockingQueue<>(1000);
   ```

2. **LinkedBlockingQueue**: 可配置有界/无界队列，基于链表实现
   ```java
   BlockingQueue<Runnable> linkedQueue = new LinkedBlockingQueue<>(1000); // 有界
   BlockingQueue<Runnable> unboundedQueue = new LinkedBlockingQueue<>(); // 无界
   ```

3. **SynchronousQueue**: 不存储元素的阻塞队列，每个插入操作必须等待另一个线程调用移除操作
   ```java
   BlockingQueue<Runnable> synchronousQueue = new SynchronousQueue<>();
   ```

4. **PriorityBlockingQueue**: 带优先级的无界阻塞队列
   ```java
   BlockingQueue<Runnable> priorityQueue = new PriorityBlockingQueue<>();
   ```

5. **DelayQueue**: 延迟队列，元素只有到了指定的延迟时间才能被取出
   ```java
   BlockingQueue<Runnable> delayQueue = new DelayQueue<>();
   ```

### 3.3 拒绝策略

ThreadPoolExecutor提供了四种拒绝策略：

1. **AbortPolicy**: 直接抛出RejectedExecutionException异常（默认策略）
   ```java
   new ThreadPoolExecutor.AbortPolicy()
   ```

2. **CallerRunsPolicy**: 在调用者线程中执行任务
   ```java
   new ThreadPoolExecutor.CallerRunsPolicy()
   ```

3. **DiscardPolicy**: 直接丢弃任务，不做任何处理
   ```java
   new ThreadPoolExecutor.DiscardPolicy()
   ```

4. **DiscardOldestPolicy**: 丢弃队列头部（最旧）的任务，然后重新尝试执行当前任务
   ```java
   new ThreadPoolExecutor.DiscardOldestPolicy()
   ```

5. **自定义拒绝策略**:
   ```java
   new RejectedExecutionHandler() {
       @Override
       public void rejectedExecution(Runnable r, ThreadPoolExecutor executor) {
           // 自定义处理逻辑
           System.out.println("任务被拒绝: " + r.toString());
           // 可以记录日志、发送报警等
       }
   }
   ```

## 4. 线程池监控与动态调整

### 4.1 线程池状态监控

```java
public class ThreadPoolMonitor {
    private ThreadPoolExecutor executor;
    
    public ThreadPoolMonitor(ThreadPoolExecutor executor) {
        this.executor = executor;
    }
    
    public void printStatus() {
        System.out.println("=========================");
        System.out.println("线程池大小: " + executor.getPoolSize());
        System.out.println("活跃线程数: " + executor.getActiveCount());
        System.out.println("队列任务数: " + executor.getQueue().size());
        System.out.println("已完成任务数: " + executor.getCompletedTaskCount());
        System.out.println("=========================");
    }
}
```

### 4.2 动态调整线程池参数

ThreadPoolExecutor提供了一些方法来动态调整线程池参数：

```java
// 调整核心线程数
executor.setCorePoolSize(newCoreSize);

// 调整最大线程数
executor.setMaximumPoolSize(newMaxSize);

// 调整线程存活时间
executor.setKeepAliveTime(newTime, unit);

// 预先创建核心线程
executor.prestartAllCoreThreads();

// 允许核心线程超时
executor.allowCoreThreadTimeOut(true);
```

## 5. 线程池最佳实践

### 5.1 合理命名线程

使用自定义ThreadFactory为线程池中的线程提供有意义的名称，便于调试和问题排查：

```java
public class NamedThreadFactory implements ThreadFactory {
    private static final AtomicInteger poolNumber = new AtomicInteger(1);
    private final AtomicInteger threadNumber = new AtomicInteger(1);
    private final String namePrefix;
    
    public NamedThreadFactory(String poolName) {
        namePrefix = poolName + "-thread-";
    }
    
    @Override
    public Thread newThread(Runnable r) {
        Thread t = new Thread(r, namePrefix + threadNumber.getAndIncrement());
        if (t.isDaemon()) {
            t.setDaemon(false);
        }
        if (t.getPriority() != Thread.NORM_PRIORITY) {
            t.setPriority(Thread.NORM_PRIORITY);
        }
        return t;
    }
}
```

使用方式：

```java
ThreadPoolExecutor executor = new ThreadPoolExecutor(
    5, 10, 60, TimeUnit.SECONDS,
    new LinkedBlockingQueue<>(100),
    new NamedThreadFactory("business-service")
);
```

### 5.2 区分业务线程池

根据不同的业务需求创建不同的线程池，避免相互影响：

```java
// 处理用户请求的线程池
ThreadPoolExecutor userRequestPool = new ThreadPoolExecutor(
    10, 20, 60, TimeUnit.SECONDS,
    new LinkedBlockingQueue<>(500),
    new NamedThreadFactory("user-request"),
    new ThreadPoolExecutor.CallerRunsPolicy()
);

// 处理后台任务的线程池
ThreadPoolExecutor backgroundTaskPool = new ThreadPoolExecutor(
    5, 10, 60, TimeUnit.SECONDS,
    new LinkedBlockingQueue<>(200),
    new NamedThreadFactory("background-task"),
    new ThreadPoolExecutor.DiscardOldestPolicy()
);
```

### 5.3 定期清理空闲线程

对于长期运行的应用，考虑定期清理空闲线程，释放系统资源：

```java
// 允许核心线程超时
executor.allowCoreThreadTimeOut(true);

// 或者定期执行清理操作
ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);
scheduler.scheduleAtFixedRate(() -> {
    // 记录当前线程池状态
    int poolSize = executor.getPoolSize();
    int activeCount = executor.getActiveCount();
    
    // 如果线程池大小超过核心线程数，且活跃线程比例小于50%，执行清理
    if (poolSize > executor.getCorePoolSize() && activeCount < poolSize * 0.5) {
        executor.purge(); // 清理已取消的任务
    }
}, 1, 1, TimeUnit.HOURS);
```

### 5.4 任务优先级处理

实现任务优先级处理，确保重要任务优先执行：

```java
class PriorityTask implements Runnable, Comparable<PriorityTask> {
    private final Runnable task;
    private final int priority;
    
    public PriorityTask(Runnable task, int priority) {
        this.task = task;
        this.priority = priority;
    }
    
    @Override
    public void run() {
        task.run();
    }
    
    @Override
    public int compareTo(PriorityTask other) {
        return Integer.compare(other.priority, this.priority); // 高优先级排在前面
    }
}

// 使用优先级队列的线程池
ThreadPoolExecutor priorityExecutor = new ThreadPoolExecutor(
    5, 10, 60, TimeUnit.SECONDS,
    new PriorityBlockingQueue<>(),
    new NamedThreadFactory("priority-pool")
);

// 提交不同优先级的任务
priorityExecutor.execute(new PriorityTask(() -> System.out.println("低优先级任务"), 1));
priorityExecutor.execute(new PriorityTask(() -> System.out.println("高优先级任务"), 10));
```

### 5.5 异常处理机制

确保线程池中的任务异常被正确处理，避免线程静默死亡：

```java
// 自定义UncaughtExceptionHandler
Thread.UncaughtExceptionHandler handler = (thread, throwable) -> {
    System.err.println("线程 " + thread.getName() + " 发生异常: " + throwable.getMessage());
    // 记录异常日志，发送报警等
};

// 设置线程工厂
ThreadFactory factory = r -> {
    Thread thread = new Thread(r);
    thread.setName("monitored-thread-" + thread.getId());
    thread.setUncaughtExceptionHandler(handler);
    return thread;
};

// 创建使用自定义线程工厂的线程池
ThreadPoolExecutor executor = new ThreadPoolExecutor(
    5, 10, 60, TimeUnit.SECONDS,
    new LinkedBlockingQueue<>(100),
    factory
);

// 对于submit方法提交的任务，需要通过Future获取异常
Future<?> future = executor.submit(() -> {
    throw new RuntimeException("Task failed");
});

try {
    future.get(); // 会抛出ExecutionException
} catch (InterruptedException | ExecutionException e) {
    System.err.println("任务执行异常: " + e.getMessage());
}
```

### 5.6 线程池隔离与容错

使用Hystrix或类似工具实现线程池隔离，防止服务雪崩：

```java
// 使用Semaphore实现简单的隔离
public class IsolatedExecutor {
    private final ThreadPoolExecutor executor;
    private final Semaphore semaphore;
    
    public IsolatedExecutor(int coreSize, int maxSize, int queueCapacity, int maxConcurrency) {
        this.executor = new ThreadPoolExecutor(
            coreSize, maxSize, 60, TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(queueCapacity)
        );
        this.semaphore = new Semaphore(maxConcurrency);
    }
    
    public <T> Future<T> submit(Callable<T> task, long timeout, TimeUnit unit, T fallback) {
        return executor.submit(() -> {
            try {
                if (semaphore.tryAcquire(timeout, unit)) {
                    try {
                        return task.call();
                    } finally {
                        semaphore.release();
                    }
                } else {
                    // 超时无法获取信号量，返回降级结果
                    return fallback;
                }
            } catch (Exception e) {
                // 异常发生，返回降级结果
                return fallback;
            }
        });
    }
}
```

## 6. 线程池使用注意事项

### 6.1 避免使用Executors创建线程池

Java官方不建议使用Executors创建线程池，主要原因有：

- FixedThreadPool和SingleThreadExecutor使用无界队列，可能导致OOM
- CachedThreadPool和ScheduledThreadPool允许创建无限线程，可能导致OOM
- 不支持自定义ThreadFactory和拒绝策略

### 6.2 正确关闭线程池

优雅地关闭线程池，确保任务得到正确处理：

```java
public void shutdownThreadPoolGracefully(ThreadPoolExecutor executor) {
    // 停止接收新任务
    executor.shutdown();
    try {
        // 等待已提交任务完成
        if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
            // 强制关闭
            executor.shutdownNow();
            // 等待任务响应中断
            if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                System.err.println("线程池未正常终止");
            }
        }
    } catch (InterruptedException ie) {
        // 重新中断当前线程
        executor.shutdownNow();
        Thread.currentThread().interrupt();
    }
}
```

### 6.3 避免在线程池中执行阻塞操作

避免在线程池中执行长时间阻塞的操作，特别是不受控制的阻塞：

```java
// 不推荐
executor.execute(() -> {
    try {
        // 不确定何时返回的阻塞操作
        someBlockingOperation();
    } catch (Exception e) {
        e.printStackTrace();
    }
});

// 推荐：设置超时
executor.execute(() -> {
    try {
        // 设置超时的阻塞操作
        CompletableFuture<Object> future = CompletableFuture.supplyAsync(
            () -> someBlockingOperation(), 
            anotherExecutor
        );
        future.get(5, TimeUnit.SECONDS);
    } catch (TimeoutException e) {
        // 超时处理
    } catch (Exception e) {
        e.printStackTrace();
    }
});
```

### 6.4 正确处理上下文传递

确保线程池中的任务能够正确传递上下文（如事务、安全上下文、线程本地变量等）：

```java
// MDC上下文传递示例
executor.execute(() -> {
    // 复制当前线程的MDC上下文
    Map<String, String> contextMap = MDC.getCopyOfContextMap();
    try {
        // 在新线程中设置MDC上下文
        if (contextMap != null) {
            MDC.setContextMap(contextMap);
        }
        // 执行任务
        doWork();
    } finally {
        // 清理MDC上下文
        MDC.clear();
    }
});
```

### 6.5 避免嵌套使用线程池

避免在线程池任务中再创建或使用其他线程池，可能导致线程资源竞争和死锁：

```java
// 不推荐
executor1.execute(() -> {
    // 在线程池1中使用线程池2
    executor2.execute(() -> {
        // 任务逻辑
    });
});

// 推荐：使用CompletableFuture组合异步操作
CompletableFuture<Void> future = CompletableFuture
    .runAsync(() -> step1(), executor1)
    .thenRunAsync(() -> step2(), executor2);
```

## 总结

线程池是并发编程的重要工具，根据任务类型是IO密集型还是CPU密集型、CPU核数，合理设置线程池大小、队列大小、拒绝策略，并进行压测和不断调优来决定适合自己场景的参数。遵循本文介绍的最佳实践，可以更高效、更安全地使用线程池，提升系统性能和稳定性。