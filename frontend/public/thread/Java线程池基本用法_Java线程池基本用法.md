# Java线程池基本用法

## 1. 线程池概述

线程池是Java并发编程中的重要组件，它可以有效管理和复用线程资源，避免频繁创建和销毁线程带来的性能开销。通过线程池，我们可以控制系统中并发线程的数量，提高系统的稳定性和响应性。

## 2. 线程池核心参数

以下是创建线程池时的核心参数：

```java
ThreadPoolExecutor executor = new ThreadPoolExecutor(
        corePoolSize,     // 核心线程数
        maximumPoolSize,   // 最大线程数
        keepAliveTime,     // 空闲线程存活时间
        timeUnit,          // 时间单位
        workQueue,         // 工作队列
        rejectedExecutionHandler // 拒绝策略
);
```

- **核心线程数（corePoolSize）**：线程池中长期保持活动的线程数量，即使这些线程处于空闲状态。
- **最大线程数（maximumPoolSize）**：线程池允许创建的最大线程数量。
- **空闲线程存活时间（keepAliveTime）**：当线程数大于核心线程数时，多余的空闲线程在终止前等待新任务的最长时间。
- **时间单位（timeUnit）**：keepAliveTime参数的时间单位。
- **工作队列（workQueue）**：用于存放等待执行的任务的阻塞队列。
- **拒绝策略（rejectedExecutionHandler）**：当线程池和队列都满了，新提交的任务将被拒绝，此时会触发拒绝策略。

## 3. 线程池工作原理

线程池处理任务的流程如下：

1. 当提交一个新任务到线程池时，线程池会做如下判断：
   - 如果正在运行的线程数小于核心线程数，那么会创建一个新线程来执行任务。
   - 如果正在运行的线程数大于或等于核心线程数，那么会将任务放入队列中。
   - 如果队列已满，且正在运行的线程数小于最大线程数，那么会创建新的线程来处理任务。
   - 如果队列已满，且正在运行的线程数大于或等于最大线程数，那么会触发拒绝策略。

## 4. 示例代码分析

以下是`ThreadPoolDemo.java`中线程池的创建和使用示例：

```java
// 创建线程池，核心2，最大3，队列容量1，拒绝策略打印信息
ThreadPoolExecutor executor = new ThreadPoolExecutor(
        2, // 核心线程数
        3, // 最大线程数
        1, TimeUnit.SECONDS,
        new ArrayBlockingQueue<>(1), // 队列容量1
        new RejectedExecutionHandler() {
            @Override
            public void rejectedExecution(Runnable r, ThreadPoolExecutor executor) {
                System.out.println("任务被拒绝: " + r.toString());
            }
        });
```

在这个示例中：
- 核心线程数为2，表示线程池会保持2个线程长期运行
- 最大线程数为3，表示线程池最多可以同时运行3个线程
- 队列容量为1，表示当核心线程都在工作时，最多可以有1个任务在队列中等待
- 自定义拒绝策略，当任务被拒绝时打印信息

## 5. 任务提交与拒绝策略触发

```java
// 快速提交多个任务，触发拒绝策略
for (int i = 0; i < 6; i++) {
    try {
        executor.execute(task);
        System.out.println("提交任务 " + (i + 1));
    } catch (RejectedExecutionException e) {
        System.out.println("任务 " + (i + 1) + " 提交时被拒绝");
    }
    // 不等待或只等待很短时间，确保任务快速提交
    if (i < 2) {
        Thread.sleep(10); // 前几个任务几乎同时提交
    }
}
```

在这个示例中，我们快速提交了6个任务，每个任务的处理时间为1秒。由于线程池的配置（核心线程数2，最大线程数3，队列容量1），当提交的任务数超过线程池的处理能力时，会触发拒绝策略：

1. 前2个任务会被核心线程直接处理
2. 第3个任务会被放入队列等待
3. 第4个任务会创建一个新的非核心线程处理（此时达到最大线程数3）
4. 第5个和第6个任务会被拒绝，因为线程池已达到最大线程数，且队列已满

## 6. 线程池关闭

```java
executor.shutdown();
executor.awaitTermination(5, TimeUnit.SECONDS); // 等待所有任务完成
```

- `shutdown()`：平缓关闭线程池，不再接受新任务，但会等待已提交的任务完成
- `awaitTermination()`：等待指定时间让任务完成，如果超时，则返回false

## 7. 常见的拒绝策略

Java提供了四种标准的拒绝策略：

1. **AbortPolicy**：默认策略，直接抛出RejectedExecutionException异常
2. **CallerRunsPolicy**：在调用者线程中执行任务，而不是在线程池中
3. **DiscardPolicy**：直接丢弃任务，不做任何处理
4. **DiscardOldestPolicy**：丢弃队列中最早的任务，然后尝试重新提交新任务

在示例中，我们使用了自定义的拒绝策略，只是简单地打印被拒绝的任务信息。

## 8. 线程池的选择与使用建议

1. **合理设置参数**：根据任务的特性和系统资源合理设置核心线程数、最大线程数和队列容量
2. **选择合适的队列**：不同的队列有不同的特性，如ArrayBlockingQueue（有界队列）、LinkedBlockingQueue（无界队列）等
3. **选择合适的拒绝策略**：根据业务需求选择或自定义拒绝策略
4. **监控线程池状态**：定期检查线程池的运行状态，如活动线程数、队列大小等
5. **及时关闭线程池**：在不需要线程池时，及时关闭以释放资源

## 9. 总结

线程池是Java并发编程中非常重要的工具，通过合理配置线程池参数，可以有效控制系统资源的使用，提高系统的性能和稳定性。在实际应用中，需要根据具体的业务场景和系统资源来选择合适的线程池配置。