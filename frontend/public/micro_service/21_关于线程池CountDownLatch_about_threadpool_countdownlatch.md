# CountdownLatch与CyclicBarrier详解

## 1. CountdownLatch（倒计时门闩）

### 1.1 基本概念

CountdownLatch是一个同步辅助类，允许一个或多个线程等待，直到其他线程完成一组操作。它的工作原理是：初始化时设定一个计数值，当调用`countDown()`方法时，计数值减1，当计数值减为0时，所有等待的线程（调用了`await()`方法的线程）将被释放继续执行。

CountdownLatch的主要特点：
- 计数器初始化后无法重置，使用一次后即被销毁
- 可以实现一个或多个线程等待其他线程完成
- 提供超时等待机制

### 1.2 主要方法

```java
// 创建实例时指定计数值
CountDownLatch latch = new CountDownLatch(count);

// 使计数器减1
latch.countDown();

// 等待计数器变为0，会阻塞当前线程
latch.await();

// 等待指定时间，如果超时则返回false
latch.await(timeout, TimeUnit.SECONDS);
```

### 1.3 典型使用场景

#### 场景一：启动信号

确保所有线程同时开始执行任务。主线程创建CountdownLatch(1)，所有工作线程await()等待，主线程准备就绪后countDown()，所有工作线程同时开始执行。

```java
public class RaceStartExample {
    public static void main(String[] args) throws InterruptedException {
        int numOfThreads = 5;
        CountDownLatch startSignal = new CountDownLatch(1);
        
        for (int i = 0; i < numOfThreads; i++) {
            final int threadId = i;
            new Thread(() -> {
                try {
                    System.out.println("线程" + threadId + "准备就绪，等待发令枪...");
                    startSignal.await();
                    System.out.println("线程" + threadId + "开始执行任务");
                    // 执行任务
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }).start();
        }
        
        // 确保所有线程都准备就绪
        Thread.sleep(1000);
        System.out.println("发令枪响，所有线程开始执行！");
        startSignal.countDown();
    }
}
```

#### 场景二：完成信号

主线程等待所有工作线程完成任务。主线程创建CountdownLatch(N)，工作线程执行完任务后countDown()，主线程await()等待所有工作线程完成。

```java
public class TaskCompletionExample {
    public static void main(String[] args) throws InterruptedException {
        int numOfThreads = 5;
        CountDownLatch finishSignal = new CountDownLatch(numOfThreads);
        
        for (int i = 0; i < numOfThreads; i++) {
            final int threadId = i;
            new Thread(() -> {
                try {
                    System.out.println("线程" + threadId + "开始执行任务");
                    // 模拟任务执行时间
                    Thread.sleep((long) (Math.random() * 1000));
                    System.out.println("线程" + threadId + "完成任务");
                    finishSignal.countDown();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }).start();
        }
        
        System.out.println("主线程等待所有任务完成...");
        finishSignal.await();
        System.out.println("所有任务已完成，主线程继续执行");
    }
}
```

### 1.4 实际应用案例

#### 案例：多服务依赖检查

在微服务架构中，一个服务可能依赖多个其他服务。在启动时需要确保所有依赖的服务都已就绪才能正常运行。

```java
public class ServiceDependencyChecker {
    private final List<String> dependentServices;
    private final CountDownLatch dependencyLatch;
    
    public ServiceDependencyChecker(List<String> services) {
        this.dependentServices = services;
        this.dependencyLatch = new CountDownLatch(services.size());
    }
    
    public void startService() {
        // 启动检查线程
        for (String service : dependentServices) {
            new Thread(() -> checkServiceAvailability(service)).start();
        }
        
        try {
            System.out.println("等待所有依赖服务就绪...");
            // 等待所有依赖服务就绪，最多等待2分钟
            boolean allServicesReady = dependencyLatch.await(2, TimeUnit.MINUTES);
            
            if (allServicesReady) {
                System.out.println("所有依赖服务已就绪，启动应用服务");
                // 启动应用服务
                startApplicationService();
            } else {
                System.out.println("部分服务未就绪，启动失败");
                // 处理启动失败情况
                handleStartupFailure();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.out.println("等待依赖服务过程被中断");
        }
    }
    
    private void checkServiceAvailability(String serviceName) {
        try {
            System.out.println("检查服务: " + serviceName);
            // 实际实现中，这里可能是HTTP请求或RPC调用来检查服务状态
            boolean isAvailable = checkServiceStatus(serviceName);
            
            if (isAvailable) {
                System.out.println("服务 " + serviceName + " 已就绪");
                dependencyLatch.countDown();
            } else {
                System.out.println("服务 " + serviceName + " 未就绪，进行重试");
                // 重试逻辑
                retryCheck(serviceName);
            }
        } catch (Exception e) {
            System.out.println("检查服务 " + serviceName + " 时发生异常: " + e.getMessage());
        }
    }
    
    private boolean checkServiceStatus(String serviceName) {
        // 模拟服务检查
        try {
            Thread.sleep((long) (Math.random() * 1000));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        return Math.random() > 0.3; // 70%概率服务可用
    }
    
    private void retryCheck(String serviceName) {
        // 实现重试逻辑
        new Thread(() -> {
            try {
                Thread.sleep(5000); // 等待5秒后重试
                checkServiceAvailability(serviceName);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }).start();
    }
    
    private void startApplicationService() {
        // 启动应用服务的逻辑
        System.out.println("应用服务启动成功");
    }
    
    private void handleStartupFailure() {
        // 处理启动失败的逻辑
        System.out.println("应用启动失败，记录日志并通知管理员");
    }
}
```

## 2. CyclicBarrier（循环栅栏）

### 2.1 基本概念

CyclicBarrier是一个同步辅助类，允许一组线程相互等待，直到所有线程都到达一个共同的屏障点。它的工作原理是：初始化时设定一个参与方的数量，当调用`await()`方法时，当前线程会被阻塞，直到所有参与方都调用了`await()`，屏障才会打开，所有线程才能继续执行。

CyclicBarrier的主要特点：
- 可以重用，当所有线程都到达屏障点后，屏障会重置
- 可以设置屏障动作，当所有线程到达屏障点时执行
- 提供超时等待机制

### 2.2 主要方法

```java
// 创建实例时指定参与方数量
CyclicBarrier barrier = new CyclicBarrier(parties);

// 创建实例并指定屏障动作
CyclicBarrier barrier = new CyclicBarrier(parties, barrierAction);

// 等待所有参与方到达屏障点
barrier.await();

// 等待指定时间，如果超时则抛出异常
barrier.await(timeout, TimeUnit.SECONDS);

// 重置屏障
barrier.reset();
```

### 2.3 典型使用场景

#### 场景一：并行计算

在并行计算中，一个大任务被分解为多个小任务，每个小任务由一个线程处理，所有线程处理完当前阶段后再一起进入下一阶段。

```java
public class ParallelComputationExample {
    public static void main(String[] args) {
        int numOfThreads = 3;
        int numOfPhases = 3;
        CyclicBarrier barrier = new CyclicBarrier(numOfThreads, () -> {
            // 每个阶段完成后执行的操作
            System.out.println("======= 所有线程完成当前阶段，进入下一阶段 =======");
        });
        
        for (int i = 0; i < numOfThreads; i++) {
            final int threadId = i;
            new Thread(() -> {
                try {
                    for (int phase = 1; phase <= numOfPhases; phase++) {
                        System.out.println("线程" + threadId + "开始执行第" + phase + "阶段");
                        // 模拟计算任务
                        Thread.sleep((long) (Math.random() * 1000));
                        System.out.println("线程" + threadId + "完成第" + phase + "阶段，等待其他线程");
                        
                        // 等待所有线程完成当前阶段
                        barrier.await();
                    }
                } catch (InterruptedException | BrokenBarrierException e) {
                    Thread.currentThread().interrupt();
                }
            }).start();
        }
    }
}
```

#### 场景二：游戏匹配

在游戏服务器中，等待足够多的玩家加入一个游戏会话后才开始游戏。

```java
public class GameMatchingExample {
    private static final int REQUIRED_PLAYERS = 4;
    
    public static void main(String[] args) {
        CyclicBarrier gameBarrier = new CyclicBarrier(REQUIRED_PLAYERS, () -> {
            System.out.println("游戏开始！4人组队已满");
            startGame();
        });
        
        // 模拟玩家陆续加入
        for (int i = 1; i <= 8; i++) {
            final int playerId = i;
            new Thread(() -> {
                try {
                    // 模拟玩家匹配延迟
                    Thread.sleep((long) (Math.random() * 2000));
                    System.out.println("玩家" + playerId + "加入匹配队列");
                    
                    try {
                        int arrivalIndex = gameBarrier.await();
                        System.out.println("玩家" + playerId + "已准备，是第" + (REQUIRED_PLAYERS - arrivalIndex) + "个准备好的玩家");
                    } catch (BrokenBarrierException e) {
                        System.out.println("玩家" + playerId + "匹配失败，有玩家退出");
                    }
                } catch (InterruptedException e) {
                    System.out.println("玩家" + playerId + "取消匹配");
                    Thread.currentThread().interrupt();
                }
            }).start();
        }
    }
    
    private static void startGame() {
        System.out.println("创建游戏房间，分配服务器资源，初始化游戏环境");
    }
}
```

### 2.4 实际应用案例

#### 案例：银行交易日终结算

在银行系统中，日终结算需要多个子系统完成各自的处理工作，然后统一进行最终结算。

```java
public class BankSettlementSystem {
    private static final int SUBSYSTEM_COUNT = 4;
    private final ExecutorService executor = Executors.newFixedThreadPool(SUBSYSTEM_COUNT);
    private final CyclicBarrier settlementBarrier;
    
    public BankSettlementSystem() {
        settlementBarrier = new CyclicBarrier(SUBSYSTEM_COUNT, this::finalSettlement);
    }
    
    public void startDailySettlement() {
        System.out.println("开始日终结算流程");
        
        // 启动各子系统结算任务
        executor.submit(() -> processSubsystem("核心账务系统"));
        executor.submit(() -> processSubsystem("信贷系统"));
        executor.submit(() -> processSubsystem("支付系统"));
        executor.submit(() -> processSubsystem("票据系统"));
    }
    
    private void processSubsystem(String subsystemName) {
        try {
            System.out.println(subsystemName + " 开始处理...");
            
            // 第一阶段：数据准备
            System.out.println(subsystemName + " 正在准备数据...");
            Thread.sleep((long) (Math.random() * 1000 + 1000));
            System.out.println(subsystemName + " 数据准备完成");
            settlementBarrier.await();
            
            // 第二阶段：交易对账
            System.out.println(subsystemName + " 正在交易对账...");
            Thread.sleep((long) (Math.random() * 1000 + 1000));
            System.out.println(subsystemName + " 交易对账完成");
            settlementBarrier.await();
            
            // 第三阶段：清算处理
            System.out.println(subsystemName + " 正在清算处理...");
            Thread.sleep((long) (Math.random() * 1000 + 1000));
            System.out.println(subsystemName + " 清算处理完成");
            settlementBarrier.await();
            
        } catch (InterruptedException | BrokenBarrierException e) {
            System.out.println(subsystemName + " 处理异常: " + e.getMessage());
            Thread.currentThread().interrupt();
        }
    }
    
    private void finalSettlement() {
        System.out.println("\n==== 所有子系统处理完毕，开始最终结算 ====\n");
        try {
            // 模拟最终结算过程
            Thread.sleep(1000);
            System.out.println("生成结算报表...");
            Thread.sleep(500);
            System.out.println("更新总账...");
            Thread.sleep(500);
            System.out.println("结算完成，锁定账务系统");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
    
    public void shutdown() {
        executor.shutdown();
    }
    
    public static void main(String[] args) {
        BankSettlementSystem settlementSystem = new BankSettlementSystem();
        settlementSystem.startDailySettlement();
        
        // 等待结算完成后关闭线程池
        new Thread(() -> {
            try {
                Thread.sleep(10000); // 给足够时间完成结算
                settlementSystem.shutdown();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }).start();
    }
}
```

## 3. CountDownLatch与CyclicBarrier的比较

| 特性 | CountDownLatch | CyclicBarrier |
|------|----------------|---------------|
| 可重用性 | 不可重用，计数为0后不能重置 | 可重用，自动重置 |
| 使用方式 | 一个或多个线程等待其他线程 | 多个线程相互等待 |
| 计数方式 | 通过countDown()方法显式减少计数 | 通过await()方法自动减少计数 |
| 执行动作 | 无内置屏障动作 | 可以定义屏障动作 |
| 异常处理 | 不会因为线程异常而破坏 | 如果有线程发生异常，会导致屏障被破坏 |
| 超时处理 | await()可以指定超时时间 | await()可以指定超时时间 |
| 使用场景 | 适合一个线程等待多个线程完成任务 | 适合多个线程相互等待，共同完成工作 |

## 4. 使用时的注意事项

### 4.1 CountDownLatch注意事项

1. **计数器不可重置**：一旦计数器减到0，就不能再重置。如果需要重新计数，必须创建新的CountDownLatch实例。
2. **死锁风险**：如果某些线程未能调用countDown()，等待的线程将永远阻塞。应考虑使用带超时的await()方法。
3. **异常处理**：确保在finally块中调用countDown()，以防止因异常导致计数器无法归零。

```java
// 推荐的使用方式
try {
    // 执行任务
} catch (Exception e) {
    // 处理异常
} finally {
    latch.countDown(); // 确保计数器减1
}
```

### 4.2 CyclicBarrier注意事项

1. **屏障被破坏**：如果任意线程在等待过程中被中断，或者调用reset()方法，屏障就会被破坏，所有等待的线程会抛出BrokenBarrierException。
2. **死锁风险**：如果参与线程数量少于创建CyclicBarrier时指定的数量，所有线程将永远等待。应考虑使用带超时的await()方法。
3. **性能考虑**：屏障动作是在最后一个到达的线程中执行的，如果屏障动作耗时较长，会延迟所有线程的继续执行。

```java
// 处理BrokenBarrierException的方式
try {
    barrier.await();
} catch (InterruptedException e) {
    Thread.currentThread().interrupt();
    // 处理中断
} catch (BrokenBarrierException e) {
    // 处理屏障被破坏的情况
    // 可能需要重新协调线程或退出当前操作
}
```

## 5. 最佳实践

### 5.1 何时选择CountDownLatch

1. 线程启动信号：一个主线程控制多个工作线程的启动
2. 任务完成信号：等待一组任务全部完成
3. 资源初始化：等待所有必要资源都初始化完成
4. 一次性等待场景：只需要等待一次，不需要重复使用

### 5.2 何时选择CyclicBarrier

1. 分阶段计算：多个线程完成第一阶段后再一起进入第二阶段
2. 并行数据处理：需要等待所有线程处理完当前批次数据后再处理下一批次
3. 需要在每个同步点执行某些操作：利用CyclicBarrier的屏障动作
4. 需要重复使用同步点的场景：比如迭代算法中的多次同步

### 5.3 组合使用

在一些复杂场景中，可以组合使用CountDownLatch和CyclicBarrier：

```java
public class CombinedExample {
    public static void main(String[] args) throws InterruptedException {
        int numOfThreads = 3;
        int numOfPhases = 2;
        
        // 用于等待所有线程完成全部工作
        CountDownLatch completionLatch = new CountDownLatch(numOfThreads);
        
        // 用于各阶段的同步
        CyclicBarrier phaseBarrier = new CyclicBarrier(numOfThreads, () -> {
            System.out.println("=== 所有线程完成当前阶段 ===");
        });
        
        for (int i = 0; i < numOfThreads; i++) {
            final int threadId = i;
            new Thread(() -> {
                try {
                    for (int phase = 0; phase < numOfPhases; phase++) {
                        System.out.println("线程" + threadId + "执行阶段" + phase);
                        Thread.sleep((long) (Math.random() * 1000));
                        
                        // 等待所有线程完成当前阶段
                        phaseBarrier.await();
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                } finally {
                    // 标记该线程已完成所有工作
                    completionLatch.countDown();
                }
            }).start();
        }
        
        // 主线程等待所有工作线程完成
        System.out.println("主线程等待所有工作完成...");
        completionLatch.await();
        System.out.println("所有线程已完成所有阶段的工作");
    }
}
```

在这个例子中，我们使用CyclicBarrier来同步各个阶段的工作，使用CountDownLatch来通知主线程所有工作都已完成。这种组合使用可以应对更复杂的同步需求。