以下是关于 **CyclicBarrier** 的详细案例说明，涵盖多线程协作、分阶段任务处理等典型场景，结合代码实现和原理解析：

---

### 一、CyclicBarrier 核心特性
1. **可重复使用**：与 `CountDownLatch` 不同，屏障可重置后重复使用。
2. **多阶段同步**：支持线程分批次到达屏障点后触发统一动作。
3. **回调函数**：所有线程到达屏障后，可执行指定 `Runnable` 任务（如汇总结果）。

---

### 二、真实案例场景与代码实现

#### 案例 1：多线程分阶段数据处理（如数据清洗 → 转换 → 入库）
**场景描述**：  
批量数据处理分为三个阶段：清洗、转换、入库，每个阶段需要所有线程完成后才能进入下一阶段。

```java
public class MultiStageDataProcessing {
    private static final int THREAD_COUNT = 3;
    private static final CyclicBarrier barrier = new CyclicBarrier(THREAD_COUNT, () -> {
        System.out.println("\n===== 阶段完成，进入下一阶段 =====");
    });

    public static void main(String[] args) {
        ExecutorService executor = Executors.newFixedThreadPool(THREAD_COUNT);
        for (int i = 0; i < THREAD_COUNT; i++) {
            executor.execute(new DataProcessor("线程-" + (i + 1)));
        }
        executor.shutdown();
    }

    static class DataProcessor implements Runnable {
        private final String name;

        public DataProcessor(String name) {
            this.name = name;
        }

        @Override
        public void run() {
            try {
                // 阶段1：数据清洗
                System.out.println(name + " 完成数据清洗");
                barrier.await();

                // 阶段2：数据转换
                System.out.println(name + " 完成数据转换");
                barrier.await();

                // 阶段3：数据入库
                System.out.println(name + " 完成数据入库");
                barrier.await();
            } catch (InterruptedException | BrokenBarrierException e) {
                e.printStackTrace();
            }
        }
    }
}
```

**输出示例**：
```plaintext
线程-1 完成数据清洗
线程-2 完成数据清洗
线程-3 完成数据清洗

===== 阶段完成，进入下一阶段 =====
线程-1 完成数据转换
线程-3 完成数据转换
线程-2 完成数据转换

===== 阶段完成，进入下一阶段 =====
线程-1 完成数据入库
线程-2 完成数据入库
线程-3 完成数据入库

===== 阶段完成，进入下一阶段 =====
```

**关键设计**：
- **阶段同步**：每个线程完成当前阶段后调用 `await()`，所有线程到达屏障后触发回调提示。
- **线程池复用**：固定线程池大小与屏障阈值一致，避免线程不足导致死锁。

---

#### 案例 2：分布式任务协同（如多服务准备就绪后触发全局操作）
**场景描述**：  
系统包含多个微服务（如支付服务、库存服务、物流服务），需等待所有服务就绪后触发全局事务提交。

```java
public class DistributedTransactionCoordinator {
    private static final int SERVICE_COUNT = 3;
    private static final CyclicBarrier barrier = new CyclicBarrier(SERVICE_COUNT, () -> {
        System.out.println("\n===== 所有服务准备就绪，提交全局事务 =====");
    });

    public static void main(String[] args) {
        ExecutorService executor = Executors.newCachedThreadPool();
        executor.execute(new Service("支付服务", 2000));
        executor.execute(new Service("库存服务", 1500));
        executor.execute(new Service("物流服务", 3000));
        executor.shutdown();
    }

    static class Service implements Runnable {
        private final String name;
        private final int prepareTime;

        public Service(String name, int prepareTime) {
            this.name = name;
            this.prepareTime = prepareTime;
        }

        @Override
        public void run() {
            try {
                Thread.sleep(prepareTime); // 模拟服务准备耗时
                System.out.println(name + " 准备完成");
                barrier.await(); // 等待其他服务
                System.out.println(name + " 提交本地事务");
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
```

**输出示例**：
```plaintext
库存服务 准备完成
支付服务 准备完成
物流服务 准备完成

===== 所有服务准备就绪，提交全局事务 =====
物流服务 提交本地事务
支付服务 提交本地事务
库存服务 提交本地事务
```

**关键设计**：
- **异构耗时处理**：不同服务准备时间不同，屏障确保最长等待时间的任务完成后触发全局提交。
- **事务协调**：适用于分布式事务的 `2PC`（两阶段提交）场景。

---

#### 案例 3：并行计算与结果合并（如 MapReduce 模型）
**场景描述**：  
将数据分片并行处理（Map阶段），所有分片处理完成后合并结果（Reduce阶段）。

```java
public class MapReduceDemo {
    private static final int PARTITIONS = 4;
    private static final CyclicBarrier barrier = new CyclicBarrier(PARTITIONS, () -> {
        System.out.println("\n===== Map阶段完成，开始Reduce操作 =====");
        // 模拟Reduce操作（如统计词频总和）
    });

    public static void main(String[] args) {
        ExecutorService executor = Executors.newFixedThreadPool(PARTITIONS);
        for (int i = 0; i < PARTITIONS; i++) {
            executor.execute(new MapTask("分片-" + (i + 1)));
        }
        executor.shutdown();
    }

    static class MapTask implements Runnable {
        private final String partition;

        public MapTask(String partition) {
            this.partition = partition;
        }

        @Override
        public void run() {
            try {
                // 模拟Map阶段处理
                System.out.println(partition + " 完成Map计算");
                barrier.await(); // 等待所有分片完成
                // 可在此继续下一阶段操作（如二次Map-Reduce）
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
```

**输出示例**：
```plaintext
分片-1 完成Map计算
分片-2 完成Map计算
分片-3 完成Map计算
分片-4 完成Map计算

===== Map阶段完成，开始Reduce操作 =====
```

---

### 三、CyclicBarrier 原理解析
1. **核心机制**：
   - 内部通过 `ReentrantLock` 和 `Condition` 实现线程等待。
   - 每调用一次 `await()`，计数器减1，当计数器归零时，唤醒所有等待线程并重置计数器。

2. **重要方法**：
   - `await()`：阻塞当前线程，直到所有线程到达屏障。
   - `reset()`：强制重置屏障，导致正在等待的线程抛出 `BrokenBarrierException`。

3. **异常处理**：
   - 若线程在等待时被中断，屏障会进入 `broken` 状态，其他线程将抛出 `BrokenBarrierException`。
   - 可通过 `isBroken()` 检查屏障状态，必要时调用 `reset()` 恢复。

---

### 四、CyclicBarrier vs CountDownLatch
| **特性**               | **CyclicBarrier**                          | **CountDownLatch**                  |
|------------------------|--------------------------------------------|--------------------------------------|
| 重用性                 | ✅ 支持多次重置                             | ❌ 一次性使用                         |
| 触发动作               | 所有线程到达后执行回调                     | 无回调，仅等待计数器归零             |
| 线程角色               | 所有线程对等，互相等待                     | 主线程等待子线程完成（单向依赖）     |
| 适用场景               | 多阶段协同、复杂并行计算                   | 单次事件同步（如服务启动、任务完成） |

---

### 五、最佳实践与注意事项
1. **线程数匹配**：确保屏障的 `parties` 参数与实际线程数一致，否则会永久阻塞。
2. **超时控制**：使用 `await(long timeout, TimeUnit unit)` 避免无限期等待。
3. **异常恢复**：捕获 `BrokenBarrierException` 后决定是否重置屏障或终止任务。
4. **性能优化**：避免在回调函数中执行耗时操作，防止阻塞线程唤醒。

---

通过以上案例和解析，可深入理解 `CyclicBarrier` 在多线程分阶段协作中的实际应用价值。