### Java线程使用全解析：从基础到高阶实战

---

#### **一、线程基础概念**

##### **1. 线程与进程的区别**
- **进程**：操作系统分配资源的基本单位（如一个运行的Chrome浏览器）。  
- **线程**：进程内的执行单元（如Chrome中同时运行的多个标签页）。  
- **核心区别**：  
  - 进程间内存隔离，线程共享进程内存。  
  - 线程切换成本低，进程切换成本高。  

##### **2. 线程生命周期与状态**  
Java线程通过`Thread.State`枚举定义6种状态：  
```java
public enum State {
    NEW,          // 新建未启动
    RUNNABLE,     // 可运行（可能在运行或等待CPU）
    BLOCKED,      // 阻塞（等待锁）
    WAITING,      // 无限期等待（wait()、join()）
    TIMED_WAITING, // 超时等待（sleep(n)、wait(n)）
    TERMINATED;   // 终止
}
```
**状态转换图**：  
```
NEW → RUNNABLE → (BLOCKED/WAITING/TIMED_WAITING) → TERMINATED
```

---

#### **二、创建线程的4种方式**

##### **1. 继承Thread类**
```java
class MyThread extends Thread {
    @Override
    public void run() {
        System.out.println("线程运行中");
    }
}
// 启动
new MyThread().start();
```

##### **2. 实现Runnable接口（推荐）**
```java
class MyRunnable implements Runnable {
    @Override
    public void run() {
        System.out.println("Runnable线程运行");
    }
}
// 启动
new Thread(new MyRunnable()).start();
```

##### **3. 实现Callable接口（带返回值）**
```java
class MyCallable implements Callable<String> {
    @Override
    public String call() {
        return "Callable结果";
    }
}
// 启动
FutureTask<String> task = new FutureTask<>(new MyCallable());
new Thread(task).start();
System.out.println(task.get()); // 获取结果
```

##### **4. Lambda表达式（Java 8+）**
```java
new Thread(() -> System.out.println("Lambda线程")).start();
```

---

#### **三、线程控制：sleep() vs wait()**

| **对比维度**     | **sleep()**                          | **wait()**                          |
|------------------|--------------------------------------|--------------------------------------|
| **所属类**       | `Thread`静态方法                     | `Object`实例方法                     |
| **锁释放**       | 不释放锁                             | 释放锁                               |
| **唤醒条件**     | 时间到期                             | `notify()`/`notifyAll()`或超时        |
| **使用场景**     | 暂停执行，无需同步块                 | 线程间协作，需在同步块中调用          |

**代码示例**：
```java
synchronized (lock) {
    System.out.println("进入同步块");
    lock.wait(1000); // 释放锁，等待1秒或被唤醒
}

Thread.sleep(1000); // 不释放锁，休眠1秒
```

---

#### **四、线程池的使用原则与实战**

##### **1. 为什么使用线程池？**
- **降低资源消耗**：复用已创建的线程。  
- **提高响应速度**：任务到达时无需等待线程创建。  
- **统一管理**：控制并发数，避免资源耗尽。  

##### **2. 线程池核心参数**
```java
ThreadPoolExecutor executor = new ThreadPoolExecutor(
    corePoolSize,   // 核心线程数（长期保留）
    maxPoolSize,    // 最大线程数（临时线程= max - core）
    keepAliveTime,  // 临时线程空闲存活时间
    TimeUnit.SECONDS,
    workQueue,      // 任务队列（如LinkedBlockingQueue）
    handler         // 拒绝策略（如AbortPolicy）
);
```

##### **3. 四种内置线程池（Executors）**
- **FixedThreadPool**：固定线程数，无界队列（适合稳定负载）。  
- **CachedThreadPool**：线程数无限，适合短时异步任务。  
- **SingleThreadExecutor**：单线程，保证任务顺序执行。  
- **ScheduledThreadPool**：支持定时/周期性任务。  

##### **4. 使用原则**  
- **避免无界队列**：防止内存溢出（OOM）。  
- **合理设置线程数**：  
  - CPU密集型：`核心数 + 1`。  
  - IO密集型：`核心数 * 2`。  
- **自定义拒绝策略**：记录日志或持久化未处理任务。  

---

#### **五、案例分析：多线程文件下载器**

##### **1. 需求场景**  
同时下载多个大文件，提升下载效率，限制最大并发数。

##### **2. 实现方案**  
```java
public class FileDownloader {
    private static final ExecutorService executor = Executors.newFixedThreadPool(4);

    public static void download(List<String> urls) {
        for (String url : urls) {
            executor.submit(() -> {
                // 模拟下载逻辑
                System.out.println("下载中: " + url);
                try {
                    Thread.sleep(2000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("完成: " + url);
            });
        }
        executor.shutdown();
    }

    public static void main(String[] args) {
        download(List.of("file1.zip", "file2.zip", "file3.zip"));
    }
}
```

##### **3. 关键优化点**  
- **线程池限流**：通过`FixedThreadPool`控制并发数。  
- **异常处理**：添加`try-catch`防止线程因异常退出。  
- **资源释放**：调用`shutdown()`优雅关闭线程池。  

---

#### **六、高阶主题：线程安全与锁优化**

##### **1. 线程安全问题**
- **竞态条件**：多个线程修改共享变量（如计数器）。  
- **解决方案**：  
  - **synchronized**：同步代码块或方法。  
  - **ReentrantLock**：更灵活的锁机制。  
  - **Atomic类**：如`AtomicInteger`（CAS无锁优化）。  

##### **2. 死锁与排查**
**死锁条件**：互斥、请求与保持、不可剥夺、循环等待。  
**排查工具**：  
- **jstack**：生成线程转储文件，分析锁持有情况。  
- **Arthas**：`thread -b` 直接定位死锁。  

---

#### **七、总结与最佳实践**

1. **优先选择Runnable/Callable**：避免单继承限制，支持返回值。  
2. **线程池替代裸线程**：提升性能，便于资源管理。  
3. **锁粒度最小化**：减少同步代码块范围。  
4. **监控与调优**：定期检查线程池状态，调整核心参数。  

**示例问题自查清单**：  
- 是否有未关闭的线程池？  
- 是否合理处理了线程异常？  
- 高并发下是否存在竞态条件？  

通过系统学习与实战，Java线程将成为你构建高性能应用的利器！