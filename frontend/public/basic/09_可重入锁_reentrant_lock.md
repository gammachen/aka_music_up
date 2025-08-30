### ReentrantLock 原理与 synchronized 的区别详解

---

#### **一、概述**
在 Java 并发编程中，`ReentrantLock` 和 `synchronized` 是两种核心的线程同步机制，用于控制共享资源的访问。虽然它们都实现了互斥访问，但底层实现和功能特性存在显著差异。本文将从 **原理、核心特性、性能对比** 等方面详细分析两者的区别，并给出实际应用建议。

---

#### **二、ReentrantLock 的原理**
**ReentrantLock** 是 Java 并发包（`java.util.concurrent.locks`）中提供的可重入互斥锁，基于 **AQS（AbstractQueuedSynchronizer）** 实现。

##### **1. 核心机制**
- **AQS 框架**：ReentrantLock 通过 AQS 的 FIFO 队列管理线程的等待队列，所有锁的获取和释放操作均通过 AQS 的状态（`state`）控制。
- **CAS 操作**：非公平锁模式下，ReentrantLock 使用 **CAS（Compare and Swap）** 原子操作尝试快速获取锁，无需进入等待队列。
- **可重入性**：通过 `state` 记录锁的持有次数，线程可重复获取同一锁，直到 `unlock()` 递减 `state` 至 0 时完全释放。

##### **2. 核心方法**
```java
// 非公平锁尝试获取锁（简化版）
final void lock() {
    if (compareAndSetState(0, 1)) { // CAS 尝试获取锁
        setExclusiveOwnerThread(Thread.currentThread());
    } else {
        acquire(1); // 进入 AQS 队列竞争
    }
}

// 公平锁尝试获取锁（简化版）
final void lock() {
    if (isHeldExclusively()) { // 可重入检查
        incrementHoldCount();
        return;
    }
    if (!hasQueuedPredecessors() && compareAndSetState(0, 1)) { // 公平性检查
        setExclusiveOwnerThread(current);
        return;
    }
    acquire(1);
}
```

---

#### **三、synchronized 的原理**
**synchronized** 是 Java 的内置关键字，基于 **Monitor 对象** 实现，通过 JVM 的 **字节码指令**（`monitorenter` 和 `monitorexit`）管理锁。

##### **1. 核心机制**
- **Monitor 对象**：每个 Java 对象在对象头中包含一个指向 Monitor 对象的指针。当线程进入 `synchronized` 块时，会尝试获取该对象的 Monitor 锁。
- **锁升级机制**：
  - **偏向锁**：默认尝试标记对象头为偏向某个线程，避免同步开销。
  - **轻量级锁**：通过 CAS 操作竞争锁，失败后升级为重量级锁。
  - **重量级锁**：通过操作系统互斥量（Mutex）阻塞线程。
- **自动管理**：锁的获取和释放由 JVM 自动完成，无需手动干预。

##### **2. 字节码示例**
```java
synchronized void method() {
    // 同步代码块
}
// 对应的字节码：
monitorenter
...
monitorexit
```

---

#### **四、ReentrantLock 与 synchronized 的核心区别**

| **特性**               | **ReentrantLock**                                                                 | **synchronized**                                                                 |
|------------------------|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| **锁的获取与释放**      | **显式**：需手动调用 `lock()` 和 `unlock()`。                                      | **隐式**：自动获取和释放锁（通过 `monitorenter`/`monitorexit`）。               |
| **可中断性**           | 支持：`lockInterruptibly()` 可中断等待锁的线程。                                   | 不支持：等待锁的线程无法中断，除非抛出异常。                                    |
| **尝试获取锁**         | 支持：`tryLock()` 可立即返回锁获取结果；`tryLock(long, TimeUnit)` 可设置超时。    | 不支持：必须等待锁释放。                                                        |
| **公平性**             | 可选：通过构造函数 `new ReentrantLock(true)` 设置公平锁。                          | 默认非公平，但 JVM 通过锁升级优化（如偏向锁）实现部分公平性。                    |
| **性能**               | 高并发下性能更优，非公平锁通过 CAS 减少线程阻塞。                                  | 低竞争时性能较好（偏向锁、轻量级锁优化），高竞争时可能因重量级锁性能下降。       |
| **异常安全性**         | **需手动释放锁**：必须在 `finally` 块中调用 `unlock()`，否则可能导致死锁。         | **自动释放锁**：即使发生异常，锁也会自动释放。                                  |
| **条件变量（Condition）** | 支持：通过 `newCondition()` 绑定多个 `Condition`，实现精准唤醒。                  | 依赖 `Object` 的 `wait()/notify()`，只能唤醒全部或随机线程。                    |
| **适用场景**           | 复杂场景：需中断、超时、多条件变量控制。                                           | 简单场景：代码块短小，无需高级控制。                                             |

---

#### **五、核心特性对比详解**

##### **1. 锁的获取与释放**
- **ReentrantLock**：
  - 需手动管理锁，例如：
    ```java
    lock.lock();
    try {
        // 临界区代码
    } finally {
        lock.unlock();
    }
    ```
  - **风险**：若忘记调用 `unlock()`，可能导致死锁。
- **synchronized**：
  - 自动管理锁，无需手动干预：
    ```java
    synchronized (obj) {
        // 临界区代码
    }
    ```

##### **2. 公平性**
- **ReentrantLock**：
  - **公平锁**：严格按线程请求顺序分配锁。
  - **非公平锁（默认）**：允许插队，吞吐量更高。
- **synchronized**：
  - 默认非公平，但通过 **偏向锁** 和 **自旋锁** 优化，减少饥饿问题。

##### **3. 可中断性**
- **ReentrantLock**：
  - `lockInterruptibly()` 允许线程在等待锁时响应中断。
  - 示例：
    ```java
    try {
        lock.lockInterruptibly();
    } catch (InterruptedException e) {
        // 处理中断
    }
    ```
- **synchronized**：
  - 无法中断等待锁的线程，除非抛出异常。

##### **4. 条件变量（Condition）**
- **ReentrantLock**：
  - 支持绑定多个 `Condition` 对象，实现精准唤醒：
    ```java
    Condition condition = lock.newCondition();
    lock.lock();
    try {
        condition.await(); // 等待条件
        condition.signal(); // 唤醒符合条件的线程
    } finally {
        lock.unlock();
    }
    ```
- **synchronized**：
  - 依赖 `Object` 的 `wait()`/`notify()`，无法精准控制唤醒目标。

##### **5. 性能对比**
- **ReentrantLock**：
  - 非公平锁通过 **CAS** 快速获取锁，减少线程阻塞。
  - 在高竞争场景下性能更优。
- **synchronized**：
  - 低竞争时性能更好（偏向锁、轻量级锁优化）。
  - 高竞争时可能因升级为重量级锁导致性能下降。

---

#### **六、代码示例对比**

##### **1. 基础用法**
```java
// ReentrantLock
ReentrantLock lock = new ReentrantLock();
lock.lock();
try {
    // 临界区
} finally {
    lock.unlock();
}

// synchronized
synchronized (obj) {
    // 临界区
}
```

##### **2. 公平性设置**
```java
// ReentrantLock 公平锁
ReentrantLock fairLock = new ReentrantLock(true);

// synchronized 无法设置公平性
```

##### **3. 可中断锁**
```java
// ReentrantLock 可中断
lock.lockInterruptibly();

// synchronized 不支持
```

---

#### **七、选择建议**
| **场景**                          | **推荐方案**           | **理由**                                                                 |
|-----------------------------------|-----------------------|-------------------------------------------------------------------------|
| 简单同步，代码块短小               | `synchronized`        | 自动管理锁，代码简洁，JVM 优化性能较好。                                  |
| 高并发场景，需优化吞吐量           | `ReentrantLock`       | 非公平锁性能更优，支持 CAS 快速获取锁。                                   |
| 需要中断等待锁的线程               | `ReentrantLock`       | `lockInterruptibly()` 提供中断支持。                                     |
| 需要多条件变量控制（如生产者-消费者）| `ReentrantLock`       | `Condition` 实现精准唤醒，避免唤醒全部线程。                             |
| 需要避免死锁风险                   | `synchronized`        | 自动释放锁，无需手动管理。                                               |

---

#### **八、总结**
- **ReentrantLock** 是 **显式、灵活、高性能** 的锁机制，适合复杂场景（如需要中断、超时、多条件变量）。
- **synchronized** 是 **隐式、简洁、低开销** 的锁机制，适合简单同步需求。
- **关键权衡**：显式锁的灵活性与隐式锁的简洁性之间的取舍，需根据实际场景选择。

通过理解两者的底层原理和特性差异，开发者可以更合理地设计并发程序，平衡性能与代码复杂度。