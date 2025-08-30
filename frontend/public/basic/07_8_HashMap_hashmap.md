以下是一个典型错误使用 `HashMap` 的案例，演示因**多线程并发修改**和**键对象可变性**导致的问题：

---

### **案例 1：多线程并发修改导致数据丢失**
```java
import java.util.HashMap;
import java.util.Map;

public class HashMapConcurrencyDemo {
    private static final Map<Integer, String> map = new HashMap<>();

    public static void main(String[] args) throws InterruptedException {
        // 线程1：添加数据
        Thread t1 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) {
                map.put(i, "Value" + i);
            }
        });

        // 线程2：删除数据
        Thread t2 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) {
                map.remove(i);
            }
        });

        t1.start();
        t2.start();
        t1.join();
        t2.join();

        System.out.println("最终Map大小: " + map.size());
        // 预期结果不确定，可能抛出异常、数据丢失，甚至死循环！
    }
}
```

#### **问题分析**
1. **线程不安全**  
   `HashMap` 非线程安全，多线程并发修改（如同时 `put` 和 `remove`）可能导致：
   - **数据丢失**：部分操作被覆盖。
   - **死循环**（JDK 1.7及之前版本）：链表成环，CPU飙升至100%。
   - **异常**：如 `ConcurrentModificationException`。

---

### **案例 2：可变对象作为键导致数据不可达**
```java
import java.util.HashMap;
import java.util.Map;

public class HashMapMutableKeyDemo {
    static class MutableKey {
        private int id;

        public MutableKey(int id) {
            this.id = id;
        }

        public void setId(int id) {
            this.id = id;
        }

        @Override
        public int hashCode() {
            return id;
        }

        @Override
        public boolean equals(Object obj) {
            if (obj instanceof MutableKey) {
                return this.id == ((MutableKey) obj).id;
            }
            return false;
        }
    }

    public static void main(String[] args) {
        Map<MutableKey, String> map = new HashMap<>();
        MutableKey key = new MutableKey(1);
        map.put(key, "Value1");

        // 修改键对象的属性
        key.setId(2);

        // 尝试获取数据，返回null！
        System.out.println(map.get(key));  // 输出：null
    }
}
```

#### **问题分析**
1. **键对象哈希值变化**  
   修改 `MutableKey` 的 `id` 后，其 `hashCode()` 返回值改变，导致 `get()` 时定位到错误的哈希桶。
2. **数据不可达**  
   原始键值对存储在旧的哈希桶中，但通过修改后的键查找时，会访问新的哈希桶，因此无法找到数据。

---

### **解决方案**
#### **1. 多线程场景**
- **使用线程安全的 `ConcurrentHashMap`**：
  ```java
  Map<Integer, String> safeMap = new ConcurrentHashMap<>();
  ```

#### **2. 键对象设计**
- **键对象不可变**（如 `String`、`Integer`）：
  ```java
  public final class ImmutableKey {
      private final int id;

      public ImmutableKey(int id) {
          this.id = id;
      }

      // 无setter方法，确保不可变
  }
  ```

#### **3. 其他注意事项**
- **重写 `hashCode()` 和 `equals()`**：确保键对象的哈希一致性和等价性。
- **避免频繁扩容**：初始化时指定容量（如 `new HashMap<>(1024)`）。

---

### **总结**
| **错误类型**         | **问题现象**               | **解决方案**                     |
|----------------------|---------------------------|----------------------------------|
| 多线程并发修改        | 数据丢失、死循环、异常     | 使用 `ConcurrentHashMap`         |
| 可变对象作为键        | 数据不可达                 | 键对象设计为不可变               |
| 未正确实现哈希/等价性 | 重复键覆盖、检索失败       | 重写 `hashCode()` 和 `equals()`  |

**永远记住**：`HashMap` 是单线程利器，但多线程需谨慎！

