```java
import java.util.ArrayList;
import java.util.List;

public class MemoryPressureDemo {
    // 静态集合保留部分对象，模拟内存泄漏
    private static final List<byte[]> leakCache = new ArrayList<>();

    public static void main(String[] args) throws InterruptedException {
        int iteration = 0;
        while (true) {
            // 每100次迭代保留一个对象，其余对象在循环结束后被GC回收
            List<byte[]> tempList = new ArrayList<>();
            // 构造10*9M的对象（100M） 暂时不是1G， 这样能够让gc慢慢回收
            for (int i = 0; i < 10; i++) {
                byte[] data = new byte[10 * 1024 * 1024]; // 分配10MB
                if (i == 0) {
                    // leakCache.add(data); // 模拟内存泄漏
                } else {
                    tempList.add(data); // 临时对象
                }
            }
            System.out.println("Iteration: " + (++iteration));
            Thread.sleep(1000); // 稍作停顿，避免CPU满载
        }
    }
}
```