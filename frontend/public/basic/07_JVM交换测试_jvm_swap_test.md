```java
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

```bash
(base) shaofu@shaofu:~$ vmstat -t 1
procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu----- -----timestamp-----
 r  b 交换 空闲 缓冲 缓存   si   so    bi    bo   in   cs us sy id wa st                 CST
 1  0 380728 5065328  17448 256288    1   86    41   121   40   39  0  0 97  2  0 2025-04-24 17:50:04
 1  0 380728 2801864  17448 256288    0    0     0     0 1192  238  7 19 74  0  0 2025-04-24 17:50:05
 1  0 380728 584264  17448 256288    0    0     0     0 1184  290  6 19 74  0  0 2025-04-24 17:50:06
 0  3 451576 104480  11620 226676    0 71472     0 71592 8354 2056  3 15 48 34  0 2025-04-24 17:50:07
 1  2 586912 104628  11652 187852    0 136320    28 136832 11133 2143  2  7 35 56  0 2025-04-24 17:50:09
 0  3 673224 109136  11652 185488    0 87292   640 87472 8946 1020  2 10 48 39  0 2025-04-24 17:50:10
 1  3 727724 126376  11652 186784   68 55304  1364 55304 2050  725  1  6 23 70  0 2025-04-24 17:50:11
 0  4 776916 140508  11652 186976   96 49680   508 49680 2838  751  3  7  7 83  0 2025-04-24 17:50:12
 0  2 790028 152936  11648 187296 2860 14344  3432 14344  960 1288  2  3 23 72  0 2025-04-24 17:50:13
 0  1 790028 150920  11656 187888  964    0  1556   152  695 1085  2  1 41 56  0 2025-04-24 17:50:14
 0  1 790028 146888  11656 190116 1288    0  3516     0  565  854  1  1 73 25  0 2025-04-24 17:50:15
 0  1 790028 146384  11656 190116 1512    0  1512    12  625  884  1  1 74 24  0 2025-04-24 17:50:16
 ```

```python
(base) shaofu@shaofu:~$ cat allocate_big_memory.py
import sys
import time

def memory_hog():
    """
    持续分配内存，直到物理内存耗尽，触发Swap空间使用。
    """
    data = []  # 存储大对象的列表
    chunk_size = 7400 * 1024 * 1024  # 每次分配1MB（字节）


# 创建一个1MB的字节对象
    byte_array = bytearray(chunk_size)
    data.append(byte_array)

    # 打印当前内存使用情况
    # current_usage = sys.getsizeof(data) // (1024 * 1024)
    current_usage = len(data) * chunk_size // (1024 * 1024)
    print(f"Allocated {current_usage} MB, Total: {len(data)} chunks")

    # 每秒检查一次内存
    time.sleep(1)
    while True:
        try:
            # 打印当前内存使用情况
            # current_usage = sys.getsizeof(data) // (1024 * 1024)
            current_usage = len(data) * chunk_size // (1024 * 1024)
            print(f"Allocated {current_usage} MB, Total: {len(data)} chunks")

            # 每秒检查一次内存
            time.sleep(1)
        except MemoryError:
            print("MemoryError: Out of memory! Continuing...")
            time.sleep(1)
            continue

if __name__ == "__main__":
    print("Starting memory hog process...")
    print("WARNING: This will consume all available memory and Swap space.")
    print("Run this only in a controlled environment (e.g., VM).")
    memory_hog()
```

javac MemoryPressureDemo.java

java -Xmx1g -Xms1g -XX:+PrintGCDetails -Xloggc:gc.log MemoryPressureDemo

# 查找Java进程PID
jps -l | grep MemoryPressureDemo

# 监控GC统计（每1秒刷新一次）
jstat -gcutil <PID> 1000

