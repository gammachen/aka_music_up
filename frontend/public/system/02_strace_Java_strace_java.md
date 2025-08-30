以下是一个包含文件操作、网络IO、内存操作和多线程的Java示例程序，以及使用strace进行跟踪的操作指南：

### 1. Java程序示例（SystemStressTest.java）
```java
import java.io.*;
import java.net.*;
import java.util.*;
import java.util.concurrent.*;

public class SystemStressTest {
    private static final String FILE_PATH = "stress_test.log";
    private static final String TARGET_URL = "http://example.com";
    private static final int THREAD_POOL_SIZE = 3;

    public static void main(String[] args) {
        ExecutorService executor = Executors.newFixedThreadPool(THREAD_POOL_SIZE);
        
        // 文件操作线程
        executor.submit(() -> {
            try {
                while (true) {
                    try (BufferedWriter writer = new BufferedWriter(new FileWriter(FILE_PATH, true))) {
                        writer.write("Timestamp: " + System.currentTimeMillis() + "\n");
                        writer.flush();
                    }
                    Thread.sleep(1000);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        });

        // 网络IO线程
        executor.submit(() -> {
            try {
                while (true) {
                    URL url = new URL(TARGET_URL);
                    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                    conn.setRequestMethod("GET");
                    conn.getResponseCode();  // 触发网络请求
                    Thread.sleep(1500);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        });

        // 内存操作线程
        executor.submit(() -> {
            List<byte[]> memoryList = new ArrayList<>();
            while (true) {
                memoryList.add(new byte[1024 * 1024]);  // 每2秒分配1MB
                Thread.sleep(2000);
            }
        });
    }
}
```

### 2. 使用strace跟踪操作步骤

1. **编译程序**：
```bash
javac SystemStressTest.java
```

2. **运行程序并使用strace跟踪**：
```bash
strace -f -o trace.log -e trace=file,network,memory,clone java SystemStressTest
```

**关键参数说明**：
- `-f`：跟踪子进程（JVM会创建多个线程）
- `-o trace.log`：输出到日志文件
- `-e trace=file,network,memory,clone`：过滤系统调用类型

### 3. 各操作对应的典型系统调用

| 操作类型      | 系统调用示例                      | 说明                      |
|---------------|----------------------------------|--------------------------|
| **文件操作**  | `openat`, `write`, `fsync`      | 文件打开/写入操作         |
| **网络IO**    | `connect`, `sendto`, `recvfrom` | 网络连接和数据传输        |
| **内存操作**  | `brk`, `mmap`, `munmap`         | 内存分配和释放            |
| **多线程**    | `clone`                         | 线程创建                  |

### 4. 结果分析技巧
1. 查看文件操作记录：
```bash
grep 'openat.*stress_test' trace.log
```

2. 查看网络连接：
```bash
grep connect trace.log | grep ':80'
```

3. 查看内存分配：
```bash
grep brk trace.log
```

4. 查看线程创建：
```bash
grep clone trace.log
```

### 5. 注意事项
1. 内存操作线程会持续消耗内存，建议测试时添加JVM参数限制内存：
```bash
java -Xmx256m SystemStressTest
```

2. 网络请求可能会被防火墙拦截，如需测试可用本地web服务器：
```bash
python3 -m http.server 8080  # 然后修改TARGET_URL为localhost:8080
```

3. 按Ctrl+C停止程序后，可使用以下命令清理测试文件：
```bash
rm stress_test.log trace.log
```

```shell
stress_test.log

Timestamp: 1744461271039
Timestamp: 1744461272058
Timestamp: 1744461273059
Timestamp: 1744461274060
Timestamp: 1744461275062
Timestamp: 1744461276064
Timestamp: 1744461277066
Timestamp: 1744461278068
Timestamp: 1744461279068
Timestamp: 1744461280070
Timestamp: 1744461281072
Timestamp: 1744461282074
Timestamp: 1744461283077
Timestamp: 1744461284078
Timestamp: 1744461285081
Timestamp: 1744461286083
Timestamp: 1744461287086
Timestamp: 1744461288087
Timestamp: 1744461289089
Timestamp: 1744461290090
Timestamp: 1744461291092
Timestamp: 1744461292094
Timestamp: 1744461293096
Timestamp: 1744461294098
Timestamp: 1744461295100
Timestamp: 1744461296102
Timestamp: 1744461297104
Timestamp: 1744461298106
Timestamp: 1744461299106
Timestamp: 1744461300108
Timestamp: 1744461301111
Timestamp: 1744461302112
Timestamp: 1744461303115
Timestamp: 1744461304116
Timestamp: 1744461305119
Timestamp: 1744461306120
Timestamp: 1744461307123
Timestamp: 1744461308125
Timestamp: 1744461309127
Timestamp: 1744461310129
Timestamp: 1744461311130
Timestamp: 1744461312131
Timestamp: 1744461313133
Timestamp: 1744461314134
Timestamp: 1744461315135
Timestamp: 1744461316137
Timestamp: 1744461317140
Timestamp: 1744461318142
Timestamp: 1744461319144
Timestamp: 1744461320146
Timestamp: 1744461321148
```

```shell
trace.log

162946 execve("/usr/bin/java", ["java", "SystemStressTest"], 0x7ffc434681c0 /* 65 vars */) = 0
162946 brk(NULL)                        = 0x622c750d2000
162946 mmap(NULL, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7845cd77f000
162946 readlink("/proc/self/exe", "/usr/lib/jvm/java-21-openjdk-amd"..., 4096) = 43
162946 access("/etc/ld.so.preload", R_OK) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/glibc-hwcaps/x86-64-v2/libz.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/glibc-hwcaps/x86-64-v2", 0x7ffd72ab5600, 0) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/tls/x86_64/x86_64/libz.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/tls/x86_64/x86_64", 0x7ffd72ab5600, 0) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/tls/x86_64/libz.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/tls/x86_64", 0x7ffd72ab5600, 0) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/tls/x86_64/libz.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/tls/x86_64", 0x7ffd72ab5600, 0) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/tls/libz.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/tls", 0x7ffd72ab5600, 0) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/x86_64/x86_64/libz.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/x86_64/x86_64", 0x7ffd72ab5600, 0) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/x86_64/libz.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/x86_64", 0x7ffd72ab5600, 0) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/x86_64/libz.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/x86_64", 0x7ffd72ab5600, 0) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/libz.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin", {st_mode=S_IFDIR|0755, st_size=4096, ...}, 0) = 0
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/glibc-hwcaps/x86-64-v2/libz.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/glibc-hwcaps/x86-64-v2", 0x7ffd72ab5600, 0) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/tls/x86_64/x86_64/libz.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/tls/x86_64/x86_64", 0x7ffd72ab5600, 0) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/tls/x86_64/libz.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/tls/x86_64", 0x7ffd72ab5600, 0) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/tls/x86_64/libz.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/tls/x86_64", 0x7ffd72ab5600, 0) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/tls/libz.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/tls", 0x7ffd72ab5600, 0) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/x86_64/x86_64/libz.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/x86_64/x86_64", 0x7ffd72ab5600, 0) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/x86_64/libz.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/x86_64", 0x7ffd72ab5600, 0) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/x86_64/libz.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/x86_64", 0x7ffd72ab5600, 0) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/libz.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib", {st_mode=S_IFDIR|0755, st_size=4096, ...}, 0) = 0
162946 openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
162946 newfstatat(3, "", {st_mode=S_IFREG|0644, st_size=74771, ...}, AT_EMPTY_PATH) = 0
162946 mmap(NULL, 74771, PROT_READ, MAP_PRIVATE, 3, 0) = 0x7845cd76c000
162946 openat(AT_FDCWD, "/lib/x86_64-linux-gnu/libz.so.1", O_RDONLY|O_CLOEXEC) = 3
162946 newfstatat(3, "", {st_mode=S_IFREG|0644, st_size=108936, ...}, AT_EMPTY_PATH) = 0
162946 mmap(NULL, 110776, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7845cd750000
162946 mprotect(0x7845cd752000, 98304, PROT_NONE) = 0
162946 mmap(0x7845cd752000, 69632, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7845cd752000
162946 mmap(0x7845cd763000, 24576, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x13000) = 0x7845cd763000
162946 mmap(0x7845cd76a000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x19000) = 0x7845cd76a000
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/libjli.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/libjli.so", O_RDONLY|O_CLOEXEC) = 3
162946 newfstatat(3, "", {st_mode=S_IFREG|0644, st_size=77688, ...}, AT_EMPTY_PATH) = 0
162946 mmap(NULL, 69936, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7845cd73e000
162946 mprotect(0x7845cd740000, 57344, PROT_NONE) = 0
162946 mmap(0x7845cd740000, 40960, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7845cd740000
162946 mmap(0x7845cd74a000, 12288, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xc000) = 0x7845cd74a000
162946 mmap(0x7845cd74e000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xf000) = 0x7845cd74e000
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/libc.so.6", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/libc.so.6", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/lib/x86_64-linux-gnu/libc.so.6", O_RDONLY|O_CLOEXEC) = 3
162946 newfstatat(3, "", {st_mode=S_IFREG|0755, st_size=2220400, ...}, AT_EMPTY_PATH) = 0
162946 mmap(NULL, 2264656, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7845cd400000
162946 mprotect(0x7845cd428000, 2023424, PROT_NONE) = 0
162946 mmap(0x7845cd428000, 1658880, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x28000) = 0x7845cd428000
162946 mmap(0x7845cd5bd000, 360448, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1bd000) = 0x7845cd5bd000
162946 mmap(0x7845cd616000, 24576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x215000) = 0x7845cd616000
162946 mmap(0x7845cd61c000, 52816, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cd61c000
162946 mmap(NULL, 12288, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7845cd73b000
162946 mprotect(0x7845cd616000, 16384, PROT_READ) = 0
162946 mprotect(0x7845cd76a000, 4096, PROT_READ) = 0
162946 mprotect(0x7845cd74e000, 4096, PROT_READ) = 0
162946 mprotect(0x622c622e6000, 4096, PROT_READ) = 0
162946 mprotect(0x7845cd7b9000, 8192, PROT_READ) = 0
162946 munmap(0x7845cd76c000, 74771)    = 0
162946 brk(NULL)                        = 0x622c750d2000
162946 brk(0x622c750f3000)              = 0x622c750f3000
162946 readlink("/proc/self/exe", "/usr/lib/jvm/java-21-openjdk-amd"..., 4096) = 43
162946 access("/usr/lib/jvm/java-21-openjdk-amd64/lib/libjava.so", F_OK) = 0
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/jvm.cfg", O_RDONLY) = 3
162946 newfstatat(3, "", {st_mode=S_IFREG|0644, st_size=54, ...}, AT_EMPTY_PATH) = 0
162946 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so", {st_mode=S_IFREG|0644, st_size=26598536, ...}, 0) = 0
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so", O_RDONLY|O_CLOEXEC) = 3
162946 newfstatat(3, "", {st_mode=S_IFREG|0644, st_size=26598536, ...}, AT_EMPTY_PATH) = 0
162946 mmap(NULL, 22365016, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7845cbe00000
162946 mprotect(0x7845cc0a1000, 18132992, PROT_NONE) = 0
162946 mmap(0x7845cc0a1000, 15147008, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2a1000) = 0x7845cc0a1000
162946 mmap(0x7845ccf13000, 2981888, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1113000) = 0x7845ccf13000
162946 mmap(0x7845cd1ec000, 1060864, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x13eb000) = 0x7845cd1ec000
162946 mmap(0x7845cd2ef000, 414552, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cd2ef000
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/libstdc++.so.6", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/libstdc++.so.6", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
162946 newfstatat(3, "", {st_mode=S_IFREG|0644, st_size=74771, ...}, AT_EMPTY_PATH) = 0
162946 mmap(NULL, 74771, PROT_READ, MAP_PRIVATE, 3, 0) = 0x7845cd76c000
162946 openat(AT_FDCWD, "/lib/x86_64-linux-gnu/libstdc++.so.6", O_RDONLY|O_CLOEXEC) = 3
162946 newfstatat(3, "", {st_mode=S_IFREG|0644, st_size=2260296, ...}, AT_EMPTY_PATH) = 0
162946 mmap(NULL, 2275520, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7845cba00000
162946 mprotect(0x7845cba9a000, 1576960, PROT_NONE) = 0
162946 mmap(0x7845cba9a000, 1118208, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x9a000) = 0x7845cba9a000
162946 mmap(0x7845cbbab000, 454656, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1ab000) = 0x7845cbbab000
162946 mmap(0x7845cbc1b000, 57344, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x21a000) = 0x7845cbc1b000
162946 mmap(0x7845cbc29000, 10432, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cbc29000
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/libm.so.6", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/libm.so.6", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/lib/x86_64-linux-gnu/libm.so.6", O_RDONLY|O_CLOEXEC) = 3
162946 newfstatat(3, "", {st_mode=S_IFREG|0644, st_size=940560, ...}, AT_EMPTY_PATH) = 0
162946 mmap(NULL, 942344, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7845cd654000
162946 mmap(0x7845cd662000, 507904, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xe000) = 0x7845cd662000
162946 mmap(0x7845cd6de000, 372736, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x8a000) = 0x7845cd6de000
162946 mmap(0x7845cd739000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0xe4000) = 0x7845cd739000
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/libgcc_s.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/libgcc_s.so.1", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162946 openat(AT_FDCWD, "/lib/x86_64-linux-gnu/libgcc_s.so.1", O_RDONLY|O_CLOEXEC) = 3
162946 newfstatat(3, "", {st_mode=S_IFREG|0644, st_size=125488, ...}, AT_EMPTY_PATH) = 0
162946 mmap(NULL, 127720, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7845cd634000
162946 mmap(0x7845cd637000, 94208, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x3000) = 0x7845cd637000
162946 mmap(0x7845cd64e000, 16384, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1a000) = 0x7845cd64e000
162946 mmap(0x7845cd652000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x1d000) = 0x7845cd652000
162946 mprotect(0x7845cd652000, 4096, PROT_READ) = 0
162946 mprotect(0x7845cd739000, 4096, PROT_READ) = 0
162946 mprotect(0x7845cbc1b000, 45056, PROT_READ) = 0
162946 mprotect(0x7845cd1ec000, 864256, PROT_READ) = 0
162946 brk(0x622c75119000)              = 0x622c75119000
162946 munmap(0x7845cd76c000, 74771)    = 0
162946 mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7845cbd00000
162947 mmap(NULL, 134217728, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x7845c3a00000
162947 munmap(0x7845c3a00000, 6291456)  = 0
162947 munmap(0x7845c8000000, 60817408) = 0
162947 mprotect(0x7845c4000000, 135168, PROT_READ|PROT_WRITE) = 0
162947 openat(AT_FDCWD, "/sys/devices/system/cpu", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 3
162947 newfstatat(3, "", {st_mode=S_IFDIR|0755, st_size=0, ...}, AT_EMPTY_PATH) = 0
162947 openat(AT_FDCWD, "/proc/stat", O_RDONLY|O_CLOEXEC) = 3
162947 newfstatat(3, "", {st_mode=S_IFREG|0444, st_size=0, ...}, AT_EMPTY_PATH) = 0
162947 mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7845cd7b8000
162947 mprotect(0x7845cd7b8000, 4096, PROT_READ|PROT_WRITE|PROT_EXEC) = 0
162947 munmap(0x7845cd7b8000, 4096)     = 0
162947 madvise(NULL, 0, MADV_POPULATE_WRITE) = 0
162947 readlink("/usr", 0x7845cbdfd1f0, 1023) = -1 EINVAL (无效的参数)
162947 readlink("/usr/lib", 0x7845cbdfd1f0, 1023) = -1 EINVAL (无效的参数)
162947 readlink("/usr/lib/jvm", 0x7845cbdfd1f0, 1023) = -1 EINVAL (无效的参数)
162947 readlink("/usr/lib/jvm/java-21-openjdk-amd64", 0x7845cbdfd1f0, 1023) = -1 EINVAL (无效的参数)
162947 readlink("/usr/lib/jvm/java-21-openjdk-amd64/lib", 0x7845cbdfd1f0, 1023) = -1 EINVAL (无效的参数)
162947 readlink("/usr/lib/jvm/java-21-openjdk-amd64/lib/server", 0x7845cbdfd1f0, 1023) = -1 EINVAL (无效的参数)
162947 readlink("/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so", 0x7845cbdfd1f0, 1023) = -1 EINVAL (无效的参数)
162947 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/modules", {st_mode=S_IFREG|0644, st_size=140574377, ...}, 0) = 0
162947 openat(AT_FDCWD, "/etc/localtime", O_RDONLY|O_CLOEXEC) = 3
162947 newfstatat(3, "", {st_mode=S_IFREG|0644, st_size=561, ...}, AT_EMPTY_PATH) = 0
162947 newfstatat(3, "", {st_mode=S_IFREG|0644, st_size=561, ...}, AT_EMPTY_PATH) = 0
162947 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/libjimage.so", {st_mode=S_IFREG|0644, st_size=39792, ...}, 0) = 0
162947 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/libjimage.so", O_RDONLY|O_CLOEXEC) = 3
162947 newfstatat(3, "", {st_mode=S_IFREG|0644, st_size=39792, ...}, AT_EMPTY_PATH) = 0
162947 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/libjimage.so", O_RDONLY|O_CLOEXEC) = 3
162947 newfstatat(3, "", {st_mode=S_IFREG|0644, st_size=39792, ...}, AT_EMPTY_PATH) = 0
162947 mmap(NULL, 32928, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 3, 0) = 0x7845cd776000
162947 mprotect(0x7845cd778000, 20480, PROT_NONE) = 0
162947 mmap(0x7845cd778000, 12288, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x2000) = 0x7845cd778000
162947 mmap(0x7845cd77b000, 4096, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x5000) = 0x7845cd77b000
162947 mmap(0x7845cd77d000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 3, 0x6000) = 0x7845cd77d000
162947 mprotect(0x7845cd77d000, 4096, PROT_READ) = 0
162947 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/modules", O_RDONLY) = 3
162947 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/modules", {st_mode=S_IFREG|0644, st_size=140574377, ...}, 0) = 0
162947 mmap(NULL, 140574377, PROT_READ, MAP_SHARED, 3, 0) = 0x7845bb800000
162947 newfstatat(AT_FDCWD, ".hotspotrc", 0x7845cbdfea80, 0) = -1 ENOENT (没有那个文件或目录)
162947 openat(AT_FDCWD, "/proc/cgroups", O_RDONLY|O_CLOEXEC) = 4
162947 newfstatat(4, "", {st_mode=S_IFREG|0444, st_size=0, ...}, AT_EMPTY_PATH) = 0
162947 openat(AT_FDCWD, "/proc/self/cgroup", O_RDONLY|O_CLOEXEC) = 4
162947 newfstatat(4, "", {st_mode=S_IFREG|0444, st_size=0, ...}, AT_EMPTY_PATH) = 0
162947 openat(AT_FDCWD, "/proc/self/mountinfo", O_RDONLY|O_CLOEXEC) = 4
162947 newfstatat(4, "", {st_mode=S_IFREG|0444, st_size=0, ...}, AT_EMPTY_PATH) = 0
162947 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/endorsed", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = -1 ENOENT (没有那个文件或目录)
162947 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/ext", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = -1 ENOENT (没有那个文件或目录)
162947 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/cpu.max", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162947 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/cpu.max", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162947 openat(AT_FDCWD, "/proc/meminfo", O_RDONLY|O_CLOEXEC) = 4
162947 newfstatat(4, "", {st_mode=S_IFREG|0444, st_size=0, ...}, AT_EMPTY_PATH) = 0
162947 openat(AT_FDCWD, "/sys/kernel/mm/hugepages", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 4
162947 newfstatat(4, "", {st_mode=S_IFDIR|0755, st_size=0, ...}, AT_EMPTY_PATH) = 0
162947 openat(AT_FDCWD, "/sys/kernel/mm/transparent_hugepage/enabled", O_RDONLY) = 4
162947 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=4096, ...}, AT_EMPTY_PATH) = 0
162947 openat(AT_FDCWD, "/sys/kernel/mm/transparent_hugepage/hpage_pmd_size", O_RDONLY) = 4
162947 newfstatat(4, "", {st_mode=S_IFREG|0444, st_size=4096, ...}, AT_EMPTY_PATH) = 0
162947 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 4
162947 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162947 openat(AT_FDCWD, "/proc/self/coredump_filter", O_RDWR|O_CLOEXEC) = 4
162947 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162947 openat(AT_FDCWD, "/proc/self/coredump_filter", O_RDWR|O_CLOEXEC) = 4
162947 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162947 mmap(NULL, 4096, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x7845cd7b8000
162947 mmap(0x7845cd7b8000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cd7b8000
162947 mprotect(0x7845cd7b8000, 4096, PROT_NONE) = 0
162947 mmap(NULL, 8192, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x7845cd774000
162947 mmap(0x7845cd774000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cd774000
162947 mprotect(0x7845cd774000, 4096, PROT_NONE) = 0
162947 mprotect(0x7845cd775000, 4096, PROT_READ) = 0
162947 socket(AF_UNIX, SOCK_STREAM|SOCK_CLOEXEC|SOCK_NONBLOCK, 0) = 4
162947 connect(4, {sa_family=AF_UNIX, sun_path="/var/run/nscd/socket"}, 110) = -1 ENOENT (没有那个文件或目录)
162947 socket(AF_UNIX, SOCK_STREAM|SOCK_CLOEXEC|SOCK_NONBLOCK, 0) = 4
162947 connect(4, {sa_family=AF_UNIX, sun_path="/var/run/nscd/socket"}, 110) = -1 ENOENT (没有那个文件或目录)
162947 newfstatat(AT_FDCWD, "/etc/nsswitch.conf", {st_mode=S_IFREG|0644, st_size=542, ...}, 0) = 0
162947 newfstatat(AT_FDCWD, "/", {st_mode=S_IFDIR|0755, st_size=4096, ...}, 0) = 0
162947 openat(AT_FDCWD, "/etc/nsswitch.conf", O_RDONLY|O_CLOEXEC) = 4
162947 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=542, ...}, AT_EMPTY_PATH) = 0
162947 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=542, ...}, AT_EMPTY_PATH) = 0
162947 openat(AT_FDCWD, "/etc/passwd", O_RDONLY|O_CLOEXEC) = 4
162947 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=3089, ...}, AT_EMPTY_PATH) = 0
162947 openat(AT_FDCWD, "/tmp/hsperfdata_shaofu", O_RDONLY|O_NOFOLLOW) = -1 ENOENT (没有那个文件或目录)
162947 mkdir("/tmp/hsperfdata_shaofu", 0755) = 0
162947 openat(AT_FDCWD, "/tmp/hsperfdata_shaofu", O_RDONLY|O_NOFOLLOW) = 4
162947 newfstatat(4, "", {st_mode=S_IFDIR|0755, st_size=4096, ...}, AT_EMPTY_PATH) = 0
162947 openat(AT_FDCWD, "/tmp/hsperfdata_shaofu", O_RDONLY|O_NONBLOCK|O_CLOEXEC|O_DIRECTORY) = 5
162947 newfstatat(5, "", {st_mode=S_IFDIR|0755, st_size=4096, ...}, AT_EMPTY_PATH) = 0
162947 mprotect(0x7845c4021000, 8192, PROT_READ|PROT_WRITE) = 0
162947 newfstatat(4, "", {st_mode=S_IFDIR|0755, st_size=4096, ...}, AT_EMPTY_PATH) = 0
162947 newfstatat(5, "", {st_mode=S_IFDIR|0755, st_size=4096, ...}, AT_EMPTY_PATH) = 0
162947 openat(AT_FDCWD, ".", O_RDONLY)  = 4
162947 openat(AT_FDCWD, "162946", O_RDWR|O_CREAT|O_NOFOLLOW|O_CLOEXEC, 0600) = 6
162947 newfstatat(6, "", {st_mode=S_IFREG|0600, st_size=0, ...}, AT_EMPTY_PATH) = 0
162947 newfstatat(6, "", {st_mode=S_IFREG|0600, st_size=0, ...}, AT_EMPTY_PATH) = 0
162947 mmap(NULL, 32768, PROT_READ|PROT_WRITE, MAP_SHARED, 6, 0) = 0x7845cd76c000
162947 mmap(0x7845cbd00000, 16384, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cbd00000
162947 mprotect(0x7845cbd00000, 16384, PROT_NONE) = 0
162947 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/libjava.so", {st_mode=S_IFREG|0644, st_size=177296, ...}, 0) = 0
162947 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/libjava.so", O_RDONLY|O_CLOEXEC) = 4
162947 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=177296, ...}, AT_EMPTY_PATH) = 0
162947 mprotect(0x7845c4023000, 4096, PROT_READ|PROT_WRITE) = 0
162947 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/libjava.so", O_RDONLY|O_CLOEXEC) = 4
162947 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=177296, ...}, AT_EMPTY_PATH) = 0
162947 mmap(NULL, 143736, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 4, 0) = 0x7845cd3dc000
162947 mprotect(0x7845cd3e9000, 81920, PROT_NONE) = 0
162947 mmap(0x7845cd3e9000, 57344, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0xd000) = 0x7845cd3e9000
162947 mmap(0x7845cd3f7000, 20480, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x1b000) = 0x7845cd3f7000
162947 mmap(0x7845cd3fd000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x20000) = 0x7845cd3fd000
162947 mmap(0x7845cd3ff000, 376, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cd3ff000
162947 mprotect(0x7845cd3fd000, 4096, PROT_READ) = 0
162947 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/modules", {st_mode=S_IFREG|0644, st_size=140574377, ...}, 0) = 0
162947 mprotect(0x7845c4024000, 28672, PROT_READ|PROT_WRITE) = 0
162947 readlink("/usr", 0x7845cbdfd0a0, 1023) = -1 EINVAL (无效的参数)
162947 readlink("/usr/lib", 0x7845cbdfd0a0, 1023) = -1 EINVAL (无效的参数)
162947 readlink("/usr/lib/jvm", 0x7845cbdfd0a0, 1023) = -1 EINVAL (无效的参数)
162947 readlink("/usr/lib/jvm/java-21-openjdk-amd64", 0x7845cbdfd0a0, 1023) = -1 EINVAL (无效的参数)
162947 readlink("/usr/lib/jvm/java-21-openjdk-amd64/lib", 0x7845cbdfd0a0, 1023) = -1 EINVAL (无效的参数)
162947 readlink("/usr/lib/jvm/java-21-openjdk-amd64/lib/modules", 0x7845cbdfd0a0, 1023) = -1 EINVAL (无效的参数)
162947 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/cpu.max", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162947 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/cpu.max", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162947 mmap(NULL, 251658240, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x7845ac800000
162947 mmap(0x7845b3d37000, 2555904, PROT_READ|PROT_WRITE|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845b3d37000
162947 mmap(NULL, 49152, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x7845cd3d0000
162947 mmap(0x7845cd3d0000, 20480, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cd3d0000
162947 mprotect(0x7845c402b000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mmap(0x7845ac800000, 2555904, PROT_READ|PROT_WRITE|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845ac800000
162947 mmap(NULL, 962560, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x7845cb915000
162947 mmap(0x7845cb915000, 20480, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb915000
162947 mmap(0x7845b42c8000, 2555904, PROT_READ|PROT_WRITE|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845b42c8000
162947 mmap(NULL, 962560, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x7845cb82a000
162947 mmap(0x7845cb82a000, 20480, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb82a000
162947 --- SIGSEGV {si_signo=SIGSEGV, si_code=SEGV_MAPERR, si_addr=NULL} ---
162947 openat(AT_FDCWD, "/sys/devices/system/cpu/cpu0/microcode/version", O_RDONLY|O_CLOEXEC) = 4
162947 newfstatat(4, "", {st_mode=S_IFREG|0444, st_size=4096, ...}, AT_EMPTY_PATH) = 0
162947 mprotect(0x7845c402c000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c402d000, 20480, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4032000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4033000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4034000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mmap(NULL, 1052672, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7845cb729000
162947 mprotect(0x7845c4035000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mmap(NULL, 1052672, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7845cb628000
162947 mmap(NULL, 1052672, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7845cb527000
162947 mmap(NULL, 1052672, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7845cb426000
162947 mmap(0x85c00000, 2051014656, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x85c00000
162947 mprotect(0x7845c4036000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4037000, 61440, PROT_READ|PROT_WRITE) = 0
162947 mmap(NULL, 4005888, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x7845cb000000
162947 mmap(NULL, 4005888, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x7845cac00000
162947 mmap(NULL, 32047104, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x7845c8c00000
162947 mprotect(0x7845c4046000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4047000, 16384, PROT_READ|PROT_WRITE) = 0
162947 mmap(NULL, 266240, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x7845cd38f000
162947 mmap(0x7845cd38f000, 266240, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cd38f000
162947 mprotect(0x7845c404b000, 8192, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c404d000, 122880, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c406b000, 16384, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c406f000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4070000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mmap(NULL, 1052672, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7845caaff000
162947 mprotect(0x7845cab00000, 1048576, PROT_READ|PROT_WRITE) = 0
162948 mmap(NULL, 134217728, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x7845a4800000
162948 munmap(0x7845a4800000, 58720256) = 0
162948 munmap(0x7845ac000000, 8388608)  = 0
162948 mprotect(0x7845a8000000, 135168, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4071000, 32768, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4079000, 16384, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c407d000, 16384, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4081000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mmap(NULL, 1052672, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7845c8aff000
162947 mprotect(0x7845c8b00000, 1048576, PROT_READ|PROT_WRITE) = 0
162949 mmap(NULL, 134217728, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x7845a0000000
162949 munmap(0x7845a4000000, 67108864) = 0
162949 mprotect(0x7845a0000000, 135168, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4082000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mmap(NULL, 1052672, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7845c89fe000
162947 mprotect(0x7845c89ff000, 1048576, PROT_READ|PROT_WRITE) = 0
162950 mmap(0x7845a4000000, 67108864, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x78459c000000
162950 mprotect(0x78459c000000, 135168, PROT_READ|PROT_WRITE) = 0
162947 mmap(NULL, 33554432, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x7845a6000000
162947 mmap(0x7845a6000000, 33554432, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845a6000000
162947 mmap(NULL, 1052672, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7845c88fd000
162947 mprotect(0x7845c4083000, 16384, PROT_READ|PROT_WRITE) = 0
162947 mmap(NULL, 1052672, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7845c87fc000
162947 mprotect(0x7845c4087000, 16384, PROT_READ|PROT_WRITE) = 0
162947 mmap(NULL, 1052672, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7845c86fb000
162947 mprotect(0x7845c408b000, 16384, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c408f000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mmap(NULL, 1052672, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7845c85fa000
162947 mprotect(0x7845c4090000, 16384, PROT_READ|PROT_WRITE) = 0
162947 mmap(0x85c00000, 130023424, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x85c00000
162947 mmap(0x7845c8c00000, 2031616, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845c8c00000
162947 mmap(0x7845cb000000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb000000
162947 mmap(0x7845cb001000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb001000
162947 mmap(0x7845cb002000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb002000
162947 mmap(0x7845cb003000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb003000
162947 mmap(0x7845cb004000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb004000
162947 mmap(0x7845cb005000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb005000
162947 mmap(0x7845cb006000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb006000
162947 mmap(0x7845cb007000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb007000
162947 mmap(0x7845cb008000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb008000
162947 mmap(0x7845cb009000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb009000
162947 mmap(0x7845cb00a000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb00a000
162947 mmap(0x7845cb00b000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb00b000
162947 mmap(0x7845cb00c000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb00c000
162947 mmap(0x7845cb00d000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb00d000
162947 mmap(0x7845cb00e000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb00e000
162947 mmap(0x7845cb00f000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb00f000
162947 mmap(0x7845cb010000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb010000
162947 mmap(0x7845cb011000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb011000
162947 mmap(0x7845cb012000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb012000
162947 mmap(0x7845cb013000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb013000
162947 mmap(0x7845cb014000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb014000
162947 mmap(0x7845cb015000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb015000
162947 mmap(0x7845cb016000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb016000
162947 mmap(0x7845cb017000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb017000
162947 mmap(0x7845cb018000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb018000
162947 mmap(0x7845cb019000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb019000
162947 mmap(0x7845cb01a000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb01a000
162947 mmap(0x7845cb01b000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb01b000
162947 mmap(0x7845cb01c000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb01c000
162947 mmap(0x7845cb01d000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb01d000
162947 mmap(0x7845cb01e000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb01e000
162947 mmap(0x7845cb01f000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb01f000
162947 mmap(0x7845cb020000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb020000
162947 mmap(0x7845cb021000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb021000
162947 mmap(0x7845cb022000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb022000
162947 mmap(0x7845cb023000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb023000
162947 mmap(0x7845cb024000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb024000
162947 mmap(0x7845cb025000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb025000
162947 mmap(0x7845cb026000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb026000
162947 mmap(0x7845cb027000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb027000
162947 mmap(0x7845cb028000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb028000
162947 mmap(0x7845cb029000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb029000
162947 mmap(0x7845cb02a000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb02a000
162947 mmap(0x7845cb02b000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb02b000
162947 mmap(0x7845cb02c000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb02c000
162947 mmap(0x7845cb02d000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb02d000
162947 mmap(0x7845cb02e000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb02e000
162947 mmap(0x7845cb02f000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb02f000
162947 mmap(0x7845cb030000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb030000
162947 mmap(0x7845cb031000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb031000
162947 mmap(0x7845cb032000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb032000
162947 mmap(0x7845cb033000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb033000
162947 mmap(0x7845cb034000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb034000
162947 mmap(0x7845cb035000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb035000
162947 mmap(0x7845cb036000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb036000
162947 mmap(0x7845cb037000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb037000
162947 mmap(0x7845cb038000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb038000
162947 mmap(0x7845cb039000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb039000
162947 mmap(0x7845cb03a000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb03a000
162947 mmap(0x7845cb03b000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb03b000
162947 mmap(0x7845cb03c000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb03c000
162947 mmap(0x7845cb03d000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb03d000
162947 mmap(0x7845cac00000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac00000
162947 mmap(0x7845cac01000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac01000
162947 mmap(0x7845cac02000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac02000
162947 mmap(0x7845cac03000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac03000
162947 mmap(0x7845cac04000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac04000
162947 mmap(0x7845cac05000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac05000
162947 mmap(0x7845cac06000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac06000
162947 mmap(0x7845cac07000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac07000
162947 mmap(0x7845cac08000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac08000
162947 mmap(0x7845cac09000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac09000
162947 mmap(0x7845cac0a000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac0a000
162947 mmap(0x7845cac0b000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac0b000
162947 mmap(0x7845cac0c000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac0c000
162947 mmap(0x7845cac0d000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac0d000
162947 mmap(0x7845cac0e000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac0e000
162947 mmap(0x7845cac0f000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac0f000
162947 mmap(0x7845cac10000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac10000
162947 mmap(0x7845cac11000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac11000
162947 mmap(0x7845cac12000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac12000
162947 mmap(0x7845cac13000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac13000
162947 mmap(0x7845cac14000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac14000
162947 mmap(0x7845cac15000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac15000
162947 mmap(0x7845cac16000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac16000
162947 mmap(0x7845cac17000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac17000
162947 mmap(0x7845cac18000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac18000
162947 mmap(0x7845cac19000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac19000
162947 mmap(0x7845cac1a000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac1a000
162947 mmap(0x7845cac1b000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac1b000
162947 mmap(0x7845cac1c000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac1c000
162947 mmap(0x7845cac1d000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac1d000
162947 mmap(0x7845cac1e000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac1e000
162947 mmap(0x7845cac1f000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac1f000
162947 mmap(0x7845cac20000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac20000
162947 mmap(0x7845cac21000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac21000
162947 mmap(0x7845cac22000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac22000
162947 mmap(0x7845cac23000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac23000
162947 mmap(0x7845cac24000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac24000
162947 mmap(0x7845cac25000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac25000
162947 mmap(0x7845cac26000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac26000
162947 mmap(0x7845cac27000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac27000
162947 mmap(0x7845cac28000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac28000
162947 mmap(0x7845cac29000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac29000
162947 mmap(0x7845cac2a000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac2a000
162947 mmap(0x7845cac2b000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac2b000
162947 mmap(0x7845cac2c000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac2c000
162947 mmap(0x7845cac2d000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac2d000
162947 mmap(0x7845cac2e000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac2e000
162947 mmap(0x7845cac2f000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac2f000
162947 mmap(0x7845cac30000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac30000
162947 mmap(0x7845cac31000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac31000
162947 mmap(0x7845cac32000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac32000
162947 mmap(0x7845cac33000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac33000
162947 mmap(0x7845cac34000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac34000
162947 mmap(0x7845cac35000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac35000
162947 mmap(0x7845cac36000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac36000
162947 mmap(0x7845cac37000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac37000
162947 mmap(0x7845cac38000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac38000
162947 mmap(0x7845cac39000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac39000
162947 mmap(0x7845cac3a000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac3a000
162947 mmap(0x7845cac3b000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac3b000
162947 mmap(0x7845cac3c000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac3c000
162947 mmap(0x7845cac3d000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cac3d000
162947 mprotect(0x7845c4094000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4095000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4096000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4097000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4098000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4099000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c409a000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c409b000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c409c000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c409d000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c409e000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c409f000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40a0000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40a1000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40a2000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40a3000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40a4000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40a5000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40a6000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40a7000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40a8000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40a9000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40aa000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40ab000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40ac000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40ad000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40ae000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40af000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40b0000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40b1000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40b2000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40b3000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40b4000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40b5000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40b6000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40b7000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40b8000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40b9000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40ba000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40bb000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40bc000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40bd000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40be000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40bf000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40c0000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40c1000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40c2000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40c3000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40c4000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40c5000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40c6000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40c7000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40c8000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40c9000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40ca000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40cb000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40cc000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40cd000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40ce000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40cf000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40d0000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40d1000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40d2000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40d3000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40d4000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40d5000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40d6000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40d7000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40d8000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40d9000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40da000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40db000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40dc000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40dd000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40de000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40df000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40e0000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40e1000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40e2000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40e3000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40e4000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40e5000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40e6000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40e7000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40e8000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40e9000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40ea000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40eb000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40ec000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40ed000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40ee000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40ef000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40f0000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40f1000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40f2000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40f3000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40f4000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40f5000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40f6000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40f7000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40f8000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40f9000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40fa000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40fb000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40fc000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40fd000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40fe000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c40ff000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4100000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4101000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4102000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4103000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4104000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mmap(NULL, 1052672, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7845c84f9000
162947 mprotect(0x7845c84fa000, 1048576, PROT_READ|PROT_WRITE) = 0
162951 mmap(NULL, 134217728, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x784594000000
162951 munmap(0x784598000000, 67108864) = 0
162951 mprotect(0x784594000000, 135168, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4105000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mmap(NULL, 1052672, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7845c83f8000
162947 mprotect(0x7845c83f9000, 1048576, PROT_READ|PROT_WRITE) = 0
162952 mmap(0x784598000000, 67108864, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x784590000000
162952 mprotect(0x784590000000, 135168, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4106000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4107000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4108000, 8192, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c410a000, 4096, PROT_READ|PROT_WRITE) = 0
162947 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 4
162947 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162947 mprotect(0x7845c410b000, 4096, PROT_READ|PROT_WRITE) = 0
162947 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/server/classes.jsa", O_RDONLY|O_CLOEXEC) = 4
162947 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=14561280, ...}, AT_EMPTY_PATH) = 0
162947 mmap(NULL, 1090519040, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x78454f000000
162947 mmap(0x78454f000000, 4706304, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED, 4, 0x1000) = 0x78454f000000
162947 mmap(0x78454f47d000, 8495104, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED, 4, 0x47e000) = 0x78454f47d000
162947 mmap(NULL, 253952, PROT_READ, MAP_PRIVATE, 4, 0xc98000) = 0x7845cbcc2000
162947 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/modules", {st_mode=S_IFREG|0644, st_size=140574377, ...}, 0) = 0
162947 mmap(0xffe00000, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0xffe00000
162947 mmap(0x7845caa88000, 16384, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845caa88000
162947 mmap(0x7845cb3d1000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cb3d1000
162947 mmap(0x7845cafd1000, 4096, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cafd1000
162947 mprotect(0x7845c410c000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mmap(0xfff00000, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0xfff00000
162947 mmap(0x7845caa8c000, 16384, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845caa8c000
162947 mprotect(0x7845c410d000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mmap(0xffe00000, 1099528, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED, 4, 0xcd6000) = 0xffe00000
162947 mprotect(0x7845c410e000, 8192, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4110000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4111000, 8192, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4113000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4114000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4115000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4116000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4117000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4118000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4119000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c411a000, 4096, PROT_READ|PROT_WRITE) = 0
162947 munmap(0x7845cbcc2000, 253952)   = 0
162947 mprotect(0x7845c411b000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mmap(NULL, 266240, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7845cbcbf000
162947 mmap(NULL, 372736, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7845cbc64000
162947 mmap(NULL, 528384, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7845c8377000
162947 mprotect(0x7845c411c000, 8192, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c411e000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c411f000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mmap(NULL, 1052672, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7845c8276000
162947 mprotect(0x7845c8277000, 1048576, PROT_READ|PROT_WRITE) = 0
162953 mmap(NULL, 134217728, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x784547000000
162953 munmap(0x784547000000, 16777216) = 0
162953 munmap(0x78454c000000, 50331648) = 0
162953 mprotect(0x784548000000, 135168, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4120000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4121000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4122000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4123000, 8192, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4125000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4126000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4127000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4128000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4129000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c412a000, 4096, PROT_READ|PROT_WRITE) = 0
162947 newfstatat(AT_FDCWD, ".hotspot_compiler", 0x7845cbdfe730, 0) = -1 ENOENT (没有那个文件或目录)
162947 mprotect(0x7845c412b000, 8192, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c412d000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c412e000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c412f000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mmap(0x784550040000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x784550040000
162947 mmap(NULL, 67108864, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x784544000000
162947 mmap(0x784544000000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x784544000000
162947 mprotect(0x7845c4130000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4131000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mmap(NULL, 1052672, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7845c8175000
162947 mprotect(0x7845c8176000, 1048576, PROT_READ|PROT_WRITE) = 0
162954 mmap(NULL, 134217728, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x78453c000000
162954 munmap(0x784540000000, 67108864) = 0
162954 mprotect(0x78453c000000, 135168, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4132000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4133000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4134000, 4096, PROT_READ|PROT_WRITE) = 0
162947 openat(AT_FDCWD, "/usr/lib/locale/locale-archive", O_RDONLY|O_CLOEXEC) = 4
162947 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=8876560, ...}, AT_EMPTY_PATH) = 0
162947 mmap(NULL, 8876560, PROT_READ, MAP_PRIVATE, 4, 0) = 0x7845a5600000
162947 mprotect(0x7845c4135000, 4096, PROT_READ|PROT_WRITE) = 0
162947 newfstatat(AT_FDCWD, "/etc/nsswitch.conf", {st_mode=S_IFREG|0644, st_size=542, ...}, 0) = 0
162947 openat(AT_FDCWD, "/etc/passwd", O_RDONLY|O_CLOEXEC) = 4
162947 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=3089, ...}, AT_EMPTY_PATH) = 0
162947 mprotect(0x7845c4136000, 4096, PROT_READ|PROT_WRITE) = 0
162947 newfstatat(AT_FDCWD, "/etc/localtime", {st_mode=S_IFREG|0644, st_size=561, ...}, 0) = 0
162947 getcwd("/home/shaofu", 4096)     = 13
162947 mprotect(0x7845c4137000, 4096, PROT_READ|PROT_WRITE) = 0
162947 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/cpu.max", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162947 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/cpu.max", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162947 mprotect(0x7845c4138000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4139000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c413a000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7845c8075000
162955 mmap(0x784540000000, 67108864, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x784538000000
162955 mprotect(0x784538000000, 135168, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c413b000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c413c000, 8192, PROT_READ|PROT_WRITE) = 0
162955 mmap(0x7845c8075000, 16384, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845c8075000
162947 mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0 <unfinished ...>
162955 mprotect(0x7845c8075000, 16384, PROT_NONE <unfinished ...>
162947 <... mmap resumed>)              = 0x7845c3f00000
162955 <... mprotect resumed>)          = 0
162956 mmap(NULL, 134217728, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x784530000000
162956 munmap(0x784534000000, 67108864) = 0
162956 mprotect(0x784530000000, 135168, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c413e000, 4096, PROT_READ|PROT_WRITE) = 0
162956 mmap(0x7845c3f00000, 16384, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845c3f00000
162956 mprotect(0x7845c3f00000, 16384, PROT_NONE) = 0
162947 mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7845ac700000
162957 mmap(0x784534000000, 67108864, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x78452c000000
162957 mprotect(0x78452c000000, 135168, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c413f000, 4096, PROT_READ|PROT_WRITE) = 0
162957 mmap(0x7845ac700000, 16384, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0 <unfinished ...>
162947 newfstatat(AT_FDCWD, "/tmp/.java_pid162946",  <unfinished ...>
162957 <... mmap resumed>)              = 0x7845ac700000
162957 mprotect(0x7845ac700000, 16384, PROT_NONE <unfinished ...>
162947 <... newfstatat resumed>0x7845cbdfeab0, 0) = -1 ENOENT (没有那个文件或目录)
162957 <... mprotect resumed>)          = 0
162947 mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7845ac600000
162958 mmap(NULL, 134217728, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x784524000000
162958 munmap(0x784528000000, 67108864) = 0
162958 mprotect(0x784524000000, 135168, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4140000, 8192, PROT_READ|PROT_WRITE) = 0
162958 mmap(0x7845ac600000, 16384, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0 <unfinished ...>
162947 mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0 <unfinished ...>
162958 <... mmap resumed>)              = 0x7845ac600000
162947 <... mmap resumed>)              = 0x7845ac500000
162958 mprotect(0x7845ac600000, 16384, PROT_NONE) = 0
162959 mmap(0x784528000000, 67108864, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x784520000000
162959 mprotect(0x784520000000, 135168, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4142000, 4096, PROT_READ|PROT_WRITE) = 0
162959 mmap(0x7845ac500000, 16384, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0 <unfinished ...>
162947 mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0 <unfinished ...>
162959 <... mmap resumed>)              = 0x7845ac500000
162947 <... mmap resumed>)              = 0x7845ac400000
162959 mprotect(0x7845ac500000, 16384, PROT_NONE) = 0
162960 mmap(NULL, 134217728, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x784518000000
162960 munmap(0x78451c000000, 67108864) = 0
162960 mprotect(0x784518000000, 135168, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4143000, 8192, PROT_READ|PROT_WRITE) = 0
162960 mmap(0x7845ac400000, 16384, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845ac400000
162947 mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0 <unfinished ...>
162960 mprotect(0x7845ac400000, 16384, PROT_NONE <unfinished ...>
162947 <... mmap resumed>)              = 0x7845ac300000
162960 <... mprotect resumed>)          = 0
162961 mmap(0x78451c000000, 67108864, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x784514000000
162961 mprotect(0x784514000000, 135168, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4145000, 4096, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x784518021000, 4096, PROT_READ|PROT_WRITE) = 0
162961 mmap(0x7845ac300000, 16384, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845ac300000
162961 mprotect(0x7845ac300000, 16384, PROT_NONE <unfinished ...>
162960 mprotect(0x784518022000, 4096, PROT_READ|PROT_WRITE <unfinished ...>
162961 <... mprotect resumed>)          = 0
162960 <... mprotect resumed>)          = 0
162947 mprotect(0x7845c4146000, 28672, PROT_READ|PROT_WRITE) = 0
162960 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/libjsvml.so", {st_mode=S_IFREG|0644, st_size=870880, ...}, 0) = 0
162947 mprotect(0x7845c414d000, 32768, PROT_READ|PROT_WRITE) = 0
162960 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/libjsvml.so", O_RDONLY|O_CLOEXEC) = 4
162960 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=870880, ...}, AT_EMPTY_PATH) = 0
162960 mprotect(0x784518023000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4155000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4156000, 4096, PROT_READ|PROT_WRITE) = 0
162947 readlink("/home", 0x7845cbdfb4d0, 1023) = -1 EINVAL (无效的参数)
162947 readlink("/home/shaofu", 0x7845cbdfb4d0, 1023) = -1 EINVAL (无效的参数)
162947 faccessat2(AT_FDCWD, "/home/shaofu/", F_OK, AT_EACCESS) = 0
162947 newfstatat(AT_FDCWD, "/home/shaofu", {st_mode=S_IFDIR|0750, st_size=4096, ...}, 0) = 0
162947 mprotect(0x7845c4157000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4158000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4159000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c415a000, 8192, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c415c000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c415d000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c415e000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c415f000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7845ac200000
162962 mmap(NULL, 134217728, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x78450c000000
162962 munmap(0x784510000000, 67108864) = 0
162962 mprotect(0x78450c000000, 135168, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4160000, 8192, PROT_READ|PROT_WRITE) = 0
162962 mmap(0x7845ac200000, 16384, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845ac200000
162962 mprotect(0x7845ac200000, 16384, PROT_NONE) = 0
162960 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/libjsvml.so", O_RDONLY|O_CLOEXEC) = 4
162960 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=870880, ...}, AT_EMPTY_PATH) = 0
162960 mmap(NULL, 852112, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 4, 0) = 0x7845c3e2f000
162960 mmap(0x7845c3e34000, 266240, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x5000) = 0x7845c3e34000
162960 mmap(0x7845c3e75000, 561152, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x46000) = 0x7845c3e75000
162960 mmap(0x7845c3efe000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0xce000) = 0x7845c3efe000
162960 mprotect(0x7845c3efe000, 4096, PROT_READ) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 4
162961 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162960 mprotect(0x784518024000, 32768, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x78451802c000, 32768, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x784518034000, 8192, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x784518036000, 32768, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x78451803e000, 32768, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4162000, 28672, PROT_READ|PROT_WRITE) = 0
162947 readlink("/home", 0x7845cbdfad60, 1023) = -1 EINVAL (无效的参数)
162947 readlink("/home/shaofu", 0x7845cbdfad60, 1023) = -1 EINVAL (无效的参数)
162947 newfstatat(AT_FDCWD, "/home/shaofu/SystemStressTest.class",  <unfinished ...>
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC <unfinished ...>
162947 <... newfstatat resumed>{st_mode=S_IFREG|0644, st_size=3000, ...}, 0) = 0
162961 <... openat resumed>)            = 4
162961 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162947 openat(AT_FDCWD, "/home/shaofu/SystemStressTest.class", O_RDONLY) = 4
162947 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=3000, ...}, AT_EMPTY_PATH) = 0
162947 mprotect(0x7845c4169000, 4096, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x784518046000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c416a000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7845ac100000
162963 mmap(0x784510000000, 67108864, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x784508000000
162963 mprotect(0x784508000000, 135168, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c416b000, 8192, PROT_READ|PROT_WRITE) = 0
162963 mmap(0x7845ac100000, 16384, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845ac100000
162963 mprotect(0x7845ac100000, 16384, PROT_NONE) = 0
162947 newfstatat(AT_FDCWD, "/home/shaofu/SystemStressTest.class", {st_mode=S_IFREG|0644, st_size=3000, ...}, 0) = 0
162947 mmap(0x784544400000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x784544400000
162947 mmap(0x784550000000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0 <unfinished ...>
162960 mprotect(0x784518047000, 32768, PROT_READ|PROT_WRITE <unfinished ...>
162947 <... mmap resumed>)              = 0x784550000000
162960 <... mprotect resumed>)          = 0
162961 mmap(0x784544010000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x784544010000
162961 mprotect(0x784514021000, 12288, PROT_READ|PROT_WRITE <unfinished ...>
162960 mprotect(0x78451804f000, 32768, PROT_READ|PROT_WRITE <unfinished ...>
162961 <... mprotect resumed>)          = 0
162960 <... mprotect resumed>)          = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 4
162961 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162961 mprotect(0x784514024000, 4096, PROT_READ|PROT_WRITE) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 4
162961 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162947 mprotect(0x7845c416d000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mmap(0x784544020000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x784544020000
162947 mprotect(0x7845c416e000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c416f000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4170000, 4096, PROT_READ|PROT_WRITE) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 4
162961 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162947 mprotect(0x7845c4171000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4172000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4173000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mmap(0x784544030000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x784544030000
162947 mprotect(0x7845c4174000, 4096, PROT_READ|PROT_WRITE) = 0
162960 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 4
162960 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162960 mprotect(0x784518057000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4175000, 4096, PROT_READ|PROT_WRITE) = 0
162947 mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7845ac000000
162964 mmap(NULL, 134217728, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x784500000000
162964 munmap(0x784504000000, 67108864) = 0
162964 mprotect(0x784500000000, 135168, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4176000, 8192, PROT_READ|PROT_WRITE) = 0
162964 mmap(0x7845ac000000, 16384, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845ac000000
162964 mprotect(0x7845ac000000, 16384, PROT_NONE <unfinished ...>
162947 mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7845a5f00000
162965 mmap(0x784504000000, 67108864, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x7844fc000000
162965 mprotect(0x7844fc000000, 135168, PROT_READ|PROT_WRITE) = 0
162964 <... mprotect resumed>)          = 0
162947 mprotect(0x7845c4178000, 4096, PROT_READ|PROT_WRITE) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 4
162964 newfstatat(4, "", {st_mode=S_IFREG|0664, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 mmap(0x7845a5f00000, 16384, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845a5f00000
162965 mprotect(0x7845a5f00000, 16384, PROT_NONE <unfinished ...>
162947 mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0 <unfinished ...>
162965 <... mprotect resumed>)          = 0
162947 <... mmap resumed>)              = 0x7845a5500000
162966 mmap(NULL, 134217728, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x7844f4000000
162966 munmap(0x7844f8000000, 67108864) = 0
162966 mprotect(0x7844f4000000, 135168, PROT_READ|PROT_WRITE) = 0
162947 mprotect(0x7845c4179000, 8192, PROT_READ|PROT_WRITE) = 0
162947 mmap(0x7845cbd00000, 16384, PROT_NONE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0 <unfinished ...>
162966 mmap(0x7845a5500000, 16384, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0 <unfinished ...>
162947 <... mmap resumed>)              = 0x7845cbd00000
162966 <... mmap resumed>)              = 0x7845a5500000
162966 mprotect(0x7845a5500000, 16384, PROT_NONE) = 0
162947 mmap(0x7845cbd00000, 16384, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845cbd00000
162947 mprotect(0x7845cbd00000, 16384, PROT_NONE) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 5
162961 newfstatat(5, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162961 mprotect(0x784514025000, 4096, PROT_READ|PROT_WRITE) = 0
162964 mmap(0x784544040000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x784544040000
162961 mmap(0x784544050000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x784544050000
162961 mprotect(0x784514026000, 32768, PROT_READ|PROT_WRITE) = 0
162961 mprotect(0x78451402e000, 32768, PROT_READ|PROT_WRITE) = 0
162961 mprotect(0x784514036000, 32768, PROT_READ|PROT_WRITE) = 0
162961 mprotect(0x78451403e000, 32768, PROT_READ|PROT_WRITE) = 0
162961 mprotect(0x784514046000, 32768, PROT_READ|PROT_WRITE) = 0
162961 mprotect(0x78451404e000, 32768, PROT_READ|PROT_WRITE) = 0
162965 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/libnio.so", {st_mode=S_IFREG|0644, st_size=109624, ...}, 0) = 0
162965 readlink("/usr", 0x7845a5ff7f20, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib", 0x7845a5ff7f20, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm", 0x7845a5ff7f20, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm/java-21-openjdk-amd64", 0x7845a5ff7f20, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm/java-21-openjdk-amd64/lib", 0x7845a5ff7f20, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm/java-21-openjdk-amd64/lib/libnio.so", 0x7845a5ff7f20, 1023) = -1 EINVAL (无效的参数)
162961 mprotect(0x784514056000, 32768, PROT_READ|PROT_WRITE) = 0
162965 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/libnio.so", O_RDONLY|O_CLOEXEC) = 4
162965 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=109624, ...}, AT_EMPTY_PATH) = 0
162965 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/libnio.so", O_RDONLY|O_CLOEXEC) = 4
162965 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=109624, ...}, AT_EMPTY_PATH) = 0
162965 mmap(NULL, 90456, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 4, 0) = 0x7845cd378000
162965 mprotect(0x7845cd37f000, 57344, PROT_NONE) = 0
162965 mmap(0x7845cd37f000, 36864, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x7000) = 0x7845cd37f000
162965 mmap(0x7845cd388000, 16384, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x10000) = 0x7845cd388000
162965 mmap(0x7845cd38d000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x14000) = 0x7845cd38d000
162965 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/glibc-hwcaps/x86-64-v2/libnet.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162965 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/glibc-hwcaps/x86-64-v2", 0x7845a5ff8da0, 0) = -1 ENOENT (没有那个文件或目录)
162965 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/tls/x86_64/x86_64/libnet.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162965 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/tls/x86_64/x86_64", 0x7845a5ff8da0, 0) = -1 ENOENT (没有那个文件或目录)
162965 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/tls/x86_64/libnet.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162965 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/tls/x86_64", 0x7845a5ff8da0, 0) = -1 ENOENT (没有那个文件或目录)
162965 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/tls/x86_64/libnet.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162965 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/tls/x86_64", 0x7845a5ff8da0, 0) = -1 ENOENT (没有那个文件或目录)
162965 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/tls/libnet.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162965 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/tls", 0x7845a5ff8da0, 0) = -1 ENOENT (没有那个文件或目录)
162965 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/x86_64/x86_64/libnet.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162965 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/x86_64/x86_64", 0x7845a5ff8da0, 0) = -1 ENOENT (没有那个文件或目录)
162965 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/x86_64/libnet.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162965 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/x86_64", 0x7845a5ff8da0, 0) = -1 ENOENT (没有那个文件或目录)
162965 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/x86_64/libnet.so", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162965 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/x86_64", 0x7845a5ff8da0, 0) = -1 ENOENT (没有那个文件或目录)
162965 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/libnet.so", O_RDONLY|O_CLOEXEC) = 4
162965 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=65568, ...}, AT_EMPTY_PATH) = 0
162965 mmap(NULL, 57824, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 4, 0) = 0x7845cd369000
162965 mmap(0x7845cd36c000, 32768, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0x3000) = 0x7845cd36c000
162965 mmap(0x7845cd374000, 8192, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0xb000) = 0x7845cd374000
162965 mmap(0x7845cd376000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 4, 0xc000) = 0x7845cd376000
162965 mprotect(0x7845cd376000, 4096, PROT_READ) = 0
162965 mprotect(0x7845cd38d000, 4096, PROT_READ) = 0
162965 getcwd("/home/shaofu", 4097)     = 13
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 4
162965 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/libjimage.so",  <unfinished ...>
162961 newfstatat(4, "",  <unfinished ...>
162965 <... newfstatat resumed>{st_mode=S_IFREG|0644, st_size=39792, ...}, 0) = 0
162961 <... newfstatat resumed>{st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 readlink("/usr",  <unfinished ...>
162961 mprotect(0x78451405e000, 4096, PROT_READ|PROT_WRITE <unfinished ...>
162965 <... readlink resumed>0x7845a5ff9510, 1023) = -1 EINVAL (无效的参数)
162961 <... mprotect resumed>)          = 0
162965 readlink("/usr/lib", 0x7845a5ff9510, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm", 0x7845a5ff9510, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm/java-21-openjdk-amd64", 0x7845a5ff9510, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm/java-21-openjdk-amd64/lib", 0x7845a5ff9510, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm/java-21-openjdk-amd64/lib/libjimage.so", 0x7845a5ff9510, 1023) = -1 EINVAL (无效的参数)
162965 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/libjimage.so", O_RDONLY|O_CLOEXEC) = 4
162965 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=39792, ...}, AT_EMPTY_PATH) = 0
162965 newfstatat(AT_FDCWD, "/home/shaofu/META-INF/services/java.net.spi.URLStreamHandlerProvider", 0x7845a5ffd970, 0) = -1 ENOENT (没有那个文件或目录)
162965 mmap(0x784544060000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x784544060000
162965 mmap(0x784544070000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x784544070000
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 4
162961 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162961 mprotect(0x78451405f000, 4096, PROT_READ|PROT_WRITE) = 0
162965 readlink("/usr", 0x7845a5ffa9f0, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib", 0x7845a5ffa9f0, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm", 0x7845a5ffa9f0, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm/java-21-openjdk-amd64", 0x7845a5ffa9f0, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm/java-21-openjdk-amd64/conf", 0x7845a5ffa9f0, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm/java-21-openjdk-amd64/conf/net.properties", "/etc/java-21-openjdk/net.propert"..., 1023) = 35
162965 readlink("/etc", 0x7845a5ffa9f0, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/etc/java-21-openjdk", 0x7845a5ffa9f0, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/etc/java-21-openjdk/net.properties", 0x7845a5ffa9f0, 1023) = -1 EINVAL (无效的参数)
162965 openat(AT_FDCWD, "/etc/java-21-openjdk/net.properties", O_RDONLY) = 4
162965 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=7441, ...}, AT_EMPTY_PATH) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 4
162961 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 mmap(0x784544080000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x784544080000
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 4
162961 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162961 mprotect(0x784514060000, 4096, PROT_READ|PROT_WRITE) = 0
162965 newfstatat(AT_FDCWD, "/home/shaofu/META-INF/services/java.lang.System$LoggerFinder", 0x7845a5ffcd50, 0) = -1 ENOENT (没有那个文件或目录)
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 4
162961 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 mmap(0x784544090000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x784544090000
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 4
162961 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 mmap(0x7845440a0000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845440a0000
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 4
162961 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 mmap(0x7845440b0000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845440b0000
162965 mmap(0x7845440c0000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845440c0000
162965 mmap(0x784550050000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x784550050000
162965 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/libnet.so", {st_mode=S_IFREG|0644, st_size=65568, ...}, 0) = 0
162965 readlink("/usr", 0x7845a5ff9cb0, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib", 0x7845a5ff9cb0, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm", 0x7845a5ff9cb0, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm/java-21-openjdk-amd64", 0x7845a5ff9cb0, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm/java-21-openjdk-amd64/lib", 0x7845a5ff9cb0, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm/java-21-openjdk-amd64/lib/libnet.so", 0x7845a5ff9cb0, 1023) = -1 EINVAL (无效的参数)
162965 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/libnet.so", O_RDONLY|O_CLOEXEC) = 4
162965 newfstatat(4, "", {st_mode=S_IFREG|0644, st_size=65568, ...}, AT_EMPTY_PATH) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 5
162961 newfstatat(5, "",  <unfinished ...>
162965 socket(AF_INET, SOCK_STREAM, IPPROTO_IP <unfinished ...>
162961 <... newfstatat resumed>{st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 <... socket resumed>)            = 4
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 4
162965 openat(AT_FDCWD, "/proc/net/if_inet6", O_RDONLY) = 4
162965 newfstatat(4, "", {st_mode=S_IFREG|0444, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 4
162965 setsockopt(4, SOL_SOCKET, SO_REUSEPORT, [1], 4) = 0
162965 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/libnio.so", {st_mode=S_IFREG|0644, st_size=109624, ...}, 0) = 0
162965 readlink("/usr", 0x7845a5ff9cb0, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib", 0x7845a5ff9cb0, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm", 0x7845a5ff9cb0, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm/java-21-openjdk-amd64", 0x7845a5ff9cb0, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm/java-21-openjdk-amd64/lib", 0x7845a5ff9cb0, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm/java-21-openjdk-amd64/lib/libnio.so", 0x7845a5ff9cb0, 1023) = -1 EINVAL (无效的参数)
162965 socketpair(AF_UNIX, SOCK_STREAM, 0, [4, 5]) = 0
162965 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/libnet.so", {st_mode=S_IFREG|0644, st_size=65568, ...}, 0) = 0
162965 readlink("/usr", 0x7845a5ffa8d0, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib", 0x7845a5ffa8d0, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm", 0x7845a5ffa8d0, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm/java-21-openjdk-amd64", 0x7845a5ffa8d0, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm/java-21-openjdk-amd64/lib", 0x7845a5ffa8d0, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm/java-21-openjdk-amd64/lib/libnet.so", 0x7845a5ffa8d0, 1023) = -1 EINVAL (无效的参数)
162960 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC <unfinished ...>
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 5
162961 newfstatat(5, "",  <unfinished ...>
162960 <... openat resumed>)            = 6
162961 <... newfstatat resumed>{st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162960 newfstatat(6, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162960 mprotect(0x784518058000, 4096, PROT_READ|PROT_WRITE) = 0
162965 newfstatat(AT_FDCWD, "/home/shaofu/META-INF/services/java.net.spi.InetAddressResolverProvider", 0x7845a5ffd320, 0) = -1 ENOENT (没有那个文件或目录)
162965 socket(AF_UNIX, SOCK_STREAM|SOCK_CLOEXEC|SOCK_NONBLOCK, 0) = 5
162965 connect(5, {sa_family=AF_UNIX, sun_path="/var/run/nscd/socket"}, 110) = -1 ENOENT (没有那个文件或目录)
162965 socket(AF_UNIX, SOCK_STREAM|SOCK_CLOEXEC|SOCK_NONBLOCK, 0) = 5
162965 connect(5, {sa_family=AF_UNIX, sun_path="/var/run/nscd/socket"}, 110) = -1 ENOENT (没有那个文件或目录)
162965 newfstatat(AT_FDCWD, "/etc/nsswitch.conf", {st_mode=S_IFREG|0644, st_size=542, ...}, 0) = 0
162965 newfstatat(AT_FDCWD, "/etc/resolv.conf", {st_mode=S_IFREG|0644, st_size=920, ...}, 0) = 0
162965 openat(AT_FDCWD, "/etc/host.conf", O_RDONLY|O_CLOEXEC) = 5
162965 newfstatat(5, "", {st_mode=S_IFREG|0644, st_size=92, ...}, AT_EMPTY_PATH) = 0
162965 openat(AT_FDCWD, "/etc/resolv.conf", O_RDONLY|O_CLOEXEC) = 5
162965 newfstatat(5, "", {st_mode=S_IFREG|0644, st_size=920, ...}, AT_EMPTY_PATH) = 0
162965 newfstatat(5, "", {st_mode=S_IFREG|0644, st_size=920, ...}, AT_EMPTY_PATH) = 0
162965 openat(AT_FDCWD, "/etc/hosts", O_RDONLY|O_CLOEXEC) = 5
162965 newfstatat(5, "", {st_mode=S_IFREG|0644, st_size=458, ...}, AT_EMPTY_PATH) = 0
162965 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/libnss_mdns4_minimal.so.2", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162965 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/bin/../lib/libnss_mdns4_minimal.so.2", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162965 openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 5
162965 newfstatat(5, "", {st_mode=S_IFREG|0644, st_size=74771, ...}, AT_EMPTY_PATH) = 0
162965 mmap(NULL, 74771, PROT_READ, MAP_PRIVATE, 5, 0) = 0x7845cd356000
162965 openat(AT_FDCWD, "/lib/x86_64-linux-gnu/libnss_mdns4_minimal.so.2", O_RDONLY|O_CLOEXEC) = 5
162965 newfstatat(5, "", {st_mode=S_IFREG|0644, st_size=18664, ...}, AT_EMPTY_PATH) = 0
162965 mmap(NULL, 20496, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 5, 0) = 0x7845cd62e000
162965 mmap(0x7845cd62f000, 8192, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 5, 0x1000) = 0x7845cd62f000
162965 mmap(0x7845cd631000, 4096, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 5, 0x3000) = 0x7845cd631000
162965 mmap(0x7845cd632000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 5, 0x3000) = 0x7845cd632000
162965 mprotect(0x7845cd632000, 4096, PROT_READ) = 0
162965 munmap(0x7845cd356000, 74771)    = 0
162965 socket(AF_INET, SOCK_DGRAM|SOCK_CLOEXEC|SOCK_NONBLOCK, IPPROTO_IP) = 5
162965 setsockopt(5, SOL_IP, IP_RECVERR, [1], 4) = 0
162965 connect(5, {sa_family=AF_INET, sin_port=htons(53), sin_addr=inet_addr("127.0.0.53")}, 16) = 0
162965 sendmmsg(5, [{msg_hdr={msg_name=NULL, msg_namelen=0, msg_iov=[{iov_base="\21\304\1 \0\1\0\0\0\0\0\1\7example\3com\0\0\1\0\1\0\0)"..., iov_len=40}], msg_iovlen=1, msg_controllen=0, msg_flags=0}, msg_len=40}, {msg_hdr={msg_name=NULL, msg_namelen=0, msg_iov=[{iov_base="Y\316\1 \0\1\0\0\0\0\0\1\7example\3com\0\0\34\0\1\0\0)"..., iov_len=40}], msg_iovlen=1, msg_controllen=0, msg_flags=0}, msg_len=40}], 2, MSG_NOSIGNAL) = 2
162965 recvfrom(5, "\21\304\201\200\0\1\0\6\0\0\0\1\7example\3com\0\0\1\0\1\300\f\0"..., 2048, 0, {sa_family=AF_INET, sin_port=htons(53), sin_addr=inet_addr("127.0.0.53")}, [28 => 16]) = 136
162965 mprotect(0x7844fc021000, 28672, PROT_READ|PROT_WRITE) = 0
162965 recvfrom(5, "Y\316\201\200\0\1\0\6\0\0\0\1\7example\3com\0\0\34\0\1\300\f\0"..., 65536, 0, {sa_family=AF_INET, sin_port=htons(53), sin_addr=inet_addr("127.0.0.53")}, [28 => 16]) = 208
162965 openat(AT_FDCWD, "/etc/gai.conf", O_RDONLY|O_CLOEXEC) = 5
162965 newfstatat(5, "", {st_mode=S_IFREG|0644, st_size=2584, ...}, AT_EMPTY_PATH) = 0
162965 newfstatat(5, "", {st_mode=S_IFREG|0644, st_size=2584, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_NETLINK, SOCK_RAW|SOCK_CLOEXEC, NETLINK_ROUTE) = 5
162965 bind(5, {sa_family=AF_NETLINK, nl_pid=0, nl_groups=00000000}, 12) = 0
162965 getsockname(5, {sa_family=AF_NETLINK, nl_pid=162946, nl_groups=00000000}, [12]) = 0
162965 sendto(5, [{nlmsg_len=20, nlmsg_type=RTM_GETADDR, nlmsg_flags=NLM_F_REQUEST|NLM_F_DUMP, nlmsg_seq=1744461271, nlmsg_pid=0}, {ifa_family=AF_UNSPEC, ...}], 20, 0, {sa_family=AF_NETLINK, nl_pid=0, nl_groups=00000000}, 12) = 20
162965 recvmsg(5, {msg_name={sa_family=AF_NETLINK, nl_pid=0, nl_groups=00000000}, msg_namelen=12, msg_iov=[{iov_base=[[{nlmsg_len=76, nlmsg_type=RTM_NEWADDR, nlmsg_flags=NLM_F_MULTI, nlmsg_seq=1744461271, nlmsg_pid=162946}, {ifa_family=AF_INET, ifa_prefixlen=8, ifa_flags=IFA_F_PERMANENT, ifa_scope=RT_SCOPE_HOST, ifa_index=if_nametoindex("lo")}, [[{nla_len=8, nla_type=IFA_ADDRESS}, inet_addr("127.0.0.1")], [{nla_len=8, nla_type=IFA_LOCAL}, inet_addr("127.0.0.1")], [{nla_len=7, nla_type=IFA_LABEL}, "lo"], [{nla_len=8, nla_type=IFA_FLAGS}, IFA_F_PERMANENT], [{nla_len=20, nla_type=IFA_CACHEINFO}, {ifa_prefered=4294967295, ifa_valid=4294967295, cstamp=2672, tstamp=2672}]]], [{nlmsg_len=88, nlmsg_type=RTM_NEWADDR, nlmsg_flags=NLM_F_MULTI, nlmsg_seq=1744461271, nlmsg_pid=162946}, {ifa_family=AF_INET, ifa_prefixlen=24, ifa_flags=0, ifa_scope=RT_SCOPE_UNIVERSE, ifa_index=if_nametoindex("eno1")}, [[{nla_len=8, nla_type=IFA_ADDRESS}, inet_addr("192.168.31.96")], [{nla_len=8, nla_type=IFA_LOCAL}, inet_addr("192.168.31.96")], [{nla_len=8, nla_type=IFA_BROADCAST}, inet_addr("192.168.31.255")], [{nla_len=9, nla_type=IFA_LABEL}, "eno1"], [{nla_len=8, nla_type=IFA_FLAGS}, IFA_F_NOPREFIXROUTE], [{nla_len=20, nla_type=IFA_CACHEINFO}, {ifa_prefered=40139, ifa_valid=40139, cstamp=8496, tstamp=271998507}]]]], iov_len=4096}], msg_iovlen=1, msg_controllen=0, msg_flags=0}, 0) = 164
162965 recvmsg(5, {msg_name={sa_family=AF_NETLINK, nl_pid=0, nl_groups=00000000}, msg_namelen=12, msg_iov=[{iov_base=[[{nlmsg_len=72, nlmsg_type=RTM_NEWADDR, nlmsg_flags=NLM_F_MULTI, nlmsg_seq=1744461271, nlmsg_pid=162946}, {ifa_family=AF_INET6, ifa_prefixlen=128, ifa_flags=IFA_F_PERMANENT, ifa_scope=RT_SCOPE_HOST, ifa_index=if_nametoindex("lo")}, [[{nla_len=20, nla_type=IFA_ADDRESS}, inet_pton(AF_INET6, "::1")], [{nla_len=20, nla_type=IFA_CACHEINFO}, {ifa_prefered=4294967295, ifa_valid=4294967295, cstamp=2672, tstamp=2672}], [{nla_len=8, nla_type=IFA_FLAGS}, IFA_F_PERMANENT]]], [{nlmsg_len=72, nlmsg_type=RTM_NEWADDR, nlmsg_flags=NLM_F_MULTI, nlmsg_seq=1744461271, nlmsg_pid=162946}, {ifa_family=AF_INET6, ifa_prefixlen=64, ifa_flags=IFA_F_PERMANENT, ifa_scope=RT_SCOPE_LINK, ifa_index=if_nametoindex("eno1")}, [[{nla_len=20, nla_type=IFA_ADDRESS}, inet_pton(AF_INET6, "fe80::7907:313e:8162:3d84")], [{nla_len=20, nla_type=IFA_CACHEINFO}, {ifa_prefered=4294967295, ifa_valid=4294967295, cstamp=8181, tstamp=271998507}], [{nla_len=8, nla_type=IFA_FLAGS}, IFA_F_PERMANENT|IFA_F_NOPREFIXROUTE]]]], iov_len=4096}], msg_iovlen=1, msg_controllen=0, msg_flags=0}, 0) = 144
162965 recvmsg(5, {msg_name={sa_family=AF_NETLINK, nl_pid=0, nl_groups=00000000}, msg_namelen=12, msg_iov=[{iov_base=[{nlmsg_len=20, nlmsg_type=NLMSG_DONE, nlmsg_flags=NLM_F_MULTI, nlmsg_seq=1744461271, nlmsg_pid=162946}, 0], iov_len=4096}], msg_iovlen=1, msg_controllen=0, msg_flags=0}, 0) = 20
162965 socket(AF_INET, SOCK_DGRAM|SOCK_CLOEXEC, IPPROTO_IP) = 5
162965 connect(5, {sa_family=AF_INET, sin_port=htons(0), sin_addr=inet_addr("23.215.0.136")}, 16) = 0
162965 getsockname(5, {sa_family=AF_INET, sin_port=htons(44094), sin_addr=inet_addr("192.168.31.96")}, [28 => 16]) = 0
162965 connect(5, {sa_family=AF_UNSPEC, sa_data="\0\0\0\0\0\0\0\0\0\0\0\0\0\0"}, 16) = 0
162965 connect(5, {sa_family=AF_INET, sin_port=htons(0), sin_addr=inet_addr("96.7.128.198")}, 16) = 0
162965 getsockname(5, {sa_family=AF_INET, sin_port=htons(44241), sin_addr=inet_addr("192.168.31.96")}, [28 => 16]) = 0
162965 connect(5, {sa_family=AF_UNSPEC, sa_data="\0\0\0\0\0\0\0\0\0\0\0\0\0\0"}, 16) = 0
162965 connect(5, {sa_family=AF_INET, sin_port=htons(0), sin_addr=inet_addr("23.192.228.80")}, 16) = 0
162965 getsockname(5, {sa_family=AF_INET, sin_port=htons(41505), sin_addr=inet_addr("192.168.31.96")}, [28 => 16]) = 0
162965 connect(5, {sa_family=AF_UNSPEC, sa_data="\0\0\0\0\0\0\0\0\0\0\0\0\0\0"}, 16) = 0
162965 connect(5, {sa_family=AF_INET, sin_port=htons(0), sin_addr=inet_addr("23.215.0.138")}, 16) = 0
162965 getsockname(5, {sa_family=AF_INET, sin_port=htons(59620), sin_addr=inet_addr("192.168.31.96")}, [28 => 16]) = 0
162965 connect(5, {sa_family=AF_UNSPEC, sa_data="\0\0\0\0\0\0\0\0\0\0\0\0\0\0"}, 16) = 0
162965 connect(5, {sa_family=AF_INET, sin_port=htons(0), sin_addr=inet_addr("96.7.128.175")}, 16) = 0
162965 getsockname(5, {sa_family=AF_INET, sin_port=htons(46135), sin_addr=inet_addr("192.168.31.96")}, [28 => 16]) = 0
162965 connect(5, {sa_family=AF_UNSPEC, sa_data="\0\0\0\0\0\0\0\0\0\0\0\0\0\0"}, 16) = 0
162965 connect(5, {sa_family=AF_INET, sin_port=htons(0), sin_addr=inet_addr("23.192.228.84")}, 16) = 0
162965 getsockname(5, {sa_family=AF_INET, sin_port=htons(49773), sin_addr=inet_addr("192.168.31.96")}, [28 => 16]) = 0
162965 socket(AF_INET6, SOCK_DGRAM|SOCK_CLOEXEC, IPPROTO_IP) = 5
162965 connect(5, {sa_family=AF_INET6, sin6_port=htons(0), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "2600:1408:ec00:36::1736:7f24", &sin6_addr), sin6_scope_id=0}, 28) = -1 ENETUNREACH (网络不可达)
162965 connect(5, {sa_family=AF_UNSPEC, sa_data="\0\0\0\0\0\0\0\0\0\0\0\0\0\0"}, 16) = 0
162965 connect(5, {sa_family=AF_INET6, sin6_port=htons(0), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "2600:1408:ec00:36::1736:7f31", &sin6_addr), sin6_scope_id=0}, 28) = -1 ENETUNREACH (网络不可达)
162965 connect(5, {sa_family=AF_UNSPEC, sa_data="\0\0\0\0\0\0\0\0\0\0\0\0\0\0"}, 16) = 0
162965 connect(5, {sa_family=AF_INET6, sin6_port=htons(0), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "2600:1406:bc00:53::b81e:94ce", &sin6_addr), sin6_scope_id=0}, 28) = -1 ENETUNREACH (网络不可达)
162965 connect(5, {sa_family=AF_UNSPEC, sa_data="\0\0\0\0\0\0\0\0\0\0\0\0\0\0"}, 16) = 0
162965 connect(5, {sa_family=AF_INET6, sin6_port=htons(0), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "2600:1406:3a00:21::173e:2e66", &sin6_addr), sin6_scope_id=0}, 28) = -1 ENETUNREACH (网络不可达)
162965 connect(5, {sa_family=AF_UNSPEC, sa_data="\0\0\0\0\0\0\0\0\0\0\0\0\0\0"}, 16) = 0
162965 connect(5, {sa_family=AF_INET6, sin6_port=htons(0), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "2600:1406:3a00:21::173e:2e65", &sin6_addr), sin6_scope_id=0}, 28) = -1 ENETUNREACH (网络不可达)
162965 connect(5, {sa_family=AF_UNSPEC, sa_data="\0\0\0\0\0\0\0\0\0\0\0\0\0\0"}, 16) = 0
162965 connect(5, {sa_family=AF_INET6, sin6_port=htons(0), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "2600:1406:bc00:53::b81e:94c8", &sin6_addr), sin6_scope_id=0}, 28) = -1 ENETUNREACH (网络不可达)
162965 mmap(0x7845440d0000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845440d0000
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 5
162961 newfstatat(5, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 5
162964 newfstatat(5, "", {st_mode=S_IFREG|0664, st_size=25, ...}, AT_EMPTY_PATH) = 0
162965 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/conf/security/java.security",  <unfinished ...>
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC <unfinished ...>
162965 <... newfstatat resumed>{st_mode=S_IFREG|0644, st_size=64427, ...}, 0) = 0
162961 <... openat resumed>)            = 5
162965 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/conf/security/java.security", O_RDONLY <unfinished ...>
162961 newfstatat(5, "",  <unfinished ...>
162965 <... openat resumed>)            = 6
162961 <... newfstatat resumed>{st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 newfstatat(6, "",  <unfinished ...>
162961 mprotect(0x784514061000, 4096, PROT_READ|PROT_WRITE <unfinished ...>
162965 <... newfstatat resumed>{st_mode=S_IFREG|0644, st_size=64427, ...}, AT_EMPTY_PATH) = 0
162961 <... mprotect resumed>)          = 0
162961 mprotect(0x784514062000, 28672, PROT_READ|PROT_WRITE) = 0
162961 mprotect(0x784514069000, 32768, PROT_READ|PROT_WRITE) = 0
162961 mprotect(0x784514071000, 32768, PROT_READ|PROT_WRITE) = 0
162961 mprotect(0x784514079000, 32768, PROT_READ|PROT_WRITE) = 0
162965 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/cpu.max", O_RDONLY|O_CLOEXEC <unfinished ...>
162961 mprotect(0x784514081000, 32768, PROT_READ|PROT_WRITE <unfinished ...>
162965 <... openat resumed>)            = -1 ENOENT (没有那个文件或目录)
162961 <... mprotect resumed>)          = 0
162965 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/cpu.max", O_RDONLY|O_CLOEXEC) = -1 ENOENT (没有那个文件或目录)
162961 mprotect(0x784514089000, 32768, PROT_READ|PROT_WRITE) = 0
162961 mprotect(0x784514091000, 32768, PROT_READ|PROT_WRITE) = 0
162965 mprotect(0x7844fc028000, 32768, PROT_READ|PROT_WRITE) = 0
162965 mprotect(0x7844fc030000, 32768, PROT_READ|PROT_WRITE) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 5
162961 newfstatat(5, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162961 mprotect(0x784514099000, 4096, PROT_READ|PROT_WRITE) = 0
162965 mmap(0x7845440e0000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845440e0000
162965 newfstatat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/libextnet.so", {st_mode=S_IFREG|0644, st_size=16776, ...}, 0) = 0
162965 readlink("/usr", 0x7845a5ff8e90, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib", 0x7845a5ff8e90, 1023) = -1 EINVAL (无效的参数)
162965 readlink("/usr/lib/jvm", 0x7845a5ff8e90, 1023) = -1 EINVAL (无效的参数)
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC <unfinished ...>
162965 readlink("/usr/lib/jvm/java-21-openjdk-amd64", 0x7845a5ff8e90, 1023) = -1 EINVAL (无效的参数)
162961 <... openat resumed>)            = 5
162965 readlink("/usr/lib/jvm/java-21-openjdk-amd64/lib", 0x7845a5ff8e90, 1023) = -1 EINVAL (无效的参数)
162961 newfstatat(5, "",  <unfinished ...>
162965 readlink("/usr/lib/jvm/java-21-openjdk-amd64/lib/libextnet.so",  <unfinished ...>
162961 <... newfstatat resumed>{st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 <... readlink resumed>0x7845a5ff8e90, 1023) = -1 EINVAL (无效的参数)
162961 mprotect(0x78451409a000, 4096, PROT_READ|PROT_WRITE <unfinished ...>
162965 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/libextnet.so", O_RDONLY|O_CLOEXEC <unfinished ...>
162961 <... mprotect resumed>)          = 0
162965 <... openat resumed>)            = 6
162965 newfstatat(6, "", {st_mode=S_IFREG|0644, st_size=16776, ...}, AT_EMPTY_PATH) = 0
162965 openat(AT_FDCWD, "/usr/lib/jvm/java-21-openjdk-amd64/lib/libextnet.so", O_RDONLY|O_CLOEXEC) = 5
162965 newfstatat(5, "", {st_mode=S_IFREG|0644, st_size=16776, ...}, AT_EMPTY_PATH) = 0
162965 mmap(NULL, 16400, PROT_READ, MAP_PRIVATE|MAP_DENYWRITE, 5, 0) = 0x7845cd629000
162965 mmap(0x7845cd62a000, 4096, PROT_READ|PROT_EXEC, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 5, 0x1000) = 0x7845cd62a000
162965 mmap(0x7845cd62b000, 4096, PROT_READ, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 5, 0x2000) = 0x7845cd62b000
162965 mmap(0x7845cd62c000, 8192, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_DENYWRITE, 5, 0x2000) = 0x7845cd62c000
162965 mprotect(0x7845cd62c000, 4096, PROT_READ) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_TCP) = 5
162965 getsockopt(5, SOL_TCP, TCP_QUICKACK, [1], [4]) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_TCP) = 5
162965 getsockopt(5, SOL_TCP, TCP_KEEPIDLE, [7200], [4]) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_TCP) = 5
162965 getsockopt(5, SOL_TCP, TCP_KEEPCNT, [9], [4]) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_TCP) = 5
162965 getsockopt(5, SOL_TCP, TCP_KEEPINTVL, [75], [4]) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_TCP) = 5
162965 getsockopt(5, SOL_SOCKET, SO_INCOMING_NAPI_ID, [0], [4]) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 5
162965 setsockopt(5, SOL_IPV6, IPV6_V6ONLY, [0], 4) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 6
162961 newfstatat(6, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 mmap(0x7845440f0000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845440f0000
162965 connect(5, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.215.0.136", &sin6_addr), sin6_scope_id=0}, 28) = 0
162965 getsockname(5, {sa_family=AF_INET6, sin6_port=htons(51016), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(5, {sa_family=AF_INET6, sin6_port=htons(51016), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 6
162961 newfstatat(6, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 setsockopt(5, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162965 mmap(0x784544100000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x784544100000
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 6
162961 newfstatat(6, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 mmap(0x784544110000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x784544110000
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 6
162964 newfstatat(6, "", {st_mode=S_IFREG|0664, st_size=50, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 6
162964 newfstatat(6, "", {st_mode=S_IFREG|0664, st_size=75, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP <unfinished ...>
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC <unfinished ...>
162965 <... socket resumed>)            = 6
162965 setsockopt(6, SOL_IPV6, IPV6_V6ONLY, [0], 4 <unfinished ...>
162961 <... openat resumed>)            = 7
162965 <... setsockopt resumed>)        = 0
162961 newfstatat(7, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 connect(6, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.215.0.136", &sin6_addr), sin6_scope_id=0}, 28) = 0
162965 getsockname(6, {sa_family=AF_INET6, sin6_port=htons(48820), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(6, {sa_family=AF_INET6, sin6_port=htons(48820), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(6, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 7
162961 newfstatat(7, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 mprotect(0x7844fc038000, 4096, PROT_READ|PROT_WRITE <unfinished ...>
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC <unfinished ...>
162965 <... mprotect resumed>)          = 0
162961 <... openat resumed>)            = 7
162961 newfstatat(7, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 7
162964 newfstatat(7, "", {st_mode=S_IFREG|0664, st_size=100, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 7
162964 newfstatat(7, "", {st_mode=S_IFREG|0664, st_size=125, ...}, AT_EMPTY_PATH) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC <unfinished ...>
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 8
162961 <... openat resumed>)            = 7
162965 setsockopt(8, SOL_IPV6, IPV6_V6ONLY, [0], 4 <unfinished ...>
162961 newfstatat(7, "",  <unfinished ...>
162965 <... setsockopt resumed>)        = 0
162961 <... newfstatat resumed>{st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 connect(8, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.215.0.136", &sin6_addr), sin6_scope_id=0}, 28) = 0
162965 getsockname(8, {sa_family=AF_INET6, sin6_port=htons(48834), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(8, {sa_family=AF_INET6, sin6_port=htons(48834), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(8, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 7
162961 newfstatat(7, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 7
162961 newfstatat(7, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 7
162964 newfstatat(7, "", {st_mode=S_IFREG|0664, st_size=150, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 7
162964 newfstatat(7, "", {st_mode=S_IFREG|0664, st_size=175, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 7
162965 setsockopt(7, SOL_IPV6, IPV6_V6ONLY, [0], 4) = 0
162965 connect(7, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.215.0.136", &sin6_addr), sin6_scope_id=0}, 28 <unfinished ...>
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 9
162961 newfstatat(9, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 <... connect resumed>)           = 0
162965 getsockname(7, {sa_family=AF_INET6, sin6_port=htons(48840), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(7, {sa_family=AF_INET6, sin6_port=htons(48840), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(7, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 9
162964 newfstatat(9, "", {st_mode=S_IFREG|0664, st_size=200, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 9
162964 newfstatat(9, "", {st_mode=S_IFREG|0664, st_size=225, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 9
162965 setsockopt(9, SOL_IPV6, IPV6_V6ONLY, [0], 4) = 0
162965 connect(9, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.215.0.136", &sin6_addr), sin6_scope_id=0}, 28) = 0
162965 getsockname(9, {sa_family=AF_INET6, sin6_port=htons(48842), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(9, {sa_family=AF_INET6, sin6_port=htons(48842), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(9, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162960 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 10
162960 newfstatat(10, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 10
162961 newfstatat(10, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 10
162964 newfstatat(10, "", {st_mode=S_IFREG|0664, st_size=250, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 10
162964 newfstatat(10, "", {st_mode=S_IFREG|0664, st_size=275, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP <unfinished ...>
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC <unfinished ...>
162965 <... socket resumed>)            = 10
162965 setsockopt(10, SOL_IPV6, IPV6_V6ONLY, [0], 4 <unfinished ...>
162961 <... openat resumed>)            = 11
162965 <... setsockopt resumed>)        = 0
162961 newfstatat(11, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 connect(10, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.215.0.136", &sin6_addr), sin6_scope_id=0}, 28) = 0
162965 getsockname(10, {sa_family=AF_INET6, sin6_port=htons(48846), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(10, {sa_family=AF_INET6, sin6_port=htons(48846), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(10, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 11
162961 newfstatat(11, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 11
162964 newfstatat(11, "", {st_mode=S_IFREG|0664, st_size=300, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 11
162964 newfstatat(11, "", {st_mode=S_IFREG|0664, st_size=325, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 11
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC <unfinished ...>
162965 setsockopt(11, SOL_IPV6, IPV6_V6ONLY, [0], 4 <unfinished ...>
162961 <... openat resumed>)            = 12
162965 <... setsockopt resumed>)        = 0
162961 newfstatat(12, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 connect(11, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.215.0.136", &sin6_addr), sin6_scope_id=0}, 28) = 0
162965 getsockname(11, {sa_family=AF_INET6, sin6_port=htons(49244), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(11, {sa_family=AF_INET6, sin6_port=htons(49244), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(11, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 12
162961 newfstatat(12, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 12
162961 newfstatat(12, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 12
162964 newfstatat(12, "", {st_mode=S_IFREG|0664, st_size=350, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 12
162964 newfstatat(12, "", {st_mode=S_IFREG|0664, st_size=375, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 12
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC <unfinished ...>
162965 setsockopt(12, SOL_IPV6, IPV6_V6ONLY, [0], 4) = 0
162961 <... openat resumed>)            = 13
162961 newfstatat(13, "",  <unfinished ...>
162965 connect(12, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.215.0.136", &sin6_addr), sin6_scope_id=0}, 28 <unfinished ...>
162961 <... newfstatat resumed>{st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 <... connect resumed>)           = 0
162965 getsockname(12, {sa_family=AF_INET6, sin6_port=htons(49252), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(12, {sa_family=AF_INET6, sin6_port=htons(49252), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(12, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162960 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 13
162960 newfstatat(13, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 13
162961 newfstatat(13, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 13
162964 newfstatat(13, "", {st_mode=S_IFREG|0664, st_size=400, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 13
162964 newfstatat(13, "", {st_mode=S_IFREG|0664, st_size=425, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP <unfinished ...>
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC <unfinished ...>
162965 <... socket resumed>)            = 13
162961 <... openat resumed>)            = 14
162965 setsockopt(13, SOL_IPV6, IPV6_V6ONLY, [0], 4 <unfinished ...>
162961 newfstatat(14, "",  <unfinished ...>
162965 <... setsockopt resumed>)        = 0
162961 <... newfstatat resumed>{st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 connect(13, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.215.0.136", &sin6_addr), sin6_scope_id=0}, 28) = 0
162965 getsockname(13, {sa_family=AF_INET6, sin6_port=htons(49260), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(13, {sa_family=AF_INET6, sin6_port=htons(49260), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(13, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 14
162961 newfstatat(14, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 14
162961 newfstatat(14, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 14
162964 newfstatat(14, "", {st_mode=S_IFREG|0664, st_size=450, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 14
162964 newfstatat(14, "", {st_mode=S_IFREG|0664, st_size=475, ...}, AT_EMPTY_PATH) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 14
162961 newfstatat(14, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 14
162965 setsockopt(14, SOL_IPV6, IPV6_V6ONLY, [0], 4) = 0
162965 connect(14, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.215.0.136", &sin6_addr), sin6_scope_id=0}, 28) = 0
162965 getsockname(14, {sa_family=AF_INET6, sin6_port=htons(49266), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(14, {sa_family=AF_INET6, sin6_port=htons(49266), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(14, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 15
162961 newfstatat(15, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 15
162964 newfstatat(15, "", {st_mode=S_IFREG|0664, st_size=500, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 15
162964 newfstatat(15, "", {st_mode=S_IFREG|0664, st_size=525, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 15
162965 setsockopt(15, SOL_IPV6, IPV6_V6ONLY, [0], 4) = 0
162965 connect(15, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.215.0.136", &sin6_addr), sin6_scope_id=0}, 28) = 0
162965 getsockname(15, {sa_family=AF_INET6, sin6_port=htons(49268), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(15, {sa_family=AF_INET6, sin6_port=htons(49268), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(15, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 16
162961 newfstatat(16, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 16
162961 newfstatat(16, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 16
162964 newfstatat(16, "", {st_mode=S_IFREG|0664, st_size=550, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 16
162964 newfstatat(16, "", {st_mode=S_IFREG|0664, st_size=575, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 16
162965 setsockopt(16, SOL_IPV6, IPV6_V6ONLY, [0], 4) = 0
162965 connect(16, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.215.0.136", &sin6_addr), sin6_scope_id=0}, 28 <unfinished ...>
162961 mprotect(0x78451409b000, 32768, PROT_READ|PROT_WRITE) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 17
162961 newfstatat(17, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 <... connect resumed>)           = 0
162965 getsockname(16, {sa_family=AF_INET6, sin6_port=htons(35804), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(16, {sa_family=AF_INET6, sin6_port=htons(35804), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(16, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 17
162964 newfstatat(17, "", {st_mode=S_IFREG|0664, st_size=600, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 17
162964 newfstatat(17, "", {st_mode=S_IFREG|0664, st_size=625, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 17
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC <unfinished ...>
162965 setsockopt(17, SOL_IPV6, IPV6_V6ONLY, [0], 4) = 0
162961 <... openat resumed>)            = 18
162961 newfstatat(18, "",  <unfinished ...>
162965 connect(17, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.215.0.136", &sin6_addr), sin6_scope_id=0}, 28 <unfinished ...>
162961 <... newfstatat resumed>{st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 <... connect resumed>)           = 0
162965 getsockname(17, {sa_family=AF_INET6, sin6_port=htons(35816), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(17, {sa_family=AF_INET6, sin6_port=htons(35816), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(17, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 18
162961 newfstatat(18, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 18
162964 newfstatat(18, "", {st_mode=S_IFREG|0664, st_size=650, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 18
162964 newfstatat(18, "", {st_mode=S_IFREG|0664, st_size=675, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 18
162965 setsockopt(18, SOL_IPV6, IPV6_V6ONLY, [0], 4) = 0
162965 connect(18, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.215.0.136", &sin6_addr), sin6_scope_id=0}, 28 <unfinished ...>
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 19
162964 newfstatat(19, "", {st_mode=S_IFREG|0664, st_size=700, ...}, AT_EMPTY_PATH) = 0
162965 <... connect resumed>)           = 0
162965 getsockname(18, {sa_family=AF_INET6, sin6_port=htons(35822), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(18, {sa_family=AF_INET6, sin6_port=htons(35822), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(18, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 19
162961 newfstatat(19, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 19
162964 newfstatat(19, "", {st_mode=S_IFREG|0664, st_size=725, ...}, AT_EMPTY_PATH) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 19
162961 newfstatat(19, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 19
162964 newfstatat(19, "", {st_mode=S_IFREG|0664, st_size=750, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 19
162965 setsockopt(19, SOL_IPV6, IPV6_V6ONLY, [0], 4) = 0
162965 connect(19, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.215.0.136", &sin6_addr), sin6_scope_id=0}, 28) = 0
162965 getsockname(19, {sa_family=AF_INET6, sin6_port=htons(35834), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(19, {sa_family=AF_INET6, sin6_port=htons(35834), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(19, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 20
162964 newfstatat(20, "", {st_mode=S_IFREG|0664, st_size=775, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 20
162964 newfstatat(20, "", {st_mode=S_IFREG|0664, st_size=800, ...}, AT_EMPTY_PATH) = 0
162965 newfstatat(AT_FDCWD, "/etc/nsswitch.conf", {st_mode=S_IFREG|0644, st_size=542, ...}, 0) = 0
162965 newfstatat(AT_FDCWD, "/etc/resolv.conf", {st_mode=S_IFREG|0644, st_size=920, ...}, 0) = 0
162965 openat(AT_FDCWD, "/etc/hosts", O_RDONLY|O_CLOEXEC) = 20
162965 newfstatat(20, "", {st_mode=S_IFREG|0644, st_size=458, ...}, AT_EMPTY_PATH) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 21
162961 newfstatat(21, "",  <unfinished ...>
162965 socket(AF_INET, SOCK_DGRAM|SOCK_CLOEXEC|SOCK_NONBLOCK, IPPROTO_IP <unfinished ...>
162961 <... newfstatat resumed>{st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 <... socket resumed>)            = 20
162965 setsockopt(20, SOL_IP, IP_RECVERR, [1], 4) = 0
162965 connect(20, {sa_family=AF_INET, sin_port=htons(53), sin_addr=inet_addr("127.0.0.53")}, 16) = 0
162965 sendmmsg(20, [{msg_hdr={msg_name=NULL, msg_namelen=0, msg_iov=[{iov_base="\3778\1 \0\1\0\0\0\0\0\1\7example\3com\0\0\1\0\1\0\0)"..., iov_len=40}], msg_iovlen=1, msg_controllen=0, msg_flags=0}, msg_len=40}, {msg_hdr={msg_name=NULL, msg_namelen=0, msg_iov=[{iov_base="\245?\1 \0\1\0\0\0\0\0\1\7example\3com\0\0\34\0\1\0\0)"..., iov_len=40}], msg_iovlen=1, msg_controllen=0, msg_flags=0}, msg_len=40}], 2, MSG_NOSIGNAL) = 2
162965 recvfrom(20, "\3778\201\200\0\1\0\6\0\0\0\1\7example\3com\0\0\1\0\1\300\f\0"..., 2048, 0, {sa_family=AF_INET, sin_port=htons(53), sin_addr=inet_addr("127.0.0.53")}, [28 => 16]) = 136
162965 recvfrom(20, "\245?\201\200\0\1\0\6\0\0\0\1\7example\3com\0\0\34\0\1\300\f\0"..., 65536, 0, {sa_family=AF_INET, sin_port=htons(53), sin_addr=inet_addr("127.0.0.53")}, [28 => 16]) = 208
162965 socket(AF_NETLINK, SOCK_RAW|SOCK_CLOEXEC, NETLINK_ROUTE) = 20
162965 bind(20, {sa_family=AF_NETLINK, nl_pid=0, nl_groups=00000000}, 12) = 0
162965 getsockname(20, {sa_family=AF_NETLINK, nl_pid=162946, nl_groups=00000000}, [12]) = 0
162965 sendto(20, [{nlmsg_len=20, nlmsg_type=RTM_GETADDR, nlmsg_flags=NLM_F_REQUEST|NLM_F_DUMP, nlmsg_seq=1744461303, nlmsg_pid=0}, {ifa_family=AF_UNSPEC, ...}], 20, 0, {sa_family=AF_NETLINK, nl_pid=0, nl_groups=00000000}, 12) = 20
162965 recvmsg(20, {msg_name={sa_family=AF_NETLINK, nl_pid=0, nl_groups=00000000}, msg_namelen=12, msg_iov=[{iov_base=[[{nlmsg_len=76, nlmsg_type=RTM_NEWADDR, nlmsg_flags=NLM_F_MULTI, nlmsg_seq=1744461303, nlmsg_pid=162946}, {ifa_family=AF_INET, ifa_prefixlen=8, ifa_flags=IFA_F_PERMANENT, ifa_scope=RT_SCOPE_HOST, ifa_index=if_nametoindex("lo")}, [[{nla_len=8, nla_type=IFA_ADDRESS}, inet_addr("127.0.0.1")], [{nla_len=8, nla_type=IFA_LOCAL}, inet_addr("127.0.0.1")], [{nla_len=7, nla_type=IFA_LABEL}, "lo"], [{nla_len=8, nla_type=IFA_FLAGS}, IFA_F_PERMANENT], [{nla_len=20, nla_type=IFA_CACHEINFO}, {ifa_prefered=4294967295, ifa_valid=4294967295, cstamp=2672, tstamp=2672}]]], [{nlmsg_len=88, nlmsg_type=RTM_NEWADDR, nlmsg_flags=NLM_F_MULTI, nlmsg_seq=1744461303, nlmsg_pid=162946}, {ifa_family=AF_INET, ifa_prefixlen=24, ifa_flags=0, ifa_scope=RT_SCOPE_UNIVERSE, ifa_index=if_nametoindex("eno1")}, [[{nla_len=8, nla_type=IFA_ADDRESS}, inet_addr("192.168.31.96")], [{nla_len=8, nla_type=IFA_LOCAL}, inet_addr("192.168.31.96")], [{nla_len=8, nla_type=IFA_BROADCAST}, inet_addr("192.168.31.255")], [{nla_len=9, nla_type=IFA_LABEL}, "eno1"], [{nla_len=8, nla_type=IFA_FLAGS}, IFA_F_NOPREFIXROUTE], [{nla_len=20, nla_type=IFA_CACHEINFO}, {ifa_prefered=40108, ifa_valid=40108, cstamp=8496, tstamp=271998507}]]]], iov_len=4096}], msg_iovlen=1, msg_controllen=0, msg_flags=0}, 0) = 164
162965 recvmsg(20, {msg_name={sa_family=AF_NETLINK, nl_pid=0, nl_groups=00000000}, msg_namelen=12, msg_iov=[{iov_base=[[{nlmsg_len=72, nlmsg_type=RTM_NEWADDR, nlmsg_flags=NLM_F_MULTI, nlmsg_seq=1744461303, nlmsg_pid=162946}, {ifa_family=AF_INET6, ifa_prefixlen=128, ifa_flags=IFA_F_PERMANENT, ifa_scope=RT_SCOPE_HOST, ifa_index=if_nametoindex("lo")}, [[{nla_len=20, nla_type=IFA_ADDRESS}, inet_pton(AF_INET6, "::1")], [{nla_len=20, nla_type=IFA_CACHEINFO}, {ifa_prefered=4294967295, ifa_valid=4294967295, cstamp=2672, tstamp=2672}], [{nla_len=8, nla_type=IFA_FLAGS}, IFA_F_PERMANENT]]], [{nlmsg_len=72, nlmsg_type=RTM_NEWADDR, nlmsg_flags=NLM_F_MULTI, nlmsg_seq=1744461303, nlmsg_pid=162946}, {ifa_family=AF_INET6, ifa_prefixlen=64, ifa_flags=IFA_F_PERMANENT, ifa_scope=RT_SCOPE_LINK, ifa_index=if_nametoindex("eno1")}, [[{nla_len=20, nla_type=IFA_ADDRESS}, inet_pton(AF_INET6, "fe80::7907:313e:8162:3d84")], [{nla_len=20, nla_type=IFA_CACHEINFO}, {ifa_prefered=4294967295, ifa_valid=4294967295, cstamp=8181, tstamp=271998507}], [{nla_len=8, nla_type=IFA_FLAGS}, IFA_F_PERMANENT|IFA_F_NOPREFIXROUTE]]]], iov_len=4096}], msg_iovlen=1, msg_controllen=0, msg_flags=0}, 0) = 144
162965 recvmsg(20, {msg_name={sa_family=AF_NETLINK, nl_pid=0, nl_groups=00000000}, msg_namelen=12, msg_iov=[{iov_base=[{nlmsg_len=20, nlmsg_type=NLMSG_DONE, nlmsg_flags=NLM_F_MULTI, nlmsg_seq=1744461303, nlmsg_pid=162946}, 0], iov_len=4096}], msg_iovlen=1, msg_controllen=0, msg_flags=0}, 0) = 20
162965 socket(AF_INET, SOCK_DGRAM|SOCK_CLOEXEC, IPPROTO_IP) = 20
162965 connect(20, {sa_family=AF_INET, sin_port=htons(0), sin_addr=inet_addr("23.192.228.80")}, 16) = 0
162965 getsockname(20, {sa_family=AF_INET, sin_port=htons(51851), sin_addr=inet_addr("192.168.31.96")}, [28 => 16]) = 0
162965 connect(20, {sa_family=AF_UNSPEC, sa_data="\0\0\0\0\0\0\0\0\0\0\0\0\0\0"}, 16) = 0
162965 connect(20, {sa_family=AF_INET, sin_port=htons(0), sin_addr=inet_addr("23.215.0.136")}, 16) = 0
162965 getsockname(20, {sa_family=AF_INET, sin_port=htons(48569), sin_addr=inet_addr("192.168.31.96")}, [28 => 16]) = 0
162965 connect(20, {sa_family=AF_UNSPEC, sa_data="\0\0\0\0\0\0\0\0\0\0\0\0\0\0"}, 16) = 0
162965 connect(20, {sa_family=AF_INET, sin_port=htons(0), sin_addr=inet_addr("96.7.128.198")}, 16) = 0
162965 getsockname(20, {sa_family=AF_INET, sin_port=htons(47758), sin_addr=inet_addr("192.168.31.96")}, [28 => 16]) = 0
162965 connect(20, {sa_family=AF_UNSPEC, sa_data="\0\0\0\0\0\0\0\0\0\0\0\0\0\0"}, 16) = 0
162965 connect(20, {sa_family=AF_INET, sin_port=htons(0), sin_addr=inet_addr("23.215.0.138")}, 16) = 0
162965 getsockname(20, {sa_family=AF_INET, sin_port=htons(38007), sin_addr=inet_addr("192.168.31.96")}, [28 => 16]) = 0
162965 connect(20, {sa_family=AF_UNSPEC, sa_data="\0\0\0\0\0\0\0\0\0\0\0\0\0\0"}, 16) = 0
162965 connect(20, {sa_family=AF_INET, sin_port=htons(0), sin_addr=inet_addr("23.192.228.84")}, 16) = 0
162965 getsockname(20, {sa_family=AF_INET, sin_port=htons(59790), sin_addr=inet_addr("192.168.31.96")}, [28 => 16]) = 0
162965 connect(20, {sa_family=AF_UNSPEC, sa_data="\0\0\0\0\0\0\0\0\0\0\0\0\0\0"}, 16) = 0
162965 connect(20, {sa_family=AF_INET, sin_port=htons(0), sin_addr=inet_addr("96.7.128.175")}, 16) = 0
162965 getsockname(20, {sa_family=AF_INET, sin_port=htons(36838), sin_addr=inet_addr("192.168.31.96")}, [28 => 16]) = 0
162965 socket(AF_INET6, SOCK_DGRAM|SOCK_CLOEXEC, IPPROTO_IP) = 20
162965 connect(20, {sa_family=AF_INET6, sin6_port=htons(0), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "2600:1406:bc00:53::b81e:94c8", &sin6_addr), sin6_scope_id=0}, 28) = -1 ENETUNREACH (网络不可达)
162965 connect(20, {sa_family=AF_UNSPEC, sa_data="\0\0\0\0\0\0\0\0\0\0\0\0\0\0"}, 16) = 0
162965 connect(20, {sa_family=AF_INET6, sin6_port=htons(0), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "2600:1408:ec00:36::1736:7f24", &sin6_addr), sin6_scope_id=0}, 28) = -1 ENETUNREACH (网络不可达)
162965 connect(20, {sa_family=AF_UNSPEC, sa_data="\0\0\0\0\0\0\0\0\0\0\0\0\0\0"}, 16) = 0
162965 connect(20, {sa_family=AF_INET6, sin6_port=htons(0), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "2600:1406:3a00:21::173e:2e65", &sin6_addr), sin6_scope_id=0}, 28) = -1 ENETUNREACH (网络不可达)
162965 connect(20, {sa_family=AF_UNSPEC, sa_data="\0\0\0\0\0\0\0\0\0\0\0\0\0\0"}, 16) = 0
162965 connect(20, {sa_family=AF_INET6, sin6_port=htons(0), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "2600:1406:bc00:53::b81e:94ce", &sin6_addr), sin6_scope_id=0}, 28) = -1 ENETUNREACH (网络不可达)
162965 connect(20, {sa_family=AF_UNSPEC, sa_data="\0\0\0\0\0\0\0\0\0\0\0\0\0\0"}, 16) = 0
162965 connect(20, {sa_family=AF_INET6, sin6_port=htons(0), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "2600:1406:3a00:21::173e:2e66", &sin6_addr), sin6_scope_id=0}, 28) = -1 ENETUNREACH (网络不可达)
162965 connect(20, {sa_family=AF_UNSPEC, sa_data="\0\0\0\0\0\0\0\0\0\0\0\0\0\0"}, 16) = 0
162965 connect(20, {sa_family=AF_INET6, sin6_port=htons(0), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "2600:1408:ec00:36::1736:7f31", &sin6_addr), sin6_scope_id=0}, 28) = -1 ENETUNREACH (网络不可达)
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 20
162965 setsockopt(20, SOL_IPV6, IPV6_V6ONLY, [0], 4) = 0
162965 connect(20, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.192.228.80", &sin6_addr), sin6_scope_id=0}, 28) = 0
162965 getsockname(20, {sa_family=AF_INET6, sin6_port=htons(34434), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(20, {sa_family=AF_INET6, sin6_port=htons(34434), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(20, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 21
162961 newfstatat(21, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 21
162961 newfstatat(21, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 21
162964 newfstatat(21, "", {st_mode=S_IFREG|0664, st_size=825, ...}, AT_EMPTY_PATH) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 21
162961 newfstatat(21, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 21
162964 newfstatat(21, "", {st_mode=S_IFREG|0664, st_size=850, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 21
162965 setsockopt(21, SOL_IPV6, IPV6_V6ONLY, [0], 4) = 0
162965 connect(21, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.192.228.80", &sin6_addr), sin6_scope_id=0}, 28) = 0
162965 getsockname(21, {sa_family=AF_INET6, sin6_port=htons(47398), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(21, {sa_family=AF_INET6, sin6_port=htons(47398), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(21, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 22
162961 newfstatat(22, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162960 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 22
162960 newfstatat(22, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 22
162964 newfstatat(22, "", {st_mode=S_IFREG|0664, st_size=875, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 22
162964 newfstatat(22, "", {st_mode=S_IFREG|0664, st_size=900, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 22
162965 setsockopt(22, SOL_IPV6, IPV6_V6ONLY, [0], 4) = 0
162965 connect(22, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.192.228.80", &sin6_addr), sin6_scope_id=0}, 28) = 0
162965 getsockname(22, {sa_family=AF_INET6, sin6_port=htons(47404), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(22, {sa_family=AF_INET6, sin6_port=htons(47404), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(22, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 23
162961 newfstatat(23, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162960 mprotect(0x784518059000, 28672, PROT_READ|PROT_WRITE) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 23
162961 newfstatat(23, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 23
162964 newfstatat(23, "", {st_mode=S_IFREG|0664, st_size=925, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 23
162964 newfstatat(23, "", {st_mode=S_IFREG|0664, st_size=950, ...}, AT_EMPTY_PATH) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 23
162961 newfstatat(23, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 23
162965 setsockopt(23, SOL_IPV6, IPV6_V6ONLY, [0], 4) = 0
162965 connect(23, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.192.228.80", &sin6_addr), sin6_scope_id=0}, 28 <unfinished ...>
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 24
162961 newfstatat(24, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 <... connect resumed>)           = 0
162965 getsockname(23, {sa_family=AF_INET6, sin6_port=htons(47410), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(23, {sa_family=AF_INET6, sin6_port=htons(47410), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(23, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 24
162964 newfstatat(24, "", {st_mode=S_IFREG|0664, st_size=975, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 24
162965 setsockopt(24, SOL_IPV6, IPV6_V6ONLY, [0], 4) = 0
162965 connect(24, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.192.228.80", &sin6_addr), sin6_scope_id=0}, 28 <unfinished ...>
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 25
162961 newfstatat(25, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 25
162964 newfstatat(25, "", {st_mode=S_IFREG|0664, st_size=1000, ...}, AT_EMPTY_PATH) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 26
162961 newfstatat(26, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 <... connect resumed>)           = 0
162965 getsockname(24, {sa_family=AF_INET6, sin6_port=htons(47418), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(24, {sa_family=AF_INET6, sin6_port=htons(47418), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(24, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 25
162961 newfstatat(25, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162960 mprotect(0x784518060000, 32768, PROT_READ|PROT_WRITE) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 25
162961 newfstatat(25, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 25
162964 newfstatat(25, "", {st_mode=S_IFREG|0664, st_size=1025, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 25
162965 setsockopt(25, SOL_IPV6, IPV6_V6ONLY, [0], 4) = 0
162965 connect(25, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.192.228.80", &sin6_addr), sin6_scope_id=0}, 28) = 0
162965 getsockname(25, {sa_family=AF_INET6, sin6_port=htons(47426), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(25, {sa_family=AF_INET6, sin6_port=htons(47426), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(25, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162960 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 26
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666 <unfinished ...>
162960 newfstatat(26, "",  <unfinished ...>
162964 <... openat resumed>)            = 27
162960 <... newfstatat resumed>{st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 newfstatat(27, "", {st_mode=S_IFREG|0664, st_size=1050, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 26
162964 newfstatat(26, "", {st_mode=S_IFREG|0664, st_size=1075, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 26
162965 setsockopt(26, SOL_IPV6, IPV6_V6ONLY, [0], 4 <unfinished ...>
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC <unfinished ...>
162965 <... setsockopt resumed>)        = 0
162961 <... openat resumed>)            = 27
162965 connect(26, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.192.228.80", &sin6_addr), sin6_scope_id=0}, 28 <unfinished ...>
162961 newfstatat(27, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 <... connect resumed>)           = 0
162965 getsockname(26, {sa_family=AF_INET6, sin6_port=htons(45620), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(26, {sa_family=AF_INET6, sin6_port=htons(45620), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(26, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162961 mmap(0x784544120000, 65536, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x784544120000
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 27
162961 newfstatat(27, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 27
162964 newfstatat(27, "", {st_mode=S_IFREG|0664, st_size=1100, ...}, AT_EMPTY_PATH) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 27
162961 newfstatat(27, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162960 mprotect(0x784518068000, 16384, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x78451806c000, 49152, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x784518078000, 32768, PROT_READ|PROT_WRITE) = 0
162960 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 27
162960 newfstatat(27, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162960 mprotect(0x784518080000, 32768, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x784518088000, 32768, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x784518090000, 32768, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x784518098000, 32768, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x7845180a0000, 32768, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x7845180a8000, 32768, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x7845180b0000, 106496, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x7845180ca000, 32768, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x7845180d2000, 45056, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x7845180dd000, 32768, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x7845180e5000, 32768, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x7845180ed000, 32768, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x7845180f5000, 28672, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x7845180fc000, 32768, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x784518104000, 32768, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x78451810c000, 32768, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x784518114000, 122880, PROT_READ|PROT_WRITE) = 0
162960 mprotect(0x784518132000, 122880, PROT_READ|PROT_WRITE) = 0
162960 openat(AT_FDCWD, "/proc/sys/vm/overcommit_memory", O_RDONLY|O_CLOEXEC) = 27
162960 madvise(0x784518134000, 114688, MADV_DONTNEED) = 0
162953 madvise(0x78451812c000, 94208, MADV_DONTNEED) = 0
162953 madvise(0x784518124000, 32768, MADV_DONTNEED) = 0
162953 madvise(0x784518105000, 126976, MADV_DONTNEED) = 0
162953 madvise(0x7845180e8000, 118784, MADV_DONTNEED) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 27
162964 newfstatat(27, "", {st_mode=S_IFREG|0664, st_size=1125, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 27
162965 setsockopt(27, SOL_IPV6, IPV6_V6ONLY, [0], 4) = 0
162965 connect(27, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.192.228.80", &sin6_addr), sin6_scope_id=0}, 28) = 0
162965 getsockname(27, {sa_family=AF_INET6, sin6_port=htons(45636), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(27, {sa_family=AF_INET6, sin6_port=htons(45636), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(27, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162960 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 28
162960 newfstatat(28, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 28
162964 newfstatat(28, "", {st_mode=S_IFREG|0664, st_size=1150, ...}, AT_EMPTY_PATH) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 28
162961 newfstatat(28, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 28
162964 newfstatat(28, "", {st_mode=S_IFREG|0664, st_size=1175, ...}, AT_EMPTY_PATH) = 0
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 28
162965 setsockopt(28, SOL_IPV6, IPV6_V6ONLY, [0], 4) = 0
162965 connect(28, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.192.228.80", &sin6_addr), sin6_scope_id=0}, 28) = 0
162965 getsockname(28, {sa_family=AF_INET6, sin6_port=htons(45646), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(28, {sa_family=AF_INET6, sin6_port=htons(45646), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(28, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 29
162961 newfstatat(29, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 29
162964 newfstatat(29, "", {st_mode=S_IFREG|0664, st_size=1200, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 29
162964 newfstatat(29, "", {st_mode=S_IFREG|0664, st_size=1225, ...}, AT_EMPTY_PATH) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 30
162961 newfstatat(30, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC <unfinished ...>
162965 socket(AF_INET6, SOCK_STREAM, IPPROTO_IP) = 29
162965 setsockopt(29, SOL_IPV6, IPV6_V6ONLY, [0], 4 <unfinished ...>
162961 <... openat resumed>)            = 30
162965 <... setsockopt resumed>)        = 0
162965 connect(29, {sa_family=AF_INET6, sin6_port=htons(80), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:23.192.228.80", &sin6_addr), sin6_scope_id=0}, 28 <unfinished ...>
162961 newfstatat(30, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162965 <... connect resumed>)           = 0
162965 getsockname(29, {sa_family=AF_INET6, sin6_port=htons(45652), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 getsockname(29, {sa_family=AF_INET6, sin6_port=htons(45652), sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "::ffff:192.168.31.96", &sin6_addr), sin6_scope_id=0}, [28]) = 0
162965 setsockopt(29, SOL_TCP, TCP_NODELAY, [1], 4) = 0
162961 openat(AT_FDCWD, "/sys/fs/cgroup/user.slice/user-1000.slice/session-7229.scope/memory.max", O_RDONLY|O_CLOEXEC) = 30
162961 newfstatat(30, "", {st_mode=S_IFREG|0644, st_size=0, ...}, AT_EMPTY_PATH) = 0
162964 openat(AT_FDCWD, "stress_test.log", O_WRONLY|O_CREAT|O_APPEND, 0666) = 30
162964 newfstatat(30, "", {st_mode=S_IFREG|0664, st_size=1250, ...}, AT_EMPTY_PATH) = 0
162946 --- SIGINT {si_signo=SIGINT, si_code=SI_KERNEL} ---
162957 mmap(NULL, 1048576, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_STACK, -1, 0) = 0x7845a5400000
163018 mmap(0x7844f8000000, 67108864, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0) = 0x7844f0000000
163018 mprotect(0x7844f0000000, 135168, PROT_READ|PROT_WRITE) = 0
163018 mmap(0x7845a5400000, 16384, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS, -1, 0) = 0x7845a5400000
163018 mprotect(0x7845a5400000, 16384, PROT_NONE) = 0
162953 madvise(0x7845c8276000, 1028096, MADV_DONTNEED) = 0
162953 +++ exited with 0 +++
162951 madvise(0x7845c84f9000, 1028096, MADV_DONTNEED) = 0
162951 +++ exited with 0 +++
162952 madvise(0x7845c83f8000, 1028096, MADV_DONTNEED) = 0
162952 +++ exited with 0 +++
162949 madvise(0x7845c8aff000, 1028096, MADV_DONTNEED <unfinished ...>
162957 mmap(0x7845ac700000, 16384, PROT_NONE, MAP_PRIVATE|MAP_FIXED|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0 <unfinished ...>
162949 <... madvise resumed>)           = 0
162957 <... mmap resumed>)              = 0x7845ac700000
162949 +++ exited with 0 +++
162957 madvise(0x7845ac700000, 1024000, MADV_DONTNEED) = 0
162957 +++ exited with 0 +++
162954 unlink("/tmp/hsperfdata_shaofu/162946") = 0
162964 +++ exited with 130 +++
```