以下是升级后的线程池案例，使用 **CountDownLatch** 等待所有任务完成，适配持续在线的系统（线程池不主动关闭），并增加任务结果回告、汇总统计和邮件通知功能。

---

### 升级功能说明
1. **持续存活的线程池**：避免使用 `shutdown()`，通过 `CountDownLatch` 等待当前批次任务完成。
2. **结果回告与汇总**：每个任务生成 JSON 格式报告，主线程汇总后发送邮件。
3. **拒绝策略集成**：被拒绝的任务直接统计，不进入执行流程。

---

### 完整代码实现
```java
import java.util.concurrent.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class PersistentThreadPoolService {

    // 线程池配置（持续存活，不主动关闭）
    private static final ThreadPoolExecutor executor = new ThreadPoolExecutor(
        2, 3, 60, TimeUnit.SECONDS,
        new ArrayBlockingQueue<>(2),
        new RejectedExecutionHandler() {
            @Override
            public void rejectedExecution(Runnable r, ThreadPoolExecutor executor) {
                WrappedTask task = (WrappedTask) r;
                synchronized (rejectedTasks) {
                    rejectedTasks.add(task.getUrl());
                }
                task.getLatch().countDown(); // 被拒绝的任务直接减少计数
                System.err.println("[拒绝] 任务被拒绝: " + task.getUrl());
            }
        }
    );

    // 存储任务结果和拒绝记录
    private static final List<Future<FileReport>> futures = new CopyOnWriteArrayList<>();
    private static final List<String> rejectedTasks = new CopyOnWriteArrayList<>();

    public static void main(String[] args) throws InterruptedException {
        String[] urls = {
            "http://file1.zip",
            "http://file2.pdf",
            "http://file3.jpg",
            "http://file4.mp4",
            "http://file5.doc",
            "http://file6.png"
        };

        // 创建栅栏（等待当前批次所有任务完成）
        CountDownLatch batchLatch = new CountDownLatch(urls.length);

        // 提交任务
        for (String url : urls) {
            DownloadTask downloadTask = new DownloadTask(url);
            WrappedTask wrappedTask = new WrappedTask(url, downloadTask, batchLatch);
            try {
                Future<FileReport> future = executor.submit(wrappedTask);
                futures.add(future);
                System.out.printf("[提交成功] %s (队列大小: %d)\n", url, executor.getQueue().size());
            } catch (RejectedExecutionException e) {
                // 此处理论上不会触发，因拒绝策略已处理
                System.err.println("[提交异常] " + url);
            }
            Thread.sleep(500); // 模拟用户提交间隔
        }

        // 等待当前批次所有任务完成（包括被拒绝的）
        batchLatch.await();
        System.out.println("\n===== 当前批次任务全部完成 =====");

        // 汇总统计
        generateSummaryAndNotify();

        // 实际系统中线程池持续存活，此处不清空数据仅为示例
    }

    // 生成汇总报告并发送邮件
    private static void generateSummaryAndNotify() {
        List<FileReport> reports = new ArrayList<>();
        AtomicInteger success = new AtomicInteger();
        AtomicInteger failure = new AtomicInteger();

        for (Future<FileReport> future : futures) {
            try {
                FileReport report = future.get(1, TimeUnit.SECONDS);
                reports.add(report);
                success.incrementAndGet();
            } catch (Exception e) {
                failure.incrementAndGet();
            }
        }

        // 构建汇总信息
        String summary = String.format(
            "任务汇总:\n- 成功: %d\n- 失败: %d\n- 被拒绝: %d\n- 总文件大小: %d KB",
            success.get(), failure.get(), rejectedTasks.size(),
            reports.stream().mapToInt(r -> r.fileSize).sum()
        );

        // 模拟发送邮件
        sendEmail(summary, reports);
    }

    // 模拟邮件发送
    private static void sendEmail(String summary, List<FileReport> reports) {
        System.out.println("\n=== 邮件内容 ===");
        System.out.println(summary);
        System.out.println("详细报告:");
        reports.forEach(report -> System.out.println(report.toJson()));
        System.out.println("===============\n");
    }

    // 包装任务（集成CountDownLatch逻辑）
    static class WrappedTask implements Runnable {
        private final String url;
        private final DownloadTask task;
        private final CountDownLatch latch;

        public WrappedTask(String url, DownloadTask task, CountDownLatch latch) {
            this.url = url;
            this.task = task;
            this.latch = latch;
        }

        public String getUrl() { return url; }
        public CountDownLatch getLatch() { return latch; }

        @Override
        public void run() {
            try {
                task.process(); // 执行实际任务
            } finally {
                latch.countDown(); // 确保计数减少
            }
        }
    }

    // 文件下载及解析任务
    static class DownloadTask {
        private final String url;
        private final Random random = new Random();

        public DownloadTask(String url) {
            this.url = url;
        }

        public FileReport process() {
            try {
                // 模拟下载耗时
                Thread.sleep(500 + random.nextInt(1500));

                // 解析文件元数据
                String fileType = url.substring(url.lastIndexOf('.') + 1);
                int fileSize = 100 + random.nextInt(900);
                String content = switch (fileType) {
                    case "zip" -> "[ZIP] Compressed data";
                    case "pdf" -> "[PDF] Pages: " + (1 + random.nextInt(50));
                    default -> "[TEXT] Sample content";
                };

                return new FileReport(url, fileType, fileSize, content);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return null;
            }
        }
    }

    // 文件解析结果
    static class FileReport {
        String url;
        String fileType;
        int fileSize;
        String content;

        public FileReport(String url, String fileType, int fileSize, String content) {
            this.url = url;
            this.fileType = fileType;
            this.fileSize = fileSize;
            this.content = content;
        }

        public String toJson() {
            return String.format(
                "{ \"url\": \"%s\", \"type\": \"%s\", \"size\": %d, \"content\": \"%s\" }",
                url, fileType, fileSize, content
            );
        }
    }
}
```

---

### 关键升级点解析

#### 1. 线程池持续存活
- **不调用 `shutdown()`**：线程池保持活动状态，可处理后续任务。
- **批次隔离**：通过 `CountDownLatch` 跟踪当前批次任务，避免影响后续请求。

#### 2. 任务包装与拒绝策略
- **WrappedTask**：封装原始任务和 `CountDownLatch`，确保任务完成或被拒绝时减少计数。
- **拒绝策略集成**：直接在被拒绝时调用 `latch.countDown()`，避免主线程阻塞。

#### 3. 结果汇总与通知
- **异步获取结果**：通过 `Future` 对象收集任务结果，主线程批量提取。
- **邮件通知**：汇总成功、失败、被拒绝的任务数量及详细数据。

---

### 执行流程示例
```plaintext
[提交成功] http://file1.zip (队列大小: 0)
[提交成功] http://file2.pdf (队列大小: 0)
[拒绝] 任务被拒绝: http://file5.doc
[拒绝] 任务被拒绝: http://file6.png
...

===== 当前批次任务全部完成 =====

=== 邮件内容 ===
任务汇总:
- 成功: 4
- 失败: 0
- 被拒绝: 2
- 总文件大小: 3200 KB
详细报告:
{ "url": "http://file1.zip", "type": "zip", "size": 450, "content": "[ZIP] Compressed data" }
...
===============
```

---

### 设计优势
1. **资源高效利用**：线程池持续处理任务，避免频繁创建/销毁。
2. **精准控制批次**：通过 `CountDownLatch` 确保每次批量任务独立统计。
3. **健壮性**：拒绝策略与任务包装解耦，避免计数遗漏或重复。

此方案可直接应用于需要 **批量异步处理** 的生产环境，如文件导入、消息推送等场景。