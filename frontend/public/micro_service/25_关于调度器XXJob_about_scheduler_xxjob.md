# XXL-Job分布式任务调度系统详细指南

## 1. XXL-Job概述

### 1.1 什么是XXL-Job

XXL-Job是一个轻量级分布式任务调度平台，其核心设计目标是开发迅速、学习简单、轻量级、易扩展。现已开放并有多家公司接入线上产品环境。

### 1.2 核心特性

- 简单易用：提供Web界面，支持任务CRUD
- 动态调度：支持动态修改任务状态、启动/停止任务
- 执行策略：支持多种执行策略，如固定频率、Cron表达式等
- 任务分片：支持任务分片执行，提高任务处理能力
- 故障转移：支持任务失败重试和告警
- 任务依赖：支持配置子任务，当父任务执行完成且执行成功后才会触发子任务
- 一致性：采用"调度中心" + "执行器"的设计，确保任务调度的一致性

## 2. 系统架构

### 2.1 整体架构

```
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
|  调度中心(Admin)  |<--->|  执行器(Executor)  |<--->|     任务(Task)    |
|                  |     |                  |     |                  |
+------------------+     +------------------+     +------------------+
```

### 2.2 核心组件

1. **调度中心(Admin)**
   - 负责任务调度
   - 提供Web管理界面
   - 管理执行器
   - 任务日志管理

2. **执行器(Executor)**
   - 负责任务执行
   - 注册到调度中心
   - 接收调度请求
   - 执行任务逻辑

3. **任务(Task)**
   - 具体的业务逻辑
   - 支持多种任务模式
   - 支持任务参数传递

## 3. 快速开始

### 3.1 环境准备

1. **数据库准备**
```sql
-- 创建数据库
CREATE DATABASE xxl_job DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 执行官方SQL脚本初始化表结构
-- 下载地址：https://github.com/xuxueli/xxl-job/blob/master/doc/db/tables_xxl_job.sql
```

2. **Maven依赖**
```xml
<dependency>
    <groupId>com.xuxueli</groupId>
    <artifactId>xxl-job-core</artifactId>
    <version>2.3.0</version>
</dependency>
```

### 3.2 调度中心配置

1. **application.properties**
```properties
# 调度中心配置
server.port=8080
spring.datasource.url=jdbc:mysql://localhost:3306/xxl_job?useUnicode=true&characterEncoding=UTF-8&autoReconnect=true&serverTimezone=Asia/Shanghai
spring.datasource.username=root
spring.datasource.password=123456
spring.datasource.driver-class-name=com.mysql.jdbc.Driver

# 调度中心通讯TOKEN
xxl.job.accessToken=default_token
```

2. **启动调度中心**
```bash
# 下载源码并编译
git clone https://github.com/xuxueli/xxl-job.git
cd xxl-job
mvn clean package

# 启动调度中心
java -jar xxl-job-admin-2.3.0.jar
```

### 3.3 执行器配置

1. **application.properties**
```properties
# 执行器配置
xxl.job.admin.addresses=http://127.0.0.1:8080/xxl-job-admin
xxl.job.accessToken=default_token
xxl.job.executor.appname=xxl-job-executor-sample
xxl.job.executor.address=
xxl.job.executor.ip=127.0.0.1
xxl.job.executor.port=9999
xxl.job.executor.logpath=/data/applogs/xxl-job/jobhandler
xxl.job.executor.logretentiondays=30
```

2. **配置类**
```java
@Configuration
public class XxlJobConfig {
    private Logger logger = LoggerFactory.getLogger(XxlJobConfig.class);

    @Value("${xxl.job.admin.addresses}")
    private String adminAddresses;

    @Value("${xxl.job.accessToken}")
    private String accessToken;

    @Value("${xxl.job.executor.appname}")
    private String appname;

    @Value("${xxl.job.executor.address}")
    private String address;

    @Value("${xxl.job.executor.ip}")
    private String ip;

    @Value("${xxl.job.executor.port}")
    private int port;

    @Value("${xxl.job.executor.logpath}")
    private String logPath;

    @Value("${xxl.job.executor.logretentiondays}")
    private int logRetentionDays;

    @Bean
    public XxlJobSpringExecutor xxlJobExecutor() {
        logger.info(">>>>>>>>>>> xxl-job config init.");
        XxlJobSpringExecutor xxlJobSpringExecutor = new XxlJobSpringExecutor();
        xxlJobSpringExecutor.setAdminAddresses(adminAddresses);
        xxlJobSpringExecutor.setAppname(appname);
        xxlJobSpringExecutor.setAddress(address);
        xxlJobSpringExecutor.setIp(ip);
        xxlJobSpringExecutor.setPort(port);
        xxlJobSpringExecutor.setAccessToken(accessToken);
        xxlJobSpringExecutor.setLogPath(logPath);
        xxlJobSpringExecutor.setLogRetentionDays(logRetentionDays);
        return xxlJobSpringExecutor;
    }
}
```

## 4. 任务开发示例

### 4.1 为什么需要分布式任务调度

在单机环境下，简单的定时任务可能足以满足需求。但随着业务发展，会面临以下问题：

1. **性能瓶颈**
   - 单机处理能力有限
   - 任务执行时间长
   - 资源利用率低

2. **可靠性问题**
   - 单点故障风险
   - 任务执行失败无保障
   - 无故障转移机制

3. **扩展性问题**
   - 难以水平扩展
   - 资源分配不灵活
   - 维护成本高

### 4.2 典型业务场景示例

#### 4.2.1 数据清理任务

```java
@Component
public class DataCleanJob {
    private static final Logger logger = LoggerFactory.getLogger(DataCleanJob.class);

    /**
     * 每天凌晨清理垃圾消息
     * 场景：消息系统中存在大量已读消息，需要定期清理以节省存储空间
     * 使用分布式原因：
     * 1. 数据量大，单机处理时间长
     * 2. 需要保证清理任务的可靠性
     * 3. 可能需要多台机器并行处理
     */
    @XxlJob("cleanExpiredMessages")
    public void cleanExpiredMessages() throws Exception {
        XxlJobHelper.log("开始清理过期消息");
        
        // 获取分片参数
        int shardIndex = XxlJobHelper.getShardIndex();
        int shardTotal = XxlJobHelper.getShardTotal();
        
        // 计算每个分片需要处理的数据范围
        int batchSize = 1000;
        int offset = shardIndex * batchSize;
        
        // 分片查询数据
        List<Message> messages = messageMapper.selectExpiredMessages(offset, batchSize);
        
        // 处理数据
        for (Message message : messages) {
            try {
                // 删除消息
                messageMapper.deleteById(message.getId());
                // 记录日志
                XxlJobHelper.log("删除消息ID: " + message.getId());
            } catch (Exception e) {
                XxlJobHelper.log("删除消息失败: " + e.getMessage());
            }
        }
        
        XxlJobHelper.log("分片 {} 处理完成，共处理 {} 条数据", shardIndex, messages.size());
    }
}
```

#### 4.2.2 报表统计任务

```java
@Component
public class ReportJob {
    private static final Logger logger = LoggerFactory.getLogger(ReportJob.class);

    /**
     * 每日销售报表统计
     * 场景：需要统计每日销售数据，生成报表
     * 使用分布式原因：
     * 1. 数据量大，统计计算耗时
     * 2. 需要保证统计的准确性
     * 3. 可能需要多维度统计
     */
    @XxlJob("dailySalesReport")
    public void dailySalesReport() throws Exception {
        XxlJobHelper.log("开始生成每日销售报表");
        
        // 获取分片参数
        int shardIndex = XxlJobHelper.getShardIndex();
        int shardTotal = XxlJobHelper.getShardTotal();
        
        // 获取统计日期
        String reportDate = XxlJobHelper.getJobParam();
        if (StringUtils.isEmpty(reportDate)) {
            reportDate = LocalDate.now().minusDays(1).toString();
        }
        
        // 按商品类别分片统计
        List<String> categories = categoryService.getAllCategories();
        for (int i = 0; i < categories.size(); i++) {
            if (i % shardTotal == shardIndex) {
                String category = categories.get(i);
                // 统计该类别销售数据
                SalesReport report = salesService.generateCategoryReport(category, reportDate);
                // 保存统计结果
                reportService.saveReport(report);
                XxlJobHelper.log("类别 {} 统计完成", category);
            }
        }
        
        XxlJobHelper.log("分片 {} 报表统计完成", shardIndex);
    }
}
```

#### 4.2.3 订单处理任务

```java
@Component
public class OrderJob {
    private static final Logger logger = LoggerFactory.getLogger(OrderJob.class);

    /**
     * 处理超时未支付订单
     * 场景：电商系统中，需要定期处理超时未支付的订单
     * 使用分布式原因：
     * 1. 订单量大，需要快速处理
     * 2. 处理逻辑复杂，耗时
     * 3. 需要保证订单处理的可靠性
     */
    @XxlJob("processTimeoutOrders")
    public void processTimeoutOrders() throws Exception {
        XxlJobHelper.log("开始处理超时未支付订单");
        
        // 获取分片参数
        int shardIndex = XxlJobHelper.getShardIndex();
        int shardTotal = XxlJobHelper.getShardTotal();
        
        // 查询超时订单
        List<Order> orders = orderMapper.selectTimeoutOrders(shardIndex, shardTotal);
        
        for (Order order : orders) {
            try {
                // 处理订单
                processOrder(order);
                XxlJobHelper.log("订单 {} 处理完成", order.getId());
            } catch (Exception e) {
                XxlJobHelper.log("订单 {} 处理失败: {}", order.getId(), e.getMessage());
            }
        }
        
        XxlJobHelper.log("分片 {} 订单处理完成，共处理 {} 个订单", shardIndex, orders.size());
    }
    
    private void processOrder(Order order) {
        // 1. 更新订单状态
        order.setStatus(OrderStatus.CANCELLED);
        orderMapper.update(order);
        
        // 2. 释放库存
        inventoryService.releaseStock(order);
        
        // 3. 发送通知
        notificationService.sendOrderCancelledNotification(order);
    }
}
```

### 4.3 复杂场景示例：库存履约处理

```java
@Component
public class InventoryFulfillmentJob {
    private static final Logger logger = LoggerFactory.getLogger(InventoryFulfillmentJob.class);

    /**
     * 库存履约数据处理
     * 场景：处理库存履约数据，需要分片处理并分发到多台服务器
     * 使用分布式原因：
     * 1. 数据量大，单次处理1000条
     * 2. 需要多台服务器并行处理
     * 3. 需要保证数据处理的可靠性
     */
    @XxlJob("processInventoryFulfillment")
    public void processInventoryFulfillment() throws Exception {
        XxlJobHelper.log("开始处理库存履约数据");
        
        // 1. 获取分片参数
        int shardIndex = XxlJobHelper.getShardIndex();
        int shardTotal = XxlJobHelper.getShardTotal();
        
        // 2. 查询待处理数据
        List<InventoryFulfillment> fulfillments = 
            fulfillmentMapper.selectPendingFulfillments(shardIndex, shardTotal);
        
        // 3. 创建线程池处理数据
        ExecutorService executorService = Executors.newFixedThreadPool(5);
        List<Future<?>> futures = new ArrayList<>();
        
        // 4. 提交任务到线程池
        for (InventoryFulfillment fulfillment : fulfillments) {
            Future<?> future = executorService.submit(() -> {
                try {
                    processFulfillment(fulfillment);
                } catch (Exception e) {
                    XxlJobHelper.log("处理履约数据失败: {}", e.getMessage());
                }
            });
            futures.add(future);
        }
        
        // 5. 等待所有任务完成
        for (Future<?> future : futures) {
            try {
                future.get();
            } catch (Exception e) {
                XxlJobHelper.log("任务执行异常: {}", e.getMessage());
            }
        }
        
        // 6. 关闭线程池
        executorService.shutdown();
        
        XxlJobHelper.log("分片 {} 处理完成，共处理 {} 条数据", shardIndex, fulfillments.size());
    }
    
    private void processFulfillment(InventoryFulfillment fulfillment) {
        // 1. 更新处理状态
        fulfillment.setStatus(FulfillmentStatus.PROCESSING);
        fulfillmentMapper.update(fulfillment);
        
        try {
            // 2. 处理履约逻辑
            processFulfillmentLogic(fulfillment);
            
            // 3. 更新处理结果
            fulfillment.setStatus(FulfillmentStatus.COMPLETED);
            fulfillmentMapper.update(fulfillment);
            
            XxlJobHelper.log("履约数据 {} 处理成功", fulfillment.getId());
        } catch (Exception e) {
            // 4. 处理失败，更新状态
            fulfillment.setStatus(FulfillmentStatus.FAILED);
            fulfillment.setErrorMsg(e.getMessage());
            fulfillmentMapper.update(fulfillment);
            
            throw e;
        }
    }
    
    private void processFulfillmentLogic(InventoryFulfillment fulfillment) {
        // 1. 检查库存
        checkInventory(fulfillment);
        
        // 2. 更新库存
        updateInventory(fulfillment);
        
        // 3. 生成履约记录
        generateFulfillmentRecord(fulfillment);
        
        // 4. 发送通知
        sendNotification(fulfillment);
    }
}
```

### 4.4 任务配置示例

#### 4.4.1 库存履约任务配置

```properties
# 任务基础配置
job:
  handler: processInventoryFulfillment
  cron: 0 */5 * * * ?  # 每5分钟执行一次
  desc: 处理库存履约数据
  
# 分片配置
sharding:
  total: 10  # 总分片数
  item: 0=A,1=B,2=C,3=D,4=E,5=F,6=G,7=H,8=I,9=J  # 分片参数
  
# 执行器配置
executor:
  appname: inventory-fulfillment
  address: http://server1:9999,http://server2:9999  # 两台服务器
  
# 线程池配置
thread:
  pool: 5  # 每个执行器5个线程
```

#### 4.4.2 任务路由策略

```java
// 自定义路由策略
public class InventoryFulfillmentRouteStrategy implements ExecutorRouteStrategy {
    @Override
    public ReturnT<String> route(TriggerParam triggerParam, List<String> addressList) {
        // 根据分片参数选择执行器
        String shardingParam = triggerParam.getExecutorParams();
        int shardIndex = Integer.parseInt(shardingParam.split("=")[0]);
        
        // 将分片均匀分配到两台服务器
        int serverIndex = shardIndex % addressList.size();
        return new ReturnT<String>(addressList.get(serverIndex));
    }
}
```

## 5. 任务管理

### 5.1 任务配置

1. **基础配置**
   - 执行器：选择任务所属的执行器
   - 任务描述：描述任务的功能
   - 路由策略：选择任务的路由策略
   - Cron：设置任务的执行时间表达式
   - 运行模式：选择任务的运行模式

2. **高级配置**
   - 任务参数：设置任务的参数
   - 阻塞处理策略：选择任务阻塞时的处理策略
   - 子任务：设置任务的子任务
   - 任务超时时间：设置任务的超时时间
   - 失败重试次数：设置任务失败时的重试次数

### 5.2 任务监控

1. **执行日志**
   - 查看任务执行日志
   - 分析任务执行情况
   - 排查任务执行问题

2. **调度报表**
   - 查看任务调度情况
   - 分析任务执行趋势
   - 优化任务调度策略

## 6. 最佳实践

### 6.1 任务设计原则

1. **任务拆分**
   - 将大任务拆分为小任务
   - 合理设置任务执行时间
   - 避免任务执行时间过长

2. **参数设计**
   - 合理设计任务参数
   - 使用JSON格式传递参数
   - 参数要有默认值

3. **异常处理**
   - 合理处理任务异常
   - 记录详细的异常信息
   - 设置合理的重试策略

### 6.2 性能优化

1. **执行器优化**
   - 合理设置线程池大小
   - 优化任务执行逻辑
   - 避免资源竞争

2. **调度优化**
   - 合理设置Cron表达式
   - 避免任务集中执行
   - 使用分片执行提高效率

### 6.3 运维建议

1. **监控告警**
   - 配置任务执行监控
   - 设置合理的告警阈值
   - 及时处理告警信息

2. **日志管理**
   - 合理设置日志级别
   - 定期清理日志文件
   - 做好日志备份

## 7. 常见问题

### 7.1 任务执行问题

1. **任务不执行**
   - 检查执行器是否正常注册
   - 检查Cron表达式是否正确
   - 检查任务参数是否正确

2. **任务执行失败**
   - 检查任务日志
   - 检查任务参数
   - 检查网络连接

### 7.2 系统问题

1. **调度中心问题**
   - 检查数据库连接
   - 检查网络连接
   - 检查系统资源

2. **执行器问题**
   - 检查执行器配置
   - 检查网络连接
   - 检查系统资源

## 8. 总结

XXL-Job是一个功能强大、易于使用的分布式任务调度平台，通过合理的配置和使用，可以满足各种任务调度需求。在实际使用中，需要根据具体业务场景，选择合适的任务类型和配置参数，同时注意任务的设计原则和性能优化，确保系统的稳定性和可靠性。
