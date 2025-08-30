
# 系统优化与容灾指南

## 1. 系统优化

### 1.1 性能优化

#### 1.1.1 压测分析与优化

拿到压测报告后，接下来会分析报告，然后进行一些有针对性的优化，如硬件升级、系统扩容、参数调优、代码优化(如代码同步改异步)、架构优化(如加缓存、读写分离、历史数据归档)等。不要把别人的经验或案例拿来直接套在自己的场景下，一定要压测，相信压测数据而不是别人的案例。

在进行系统优化时，要进行代码走查，发现不合理的参数配置，如超时时间、降级策略、缓存时间等。在系统压测中进行慢查询排查，包括Redis、MySQL等，通过优化查询解决慢查询问题。

#### 1.1.2 代码优化实现

1. **参数配置优化**
   ```java
   // 超时时间配置
   @Configuration
   public class TimeoutConfig {
       @Bean
       public RestTemplate restTemplate() {
           SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
           factory.setConnectTimeout(3000);  // 连接超时
           factory.setReadTimeout(5000);     // 读取超时
           return new RestTemplate(factory);
       }
   }
   
   // 降级策略配置
   @Configuration
   public class FallbackConfig {
       @Bean
       public HystrixCommandProperties.Setter commandProperties() {
           return HystrixCommandProperties.Setter()
               .withCircuitBreakerEnabled(true)
               .withCircuitBreakerRequestVolumeThreshold(20)
               .withCircuitBreakerSleepWindowInMilliseconds(5000)
               .withCircuitBreakerErrorThresholdPercentage(50);
       }
   }
   ```

2. **慢查询优化**
   ```sql
   -- 优化前
   SELECT * FROM orders WHERE status = 'PENDING' ORDER BY create_time DESC;
   
   -- 优化后
   SELECT id, order_no, status, create_time 
   FROM orders 
   WHERE status = 'PENDING' 
   AND create_time >= DATE_SUB(NOW(), INTERVAL 7 DAY)
   ORDER BY create_time DESC
   LIMIT 100;
   ```

### 1.2 系统扩容

在应用系统扩容方面，可以根据去年流量、与运营业务方沟通促销力度、最近一段时间的流量来评估出是否需要进行扩容，需要扩容多少倍，比如，预计GMV增长100%，那么可以考虑扩容2~3倍容量。还要根据系统特点进行评估，如商品详情页可能要支持平常的十几倍流量，如秒杀系统可能要支持平常的几十倍流量。扩容之后还要预留一些机器应对突发情况，在扩容上尽量支持快速扩容，从而出现突发情况时可以几分钟内完成扩容。

#### 1.2.1 扩容实施

1. **快速扩容方案**
   ```yaml
   # Kubernetes自动扩缩容配置
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: order-service
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: order-service
     minReplicas: 3
     maxReplicas: 20
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70
   ```

2. **资源预留策略**
   - CPU预留：20-30%
   - 内存预留：30-40%
   - 磁盘预留：40-50%

## 2. 系统容灾

在扩容时要考虑系统容灾，比如分组部署、跨机房部署。容灾是通过部署多组(单机房/多机房)相同应用系统，当其中一组出现问题时，可以切换到另一个分组，保证系统可用。

### 2.1 容灾架构

#### 2.1.1 多机房部署

1. **机房选择**
   - 地理位置分散
   - 网络延迟可控
   - 基础设施完备

2. **数据同步策略**
   ```java
   @Service
   public class DataSyncService {
       @Scheduled(fixedRate = 5000)
       public void syncData() {
           // 增量同步
           List<DataChange> changes = dataChangeRepository.findUnsyncedChanges();
           for (DataChange change : changes) {
               try {
                   syncToBackup(change);
                   change.setSynced(true);
                   dataChangeRepository.save(change);
               } catch (Exception e) {
                   log.error("数据同步失败", e);
               }
           }
       }
   }
   ```

### 2.2 应急预案

在系统压测之后会发现一些系统瓶颈，在系统优化之后会提升系统吞吐量并降低响应时间，容灾之后的系统可用性得以保障，但还是会存在一些风险，如网络抖动、某台机器负载过高、某个服务变慢、数据库Load值过高等，为了防止因为这些问题而出现系统雪崩，需要针对这些情况制定应急预案，从而在出现突发情况时，有相应的措施来解决掉这些问题。

应急预案可按照如下几步进行：首先进行系统分级，然后进行全链路分析、配置监控报警，最后制定应急预案。

#### 2.2.1 系统分级

系统分级可以按照交易核心系统和交易支撑系统进行划分。交易核心系统，如购物车，如果挂了，将影响用户无法购物，因此需要投入更多资源保障系统质量，将系统优化到极致，降低事故率。而交易支撑系统是外围系统，如商品后台，即使挂了也不影响前台用户购物，这些系统允许暂时不可用。实际系统分级要根据公司特色进行，目的是对不同级别的系统实施不同的质量保障，核心系统要投入更多资源保障系统高可用，外围系统要投入较少资源允许系统暂时不可用。

1. **核心系统**
   - 交易系统
   - 支付系统
   - 库存系统

2. **支撑系统**
   - 商品后台
   - 运营系统
   - 报表系统

#### 2.2.2 全链路分析

系统分级后，接下来要对交易核心系统进行全链路分析，从用户入口到后端存储，梳理出各个关键路径，对相关路径进行评估并制定预案。即当出现问题时，该路径可以执行什么操作来保证用户可下单、可购物，并且也要防止问题的级联效应和雪崩效应。

1. **关键路径梳理**
   ```
   用户 -> CDN -> 负载均衡 -> 网关 -> 应用服务 -> 数据库
   ```

2. **各层关注点**

   - **网络接入层**：由系统工程师负责，主要关注是机房不可用、DNS故障、VIP故障等预案处理。
   
   - **应用接入层**：由开发工程师负责，主要关注点是上游应用路由切换、限流、降级、隔离等预案处理。
   
   - **Web应用层和服务层**：由开发工程师负责，Web应用层和服务层应用策略差不多，主要关注点是依赖服务的路由切换、连接池(数据库、线程池等)异常、限流、超时降级、服务异常降级、应用负载异常、数据库故障切换、缓存故障切换等。
   
   - **数据层**：由开发工程师或系统工程师负责，主要关注点是数据库/缓存负载高、数据库/缓存故障等。

#### 2.2.3 监控告警

最后，要对关联路径实施监控报警，包括服务器监控(CPU使用率、磁盘使用率、网络带宽等)、系统监控(系统存活、URL状态/内容监控、端口存活等)、JVM监控(堆内存、GC次数、线程数等)、接口监控(接口调用量(每秒/每分钟)、接口性能(TOP50/TOP99/TOP999)、接口可用率等)。然后，配置报警策略，如监控时间段(如上午10:00—13:00、00:00—5:00，不同时间段的报警阈值不一样)、报警阈值(如每5分钟调用次数少于100次则报警)、通知方式(短信/邮件)。在报警后要观察系统状态、监控数据或者日志来查看系统是否真的存在故障，如果确实是故障，则应及时执行相关的预案处理，避免故障扩散。

1. **监控指标**
   ```java
   @Configuration
   public class MonitoringConfig {
       @Bean
       public MeterRegistry meterRegistry() {
           return new PrometheusMeterRegistry(PrometheusConfig.DEFAULT);
       }
       
       @Bean
       public MetricsEndpoint metricsEndpoint(MeterRegistry registry) {
           return new MetricsEndpoint(registry);
       }
   }
   ```

2. **告警策略**
   ```yaml
   # 告警规则配置
   groups:
   - name: system
     rules:
     - alert: HighErrorRate
       expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
       for: 5m
       labels:
         severity: critical
       annotations:
         summary: "High error rate on {{ $labels.instance }}"
         description: "Error rate is {{ $value }}"
   ```

## 3. 最佳实践

### 3.1 预案演练

制定好预案后，应对预案进行演习，来验证预案的正确性，在制定预案时也要设定故障的恢复时间。有一些故障如数据库挂掉是不可降级处理的，对于这种不可降级的关键链路更应进行充分演习。演习一般在零点之后，这个时间点后用户量相对来说较少，即使出了问题影响较小。

1. **演练内容**
   - 定期故障演练
   - 预案有效性验证
   - 团队应急响应

2. **演练频率**
   - 核心系统：每月一次
   - 支撑系统：每季度一次
   - 特殊情况：重大变更前

