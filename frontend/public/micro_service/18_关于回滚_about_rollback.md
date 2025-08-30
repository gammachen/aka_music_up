# 回滚机制详解

## 1. 概述

回滚是指当程序或数据出错时，将程序或数据恢复到最近的一个正确版本的行为。通过回滚机制可以保证系统在某些场景下的高可用性。常见的回滚类型包括：

- 事务回滚：数据库事务、分布式事务的回滚
- 代码库回滚：Git版本回滚
- 部署版本回滚：应用部署版本的回滚
- 数据版本回滚：数据快照和版本回滚
- 静态资源版本回滚：前端资源版本回滚

## 2. 事务回滚

### 2.1 单库事务回滚

#### 2.1.1 Spring事务管理

```java
@Service
public class OrderService {
    
    @Autowired
    private OrderRepository orderRepository;
    
    @Autowired
    private InventoryRepository inventoryRepository;
    
    @Transactional(rollbackFor = Exception.class)
    public void createOrder(OrderDTO orderDTO) {
        try {
            // 1. 创建订单
            Order order = new Order();
            order.setUserId(orderDTO.getUserId());
            order.setAmount(orderDTO.getAmount());
            order.setStatus(OrderStatus.PENDING);
            orderRepository.save(order);
            
            // 2. 扣减库存
            inventoryRepository.decreaseStock(
                orderDTO.getProductId(), 
                orderDTO.getQuantity()
            );
            
            // 3. 如果发生异常，事务会自动回滚
        } catch (Exception e) {
            log.error("创建订单失败", e);
            throw e; // 抛出异常触发回滚
        }
    }
}
```

#### 2.1.2 手动事务控制

```java
@Service
public class OrderService {
    
    @Autowired
    private PlatformTransactionManager transactionManager;
    
    public void createOrder(OrderDTO orderDTO) {
        DefaultTransactionDefinition def = new DefaultTransactionDefinition();
        TransactionStatus status = transactionManager.getTransaction(def);
        
        try {
            // 业务逻辑
            orderRepository.save(createOrderEntity(orderDTO));
            inventoryRepository.decreaseStock(
                orderDTO.getProductId(), 
                orderDTO.getQuantity()
            );
            
            transactionManager.commit(status);
        } catch (Exception e) {
            transactionManager.rollback(status);
            throw e;
        }
    }
}
```

### 2.2 分布式事务回滚

#### 2.2.1 TCC模式实现

```java
@Service
public class OrderTccService {
    
    @Autowired
    private OrderRepository orderRepository;
    
    @Autowired
    private InventoryTccService inventoryTccService;
    
    @Autowired
    private CouponTccService couponTccService;
    
    @Transactional
    public void createOrder(OrderDTO orderDTO) {
        // 1. Try阶段：预留资源
        Order order = orderRepository.save(createOrderEntity(orderDTO));
        inventoryTccService.tryDecreaseStock(orderDTO.getProductId(), orderDTO.getQuantity());
        couponTccService.tryLockCoupon(orderDTO.getUserId(), orderDTO.getCouponId());
        
        // 2. Confirm阶段：确认操作
        try {
            inventoryTccService.confirmDecreaseStock(orderDTO.getProductId());
            couponTccService.confirmLockCoupon(orderDTO.getUserId(), orderDTO.getCouponId());
            order.setStatus(OrderStatus.CONFIRMED);
            orderRepository.save(order);
        } catch (Exception e) {
            // 3. Cancel阶段：取消操作
            inventoryTccService.cancelDecreaseStock(orderDTO.getProductId());
            couponTccService.cancelLockCoupon(orderDTO.getUserId(), orderDTO.getCouponId());
            order.setStatus(OrderStatus.CANCELLED);
            orderRepository.save(order);
            throw e;
        }
    }
}
```

#### 2.2.2 Saga模式实现

```java
@Service
public class OrderSagaService {
    
    @Autowired
    private OrderRepository orderRepository;
    
    @Autowired
    private InventoryService inventoryService;
    
    @Autowired
    private CouponService couponService;
    
    public void createOrder(OrderDTO orderDTO) {
        Order order = null;
        try {
            // 1. 创建订单
            order = orderRepository.save(createOrderEntity(orderDTO));
            
            // 2. 扣减库存
            inventoryService.decreaseStock(
                orderDTO.getProductId(), 
                orderDTO.getQuantity()
            );
            
            // 3. 使用优惠券
            couponService.useCoupon(
                orderDTO.getUserId(), 
                orderDTO.getCouponId()
            );
            
            order.setStatus(OrderStatus.COMPLETED);
            orderRepository.save(order);
        } catch (Exception e) {
            // 补偿操作
            if (order != null) {
                // 回滚订单状态
                order.setStatus(OrderStatus.CANCELLED);
                orderRepository.save(order);
                
                // 补偿库存
                inventoryService.increaseStock(
                    orderDTO.getProductId(), 
                    orderDTO.getQuantity()
                );
                
                // 补偿优惠券
                couponService.returnCoupon(
                    orderDTO.getUserId(), 
                    orderDTO.getCouponId()
                );
            }
            throw e;
        }
    }
}
```

#### 2.2.3 消息队列实现最终一致性

```java
@Service
public class OrderMessageService {
    
    @Autowired
    private OrderRepository orderRepository;
    
    @Autowired
    private RabbitTemplate rabbitTemplate;
    
    @Transactional
    public void createOrder(OrderDTO orderDTO) {
        // 1. 创建订单
        Order order = orderRepository.save(createOrderEntity(orderDTO));
        
        // 2. 发送消息到消息队列
        OrderMessage message = new OrderMessage();
        message.setOrderId(order.getId());
        message.setProductId(orderDTO.getProductId());
        message.setQuantity(orderDTO.getQuantity());
        message.setUserId(orderDTO.getUserId());
        message.setCouponId(orderDTO.getCouponId());
        
        rabbitTemplate.convertAndSend(
            "order.exchange",
            "order.create",
            message
        );
    }
}

// 消息消费者
@Component
public class OrderMessageConsumer {
    
    @Autowired
    private InventoryService inventoryService;
    
    @Autowired
    private CouponService couponService;
    
    @RabbitListener(queues = "order.queue")
    public void handleOrderMessage(OrderMessage message) {
        try {
            // 处理库存
            inventoryService.decreaseStock(
                message.getProductId(),
                message.getQuantity()
            );
            
            // 处理优惠券
            couponService.useCoupon(
                message.getUserId(),
                message.getCouponId()
            );
        } catch (Exception e) {
            // 发送补偿消息
            rabbitTemplate.convertAndSend(
                "compensation.exchange",
                "order.compensate",
                message
            );
        }
    }
}
```

## 3. 代码库回滚

### 3.1 Git版本回滚

```bash
# 1. 回滚到指定提交
git reset --hard <commit_id>

# 2. 回滚到上一个版本
git reset --hard HEAD^

# 3. 回滚特定文件
git checkout <commit_id> -- <file_path>

# 4. 创建回滚提交（推荐用于已推送到远程仓库的情况）
git revert <commit_id>
```

### 3.2 自动化回滚脚本

```bash
#!/bin/bash

# 回滚脚本
function rollback() {
    local target_commit=$1
    local branch=$2
    
    echo "开始回滚到提交: $target_commit"
    
    # 1. 切换到目标分支
    git checkout $branch
    
    # 2. 拉取最新代码
    git pull origin $branch
    
    # 3. 重置到目标提交
    git reset --hard $target_commit
    
    # 4. 强制推送到远程
    git push origin $branch --force
    
    echo "回滚完成"
}

# 使用示例
rollback "a1b2c3d" "master"
```

## 4. 部署版本回滚

### 4.1 Kubernetes部署回滚

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 3
  revisionHistoryLimit: 10  # 保留10个历史版本
  strategy:
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
    type: RollingUpdate
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myapp:1.0.0
        imagePullPolicy: IfNotPresent
```

回滚命令：

```bash
# 1. 查看部署历史
kubectl rollout history deployment/myapp

# 2. 回滚到上一个版本
kubectl rollout undo deployment/myapp

# 3. 回滚到指定版本
kubectl rollout undo deployment/myapp --to-revision=2
```

### 4.2 Docker Compose部署回滚

```yaml
# docker-compose.yml
version: '3'
services:
  app:
    image: myapp:${APP_VERSION:-latest}
    deploy:
      replicas: 3
      rollback_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
        monitor: 60s
        max_failure_ratio: 0.1
```

回滚脚本：

```bash
#!/bin/bash

# 回滚到上一个版本
function rollback() {
    local service=$1
    
    # 1. 获取当前版本
    current_version=$(docker service inspect --format '{{.Spec.TaskTemplate.ContainerSpec.Image}}' $service)
    
    # 2. 获取上一个版本
    previous_version=$(docker service inspect --format '{{.PreviousSpec.TaskTemplate.ContainerSpec.Image}}' $service)
    
    # 3. 执行回滚
    docker service update --image $previous_version $service
    
    echo "已回滚服务 $service 从 $current_version 到 $previous_version"
}

# 使用示例
rollback myapp_app
```

## 5. 数据版本回滚

### 5.1 数据库备份与恢复

```java
@Service
public class DatabaseBackupService {
    
    @Autowired
    private DataSource dataSource;
    
    public void backupDatabase() {
        try {
            // 1. 创建备份
            String backupFile = "backup_" + System.currentTimeMillis() + ".sql";
            ProcessBuilder pb = new ProcessBuilder(
                "mysqldump",
                "-u" + dataSource.getUsername(),
                "-p" + dataSource.getPassword(),
                dataSource.getDatabaseName(),
                "--result-file=" + backupFile
            );
            Process process = pb.start();
            process.waitFor();
            
            // 2. 记录备份信息
            BackupRecord record = new BackupRecord();
            record.setBackupTime(new Date());
            record.setBackupFile(backupFile);
            record.setStatus(BackupStatus.SUCCESS);
            backupRecordRepository.save(record);
            
        } catch (Exception e) {
            log.error("数据库备份失败", e);
            throw new BackupException("数据库备份失败", e);
        }
    }
    
    public void restoreDatabase(String backupFile) {
        try {
            // 1. 执行恢复
            ProcessBuilder pb = new ProcessBuilder(
                "mysql",
                "-u" + dataSource.getUsername(),
                "-p" + dataSource.getPassword(),
                dataSource.getDatabaseName(),
                "<",
                backupFile
            );
            Process process = pb.start();
            process.waitFor();
            
            // 2. 记录恢复信息
            RestoreRecord record = new RestoreRecord();
            record.setRestoreTime(new Date());
            record.setBackupFile(backupFile);
            record.setStatus(RestoreStatus.SUCCESS);
            restoreRecordRepository.save(record);
            
        } catch (Exception e) {
            log.error("数据库恢复失败", e);
            throw new RestoreException("数据库恢复失败", e);
        }
    }
}
```

### 5.2 使用数据库工具实现回滚

```java
@Service
public class DatabaseRollbackService {
    
    @Autowired
    private JdbcTemplate jdbcTemplate;
    
    public void rollbackToVersion(String version) {
        try {
            // 1. 开始事务
            jdbcTemplate.execute("START TRANSACTION");
            
            // 2. 执行回滚SQL
            List<String> rollbackSqls = getRollbackSqls(version);
            for (String sql : rollbackSqls) {
                jdbcTemplate.execute(sql);
            }
            
            // 3. 提交事务
            jdbcTemplate.execute("COMMIT");
            
        } catch (Exception e) {
            // 4. 回滚事务
            jdbcTemplate.execute("ROLLBACK");
            throw new RollbackException("数据回滚失败", e);
        }
    }
    
    private List<String> getRollbackSqls(String version) {
        // 从版本控制系统中获取回滚SQL
        return versionControlService.getRollbackSqls(version);
    }
}
```

## 6. 静态资源版本回滚

### 6.1 前端资源版本管理

```javascript
// webpack.config.js
module.exports = {
  output: {
    filename: '[name].[contenthash].js',
    path: path.resolve(__dirname, 'dist')
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: './src/index.html',
      filename: 'index.html'
    })
  ]
};

// 版本回滚脚本
const fs = require('fs');
const path = require('path');

function rollbackStatic(version) {
  // 1. 读取版本清单
  const manifest = JSON.parse(fs.readFileSync('dist/manifest.json'));
  
  // 2. 获取目标版本的资源路径
  const targetAssets = manifest[version];
  
  // 3. 更新HTML文件中的资源引用
  let html = fs.readFileSync('dist/index.html', 'utf8');
  Object.entries(targetAssets).forEach(([key, value]) => {
    html = html.replace(new RegExp(`${key}="[^"]+"`), `${key}="${value}"`);
  });
  
  // 4. 保存更新后的HTML
  fs.writeFileSync('dist/index.html', html);
  
  console.log(`已回滚到版本 ${version}`);
}
```

### 6.2 CDN资源回滚

```javascript
// CDN回滚脚本
const AWS = require('aws-sdk');
const s3 = new AWS.S3();

async function rollbackCDN(version) {
  try {
    // 1. 获取目标版本的资源列表
    const objects = await s3.listObjectsV2({
      Bucket: 'my-cdn-bucket',
      Prefix: `versions/${version}/`
    }).promise();
    
    // 2. 复制资源到当前版本目录
    for (const object of objects.Contents) {
      const sourceKey = object.Key;
      const targetKey = sourceKey.replace(`versions/${version}/`, '');
      
      await s3.copyObject({
        Bucket: 'my-cdn-bucket',
        CopySource: `my-cdn-bucket/${sourceKey}`,
        Key: targetKey
      }).promise();
    }
    
    // 3. 更新版本索引
    await s3.putObject({
      Bucket: 'my-cdn-bucket',
      Key: 'current-version',
      Body: version
    }).promise();
    
    console.log(`CDN资源已回滚到版本 ${version}`);
  } catch (error) {
    console.error('CDN回滚失败:', error);
    throw error;
  }
}
```

## 7. 回滚最佳实践

1. **事前准备**
   - 制定详细的回滚计划
   - 准备回滚脚本和工具
   - 进行回滚演练

2. **监控与告警**
   - 监控关键指标
   - 设置告警阈值
   - 建立快速响应机制

3. **版本控制**
   - 保留足够的历史版本
   - 清晰的版本命名规范
   - 完整的版本变更记录

4. **自动化工具**
   - 自动化回滚脚本
   - 一键回滚功能
   - 回滚进度可视化

5. **回滚策略**
   - 分级回滚（全量/增量）
   - 渐进式回滚
   - 回滚验证机制

