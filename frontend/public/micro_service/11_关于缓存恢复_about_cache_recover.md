# 缓存崩溃与快速恢复指南

## 1. 缓存集群架构

### 1.1 取模机制

对于取模机制，如果其中一个实例坏了，摘除此实例将导致大量缓存不命中，则瞬间大流量可能导致后端DB/服务出现问题。

#### 1.1.1 主从机制实现

```java
public class CacheCluster {
    private final Map<String, CacheNode> primaryNodes = new ConcurrentHashMap<>();
    private final Map<String, CacheNode> backupNodes = new ConcurrentHashMap<>();
    
    public void addNode(String nodeId, CacheNode node, boolean isPrimary) {
        if (isPrimary) {
            primaryNodes.put(nodeId, node);
        } else {
            backupNodes.put(nodeId, node);
        }
    }
    
    public void handleNodeFailure(String nodeId) {
        CacheNode failedNode = primaryNodes.remove(nodeId);
        if (failedNode != null) {
            // 从备份节点中选取一个作为新的主节点
            CacheNode backupNode = selectBackupNode(nodeId);
            if (backupNode != null) {
                primaryNodes.put(nodeId, backupNode);
                backupNodes.remove(backupNode.getId());
            }
        }
    }
}
```

#### 1.1.2 数据迁移方案

```java
public class CacheMigration {
    private final CacheCluster oldCluster;
    private final CacheCluster newCluster;
    
    public void migrateData() {
        // 1. 预热新集群
        warmUpNewCluster();
        
        // 2. 逐步迁移流量
        migrateTraffic();
        
        // 3. 验证数据一致性
        verifyDataConsistency();
    }
    
    private void warmUpNewCluster() {
        // 使用后台任务预热数据
        ScheduledExecutorService executor = Executors.newScheduledThreadPool(1);
        executor.scheduleAtFixedRate(() -> {
            List<String> hotKeys = getHotKeys();
            for (String key : hotKeys) {
                Object value = oldCluster.get(key);
                newCluster.put(key, value);
            }
        }, 0, 5, TimeUnit.MINUTES);
    }
}
```

### 1.2 一致性哈希机制

对于一致性哈希机制，如果其中一个实例坏了，摘除此实例只影响一致性哈希环上的部分缓存不命中，不会导致大量缓存瞬间回源到后端DB/服务。

#### 1.2.1 一致性哈希实现

```java
public class ConsistentHash {
    private final TreeMap<Long, CacheNode> circle = new TreeMap<>();
    private final int virtualNodes;
    
    public ConsistentHash(int virtualNodes) {
        this.virtualNodes = virtualNodes;
    }
    
    public void addNode(CacheNode node) {
        for (int i = 0; i < virtualNodes; i++) {
            String virtualNodeName = node.getId() + "#" + i;
            long hash = hash(virtualNodeName);
            circle.put(hash, node);
        }
    }
    
    public CacheNode getNode(String key) {
        if (circle.isEmpty()) {
            return null;
        }
        long hash = hash(key);
        if (!circle.containsKey(hash)) {
            SortedMap<Long, CacheNode> tail = circle.tailMap(hash);
            hash = tail.isEmpty() ? circle.firstKey() : tail.firstKey();
        }
        return circle.get(hash);
    }
}
```

## 2. 快速恢复方案

### 2.1 主从冗余机制

```java
public class CacheRecovery {
    private final CacheCluster cluster;
    private final CacheWarmupService warmupService;
    
    public void handleClusterFailure() {
        // 1. 检测故障节点
        List<String> failedNodes = detectFailedNodes();
        
        // 2. 启用备份节点
        for (String nodeId : failedNodes) {
            cluster.activateBackupNode(nodeId);
        }
        
        // 3. 预热缓存数据
        warmupService.warmup(failedNodes);
    }
}
```

### 2.2 用户降级策略

```java
public class UserDegradation {
    private final CacheDegradationService degradationService;
    private final CacheWarmupService warmupService;
    
    public void handleCacheFailure() {
        // 1. 根据系统负载确定降级比例
        double degradationRatio = calculateDegradationRatio();
        
        // 2. 实施用户降级
        degradationService.degradeUsers(degradationRatio);
        
        // 3. 后台预热缓存
        warmupService.warmupInBackground();
        
        // 4. 逐步恢复服务
        graduallyRecoverService();
    }
    
    private void graduallyRecoverService() {
        ScheduledExecutorService executor = Executors.newScheduledThreadPool(1);
        executor.scheduleAtFixedRate(() -> {
            // 每5分钟减少10%的降级比例
            degradationService.reduceDegradationRatio(0.1);
        }, 5, 5, TimeUnit.MINUTES);
    }
}
```

### 2.3 缓存预热实现

```java
public class CacheWarmupService {
    private final CacheCluster cluster;
    private final DataSource dataSource;
    
    public void warmupInBackground() {
        // 1. 识别热点数据
        List<String> hotKeys = identifyHotKeys();
        
        // 2. 多线程预热
        ExecutorService executor = Executors.newFixedThreadPool(10);
        for (String key : hotKeys) {
            executor.submit(() -> {
                try {
                    Object value = dataSource.load(key);
                    cluster.put(key, value);
                } catch (Exception e) {
                    log.error("Failed to warmup key: " + key, e);
                }
            });
        }
    }
    
    private List<String> identifyHotKeys() {
        // 从监控系统获取热点数据
        return monitoringService.getHotKeys();
    }
}
```

## 3. 最佳实践

### 3.1 预防措施

1. **多级缓存架构**
   ```java
   public class MultiLevelCache {
       private final LocalCache localCache;
       private final DistributedCache distributedCache;
       
       public Object get(String key) {
           // 1. 检查本地缓存
           Object value = localCache.get(key);
           if (value != null) {
               return value;
           }
           
           // 2. 检查分布式缓存
           value = distributedCache.get(key);
           if (value != null) {
               localCache.put(key, value);
               return value;
           }
           
           // 3. 回源加载
           value = loadFromSource(key);
           if (value != null) {
               distributedCache.put(key, value);
               localCache.put(key, value);
           }
           return value;
       }
   }
   ```

2. **监控告警**
   ```java
   public class CacheMonitor {
       private final CacheCluster cluster;
       
       public void monitor() {
           // 1. 监控缓存命中率
           monitorHitRate();
           
           // 2. 监控节点健康状态
           monitorNodeHealth();
           
           // 3. 监控响应时间
           monitorResponseTime();
       }
   }
   ```

### 3.2 恢复策略

1. **分级恢复**
   - 核心业务优先恢复
   - 非核心业务逐步恢复
   - 根据系统负载动态调整恢复速度

2. **数据一致性保证**
   ```java
   public class DataConsistency {
       private final CacheCluster cluster;
       private final DataSource dataSource;
       
       public void verifyConsistency() {
           // 1. 抽样检查
           List<String> sampleKeys = getSampleKeys();
           for (String key : sampleKeys) {
               Object cacheValue = cluster.get(key);
               Object sourceValue = dataSource.load(key);
               if (!Objects.equals(cacheValue, sourceValue)) {
                   // 记录不一致并修复
                   logInconsistency(key, cacheValue, sourceValue);
                   cluster.put(key, sourceValue);
               }
           }
       }
   }
   ```

### 3.3 运维建议

1. **定期演练**
   - 模拟节点故障
   - 测试恢复流程
   - 验证监控告警

2. **文档维护**
   - 更新应急预案
   - 记录故障处理经验
   - 完善运维手册

