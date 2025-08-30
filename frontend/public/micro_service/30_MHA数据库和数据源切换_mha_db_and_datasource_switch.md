为了将 Spring Cloud 与 MySQL 高可用方案 **MHA（Master High Availability）** 结合使用，实现数据库主从切换时微服务无感知自动切换，可以按以下步骤进行架构设计与配置：

---

### **1. 整体架构设计**
![Spring Cloud + MHA 架构](https://example.com/spring-cloud-mha-arch.png)

- **MHA 层**：监控 MySQL 主从集群，自动完成故障检测、主库切换、数据补偿。
- **Spring Cloud 层**：通过服务发现（Eureka）、动态配置（Config Server）、客户端负载均衡（Ribbon）实现数据源动态切换。
- **中间件层**：使用代理（如 ProxySQL）或连接池（如 HikariCP）屏蔽底层数据库切换细节。

---

### **2. 关键组件配置**

#### **2.1 MHA 配置**
1. **部署 MHA Manager**：  
   - 在独立节点安装 MHA Manager，配置 `app1.cnf` 定义主从节点：
     ```ini
     [server default]
     manager_workdir=/var/log/mha
     master_binlog_dir=/var/lib/mysql
     user=mha_admin
     password=mha_password
     ssh_user=root

     [server1]
     hostname=master_a_ip
     candidate_master=1

     [server2]
     hostname=slave_b_ip

     [server3]
     hostname=slave_c_ip
     ```

2. **启用自动故障转移**：  
   - 配置 MHA 触发脚本（`master_ip_failover`），在切换时更新 ProxySQL 或 DNS。

---

#### **2.2 Spring Cloud 动态数据源配置**
1. **抽象动态数据源**：  
   使用 `AbstractRoutingDataSource` 实现运行时数据源切换：
   ```java
   public class DynamicDataSource extends AbstractRoutingDataSource {
       @Override
       protected Object determineCurrentLookupKey() {
           return DatabaseContextHolder.getDataSourceKey();
       }
   }
   ```

2. **监听配置中心变更**：  
   - 当 MHA 切换主库后，通过 Spring Cloud Config 动态下发新主库信息：
     ```yaml
     # application.yml
     spring:
       datasource:
         url: jdbc:mysql://${CURRENT_MASTER_IP}:3306/db
         username: user
         password: pass
     ```

3. **刷新数据源**：  
   通过 `@RefreshScope` 和 `/actuator/refresh` 端点触发数据源重建：
   ```java
   @RefreshScope
   @Configuration
   public class DataSourceConfig {
       @Bean
       @ConfigurationProperties(prefix = "spring.datasource")
       public DataSource dataSource() {
           return DataSourceBuilder.create().build();
       }
   }
   ```

---

### **3. 自动化切换流程**
#### **步骤 1：MHA 检测到主库故障**
- MHA Manager 通过心跳检测确认主库（`master_a`）不可用。
- 选举数据最新的从库（如 `slave_b`）为新主库，并修复未同步的 Binlog。

#### **步骤 2：更新代理或 DNS**
- **方案 1：ProxySQL 自动路由**  
  配置 MHA 触发脚本，切换 ProxySQL 后端主库：
  ```sql
  -- 更新 ProxySQL 主库组
  UPDATE mysql_servers SET status='OFFLINE' WHERE hostname='master_a_ip';
  UPDATE mysql_servers SET status='ONLINE' WHERE hostname='slave_b_ip';
  LOAD MYSQL SERVERS TO RUNTIME;
  ```

- **方案 2：动态 DNS 更新**  
  使用 `nsupdate` 或云厂商 API 修改 DNS 记录指向 `slave_b_ip`。

#### **步骤 3：Spring Cloud 动态感知**
- **通过 Config Server 推送新主库 IP**：  
  ```shell
  # 更新配置仓库中的数据库连接信息
  curl -X POST http://config-server:8888/actuator/bus-refresh
  ```

- **微服务自动刷新数据源**：  
  各服务通过 `@RefreshScope` 重新加载数据源配置，连接到新主库。

---

### **4. 代码实现示例**
#### **4.1 动态数据源切换**
```java
// 数据库上下文管理器
public class DatabaseContextHolder {
    private static final ThreadLocal<String> CONTEXT = new ThreadLocal<>();

    public static void setDataSourceKey(String key) {
        CONTEXT.set(key);
    }

    public static String getDataSourceKey() {
        return CONTEXT.get();
    }

    public static void clear() {
        CONTEXT.remove();
    }
}

// 数据源配置
@Configuration
public class DataSourceConfig {
    @Bean
    @Primary
    public DataSource dynamicDataSource() {
        Map<Object, Object> targetDataSources = new HashMap<>();
        targetDataSources.put("master", masterDataSource());
        targetDataSources.put("slave", slaveDataSource());

        DynamicDataSource dataSource = new DynamicDataSource();
        dataSource.setTargetDataSources(targetDataSources);
        dataSource.setDefaultTargetDataSource(masterDataSource());
        return dataSource;
    }

    @Bean
    @ConfigurationProperties("spring.datasource.master")
    public DataSource masterDataSource() {
        return DataSourceBuilder.create().build();
    }

    @Bean
    @ConfigurationProperties("spring.datasource.slave")
    public DataSource slaveDataSource() {
        return DataSourceBuilder.create().build();
    }
}
```

#### **4.2 监听配置变更**
```java
@Component
public class DatabaseSwitchListener {
    @Autowired
    private DynamicDataSource dynamicDataSource;

    @EventListener
    public void onApplicationEvent(EnvironmentChangeEvent event) {
        if (event.getKeys().contains("spring.datasource.url")) {
            // 重新初始化数据源
            dynamicDataSource.afterPropertiesSet();
        }
    }
}
```

---

### **5. 容灾与回退策略**
1. **数据一致性校验**：  
   - 使用 `pt-table-checksum` 定期校验主从数据一致性。
   - 发现不一致时，通过 `pt-table-sync` 自动修复。

2. **故障回退**：  
   - 若新主库（`slave_b`）出现问题，手动或通过 MHA 切回原主库（修复后需重新同步数据）。

3. **连接池清理**：  
   - 在数据源切换后，强制关闭旧连接池，避免残留连接访问失效主库：
     ```java
     ((HikariDataSource) oldDataSource).close();
     ```

---

### **6. 监控与告警**
- **监控指标**：  
  - MySQL 复制状态（`Seconds_Behind_Master`）。  
  - Spring Cloud 服务健康状态（通过 `actuator/health`）。  
  - 数据源连接池使用率（活跃连接、空闲连接）。

- **告警规则**：  
  - 主从延迟超过阈值（如 60 秒）。  
  - 数据源切换次数异常升高。  
  - 微服务数据库连接失败率突增。

---

### **总结**
通过整合 Spring Cloud 动态配置与 MHA 自动化故障转移能力，可以实现数据库主从切换时微服务的无缝衔接。核心要点包括：

1. **MHA 自动化切换**：确保主库故障时快速提升从库并修复数据。  
2. **动态数据源管理**：通过 Spring Cloud Config 和 `AbstractRoutingDataSource` 实现运行时切换。  
3. **代理层屏蔽细节**：使用 ProxySQL 或 DNS 更新，减少应用层改动。  
4. **完备的监控与回退**：确保数据一致性，快速响应异常。  

此方案适用于对数据库高可用和微服务连续性要求较高的场景，如金融、电商等核心业务系统。

