# Oozie作业编排与调度的架构与示例

## Oozie架构概述

Oozie是一个基于工作流引擎的Hadoop作业调度系统，主要包含以下核心组件：

1. **Oozie Server**：负责接收客户端请求、解析工作流定义、调度作业执行
2. **Oozie Client**：提供命令行和REST API接口与服务器交互
3. **数据库**：存储工作流定义、作业状态等信息（通常使用MySQL或PostgreSQL）
4. **Hadoop集群**：实际执行工作流中定义的MapReduce、Pig、Hive等作业
5. **协调器引擎(Coordinator Engine)**：基于时间或数据可用性触发工作流
6. **Bundle引擎**：管理一组协调器作业

## 核心概念

1. **Workflow**：定义一系列动作及其依赖关系的DAG（有向无环图）
2. **Coordinator**：基于时间或数据可用性调度工作流
3. **Bundle**：将多个协调器作业打包成一个逻辑单元

## 工作流示例

### 简单工作流示例 (workflow.xml)

```xml
<workflow-app name="sample-wf" xmlns="uri:oozie:workflow:0.5">
    <start to="first-action"/>
    
    <action name="first-action">
        <shell xmlns="uri:oozie:shell-action:0.2">
            <job-tracker>${jobTracker}</job-tracker>
            <name-node>${nameNode}</name-node>
            <configuration>
                <property>
                    <name>mapred.job.queue.name</name>
                    <value>${queueName}</value>
                </property>
            </configuration>
            <exec>echo</exec>
            <argument>Hello Oozie</argument>
        </shell>
        <ok to="second-action"/>
        <error to="fail"/>
    </action>
    
    <action name="second-action">
        <java>
            <job-tracker>${jobTracker}</job-tracker>
            <name-node>${nameNode}</name-node>
            <main-class>com.example.MyJavaJob</main-class>
            <arg>inputPath=${inputDir}</arg>
            <arg>outputPath=${outputDir}</arg>
        </java>
        <ok to="end"/>
        <error to="fail"/>
    </action>
    
    <kill name="fail">
        <message>Workflow failed, error message[${wf:errorMessage(wf:lastErrorNode())}]</message>
    </kill>
    
    <end name="end"/>
</workflow-app>
```

### 协调器示例 (coordinator.xml)

```xml
<coordinator-app name="sample-coord" frequency="${coord:days(1)}"
    start="2023-01-01T00:00Z" end="2023-12-31T00:00Z" timezone="UTC"
    xmlns="uri:oozie:coordinator:0.4">
    
    <controls>
        <timeout>10</timeout>
        <concurrency>1</concurrency>
        <execution>FIFO</execution>
    </controls>
    
    <datasets>
        <dataset name="input1" frequency="${coord:days(1)}"
            initial-instance="2023-01-01T00:00Z" timezone="UTC">
            <uri-template>hdfs://namenode/path/to/data/${YEAR}/${MONTH}/${DAY}</uri-template>
        </dataset>
    </datasets>
    
    <input-events>
        <data-in name="input" dataset="input1">
            <instance>${coord:current(0)}</instance>
        </data-in>
    </input-events>
    
    <action>
        <workflow>
            <app-path>hdfs://namenode/path/to/workflow</app-path>
            <configuration>
                <property>
                    <name>inputDir</name>
                    <value>${coord:dataIn('input')}</value>
                </property>
                <property>
                    <name>outputDir</name>
                    <value>hdfs://namenode/path/to/output/${YEAR}/${MONTH}/${DAY}</value>
                </property>
            </configuration>
        </workflow>
    </action>
</coordinator-app>
```

### Bundle示例 (bundle.xml)

```xml
<bundle-app name="sample-bundle" xmlns="uri:oozie:bundle:0.2">
    <parameters>
        <property>
            <name>jobTracker</name>
            <value>jt.example.com:8032</value>
        </property>
        <property>
            <name>nameNode</name>
            <value>hdfs://nn.example.com:8020</value>
        </property>
    </parameters>
    
    <coordinator name="coord1">
        <app-path>hdfs://namenode/path/to/coordinator1</app-path>
        <configuration>
            <property>
                <name>inputDir</name>
                <value>/data/source1</value>
            </property>
        </configuration>
    </coordinator>
    
    <coordinator name="coord2">
        <app-path>hdfs://namenode/path/to/coordinator2</app-path>
        <configuration>
            <property>
                <name>inputDir</name>
                <value>/data/source2</value>
            </property>
        </configuration>
    </coordinator>
</bundle-app>
```

## 实际应用场景示例

### 数据ETL管道

```xml
<workflow-app name="etl-pipeline" xmlns="uri:oozie:workflow:0.5">
    <start to="sqoop-import"/>
    
    <!-- 从RDBMS导入数据到HDFS -->
    <action name="sqoop-import">
        <sqoop xmlns="uri:oozie:sqoop-action:0.2">
            <job-tracker>${jobTracker}</job-tracker>
            <name-node>${nameNode}</name-node>
            <command>import --connect jdbc:mysql://db.example.com/source --table customers --target-dir /data/raw/customers/${YEAR}${MONTH}${DAY} -m 1</command>
        </sqoop>
        <ok to="hive-transform"/>
        <error to="fail"/>
    </action>
    
    <!-- 使用Hive转换数据 -->
    <action name="hive-transform">
        <hive xmlns="uri:oozie:hive-action:0.2">
            <job-tracker>${jobTracker}</job-tracker>
            <name-node>${nameNode}</name-node>
            <script>transform_customers.hql</script>
            <param>inputDir=/data/raw/customers/${YEAR}${MONTH}${DAY}</param>
            <param>outputDir=/data/processed/customers/${YEAR}${MONTH}${DAY}</param>
        </hive>
        <ok to="pig-aggregate"/>
        <error to="fail"/>
    </action>
    
    <!-- 使用Pig进行聚合 -->
    <action name="pig-aggregate">
        <pig>
            <job-tracker>${jobTracker}</job-tracker>
            <name-node>${nameNode}</name-node>
            <script>aggregate_sales.pig</script>
            <param>input=/data/processed/customers/${YEAR}${MONTH}${DAY}</param>
            <param>output=/data/aggregated/sales/${YEAR}${MONTH}${DAY}</param>
        </pig>
        <ok to="end"/>
        <error to="fail"/>
    </action>
    
    <kill name="fail">
        <message>ETL Pipeline failed at [${wf:errorMessage(wf:lastErrorNode())}]</message>
    </kill>
    
    <end name="end"/>
</workflow-app>
```

## 最佳实践

1. **模块化设计**：将复杂工作流分解为多个小工作流
2. **参数化配置**：使用变量代替硬编码值
3. **错误处理**：为每个动作定义明确的错误处理路径
4. **监控和告警**：集成监控系统跟踪作业状态
5. **资源管理**：合理设置并发和超时参数
6. **版本控制**：将工作流定义文件纳入版本控制系统

Oozie提供了强大的作业编排能力，特别适合复杂的数据处理管道和定期执行的批处理作业。