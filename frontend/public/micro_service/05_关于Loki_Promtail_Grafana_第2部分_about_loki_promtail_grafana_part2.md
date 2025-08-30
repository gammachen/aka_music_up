# Loki + Promtail + Grafana 日志收集方案详解（续）

## 4.3 与 Prometheus 集成的可观测性案例（续）

#### 4.3.1 场景描述

一个云原生应用平台，需要将日志和指标数据结合起来，提供完整的可观测性解决方案。

#### 4.3.2 集成架构

```
应用 → Promtail → Loki → Grafana ← Prometheus ← 应用
```

#### 4.3.3 实现步骤

1. **部署 Prometheus 和 Loki**：

```bash
# 部署 Prometheus Stack
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace

# 部署 Loki Stack
helm install loki grafana/loki-stack \
  --set grafana.enabled=false \
  --namespace monitoring
```

2. **配置 Grafana 数据源**：

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus-server:80
    version: 1
    editable: false
    isDefault: false

  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    version: 1
    editable: false
    isDefault: true
```

3. **创建混合仪表板**：

```
# Grafana 仪表板 JSON 模型（部分）
{
  "panels": [
    {
      "title": "HTTP Request Rate",
      "datasource": "Prometheus",
      "targets": [
        {
          "expr": "sum(rate(http_requests_total[5m])) by (service)"
        }
      ]
    },
    {
      "title": "Error Logs",
      "datasource": "Loki",
      "targets": [
        {
          "expr": "{app=\"myapp\"} |= \"error\" | json"
        }
      ]
    },
    {
      "title": "Latency vs Error Correlation",
      "datasource": "Prometheus",
      "targets": [
        {
          "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))"
        }
      ]
    }
  ]
}
```

4. **设置关联查询**：

在 Grafana 中配置从指标到日志的钻取功能，例如：

- 从 HTTP 错误率图表点击到相关的错误日志
- 从高延迟时间段查看对应时间范围的日志

5. **创建复合告警**：

```yaml
groups:
  - name: combined_alerts
    rules:
      - alert: HighErrorRateWithLogs
        expr: sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) > 0.05
        for: 2m
        annotations:
          summary: High error rate detected
          description: "Error rate is above 5%. Check logs: https://grafana/explore?orgId=1&left=[\"now-1h\",\"now\",\"Loki\",{\"expr\":\"{app=\\\"myapp\\\"} |= \\\"error\\\"\"}"
```

## 5. 常见问题与解决方案

### 5.1 性能问题

#### 5.1.1 Loki 查询性能慢

**症状**：
- 查询响应时间长
- Grafana 仪表板加载缓慢

**解决方案**：
- 优化标签使用，减少高基数标签
- 增加查询前端缓存
- 调整查询范围和限制
- 使用并行查询
- 考虑使用微服务模式部署

#### 5.1.2 Promtail 资源使用高

**症状**：
- Promtail 占用过多 CPU 或内存
- 日志传输延迟增加

**解决方案**：
- 减少标签数量和复杂度
- 优化正则表达式
- 增加资源限制
- 考虑使用批处理模式

### 5.2 存储问题

#### 5.2.1 存储空间快速增长

**症状**：
- Loki 存储空间迅速耗尽
- 磁盘使用率高

**解决方案**：
- 配置适当的保留策略
- 使用压缩功能
- 实施日志采样
- 使用对象存储作为后端
- 配置索引和块的生命周期管理

#### 5.2.2 索引性能下降

**症状**：
- 查询变慢
- 索引操作耗时增加

**解决方案**：
- 优化标签策略，减少高基数标签
- 配置适当的索引周期
- 使用 BoltDB Shipper 模式
- 考虑使用更强大的索引存储后端

### 5.3 部署问题

#### 5.3.1 Kubernetes 集成问题

**症状**：
- Promtail 无法收集容器日志
- 标签不正确或缺失

**解决方案**：
- 确保 Promtail 有适当的权限
- 检查 Kubernetes 服务发现配置
- 验证日志路径映射
- 使用 Helm 图表进行部署

#### 5.3.2 多租户配置问题

**症状**：
- 租户数据混合或访问控制问题
- 授权错误

**解决方案**：
- 正确配置 X-Scope-OrgID 头
- 设置适当的租户限制
- 使用 auth_enabled: true
- 配置租户特定的数据源

## 6. 总结与最佳实践

### 6.1 Loki + Promtail + Grafana 优势

- **资源效率**：相比全文索引系统，资源消耗低
- **成本效益**：存储成本低，适合大规模部署
- **简单部署**：部署和维护简单
- **云原生**：为 Kubernetes 环境设计
- **与 Prometheus 集成**：提供完整的可观测性解决方案
- **多租户支持**：原生支持多租户

### 6.2 最佳实践

#### 6.2.1 标签策略

- 使用少量、低基数的标签（如环境、应用名称、服务名称）
- 避免使用高基数标签（如用户ID、请求ID）
- 使用静态标签标识日志源
- 保持标签命名一致性

#### 6.2.2 查询优化

- 使用标签过滤缩小查询范围
- 限制时间范围
- 使用管道操作符优化查询
- 利用 LogQL 聚合功能

#### 6.2.3 存储优化

- 配置适当的保留策略
- 使用对象存储作为长期存储
- 实施压缩策略
- 配置索引和块的生命周期管理

#### 6.2.4 可观测性集成

- 将 Loki 与 Prometheus 和 Tempo 集成
- 创建混合仪表板
- 实施跨数据源关联
- 使用 Grafana Explore 进行交互式查询

### 6.3 适用场景

- **Kubernetes 环境**：原生支持容器化环境
- **微服务架构**：适合分布式系统日志收集
- **资源受限环境**：适合资源有限的环境
- **大规模部署**：适合大量日志数据的场景
- **多租户需求**：适合需要租户隔离的场景

### 6.4 未来发展

- 增强 LogQL 查询能力
- 改进与 Prometheus 和 Tempo 的集成
- 扩展告警功能
- 优化存储效率
- 增强多租户功能

## 7. 参考资源

- [Grafana Loki 官方文档](https://grafana.com/docs/loki/latest/)
- [Promtail 官方文档](https://grafana.com/docs/loki/latest/clients/promtail/)
- [Grafana 官方文档](https://grafana.com/docs/grafana/latest/)
- [LogQL 查询语言文档](https://grafana.com/docs/loki/latest/logql/)
- [Loki 架构文档](https://grafana.com/docs/loki/latest/fundamentals/architecture/)
- [Grafana Labs 博客](https://grafana.com/blog/)
- [Loki GitHub 仓库](https://github.com/grafana/loki)