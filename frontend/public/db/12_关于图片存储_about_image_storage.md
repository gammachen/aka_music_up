# 图片存储方案技术选型分析

## 1. 存储方案对比维度
### 1.1 基础特性
| 维度        | BLOB存储                          | 外部文件系统                   |
|------------|----------------------------------|------------------------------|
| 存储位置    | 数据库内部                        | 独立存储服务/NAS              |
| 事务支持    | ACID特性完整                     | 需额外实现一致性机制          |
| 访问方式    | SQL接口直接访问                   | REST API/文件系统接口         |
| 运维管理    | 需专业DBA维护                    | 可独立进行文件系统维护        |
| 权限管理    | 通过SQL权限精细控制               | 依赖操作系统权限体系          |

### 1.2 性能指标
- **吞吐量**：
  - BLOB存储受数据库连接池限制（参考tech_design_v3.md第5节）
  - 文件系统可水平扩展（见23_about_scaling_db.md NoSQL方案）
- **延迟**：
  - 小文件(<1MB)场景BLOB延迟降低30-40%
  - 大文件(>10MB)场景文件系统优势明显

## 2. 扩展性成本分析
### 2.1 存储扩容
- **BLOB方案**：
  - 需数据库垂直扩容（23_about_scaling_db.md第2章）
  - 单实例存储上限受RDBMS限制
- MySQL实测数据：
  - 备份时间减少50%（10GB数据库测试）
  - 备份文件大小缩减70%
- **文件系统**：
  - 支持分布式存储（参考nodejs_analysis.md S3配置）
  - 按需扩展存储节点

### 2.2 流量承载
| 场景        | BLOB存储                    | 文件系统                 |
|------------|----------------------------|-------------------------|
| 高频读      | 需增加数据库从库            | 可直接对接CDN          |
| 突发写入    | 受事务锁限制                | 支持分片上传机制       |

## 3. 项目适配方案

### 3.0 外部存储核心风险
1. **事务隔离缺失**：文件操作无法实现ACID特性
2. **回滚机制失效**：文件删除后无法通过事务回滚恢复
3. **备份工具不兼容**：需额外实现文件系统备份策略
4. **权限管理绕过**：GRANT/REVOKE对文件系统无效
5. **路径验证缺失**：数据库不验证路径有效性（详见3.1节）
6. **版本控制困难**：文件修改历史难以追踪

### 3.1 路径验证机制
```text
/media/images/
├── equation_solver_visual/
├── kalman_animation/
└── product_rule_*/
```
**路径同步要求**：
1. 文件操作日志需记录到数据库（参考tech_design_v3.md审计模块）
2. 定期执行路径校验脚本：
```bash
# 与数据库记录比对路径有效性
find /media/images -type f | xargs -I {} grep -q {} backup/db/media_registry.csv || echo "{} 路径失效"
```
3. 实现文件操作hook：
```python
# 文件删除时同步数据库状态
def on_file_delete(path):
    db.execute("UPDATE media SET status='deleted' WHERE path=%s", (path,))
```
### 3.1 路径规则实现
采用混合存储策略：
```text
/meida/images/  # 存储原始图片（文件系统）
/thumbnails/    # 存储缩略图（BLOB+缓存）

**批处理示例**（关联/media/images目录）：
```bash
# 批量调整图片尺寸（参考backend/README_about_deploy.md备份策略）
find /media/images -name '*.jpg' -exec convert {} -resize 1024x768 {} \;

# 结合项目实际目录结构（检测到存在equation_solver_visual等子目录）
rsync -avz /media/images/ backup-server:/nightly_backup/$(date +%Y%m%d)
```
```

### 3.2 CDN加速集成
- **元数据更新**：数据库事务提交后触发CDN预热（需实现after_commit钩子）
- **版本控制**：文件hash值作为缓存标识（参考tech_design_v3.md版本机制）

## 4. 选型建议
| 场景                | 推荐方案          | 技术依据                          |
|--------------------|------------------|---------------------------------|
| 用户头像等小文件    | BLOB存储         | 高并发读取+强一致性要求          |
| 视频/高清图片       | 文件系统+S3       | 23_about_scaling_db.md扩容建议   |
| 临时缓存文件        | 内存数据库        | 高吞吐量需求（见nodejs_analysis.md） |

## 5. 运维监控要点
- **BLOB存储**：监控表空间增长速率（警戒线参考tech_design_v3.md第8节）
- **文件系统**：设置inode使用率告警（预防小文件存储瓶颈）