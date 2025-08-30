# Apache Superset 技术文档

## 1. 组件简介
Apache Superset 是一个现代化、开源的数据可视化和数据分析平台，支持自助式仪表盘和报表。

## 2. 主要功能
- 可视化仪表盘与报表设计
- 支持多种数据源（SQL、Druid、Presto 等）
- 丰富的图表类型与交互分析
- 权限与多租户管理

## 3. 架构原理
Superset 采用前后端分离架构，前端基于 React，后端基于 Flask。通过 SQLAlchemy 连接多种数据源，支持自定义扩展。

## 4. 典型应用场景
- 业务数据可视化
- 自助式数据分析
- 实时监控与报表

## 5. 与本平台的集成方式
- 作为数据服务层的可视化门户
- 与 Hive、Presto、Trino、Spark SQL 等数据源集成

## 6. 优势与局限
**优势：**
- 交互性强，易用性高
- 支持多数据源和多种图表

**局限：**
- 对超大数据量需优化查询
- 依赖底层数据源性能

## 7. 关键配置与运维建议
- 合理配置数据库连接池
- 优化查询与缓存策略
- 配置权限与安全策略

## 8. 相关开源社区与文档链接
- 官方文档：https://superset.apache.org/docs/intro
- GitHub：https://github.com/apache/superset 
- Docks: https://superset.apache.org/docs/configuration/databases/#supported-databases-and-dependencies
  - SQLite	No additional library needed	sqlite://path/to/file.db?check_same_thread=false

```shell
ERROR: (builtins.NoneType) None [SQL: (sqlite3.OperationalError) unable to open database file (Background on this error at: https://sqlalche.me/e/14/e3q8)] (Background on this error at: https://sqlalche.me/e/14/dbapi)

{
    "errors": [
        {
            "message": "(builtins.NoneType) None\n[SQL: (sqlite3.OperationalError) unable to open database file\n(Background on this error at: https://sqlalche.me/e/14/e3q8)]\n(Background on this error at: https://sqlalche.me/e/14/dbapi)",
            "error_type": "GENERIC_DB_ENGINE_ERROR",
            "level": "error",
            "extra": {
                "engine_name": "SQLite",
                "issue_codes": [
                    {
                        "code": 1002,
                        "message": "Issue 1002 - The database returned an unexpected error."
                    }
                ]
            }
        }
    ]
}

superset_app          | 192.168.65.1 - - [20/Jun/2025:16:27:08 +0000] "POST /api/v1/database/test_connection/ HTTP/1.1" 500 440 "http://localhost:8088/superset/welcome/" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15"
superset_app          | 2025-06-20 16:27:08,037:WARNING:superset.views.error_handling:SupersetErrorsException
superset_app          | Traceback (most recent call last):
superset_app          |   File "/app/superset/commands/database/test_connection.py", line 170, in run
superset_app          |     raise DBAPIError(ex_str or None, None, None)
superset_app          | sqlalchemy.exc.DBAPIError: (builtins.NoneType) None
superset_app          | [SQL: (sqlite3.OperationalError) unable to open database file
superset_app          | (Background on this error at: https://sqlalche.me/e/14/e3q8)]
superset_app          | (Background on this error at: https://sqlalche.me/e/14/dbapi)
superset_app          |
superset_app          | The above exception was the direct cause of the following exception:
superset_app          |
superset_app          | Traceback (most recent call last):
superset_app          |   File "/usr/local/lib/python3.10/site-packages/flask/app.py", line 1484, in full_dispatch_request
superset_app          |     rv = self.dispatch_request()
superset_app          |   File "/usr/local/lib/python3.10/site-packages/flask/app.py", line 1469, in dispatch_request
superset_app          |     return self.ensure_sync(self.view_functions[rule.endpoint])(**view_args)
superset_app          |   File "/usr/local/lib/python3.10/site-packages/flask_appbuilder/security/decorators.py", line 95, in wraps
superset_app          |     return f(self, *args, **kwargs)
superset_app          |   File "/app/superset/views/base_api.py", line 119, in wraps
superset_app          |     duration, response = time_function(f, self, *args, **kwargs)
superset_app          |   File "/app/superset/utils/core.py", line 1364, in time_function
superset_app          |     response = func(*args, **kwargs)
superset_app          |   File "/app/superset/utils/log.py", line 303, in wrapper
superset_app          |     value = f(*args, **kwargs)
superset_app          |   File "/app/superset/views/base_api.py", line 91, in wraps
superset_app          |     return f(self, *args, **kwargs)
superset_app          |   File "/app/superset/databases/api.py", line 1215, in test_connection
superset_app          |     TestConnectionDatabaseCommand(item).run()
superset_app          |   File "/app/superset/commands/database/test_connection.py", line 199, in run
superset_app          |     raise SupersetErrorsException(errors) from ex
superset_app          | superset.exceptions.SupersetErrorsException: [SupersetError(message='(builtins.NoneType) None\n[SQL: (sqlite3.OperationalError) unable to open database file\n(Background on this error at: https://sqlalche.me/e/14/e3q8)]\n(Background on this error at: https://sqlalche.me/e/14/dbapi)', error_type=<SupersetErrorType.GENERIC_DB_ENGINE_ERROR: 'GENERIC_DB_ENGINE_ERROR'>, level=<ErrorLevel.ERROR: 'error'>, extra={'engine_name': 'SQLite', 'issue_codes': [{'code': 1002, 'message': 'Issue 1002 - The database returned an unexpected error.'}]})]
```

```shell
sqlite:////Users/shhaofu/Code/cursor-projects/aka_music/backend/instance/aka_music.db?check_same_thread=false

sqlite:///Users/shhaofu/Code/cursor-projects/p-sso-web/p-sso-b-web/instance/app_b.db?check_same_thread=false
```

https://github.com/dbeaver/dbeaver/releases/download/25.1.0/dbeaver-ce-25.1.0-macos-aarch64.dmg

jdbc:mysql://localhost:3306/movie_database?allowPublicKeyRetrieval=true