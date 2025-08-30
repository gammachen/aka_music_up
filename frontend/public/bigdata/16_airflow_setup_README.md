使用 Apache Airflow 构建一个简单的 **POC（Proof of Concept）** 可以帮助你快速验证 Airflow 的基本功能（如任务调度、依赖管理、监控等）。以下是一个最小化的 POC 实施步骤，涵盖安装、DAG 编写、任务执行和验证。

---

### **一、POC 目标**
1. **验证 Airflow 的任务调度能力**：定义一个简单的 DAG，包含两个任务。
2. **展示任务依赖关系**：确保任务按顺序执行。
3. **通过 Web 界面监控任务状态**：查看任务运行日志和状态。

---

### **二、实施步骤**

#### **1. 安装 Airflow 3.x 及标准 Provider**
```bash
# 使用虚拟环境（可选）
python3 -m venv airflow_venv
source airflow_venv/bin/activate

# 安装 Airflow 3.0.2 及标准 Provider
pip install "apache-airflow==3.0.2" "apache-airflow-providers-standard"

pip install "apache-airflow" "apache-airflow-providers-standard"
```

#### **2. 初始化数据库**
```bash
airflow db migrate
```

#### **3. 创建管理员用户**
```bash
airflow users create \
  --username admin \
  --password admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com
```

#### **4. 启动 Airflow 服务**
```bash
# 启动 API Server（默认端口 8080）
airflow api-server -p 8080 &

# 启动 Scheduler
airflow scheduler &

# 启动 DAG Processor（Airflow 3.x 必须单独启动）
airflow dag-processor &
```

---

#### **5. 编写 DAG 文件**
在 Airflow 的 `dags` 目录下创建一个 Python 文件（例如 `poc_dag.py`）：

```python
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta

# 默认参数
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# 定义任务函数
def task1():
    print("Executing Task 1: Data Extraction")

def task2():
    print("Executing Task 2: Data Transformation")

# 定义 DAG
with DAG(
    dag_id='poc_dag',
    default_args=default_args,
    start_date=datetime(2025, 6, 19),  # 设置为当前日期或过去日期
    # schedule_interval='@once',         # 仅运行一次
    catchup=False                       # 不回填历史任务
) as dag:
    # 定义任务
    task_1 = PythonOperator(
        task_id='task_1',
        python_callable=task1,
        dag=dag,
    )

    task_2 = PythonOperator(
        task_id='task_2',
        python_callable=task2,
        dag=dag,
    )

    # 设置任务依赖关系
    task_1 >> task_2
```

---

#### **6. 访问 Airflow Web 界面**
1. 打开浏览器，访问 [http://localhost:8080](http://localhost:8080)。
2. 使用管理员账号登录。
3. 在 DAG 列表中找到 `poc_dag`，点击进入 DAG 页面。
4. 点击 **"Trigger DAG"** 手动触发任务执行。

---

#### **7. 验证执行结果**
1. **查看任务状态**：
   - 在 DAG 图形视图中，绿色表示任务成功，红色表示失败。
2. **查看日志**：
   - 点击任务名称，进入日志页面，验证 `print` 输出是否显示。

---

### **三、POC 扩展建议**
1. **添加更多任务**：
   - 使用 `BashOperator` 执行 Shell 命令。
   - 使用 `SqlOperator` 操作数据库。
2. **调整调度策略**：
   - 修改 `schedule_interval` 为 `@daily` 或自定义 Cron 表达式。
3. **集成外部工具**：
   - 使用 `HttpOperator` 调用 API。
   - 使用 `EmailOperator` 发送通知。

---

### **四、常见问题**
1. **DAG 未显示**：
   - 确保 DAG 文件保存在 `AIRFLOW_HOME/dags/` 目录下。
   - 检查文件名是否符合命名规范（无 `.pyc` 等后缀）。
2. **任务失败**：
   - 检查日志中的错误信息。
   - 调整 `retries` 和 `retry_delay` 参数。
3. **Web 界面无法访问**：
   - 确认 API Server 是否正常运行。
   - 检查防火墙设置（如本地开发环境需允许 `localhost:8080`）。

---

### **五、总结**
通过以上步骤，你可以快速构建一个最小化的 Airflow POC，验证其核心功能：
- DAG 的定义与任务依赖管理。
- 任务调度与执行监控。
- Airflow Web 界面的基本使用。

后续可根据需求扩展为更复杂的数据流水线（如 ETL 流程、机器学习训练等）。