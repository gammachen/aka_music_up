Read file: README.md
下面是根据你的所有脚本和 DAG 流程图，重新整理和丰富后的 README 技术方案文档。内容包含了所有脚本的代码块、流程图（Mermaid 语法）、以及每个脚本的功能说明，便于团队理解和复用。

---

# Airflow POC 技术方案

## 一、项目简介

本项目通过 Apache Airflow 3.x 构建多种数据处理和业务自动化流程，涵盖 ETL、动态定价、金融风控等场景。文档包含了所有 DAG 脚本、流程图（Mermaid）、以及关键技术点说明，适合团队快速复用和扩展。

---

## 二、环境安装与基础配置

```bash
# 推荐使用虚拟环境
python3 -m venv airflow_venv
source airflow_venv/bin/activate

# 安装 Airflow 3.x 及标准 Provider
pip install "apache-airflow==3.0.2" "apache-airflow-providers-standard"
# 如需用到 Postgres、Redshift、Spark、AWS 等，补充安装相关 provider
pip install "apache-airflow-providers-postgres" "apache-airflow-providers-amazon" "apache-airflow-providers-apache-spark" "apache-airflow-providers-slack"
```

初始化数据库、创建用户、启动服务等步骤同原文档。

---

## 三、DAG 脚本与流程

### 1. poc_dag.py —— 复杂分支与 TaskGroup 演示

#### 代码

```python
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import pendulum
import re
from bs4 import BeautifulSoup

# 默认参数
default_args = {
    'owner': 'airflow',
    'retries': 2,
    'retry_delay': timedelta(minutes=1),
}

def extract_data(**context):
    try:
        with open('/tmp/page.html', 'r', encoding='utf-8') as f:
            html = f.read()
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string.strip() if soup.title else ''
        
        # 用正则表达式提取title，避免BeautifulSoup
        # title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        # title = title_match.group(1).strip() if title_match else ''
        
        result = {"title": title, "success": True}
    except Exception as e:
        result = {"title": str(e), "success": False}
    print(f"[Extract] result: {result}")
    return result

def is_weekend_func(**context):
    logical_date = context['dag_run'].logical_date
    weekday = pendulum.instance(logical_date).weekday()
    
    print(f"[is_weekend_func] ***** weekday: {weekday}")
    
    if weekday >= 5:
        return 'weekend_task'
    else:
        return 'etl_group.start_etl'

def transform(**context):
    # 返回一个默认json结构
    result = {"title": "transform", "success": True} # TODO
    print(f"[Transform] result: {result}")
    return result

def load(**context):
    result = {"title": "load", "success": True} ## TODO
    print(f"[Transform] result: {result}")
    return result

def validate(**context):
    # 获取extract_data和transform的结果
    load_result = context['ti'].xcom_pull(task_ids='etl_group.load')
    transform_result = context['ti'].xcom_pull(task_ids='etl_group.transform')
    # 只要有一个失败就失败
    if not load_result or not load_result.get('success'):
        print("[Validate] load_result failed")
        return 'end'
    if not transform_result or not transform_result.get('success'):
        print("[Validate] transform failed")
        return 'end'
    print("[Validate] 验证通过")
    print("extract_result: ", load_result)
    print("transform_result: ", transform_result)
    return 'etl_group.notify_workday'

with DAG(
    dag_id='poc_dag',
    default_args=default_args,
    start_date=datetime(2025, 6, 19),
    # schedule_interval='@daily',
    catchup=False,
    description='POC复杂DAG演示',
) as dag:
    start = EmptyOperator(task_id='start')

    branch = BranchPythonOperator(
        task_id='is_weekend',
        python_callable=is_weekend_func,
        # provide_context=True,
    )

    # 周末流程
    weekend = EmptyOperator(task_id='weekend_task')
    notify_weekend = BashOperator(
        task_id='notify_weekend',
        bash_command='echo "周末流程已完成，发送通知！"',
    )

    end = EmptyOperator(task_id='end', trigger_rule='none_failed_min_one_success')

    # 工作日流程（ETL TaskGroup）
    with TaskGroup('etl_group', tooltip='ETL流程') as etl_group:
        start_etl = EmptyOperator(task_id='start_etl')
        fetch_page = BashOperator(
            task_id='fetch_page',
            # bash_command='curl -s "https://segmentfault.com/a/1190000046041252" -o /tmp/page.html',
            bash_command='echo $PYTHONFAULTHANDLER',
        )
            
        extract_data_task = PythonOperator(
            task_id='extract_data',
            python_callable=extract_data,
        )
        transform_task = PythonOperator(
            task_id='transform',
            python_callable=transform,
        )
        load_task = PythonOperator(
            task_id='load',
            python_callable=load,
            # provide_context=True,
        )
        validate_task = BranchPythonOperator(
            task_id='validate',
            python_callable=validate,
            # provide_context=True,
        )
        notify_workday = BashOperator(
            task_id='notify_workday',
            bash_command='echo "工作日ETL流程已完成，发送通知！"',
        )
        # ETL流程依赖
        start_etl >> fetch_page >> extract_data_task >> [transform_task, load_task]
        [transform_task, load_task] >> validate_task
        validate_task >> [notify_workday, end]

    # DAG依赖关系（全部用对象，不用字符串）
    start >> branch
    branch >> weekend >> notify_weekend >> end
    branch >> etl_group >> end
    
'''
[2025-06-20, 11:52:56] WARNING - Could not import pandas. Holidays will not be considered.: source="airflow.example_dags.plugins.workday"
[2025-06-20, 11:52:56] INFO - DAG bundles loaded: dags-folder, example_dags: source="airflow.dag_processing.bundles.manager.DagBundlesManager"
[2025-06-20, 11:52:56] INFO - Filling up the DagBag from /Users/shhaofu/airflow/dags/poc_dag.py: source="airflow.models.dagbag.DagBag"
[2025-06-20, 11:52:56] INFO - Done. Returned value was: etl_group.start_etl: source="airflow.task.operators.airflow.providers.standard.operators.python.BranchPythonOperator"
[2025-06-20, 11:52:56] INFO - Branch into etl_group.start_etl: source="airflow.task.operators.airflow.providers.standard.operators.python.BranchPythonOperator"
[2025-06-20, 11:52:56] INFO - Following branch {'etl_group.start_etl'}: source="airflow.task.operators.airflow.providers.standard.operators.python.BranchPythonOperator"
[2025-06-20, 11:52:56] INFO - Skipping tasks [('weekend_task', -1)]: source="airflow.task.operators.airflow.providers.standard.operators.python.BranchPythonOperator"
[2025-06-20, 11:52:56] INFO - Skipping downstream tasks.: source="task"
[2025-06-20, 11:52:56] INFO - Task instance is in running state: chan="stdout": source="task"
[2025-06-20, 11:52:56] INFO -  Previous state of the Task instance: TaskInstanceState.QUEUED: chan="stdout": source="task"
[2025-06-20, 11:52:56] INFO - Current task name:is_weekend: chan="stdout": source="task"
[2025-06-20, 11:52:56] INFO - Dag name:poc_dag: chan="stdout": source="task"
[2025-06-20, 11:52:56] INFO - [is_weekend_func] ***** weekday: 4: chan="stdout": source="task"
[2025-06-20, 11:52:56] INFO - Task instance in success state: chan="stdout": source="task"
[2025-06-20, 11:52:56] INFO -  Previous state of the Task instance: TaskInstanceState.RUNNING: chan="stdout": source="task"
[2025-06-20, 11:52:56] INFO - Task operator:<Task(BranchPythonOperator): is_weekend>: chan="stdout": source="task"
'''
```

#### Mermaid 流程图

```mermaid
flowchart TD
    start([start])
    extract_data([extract_data])
    is_weekend([is_weekend])
    weekend_task([weekend_task])
    notify_weekend([notify_weekend])
    etl_group_start([etl_group.start_etl])
    fetch_page([etl_group.fetch_page])
    extract_data_task([etl_group.extract_data])
    transform([etl_group.transform])
    load([etl_group.load])
    validate([etl_group.validate])
    notify_workday([etl_group.notify_workday])
    finish([end])

    start --> is_weekend
    is_weekend -->|周末| weekend_task --> notify_weekend --> finish
    is_weekend -->|工作日| etl_group_start --> fetch_page --> extract_data_task --> transform
    extract_data_task --> load
    transform --> validate
    load --> validate
    validate -->|通过| notify_workday --> finish
    validate -->|失败| finish
```

#### 运行效果截图

![poc_dag-graph.png](images/poc_dag-graph.png)
![poc_dag-graph-2.png](images/poc_dag-graph-2.png)
![poc_dag-graph-3.png](images/poc_dag-graph-3.png)
![poc_dag-graph-4.png](images/poc_dag-graph-4.png)

---

### 2. fresh_grocery_etl.py —— 生鲜电商 ETL 流程

#### 代码

```python
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import PythonOperator

# from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.providers.amazon.aws.transfers.s3_to_redshift import S3ToRedshiftOperator
from datetime import datetime, timedelta
import pandas as pd

default_args = {
    'owner': 'data_team',
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

def extract_order_data(**context):
    orders = pd.DataFrame({
        'order_id': [101, 102],
        'user_id': [1, 2],
        'product_id': ['fresh_001', 'fresh_002'],
        'quantity': [3, 5],
        'timestamp': [datetime.now()] * 2
    })
    orders.to_parquet('/tmp/orders.parquet')
    context['ti'].xcom_push(key='order_path', value='/tmp/orders.parquet')

def analyze_inventory(**context):
    stock_path = context['ti'].xcom_pull(task_ids='extract_stock_data')
    order_path = context['ti'].xcom_pull(task_ids='extract_order_data', key='order_path')
    stock = pd.read_parquet(stock_path)
    orders = pd.read_parquet(order_path)
    merged = pd.merge(orders, stock, on='product_id')
    merged['consumption_rate'] = merged['quantity'] / merged['current_stock']
    low_stock = merged[merged['consumption_rate'] > 0.7]['product_id'].tolist()
    context['ti'].xcom_push(key='low_stock_items', value=low_stock)

def alert_branch(**context):
    low_stock = context['ti'].xcom_pull(task_ids='analyze_inventory', key='low_stock_items')
    return 'send_supplier_alert' if low_stock else 'no_alert_needed'

with DAG(
    'fresh_grocery_etl',
    start_date=datetime(2025, 6, 1),
    # schedule_interval='@daily',
    default_args=default_args
) as dag:
    start = EmptyOperator(task_id='start')
    extract_tasks = [
        PythonOperator(
            task_id='extract_order_data',
            python_callable=extract_order_data
        ),
        PythonOperator(
            task_id='extract_user_behavior',
            python_callable=lambda: print("Extracting user clickstream...")
        ),
        PythonOperator(
            task_id='extract_stock_data',
            python_callable=lambda **ctx: ctx['ti'].xcom_push(key='path', value='/tmp/stock.parquet')
        )
    ]
    transform_orders = SQLExecuteQueryOperator(
        task_id='transform_orders',
        conn_id='postgres_conn',
        database='postgres',
        sql="""
        CREATE TABLE IF NOT EXISTS cleaned_orders AS
        SELECT 
            order_id, 
            user_id,
            product_id,
            quantity,
            timestamp::DATE AS order_date
        FROM staging_orders
        WHERE quantity > 0
        """
    )
    inventory_analysis = PythonOperator(
        task_id='analyze_inventory',
        python_callable=analyze_inventory
    )
    branch = BranchPythonOperator(
        task_id='check_low_stock',
        python_callable=alert_branch
    )
    alert_task = PythonOperator(
        task_id='send_supplier_alert',
        python_callable=lambda: print("Alerting suppliers...")
    )
    no_alert = EmptyOperator(task_id='no_alert_needed')
    load_to_dw = S3ToRedshiftOperator(
        task_id='load_to_redshift',
        s3_bucket='data-lake',
        s3_key='{{ ds }}/processed_data.parquet',
        redshift_conn_id='redshift_conn',
        table='fresh_grocery_dm'
    )
    end = EmptyOperator(task_id='end', trigger_rule='none_failed')
    start >> extract_tasks 
    extract_tasks[0] >> transform_orders
    extract_tasks[2] >> inventory_analysis
    transform_orders >> load_to_dw
    inventory_analysis >> branch >> [alert_task, no_alert]
    [load_to_dw, alert_task, no_alert] >> end

```

#### Mermaid 流程图

```mermaid
flowchart TD
    start([start])
    extract_order_data([extract_order_data])
    extract_user_behavior([extract_user_behavior])
    extract_stock_data([extract_stock_data])
    transform_orders([transform_orders])
    inventory_analysis([analyze_inventory])
    branch([check_low_stock])
    send_supplier_alert([send_supplier_alert])
    no_alert_needed([no_alert_needed])
    load_to_dw([load_to_redshift])
    finish([end])

    start --> extract_order_data
    start --> extract_user_behavior
    start --> extract_stock_data
    extract_order_data --> transform_orders --> load_to_dw --> finish
    extract_stock_data --> inventory_analysis --> branch
    branch --> send_supplier_alert --> finish
    branch --> no_alert_needed --> finish
    extract_user_behavior --> finish
```

---

### 3. fresh_egg_dynamic_pricing.py —— 鲜蛋动态定价

#### 代码

```python
from airflow import DAG
from airflow.decorators import task, dag
from airflow.providers.amazon.aws.operators.sns import SnsPublishOperator

from airflow.providers.amazon.aws.operators.lambda_function import LambdaInvokeFunctionOperator
from airflow.providers.standard.operators.python import PythonOperator

import time
import pendulum

default_args = {
    "owner": "airflow",
    "retries": 3,
    "retry_delay": 30,
}

@dag(
    dag_id="fresh_egg_dynamic_pricing",
    default_args=default_args,
    start_date=pendulum.now().subtract(days=1),  # 或直接写 datetime(2024, 6, 20)
    # schedule_interval="0 */4 * * *",
    catchup=False,
    tags=["dynamic_pricing", "fresh_egg"],
)
def dynamic_pricing_dag():
    @task
    def monitor_inventory():
        # simulate slow task
        time.sleep(2)
        return {"low_stock_products": ["product_123", "product_456"], "threshold": 50}

    @task
    def analyze_demand_trends():
        time.sleep(2)
        return {"suggested_price": 19.99, "demand_score": 0.85}

    @task
    def check_stockout_alert(inventory_data: dict):
        if inventory_data["low_stock_products"]:
            return {"alert": True, "message": "低库存产品: " + ", ".join(inventory_data["low_stock_products"])}
        return {"alert": False}

    @task
    def check_price_fluctuations(demand_data: dict):
        if demand_data["suggested_price"] < 15:
            return {"alert": True, "message": "价格低于成本线"}
        return {"alert": False}

    @task
    def adjust_pricing(product_id: str, new_price: float):
        print(f"Adjusting price for {product_id} to {new_price}")
        # 实际应调用Lambda或API

    @task
    def trigger_restock(products: list):
        print(f"Triggering restock for: {products}")
        # 实际应触发DAG

    inventory_data = monitor_inventory()
    demand_data = analyze_demand_trends()

    stockout_alert = check_stockout_alert(inventory_data)
    price_alert = check_price_fluctuations(demand_data)

    adjust_pricing.expand(product_id=inventory_data["low_stock_products"], new_price=[demand_data["suggested_price"]]*2)
    trigger_restock(products=inventory_data["low_stock_products"])

    notify_stockout = SnsPublishOperator(
        task_id="notify_stockout",
        target_arn="arn:aws:sns:us-east-1:123456789012:stockout-alerts",
        message="{{ ti.xcom_pull(task_ids='check_stockout_alert')['message'] }}",
        subject="库存告警",
    )

    notify_price_anomaly = SnsPublishOperator(
        task_id="notify_price_anomaly",
        target_arn="arn:aws:sns:us-east-1:123456789012:price-alerts",
        message="{{ ti.xcom_pull(task_ids='check_price_fluctuations')['message'] }}",
        subject="价格异常",
    )

    stockout_alert >> notify_stockout
    price_alert >> notify_price_anomaly

# dynamic_pricing_dag()

```

#### Mermaid 流程图

```mermaid
flowchart TD
    monitor_inventory([monitor_inventory])
    analyze_demand_trends([analyze_demand_trends])
    check_stockout_alert([check_stockout_alert])
    check_price_fluctuations([check_price_fluctuations])
    adjust_pricing([adjust_pricing])
    trigger_restock([trigger_restock])
    notify_stockout([notify_stockout])
    notify_price_anomaly([notify_price_anomaly])

    monitor_inventory --> check_stockout_alert --> notify_stockout
    analyze_demand_trends --> check_price_fluctuations --> notify_price_anomaly
    monitor_inventory --> adjust_pricing
    analyze_demand_trends --> adjust_pricing
    monitor_inventory --> trigger_restock
```

---

### 4. fresh_egg_etl_pipeline.py —— 鲜蛋多源数据 ETL

#### 代码

```python
from airflow import DAG
from airflow.decorators import task
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

import pandas as pd
import sqlalchemy
import pendulum

# 默认参数
default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": 30,
}

with DAG(
    dag_id="fresh_egg_etl_pipeline",
    default_args=default_args,
    start_date=pendulum.now().subtract(days=1),  # 或直接写 datetime(2024, 6, 20)
    # schedule_interval="0 0 * * *",
    catchup=False,
    tags=["etl", "fresh_egg"],
) as dag:
    @task
    def extract_from_mysql(table_name: str):
        engine = sqlalchemy.create_engine("mysql+pymysql://user:password@localhost/dbname")
        query = f"SELECT * FROM {table_name} WHERE date = CURRENT_DATE - INTERVAL 1 DAY"
        df = pd.read_sql(query, engine)
        return df.to_json()

    @task
    def extract_from_s3(bucket: str, prefix: str):
        s3_hook = S3Hook(aws_conn_id="aws_default")
        key = s3_hook.get_key(bucket_name=bucket, key=prefix + "latest.json")
        df = pd.read_json(key.get()["Body"].read().decode("utf-8"))
        return df.to_json()

    @task
    def clean_data(source_data: str, source_type: str):
        df = pd.read_json(source_data)
        if source_type == "orders":
            df["order_time"] = pd.to_datetime(df["order_time"])
        elif source_type == "behavior":
            df["event_time"] = pd.to_datetime(df["event_time"])
        return df.drop_duplicates().to_json()

    @task
    def calculate_gmv(cleaned_orders: str):
        df = pd.read_json(cleaned_orders)
        gmv_df = df.groupby("date")["total_amount"].sum().reset_index()
        return gmv_df.to_json()

    @task
    def join_data(gmv_data: str, inventory_data: str, behavior_data: str):
        gmv_df = pd.read_json(gmv_data)
        inventory_df = pd.read_json(inventory_data)
        behavior_df = pd.read_json(behavior_data)
        final_df = gmv_df.merge(inventory_df, on="date").merge(behavior_df, on="date")
        return final_df.to_json()

    @task
    def load_to_redshift(final_data: str):
        df = pd.read_json(final_data)
        engine = sqlalchemy.create_engine("postgresql+psycopg2://user:password@redshift-host/dbname")
        df.to_sql("daily_sales_report", engine, if_exists="replace", index=False)

    orders_data = extract_from_mysql("orders")
    inventory_data = extract_from_mysql("inventory")
    behavior_data = extract_from_s3("user-logs", "behavior/")

    cleaned_orders = clean_data(orders_data, "orders")
    cleaned_inventory = clean_data(inventory_data, "inventory")
    cleaned_behavior = clean_data(behavior_data, "behavior")

    gmv_result = calculate_gmv(cleaned_orders)
    final_report = join_data(gmv_result, cleaned_inventory, cleaned_behavior)

    load_to_redshift(final_report)

```

#### Mermaid 流程图

```mermaid
flowchart TD
    extract_mysql_orders["extract_from_mysql(orders)"]
    extract_mysql_inventory["extract_from_mysql(inventory)"]
    extract_s3["extract_from_s3"]
    clean_orders["clean_data(orders)"]
    clean_inventory["clean_data(inventory)"]
    clean_behavior["clean_data(behavior)"]
    calculate_gmv["calculate_gmv"]
    join_data["join_data"]
    load_redshift["load_to_redshift"]

    extract_mysql_orders --> clean_orders --> calculate_gmv
    extract_mysql_inventory --> clean_inventory
    extract_s3 --> clean_behavior
    calculate_gmv --> join_data
    clean_inventory --> join_data
    clean_behavior --> join_data
    join_data --> load_redshift
```

---

### 5. financial_risk_management.py —— 金融风控全流程

#### 代码

```python
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
from airflow.decorators import task

def slack_fail_callback(context):
    SlackWebhookOperator(
        task_id='slack_alert',
        message=f"DAG Failed: {context['dag']}",
        slack_webhook_conn_id='slack_conn'
    ).execute(context)

default_args = {
    'owner': 'risk_team',
    'retries': 3,
    'retry_delay': timedelta(minutes=10),
    'on_failure_callback': slack_fail_callback
}

with DAG(
    'financial_risk_management',
    start_date=datetime(2025, 1, 1),
    # schedule_interval='@hourly',
    max_active_runs=1,
    default_args=default_args
) as dag:
    
    # 1. 多源数据并行摄取
    ingest_transactions = SparkSubmitOperator(
        task_id='ingest_transactions',
        application='/jobs/ingest_transactions.py',
        conn_id='spark_cluster'
    )
    
    ingest_user_profiles = PythonOperator(
        task_id='ingest_user_profiles',
        python_callable=lambda: print("Syncing user profiles from CRM...")
    )
    
    # 2. 外部征信检查（API调用）
    @task(task_id='check_credit_bureau')
    def check_credit_bureau():
        # 模拟调用外部征信系统
        return {"user_123": 750, "user_456": 680}
    
    # 3. 特征工程
    feature_engineering = SparkSubmitOperator(
        task_id='feature_engineering',
        application='/jobs/build_features.py',
        conn_id='spark_cluster'
    )
    
    # 4. 并行模型评分
    fraud_model = SparkSubmitOperator(
        task_id='run_fraud_model',
        application='/models/fraud_detection.py',
        conn_id='spark_cluster'
    )
    
    credit_risk_model = PythonOperator(
        task_id='run_credit_risk_model',
        python_callable=lambda: print("Executing credit risk model...")
    )
    
    # 5. 决策引擎
    @task.branch(task_id='risk_decision_engine')
    def decision_engine(**context):
        fraud_score = context['ti'].xcom_pull(task_ids='run_fraud_model', key='fraud_score')
        credit_score = context['ti'].xcom_pull(task_ids='run_credit_risk_model', key='credit_score')
        
        if fraud_score and fraud_score > 0.9:
            return 'block_transaction'
        elif credit_score and credit_score < 600:
            return 'manual_review'
        else:
            return 'approve_transaction'
    
    # 6. 处置动作
    block_transaction = PythonOperator(
        task_id='block_transaction',
        python_callable=lambda: print("Blocking high-risk transaction!")
    )
    
    manual_review = TriggerDagRunOperator(
        task_id='manual_review',
        trigger_dag_id='fraud_investigation_workflow'
    )
    
    approve_transaction = EmptyOperator(task_id='approve_transaction')
    
    # 7. 监管报告
    generate_reports = SparkSubmitOperator(
        task_id='generate_regulatory_reports',
        application='/compliance/generate_fincen_report.py',
        conn_id='spark_cluster'
    )
    
    # 8. 数据归档
    archive_data = PythonOperator(
        task_id='archive_processed_data',
        python_callable=lambda: print("Moving to cold storage...")
    )
    
    # 任务依赖
    [ingest_transactions, ingest_user_profiles] >> check_credit_bureau() >> feature_engineering
    feature_engineering >> [fraud_model, credit_risk_model] >> decision_engine()
    decision_engine() >> [block_transaction, manual_review, approve_transaction]
    [block_transaction, manual_review, approve_transaction] >> generate_reports >> archive_data

```

#### Mermaid 流程图

```mermaid
flowchart TD
    ingest_transactions([ingest_transactions])
    ingest_user_profiles([ingest_user_profiles])
    check_credit_bureau([check_credit_bureau])
    feature_engineering([feature_engineering])
    fraud_model([run_fraud_model])
    credit_risk_model([run_credit_risk_model])
    decision_engine([risk_decision_engine])
    block_transaction([block_transaction])
    manual_review([manual_review])
    approve_transaction([approve_transaction])
    generate_reports([generate_regulatory_reports])
    archive_data([archive_processed_data])

    ingest_transactions --> check_credit_bureau
    ingest_user_profiles --> check_credit_bureau
    check_credit_bureau --> feature_engineering
    feature_engineering --> fraud_model
    feature_engineering --> credit_risk_model
    fraud_model --> decision_engine
    credit_risk_model --> decision_engine
    decision_engine --> block_transaction
    decision_engine --> manual_review
    decision_engine --> approve_transaction
    block_transaction --> generate_reports
    manual_review --> generate_reports
    approve_transaction --> generate_reports
    generate_reports --> archive_data
```

---

## 四、DAG 运行效果图

![simple_demo_dag-graph.png](images/simple_demo_dag-graph.png)

---



