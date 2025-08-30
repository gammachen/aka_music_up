下面是一个使用 Apache Airflow 做有趣自动化任务的完整 Demo，我们将创建一个"每日趣味邮件"工作流，自动收集有趣内容并发送给指定邮箱：

### 有趣场景：每日自动发送趣味邮件
**功能包含**：
1. 获取每日趣味编程笑话
2. 抓取 NASA 每日天文图片
3. 生成随机趣味事实
4. 组合内容发送精美邮件

---

### 环境准备 (使用 Docker)
```bash
# 创建项目目录
mkdir airflow-fun-demo && cd airflow-fun-demo

# 创建 docker-compose.yml
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/stable/docker-compose.yaml'

# 创建环境文件
echo -e "AIRFLOW_UID=$(id -u)\nAIRFLOW_GID=0" > .env

# 创建目录
mkdir -p ./dags ./logs ./plugins
```

---

### DAG 实现 (`dags/fun_daily_email.py`)
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.dates import days_ago
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import random
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'retries': 1
}

def get_programming_joke():
    """获取编程笑话"""
    url = "https://v2.jokeapi.dev/joke/Programming?type=single"
    response = requests.get(url)
    return response.json().get('joke', "Why do programmers prefer dark mode? Because light attracts bugs!")

def get_nasa_image():
    """获取NASA每日图片"""
    url = "https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY"
    response = requests.get(url)
    data = response.json()
    return {
        'title': data.get('title', "Space Image"),
        'url': data.get('url', "https://apod.nasa.gov/apod/image/2208/M13_final2_sinfirma.jpg"),
        'explanation': data.get('explanation', "Beautiful space image")
    }

def get_random_fact():
    """获取随机趣味事实"""
    facts = [
        "蜂鸟是唯一可以倒飞的鸟",
        "章鱼有三颗心脏",
        "蜂蜜永远不会变质，考古学家曾发现过3000年前的蜂蜜",
        "你的胃酸可以溶解刀片",
        "云看起来轻飘飘的，但一片典型的积云重量约等于100头大象"
    ]
    return random.choice(facts)

def send_fun_email(**context):
    """发送组合邮件"""
    # 获取任务结果
    ti = context['ti']
    joke = ti.xcom_pull(task_ids='get_joke')
    nasa_data = ti.xcom_pull(task_ids='get_nasa_image')
    fact = ti.xcom_pull(task_ids='get_fact')
    
    # 邮件内容
    date_str = datetime.now().strftime("%Y年%m月%d日")
    subject = f"每日趣味包 - {date_str}"
    
    html_content = f"""
    <html>
      <body>
        <h2>🌞 早上好！这是今天的趣味包：</h2>
        
        <h3>😂 编程笑话</h3>
        <p>{joke}</p>
        
        <h3>🪐 NASA 天文每日一图</h3>
        <h4>{nasa_data['title']}</h4>
        <img src="{nasa_data['url']}" alt="NASA Image" width="500">
        <p><i>{nasa_data['explanation'][:200]}...</i></p>
        <p><a href="{nasa_data['url']}">查看大图</a></p>
        
        <h3>🔍 趣味冷知识</h3>
        <p>{fact}</p>
        
        <hr>
        <p>由 Apache Airflow 自动生成 | 每日快乐 😊</p>
      </body>
    </html>
    """
    
    # 发送邮件
    from_email = "your_email@gmail.com"
    to_email = "recipient@example.com"
    password = "your_app_password"  # 使用应用专用密码
    
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email
    
    msg.attach(MIMEText(html_content, 'html'))
    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
        print("邮件发送成功！")
    except Exception as e:
        print(f"邮件发送失败: {str(e)}")

# 定义DAG
with DAG(
    'daily_fun_email',
    default_args=default_args,
    description='每日发送趣味邮件',
    schedule_interval='0 8 * * *',  # 每天上午8点运行
    catchup=False,
    tags=['fun'],
) as dag:

    create_table = PostgresOperator(
        task_id='create_log_table',
        postgres_conn_id='airflow_db',
        sql="""
        CREATE TABLE IF NOT EXISTS fun_email_log (
            date DATE PRIMARY KEY,
            joke TEXT,
            fact TEXT,
            nasa_title TEXT
        );
        """
    )

    get_joke = PythonOperator(
        task_id='get_joke',
        python_callable=get_programming_joke
    )

    get_nasa_image = PythonOperator(
        task_id='get_nasa_image',
        python_callable=get_nasa_image
    )

    get_fact = PythonOperator(
        task_id='get_fact',
        python_callable=get_random_fact
    )

    send_email = PythonOperator(
        task_id='send_email',
        python_callable=send_fun_email,
        provide_context=True
    )

    log_result = PostgresOperator(
        task_id='log_result',
        postgres_conn_id='airflow_db',
        sql="""
        INSERT INTO fun_email_log (date, joke, fact, nasa_title)
        VALUES ('{{ ds }}', 
                '{{ ti.xcom_pull(task_ids="get_joke") }}',
                '{{ ti.xcom_pull(task_ids="get_fact") }}',
                '{{ ti.xcom_pull(task_ids="get_nasa_image")["title"] }}')
        ON CONFLICT (date) DO NOTHING;
        """
    )

    # 任务依赖关系
    create_table >> [get_joke, get_nasa_image, get_fact] >> send_email >> log_result
```

---

### 启动 Airflow
```bash
# 初始化数据库
docker-compose up airflow-init

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

访问: http://localhost:8080 (默认账号: airflow, 密码: airflow)

---

### 配置步骤
1. **设置邮件参数**：
   - 修改代码中的邮箱配置：
     ```python
     from_email = "your_email@gmail.com"
     to_email = "recipient@example.com"
     password = "your_app_password"  # 对于Gmail需创建应用专用密码
     ```

2. **启用Email连接**：
   - 在Airflow UI: Admin -> Connections
   - 添加新连接：
     - Conn Id: `smtp_default`
     - Conn Type: SMTP
     - Host: `smtp.gmail.com`
     - Port: `587`
     - Login: 你的邮箱
     - Password: 邮箱密码/应用密码

3. **设置数据库连接**：
   - 添加Postgres连接：
     - Conn Id: `airflow_db`
     - Conn Type: Postgres
     - Host: `postgres`
     - Schema: `airflow`
     - Login: `airflow`
     - Password: `airflow`
     - Port: `5432`

---

### 效果展示
1. **邮件内容**：
   ![趣味邮件示例](https://via.placeholder.com/600x400?text=趣味邮件截图)

2. **Airflow DAG视图**：
   ```mermaid
   graph TD
     A[创建日志表] --> B[获取笑话]
     A --> C[获取NASA图片]
     A --> D[获取趣味事实]
     B --> E[发送邮件]
     C --> E
     D --> E
     E --> F[记录日志]
   ```

3. **日志记录**：
   ```sql
   SELECT * FROM fun_email_log;
   ```
   结果：
   | date       | joke                                | fact                        | nasa_title           |
   |------------|-------------------------------------|-----------------------------|---------------------|
   | 2023-08-01 | Why do Java developers...           | 章鱼有三颗心脏              | Jupiter's Swirling |
   | 2023-08-02 | There are 10 types of people...     | 蜂蜜永远不会变质            | Earth's Moon       |

---

### 扩展更多有趣功能
1. **添加天气预报**：
   ```python
   def get_weather(city="Beijing"):
       url = f"http://wttr.in/{city}?format=%C+%t"
       response = requests.get(url)
       return response.text
   ```

2. **添加每日名言**：
   ```python
   def get_quote():
       url = "https://api.quotable.io/random"
       response = requests.get(url)
       return f"{response.json()['content']} - {response.json()['author']}"
   ```

3. **添加生日提醒**：
   ```python
   def check_birthdays():
       today = datetime.now().strftime("%m-%d")
       birthdays = {
           "01-15": "张三",
           "08-20": "李四"
       }
       return birthdays.get(today, "今天没有生日哦~")
   ```

4. **添加随机猫图**：
   ```python
   def get_cat_image():
       return "https://cataas.com/cat?type=" + random.choice(["","gif","funny"])
   ```

---

### 生产环境建议
1. **安全增强**：
   - 使用Airflow的`Variables`存储敏感信息
   - 启用SSL加密
   - 设置访问控制

2. **错误处理**：
   ```python
   try:
       # API调用
   except requests.exceptions.RequestException as e:
       logging.error(f"API请求失败: {str(e)}")
       return "内容获取失败，请稍后再试"
   ```

3. **性能优化**：
   - 设置任务超时时间
   - 使用`retries`和`retry_delay`
   - 添加监控告警

这个Demo展示了如何使用Airflow创建有趣实用的自动化工作流。通过扩展API和增加更多任务，你可以打造属于自己的个性化每日简报系统！