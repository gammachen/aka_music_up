在 Python 中直接使用 **纯 SQLAlchemy**（不依赖 Flask 等框架）时，数据库路径 `sqlite:///app_b.db` 的解析规则如下：

---

### 1. **默认行为（无框架场景）**
- **相对路径**：若直接使用 `sqlite:///app_b.db`，SQLAlchemy 默认会在 **当前工作目录**（即运行 Python 脚本的目录）创建数据库文件，**不会自动添加 `instance` 目录**。
  ```python
  from sqlalchemy import create_engine
  engine = create_engine("sqlite:///app_b.db")  # 文件生成在运行脚本的目录下
  ```

---

### 2. **Flask 框架的特殊行为**
如果通过 **Flask-SQLAlchemy 扩展** 连接数据库，则路径解析会受 Flask 的 `instance_path` 配置影响：
- **默认 `instance` 目录**：Flask 默认将 `instance` 目录（位于项目根目录下）用于存储应用实例相关的文件（如配置、本地数据库）。若未显式配置 `SQLALCHEMY_DATABASE_URI`，Flask-SQLAlchemy 会默认将数据库文件放在 `instance` 目录中。
- **显式设置路径**：即使手动设置 `SQLALCHEMY_DATABASE_URI="sqlite:///app_b.db"`，Flask-SQLAlchemy 仍可能将路径解析为相对于 `instance` 目录。例如：
  ```python
  from flask import Flask
  from flask_sqlalchemy import SQLAlchemy

  app = Flask(__name__)
  app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app_b.db"
  db = SQLAlchemy(app)  # 文件生成在 instance/app_b.db
  ```

---

### 3. **如何控制路径？**
#### **场景 1：强制数据库在项目根目录（非 `instance` 目录）**
- **Flask 配置**：通过设置 `instance_path` 或直接使用绝对路径：
  ```python
  app = Flask(__name__, instance_path=os.path.abspath("."))  # 修改 instance_path 为当前目录
  app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app_b.db"
  ```
  或：
  ```python
  import os
  app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.abspath('app_b.db')}"  # 绝对路径
  ```

#### **场景 2：纯 SQLAlchemy 指定路径**
- **明确路径前缀**：使用 `sqlite:////` 表示绝对路径（注意四个斜杠）：
  ```python
  engine = create_engine("sqlite:////tmp/app_b.db")  # Linux/macOS 绝对路径
  engine = create_engine("sqlite:///C:\\path\\app_b.db")  # Windows 绝对路径
  ```

---

### 4. **总结**
| 场景                | 行为                                                                 |
|---------------------|--------------------------------------------------------------------|
| **纯 SQLAlchemy**   | 路径相对于 **当前工作目录**，无 `instance` 目录参与。                 |
| **Flask-SQLAlchemy** | 默认路径相对于 `instance` 目录，需通过配置调整路径。                   |

建议通过打印数据库路径验证实际位置：
```python
print(engine.url.database)  # 查看 SQLAlchemy 解析后的绝对路径
```