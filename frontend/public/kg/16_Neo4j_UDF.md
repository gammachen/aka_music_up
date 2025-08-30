在 Neo4j 中，**自定义插件（Plugin）和用户自定义函数（UDF, User Defined Functions）** 是扩展数据库功能的两种主要方式。它们允许你通过 Java 编写自己的逻辑，并将其集成到 Cypher 查询中。

---

## ✅ 一、自定义 **User Defined Function (UDF)**

### 📌 UDF 的作用：
编写一个 Java 方法，让它像 `split()`, `size()` 这样的函数一样，在 Cypher 查询中使用，比如：

```cypher
RETURN example.myFunction("hello") AS result
```

---

### 🔧 步骤：创建 UDF 插件

#### 1. 创建 Maven 项目（pom.xml）

```xml
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>neo4j-udf-example</artifactId>
    <version>1.0-SNAPSHOT</version>

    <dependencies>
        <dependency>
            <groupId>org.neo4j</groupId>
            <artifactId>neo4j</artifactId>
            <version>5.21.0</version> <!-- 根据你的 Neo4j 版本修改 -->
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-jar-plugin</artifactId>
                <version>3.2.0</version>
                <configuration>
                    <archive>
                        <manifest>
                            <addClasspath>true</addClasspath>
                        </manifest>
                    </archive>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```

---

#### 2. 编写 Java 类实现 UDF

```java
package com.example;

import org.neo4j.procedure.*;

public class MyFunctions {

    @UserFunction
    @Description("Returns the length of a string")
    public long myStringLength(@Name("text") String text) {
        return text != null ? text.length() : 0;
    }

    @UserFunction
    @Description("Returns uppercase version of a string")
    public String toUpperCase(@Name("text") String text) {
        return text != null ? text.toUpperCase() : null;
    }
}
```

---

#### 3. 构建 JAR 文件

```bash
mvn clean package
```

生成的 JAR 文件位于：`target/neo4j-udf-example-1.0-SNAPSHOT.jar`

---

#### 4. 安装插件

将 `.jar` 文件复制到 Neo4j 的 `plugins/` 目录下，例如：

```
$NEO4J_HOME/plugins/
```

---

#### 5. 配置 Neo4j 允许调用该函数

编辑 `neo4j.conf`：

```properties
dbms.security.procedures.unrestricted=com.example.*
```

---

#### 6. 使用自定义函数

重启 Neo4j 后即可使用：

```cypher
RETURN com.example.myStringLength("Neo4j") AS len;
// 返回: 5

RETURN com.example.toUpperCase("hello") AS upper;
// 返回: HELLO
```

---

## ✅ 二、自定义 **Stored Procedures（存储过程）**

与 UDF 不同，存储过程可以返回多个值、节点、关系等，甚至执行图操作。

### 示例：创建一个返回所有用户的存储过程

```java
package com.example;

import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Result;
import org.neo4j.graphdb.Transaction;
import org.neo4j.procedure.*;

import java.util.stream.Stream;

public class MyProcedures {

    // 注入系统提供的 GraphDatabaseService
    @Context
    public GraphDatabaseService db;

    @Procedure(name = "com.example.getAllUsers", mode = Mode.READ)
    @Description("Returns all users in the database")
    public Stream<UserResult> getAllUsers() {
        try (Transaction tx = db.beginTx()) {
            Result result = tx.execute("MATCH (u:SN_User) RETURN u.name AS name, u.uid AS uid");
            return result.stream().map(record -> new UserResult(
                record.get("uid").asString(),
                record.get("name").asString()
            ));
        }
    }

    // 定义返回类型
    public static class UserResult {
        public final String uid;
        public final String name;

        public UserResult(String uid, String name) {
            this.uid = uid;
            this.name = name;
        }
    }
}
```

---

#### 使用方式：

```cypher
CALL com.example.getAllUsers() YIELD uid, name
RETURN *;
```

---

## ✅ 三、注意事项

| 内容 | 说明 |
|------|------|
| Neo4j 版本兼容性 | 插件必须与 Neo4j 的版本兼容（建议开发时使用相同版本） |
| 安全权限 | 默认情况下，新函数/过程是受限的，需要配置 `neo4j.conf` |
| 日志调试 | 可以使用 `@Context public Log log;` 记录日志信息 |
| 异常处理 | 在方法中抛出异常会传递给 Cypher，记得 try-catch 处理 |

---

## ✅ 四、总结对比

| 类型 | 是否支持返回表数据 | 是否能修改图数据 | 能否作为表达式使用 |
|------|------------------|------------------|-------------------|
| UDF（用户自定义函数） | ❌ 否 | ❌ 否 | ✅ 是（如 `example.func(x)`） |
| 存储过程（Procedure） | ✅ 是 | ✅ 是 | ❌ 否（需 `CALL` 调用） |

---

## 📦 五、推荐工具链

- IDE：IntelliJ IDEA / Eclipse
- 构建工具：Maven 或 Gradle
- 测试方式：
  - 使用 `@RunWith(ExternalResourceTestRunner.class)` + `TestGraphDatabaseFactory`
  - 或者直接部署后在 Neo4j Browser 中测试

