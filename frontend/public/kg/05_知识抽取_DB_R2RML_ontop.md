## 1. 生成用于初始化 SQLite 数据库的 Python 脚本

保存为 `init_db.py`：

```python
import sqlite3

conn = sqlite3.connect('demo.db')
cur = conn.cursor()

cur.executescript("""
DROP TABLE IF EXISTS works_in;
DROP TABLE IF EXISTS person;
DROP TABLE IF EXISTS department;

CREATE TABLE person (
  id INTEGER PRIMARY KEY,
  name TEXT,
  birth_date TEXT
);

CREATE TABLE department (
  dept_id INTEGER PRIMARY KEY,
  dept_name TEXT
);

CREATE TABLE works_in (
  person_id INTEGER,
  dept_id INTEGER,
  FOREIGN KEY(person_id) REFERENCES person(id),
  FOREIGN KEY(dept_id) REFERENCES department(dept_id)
);

INSERT INTO person VALUES (1, 'Alice', '1990-01-01');
INSERT INTO person VALUES (2, 'Bob', '1985-05-12');
INSERT INTO department VALUES (10, 'HR');
INSERT INTO department VALUES (20, 'Engineering');
INSERT INTO works_in VALUES (1, 10);
INSERT INTO works_in VALUES (2, 20);
""")

conn.commit()
conn.close()
print("Database demo.db initialized.")
```

运行方法（在终端）：

```sh
python3 init_db.py
```

---

## 2. 使用 Ontop CLI 工具进行 RDF 生成

### 2.1. 前提

- 已安装 [Ontop CLI](https://ontop-vkg.org/guide/cli/).
- 已有 `demo.db`（上一步生成）、`mapping.ttl`（R2RML 映射文件，见上条回答）、`ontology.owl`（可选，若有本体）。

### 2.2. 创建 Ontop 连接配置文件 `demo.properties`：

```properties
# demo.properties
jdbc.url=jdbc:sqlite:demo.db
jdbc.driver=org.sqlite.JDBC
```

### 2.3. 运行 Ontop 生成 RDF（N-Triples 格式为例）

```sh
ontop materialize \
  --properties demo.properties \
  --mapping mapping.ttl \
  --output demo_output.nt
```

- `--properties`：数据库连接配置
- `--mapping`：R2RML 映射文件
- `--output`：输出的RDF文件

### 2.4. 查看结果

`demo_output.nt` 即为生成的RDF三元组文件。

---

### 总结

1. 用 `init_db.py` 初始化数据库。
2. 用 Ontop CLI + R2RML 映射文件 + 配置文件，生成 RDF。

如需 `mapping.ttl` 文件内容或 Ontop CLI 安装方法等细节，请随时告知！


### 执行指令

```bash
~/ontop-cli-5.3.0/ontop materialize --properties 05_graph_sqlite3.properties --mapping 05_graph_sqlite3_mapping.ttl --output 05_graph_sqlite_graph_output.nt
```

异常情况：
```bash
(base) shhaofu@shhaofudeMacBook-Pro kg % /ontop-cli-5.3.0/ontop materialize --properties 05_graph_sqlite3.properties --mapping 05_graph_sqlite3_mapping.ttl --output sqlite_graph_output.nt
java.lang.IllegalArgumentException: Nullable attribute "id" INTEGER cannot be in a PK
        at it.unibz.inf.ontop.dbschema.impl.UniqueConstraintImpl$PrimaryKeyBuilder.addDeterminant(UniqueConstraintImpl.java:79)
        at it.unibz.inf.ontop.dbschema.impl.AbstractDBMetadataProvider.insertPrimaryKey(AbstractDBMetadataProvider.java:286)
        at it.unibz.inf.ontop.dbschema.impl.AbstractDBMetadataProvider.insertIntegrityConstraints(AbstractDBMetadataProvider.java:224)
        at it.unibz.inf.ontop.dbschema.impl.CachingMetadataLookup.extractImmutableMetadata(CachingMetadataLookup.java:57)
        at it.unibz.inf.ontop.spec.mapping.impl.SQLMappingExtractor.convert(SQLMappingExtractor.java:250)
        at it.unibz.inf.ontop.spec.mapping.impl.SQLMappingExtractor.convert(SQLMappingExtractor.java:219)
        at it.unibz.inf.ontop.spec.mapping.impl.SQLMappingExtractor.convert(SQLMappingExtractor.java:188)
        at it.unibz.inf.ontop.spec.mapping.impl.SQLMappingExtractor.convertPPMapping(SQLMappingExtractor.java:153)
        at it.unibz.inf.ontop.spec.mapping.impl.SQLMappingExtractor.extract(SQLMappingExtractor.java:108)
        at it.unibz.inf.ontop.spec.impl.DefaultOBDASpecificationExtractor.extract(DefaultOBDASpecificationExtractor.java:51)
        at it.unibz.inf.ontop.injection.impl.OntopMappingConfigurationImpl.loadSpecification(OntopMappingConfigurationImpl.java:130)
        at it.unibz.inf.ontop.injection.impl.OntopMappingSQLConfigurationImpl.loadSpecification(OntopMappingSQLConfigurationImpl.java:93)
        at it.unibz.inf.ontop.injection.impl.OntopMappingSQLAllConfigurationImpl.loadSpecification(OntopMappingSQLAllConfigurationImpl.java:55)
        at it.unibz.inf.ontop.injection.impl.OntopSQLOWLAPIConfigurationImpl.loadOBDASpecification(OntopSQLOWLAPIConfigurationImpl.java:42)
        at it.unibz.inf.ontop.injection.impl.OntopOBDAConfigurationImpl.loadSpecification(OntopOBDAConfigurationImpl.java:44)
        at it.unibz.inf.ontop.materialization.impl.DefaultOntopRDFMaterializer.<init>(DefaultOntopRDFMaterializer.java:48)
        at it.unibz.inf.ontop.rdf4j.materialization.impl.DefaultRDF4JMaterializer.<init>(DefaultRDF4JMaterializer.java:39)
        at it.unibz.inf.ontop.rdf4j.materialization.RDF4JMaterializer.defaultMaterializer(RDF4JMaterializer.java:33)
        at it.unibz.inf.ontop.cli.OntopMaterialize.createMaterializer(OntopMaterialize.java:110)
        at it.unibz.inf.ontop.cli.OntopMaterialize.run(OntopMaterialize.java:97)
        at it.unibz.inf.ontop.cli.Ontop.main(Ontop.java:19)
Exception in thread "main" java.lang.AssertionError
        at it.unibz.inf.ontop.cli.Ontop.main(Ontop.java:31)

必须将id等pk这个主键显示设置为not null，比如："id" interger not null,primary key("id")，不能够简单的只是设置为：primary key("id")
```

## ontop的使用

Ontop CLI 的安装方式有多种，最常用的是通过下载官方发布的 zip 包。下面是标准的安装步骤（以 macOS/Linux 为例）：

## jdbc driver

```bash
(base) shhaofu@shhaofudeMacBook-Pro ontop-cli-5.3.0 % tree
.
├── README.md
├── copyright
│   ├── AIRLINE-LICENSE.txt
│   ├── ANTLR-LICENSE.txt
│   ├── COMMONS-LICENSE.txt
│   ├── FINDBUGSJSR305-LICENSE.txt
│   ├── FUNCTIONALJAVA-LICENSE.txt
│   ├── GSON-LICENSE.txt
│   ├── GUAVA-LICENSE.txt
│   ├── GUICE-LICENSE.txt
│   ├── H2-LICENSE.txt
│   ├── HIKARICP-LICENSE.txt
│   ├── JACKSON-LICENSE.txt
│   ├── JETTY-LICENSE.txt
│   ├── JGRAPHT-LICENSE.txt
│   ├── JSONLDJAVA-LICENSE.txt
│   ├── JSQLPARSER-LICENSE.txt
│   ├── JUNIT-LICENSE.txt
│   ├── LOGBACK-LICENSE.txt
│   ├── OSGI-LICENSE.txt
│   ├── OWLAPI-LICENSE.txt
│   ├── PROTEGE-LICENSE.txt
│   ├── R2RMLAPI-LICENSE.txt
│   ├── RDF4J-LICENSE.txt
│   ├── SLF4J-LICENSE.txt
│   ├── SPRING-LICENSE.txt
│   ├── TOM4J-LICENSE.txt
│   ├── TOMCAT-LICENSE.txt
│   └── URLBUILDER-LICENSE.txt
├── jdbc
│   ├── DROP_YOUR_JDBC_DRIVERS_IN_THIS_DIR.txt
│   └── sqlite-jdbc-3.48.0.0.jar
├── lib
│   ├── HdrHistogram-2.1.12.jar
│   ├── HikariCP-3.4.5.jar
│   ├── LatencyUtils-2.0.3.jar
│   ├── airline-3.0.0.jar
│   ├── airline-help-bash-3.0.0.jar
│   ├── airline-io-3.0.0.jar
│   ├── antlr4-runtime-4.13.1.jar
│   ├── aopalliance-1.0.jar
│   ├── attoparser-2.0.5.RELEASE.jar
│   ├── caffeine-3.1.8.jar
│   ├── checker-qual-3.8.0.jar
│   ├── commons-beanutils-1.9.4.jar
│   ├── commons-codec-1.14.jar
│   ├── commons-collections-3.2.2.jar
│   ├── commons-collections4-4.3.jar
│   ├── commons-io-2.14.0.jar
│   ├── commons-lang3-3.11.jar
│   ├── commons-math3-3.6.1.jar
│   ├── commons-rdf-api-0.5.0.jar
│   ├── commons-rdf-rdf4j-0.5.0.jar
│   ├── commons-rdf-simple-0.5.0.jar
│   ├── commons-text-1.10.0.jar
│   ├── error_prone_annotations-2.5.1.jar
│   ├── failureaccess-1.0.1.jar
│   ├── gson-2.10.1.jar
│   ├── guava-32.0.1-jre.jar
│   ├── guice-5.0.1.jar
│   ├── guice-assistedinject-5.0.1.jar
│   ├── hasmac-json-ld-0.9.0.jar
│   ├── hppcrt-0.7.5.jar
│   ├── httpclient-4.5.13.jar
│   ├── httpclient-cache-4.5.13.jar
│   ├── httpcore-4.4.14.jar
│   ├── j2objc-annotations-2.8.jar
│   ├── jackson-annotations-2.13.2.jar
│   ├── jackson-core-2.13.2.jar
│   ├── jackson-databind-2.13.4.2.jar
│   ├── jackson-datatype-guava-2.13.2.jar
│   ├── jackson-datatype-jdk8-2.13.2.jar
│   ├── jackson-datatype-jsr310-2.13.5.jar
│   ├── jackson-module-parameter-names-2.13.5.jar
│   ├── jakarta.annotation-api-1.3.5.jar
│   ├── jakarta.json-2.0.1.jar
│   ├── jakarta.json-api-2.0.1.jar
│   ├── javax.inject-1.jar
│   ├── jcl-over-slf4j-1.7.36.jar
│   ├── jgrapht-core-0.9.3.jar
│   ├── jsonld-java-0.13.0.jar
│   ├── jsqlparser-4.4.jar
│   ├── jsr305-3.0.2.jar
│   ├── jul-to-slf4j-1.7.36.jar
│   ├── listenablefuture-9999.0-empty-to-avoid-conflict-with-guava.jar
│   ├── log4j-over-slf4j-1.7.36.jar
│   ├── logback-classic-1.2.13.jar
│   ├── logback-core-1.2.13.jar
│   ├── micrometer-core-1.9.17.jar
│   ├── nullanno-3.0.0.jar
│   ├── ontop-cli-5.3.0.jar
│   ├── ontop-endpoint-5.3.0.jar
│   ├── ontop-endpoint-core-5.3.0.jar
│   ├── ontop-kg-query-5.3.0.jar
│   ├── ontop-mapping-core-5.3.0.jar
│   ├── ontop-mapping-native-5.3.0.jar
│   ├── ontop-mapping-owlapi-5.3.0.jar
│   ├── ontop-mapping-r2rml-5.3.0.jar
│   ├── ontop-mapping-sql-all-5.3.0.jar
│   ├── ontop-mapping-sql-core-5.3.0.jar
│   ├── ontop-mapping-sql-owlapi-5.3.0.jar
│   ├── ontop-model-5.3.0.jar
│   ├── ontop-obda-core-5.3.0.jar
│   ├── ontop-ontology-owlapi-5.3.0.jar
│   ├── ontop-optimization-5.3.0.jar
│   ├── ontop-owlapi-5.3.0.jar
│   ├── ontop-rdb-5.3.0.jar
│   ├── ontop-rdf4j-5.3.0.jar
│   ├── ontop-reformulation-core-5.3.0.jar
│   ├── ontop-reformulation-sql-5.3.0.jar
│   ├── ontop-system-core-5.3.0.jar
│   ├── ontop-system-owlapi-5.3.0.jar
│   ├── ontop-system-sql-core-5.3.0.jar
│   ├── ontop-system-sql-owlapi-5.3.0.jar
│   ├── opencsv-5.7.1.jar
│   ├── org.protege.xmlcatalog-1.0.5.jar
│   ├── owlapi-api-5.5.1.jar
│   ├── owlapi-apibinding-5.5.1.jar
│   ├── owlapi-impl-5.5.1.jar
│   ├── owlapi-oboformat-5.5.1.jar
│   ├── owlapi-parsers-5.5.1.jar
│   ├── owlapi-rio-5.5.1.jar
│   ├── owlapi-tools-5.5.1.jar
│   ├── proj4j-1.1.1.jar
│   ├── r2rml-api-core-0.9.1.jar
│   ├── r2rml-api-rdf4j-binding-0.9.1.jar
│   ├── rdf4j-collection-factory-api-5.1.0.jar
│   ├── rdf4j-common-annotation-5.1.0.jar
│   ├── rdf4j-common-exception-5.1.0.jar
│   ├── rdf4j-common-io-5.1.0.jar
│   ├── rdf4j-common-iterator-5.1.0.jar
│   ├── rdf4j-common-order-5.1.0.jar
│   ├── rdf4j-common-text-5.1.0.jar
│   ├── rdf4j-common-transaction-5.1.0.jar
│   ├── rdf4j-common-xml-5.1.0.jar
│   ├── rdf4j-http-client-5.1.0.jar
│   ├── rdf4j-http-protocol-5.1.0.jar
│   ├── rdf4j-model-5.1.0.jar
│   ├── rdf4j-model-api-5.1.0.jar
│   ├── rdf4j-model-vocabulary-5.1.0.jar
│   ├── rdf4j-query-5.1.0.jar
│   ├── rdf4j-queryalgebra-evaluation-5.1.0.jar
│   ├── rdf4j-queryalgebra-model-5.1.0.jar
│   ├── rdf4j-queryparser-api-5.1.0.jar
│   ├── rdf4j-queryparser-sparql-5.1.0.jar
│   ├── rdf4j-queryrender-5.1.0.jar
│   ├── rdf4j-queryresultio-api-5.1.0.jar
│   ├── rdf4j-queryresultio-binary-5.1.0.jar
│   ├── rdf4j-queryresultio-sparqljson-5.1.0.jar
│   ├── rdf4j-queryresultio-sparqlxml-5.1.0.jar
│   ├── rdf4j-queryresultio-text-5.1.0.jar
│   ├── rdf4j-repository-api-5.1.0.jar
│   ├── rdf4j-repository-sail-5.1.0.jar
│   ├── rdf4j-repository-sparql-5.1.0.jar
│   ├── rdf4j-rio-api-5.1.0.jar
│   ├── rdf4j-rio-binary-5.1.0.jar
│   ├── rdf4j-rio-datatypes-5.1.0.jar
│   ├── rdf4j-rio-hdt-5.1.0.jar
│   ├── rdf4j-rio-jsonld-5.1.0.jar
│   ├── rdf4j-rio-languages-5.1.0.jar
│   ├── rdf4j-rio-n3-5.1.0.jar
│   ├── rdf4j-rio-nquads-5.1.0.jar
│   ├── rdf4j-rio-ntriples-5.1.0.jar
│   ├── rdf4j-rio-rdfjson-5.1.0.jar
│   ├── rdf4j-rio-rdfxml-5.1.0.jar
│   ├── rdf4j-rio-trig-5.1.0.jar
│   ├── rdf4j-rio-trix-5.1.0.jar
│   ├── rdf4j-rio-turtle-5.1.0.jar
│   ├── rdf4j-sail-api-5.1.0.jar
│   ├── rdf4j-sail-base-5.1.0.jar
│   ├── rdf4j-sail-memory-5.1.0.jar
│   ├── slf4j-api-1.7.36.jar
│   ├── spring-aop-5.3.31.jar
│   ├── spring-beans-5.3.31.jar
│   ├── spring-boot-2.7.18.jar
│   ├── spring-boot-actuator-2.7.18.jar
│   ├── spring-boot-actuator-autoconfigure-2.7.18.jar
│   ├── spring-boot-autoconfigure-2.7.18.jar
│   ├── spring-boot-starter-2.7.18.jar
│   ├── spring-boot-starter-actuator-2.7.18.jar
│   ├── spring-boot-starter-json-2.7.18.jar
│   ├── spring-boot-starter-logging-2.7.18.jar
│   ├── spring-boot-starter-thymeleaf-2.7.18.jar
│   ├── spring-boot-starter-tomcat-2.7.18.jar
│   ├── spring-boot-starter-web-2.7.18.jar
│   ├── spring-context-5.3.31.jar
│   ├── spring-core-5.3.31.jar
│   ├── spring-expression-5.3.31.jar
│   ├── spring-jcl-5.3.31.jar
│   ├── spring-web-5.3.31.jar
│   ├── spring-webmvc-5.3.31.jar
│   ├── thymeleaf-3.0.15.RELEASE.jar
│   ├── thymeleaf-extras-java8time-3.0.4.RELEASE.jar
│   ├── thymeleaf-spring5-3.0.15.RELEASE.jar
│   ├── tomcat-embed-core-9.0.98.jar
│   ├── tomcat-embed-el-9.0.83.jar
│   ├── tomcat-embed-websocket-9.0.83.jar
│   ├── tomcat-jdbc-10.0.0-M7.jar
│   ├── tomcat-juli-10.0.0-M7.jar
│   ├── toml4j-0.7.2.jar
│   ├── unbescape-1.1.6.RELEASE.jar
│   ├── urlbuilder-2.0.9.jar
│   ├── xml-resolver-1.2.jar
│   └── xz-1.9.jar
├── log
│   └── logback.xml
├── ontop
├── ontop-completion.sh
└── ontop.bat
```

如果需要链接sqlite3，需要将dirver jar包放在 `ontop-cli/jdbc` 目录下。

比如：sqlite-jdbc-3.48.0.0.jar（自行找地方下载）

---

### 1. 下载 Ontop CLI

```sh
wget https://github.com/ontop/ontop/releases/latest/download/ontop.zip
```
如果没有 `wget`，可以用 `curl`：
```sh
curl -L -o ontop.zip https://github.com/ontop/ontop/releases/latest/download/ontop.zip
```

---

### 2. 解压 Ontop

```sh
unzip ontop.zip -d ontop-cli
```

---

### 3. 进入 Ontop 目录

```sh
cd ontop-cli
```

---

### 4. 运行 Ontop CLI

```sh
./ontop
```
或者加上参数查看帮助：
```sh
./ontop --help
```

---

### 5. （可选）将 Ontop CLI 加入 PATH

你可以将 `ontop-cli` 目录加入 PATH，或将 `ontop` 脚本软链接到 `/usr/local/bin`：

```sh
ln -s "$(pwd)/ontop" /usr/local/bin/ontop
```

---

### 6. 检查 Java 环境

Ontop 需要 Java 8+，请确保已安装 Java（可用 `java -version` 检查）。

---

如需 SQLite JDBC 驱动，Ontop 已内置，无需单独下载。

---

**参考：**  
- 官方文档：https://ontop-vkg.org/guide/cli/
- GitHub Releases：https://github.com/ontop/ontop/releases
