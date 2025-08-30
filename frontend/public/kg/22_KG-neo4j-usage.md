当您使用Docker启动Neo4j后，可以通过命令行与Neo4j进行交互。以下是一些常用的操作指令：

### 1. 启动Neo4j容器
首先，确保您已经下载了Neo4j的Docker镜像，并使用以下命令启动容器：

```bash
docker run --name neo4j-container -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password -d neo4j

sudo docker run --name neo4j-container -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/123123csd -d neo4j
```

这里：
- `--name neo4j-container` 为容器指定一个名称。
- `-p 7474:7474` 将容器的7474端口映射到宿主机的7474端口（Neo4j Browser使用）。
- `-p 7687:7687` 将容器的7687端口映射到宿主机的7687端口（Bolt协议使用）。
- `-e NEO4J_AUTH=neo4j/password` 设置Neo4j的用户名和密码。
- `-d` 表示以 detached 模式运行容器。
- `neo4j` 是Neo4j的Docker镜像名称。

### 2. 访问Neo4j Browser
在浏览器中访问 `http://localhost:7474` 来使用Neo4j Browser。

### 3. 进入Neo4j容器
使用以下命令进入正在运行的Neo4j容器：

```bash
docker exec -it neo4j-container /bin/bash
```

### 4. 使用Cypher Shell
在容器内部，您可以使用Cypher Shell执行Cypher查询：

```bash
neo4j-shell
```

### 5. 一些常用的Cypher查询指令
在Cypher Shell中，您可以执行以下操作：

- **创建节点**：
  ```cypher
  CREATE (a:Person {name: 'Alice', age: 23})
  ```

- **创建关系**：
  ```cypher
  MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
  CREATE (a)-[:KNOWS]->(b)
  ```

- **查询节点和关系**：
  ```cypher
  MATCH (p:Person)-[r:KNOWS]->(friend)
  RETURN p.name, friend.name
  ```

- **更新节点属性**：
  ```cypher
  MATCH (p:Person {name: 'Alice'})
  SET p.age = 24
  ```

- **删除节点或关系**：
  ```cypher
  MATCH (p:Person {name: 'Alice'})
  DETACH DELETE p
  ```

- **使用索引查询**：
  ```cypher
  CREATE INDEX ON :Person(name)
  MATCH (p:Person)
  WHERE p.name = 'Alice'
  RETURN p
  ```

- **聚合查询**：
  ```cypher
  MATCH (p:Person)
  RETURN COUNT(p), AVG(p.age)
  ```


```shell
// 创建杜甫节点
CREATE (dufu:Persony {
  name: "杜甫",
  字号: "字子美",
  朝代: "唐朝",
  年龄: "58岁（公元712年－770年）",
  称号: "诗圣",
  事迹: "杜甫是唐代伟大的现实主义诗人，与李白并称“李杜”。他的诗歌深刻反映了唐代安史之乱前后的社会现实，表达了对国家和人民的深切关怀。杜甫的诗歌风格多样，既有豪放的边塞诗，也有深沉的忧国忧民之作。他的诗作对后世影响深远，被后人尊称为“诗圣”。"
})

// 创建代表作节点并链接到杜甫
with ["《春望》", "《登高》", "《茅屋为秋风所破歌》", "《三吏》", "《三别》"] AS 代表作0
unwind 代表作0 as 代表作 
merge (dufu)-[:AUTHORED]->(:Poem {name: 代表作})

// 创建链接节点并链接到杜甫
merge (dufu)-[:REFERENCES]->(:Wiki {
  baiduBaike: "https://baike.baidu.com/item/%E6%9D%9C%E5%BF%97/7397",
  wikipedia: "https://zh.wikipedia.org/wiki/%E6%9D%9C%E5%BF%97"
})

```