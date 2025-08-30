在实际开发中，确实可以通过简单的哈希（如 `hash(uid) % N`）实现流量分桶，但在**大规模系统**（如 AB 测试、分布式存储、负载均衡）中，直接使用简单哈希函数（如 `hashCode`、`MD5`）可能会导致以下问题：

---

### **1. 为什么不能直接用简单哈希 + 取模？**
#### **(1) 性能问题**
- **MD5、SHA 等加密哈希函数**虽然分布均匀，但计算成本高（尤其是处理大量数据时）。例如：
  - `MD5` 的计算速度通常比 `MurmurHash` 慢 **50% 以上**（参考知识库中的实测数据）。
  - 在 AB 测试系统中，如果每个请求都要计算 `MD5(uid)`，会显著增加延迟。
- **JDK 的 `hashCode`** 虽然快，但分布性差（尤其是对字符串等复杂数据），容易导致数据倾斜。

#### **(2) 分布不均匀**
- **简单哈希函数的随机性不足**：例如，`hashCode` 对某些特定输入（如连续数字、短字符串）可能产生聚集效应，导致分桶不均。
- **碰撞率高**：如果哈希函数设计不佳，不同用户可能被分配到相同桶中（即哈希冲突），影响实验的公平性。

#### **(3) 跨语言/平台不一致**
- **`hashCode` 在不同语言/平台上的实现可能不同**：例如，Java 的 `String.hashCode()` 与 Python 的 `hash()` 对同一字符串的输出可能不同，导致跨系统分桶不一致。
- **加密哈希（如 MD5）虽然跨平台一致，但计算开销大**。

#### **(4) 动态扩展问题**
- **当分桶数变化时（如从 10 个桶扩展到 20 个桶）**，直接取模会导致大量用户重新分配到新桶，迁移成本高。
- **需要结合一致性哈希或虚拟节点**（如 `MurmurHash` + 虚拟节点）来减少迁移影响。

---

### **2. 为什么选择 MurmurHash、CityHash？**
#### **(1) 高性能**
- **MurmurHash3** 和 **CityHash** 是专为 **非加密场景** 设计的高性能哈希函数：
  - **速度极快**：例如，`MurmurHash3` 的速度是 `MD5` 的 **10 倍**（参考知识库中的测试数据）。
  - **适合高频场景**：如 AB 测试系统中每秒数万次的哈希计算，性能优势显著。

#### **(2) 低碰撞率**
- **MurmurHash3** 和 **CityHash** 的设计目标是 **均匀分布** 和 **低碰撞率**：
  - 对于 1 亿用户的 UID，哈希值无冲突（参考知识库中的实测数据）。
  - 在 AB 测试中，确保不同用户被均匀分配到不同桶，避免实验偏差。

#### **(3) 良好的随机性**
- **对相似输入敏感**：例如，`MurmurHash("user123_layer1")` 和 `MurmurHash("user123_layer2")` 的输出差异显著，适合多层实验分桶（正交性验证）。
- **避免“近似输入导致相同输出”**：这是 AB 测试中多层实验的关键需求。

#### **(4) 跨平台一致性**
- **MurmurHash** 和 **CityHash** 在主流语言（Java、Python、C++、Go 等）中都有成熟实现，且输出结果一致。
  - 例如，Java 中使用 `Guava` 的 `MurmurHash3`，Python 中使用 `mmh3`，结果完全一致。
  - 这在跨系统（前端 + 后端 + 数据分析）的 AB 测试中至关重要。

#### **(5) 支持虚拟节点**
- **结合一致性哈希**：通过虚拟节点（Virtual Node）技术，可以进一步优化数据分布均匀性。
  - 例如，将每个物理节点拆分为多个虚拟节点，均匀分布在哈希环上。

---

### **3. 实际场景对比**
#### **场景 1：AB 测试系统**
- **需求**：将用户均匀分配到 100 个桶，支持多层实验（如推荐策略、价格策略）。
- **简单哈希方案**：
  ```python
  # 使用 MD5 的方案（低效）
  def get_bucket(uid, salt):
      return int(hashlib.md5(f"{uid}_{salt}".encode()).hexdigest(), 16) % 100
  ```
- **MurmurHash 方案**：
  ```python
  # 使用 mmh3（高效且分布均匀）
  import mmh3
  def get_bucket(uid, salt):
      return mmh3.hash(f"{uid}_{salt}") % 100
  ```

#### **场景 2：分布式存储**
- **需求**：将数据均匀分布到 10 个节点。
- **简单哈希方案**：
  ```java
  // 使用 JDK hashCode（分布不均）
  int bucket = uid.hashCode() % 10;
  ```
- **MurmurHash 方案**：
  ```java
  // 使用 Guava 的 MurmurHash3（分布均匀）
  HashFunction hashFunc = Hashing.murmur3_128();
  int bucket = (int) (hashFunc.hashString(uid).asLong() % 10);
  ```

---

### **4. 总结**
| 特性                | 简单哈希（如 MD5、hashCode） | MurmurHash / CityHash         |
|---------------------|-----------------------------|-------------------------------|
| **性能**            | 慢                          | 极快（10 倍于 MD5）           |
| **分布均匀性**      | 差                          | 优秀                          |
| **碰撞率**          | 高                          | 极低                          |
| **跨平台一致性**    | 不一致                      | 一致                          |
| **动态扩展支持**    | 需额外算法（如一致性哈希）  | 可结合虚拟节点优化            |
| **适用场景**        | 小规模、非性能敏感场景      | 大规模、高性能需求场景        |

---

### **5. 如何选择？**
- **小规模系统**：直接使用 `hashCode` 或 `MD5` 可能足够。
- **大规模系统**（如 AB 测试、分布式存储）：
  - **首选 MurmurHash3**（平衡性能、分布性和跨平台一致性）。
  - **可选 CityHash**（速度略快，但语言支持较少）。

---

### **6. 代码示例**
#### **Python（MurmurHash）**
```python
import mmh3

def get_bucket(uid, salt, total_buckets=100):
    return mmh3.hash(f"{uid}_{salt}") % total_buckets
```

#### **Java（Guava 的 MurmurHash3）**
```java
import com.google.common.hash.Hashing;

public class BucketUtil {
    public static int getBucket(String uid, String salt, int totalBuckets) {
        long hash = Hashing.murmur3_128().hashString(uid + salt).asLong();
        return (int) (hash % totalBuckets);
    }
}
```

#### **Go（CityHash）**
```go
import "github.com/dgryski/go-cityhash"

func GetBucket(uid, salt string, totalBuckets int) int {
    hash := cityhash.CityHash64([]byte(uid + salt))
    return int(hash % uint64(totalBuckets))
}
```

---

通过选择 **MurmurHash** 或 **CityHash**，可以在保证性能的同时，解决分布不均、跨平台一致性等问题，是大规模系统中更优的选择。