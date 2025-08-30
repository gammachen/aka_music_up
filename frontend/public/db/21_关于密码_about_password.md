# 数据库密码存储技术详解

## 一、密码存储概述

在数据库设计中，用户密码的安全存储是系统安全的关键环节。不当的密码存储方式可能导致用户数据泄露，造成严重的安全事件。本文将详细介绍数据库中密码存储的关键技术，包括多种实施方案、技术实现以及相关风险。

### 1.1 密码存储的基本原则

- **永不明文存储**：密码绝不应以明文形式存储在数据库中
- **单向转换**：密码应通过单向函数转换，无法从存储值反推原始密码
- **抵抗彩虹表**：存储机制应能抵抗预计算攻击（如彩虹表）
- **计算成本**：验证过程应有适当的计算成本，以抵抗暴力破解
- **现代算法**：使用经过时间验证的现代加密算法

## 二、密码存储方案对比

### 2.1 明文存储（极不推荐）

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(50) NOT NULL  -- 明文密码
);
```

**风险**：
- 数据库被入侵时，所有用户密码立即泄露
- 内部人员可直接查看用户密码
- 违反多数数据保护法规和安全标准
- 用户在多平台使用相同密码时，风险扩大

### 2.2 简单哈希（不推荐）

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password_hash VARCHAR(64) NOT NULL  -- MD5/SHA-1/SHA-256哈希值
);
```

**实现示例**（SHA-256）：
```python
import hashlib

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hash):
    return hash_password(password) == hash
```

**风险**：
- 易受彩虹表攻击
- 相同密码产生相同哈希值
- 计算速度快，易受暴力破解

### 2.3 加盐哈希（基本可接受）

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password_hash VARCHAR(64) NOT NULL,  -- 哈希值
    salt VARCHAR(32) NOT NULL  -- 随机盐值
);
```

**实现示例**（SHA-256 + 盐值）：
```python
import hashlib
import os

def generate_salt():
    return os.urandom(16).hex()

def hash_password(password, salt):
    return hashlib.sha256((password + salt).encode()).hexdigest()

def create_user(username, password):
    salt = generate_salt()
    hash = hash_password(password, salt)
    # 将username, hash, salt存入数据库
    
def verify_password(password, stored_hash, salt):
    return hash_password(password, salt) == stored_hash
```

**优势**：
- 抵抗彩虹表攻击
- 相同密码产生不同哈希值

**局限性**：
- 计算速度仍然较快，可能受到GPU加速的暴力破解
- 缺乏迭代机制，无法增加计算成本

### 2.4 密钥拉伸算法（推荐）

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password_hash VARCHAR(128) NOT NULL,  -- 哈希值
    salt VARCHAR(32) NOT NULL,  -- 随机盐值
    algorithm VARCHAR(20) NOT NULL  -- 算法标识
);
```

#### 2.4.1 PBKDF2（密码基础密钥派生函数）

**实现示例**：
```python
import os
import hashlib
import binascii

def hash_password(password, salt=None, iterations=100000):
    if salt is None:
        salt = os.urandom(16)  # 16字节的随机盐值
    
    # PBKDF2-HMAC-SHA256算法
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        iterations,
        dklen=32  # 32字节的派生密钥
    )
    
    # 转换为十六进制字符串
    hash = binascii.hexlify(key).decode('ascii')
    salt_hex = binascii.hexlify(salt).decode('ascii')
    
    return f"pbkdf2:sha256:{iterations}:{salt_hex}:{hash}"

def verify_password(password, stored_hash):
    # 解析存储的哈希字符串
    algorithm, hash_name, iterations, salt_hex, hash = stored_hash.split(':')
    iterations = int(iterations)
    salt = binascii.unhexlify(salt_hex)
    
    # 使用相同参数重新计算哈希值
    key = hashlib.pbkdf2_hmac(
        hash_name,
        password.encode('utf-8'),
        salt,
        iterations,
        dklen=32
    )
    
    # 比较计算得到的哈希值与存储的哈希值
    new_hash = binascii.hexlify(key).decode('ascii')
    return new_hash == hash
```

#### 2.4.2 Bcrypt

**实现示例**：
```python
import bcrypt

def hash_password(password, cost=12):
    # 自动生成盐值并哈希密码
    password_bytes = password.encode('utf-8')
    hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt(cost))
    return hashed.decode('utf-8')

def verify_password(password, stored_hash):
    password_bytes = password.encode('utf-8')
    stored_hash_bytes = stored_hash.encode('utf-8')
    return bcrypt.checkpw(password_bytes, stored_hash_bytes)
```

#### 2.4.3 Argon2（推荐）

**实现示例**：
```python
from argon2 import PasswordHasher

def hash_password(password):
    ph = PasswordHasher()
    return ph.hash(password)

def verify_password(password, stored_hash):
    ph = PasswordHasher()
    try:
        return ph.verify(stored_hash, password)
    except:
        return False
```

**优势**：
- 可调节的计算成本（时间、内存、并行度）
- 抵抗GPU/ASIC加速攻击
- Argon2专为密码哈希设计，获得密码哈希竞赛冠军

## 三、常见攻击与防御

### 3.1 彩虹表攻击

**攻击原理**：
彩虹表是预计算的哈希值表，攻击者通过查表快速找到哈希值对应的原始密码。

**防御措施**：
- 使用唯一盐值
- 增加哈希计算复杂度
- 使用现代密钥派生函数

### 3.2 暴力破解攻击

**攻击原理**：
攻击者尝试所有可能的密码组合，直到找到匹配的哈希值。

**防御措施**：
- 使用计算密集型哈希算法
- 实施密码策略（长度、复杂度）
- 实施登录尝试限制
- 使用内存密集型算法（如Argon2）抵抗GPU加速

### 3.3 旁路攻击

**攻击原理**：
通过分析系统行为（如响应时间）推断密码信息。

**防御措施**：
- 实现恒定时间比较
- 避免在错误消息中泄露信息

## 四、当前系统实现分析

我们的系统目前使用SHA-256 + 盐值加密方案存储密码：

```sql
CREATE TABLE user (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(20) NOT NULL UNIQUE,
    password_hash VARCHAR(64) NOT NULL,
    -- 其他字段
);
```

**当前实现的优势**：
- 使用了盐值防止彩虹表攻击
- SHA-256是一种强哈希算法

**改进建议**：
1. 迁移到密钥拉伸算法（如Argon2或Bcrypt）
2. 存储算法标识符，便于未来算法升级
3. 实现密码重置时的渐进式升级

## 五、密码存储最佳实践

### 5.1 算法选择

按推荐优先级排序：
1. **Argon2id**：内存硬化、抗GPU/ASIC、可调参数
2. **Bcrypt**：经过时间验证、广泛支持
3. **PBKDF2**：NIST认证、广泛支持

### 5.2 实施建议

- **盐值长度**：至少16字节的随机值
- **迭代次数**：根据硬件定期调整，保持验证时间在~250ms
- **算法参数存储**：与哈希值一起存储所有参数
- **渐进式升级**：在用户登录时升级旧哈希算法

### 5.3 密码策略

- **最小长度**：至少10个字符
- **复杂度要求**：混合使用大小写字母、数字和特殊字符
- **密码轮换**：避免强制频繁更换（可能导致弱密码）
- **密码检查**：对照已泄露密码数据库（如Have I Been Pwned）验证

### 5.4 系统安全措施

- **传输加密**：使用HTTPS/TLS保护密码传输
- **多因素认证**：实施MFA作为额外安全层
- **登录限制**：实施基于IP的登录尝试限制
- **安全监控**：监控异常登录活动

## 六、密码存储升级路径

### 6.1 从SHA-256+盐值升级到Argon2

**步骤**：
1. 更新数据库结构，添加算法标识字段
2. 修改验证逻辑，支持多种哈希格式
3. 在用户成功登录时，重新哈希密码并更新存储

**代码示例**：
```python
from argon2 import PasswordHasher
import hashlib

def verify_and_upgrade(username, password, stored_hash, salt, algorithm="sha256"):
    # 验证现有密码
    if algorithm == "sha256":
        # 旧的SHA-256 + 盐值验证
        current_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        if current_hash != stored_hash:
            return False
            
        # 密码正确，升级到Argon2
        ph = PasswordHasher()
        new_hash = ph.hash(password)
        
        # 更新数据库中的密码哈希和算法标识
        update_password_hash(username, new_hash, algorithm="argon2")
        return True
        
    elif algorithm == "argon2":
        # 已经是Argon2，直接验证
        ph = PasswordHasher()
        try:
            return ph.verify(stored_hash, password)
        except:
            return False
    
    return False
```

## 七、总结

密码存储是系统安全的关键环节，选择合适的密码存储技术对保护用户数据至关重要。本文详细介绍了从明文存储到现代密钥派生函数的多种密码存储方案，并分析了各自的优缺点和安全风险。

在实际应用中，应当：

1. 避免使用明文或简单哈希存储密码
2. 优先选择现代密钥派生函数（Argon2、Bcrypt）
3. 正确实施盐值和迭代机制
4. 定期评估和升级密码存储方案
5. 结合其他安全措施（如MFA、登录限制）提供多层次保护

通过采用本文推荐的最佳实践，可以显著提高系统的密码安全性，有效保护用户账户免受各种常见攻击。