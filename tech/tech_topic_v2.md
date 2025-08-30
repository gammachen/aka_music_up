# 音乐站点主题功能技术方案 V2

## 1. 系统架构

### 1.1 整体架构
```
                 ┌─────────────┐
                 │   客户端    │
                 └─────┬───────┘
                       │
                       ▼
                 ┌─────────────┐
                 │   API网关   │ ←→ 限流、认证
                 └─────┬───────┘
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
    ┌─────────────┐         ┌─────────────┐
    │  主题服务   │         │  用户服务   │
    └─────┬───────┘         └─────────────┘
          │
    ┌─────┴───────┐
    ▼             ▼
┌─────────┐   ┌─────────┐
│ MySQL   │   │ Redis   │
└─────────┘   └─────────┘
```

### 1.2 核心模块

- **内容安全模块**：负责主题内容的安全检查，包括敏感词过滤、XSS防护等
- **访问控制模块**：处理用户权限验证和访问频率限制
- **缓存管理模块**：管理主题相关的缓存策略
- **计数器模块**：处理点赞、收藏等计数的原子操作
- **通知模块**：处理主题相关的消息通知

## 2. 数据模型

### 2.1 主题模型

```python
class Topic:
    """主题模型"""
    def __init__(self):
        self.id = None
        self.user_id = None
        self.title = ""
        self.content = ""
        self.hidden_content = ""
        self.hidden_cost = 0
        self.view_count = 0
        self.like_count = 0
        self.favorite_count = 0
        self.reply_count = 0
        self.status = "normal"
        self.is_top = False
        self.is_essence = False
        self.created_at = None
        self.updated_at = None
        
    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "content": self.content,
            "hidden_content": self.hidden_content if self.can_view_hidden() else None,
            "hidden_cost": self.hidden_cost,
            "stats": {
                "view_count": self.view_count,
                "like_count": self.like_count,
                "favorite_count": self.favorite_count,
                "reply_count": self.reply_count
            },
            "status": self.status,
            "is_top": self.is_top,
            "is_essence": self.is_essence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
        
    def can_view_hidden(self):
        """检查是否可以查看隐藏内容"""
        if not self.hidden_content:
            return True
            
        # 作者本人可以查看
        if current_user.id == self.user_id:
            return True
            
        # VIP用户可以查看
        if current_user.is_vip():
            return True
            
        # 已购买可以查看
        return TopicPurchase.exists(user_id=current_user.id, topic_id=self.id)
```

### 2.2 回复模型

```python
class Reply:
    """回复模型"""
    def __init__(self):
        self.id = None
        self.topic_id = None
        self.user_id = None
        self.content = ""
        self.parent_id = None
        self.like_count = 0
        self.status = "normal"
        self.created_at = None
        self.updated_at = None
        
    def build_tree(self):
        """构建回复树"""
        # 获取所有子回复
        children = Reply.query.filter_by(parent_id=self.id).all()
        
        # 递归构建树结构
        tree = self.to_dict()
        if children:
            tree["children"] = [child.build_tree() for child in children]
        return tree
```

## 3. 核心算法

### 3.1 内容安全检查

#### 3.1.1 敏感词过滤

使用DFA(确定有限自动机)算法实现高效的敏感词过滤：

```python
class SensitiveFilter:
    """敏感词过滤器"""
    def __init__(self):
        self.keyword_chains = {}  # 关键词链
        self.delimit = '\x00'     # 限定符
        
    def add_keyword(self, keyword):
        """添加敏感词到词链"""
        keyword = keyword.lower()
        chars = keyword.strip()
        if not chars:
            return
        level = self.keyword_chains
        for i in range(len(chars)):
            if chars[i] in level:
                level = level[chars[i]]
            else:
                if not isinstance(level, dict):
                    break
                for j in range(i, len(chars)):
                    level[chars[j]] = {}
                    last_level, last_char = level, chars[j]
                    level = level[chars[j]]
                last_level[last_char] = {self.delimit: 0}
                break
            if i == len(chars) - 1:
                level[self.delimit] = 0
                
    def filter(self, message, repl="*"):
        """检查文本是否包含敏感词"""
        message = message.lower()
        ret = []
        start = 0
        while start < len(message):
            level = self.keyword_chains
            step_ins = 0
            for char in message[start:]:
                if char in level:
                    step_ins += 1
                    if self.delimit in level[char]:
                        ret.append(repl * step_ins)
                        start += step_ins - 1
                        break
                    level = level[char]
                else:
                    ret.append(message[start])
                    break
            else:
                ret.append(message[start])
            start += 1
        return ''.join(ret)
```

#### 3.1.2 XSS防护

使用白名单机制实现HTML内容过滤：

```python
class XSSFilter:
    """XSS过滤器"""
    def __init__(self):
        self.allowed_tags = {
            'p': ['style', 'class'],
            'br': [],
            'strong': [],
            'em': [],
            'img': ['src', 'alt', 'title', 'width', 'height'],
            'a': ['href', 'title', 'target'],
            'code': ['class'],
            'pre': ['class'],
            'ul': ['class'],
            'li': ['class']
        }
        
    def clean(self, html_content):
        """清理HTML内容"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for tag in soup.find_all(True):
            if tag.name not in self.allowed_tags:
                tag.unwrap()
            else:
                attrs = dict(tag.attrs)
                allowed_attrs = self.allowed_tags[tag.name]
                for attr in attrs:
                    if attr not in allowed_attrs:
                        del tag[attr]
                        
        return str(soup)
```

### 3.2 访问频率限制

使用令牌桶算法实现精确的访问频率控制：

```python
class TokenBucket:
    """令牌桶算法实现"""
    def __init__(self, capacity, fill_rate):
        self.capacity = capacity    # 桶容量
        self.fill_rate = fill_rate  # 填充速率
        self.tokens = capacity      # 当前令牌数
        self.last_fill = time.time()
        self.lock = threading.Lock()
        
    def consume(self, tokens=1):
        """消费令牌"""
        with self.lock:
            now = time.time()
            # 计算需要填充的令牌
            if self.tokens < self.capacity:
                delta = (now - self.last_fill) * self.fill_rate
                self.tokens = min(self.capacity, self.tokens + delta)
                self.last_fill = now
            
            # 检查是否有足够的令牌
            if tokens <= self.tokens:
                self.tokens -= tokens
                return True
            return False

class RateLimiter:
    """访问频率限制器"""
    def __init__(self):
        self.buckets = {}
        
    def is_allowed(self, key, capacity=60, fill_rate=1):
        """检查是否允许访问"""
        if key not in self.buckets:
            self.buckets[key] = TokenBucket(capacity, fill_rate)
        return self.buckets[key].consume()
```

### 3.3 缓存更新策略

采用写直达(Write-Through)和延迟双删策略确保缓存一致性：

```python
class CacheManager:
    """缓存管理器"""
    def __init__(self):
        self.redis = Redis()
        self.delay = 0.5  # 延迟删除等待时间(秒)
        
    def get_topic(self, topic_id):
        """获取主题信息"""
        # 尝试从缓存获取
        cache_key = f'topic:{topic_id}'
        data = self.redis.get(cache_key)
        if data:
            return json.loads(data)
            
        # 从数据库获取
        topic = Topic.query.get(topic_id)
        if topic:
            # 写入缓存
            self.redis.setex(
                cache_key,
                3600,  # 1小时过期
                json.dumps(topic.to_dict())
            )
            return topic.to_dict()
        return None
        
    def update_topic(self, topic):
        """更新主题信息"""
        cache_key = f'topic:{topic.id}'
        
        # 删除缓存
        self.redis.delete(cache_key)
        
        # 更新数据库
        db.session.commit()
        
        # 延迟双删
        def delay_delete():
            time.sleep(self.delay)
            self.redis.delete(cache_key)
            
        threading.Thread(target=delay_delete).start()
```

### 3.4 并发控制

使用乐观锁机制处理并发更新：

```python
class TopicLock:
    """主题并发控制"""
    def __init__(self):
        self.redis = Redis()
        
    def acquire_lock(self, topic_id, timeout=10):
        """获取锁"""
        lock_key = f'lock:topic:{topic_id}'
        lock_value = str(uuid.uuid4())
        
        # 尝试获取锁
        if self.redis.set(lock_key, lock_value, nx=True, ex=timeout):
            return lock_value
        return None
        
    def release_lock(self, topic_id, lock_value):
        """释放锁"""
        lock_key = f'lock:topic:{topic_id}'
        
        # 使用Lua脚本保证原子性
        script = """
            if redis.call('get', KEYS[1]) == ARGV[1] then
                return redis.call('del', KEYS[1])
            end
            return 0
        """
        self.redis.eval(script, 1, lock_key, lock_value)
        
    @contextmanager
    def lock(self, topic_id, timeout=10):
        """锁上下文管理器"""
        lock_value = self.acquire_lock(topic_id, timeout)
        if not lock_value:
            raise LockError('无法获取锁')
            
        try:
            yield
        finally:
            self.release_lock(topic_id, lock_value)
```

## 4. 性能优化

### 4.1 查询优化

#### 4.1.1 索引优化

```sql
-- 主题表索引优化
CREATE INDEX idx_topics_composite ON topics(status, is_top DESC, created_at DESC);
CREATE INDEX idx_topics_user_time ON topics(user_id, created_at DESC);

-- 回复表索引优化
CREATE INDEX idx_replies_composite ON replies(topic_id, created_at DESC);
```

#### 4.1.2 分页查询优化

使用游标分页替代传统的OFFSET分页：

```python
def get_topic_list(cursor=None, limit=20):
    """获取主题列表"""
    query = Topic.query.filter_by(status='normal')
    
    if cursor: