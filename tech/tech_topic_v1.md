# 音乐站点主题功能技术方案

## 1. 数据库设计

### 1.1 主题表（topics）
```sql
CREATE TABLE topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    title VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    hidden_content TEXT,  -- 隐藏内容，需要支付金币查看
    hidden_cost INTEGER DEFAULT 0,  -- 查看隐藏内容需要的金币数
    view_count INTEGER DEFAULT 0,  -- 浏览次数
    like_count INTEGER DEFAULT 0,  -- 点赞数
    favorite_count INTEGER DEFAULT 0,  -- 收藏数
    reply_count INTEGER DEFAULT 0,  -- 回复数
    status VARCHAR(10) NOT NULL DEFAULT 'normal',  -- normal/hidden/deleted
    is_top BOOLEAN DEFAULT 0,  -- 是否置顶
    is_essence BOOLEAN DEFAULT 0,  -- 是否精华
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 主题相关索引
CREATE INDEX idx_topics_user ON topics(user_id);
CREATE INDEX idx_topics_status ON topics(status);
CREATE INDEX idx_topics_created ON topics(created_at);
```

### 1.2 回复表（replies）
```sql
CREATE TABLE replies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    parent_id INTEGER,  -- 父回复ID，用于实现回复层级
    like_count INTEGER DEFAULT 0,
    status VARCHAR(10) NOT NULL DEFAULT 'normal',  -- normal/hidden/deleted
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (topic_id) REFERENCES topics(id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (parent_id) REFERENCES replies(id)
);

-- 回复相关索引
CREATE INDEX idx_replies_topic ON replies(topic_id);
CREATE INDEX idx_replies_user ON replies(user_id);
CREATE INDEX idx_replies_parent ON replies(parent_id);
```

### 1.3 点赞表（user_likes）
```sql
CREATE TABLE user_likes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    target_type VARCHAR(10) NOT NULL,  -- topic/reply
    target_id INTEGER NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    UNIQUE (user_id, target_type, target_id)  -- 防止重复点赞
);

-- 点赞相关索引
CREATE INDEX idx_likes_user ON user_likes(user_id);
CREATE INDEX idx_likes_target ON user_likes(target_type, target_id);
```

### 1.4 收藏表（user_favorites）
```sql
CREATE TABLE user_favorites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    topic_id INTEGER NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (topic_id) REFERENCES topics(id),
    UNIQUE (user_id, topic_id)  -- 防止重复收藏
);

-- 收藏相关索引
CREATE INDEX idx_favorites_user ON user_favorites(user_id);
CREATE INDEX idx_favorites_topic ON user_favorites(topic_id);
```

## 2. 接口设计

### 2.1 主题相关接口

#### 2.1.1 创建主题
```python
@app.route('/api/topics', methods=['POST'])
@login_required
def create_topic():
    """创建主题
    请求体：
    {
        "title": "主题标题",
        "content": "主题内容",
        "hidden_content": "隐藏内容",  # 可选
        "hidden_cost": 10  # 可选，默认0
    }
    """
    # 权限检查：用户等级限制、发帖间隔等
    # 内容检查：敏感词过滤、长度限制等
    # 创建主题
    # 更新用户经验值
    pass
```

#### 2.1.2 获取主题列表
```python
@app.route('/api/topics', methods=['GET'])
def get_topics():
    """获取主题列表
    参数：
    - page: 页码，默认1
    - size: 每页数量，默认20
    - sort: 排序方式（latest/popular/essence），默认latest
    """
    # 分页查询
    # 权限过滤
    # 排序处理
    pass
```

#### 2.1.3 获取主题详情
```python
@app.route('/api/topics/<int:topic_id>', methods=['GET'])
def get_topic(topic_id):
    """获取主题详情
    返回：
    {
        "id": 1,
        "title": "标题",
        "content": "内容",
        "hidden_content": "隐藏内容",  # 需要权限验证
        "user": {},  # 发帖用户信息
        "stats": {  # 统计信息
            "view_count": 100,
            "like_count": 50,
            "favorite_count": 30,
            "reply_count": 20
        },
        "is_liked": false,  # 当前用户是否已点赞
        "is_favorited": false  # 当前用户是否已收藏
    }
    """
    # 权限检查
    # 增加浏览次数
    # 处理隐藏内容
    pass
```

### 2.2 回复相关接口

#### 2.2.1 创建回复
```python
@app.route('/api/topics/<int:topic_id>/replies', methods=['POST'])
@login_required
def create_reply(topic_id):
    """创建回复
    请求体：
    {
        "content": "回复内容",
        "parent_id": 1  # 可选，回复其他回复
    }
    """
    # 权限检查
    # 内容检查
    # 创建回复
    # 更新主题回复数
    # 更新用户经验值
    pass
```

#### 2.2.2 获取回复列表
```python
@app.route('/api/topics/<int:topic_id>/replies', methods=['GET'])
def get_replies(topic_id):
    """获取回复列表
    参数：
    - page: 页码
    - size: 每页数量
    """
    # 分页查询
    # 构建回复树
    # 权限过滤
    pass
```

### 2.3 点赞相关接口

#### 2.3.1 点赞/取消点赞
```python
@app.route('/api/likes', methods=['POST'])
@login_required
def toggle_like():
    """点赞或取消点赞
    请求体：
    {
        "target_type": "topic",  # topic/reply
        "target_id": 1
    }
    """
    # 权限检查
    # 点赞去重
    # 更新目标点赞数
    # 更新用户经验值
    pass
```

### 2.4 收藏相关接口

#### 2.4.1 收藏/取消收藏
```python
@app.route('/api/favorites', methods=['POST'])
@login_required
def toggle_favorite():
    """收藏或取消收藏
    请求体：
    {
        "topic_id": 1
    }
    """
    # 权限检查
    # 收藏去重
    # 更新主题收藏数
    # 更新用户经验值
    pass
```

## 3. 业务逻辑实现

### 3.1 隐藏内容处理
```python
def check_hidden_content_access(user, topic):
    """检查用户是否有权限查看隐藏内容"""
    if not topic.hidden_content:
        return True
    
    # VIP用户免费查看
    if user.is_vip():
        return True
    
    # 检查是否已支付
    if db.session.query(TopicPurchase).filter_by(
        user_id=user.id,
        topic_id=topic.id
    ).first():
        return True
    
    return False

def purchase_hidden_content(user, topic):
    """购买隐藏内容"""
    # 检查金币余额
    if user.coins < topic.hidden_cost:
        raise InsufficientCoinsError()
    
    # 扣除金币
    user.coins -= topic.hidden_cost
    
    # 记录购买记录
    purchase = TopicPurchase(
        user_id=user.id,
        topic_id=topic.id,
        cost=topic.hidden_cost
    )
    db.session.add(purchase)
    
    # 分成给作者
    author = topic.user
    author.coins += topic.hidden_cost * 0.7  # 作者获得70%分成
    
    db.session.commit()
```

### 3.2 权限控制
```python
def check_topic_permission(user, action):
    """检查发帖权限"""
    if action == 'create':
        # 等级限制
        if user.level < 2:
            raise PermissionError('需要达到2级才能发帖')
        
        # 发帖间隔
        last_topic = db.session.query(Topic).filter_by(
            user_id=user.id
        ).order_by(Topic.created_at.desc()).first()
        
        if last_topic and \
           (datetime.now() - last_topic.created_at).seconds < 300:
            raise PermissionError('发帖太频繁，请稍后再试')
    
    elif action == 'edit':
        # 只允许编辑24小时内的帖子
        if (datetime.now() - topic.created_at).days >= 1:
            raise PermissionError('只能编辑24小时内的帖子')
```

### 3.3 内容安全
```python
def check_content_security(content):
    """内容安全检查"""
    # 敏感词过滤
    if contains_sensitive_words(content):
        raise ContentSecurityError('内容包含敏感词')
    
    # XSS过滤
    content = bleach.clean(
        content,
        tags=['p', 'br', 'strong', 'em', 'img'],
        attributes={'img': ['src']}
    )
    
    # 图片处理
    content = process_images(content)
    
    return content
```

## 4. 性能优化

### 4.1 缓存策略
```python
def get_topic_with_cache(topic_id):
    """获取主题信息（带缓存）"""
    cache_key = f'topic:{topic_id}'
    
    # 尝试从缓存获取
    topic = redis.get(cache_key)
    if topic:
        return json.loads(topic)
    
    # 从数据库获取
    topic = db.session.query(Topic).get(topic_id)
    if not topic:
        return None
    
    # 写入缓存
    topic_data = topic.to_dict()
    redis.setex(
        cache_key,
        3600,  # 1小时过期
        json.dumps(topic_data)
    )
    
    return topic_data
```

### 4.2 计数器优化
```python
def increment_view_count(topic_id):
    """增加浏览次数（异步）"""
    cache_key = f'topic:views:{topic_id}'
    
    # 增加缓存计数
    redis.incr(cache_key)
    
    # 定时同步到数据库
    if redis.get(cache_key) % 10 == 0:  # 每10次同步一次
        count = int(redis.get(cache_key) or 0)
        db.session.query(Topic).filter_by(id=topic_id).update({
            'view_count': count
        })
        db.session.commit()
```

## 5. 安全措施

### 5.1 防重复提交
```python
def prevent_duplicate_submit(key, expire_seconds=5):
    """防重复提交"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成防重复key
            cache_key = f'duplicate:{key}:{request.user.id}'
            
            # 检查是否存在防重复标记
            if redis.exists(cache_key):
                raise DuplicateSubmitError('请勿重复提交')
            
            # 设置防重复标记
            redis.setex(cache_key, expire_seconds, 1)
            
            try:
                return func(*args, **kwargs)
            finally:
                redis.delete(cache_key)
            
        return wrapper
```

### 5.2 访问频率限制
```python
def rate_limit(key, limit=60, period=60):
    """访问频率限制
    Args:
        key: 限制key
        limit: 允许的最大访问次数
        period: 时间窗口(秒)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成限制key
            cache_key = f'ratelimit:{key}:{request.user.id}'
            
            # 获取当前访问次数
            current = redis.get(cache_key)
            if current is None:
                redis.setex(cache_key, period, 1)
            elif int(current) >= limit:
                raise RateLimitExceededError('访问太频繁，请稍后再试')
            else:
                redis.incr(cache_key)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

### 5.3 数据安全
```python
class DataSecurity:
    """数据安全处理"""
    
    @staticmethod
    def encrypt_sensitive_data(data, key):
        """加密敏感数据"""
        f = Fernet(key)
        return f.encrypt(data.encode()).decode()
    
    @staticmethod
    def decrypt_sensitive_data(encrypted_data, key):
        """解密敏感数据"""
        f = Fernet(key)
        return f.decrypt(encrypted_data.encode()).decode()
    
    @staticmethod
    def hash_password(password):
        """密码哈希"""
        return bcrypt.hashpw(
            password.encode(),
            bcrypt.gensalt()
        ).decode()
    
    @staticmethod
    def verify_password(password, hashed):
        """验证密码"""
        return bcrypt.checkpw(
            password.encode(),
            hashed.encode()
        )
```

### 5.4 XSS防护
```python
def xss_clean(content):
    """XSS清洗"""
    # 允许的HTML标签和属性
    allowed_tags = {
        'p': [],
        'br': [],
        'strong': [],
        'em': [],
        'img': ['src', 'alt'],
        'a': ['href', 'title'],
        'code': [],
        'pre': [],
        'ul': [],
        'li': []
    }
    
    # 使用bleach清洗内容
    cleaned = bleach.clean(
        content,
        tags=allowed_tags.keys(),
        attributes=allowed_tags
    )
    
    return cleaned
```

### 5.5 数据备份
```python
class DatabaseBackup:
    """数据库备份"""
    
    @staticmethod
    def create_backup():
        """创建备份"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f'backups/db_backup_{timestamp}.sql'
        
        # 导出数据库
        with open(backup_path, 'w') as f:
            for line in connection.iterdump():
                f.write(f'{line}\n')
        
        # 压缩备份文件
        with zipfile.ZipFile(
            f'{backup_path}.zip',
            'w',
            zipfile.ZIP_DEFLATED
        ) as zf:
            zf.write(backup_path)
        
        # 删除原始备份文件
        os.remove(backup_path)
        
        return f'{backup_path}.zip'
    
    @staticmethod
    def restore_backup(backup_file):
        """恢复备份"""
        # 解压备份文件
        with zipfile.ZipFile(backup_file, 'r') as zf:
            zf.extractall('backups')
        
        sql_file = backup_file.replace('.zip', '')
        
        # 执行SQL恢复
        with open(sql_file, 'r') as f:
            connection.executescript(f.read())
        
        # 清理临时文件
        os.remove(sql_file)