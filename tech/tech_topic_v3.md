# 音乐站点主题功能技术方案 V3

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
```

### 2.2 点赞模型

```python
class Like:
    """点赞模型"""
    def __init__(self):
        self.id = None
        self.user_id = None
        self.target_id = None
        self.target_type = None  # topic/reply
        self.created_at = None
        
    @classmethod
    def create(cls, user_id, target_id, target_type):
        """创建点赞"""
        with redis_lock(f'like:{target_type}:{target_id}'):
            # 检查是否已点赞
            if cls.exists(user_id, target_id, target_type):
                return False
                
            # 创建点赞记录
            like = cls()
            like.user_id = user_id
            like.target_id = target_id
            like.target_type = target_type
            like.created_at = datetime.now()
            db.session.add(like)
            
            # 更新计数
            if target_type == 'topic':
                Topic.increment_counter(target_id, 'like_count')
            else:
                Reply.increment_counter(target_id, 'like_count')
                
            # 发送通知
            NotificationManager.send_like_notification(like)
            
            db.session.commit()
            return True
```

### 2.3 收藏模型

```python
class Favorite:
    """收藏模型"""
    def __init__(self):
        self.id = None
        self.user_id = None
        self.topic_id = None
        self.created_at = None
        
    @classmethod
    def create(cls, user_id, topic_id):
        """创建收藏"""
        with redis_lock(f'favorite:topic:{topic_id}'):
            # 检查是否已收藏
            if cls.exists(user_id, topic_id):
                return False
                
            # 创建收藏记录
            favorite = cls()
            favorite.user_id = user_id
            favorite.topic_id = topic_id
            favorite.created_at = datetime.now()
            db.session.add(favorite)
            
            # 更新计数
            Topic.increment_counter(topic_id, 'favorite_count')
            
            # 发送通知
            NotificationManager.send_favorite_notification(favorite)
            
            db.session.commit()
            return True
```

### 2.4 通知模型

```python
class Notification:
    """通知模型"""
    def __init__(self):
        self.id = None
        self.user_id = None
        self.sender_id = None
        self.type = None  # like/favorite/reply
        self.target_id = None
        self.target_type = None
        self.content = ""
        self.is_read = False
        self.created_at = None
```

### 2.5 回复模型

```python
class Reply:
    """回复模型"""
    def __init__(self):
        self.id = None
        self.topic_id = None
        self.user_id = None
        self.content = ""
        self.floor = 0  # 楼层号
        self.like_count = 0
        self.status = "normal"
        self.created_at = None
        self.updated_at = None
        
    @classmethod
    def create(cls, topic_id, user_id, content):
        """创建回复"""
        with redis_lock(f'reply:topic:{topic_id}'):
            # 内容检查
            if not content or len(content) > 10000:
                raise ValueError('回复内容长度不合法')
                
            # 计算楼层号
            floor = Topic.increment_counter(topic_id, 'reply_count')
            
            # 创建回复
            reply = cls()
            reply.topic_id = topic_id
            reply.user_id = user_id
            reply.content = content
            reply.floor = floor
            reply.created_at = datetime.now()
            reply.updated_at = reply.created_at
            db.session.add(reply)
            
            # 发送通知
            NotificationManager.send_reply_notification(reply)
            
            db.session.commit()
            return reply
```

## 5. 回复模块设计

### 5.1 楼层计算

使用Redis原子操作确保楼层号的唯一性和连续性：

```python
class FloorManager:
    """楼层管理器"""
    def __init__(self):
        self.redis = Redis()
        
    def get_next_floor(self, topic_id):
        """获取下一个楼层号"""
        key = f'topic:{topic_id}:floor'
        return self.redis.incr(key)
        
    def get_floor_range(self, topic_id, start, end):
        """获取指定范围的楼层"""
        return Reply.query\
            .filter_by(topic_id=topic_id)\
            .filter(Reply.floor.between(start, end))\
            .order_by(Reply.floor.asc())\
            .all()
```

### 5.2 回复缓存策略

采用分页缓存策略，减少数据库访问：

```python
class ReplyCacheManager:
    """回复缓存管理器"""
    def __init__(self):
        self.redis = Redis()
        self.page_size = 20
        self.expire = 3600  # 1小时过期
        
    def get_page(self, topic_id, page):
        """获取指定页的回复"""
        cache_key = f'topic:{topic_id}:replies:page:{page}'
        data = self.redis.get(cache_key)
        
        if not data:
            # 从数据库获取
            start = (page - 1) * self.page_size
            replies = Reply.query\
                .filter_by(topic_id=topic_id)\
                .order_by(Reply.floor.asc())\
                .offset(start)\
                .limit(self.page_size)\
                .all()
                
            data = [reply.to_dict() for reply in replies]
            # 写入缓存
            self.redis.setex(
                cache_key,
                self.expire,
                json.dumps(data)
            )
        else:
            data = json.loads(data)
            
        return data
        
    def invalidate_cache(self, topic_id):
        """使指定主题的回复缓存失效"""
        pattern = f'topic:{topic_id}:replies:page:*'
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
```

### 5.3 回复相关接口

```python
@app.route('/api/topics/<int:topic_id>/replies', methods=['POST'])
@login_required
def create_reply(topic_id):
    """创建回复"""
    content = request.json.get('content')
    reply = Reply.create(
        topic_id=topic_id,
        user_id=current_user.id,
        content=content
    )
    return jsonify(reply.to_dict())
    
@app.route('/api/topics/<int:topic_id>/replies', methods=['GET'])
def get_topic_replies(topic_id):
    """获取主题回复列表"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    # 从缓存获取
    cache = ReplyCacheManager()
    replies = cache.get_page(topic_id, page)
    
    # 获取总数
    total = Topic.query.get(topic_id).reply_count
    pages = (total + per_page - 1) // per_page
    
    return jsonify({
        'items': replies,
        'total': total,
        'pages': pages,
        'current_page': page
    })
    
@app.route('/api/replies/<int:reply_id>', methods=['DELETE'])
@login_required
def delete_reply(reply_id):
    """删除回复"""
    reply = Reply.query.get_or_404(reply_id)
    
    # 权限检查
    if reply.user_id != current_user.id and not current_user.is_admin():
        abort(403)
        
    # 软删除
    reply.status = 'deleted'
    db.session.commit()
    
    # 清除缓存
    ReplyCacheManager().invalidate_cache(reply.topic_id)
    
    return jsonify({'success': True})
```

### 3.3 缓存更新策略

采用写直达(Write-Through)和延迟双删策略确保缓存一致性：

```python
class CacheManager:
    """缓存管理器"""
    def __init__(self):
        self.redis = Redis()
        self.delay = 0.5  # 延迟删除等待时间(秒)
        
    def get_topic_stats(self, topic_id):
        """获取主题统计信息"""
        cache_key = f'topic:stats:{topic_id}'
        data = self.redis.hgetall(cache_key)
        
        if not data:
            # 从数据库获取
            topic = Topic.query.get(topic_id)
            if topic:
                data = {
                    'view_count': topic.view_count,
                    'like_count': topic.like_count,
                    'favorite_count': topic.favorite_count,
                    'reply_count': topic.reply_count
                }
                # 写入缓存
                self.redis.hmset(cache_key, data)
                self.redis.expire(cache_key, 3600)  # 1小时过期
                
        return data
        
    def update_topic_stats(self, topic_id, field, value):
        """更新主题统计信息"""
        cache_key = f'topic:stats:{topic_id}'
        
        # 删除缓存
        self.redis.delete(cache_key)
        
        # 更新数据库
        Topic.query.filter_by(id=topic_id)\
            .update({field: value})
        db.session.commit()
        
        # 延迟双删
        def delay_delete():
            time.sleep(self.delay)
            self.redis.delete(cache_key)
            
        threading.Thread(target=delay_delete).start()
```

## 4. 接口设计

### 4.1 点赞相关接口

```python
@app.route('/api/topics/<int:topic_id>/like', methods=['POST'])
@login_required
def like_topic(topic_id):
    """点赞主题"""
    # 创建点赞
    success = Like.create(
        user_id=current_user.id,
        target_id=topic_id,
        target_type='topic'
    )
    return jsonify({'success': success})
    
@app.route('/api/topics/<int:topic_id>/like', methods=['DELETE'])
@login_required
def unlike_topic(topic_id):
    """取消点赞"""
    # 删除点赞
    success = Like.delete(
        user_id=current_user.id,
        target_id=topic_id,
        target_type='topic'
    )
    return jsonify({'success': success})
```

### 4.2 收藏相关接口

```python
@app.route('/api/topics/<int:topic_id>/favorite', methods=['POST'])
@login_required
def favorite_topic(topic_id):
    """收藏主题"""
    # 创建收藏
    success = Favorite.create(
        user_id=current_user.id,
        topic_id=topic_id
    )
    return jsonify({'success': success})
    
@app.route('/api/topics/<int:topic_id>/favorite', methods=['DELETE'])
@login_required
def unfavorite_topic(topic_id):
    """取消收藏"""
    # 删除收藏
    success = Favorite.delete(
        user_id=current_user.id,
        topic_id=topic_id
    )
    return jsonify({'success': success})
```

### 4.3 通知相关接口

```python
@app.route('/api/notifications', methods=['GET'])
@login_required
def get_notifications():
    """获取通知列表"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    # 分页查询
    pagination = Notification.query\
        .filter_by(user_id=current_user.id)\
        .order_by(Notification.created_at.desc())\
        .paginate(page=page, per_page=per_page)
        
    return jsonify({
        'items': [item.to_dict() for item in pagination.items],
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': pagination.page
    })
    
@app.route('/api/notifications/unread/count', methods=['GET'])
@login_required
def get_unread_notification_count():
    """获取未读通知数"""
    count = NotificationManager().get_unread_count(current_user.id)
    return jsonify({'count': count})
    
@app.route('/api/notifications/read', methods=['POST'])
@login_required
def mark_notifications_as_read():
    """标记通知为已读"""
    notification_ids = request.json.get('notification_ids', [])
    NotificationManager().mark_as_read(current_user.id, notification_ids)
    return jsonify({'success': True})