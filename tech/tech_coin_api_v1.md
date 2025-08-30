# 金币系统接口与服务设计文档

## 1. 通用规范

### 1.1 接口响应格式
```json
{
    "code": 0,           // 错误码，0表示成功
    "message": "成功",  // 错误信息
    "data": {}         // 响应数据
}
```

### 1.2 通用错误码
```
0     - 成功
1001  - 参数错误
1002  - 未登录或登录已过期
1003  - 权限不足
1004  - 资源不存在
1005  - 操作频率超限
2001  - 金币余额不足
2002  - 红包已抢完
2003  - 红包已过期
2004  - 已达到每日免费打赏次数上限
```

## 2. 金币变更服务

### 2.1 获取用户金币余额

```http
GET /api/v1/coins/balance
```

**响应数据：**
```json
{
    "balance": 100,            // 当前余额
    "today_earned": 10,       // 今日获得金币
    "today_spent": 5,         // 今日消费金币
    "frozen_coins": 0         // 冻结金币数量
}
```

### 2.2 获取金币交易记录

```http
GET /api/v1/coins/transactions
```

**请求参数：**
```json
{
    "page": 1,               // 页码，从1开始
    "page_size": 20,        // 每页记录数，默认20
    "type": "all",          // 交易类型：all-全部/income-收入/expense-支出
    "start_date": "2024-02-15",  // 开始日期
    "end_date": "2024-02-15"    // 结束日期
}
```

**响应数据：**
```json
{
    "total": 100,           // 总记录数
    "page": 1,             // 当前页码
    "page_size": 20,       // 每页记录数
    "items": [
        {
            "id": 1,
            "amount": 10,    // 交易金额，正数表示收入，负数表示支出
            "balance": 100,  // 交易后余额
            "type": "sign_in",  // 交易类型
            "description": "每日签到奖励",
            "created_at": "2024-02-15 10:00:00"
        }
    ]
}
```

## 3. 签到服务

### 3.1 获取签到信息

```http
GET /api/v1/coins/sign-in/info
```

**响应数据：**
```json
{
    "today_signed": true,         // 今日是否已签到
    "continuous_days": 6,        // 当前连续签到天数
    "total_signed_days": 30,     // 总签到天数
    "next_reward": {             // 下次签到可获得的奖励
        "base_coins": 1,         // 基础奖励
        "extra_coins": 2,        // 额外奖励（连续签到）
        "total_coins": 3         // 总奖励
    }
}
```

### 3.2 执行签到

```http
POST /api/v1/coins/sign-in
```

**响应数据：**
```json
{
    "success": true,
    "reward_coins": 3,           // 获得的金币数量
    "continuous_days": 7,        // 更新后的连续签到天数
    "next_extra_reward": {       // 下次额外奖励信息
        "days": 30,             // 达到天数
        "coins": 10             // 奖励金币
    }
}
```

## 4. 红包服务

### 4.1 发红包

```http
POST /api/v1/coins/red-packets
```

**请求参数：**
```json
{
    "total_amount": 100,        // 红包总金额
    "packet_count": 10,        // 红包个数
    "packet_type": "random",   // 红包类型：random-随机/fixed-固定
    "valid_hours": 24,        // 有效期小时数，默认24
    "message": "恭喜发财"      // 红包祝福语
}
```

**响应数据：**
```json
{
    "id": 1,                   // 红包ID
    "total_amount": 100,       // 红包总金额
    "remaining_amount": 100,   // 剩余金额
    "packet_count": 10,       // 红包总个数
    "remaining_count": 10,    // 剩余个数
    "expire_time": "2024-02-16 10:00:00"  // 过期时间
}
```

### 4.2 抢红包

```http
POST /api/v1/coins/red-packets/{id}/grab
```

**响应数据：**
```json
{
    "success": true,
    "amount": 10,              // 抢到的金币数量
    "message": "恭喜发财",     // 红包祝福语
    "packet_info": {          // 红包信息
        "remaining_amount": 90,  // 剩余金额
        "remaining_count": 9     // 剩余个数
    }
}
```

### 4.3 获取红包详情

```http
GET /api/v1/coins/red-packets/{id}
```

**响应数据：**
```json
{
    "id": 1,
    "user": {                  // 发红包用户信息
        "id": 100,
        "nickname": "张三",
        "avatar": "http://..."
    },
    "total_amount": 100,
    "remaining_amount": 90,
    "packet_count": 10,
    "remaining_count": 9,
    "packet_type": "random",
    "status": "active",        // 状态：active-进行中/expired-已过期/finished-已抢完
    "message": "恭喜发财",
    "expire_time": "2024-02-16 10:00:00",
    "records": [              // 抢红包记录
        {
            "user": {         // 抢红包用户信息
                "id": 101,
                "nickname": "李四",
                "avatar": "http://..."
            },
            "amount": 10,
            "created_at": "2024-02-15 10:00:00"
        }
    ]
}
```

## 5. 打赏服务

### 5.1 获取打赏配置

```http
GET /api/v1/coins/reward/config
```

**响应数据：**
```json
{
    "daily_free_count": 10,     // 每日免费打赏次数
    "remaining_free_count": 5,  // 今日剩余免费次数
    "min_amount": 1,           // 最小打赏金额
    "max_amount": 100,         // 最大打赏金额
    "suggested_amounts": [1, 5, 10, 20, 50, 100]  // 建议打赏金额
}
```

### 5.2 执行打赏

```http
POST /api/v1/coins/reward
```

**请求参数：**
```json
{
    "target_type": "topic",    // 打赏目标类型：topic-主题/comment-评论
    "target_id": 100,         // 打赏目标ID
    "amount": 10,             // 打赏金额
    "message": "感谢分享"      // 打赏留言
}
```

**响应数据：**
```json
{
    "success": true,
    "reward_id": 1,            // 打赏记录ID
    "remaining_free_count": 4,  // 更新后的免费次数
    "cost_coins": 0            // 实际消耗金币数（免费次数内为0）
}
```

## 6. 隐藏内容服务

### 6.1 检查查看权限

```http
GET /api/v1/topics/{id}/hidden-content/check
```

**响应数据：**
```json
{
    "can_view": true,           // 是否可以查看
    "reason": "VIP用户",        // 可查看原因
    "cost_coins": 0            // 查看需要消耗的金币数（0表示免费）
}
```

### 6.2 查看隐藏内容

```http
POST /api/v1/topics/{id}/hidden-content/view
```

**响应数据：**
```json
{
    "success": true,
    "content": "隐藏内容详情",   // 隐藏内容
    "cost_coins": 1,           // 实际消耗金币数
    "author_earned": 0         // 作者获得金币数（50%概率）
}
```

## 7. 服务实现

### 7.1 金币变更服务

```python
class CoinService:
    def change_coins(self, user_id: int, amount: int, type: str, related_id: int = None) -> bool:
        """金币变更核心方法
        
        Args:
            user_id: 用户ID
            amount: 变更金额（正数增加，负数减少）
            type: 变更类型
            related_id: 关联ID
            
        Returns:
            bool: 变更是否成功
        """
        with transaction.atomic():
            # 1. 检查用户余额
            if amount < 0:
                balance = self.get_user_balance(user_id)
                if balance + amount < 0:
                    return False
                    
            # 2. 更新用户余额
            User.objects.filter(id=user_id).update(
                coin_balance=F('coin_balance') + amount
            )
            
            # 3. 记录流水
            CoinTransaction.objects.create(
                user_id=user_id,
                amount=amount,
                balance=F('coin_balance'),
                type=type,
                related_id=related_id
            )
            
            return True
```

### 7.2 签到服务

```python
class SignInService:
    def do_sign_in(self, user_id: int) -> dict:
        """执行签到
        
        Args:
            user_id: 用户ID
            
        Returns:
            dict: 签到结果
        """
        with transaction.atomic():
            # 1. 检查是否已签到
            today = date.today()
            if SignInLog.objects.filter(
                user_id=user_id,
                sign_date=today
            ).exists():
                raise BusinessError('今日已签到')
                
            # 2. 获取连续签到天数
            yesterday = today - timedelta(days=1)
            yesterday_log = SignInLog.objects.filter(
                user_id=user_id,
                sign_date=yesterday
            ).first()
            
            continuous_days = yesterday_log.continuous_days + 1 if yesterday_log else 1
            
            # 3. 计算奖励金币
            base_coins = 1
            extra_coins = 0
            if continuous_days == 7:
                extra_coins = 2
            elif continuous_days == 30:
                extra_coins = 10
                
            total_coins = base_coins + extra_coins
            
            # 4. 记录签到
            SignInLog.objects.create(
                user_id=user_id,
                sign_date=today,
                continuous_days=continuous_days,
                reward_coins=total_coins
            )
            
            # 5. 发放金币
            coin_service = CoinService()
            coin_service.change_coins(
                user_id=user_id,
                amount=total_coins,
                type='sign_in'
            )
            
            return {
                'success': True,
                'reward_coins': total_coins,
                'continuous_days': continuous_days
            }
```

### 7.3 红包服务

```python
class RedPacketService:
    def create_red_packet(self, user_id: int, total_amount: int, 
                         packet_count: int, packet_type: str) -> dict:
        """创建红包
        
        Args:
            user_id: 用户ID
            total_amount: 红包总金额
            packet_count: 红包个数
            packet_type: 红包类型(random/fixed)
            
        Returns:
            dict: