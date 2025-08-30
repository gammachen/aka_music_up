# 金币系统接口与服务设计文档 V2

## 1. 充值服务

### 1.1 获取充值套餐列表

```http
GET /api/v1/coins/recharge/packages
```

**响应数据：**
```json
{
    "packages": [
        {
            "id": 1,
            "name": "新手礼包",
            "coins": 100,           // 金币数量
            "original_price": 10,   // 原价（元）
            "price": 8,            // 实际价格（元）
            "discount": 0.8,        // 折扣
            "tag": "限时特惠",      // 标签
            "first_buy_only": true  // 是否仅首充可购买
        }
    ],
    "first_buy_bonus": {
        "coins": 50,              // 首充额外赠送金币
        "valid_days": 7          // 有效期（天）
    }
}
```

### 1.2 创建充值订单

```http
POST /api/v1/coins/recharge/orders
```

**请求参数：**
```json
{
    "package_id": 1,           // 套餐ID
    "payment_type": "alipay"  // 支付方式：alipay/wechat/apple/google
}
```

**响应数据：**
```json
{
    "order_id": "202402160001",    // 订单号
    "amount": 8,                   // 支付金额（元）
    "coins": 100,                 // 充值金币数量
    "payment_info": {              // 支付信息（根据支付方式返回不同内容）
        "pay_url": "https://...",  // 支付链接
        "qr_code": "base64://..."   // 二维码图片
    }
}
```

### 1.3 查询订单状态

```http
GET /api/v1/coins/recharge/orders/{order_id}
```

**响应数据：**
```json
{
    "order_id": "202402160001",
    "status": "paid",           // 订单状态：created/paid/failed/expired
    "amount": 8,
    "coins": 100,
    "created_at": "2024-02-16 10:00:00",
    "paid_at": "2024-02-16 10:01:00"
}
```

## 2. VIP服务

### 2.1 获取VIP等级配置

```http
GET /api/v1/vip/levels
```

**响应数据：**
```json
{
    "levels": [
        {
            "level": 1,
            "name": "青铜会员",
            "icon": "http://...",
            "monthly_price": 10,      // 包月价格（元）
            "yearly_price": 100,      // 包年价格（元）
            "benefits": [             // 会员权益
                {
                    "type": "daily_coins",      // 每日金币
                    "value": 10
                },
                {
                    "type": "reward_discount",  // 打赏折扣
                    "value": 0.9
                }
            ]
        }
    ],
    "current_level": {              // 当前VIP信息
        "level": 1,
        "expire_time": "2024-03-16 00:00:00",
        "auto_renew": true
    }
}
```

### 2.2 开通/续费VIP

```http
POST /api/v1/vip/subscribe
```

**请求参数：**
```json
{
    "level": 1,                // VIP等级
    "duration": "monthly",     // 购买时长：monthly/yearly
    "payment_type": "alipay", // 支付方式
    "auto_renew": true       // 是否自动续费
}
```

**响应数据：**
```json
{
    "order_id": "202402160002",
    "amount": 10,
    "payment_info": {
        "pay_url": "https://...",
        "qr_code": "base64://..."
    }
}
```

### 2.3 取消自动续费

```http
POST /api/v1/vip/cancel-renew
```

**响应数据：**
```json
{
    "success": true,
    "expire_time": "2024-03-16 00:00:00"
}
```

## 3. 服务实现

### 3.1 充值服务

```python
class RechargeService:
    def create_order(self, user_id: int, package_id: int, payment_type: str) -> dict:
        """创建充值订单
        
        Args:
            user_id: 用户ID
            package_id: 套餐ID
            payment_type: 支付方式
            
        Returns:
            dict: 订单信息
        """
        with transaction.atomic():
            # 1. 获取套餐信息
            package = RechargePackage.objects.get(id=package_id)
            
            # 2. 校验首充限制
            if package.first_buy_only:
                if Order.objects.filter(
                    user_id=user_id,
                    status='paid'
                ).exists():
                    raise BusinessError('该套餐仅限首充用户购买')
            
            # 3. 创建订单
            order = Order.objects.create(
                user_id=user_id,
                package_id=package_id,
                order_id=self.generate_order_id(),
                amount=package.price,
                coins=package.coins,
                payment_type=payment_type
            )
            
            # 4. 调用支付网关
            payment_service = PaymentService()
            payment_info = payment_service.create_payment(
                order_id=order.order_id,
                amount=order.amount,
                type=payment_type
            )
            
            return {
                'order_id': order.order_id,
                'amount': order.amount,
                'coins': order.coins,
                'payment_info': payment_info
            }
            
    def process_payment_callback(self, order_id: str, trade_no: str) -> bool:
        """处理支付回调
        
        Args:
            order_id: 订单号
            trade_no: 支付流水号
            
        Returns:
            bool: 处理是否成功
        """
        with transaction.atomic():
            # 1. 查询并锁定订单
            order = Order.objects.select_for_update().get(
                order_id=order_id
            )
            
            # 2. 校验订单状态
            if order.status != 'created':
                return False
                
            # 3. 更新订单状态
            order.status = 'paid'
            order.trade_no = trade_no
            order.paid_at = timezone.now()
            order.save()
            
            # 4. 发放金币
            coin_service = CoinService()
            coin_service.change_coins(
                user_id=order.user_id,
                amount=order.coins,
                type='recharge',
                related_id=order.id
            )
            
            # 5. 发放首充奖励
            if self.is_first_recharge(order.user_id):
                bonus = FirstRechargeBonus.objects.first()
                if bonus:
                    coin_service.change_coins(
                        user_id=order.user_id,
                        amount=bonus.coins,
                        type='first_recharge_bonus'
                    )
            
            return True
```

### 3.2 VIP服务

```python
class VipService:
    def subscribe(self, user_id: int, level: int, 
                 duration: str, auto_renew: bool) -> dict:
        """开通/续费VIP
        
        Args:
            user_id: 用户ID
            level: VIP等级
            duration: 购买时长
            auto_renew: 是否自动续费
            
        Returns:
            dict: 订单信息
        """
        with transaction.atomic():
            # 1. 获取VIP配置
            vip_config = VipLevel.objects.get(level=level)
            
            # 2. 计算价格和有效期
            if duration == 'monthly':
                amount = vip_config.monthly_price
                months = 1
            else:
                amount = vip_config.yearly_price
                months = 12
                
            # 3. 创建订单
            order = VipOrder.objects.create(
                user_id=user_id,
                level=level,
                duration=duration,
                amount=amount,
                auto_renew=auto_renew
            )
            
            # 4. 更新会员信息
            expire_time = self.calculate_expire_time(user_id, months)
            VipInfo.objects.update_or_create(
                user_id=user_id,
                defaults={
                    'level': level,
                    'expire_time': expire_time,
                    'auto_renew': auto_renew
                }
            )
            
            return {
                'order_id': order.order_id,
                'amount': amount,
                'expire_time': expire_time
            }
    
    def process_daily_benefits(self, user_id: int):
        """处理每日会员权益
        
        Args:
            user_id: 用户ID
        """
        vip_info = VipInfo.objects.get(user_id=user_id)
        if not vip_info.is_valid():
            return
            
        # 发放每日金币
        vip_config = VipLevel.objects.get(level=vip_info.level)
        daily_coins = vip_config.get_benefit('daily_coins')
        
        coin_service = CoinService()
        coin_service.change_coins(
            user_id=user_id,
            amount=daily_coins,
            type='vip_daily_coins'
        )
```