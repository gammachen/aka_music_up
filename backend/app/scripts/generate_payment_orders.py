from datetime import datetime, timedelta
from random import randint, choice, uniform
from app import db
from app.models.payment import PaymentOrder, PaymentChannel
from app.models.user import User

def generate_order_no():
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    random_num = str(randint(1000, 9999))
    return f'ORDER{timestamp}{random_num}'

def generate_payment_orders():
    # 获取现有用户和支付渠道
    users = User.query.all()
    channels = PaymentChannel.query.all()
    
    if not users or not channels:
        print("请确保数据库中已存在用户和支付渠道数据")
        return
    
    # 支付状态列表
    status_list = ['待支付', '支付成功', '支付失败', '已过期']
    # 金币套餐列表 (金额, 金币数)
    gold_packages = [
        (6.00, 60),
        (30.00, 300),
        (68.00, 680),
        (128.00, 1280),
        (328.00, 3280),
        (648.00, 6480)
    ]
    
    # 生成30条订单数据
    for _ in range(30):
        # 随机选择用户和支付渠道
        user = choice(users)
        channel = choice(channels)
        
        # 随机选择金币套餐
        amount, gold_amount = choice(gold_packages)
        
        # 随机生成订单时间（最近30天内）
        days_ago = randint(0, 29)
        hours_ago = randint(0, 23)
        minutes_ago = randint(0, 59)
        created_time = datetime.utcnow() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
        
        # 设置过期时间（创建时间后2小时）
        expire_time = created_time + timedelta(hours=2)
        
        # 随机选择订单状态
        status = choice(status_list)
        
        # 如果订单已支付成功，生成第三方订单号
        third_party_no = f'THIRD{randint(100000, 999999)}' if status == '支付成功' else None
        
        # 创建订单对象
        order = PaymentOrder(
            user_id=user.id,
            channel_id=channel.id,
            order_no=generate_order_no(),
            third_party_no=third_party_no,
            amount=amount,
            gold_amount=gold_amount,
            status=status,
            notify_url='http://localhost:5000/api/payment/notify',
            expire_time=expire_time,
            created_at=created_time,
            updated_at=created_time
        )
        
        # 如果订单状态不是待支付，更新updated_at时间
        if status != '待支付':
            order.updated_at = expire_time if status == '已过期' else \
                             created_time + timedelta(minutes=randint(5, 30))
        
        db.session.add(order)
    
    try:
        db.session.commit()
        print("成功生成30条支付订单数据")
    except Exception as e:
        db.session.rollback()
        print(f"生成订单数据失败: {str(e)}")

if __name__ == '__main__':
    generate_payment_orders()