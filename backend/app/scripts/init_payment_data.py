import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime, timedelta
from faker import Faker
import random
from app import create_app
from app.models.payment import db, PaymentOrder, PaymentChannel
from app.models.user import User

fake = Faker()

# 预设的金币套餐
COIN_PACKAGES = [
    {"amount": 6, "coins": 60},
    {"amount": 30, "coins": 330},
    {"amount": 68, "coins": 780},
    {"amount": 128, "coins": 1580},
    {"amount": 328, "coins": 4280},
    {"amount": 648, "coins": 8880}
]

# 支付状态
PAYMENT_STATUS = ['pending', 'success', 'failed']

def generate_order_no():
    """生成订单号"""
    now = datetime.now()
    return f"P{now.strftime('%Y%m%d%H%M%S')}{random.randint(1000, 9999)}"

def generate_channel_order_no(channel_code):
    """生成渠道订单号"""
    now = datetime.now()
    return f"{channel_code}{now.strftime('%Y%m%d%H%M%S')}{random.randint(10000, 99999)}"

def init_payment_data():
    app = create_app()
    with app.app_context():
        # 获取所有用户
        users = User.query.all()
        if not users:
            print("No users found in database")
            return
        
        # 生成30条支付订单数据
        for _ in range(30):
            # 随机选择用户
            user = random.choice(users)
            # 随机选择金币套餐
            package = random.choice(COIN_PACKAGES)
            # 随机选择支付状态，成功的概率更高
            status = random.choices(PAYMENT_STATUS, weights=[20, 70, 10])[0]
            # 生成最近30天内的随机时间
            created_at = datetime.now() - timedelta(days=random.randint(0, 30))
            
            # 获取默认支付渠道
            channel = PaymentChannel.query.first()
            if not channel:
                print("No payment channel found in database")
                return

            payment = PaymentOrder(
                order_no=generate_order_no(),
                channel_order_no=generate_channel_order_no(channel.code),
                user_id=user.id,
                channel_id=channel.id,
                amount=package['amount'],
                status='已支付' if status == 'success' else '待支付' if status == 'pending' else '已失败',
                created_at=created_at,
                updated_at=created_at
            )
            
            # 如果支付成功，设置支付完成时间和渠道订单号
            if status == 'success':
                payment.paid_at = created_at + timedelta(minutes=random.randint(1, 30))
                payment.channel_order_no = generate_channel_order_no(channel.code)
            
            db.session.add(payment)
        
        try:
            db.session.commit()
            print("Successfully initialized payment data")
        except Exception as e:
            db.session.rollback()
            print(f"Error initializing payment data: {str(e)}")

if __name__ == '__main__':
    init_payment_data()