from datetime import datetime, timedelta
from uuid import uuid4
from app import db
from app.models.payment import PaymentChannel, PaymentOrder, GoldTransaction
from app.services.alipay_service import alipay_service

class RechargeService:
    @staticmethod
    def create_recharge_order(user_id: int, reference_id:int, reference_type:str, amount: float, channel_code: str) -> dict:
        """创建充值订单并生成金币交易记录
        
        Args:
            user_id: 用户ID
            reference_id: 关联ID，用于关联具体的业务实体（如商品ID、服务ID等）
            reference_type: 关联类型，如'Product'（商品）、'Service'（服务）等
            amount: 充值金额（人民币）
            channel_code: 支付渠道代码，用于指定使用的支付方式
            
        Returns:
            dict: 包含以下信息的字典：
                - order_no: 订单号
                - status: 订单状态（'待支付'）
        
        Raises:
            ValueError: 当支付渠道不可用或状态异常时
            Exception: 数据库操作异常时
        """
        # 获取支付渠道并验证
        channel = PaymentChannel.query.filter_by(code=channel_code, is_enabled=True).first()
        if not channel:
            raise ValueError('支付渠道不可用')
        
        # 计算金币数量（根据渠道汇率）（暂时没有所谓的渠道汇率rate，TODO 将来有可能会有， amount * channel.rate）
        gold_amount = int(amount * 1) # 
        
        # 生成充值订单号（RCH表示Recharge）
        order_no = f'RCH{datetime.now().strftime("%Y%m%d%H%M%S")}{uuid4().hex[:8]}'
        
        try:
            # 创建金币交易记录
            gold_transaction = GoldTransaction(
                user_id=user_id,
                order_no=order_no,
                transaction_type='充值',
                amount=gold_amount,
                order_status='待支付',  # 新增字段，默认为待支付状态
                reference_type=reference_type,
                reference_id=reference_id,
                created_at=datetime.utcnow()
            )
            
            # 生成充值交易订单号
            pay_order_no = f'{datetime.now().strftime("%Y%m%d%H%M%S")}{uuid4().hex[:8]}'
            
            # 创建支付订单
            payment_order = PaymentOrder(
                user_id=user_id,
                channel_id=channel.id,
                pay_order_no=pay_order_no,
                outer_order_no=order_no,
                amount=amount,
                status='待支付',
                # notify_data=  alipay_service.config.get('DEFAULT', 'alipay.return-url'), # 'http://localhost:5000/api/payment/notify',  # 支付回调通知地址
                expire_time=datetime.utcnow() + timedelta(minutes=300),  # 设置300分钟过期时间
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            db.session.add(gold_transaction)
            db.session.add(payment_order)
            db.session.commit()
            
            return {
                'order_no': order_no,
                'pay_order_no': pay_order_no,
                'status': '待支付'
            }
            
        except Exception as e:
            db.session.rollback()
            raise e
    
    @staticmethod
    def handle_alipay_notify(data: dict) -> bool:
        """处理支付宝回调"""
        # 查询订单（使用支付订单号查询，支付宝返回的字段内容，从前端服务器那边传递过来的）
        pay_order = PaymentOrder.query.filter_by(pay_order_no=data['out_trade_no']).first()
        if not pay_order:
            return False
            
        # 验证订单状态
        if pay_order.status != '待支付':
            return True  # 已处理过的订单直接返回成功
            
        # 验证金额
        if float(data['total_amount']) != float(pay_order.amount):
            return False
            
        try:
            # 更新订单状态
            pay_order.status = '已支付'
            # 将支付宝的订单编号赋值到支付订单中！
            pay_order.channel_order_no = data['trade_no']
            
            # 更新对应的充值订单
            gold_transaction = GoldTransaction.query.filter_by(order_no=pay_order.outer_order_no).first()
            gold_transaction.order_status = '已支付'
            
            db.session.add(pay_order)
            db.session.add(gold_transaction)
            db.session.commit()
            return True
            
        except Exception as e:
            db.session.rollback()
            raise e
    
    @staticmethod
    def query_pay_order_status(order_no: str) -> dict:
        """查询订单状态"""
        order = PaymentOrder.query.filter_by(order_no=order_no).first()
        if not order:
            raise ValueError('订单不存在')
            
        return {
            'order_no': order.order_no,
            'status': order.status,
            'amount': float(order.amount),
            'created_at': order.created_at.isoformat(),
            'expire_time': order.expire_time.isoformat()
        }
    
    
    
    
    
    