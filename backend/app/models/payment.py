from datetime import datetime
from app import db

class PaymentChannel(db.Model):
    """支付渠道模型
    用于管理不同的支付方式（如支付宝、微信支付等）及其配置信息
    """
    __tablename__ = 'payment_channels'

    id = db.Column(db.BigInteger, primary_key=True)  # 主键ID
    name = db.Column(db.String(50), nullable=False)  # 支付渠道名称，如"支付宝"
    code = db.Column(db.String(20), nullable=False, unique=True)  # 支付渠道代码，如"alipay","wechat"，唯一
    config = db.Column(db.JSON)  # 支付渠道配置信息，存储为JSON格式 （将配置信息存储在这里其实是最好的，虽然将其放置在.properties文件中也是一种选择）（TODO 搞起来费劲，暂时还是采用文件的形式）
    is_enabled = db.Column(db.Boolean, default=True)  # 是否启用该支付渠道
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # 创建时间
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)  # 更新时间

class PaymentOrder(db.Model):
    """支付订单模型
    记录用户的支付订单信息，包括支付金额、状态等
    """
    __tablename__ = 'payment_orders'

    id = db.Column(db.BigInteger, primary_key=True)  # 主键ID
    user_id = db.Column(db.BigInteger, db.ForeignKey('users.id'), nullable=False)  # 用户ID，关联用户表
    channel_id = db.Column(db.BigInteger, db.ForeignKey('payment_channels.id'), nullable=False)  # 支付渠道ID
    pay_order_no = db.Column(db.String(64), unique=True, nullable=False)
    outer_order_no = db.Column(db.String(64), db.ForeignKey('gold_transactions.order_no'), unique=True, nullable=False)  # 订单号,是outer_order_no，系统生成的唯一标识----商户的订单编号
    channel_order_no = db.Column(db.String(128))  # 支付渠道返回的订单号
    amount = db.Column(db.Numeric(10, 2), nullable=False)  # 支付金额，精确到分
    status = db.Column(db.String(20), nullable=False, default='待支付')  # 订单状态：待支付、已支付、已失败等
    notify_data = db.Column(db.String(255))  # 支付回调数据
    expire_time = db.Column(db.DateTime, default=datetime.utcnow)  # 订单超时时间
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # 订单创建时间
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)  # 订单更新时间

    user = db.relationship('User', backref='payment_orders', lazy=True)  # 与用户的关联
    channel = db.relationship('PaymentChannel', backref='orders', lazy=True)  # 与支付渠道的关联
    transactions = db.relationship('GoldTransaction', backref='payment_order', lazy=True)  # 与金币交易的关联

class GoldTransaction(db.Model):
    """金币交易记录模型
    记录用户金币的变动情况，包括充值、消费等
    """
    __tablename__ = 'gold_transactions'

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)  # 主键ID
    user_id = db.Column(db.BigInteger, db.ForeignKey('users.id'), nullable=False)  # 用户ID，关联用户表
    order_no = db.Column(db.String(64), unique=True, nullable=False)  # 订单号，系统生成的唯一标识----商户的订单编号
    transaction_type = db.Column(db.String(50), nullable=False)  # 交易类型：充值、消费等
    amount = db.Column(db.Integer, nullable=False)  # 交易金币数量
    order_status = db.Column(db.String(20), nullable=False, default='待支付')  # 订单状态：待支付、已支付、已失败等（必须要有的状态信息）
    reference_type = db.Column(db.String(50))  # 关联类型，如"Product"-实体商品、"Coin"-金币、"RedPacket"-红包，理论上可以直接写成Product，无论是实体或者其他的都应该是Product
    reference_id = db.Column(db.String(50)) # db.ForeignKey('payment_orders.id')) - 不应该是支付订单的ID，暂时不反向关联，因为是先有交易订单，再有支付订单的 # 关联ID，比如产品ID、金币ID、红包ID、服务ID
    # TODO 这里面使用Integer不是很合适，很多外部的ID可能不是数字，可能是很长的字符串，按某种规则生成的，即使是一个体系内的ID大多数也是这样的情况，所以这里面使用Integer或者BigNumber都不是很合适
    # TODO 虽然sqlite中插入的时候好像能够将字符串插入到Number类型的字段中（可能内部转了），但是还是不太合适，所以这里还是使用String
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # 交易创建时间
    # TODO 缺失了更新时间updated_at(0225)
    
    user = db.relationship('User', backref='gold_transactions', lazy=True)  # 与用户的关联