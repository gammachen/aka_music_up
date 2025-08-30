from flask import Blueprint, request, jsonify
from app.services.payment import RechargeService, PaymentOrder
from app.services.alipay_service import alipay_service
from app.utils.response import make_response
from app.utils.auth import token_required
import logging
import time
from datetime import datetime, timedelta


bp = Blueprint('payment', __name__, url_prefix='/api/payment')

@bp.route('/records', methods=['GET'])
@token_required
def get_recharge_records(current_user):
    '''
    获取当前用户的充值记录集合
    '''
    start_time = time.time()
    logging.info(f'开始获取用户[{current_user.id}]的充值记录，请求参数：page={request.args.get("page")}, pageSize={request.args.get("pageSize")}')
    
    try:
        # 获取分页参数
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('pageSize', 10, type=int)
        
        # 查询用户的支付订单记录 TODO 理论上查询充值记录GoldTransaction就能够满足的，这里查询交易订单会将充值订单、支付订单都捞出来，更加完备
        query = PaymentOrder.query.filter_by(user_id=current_user.id).order_by(PaymentOrder.created_at.desc())
        
        # 执行分页查询
        pagination = query.paginate(page=page, per_page=page_size)
        
        # 获取用户当前余额
        total_balance = current_user.gold_balance
        
        # 格式化支付订单记录
        items = [{
            'createTime': order.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'amount': float(order.amount),
            'paymentMethod': order.channel.code if order.channel else '未知',
            'pay_order_no': order.pay_order_no, # 本系统的订单号
            'outer_order_no':order.outer_order_no, # 外部订单号(充值订单号)
            'channel_order_no':order.channel_order_no, # 支付渠道返回的订单号
            'status': order.status
        } for order in pagination.items]
        
        response_data = {
            'items': items,
            'total': pagination.total,
            'totalBalance': float(total_balance)
        }
        
        end_time = time.time()
        execution_time = round((end_time - start_time) * 1000, 2)
        logging.info(f'用户[{current_user.id}]充值记录查询完成，执行时间：{execution_time}ms，返回记录数：{len(items)}')
        
        return make_response(data=response_data)
        
    except Exception as e:
        end_time = time.time()
        execution_time = round((end_time - start_time) * 1000, 2)
        logging.error(f'获取用户[{current_user.id}]充值记录失败，执行时间：{execution_time}ms，错误信息：{str(e)}', exc_info=True)
        return make_response(message='获取充值记录失败', code=500)
    
@bp.route('/create', methods=['POST'])
@token_required
def create_alipay_trade(current_user):
    '''
    创建充值订单、交易订单、支付订单（进行支付）
    '''
    try:
        data = request.get_json()
        amount = data.get('amount')
        reference_id = data.get('reference_id')
        reference_type = data.get('reference_type')
        payment_method = data.get('paymentMethod') # alipay
        
        # 参数验证
        if not amount:
            return make_response(code=400, message='缺少必要参数')
        
        # TODO 对payment_method进行校验，理论上要查询数据库获取可用的支付方式，这里先简单处理
        
        # 首先要生成充值订单
        order_data = RechargeService.create_recharge_order(
            user_id=current_user.id,
            reference_id=reference_id,
            reference_type=reference_type,
            amount=amount,
            channel_code=payment_method
        )
        
        # 创建支付宝支付订单
        logging.warn(f'创建交易订单，订单号：{order_data["order_no"]}, 金额：{amount}')
        
        response = alipay_service.create_trade_page_pay(
            out_trade_no=order_data['pay_order_no'], # 使用交易订单号，而不使用订单号order_no
            total_amount=str(amount),
            subject='充值金币'
        )
        logging.warn(f'支付宝支付订单创建成功，响应数据：{response}')
        # response会是一段html代码，直接返回给前端即可，是一个form表单，让前端直接执行form的表单提交
        # 参考文旦：https://opendocs.alipay.com/open/59da99d0_alipay.trade.page.pay?scene=22&pathHash=e26b497f#%E4%B8%9A%E5%8A%A1%E5%93%8D%E5%BA%94%E5%8F%82%E6%95%B0

        return make_response(data={
            'order_no': order_data['order_no'],
            'pay_url': response,
            'expire_time': (datetime.now() + timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        logging.error(f'创建支付订单失败：{str(e)}')
        return make_response(code=500, message='创建支付订单失败')

@bp.route('/notify/alipay', methods=['POST'])
def alipay_async_notify():
    """支付宝异步通知接口"""
    start_time = time.time()
    logging.info(f'收到支付宝异步通知，请求参数：{request.form.to_dict()}')
    '''
    charset=utf-8
    out_trade_no=20250219161651424b1614
    method=alipay.trade.page.pay.return
    total_amount=100.00
    sign=OIuGQLQM50ftCCo6gc%2FGnHOFnfHdbULMKmdUodAKhhim8LeUiVIyBRU1Uq3QUng2b9HbaZagxTuno%2Bv3DfAoz4K3LDMVlZTyQjb6ySlZYb6NqASDb89ruU9r%2FbuC151Ih9rQEbhsc%2Bw6xvgyiPEncYgIXwN8WiZmzVrx7J7p0XPXx%2BK1SiCV7Ml107aRgHb1%2BqDZ6FW%2B7f19hHFBreYcL7ATTnsZPLbQ8fI3znnFMBVWpj9HVUsKg84b0HTduW%2FH%2BffMOfvVJ0wbUXXedgPUWemuAI%2BmtE1WhzQYIy8YBaNWOCc2AmA0PMoJZykEtQIrLr2W6GmerGVmS5K2UaPWMA%3D%3D
    trade_no=2025021922001441780506525151
    auth_app_id=9021000144619580
    version=1.0
    app_id=9021000144619580
    sign_type=RSA2
    seller_id=2088721058496853
    timestamp=2025-02-19+16%3A18%3A10
    '''
    
    try:
        # 获取通知参数
        notify_data = request.form.to_dict()
        
        # 验证签名
        if not alipay_service.verify_async_notify(notify_data):
            logging.error('支付宝异步通知验签失败')
            return 'failure'
        
        # 验证通知类型
        if notify_data.get('trade_status') not in ['TRADE_SUCCESS', 'TRADE_FINISHED']:
            logging.info(f'支付宝交易状态不是成功，状态：{notify_data.get("trade_status")}')
            return 'success'
        
        # 处理支付成功逻辑
        RechargeService.handle_alipay_notify(notify_data)
        
        end_time = time.time()
        execution_time = round((end_time - start_time) * 1000, 2)
        logging.info(f'支付宝异步通知处理完成，执行时间：{execution_time}ms')
        
        return 'success'
        
    except Exception as e:
        end_time = time.time()
        execution_time = round((end_time - start_time) * 1000, 2)
        logging.error(f'处理支付宝异步通知失败，执行时间：{execution_time}ms，错误信息：{str(e)}', exc_info=True)
        return 'failure'

@bp.route('/return/alipay', methods=['GET'])
def alipay_sync_return():
    """支付宝同步返回接口"""
    start_time = time.time()
    logging.info(f'收到支付宝同步返回请求，请求参数：{request.args.to_dict()}')
    
    try:
        # 获取返回参数
        return_data = request.args.to_dict()
        logging.info(f'解析请求参数完成：{return_data}')
        
        ''' TODO 同步返回的参数暂时不验签
        # 验证签名
        logging.info('开始验证支付宝同步返回签名')
        verify_result = alipay_service.verify_sync_return(return_data)
        if not verify_result:
            logging.error('支付宝同步返回验签失败，返回参数：{return_data}')
            return make_response(code=400, message='支付宝同步返回验签失败')
        logging.info('支付宝同步返回验签成功')
        '''
        
        # 返回支付结果页面
        response_data = {
            'trade_no': return_data.get('trade_no'), # 支付宝的订单号
            'out_trade_no': return_data.get('out_trade_no'), # 商户订单号（pay_order_no交易单的订单号）
            'total_amount': return_data.get('total_amount'),
            'timestamp': return_data.get('timestamp')
        }
        
        end_time = time.time()
        execution_time = round((end_time - start_time) * 1000, 2)
        logging.info(f'支付宝同步返回处理成功，执行时间：{execution_time}ms，响应数据：{response_data}')
        
        return make_response(
            code=200,
            message='支付宝同步返回处理成功',
            data=response_data
        )
        
    except Exception as e:
        end_time = time.time()
        execution_time = round((end_time - start_time) * 1000, 2)
        logging.error(f'处理支付宝同步返回失败，执行时间：{execution_time}ms，请求参数：{request.args.to_dict()}，错误信息：{str(e)}', exc_info=True)
        return make_response(code=500, message='处理支付宝同步返回失败')