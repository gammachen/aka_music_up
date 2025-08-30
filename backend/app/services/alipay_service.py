import os
import logging
from configparser import ConfigParser
from alipay.aop.api.AlipayClientConfig import AlipayClientConfig
from alipay.aop.api.DefaultAlipayClient import DefaultAlipayClient
from alipay.aop.api.domain.AlipayTradePagePayModel import AlipayTradePagePayModel
from alipay.aop.api.request.AlipayTradePagePayRequest import AlipayTradePagePayRequest
from alipay.aop.api.response.AlipayTradePagePayResponse import AlipayTradePagePayResponse
import alipay.aop.api.util.SignatureUtils as SignatureUtils

class AlipayService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config()
        self.client = self._init_alipay_client()

    def _load_config(self):
        """加载支付宝配置文件"""
        config = ConfigParser()
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resource', 'alipay-sandbox.properties')
        config.read(config_path, encoding='utf-8')
        return {
            'app_id': config.get('DEFAULT', 'alipay.app-id'),
            'merchant_private_key': config.get('DEFAULT', 'alipay.merchant-private-key'),
            'alipay_public_key': config.get('DEFAULT', 'alipay.alipay-public-key'),
            'notify_url': config.get('DEFAULT', 'alipay.notify-url'),
            'return_url': config.get('DEFAULT', 'alipay.return-url'),
            'gateway_url': config.get('DEFAULT', 'alipay.gateway-url')
        }

    def _init_alipay_client(self):
        """初始化支付宝客户端"""
        alipay_client_config = AlipayClientConfig()
        alipay_client_config.server_url = self.config['gateway_url']
        alipay_client_config.app_id = self.config['app_id']
        alipay_client_config.app_private_key = self.config['merchant_private_key']
        alipay_client_config.alipay_public_key = self.config['alipay_public_key']
        
        return DefaultAlipayClient(alipay_client_config, self.logger)

    def create_trade_page_pay(self, out_trade_no: str, total_amount: str, subject: str) -> str:
        """创建电脑网站支付订单"""
        model = AlipayTradePagePayModel()
        model.out_trade_no = out_trade_no
        model.total_amount = total_amount
        model.subject = subject
        model.product_code = "FAST_INSTANT_TRADE_PAY"

        request = AlipayTradePagePayRequest(biz_model=model)
        request.notify_url = self.config['notify_url']
        request.return_url = self.config['return_url']

        try:
            response = self.client.page_execute(request)
            return response
        except Exception as e:
            self.logger.error(f"创建支付订单失败: {str(e)}")
            raise e

    def verify_async_notify(self, params: dict) -> bool:
        """验证支付宝异步通知签名
        Args:
            params: 支付宝异步通知的所有参数
        Returns:
            bool: 验签结果
        """
        try:
            # 调用SDK的验签方法
            signature = params.pop('sign')
            sign_type = params.pop('sign_type', 'RSA2')
            # 使用支付宝公钥验证签名
            # return self.client.verify(params, signature, sign_type)
            
            return SignatureUtils.verify_with_rsa(public_key=self.client.__config.alipay_public_key, sign=signature, sign_type=sign_type)
        except Exception as e:
            self.logger.error(f"验证支付宝异步通知签名失败: {str(e)}")
            return False

# 创建服务实例
alipay_service = AlipayService()