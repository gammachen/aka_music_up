from app import db
from app.models.reward import RewardRecord
from app.models.user import User
from app.models.topic import Topic
from app.models.payment import GoldTransaction
from datetime import datetime
import logging

class RewardService:
    @staticmethod
    def create_reward(from_user_id: int, topic_id: int, amount: int, message: str = '') -> dict:
        """
        创建打赏记录
        
        Args:
            from_user_id: 打赏用户ID
            topic_id: 主题ID
            amount: 打赏金额
            message: 打赏留言
            
        Returns:
            dict: 打赏记录信息
            
        Raises:
            ValueError: 参数验证失败
            Exception: 数据库操作失败
        """
        # 验证打赏金额
        if amount <= 0:
            logging.warning(f'打赏金额无效 - 用户ID: {from_user_id}, 金额: {amount}')
            raise ValueError('打赏金额必须大于0')

        # 获取当前用户
        from_user = User.query.get(from_user_id)
        if not from_user:
            logging.error(f'打赏用户不存在 - 用户ID: {from_user_id}')
            raise ValueError('用户不存在')

        # 检查用户金币是否足够
        if from_user.gold_balance < amount:
            logging.warning(f'用户金币不足 - 用户ID: {from_user_id}, 当前余额: {from_user.gold_balance}, 需要金币: {amount}')
            raise ValueError('金币不足')

        # 获取主题和作者信息
        topic = Topic.query.get(topic_id)
        if not topic:
            logging.error(f'打赏主题不存在 - 主题ID: {topic_id}')
            raise ValueError('主题不存在')

        to_user = User.query.get(topic.user_id)
        if not to_user:
            logging.error(f'主题作者不存在 - 作者ID: {topic.user_id}')
            raise ValueError('主题作者不存在')
        
        try:
            # 生成打赏订单号
            order_no = f'reward_{int(datetime.now().timestamp())}' # TODO 使用雪花算法生成订单号或者加上uuid
            logging.info(f'生成打赏订单 - 订单号: {order_no}')

            # 创建打赏记录
            reward = RewardRecord(
                amount=amount,
                from_user_id=from_user_id,
                to_user_id=to_user.id,
                topic_id=topic_id,
                message=message
            )
            db.session.add(reward)

            # 更新用户金币余额
            from_user.gold_balance -= amount
            to_user.gold_balance += amount
            logging.info(f'更新用户金币余额 - 打赏用户(ID: {from_user_id})余额: {from_user.gold_balance}, 接收用户(ID: {to_user.id})余额: {to_user.gold_balance}')

            # 创建打赏用户的金币交易记录（支出）
            from_transaction = GoldTransaction(
                user_id=from_user_id,
                order_no=f'{order_no}_{from_user_id}',
                transaction_type='打赏支出',
                amount=-amount,  # 负数表示支出
                order_status='已完成',
                reference_type='RewardTopic',
                reference_id=topic_id # TODO 这里应该是打赏记录的ID可能更加合适，不确定
            )
            db.session.add(from_transaction)
            db.session.flush()  # 确保第一个交易记录的主键生成完成

            # 创建被打赏用户的金币交易记录（收入）
            to_transaction = GoldTransaction(
                user_id=to_user.id,
                order_no=f'{order_no}_{to_user.id}',
                transaction_type='打赏收入',
                amount=amount,  # 正数表示收入
                order_status='已完成',
                reference_type='RewardTopic',
                reference_id=topic_id # TODO 这里应该是打赏记录的ID可能更加合适，不确定
            )
            db.session.add(to_transaction)

            db.session.commit()
            logging.info(f'打赏交易完成 - 订单号: {order_no}, 打赏ID: {reward.id}')

            return reward.to_dict()

        except Exception as e:
            db.session.rollback()
            logging.error(f'打赏交易失败 - 订单号: {order_no}, 错误信息: {str(e)}')
            raise e