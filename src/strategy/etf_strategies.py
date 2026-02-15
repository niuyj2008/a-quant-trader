"""
ETF定投策略 - Phase 6

支持三种定投策略:
1. 定期定额 (Dollar Cost Averaging, DCA)
2. 价值平均 (Value Averaging, VA)
3. 智能再平衡 (Smart Rebalancing)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger


class ETFDollarCostAveraging:
    """
    ETF定期定额策略 (Dollar Cost Averaging)

    每隔固定周期(周/双周/月)投入固定金额,
    平滑买入成本,无需择时
    """

    def __init__(self, frequency: str = 'weekly', invest_amount: float = 5000,
                 market: str = 'CN'):
        """
        Args:
            frequency: 'weekly'(周频), 'biweekly'(双周), 'monthly'(月频)
            invest_amount: 每次定投金额
            market: 'CN'(A股,100股整数倍) or 'US'(美股,可买任意股)
        """
        self.frequency = frequency
        self.invest_amount = invest_amount
        self.market = market

    def generate_signals(self, df: pd.DataFrame, date: str) -> List[Dict]:
        """
        生成定投信号

        Args:
            df: 历史价格数据
            date: 当前日期(字符串或Timestamp)

        Returns:
            交易信号列表
        """
        # 检查是否到定投日
        if not self._is_invest_day(date):
            return []

        # 转换日期格式以便索引
        date_ts = pd.to_datetime(date)

        # 检查日期是否在DataFrame中
        if date_ts not in df.index:
            return []

        # 固定金额买入
        current_price = df.loc[date_ts, 'close']

        # 计算可以买多少股
        if self.market == 'CN':
            # A股: 100股整数倍
            affordable_shares = int(self.invest_amount / current_price)
            shares = (affordable_shares // 100) * 100

            if shares == 0:
                # 如果金额太小买不到100股,则跳过
                return []
        else:
            # 美股: 可买任意股(向下取整)
            shares = int(self.invest_amount / current_price)

            if shares == 0:
                # 金额太小买不到1股
                return []

        return [{
            'action': 'buy',
            'shares': shares,
            'price': current_price,
            'amount': shares * current_price,
            'reason': f'{self.frequency}定投',
            'confidence': 1.0,
        }]

    def _is_invest_day(self, date: str) -> bool:
        """判断是否为定投日"""
        dt = pd.to_datetime(date)

        if self.frequency == 'weekly':
            # 每周一(0=Monday)
            return dt.weekday() == 0
        elif self.frequency == 'biweekly':
            # 每两周的周一
            return dt.weekday() == 0 and dt.isocalendar()[1] % 2 == 0
        elif self.frequency == 'monthly':
            # 每月第一个交易日(1-7号且是周一)
            return dt.day <= 7 and dt.weekday() == 0

        return False


class ETFValueAveraging:
    """
    ETF价值平均策略 (Value Averaging)

    设定目标增长率,根据实际价值与目标价值的差异调整投入:
    - 如果实际价值 < 目标价值 → 加大投入补齐
    - 如果实际价值 > 目标价值 → 减少投入甚至卖出
    """

    def __init__(self, target_growth_rate: float = 0.01, base_amount: float = 5000):
        """
        Args:
            target_growth_rate: 每期目标增长率(如0.01=1%)
            base_amount: 基础投入金额
        """
        self.target_growth_rate = target_growth_rate
        self.base_amount = base_amount
        self.target_value = 0
        self.periods = 0

    def generate_signals(self, df: pd.DataFrame, date: str, current_value: float) -> List[Dict]:
        """
        根据目标价值与实际价值差异调整投入

        Args:
            df: 历史价格数据
            date: 当前日期(字符串或Timestamp)
            current_value: 当前持仓市值

        Returns:
            交易信号列表
        """
        self.periods += 1

        # 计算目标价值(复利增长 + 新投入)
        self.target_value = self.target_value * (1 + self.target_growth_rate) + self.base_amount

        # 计算需要投入金额
        gap = self.target_value - current_value

        # 转换日期格式
        date_ts = pd.to_datetime(date)

        # 检查日期是否在DataFrame中
        if date_ts not in df.index:
            return []

        current_price = df.loc[date_ts, 'close']

        if gap > 100:  # 差额>100元才操作
            # 需要买入
            affordable_shares = int(gap / current_price)
            shares = (affordable_shares // 100) * 100  # 向下取整到100股整数倍

            if shares > 0:
                return [{
                    'action': 'buy',
                    'shares': shares,
                    'price': current_price,
                    'amount': shares * current_price,
                    'reason': f'价值平均补仓(差额{gap:.0f}元)',
                    'confidence': 1.0,
                }]

        elif gap < -10000:  # 超额盈利>1万,部分卖出
            sellable_shares = int(abs(gap) / current_price)
            shares_to_sell = (sellable_shares // 100) * 100  # 向下取整到100股整数倍

            if shares_to_sell > 0:
                return [{
                    'action': 'sell',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'amount': shares_to_sell * current_price,
                    'reason': f'价值平均减仓(超额{abs(gap):.0f}元)',
                    'confidence': 1.0,
                }]

        return []


class ETFSmartRebalancing:
    """
    ETF智能再平衡策略

    维护多个ETF的目标权重,当实际权重偏离目标超过阈值时,
    通过买卖调整回目标权重
    """

    def __init__(self, etf_weights: Dict[str, float],
                 rebalance_frequency: str = 'quarterly',
                 deviation_threshold: float = 0.05):
        """
        Args:
            etf_weights: ETF目标权重 {'510300': 0.4, 'QQQ': 0.3, 'TLT': 0.3}
            rebalance_frequency: 再平衡频率 'monthly', 'quarterly', 'semiannual', 'annual'
            deviation_threshold: 偏差阈值(如0.05=5%),超过时触发再平衡
        """
        self.etf_weights = etf_weights
        self.rebalance_frequency = rebalance_frequency
        self.deviation_threshold = deviation_threshold

        # 验证权重和为1
        total_weight = sum(etf_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"ETF权重之和必须为1,当前为{total_weight}")

    def generate_rebalance_plan(self, current_holdings: Dict[str, int],
                                total_value: float,
                                prices: Dict[str, float]) -> List[Dict]:
        """
        生成再平衡计划

        Args:
            current_holdings: 当前持仓 {code: shares}
            total_value: 总市值
            prices: 当前价格 {code: price}

        Returns:
            调仓指令列表
        """
        orders = []

        for code, target_weight in self.etf_weights.items():
            # 目标市值
            target_value = total_value * target_weight

            # 当前市值
            current_shares = current_holdings.get(code, 0)
            current_value = current_shares * prices[code]

            # 计算偏差
            deviation = (current_value - target_value) / total_value

            # 偏差超过阈值时调仓
            if abs(deviation) > self.deviation_threshold:
                gap_value = target_value - current_value
                tradable_shares = int(abs(gap_value) / prices[code])
                gap_shares = (tradable_shares // 100) * 100  # 向下取整到100股整数倍

                if gap_value > 0 and gap_shares > 0:
                    orders.append({
                        'code': code,
                        'action': 'buy',
                        'shares': gap_shares,
                        'price': prices[code],
                        'amount': gap_shares * prices[code],
                        'reason': f'再平衡买入(偏差{deviation:.1%})',
                        'target_weight': target_weight,
                        'current_weight': current_value / total_value,
                    })
                elif gap_value < 0 and gap_shares > 0:
                    orders.append({
                        'code': code,
                        'action': 'sell',
                        'shares': gap_shares,
                        'price': prices[code],
                        'amount': gap_shares * prices[code],
                        'reason': f'再平衡卖出(偏差{deviation:.1%})',
                        'target_weight': target_weight,
                        'current_weight': current_value / total_value,
                    })

        return orders

    def is_rebalance_day(self, date: str, last_rebalance_date: Optional[str] = None) -> bool:
        """
        判断是否到再平衡日

        Args:
            date: 当前日期
            last_rebalance_date: 上次再平衡日期

        Returns:
            是否应该再平衡
        """
        dt = pd.to_datetime(date)

        if last_rebalance_date is None:
            # 第一次,直接再平衡
            return True

        last_dt = pd.to_datetime(last_rebalance_date)

        if self.rebalance_frequency == 'monthly':
            # 每月1号
            return dt.month != last_dt.month and dt.day <= 7
        elif self.rebalance_frequency == 'quarterly':
            # 每季度首月1号
            return dt.month in [1, 4, 7, 10] and dt.month != last_dt.month and dt.day <= 7
        elif self.rebalance_frequency == 'semiannual':
            # 每半年(1月、7月)
            return dt.month in [1, 7] and dt.month != last_dt.month and dt.day <= 7
        elif self.rebalance_frequency == 'annual':
            # 每年1月
            return dt.month == 1 and dt.month != last_dt.month and dt.day <= 7

        return False
