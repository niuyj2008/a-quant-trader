"""
策略集成(Ensemble) - Phase 9.3

组合多个策略降低风险,提高稳定性
支持投票法、加权法、动态加权法
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from loguru import logger
from collections import Counter


class EnsembleStrategy:
    """
    策略集成框架

    支持的集成方法:
    1. 投票法(Voting): 多数策略同意才发出信号
    2. 加权法(Weighted): 根据历史表现加权
    3. 动态加权法(Dynamic): 根据近期表现动态调整权重
    """

    def __init__(self, strategies: List[Any], method: str = 'voting',
                 weights: Optional[List[float]] = None):
        """
        Args:
            strategies: 子策略列表
            method: 集成方法('voting', 'weighted', 'dynamic')
            weights: 各策略权重(None=等权)
        """
        self.strategies = strategies
        self.method = method

        # 权重初始化
        if weights is None:
            self.weights = [1.0 / len(strategies)] * len(strategies)
        else:
            if len(weights) != len(strategies):
                raise ValueError("weights长度必须等于strategies长度")
            # 归一化权重
            total = sum(weights)
            self.weights = [w / total for w in weights]

        # 历史表现记录(用于动态加权)
        self.strategy_performance = {i: [] for i in range(len(strategies))}

        logger.info(f"策略集成初始化: {len(strategies)}个子策略, 方法={method}")

    def generate_signals(self, df: pd.DataFrame, date: str,
                        context: Optional[Dict] = None) -> List[Dict]:
        """
        生成集成信号

        Args:
            df: 历史数据
            date: 当前日期
            context: 上下文信息(如当前持仓)

        Returns:
            信号列表
        """
        if self.method == 'voting':
            return self.generate_signals_voting(df, date, context)
        elif self.method == 'weighted':
            return self.generate_signals_weighted(df, date, context)
        elif self.method == 'dynamic':
            return self.generate_signals_dynamic(df, date, context)
        else:
            raise ValueError(f"未知的集成方法: {self.method}")

    def generate_signals_voting(self, df: pd.DataFrame, date: str,
                               context: Optional[Dict] = None) -> List[Dict]:
        """
        投票法: 多数策略同意才发出信号

        规则:
        - 买入: 超过半数策略推荐买入
        - 卖出: 超过半数策略推荐卖出
        - 持有: 否则
        """
        buy_votes = 0
        sell_votes = 0
        hold_votes = 0

        voting_details = []

        for i, strategy in enumerate(self.strategies):
            signals = strategy.generate_signals(df, date, context)

            if not signals:
                hold_votes += 1
                voting_details.append({
                    'strategy': strategy.__class__.__name__,
                    'vote': 'hold',
                })
                continue

            # 汇总信号(一个策略可能产生多个信号)
            action_counter = Counter([sig['action'] for sig in signals])

            if action_counter.get('buy', 0) > action_counter.get('sell', 0):
                buy_votes += 1
                voting_details.append({
                    'strategy': strategy.__class__.__name__,
                    'vote': 'buy',
                    'reason': signals[0].get('reason', ''),
                })
            elif action_counter.get('sell', 0) > 0:
                sell_votes += 1
                voting_details.append({
                    'strategy': strategy.__class__.__name__,
                    'vote': 'sell',
                    'reason': signals[0].get('reason', ''),
                })
            else:
                hold_votes += 1
                voting_details.append({
                    'strategy': strategy.__class__.__name__,
                    'vote': 'hold',
                })

        # 决策规则
        total_strategies = len(self.strategies)
        threshold = total_strategies / 2

        result_signals = []

        if buy_votes > threshold:
            result_signals.append({
                'action': 'buy',
                'reason': f'投票法: {buy_votes}/{total_strategies}个策略推荐买入',
                'confidence': buy_votes / total_strategies,
                'voting_details': voting_details,
                'method': 'voting',
            })
        elif sell_votes > threshold:
            result_signals.append({
                'action': 'sell',
                'reason': f'投票法: {sell_votes}/{total_strategies}个策略推荐卖出',
                'confidence': sell_votes / total_strategies,
                'voting_details': voting_details,
                'method': 'voting',
            })

        return result_signals

    def generate_signals_weighted(self, df: pd.DataFrame, date: str,
                                  context: Optional[Dict] = None) -> List[Dict]:
        """
        加权法: 根据权重加权策略信号

        计算加权得分:
        - 买入信号: +1 × weight × confidence
        - 卖出信号: -1 × weight × confidence
        - 持有信号: 0
        """
        weighted_score = 0.0
        signal_details = []

        for i, (strategy, weight) in enumerate(zip(self.strategies, self.weights)):
            signals = strategy.generate_signals(df, date, context)

            if not signals:
                signal_details.append({
                    'strategy': strategy.__class__.__name__,
                    'weight': weight,
                    'action': 'hold',
                    'contribution': 0.0,
                })
                continue

            for sig in signals:
                confidence = sig.get('confidence', 1.0)

                if sig['action'] == 'buy':
                    contribution = weight * confidence
                    weighted_score += contribution
                elif sig['action'] == 'sell':
                    contribution = -weight * confidence
                    weighted_score += contribution
                else:
                    contribution = 0.0

                signal_details.append({
                    'strategy': strategy.__class__.__name__,
                    'weight': weight,
                    'action': sig['action'],
                    'confidence': confidence,
                    'contribution': contribution,
                    'reason': sig.get('reason', ''),
                })

        # 决策规则
        result_signals = []

        if weighted_score > 0.3:  # 阈值可调
            result_signals.append({
                'action': 'buy',
                'reason': f'加权法: 综合得分{weighted_score:.2f}',
                'confidence': min(abs(weighted_score), 1.0),
                'weighted_score': weighted_score,
                'signal_details': signal_details,
                'method': 'weighted',
            })
        elif weighted_score < -0.3:
            result_signals.append({
                'action': 'sell',
                'reason': f'加权法: 综合得分{weighted_score:.2f}',
                'confidence': min(abs(weighted_score), 1.0),
                'weighted_score': weighted_score,
                'signal_details': signal_details,
                'method': 'weighted',
            })

        return result_signals

    def generate_signals_dynamic(self, df: pd.DataFrame, date: str,
                                context: Optional[Dict] = None) -> List[Dict]:
        """
        动态加权法: 根据近期表现动态调整权重

        流程:
        1. 计算各策略近期表现(如最近10个信号的准确率)
        2. 动态调整权重(表现好的策略权重上升)
        3. 使用新权重生成信号
        """
        # 更新权重(基于历史表现)
        self._update_dynamic_weights()

        # 使用动态权重生成信号(复用加权法逻辑)
        signals = self.generate_signals_weighted(df, date, context)

        # 标记为动态加权
        for sig in signals:
            sig['method'] = 'dynamic'
            sig['current_weights'] = {
                self.strategies[i].__class__.__name__: self.weights[i]
                for i in range(len(self.strategies))
            }

        return signals

    def _update_dynamic_weights(self, lookback: int = 10):
        """
        更新动态权重

        Args:
            lookback: 回溯窗口(最近N个信号)
        """
        # 计算各策略的近期胜率
        performance_scores = []

        for i in range(len(self.strategies)):
            recent_performance = self.strategy_performance[i][-lookback:]

            if len(recent_performance) == 0:
                # 无历史记录,使用初始权重
                performance_scores.append(1.0)
            else:
                # 胜率 = 盈利信号数 / 总信号数
                win_rate = sum(1 for p in recent_performance if p > 0) / len(recent_performance)
                performance_scores.append(max(win_rate, 0.1))  # 最低0.1避免权重为0

        # 归一化为权重
        total_score = sum(performance_scores)
        self.weights = [score / total_score for score in performance_scores]

        logger.debug(f"动态权重更新: {self.weights}")

    def record_performance(self, strategy_idx: int, pnl: float):
        """
        记录策略表现(用于动态加权)

        Args:
            strategy_idx: 策略索引
            pnl: 盈亏(>0为盈利, <0为亏损)
        """
        if strategy_idx < 0 or strategy_idx >= len(self.strategies):
            raise ValueError(f"无效的策略索引: {strategy_idx}")

        self.strategy_performance[strategy_idx].append(pnl)

    def optimize_weights(self, historical_returns: pd.DataFrame,
                        objective: str = 'sharpe') -> List[float]:
        """
        优化策略权重(基于历史回报)

        Args:
            historical_returns: 各策略的历史收益率DataFrame
                - index: 日期
                - columns: 策略名称
            objective: 优化目标('sharpe', 'return', 'min_variance')

        Returns:
            最优权重列表
        """
        from scipy.optimize import minimize

        n_strategies = len(self.strategies)

        # 目标函数
        def objective_function(weights):
            portfolio_returns = (historical_returns * weights).sum(axis=1)

            if objective == 'sharpe':
                # 最大化夏普比率 = 最小化负夏普
                mean_return = portfolio_returns.mean()
                std_return = portfolio_returns.std()
                sharpe = mean_return / std_return if std_return > 0 else 0
                return -sharpe
            elif objective == 'return':
                # 最大化收益 = 最小化负收益
                return -portfolio_returns.mean()
            elif objective == 'min_variance':
                # 最小化方差
                return portfolio_returns.var()
            else:
                raise ValueError(f"未知的优化目标: {objective}")

        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # 权重和=1
        ]

        # 边界条件(每个权重在0-1之间)
        bounds = [(0, 1) for _ in range(n_strategies)]

        # 初始猜测(等权)
        initial_weights = [1.0 / n_strategies] * n_strategies

        # 优化
        result = minimize(
            objective_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            optimized_weights = result.x.tolist()
            logger.info(f"权重优化成功: {optimized_weights}")
            return optimized_weights
        else:
            logger.warning(f"权重优化失败: {result.message}")
            return self.weights

    def get_strategy_summary(self) -> Dict:
        """
        获取策略集成摘要

        Returns:
            摘要信息
        """
        return {
            '子策略数量': len(self.strategies),
            '集成方法': self.method,
            '当前权重': {
                self.strategies[i].__class__.__name__: self.weights[i]
                for i in range(len(self.strategies))
            },
            '历史信号数': {
                self.strategies[i].__class__.__name__: len(self.strategy_performance[i])
                for i in range(len(self.strategies))
            },
        }


def create_ensemble_strategy(strategy_configs: List[Dict],
                            method: str = 'voting',
                            weights: Optional[List[float]] = None) -> EnsembleStrategy:
    """
    便捷函数: 创建策略集成

    Args:
        strategy_configs: 策略配置列表
            [
                {'class': MomentumStrategy, 'params': {'period': 20}},
                {'class': ValueStrategy, 'params': {'pe_threshold': 15}},
            ]
        method: 集成方法
        weights: 权重

    Returns:
        EnsembleStrategy实例
    """
    strategies = []

    for config in strategy_configs:
        strategy_class = config['class']
        params = config.get('params', {})
        strategy = strategy_class(**params)
        strategies.append(strategy)

    return EnsembleStrategy(strategies, method=method, weights=weights)


# 示例: 如何使用策略集成
if __name__ == "__main__":
    from src.strategy.interpretable_strategy import InterpretableStrategy

    # 创建3个子策略(示例)
    strategies = [
        InterpretableStrategy(name='多因子均衡'),
        InterpretableStrategy(name='动量趋势'),
        InterpretableStrategy(name='价值投资'),
    ]

    # 投票法集成
    ensemble_voting = EnsembleStrategy(strategies, method='voting')

    # 加权法集成(手动设置权重)
    ensemble_weighted = EnsembleStrategy(
        strategies,
        method='weighted',
        weights=[0.5, 0.3, 0.2]  # 多因子50%, 动量30%, 价值20%
    )

    # 动态加权法集成
    ensemble_dynamic = EnsembleStrategy(strategies, method='dynamic')

    print("策略集成框架已初始化!")
