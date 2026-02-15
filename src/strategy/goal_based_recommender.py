"""
目标导向策略推荐系统 - Phase 7

根据用户的投资目标(时间期限、目标收益率、风险承受能力)
反向推荐最合适的策略组合
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class InvestmentGoal:
    """投资目标"""

    # 时间目标
    time_horizon_years: float  # 投资期限(年)

    # 收益目标
    target_return: float  # 目标年化收益率(如0.15表示15%)

    # 风险承受能力
    risk_tolerance: str  # 'conservative'(保守), 'moderate'(稳健), 'aggressive'(激进)

    # 约束条件
    max_drawdown: float = -0.20  # 最大可接受回撤(-20%)
    min_sharpe: float = 1.0  # 最小夏普比率

    # 投资偏好
    prefer_etf: bool = False  # 是否偏好ETF定投
    initial_capital: float = 100000  # 初始资金
    monthly_invest: float = 0  # 每月定投金额(0表示一次性投入)

    def __post_init__(self):
        """验证目标参数"""
        if self.time_horizon_years <= 0:
            raise ValueError("投资期限必须>0")

        if self.target_return <= 0:
            raise ValueError("目标收益率必须>0")

        if self.risk_tolerance not in ['conservative', 'moderate', 'aggressive']:
            raise ValueError("风险承受能力必须是conservative/moderate/aggressive之一")

    def get_risk_constraints(self) -> Dict:
        """获取风险约束"""
        constraints = {
            'conservative': {
                'max_drawdown': -0.15,
                'max_volatility': 0.15,
                'min_sharpe': 1.2,
            },
            'moderate': {
                'max_drawdown': -0.25,
                'max_volatility': 0.25,
                'min_sharpe': 1.0,
            },
            'aggressive': {
                'max_drawdown': -0.35,
                'max_volatility': 0.40,
                'min_sharpe': 0.8,
            },
        }

        return constraints[self.risk_tolerance]

    def summary(self) -> str:
        """目标摘要"""
        years_text = f"{self.time_horizon_years:.1f}年" if self.time_horizon_years < 10 else f"{int(self.time_horizon_years)}年"

        total_return = (1 + self.target_return) ** self.time_horizon_years - 1

        risk_map = {
            'conservative': '保守型',
            'moderate': '稳健型',
            'aggressive': '激进型',
        }

        return (
            f"投资期限: {years_text}\n"
            f"年化收益目标: {self.target_return:.1%}\n"
            f"总收益目标: {total_return:.1%}\n"
            f"风险承受: {risk_map[self.risk_tolerance]}\n"
            f"初始资金: {self.initial_capital:,.0f}元\n"
            f"每月定投: {self.monthly_invest:,.0f}元"
        )


class StrategyRecommender:
    """
    策略推荐器

    根据投资目标匹配最合适的策略
    """

    def __init__(self):
        # 策略历史表现数据库(基于Phase 5的历史验证结果)
        # 实际应用中从数据库或历史回测结果中加载
        self.strategy_performance = {
            '多因子均衡': {
                'annual_return': 0.12,
                'sharpe_ratio': 1.5,
                'max_drawdown': -0.18,
                'volatility': 0.20,
                'win_rate': 0.65,
                'suitable_horizon': (1, 5),  # 适合1-5年
                'risk_level': 'moderate',
                'type': 'stock_picking',
            },
            '动量趋势': {
                'annual_return': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.25,
                'volatility': 0.28,
                'win_rate': 0.58,
                'suitable_horizon': (0.5, 3),
                'risk_level': 'aggressive',
                'type': 'stock_picking',
            },
            '价值投资': {
                'annual_return': 0.10,
                'sharpe_ratio': 1.8,
                'max_drawdown': -0.12,
                'volatility': 0.15,
                'win_rate': 0.68,
                'suitable_horizon': (3, 10),
                'risk_level': 'conservative',
                'type': 'stock_picking',
            },
            'ETF定投-沪深300': {
                'annual_return': 0.08,
                'sharpe_ratio': 1.6,
                'max_drawdown': -0.15,
                'volatility': 0.18,
                'win_rate': 0.70,
                'suitable_horizon': (3, 10),
                'risk_level': 'conservative',
                'type': 'etf_dca',
            },
            'ETF价值平均': {
                'annual_return': 0.12,
                'sharpe_ratio': 1.4,
                'max_drawdown': -0.10,
                'volatility': 0.16,
                'win_rate': 0.75,
                'suitable_horizon': (3, 10),
                'risk_level': 'moderate',
                'type': 'etf_va',
            },
            '股债平衡': {
                'annual_return': 0.06,
                'sharpe_ratio': 2.0,
                'max_drawdown': -0.08,
                'volatility': 0.10,
                'win_rate': 0.80,
                'suitable_horizon': (5, 20),
                'risk_level': 'conservative',
                'type': 'rebalancing',
            },
        }

    def recommend(self, goal: InvestmentGoal) -> Dict:
        """
        推荐策略

        Args:
            goal: 投资目标

        Returns:
            推荐结果
        """
        logger.info(f"开始为目标推荐策略: 目标年化收益{goal.target_return:.1%}, 期限{goal.time_horizon_years}年")

        # 1. 筛选候选策略
        candidates = self._filter_candidates(goal)

        if not candidates:
            return {
                'status': 'no_match',
                'message': '没有找到匹配的策略,建议降低目标收益或提高风险承受能力',
            }

        # 2. 评分排序
        ranked_strategies = self._rank_strategies(candidates, goal)

        # 3. 计算达成概率
        for strategy in ranked_strategies:
            strategy['success_probability'] = self._calculate_success_probability(
                strategy, goal
            )

        # 4. 生成推荐报告
        report = self._generate_report(ranked_strategies, goal)

        return {
            'status': 'success',
            'goal': goal,
            'recommended_strategies': ranked_strategies,
            'report': report,
        }

    def _filter_candidates(self, goal: InvestmentGoal) -> List[Dict]:
        """筛选候选策略"""
        candidates = []
        risk_constraints = goal.get_risk_constraints()

        for name, perf in self.strategy_performance.items():
            # 条件1: 期望收益需要接近或超过目标
            # 允许20%的波动范围
            if perf['annual_return'] < goal.target_return * 0.8:
                continue

            # 条件2: 风险约束
            if perf['max_drawdown'] < risk_constraints['max_drawdown']:
                continue

            if perf['sharpe_ratio'] < risk_constraints['min_sharpe']:
                continue

            # 条件3: 投资期限匹配
            horizon_min, horizon_max = perf['suitable_horizon']
            if not (horizon_min <= goal.time_horizon_years <= horizon_max):
                # 允许一定容差
                if goal.time_horizon_years < horizon_min * 0.8 or goal.time_horizon_years > horizon_max * 1.2:
                    continue

            # 条件4: ETF偏好
            if goal.prefer_etf and perf['type'] not in ['etf_dca', 'etf_va', 'rebalancing']:
                continue

            candidates.append({
                'name': name,
                'performance': perf.copy(),
            })

        logger.info(f"筛选出{len(candidates)}个候选策略")
        return candidates

    def _rank_strategies(self, candidates: List[Dict], goal: InvestmentGoal) -> List[Dict]:
        """
        策略评分排序

        评分维度:
        1. 收益匹配度(30%)
        2. 风险控制(30%)
        3. 稳定性(20%)
        4. 期限匹配度(20%)
        """
        for candidate in candidates:
            perf = candidate['performance']

            # 1. 收益匹配度(越接近目标越好,但不能太低)
            return_gap = perf['annual_return'] - goal.target_return
            if return_gap >= 0:
                # 超出目标,分数满分
                return_score = 100
            else:
                # 低于目标,按比例扣分
                return_score = max(0, 100 + (return_gap / goal.target_return) * 100)

            # 2. 风险控制(回撤越小越好)
            risk_constraints = goal.get_risk_constraints()
            max_dd_allowed = risk_constraints['max_drawdown']

            dd_score = 100 * (1 - abs(perf['max_drawdown']) / abs(max_dd_allowed))
            dd_score = max(0, min(100, dd_score))

            # 3. 稳定性(夏普比率)
            sharpe_score = min(100, perf['sharpe_ratio'] / 2.0 * 100)

            # 4. 期限匹配度
            horizon_min, horizon_max = perf['suitable_horizon']
            horizon_center = (horizon_min + horizon_max) / 2

            horizon_gap = abs(goal.time_horizon_years - horizon_center) / horizon_center
            horizon_score = max(0, 100 - horizon_gap * 100)

            # 加权总分
            total_score = (
                return_score * 0.30 +
                dd_score * 0.30 +
                sharpe_score * 0.20 +
                horizon_score * 0.20
            )

            candidate['scores'] = {
                'total': total_score,
                'return_match': return_score,
                'risk_control': dd_score,
                'stability': sharpe_score,
                'horizon_match': horizon_score,
            }

        # 按总分排序
        candidates.sort(key=lambda x: x['scores']['total'], reverse=True)

        return candidates

    def _calculate_success_probability(self, strategy: Dict, goal: InvestmentGoal) -> float:
        """
        计算达成目标的概率

        使用蒙特卡洛模拟:
        假设年化收益率服从正态分布 N(μ, σ²)
        模拟10000次,计算达成目标的次数
        """
        perf = strategy['performance']

        mu = perf['annual_return']  # 期望收益
        sigma = perf['volatility']  # 波动率

        n_simulations = 10000
        years = goal.time_horizon_years

        # 模拟终值
        np.random.seed(42)
        annual_returns = np.random.normal(mu, sigma, (n_simulations, int(years)))

        # 计算复利终值
        final_values = np.prod(1 + annual_returns, axis=1)

        # 计算实际年化收益
        actual_annual_returns = final_values ** (1 / years) - 1

        # 达成目标的次数
        success_count = np.sum(actual_annual_returns >= goal.target_return)

        probability = success_count / n_simulations

        return probability

    def _generate_report(self, strategies: List[Dict], goal: InvestmentGoal) -> str:
        """生成推荐报告"""
        if not strategies:
            return "未找到匹配策略"

        report = []
        report.append("=" * 60)
        report.append("目标导向策略推荐报告")
        report.append("=" * 60)

        report.append("\n📋 投资目标:")
        report.append(goal.summary())

        report.append(f"\n\n✅ 共找到 {len(strategies)} 个匹配策略,按匹配度排序:\n")

        for i, strategy in enumerate(strategies[:5], 1):  # 只显示Top 5
            name = strategy['name']
            perf = strategy['performance']
            scores = strategy['scores']
            prob = strategy['success_probability']

            report.append(f"\n{'=' * 60}")
            report.append(f"推荐{i}: {name}")
            report.append(f"{'=' * 60}")

            report.append(f"\n  📊 历史表现:")
            report.append(f"    年化收益: {perf['annual_return']:.2%}")
            report.append(f"    夏普比率: {perf['sharpe_ratio']:.2f}")
            report.append(f"    最大回撤: {perf['max_drawdown']:.2%}")
            report.append(f"    胜率: {perf['win_rate']:.1%}")
            report.append(f"    波动率: {perf['volatility']:.2%}")

            report.append(f"\n  🎯 匹配度评分:")
            report.append(f"    总分: {scores['total']:.1f}/100")
            report.append(f"    收益匹配: {scores['return_match']:.1f}/100")
            report.append(f"    风险控制: {scores['risk_control']:.1f}/100")
            report.append(f"    稳定性: {scores['stability']:.1f}/100")
            report.append(f"    期限匹配: {scores['horizon_match']:.1f}/100")

            report.append(f"\n  📈 达成概率: {prob:.1%}")

            # 预期终值
            expected_final = goal.initial_capital * (1 + perf['annual_return']) ** goal.time_horizon_years
            target_final = goal.initial_capital * (1 + goal.target_return) ** goal.time_horizon_years

            report.append(f"\n  💰 预期结果:")
            report.append(f"    初始资金: {goal.initial_capital:,.0f}元")
            report.append(f"    预期终值: {expected_final:,.0f}元")
            report.append(f"    目标终值: {target_final:,.0f}元")
            report.append(f"    预期收益: {expected_final - goal.initial_capital:,.0f}元")

        report.append("\n\n" + "=" * 60)
        report.append("💡 建议:")

        best_strategy = strategies[0]
        if best_strategy['success_probability'] >= 0.7:
            report.append(f"  ✅ 推荐采用【{best_strategy['name']}】策略")
            report.append(f"     达成概率{best_strategy['success_probability']:.1%},风险可控")
        elif best_strategy['success_probability'] >= 0.5:
            report.append(f"  ⚠️  可以尝试【{best_strategy['name']}】策略")
            report.append(f"     达成概率{best_strategy['success_probability']:.1%},但存在一定风险")
        else:
            report.append(f"  ❌ 当前目标难以达成(最佳策略概率仅{best_strategy['success_probability']:.1%})")
            report.append(f"     建议: 降低收益目标或延长投资期限")

        report.append("=" * 60)

        return "\n".join(report)


def quick_recommend(
    target_return: float,
    years: float,
    risk_tolerance: str = 'moderate',
    initial_capital: float = 100000
) -> Dict:
    """
    快速推荐(便捷函数)

    Args:
        target_return: 目标年化收益率(如0.15表示15%)
        years: 投资期限(年)
        risk_tolerance: 风险承受能力 'conservative'/'moderate'/'aggressive'
        initial_capital: 初始资金

    Returns:
        推荐结果
    """
    goal = InvestmentGoal(
        time_horizon_years=years,
        target_return=target_return,
        risk_tolerance=risk_tolerance,
        initial_capital=initial_capital,
    )

    recommender = StrategyRecommender()
    return recommender.recommend(goal)
