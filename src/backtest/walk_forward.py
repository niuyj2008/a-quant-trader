"""
股票量化策略决策支持系统 - Walk-Forward 闭环训练验证

滚动窗口训练-验证框架，将10年数据按滑动窗口划分，
每个窗口独立训练模型并在样本外验证，形成闭环。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class WalkForwardWindow:
    """单个滚动窗口的结果"""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    # 训练期结果
    train_score: float = 0.0
    # 验证期结果
    test_return: float = 0.0
    test_sharpe: float = 0.0
    test_max_drawdown: float = 0.0
    test_win_rate: float = 0.0
    n_trades: int = 0
    # 策略参数
    best_params: Dict = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Walk-Forward整体结果"""
    windows: List[WalkForwardWindow] = field(default_factory=list)
    total_return: float = 0.0
    annualized_return: float = 0.0
    overall_sharpe: float = 0.0
    overall_max_drawdown: float = 0.0
    param_stability: float = 0.0  # 参数稳定性分数

    def summary(self) -> Dict:
        n = len(self.windows)
        if n == 0:
            return {"状态": "无结果"}
        returns = [w.test_return for w in self.windows]
        win_wins = sum(1 for r in returns if r > 0)
        return {
            "滚动窗口数": n,
            "总收益率": f"{self.total_return:.2%}",
            "年化收益率": f"{self.annualized_return:.2%}",
            "夏普比率": f"{self.overall_sharpe:.2f}",
            "最大回撤": f"{self.overall_max_drawdown:.2%}",
            "窗口胜率": f"{win_wins / n:.1%}",
            "平均窗口收益": f"{np.mean(returns):.2%}",
            "参数稳定性": f"{self.param_stability:.2f}",
        }


class WalkForwardValidator:
    """
    Walk-Forward 滚动验证器

    将长期历史数据按滚动窗口划分：
    - 训练期: N年 → 训练策略参数/模型
    - 验证期: M年 → 在未见数据上验证
    - 步进: 每次滑动Delta年

    示例(10年数据, 训练3年, 验证1年, 步进1年):
      窗口1: 训练 2015-2017, 验证 2018
      窗口2: 训练 2016-2018, 验证 2019
      窗口3: 训练 2017-2019, 验证 2020
      ...
    """

    def __init__(self, train_years: int = 3, test_years: int = 1,
                 step_years: int = 1):
        self.train_years = train_years
        self.test_years = test_years
        self.step_years = step_years

    def generate_windows(self, start_date: str, end_date: str) -> List[Dict]:
        """生成所有滚动窗口的日期范围"""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        windows = []
        window_id = 0

        current_train_start = start
        while True:
            train_end = current_train_start + pd.DateOffset(years=self.train_years)
            test_start = train_end
            test_end = test_start + pd.DateOffset(years=self.test_years)

            if test_end > end:
                break

            windows.append({
                'window_id': window_id,
                'train_start': current_train_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'test_start': test_start.strftime('%Y-%m-%d'),
                'test_end': test_end.strftime('%Y-%m-%d'),
            })
            window_id += 1
            current_train_start += pd.DateOffset(years=self.step_years)

        return windows

    def run(self, data: Dict[str, pd.DataFrame],
            strategy_factory: Callable,
            train_func: Callable,
            test_func: Callable,
            start_date: str = "2015-01-01",
            end_date: str = "2025-01-01") -> WalkForwardResult:
        """
        执行Walk-Forward验证

        Args:
            data: {code: full_history_df} 全部历史数据
            strategy_factory: 创建策略实例的工厂函数
            train_func: 训练函数 (strategy, train_data) -> train_score
            test_func: 测试函数 (strategy, test_data) -> test_metrics_dict
            start_date: 数据起始日期
            end_date: 数据结束日期
        """
        windows = self.generate_windows(start_date, end_date)
        logger.info(f"Walk-Forward: {len(windows)}个滚动窗口")

        result = WalkForwardResult()
        cumulative_return = 1.0

        for w in windows:
            logger.info(f"窗口 {w['window_id']}: 训练[{w['train_start']} - {w['train_end']}] "
                        f"验证[{w['test_start']} - {w['test_end']}]")

            # 分割数据
            train_data = self._split_data(data, w['train_start'], w['train_end'])
            test_data = self._split_data(data, w['test_start'], w['test_end'])

            if not train_data or not test_data:
                logger.warning(f"窗口 {w['window_id']} 数据不足，跳过")
                continue

            # 训练
            strategy = strategy_factory()
            try:
                train_score = train_func(strategy, train_data)
            except Exception as e:
                logger.error(f"训练失败: {e}")
                train_score = 0.0

            # 验证
            try:
                test_metrics = test_func(strategy, test_data)
            except Exception as e:
                logger.error(f"验证失败: {e}")
                test_metrics = {'return': 0, 'sharpe': 0, 'max_drawdown': 0,
                                'win_rate': 0, 'n_trades': 0}

            wf_window = WalkForwardWindow(
                window_id=w['window_id'],
                train_start=w['train_start'], train_end=w['train_end'],
                test_start=w['test_start'], test_end=w['test_end'],
                train_score=train_score,
                test_return=test_metrics.get('return', 0),
                test_sharpe=test_metrics.get('sharpe', 0),
                test_max_drawdown=test_metrics.get('max_drawdown', 0),
                test_win_rate=test_metrics.get('win_rate', 0),
                n_trades=test_metrics.get('n_trades', 0),
            )
            result.windows.append(wf_window)
            cumulative_return *= (1 + wf_window.test_return)

        # 汇总统计
        if result.windows:
            result.total_return = cumulative_return - 1
            n_years = len(result.windows) * self.test_years
            result.annualized_return = cumulative_return ** (1 / max(n_years, 1)) - 1
            returns = [w.test_return for w in result.windows]
            result.overall_sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252 / (252 * self.test_years))
                                      if np.std(returns) > 0 else 0)
            result.overall_max_drawdown = min(w.test_max_drawdown for w in result.windows)

        return result

    def _split_data(self, data: Dict[str, pd.DataFrame],
                    start: str, end: str) -> Dict[str, pd.DataFrame]:
        """按日期范围分割数据"""
        result = {}
        for code, df in data.items():
            mask = (df.index >= start) & (df.index < end)
            subset = df[mask]
            if len(subset) > 20:  # 至少20个交易日
                result[code] = subset
        return result

    def calculate_param_stability(self, windows: List[WalkForwardWindow]) -> float:
        """
        计算参数稳定性

        如果策略参数在不同窗口中变化很大,说明策略不稳定(过拟合)
        稳定性评分: 0-100, 100为最稳定
        """
        if len(windows) < 2:
            return 100.0

        # 这里简化处理,实际应该分析best_params的变化
        # 当前基于窗口收益的标准差来评估
        returns = [w.test_return for w in windows]
        if not returns:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if mean_return <= 0:
            return 0.0

        # 变异系数: std/mean, 越小越稳定
        cv = std_return / abs(mean_return) if mean_return != 0 else 999

        # 转换为0-100分数 (cv<0.5 → 100分, cv>2 → 0分)
        stability = max(0, min(100, 100 * (1 - cv / 2)))

        return stability


class StrategyHealthScorer:
    """策略健康评分器"""

    def __init__(self):
        self.weights = {
            'window_winrate': 0.30,      # 窗口胜率 30%
            'avg_return': 0.25,          # 平均收益 25%
            'sharpe_ratio': 0.20,        # 夏普比率 20%
            'max_drawdown': 0.15,        # 最大回撤 15%
            'stability': 0.10,           # 稳定性 10%
        }

    def score(self, result: WalkForwardResult) -> Dict:
        """
        综合评分策略健康度

        Returns:
            {
                'total_score': 85,  # 总分0-100
                'grade': 'A',       # 评级A+/A/B/C/D
                'subscores': {...}, # 各维度得分
                'recommendation': '策略稳定可靠,可实盘使用'
            }
        """
        if not result.windows:
            return {
                'total_score': 0,
                'grade': 'F',
                'recommendation': '无验证数据'
            }

        subscores = {}

        # 1. 窗口胜率得分 (0-100)
        returns = [w.test_return for w in result.windows]
        win_windows = sum(1 for r in returns if r > 0)
        window_winrate = win_windows / len(result.windows)
        subscores['window_winrate'] = self._winrate_score(window_winrate)

        # 2. 平均收益得分 (0-100)
        avg_return = np.mean(returns)
        subscores['avg_return'] = self._return_score(avg_return)

        # 3. 夏普比率得分 (0-100)
        subscores['sharpe_ratio'] = self._sharpe_score(result.overall_sharpe)

        # 4. 最大回撤得分 (0-100)
        subscores['max_drawdown'] = self._drawdown_score(result.overall_max_drawdown)

        # 5. 稳定性得分 (0-100)
        subscores['stability'] = result.param_stability

        # 加权总分
        total_score = sum(
            subscores[k] * self.weights[k]
            for k in self.weights.keys()
        )

        # 评级
        grade = self._get_grade(total_score)

        # 建议
        recommendation = self._generate_recommendation(
            total_score, window_winrate, avg_return, result.overall_sharpe
        )

        return {
            'total_score': round(total_score, 1),
            'grade': grade,
            'subscores': {k: round(v, 1) for k, v in subscores.items()},
            'recommendation': recommendation,
            'details': {
                'window_winrate': f"{window_winrate:.1%}",
                'avg_return': f"{avg_return:.2%}",
                'sharpe_ratio': f"{result.overall_sharpe:.2f}",
                'max_drawdown': f"{result.overall_max_drawdown:.2%}",
            }
        }

    def _winrate_score(self, winrate: float) -> float:
        """胜率得分 (>70% → 100分, <40% → 0分)"""
        if winrate >= 0.70:
            return 100
        elif winrate <= 0.40:
            return 0
        else:
            return (winrate - 0.40) / 0.30 * 100

    def _return_score(self, annual_return: float) -> float:
        """收益得分 (>20% → 100分, <0% → 0分)"""
        if annual_return >= 0.20:
            return 100
        elif annual_return <= 0:
            return 0
        else:
            return annual_return / 0.20 * 100

    def _sharpe_score(self, sharpe: float) -> float:
        """夏普比率得分 (>2 → 100分, <0.5 → 0分)"""
        if sharpe >= 2.0:
            return 100
        elif sharpe <= 0.5:
            return 0
        else:
            return (sharpe - 0.5) / 1.5 * 100

    def _drawdown_score(self, max_dd: float) -> float:
        """回撤得分 (回撤越小越好, <-5% → 100分, >-30% → 0分)"""
        dd_abs = abs(max_dd)
        if dd_abs <= 0.05:
            return 100
        elif dd_abs >= 0.30:
            return 0
        else:
            return (0.30 - dd_abs) / 0.25 * 100

    def _get_grade(self, score: float) -> str:
        """分数转评级"""
        if score >= 90:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 70:
            return 'B+'
        elif score >= 60:
            return 'B'
        elif score >= 50:
            return 'C'
        else:
            return 'D'

    def _generate_recommendation(self, score: float, winrate: float,
                                 avg_return: float, sharpe: float) -> str:
        """生成操作建议"""
        if score >= 80 and winrate >= 0.65:
            return "✅ 策略稳定可靠,可实盘使用"
        elif score >= 70 and winrate >= 0.60:
            return "✅ 策略表现良好,建议小仓位试用"
        elif score >= 60:
            return "⚠️  策略表现一般,需进一步优化参数"
        elif winrate < 0.50:
            return "❌ 窗口胜率<50%,策略可能失效,建议放弃"
        elif avg_return < 0:
            return "❌ 平均收益为负,策略无效"
        else:
            return "⚠️  策略不稳定,风险较高,不建议实盘"
