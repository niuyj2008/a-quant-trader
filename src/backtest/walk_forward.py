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
