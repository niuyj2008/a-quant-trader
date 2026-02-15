"""
策略历史验证器 - Phase 5

提供简化的策略历史验证功能,无需完整回测引擎
直接基于推荐记录进行验证
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger

from src.trading.trade_journal import TradeJournal
from src.backtest.walk_forward import WalkForwardResult, WalkForwardWindow, StrategyHealthScorer


class SimpleStrategyValidator:
    """
    简化策略验证器

    基于历史推荐记录进行验证,无需完整回测引擎
    """

    def __init__(self, db_path: str = "data/trade_journal.db"):
        self.journal = TradeJournal(db_path)
        self.scorer = StrategyHealthScorer()

    def validate_strategy_by_recommendations(
        self,
        strategy_name: str,
        market: str = "CN",
        lookback_days: int = 365,
        window_days: int = 90
    ) -> Dict:
        """
        基于推荐记录验证策略

        Args:
            strategy_name: 策略名称
            market: 市场(CN/US)
            lookback_days: 回溯天数(默认1年)
            window_days: 窗口大小(默认90天=3个月)

        Returns:
            验证结果
        """
        logger.info(f"开始验证策略: {strategy_name} (市场: {market})")

        # 获取历史推荐
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

        with self.journal.db_path.parent:
            import sqlite3
            conn = sqlite3.connect(self.journal.db_path)
            query = """
                SELECT * FROM recommendations
                WHERE strategy = ? AND market = ? AND date >= ?
                  AND return_3m IS NOT NULL
                ORDER BY date ASC
            """
            recs = pd.read_sql_query(
                query, conn,
                params=(strategy_name, market, cutoff_date)
            )
            conn.close()

        if recs.empty:
            logger.warning(f"未找到策略 {strategy_name} 的历史推荐记录")
            return {
                'status': 'no_data',
                'message': f'未找到策略 {strategy_name} 的历史推荐记录'
            }

        logger.info(f"找到{len(recs)}条推荐记录")

        # 按时间窗口分组
        windows = self._create_windows(recs, window_days)

        logger.info(f"生成{len(windows)}个滚动窗口")

        # 构建WalkForwardResult
        wf_result = WalkForwardResult()

        for i, window_recs in enumerate(windows):
            if window_recs.empty:
                continue

            # 计算窗口指标
            returns = window_recs['return_3m'].values
            avg_return = np.mean(returns)
            sharpe = self._calculate_sharpe(returns)
            max_dd = self._calculate_max_drawdown(returns)
            win_rate = (returns > 0).sum() / len(returns)

            window = WalkForwardWindow(
                window_id=i,
                train_start="",  # 简化版不需要训练期
                train_end="",
                test_start=window_recs['date'].iloc[0],
                test_end=window_recs['date'].iloc[-1],
                test_return=avg_return,
                test_sharpe=sharpe,
                test_max_drawdown=max_dd,
                test_win_rate=win_rate,
                n_trades=len(window_recs)
            )
            wf_result.windows.append(window)

        if not wf_result.windows:
            return {
                'status': 'no_windows',
                'message': '无法生成验证窗口'
            }

        # 计算总体指标
        all_returns = [w.test_return for w in wf_result.windows]
        wf_result.total_return = np.mean(all_returns) * len(all_returns)
        wf_result.annualized_return = np.mean(all_returns) * (365 / window_days)
        wf_result.overall_sharpe = np.mean([w.test_sharpe for w in wf_result.windows])
        wf_result.overall_max_drawdown = min(w.test_max_drawdown for w in wf_result.windows)

        # 计算稳定性
        validator = self.journal  # 复用
        from src.backtest.walk_forward import WalkForwardValidator
        wfv = WalkForwardValidator()
        wf_result.param_stability = wfv.calculate_param_stability(wf_result.windows)

        # 健康评分
        health_score = self.scorer.score(wf_result)

        return {
            'status': 'success',
            'strategy': strategy_name,
            'market': market,
            'summary': wf_result.summary(),
            'health_score': health_score,
            'windows': wf_result.windows,
            'total_recommendations': len(recs),
        }

    def _create_windows(self, recs: pd.DataFrame, window_days: int) -> List[pd.DataFrame]:
        """将推荐记录按时间窗口分组"""
        recs = recs.copy()
        recs['date'] = pd.to_datetime(recs['date'])
        recs = recs.sort_values('date')

        start_date = recs['date'].min()
        end_date = recs['date'].max()

        windows = []
        current_start = start_date

        while current_start < end_date:
            current_end = current_start + pd.Timedelta(days=window_days)

            window_recs = recs[
                (recs['date'] >= current_start) &
                (recs['date'] < current_end)
            ]

            if len(window_recs) >= 3:  # 至少3条推荐
                windows.append(window_recs)

            # 滑动窗口(步进window_days/2,50%重叠)
            current_start += pd.Timedelta(days=window_days // 2)

        return windows

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """计算夏普比率"""
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # 年化夏普比率 (假设3个月窗口)
        sharpe = (mean_return / std_return) * np.sqrt(4)  # 4个季度
        return sharpe

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """计算最大回撤"""
        if len(returns) == 0:
            return 0.0

        # 累积收益曲线
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        return np.min(drawdown)

    def compare_strategies(
        self,
        strategies: List[str],
        market: str = "CN",
        lookback_days: int = 365
    ) -> pd.DataFrame:
        """
        对比多个策略的表现

        Args:
            strategies: 策略名称列表
            market: 市场
            lookback_days: 回溯天数

        Returns:
            对比结果DataFrame
        """
        results = []

        for strategy in strategies:
            result = self.validate_strategy_by_recommendations(
                strategy, market, lookback_days
            )

            if result['status'] == 'success':
                health = result['health_score']
                summary = result['summary']

                results.append({
                    '策略': strategy,
                    '总分': health['total_score'],
                    '评级': health['grade'],
                    '窗口胜率': summary.get('窗口胜率', 'N/A'),
                    '年化收益': summary.get('年化收益率', 'N/A'),
                    '夏普比率': summary.get('夏普比率', 'N/A'),
                    '最大回撤': summary.get('最大回撤', 'N/A'),
                    '推荐数': result['total_recommendations'],
                    '建议': health['recommendation'],
                })

        return pd.DataFrame(results)


def quick_validate_strategy(
    strategy_name: str,
    market: str = "CN",
    lookback_days: int = 365
) -> Dict:
    """
    快速验证策略(便捷函数)

    Args:
        strategy_name: 策略名称
        market: 市场
        lookback_days: 回溯天数

    Returns:
        验证结果
    """
    validator = SimpleStrategyValidator()
    return validator.validate_strategy_by_recommendations(
        strategy_name, market, lookback_days
    )
