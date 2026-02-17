"""
策略冗余度分析

分析6个策略之间的信号相关性和收益相关性，
识别冗余策略（可合并）和互补策略（应保留），
评估每个策略的增量贡献。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

from src.strategy.interpretable_strategy import STRATEGY_REGISTRY


@dataclass
class RedundancyReport:
    """冗余度分析报告"""
    signal_correlation: Optional[pd.DataFrame] = None   # 信号相关性矩阵
    return_correlation: Optional[pd.DataFrame] = None    # 收益相关性矩阵
    incremental_value: Dict[str, float] = field(default_factory=dict)  # 增量夏普贡献
    redundant_pairs: List[Tuple[str, str, float]] = field(default_factory=list)  # 冗余对
    recommended_removals: List[str] = field(default_factory=list)  # 建议移除的策略


class StrategyRedundancyAnalyzer:
    """策略冗余度分析器

    分析内容：
    1. 信号一致性矩阵（>80%一致=冗余）
    2. 收益相关性（低相关=互补，高相关=可合并）
    3. 增量贡献：逐个加入集成，夏普无提升=可删除
    """

    def __init__(self, hold_days: int = 20):
        self.hold_days = hold_days

    def analyze(self, data_dict: Dict[str, pd.DataFrame],
                financial_dict: Optional[Dict[str, Dict]] = None,
                strategy_keys: Optional[List[str]] = None) -> RedundancyReport:
        """全面冗余度分析

        Args:
            data_dict: {股票代码: DataFrame}
            financial_dict: {股票代码: 基本面Dict}
            strategy_keys: 要分析的策略列表，默认全部6个

        Returns:
            RedundancyReport
        """
        if strategy_keys is None:
            strategy_keys = list(STRATEGY_REGISTRY.keys())

        report = RedundancyReport()

        # 1. 收集各策略对所有股票的信号和得分
        signals, scores = self._collect_signals(
            data_dict, financial_dict, strategy_keys
        )

        if signals.empty:
            logger.warning("无法收集到有效信号，分析终止")
            return report

        # 2. 信号相关性矩阵
        report.signal_correlation = self._compute_signal_correlation(signals)

        # 3. 收益相关性矩阵
        returns_df = self._compute_strategy_returns(data_dict, strategy_keys)
        if not returns_df.empty:
            report.return_correlation = returns_df.corr(method='spearman').round(3)

        # 4. 增量贡献分析
        report.incremental_value = self._compute_incremental_value(
            returns_df, strategy_keys
        )

        # 5. 识别冗余对
        if report.signal_correlation is not None:
            for i, k1 in enumerate(strategy_keys):
                for j, k2 in enumerate(strategy_keys):
                    if j <= i:
                        continue
                    corr = report.signal_correlation.loc[k1, k2]
                    if abs(corr) > 0.8:
                        report.redundant_pairs.append((k1, k2, round(corr, 3)))

        # 6. 建议移除
        for k1, k2, corr in report.redundant_pairs:
            # 移除增量贡献更低的那个
            v1 = report.incremental_value.get(k1, 0)
            v2 = report.incremental_value.get(k2, 0)
            removal = k2 if v1 >= v2 else k1
            if removal not in report.recommended_removals:
                report.recommended_removals.append(removal)

        logger.info(
            f"冗余度分析完成: {len(report.redundant_pairs)}对冗余, "
            f"建议移除: {report.recommended_removals or '无'}"
        )

        return report

    def _collect_signals(
        self, data_dict: Dict[str, pd.DataFrame],
        financial_dict: Optional[Dict[str, Dict]],
        strategy_keys: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """收集各策略对各股票的信号（buy=1, hold=0, sell=-1）和得分"""
        strategies = {}
        for key in strategy_keys:
            if key in STRATEGY_REGISTRY:
                strategies[key] = STRATEGY_REGISTRY[key]()

        signal_rows = []
        score_rows = []

        for code, df in data_dict.items():
            if df.empty or len(df) < 120:
                continue

            test_df = df.iloc[len(df) // 2:]
            if len(test_df) < 60:
                continue

            sig_row = {'code': code}
            score_row = {'code': code}

            for key, strategy in strategies.items():
                try:
                    fin = financial_dict.get(code) if financial_dict else None
                    report = strategy.analyze_stock(code, test_df, financial_data=fin)
                    if report is None:
                        sig_row[key] = 0
                        score_row[key] = 50
                        continue

                    action_map = {'buy': 1, 'add': 1, 'sell': -1, 'reduce': -1, 'hold': 0}
                    sig_row[key] = action_map.get(report.action, 0)
                    score_row[key] = report.score
                except Exception:
                    sig_row[key] = 0
                    score_row[key] = 50

            signal_rows.append(sig_row)
            score_rows.append(score_row)

        if not signal_rows:
            return pd.DataFrame(), pd.DataFrame()

        signals_df = pd.DataFrame(signal_rows).set_index('code')
        scores_df = pd.DataFrame(score_rows).set_index('code')
        return signals_df, scores_df

    def _compute_signal_correlation(self, signals: pd.DataFrame) -> pd.DataFrame:
        """计算策略间信号一致性（Spearman相关）"""
        if signals.empty or len(signals) < 5:
            return pd.DataFrame()
        return signals.corr(method='spearman').round(3)

    def _compute_strategy_returns(
        self, data_dict: Dict[str, pd.DataFrame],
        strategy_keys: List[str],
    ) -> pd.DataFrame:
        """计算各策略在买入信号时的实际收益"""
        strategies = {}
        for key in strategy_keys:
            if key in STRATEGY_REGISTRY:
                strategies[key] = STRATEGY_REGISTRY[key]()

        returns = {key: [] for key in strategy_keys}
        codes = []

        for code, df in data_dict.items():
            if df.empty or len(df) < 120:
                continue

            test_df = df.iloc[len(df) // 2:]
            if len(test_df) < self.hold_days + 1:
                continue

            # 计算前向收益
            buy_price = test_df.iloc[-self.hold_days - 1]['close']
            sell_price = test_df.iloc[-1]['close']
            if buy_price <= 0:
                continue
            fwd_return = (sell_price - buy_price) / buy_price

            has_signal = False
            for key, strategy in strategies.items():
                try:
                    report = strategy.analyze_stock(code, test_df)
                    if report and report.action in ('buy', 'add'):
                        returns[key].append(fwd_return)
                        has_signal = True
                    else:
                        returns[key].append(np.nan)
                except Exception:
                    returns[key].append(np.nan)

            if has_signal:
                codes.append(code)

        if not codes:
            return pd.DataFrame()

        # 对齐长度
        min_len = min(len(v) for v in returns.values())
        returns_df = pd.DataFrame({k: v[:min_len] for k, v in returns.items()})
        return returns_df

    def _compute_incremental_value(
        self, returns_df: pd.DataFrame, strategy_keys: List[str]
    ) -> Dict[str, float]:
        """计算每个策略的增量贡献

        逐一将策略加入"集成池"，测量夏普比率的提升。
        增量 ≤ 0 表示该策略是冗余的。
        """
        if returns_df.empty:
            return {}

        incremental = {}

        # 基准：所有策略集成的夏普
        all_returns = returns_df.mean(axis=1).dropna()
        all_sharpe = self._calc_sharpe(all_returns)

        for key in strategy_keys:
            # 去掉该策略后的集成夏普
            other_keys = [k for k in strategy_keys if k != key]
            if not other_keys:
                incremental[key] = all_sharpe
                continue

            other_cols = [k for k in other_keys if k in returns_df.columns]
            if not other_cols:
                incremental[key] = 0
                continue

            without_returns = returns_df[other_cols].mean(axis=1).dropna()
            without_sharpe = self._calc_sharpe(without_returns)

            # 增量 = 有该策略 - 无该策略
            incremental[key] = round(all_sharpe - without_sharpe, 4)

        return incremental

    def _calc_sharpe(self, returns: pd.Series) -> float:
        """计算夏普比率"""
        valid = returns.dropna()
        if len(valid) < 3:
            return 0.0
        mean_ret = valid.mean()
        std_ret = valid.std()
        if std_ret < 1e-10:
            return 0.0
        return mean_ret / std_ret * np.sqrt(252 / self.hold_days)

    def generate_summary(self, report: RedundancyReport) -> str:
        """生成文字分析报告"""
        lines = ["=" * 50, "策略冗余度分析报告", "=" * 50]

        if report.signal_correlation is not None:
            lines.append("\n■ 信号相关性矩阵:")
            lines.append(report.signal_correlation.to_string())

        if report.return_correlation is not None:
            lines.append("\n■ 收益相关性矩阵:")
            lines.append(report.return_correlation.to_string())

        lines.append("\n■ 增量贡献 (Δ夏普):")
        for key, val in sorted(report.incremental_value.items(),
                                key=lambda x: x[1], reverse=True):
            status = "有效" if val > 0.05 else "边际" if val > 0 else "冗余"
            lines.append(f"  {key:12s}: {val:+.4f}  ({status})")

        if report.redundant_pairs:
            lines.append(f"\n■ 冗余策略对 (相关>0.8):")
            for k1, k2, corr in report.redundant_pairs:
                lines.append(f"  {k1} <-> {k2}: {corr:.3f}")

        if report.recommended_removals:
            lines.append(f"\n■ 建议移除: {', '.join(report.recommended_removals)}")
        else:
            lines.append("\n■ 无冗余策略，全部保留")

        return "\n".join(lines)
