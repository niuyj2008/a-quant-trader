"""
股票量化策略决策支持系统 - 因子分析器

提供因子有效性评估：IC/IR分析、因子衰减、分组回测
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger


class FactorAnalyzer:
    """因子有效性分析器"""

    @staticmethod
    def calculate_ic(factor_values: pd.Series, future_returns: pd.Series) -> float:
        """
        计算IC (Information Coefficient)
        因子值与未来收益的截面相关系数
        """
        valid = pd.concat([factor_values, future_returns], axis=1).dropna()
        if len(valid) < 10:
            return 0.0
        return valid.iloc[:, 0].corr(valid.iloc[:, 1])

    @staticmethod
    def calculate_ic_series(factor_df: pd.DataFrame, returns_df: pd.DataFrame,
                            factor_col: str, return_col: str = 'future_return',
                            periods: int = 5) -> pd.Series:
        """
        计算IC时序：在每个截面日期上计算因子预测力
        """
        result = {}
        dates = factor_df.index.unique()
        for date in dates:
            f = factor_df.loc[date, factor_col] if date in factor_df.index else None
            r = returns_df.loc[date, return_col] if date in returns_df.index else None
            if f is not None and r is not None:
                try:
                    ic = float(pd.Series(f).corr(pd.Series(r)))
                    if not np.isnan(ic):
                        result[date] = ic
                except Exception:
                    pass
        return pd.Series(result)

    @staticmethod
    def calculate_ir(ic_series: pd.Series) -> float:
        """
        计算IR (Information Ratio) = IC均值 / IC标准差
        IR > 0.5 表示因子有较好的预测稳定性
        """
        if len(ic_series) < 5 or ic_series.std() == 0:
            return 0.0
        return ic_series.mean() / ic_series.std()

    @staticmethod
    def factor_quintile_returns(data: pd.DataFrame, factor_col: str,
                                 return_col: str, n_groups: int = 5) -> pd.DataFrame:
        """
        因子分组回测：按因子值分N组，计算各组平均收益
        """
        data = data.dropna(subset=[factor_col, return_col])
        if len(data) < n_groups * 2:
            return pd.DataFrame()
        data['group'] = pd.qcut(data[factor_col], n_groups, labels=False, duplicates='drop')
        result = data.groupby('group')[return_col].agg(['mean', 'std', 'count'])
        result.columns = ['平均收益', '收益标准差', '样本数']
        result.index = [f'第{i+1}组' for i in result.index]
        return result

    @staticmethod
    def factor_decay(df: pd.DataFrame, factor_col: str, max_periods: int = 20) -> Dict[int, float]:
        """
        因子衰减分析：因子在不同预测周期的IC值
        """
        result = {}
        for period in [1, 3, 5, 10, 15, 20]:
            if period > max_periods:
                break
            future_return = df['close'].pct_change(period).shift(-period)
            valid = pd.concat([df[factor_col], future_return], axis=1).dropna()
            if len(valid) > 20:
                ic = valid.iloc[:, 0].corr(valid.iloc[:, 1])
                result[period] = ic
        return result

    @staticmethod
    def factor_report(df: pd.DataFrame, factor_col: str, future_periods: List[int] = None) -> Dict:
        """生成因子质量报告"""
        if future_periods is None:
            future_periods = [5, 10, 20]

        report = {"因子名称": factor_col}

        for period in future_periods:
            future_ret = df['close'].pct_change(period).shift(-period)
            valid = pd.concat([df[factor_col], future_ret], axis=1).dropna()
            if len(valid) > 20:
                ic = valid.iloc[:, 0].corr(valid.iloc[:, 1])
                report[f"IC({period}日)"] = round(ic, 4) if not np.isnan(ic) else 0.0
            else:
                report[f"IC({period}日)"] = 0.0

        # 基本统计
        factor_data = df[factor_col].dropna()
        report["均值"] = round(factor_data.mean(), 4) if len(factor_data) > 0 else 0.0
        report["标准差"] = round(factor_data.std(), 4) if len(factor_data) > 0 else 0.0
        report["偏度"] = round(factor_data.skew(), 4) if len(factor_data) > 0 else 0.0
        report["覆盖率"] = round(len(factor_data) / len(df) * 100, 1)

        return report
