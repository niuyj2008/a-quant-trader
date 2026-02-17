"""
因子有效性验证器

通过IC（信息系数）和IC_IR（信息比率）评估每个因子的预测能力，
用数据回答"这个因子到底有没有用"。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
from scipy import stats


@dataclass
class FactorICResult:
    """单个因子的IC检验结果"""
    factor_name: str
    ic_mean: float = 0.0          # IC均值
    ic_std: float = 0.0           # IC标准差
    ic_ir: float = 0.0            # IC信息比率 = mean/std
    ic_series: List[float] = field(default_factory=list)  # IC时间序列
    ic_positive_ratio: float = 0.0  # IC为正的比例
    effectiveness: str = "无效"     # 强有效/中等有效/弱有效/无效
    forward_days: int = 10         # 预测周期（天）


@dataclass
class FactorValidationReport:
    """因子验证报告"""
    market: str = "US"
    stock_count: int = 0
    date_range: Tuple[str, str] = ("", "")
    factor_results: Dict[str, FactorICResult] = field(default_factory=dict)
    correlation_matrix: Optional[pd.DataFrame] = None  # 因子间相关性


class FactorValidator:
    """因子有效性验证器

    对股票池中的所有股票计算截面IC，评估每个因子对未来收益的预测能力。
    """

    def __init__(self):
        from src.factors.factor_engine import FactorEngine
        self.factor_engine = FactorEngine()

    def validate(self, data_dict: Dict[str, pd.DataFrame],
                 forward_days_list: List[int] = None,
                 min_stocks: int = 20) -> FactorValidationReport:
        """对股票池数据进行因子有效性验证

        Args:
            data_dict: {股票代码: 日线DataFrame} 的字典
            forward_days_list: 预测周期列表，默认[5, 10, 20]
            min_stocks: 每个截面最少需要的股票数（少于此数不计算IC）

        Returns:
            FactorValidationReport
        """
        if forward_days_list is None:
            forward_days_list = [5, 10, 20]

        report = FactorValidationReport(stock_count=len(data_dict))

        # 1. 计算所有股票的因子值
        logger.info(f"开始因子验证: {len(data_dict)}只股票")
        factor_data = {}  # {code: DataFrame with factors}
        for code, df in data_dict.items():
            if df.empty or len(df) < 60:
                continue
            try:
                factors = self.factor_engine.compute_all_core_factors(df)
                if not factors.empty:
                    factor_data[code] = factors
            except Exception as e:
                logger.debug(f"因子计算失败 {code}: {e}")

        if len(factor_data) < min_stocks:
            logger.warning(f"有效股票数不足: {len(factor_data)} < {min_stocks}")
            return report

        # 获取因子名列表（取第一只股票的列名）
        sample_factors = list(next(iter(factor_data.values())).columns)
        # 过滤掉价格列，只保留因子列
        price_cols = {'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover'}
        factor_names = [f for f in sample_factors if f not in price_cols]

        logger.info(f"有效股票: {len(factor_data)}只, 因子数: {len(factor_names)}")

        # 2. 获取日期范围
        all_dates = set()
        for df in factor_data.values():
            all_dates.update(df.index.strftime('%Y-%m-%d'))
        sorted_dates = sorted(all_dates)
        if sorted_dates:
            report.date_range = (sorted_dates[0], sorted_dates[-1])

        # 3. 对每个forward_days计算IC
        for fwd in forward_days_list:
            ic_results = self._compute_cross_sectional_ic(
                factor_data, factor_names, fwd, min_stocks
            )
            for fname, ic_result in ic_results.items():
                key = f"{fname}_fwd{fwd}"
                report.factor_results[key] = ic_result

        # 4. 计算因子间相关性矩阵（用默认fwd=10的截面数据）
        report.correlation_matrix = self._compute_factor_correlation(
            factor_data, factor_names
        )

        return report

    def _compute_cross_sectional_ic(
        self, factor_data: Dict[str, pd.DataFrame],
        factor_names: List[str], forward_days: int,
        min_stocks: int
    ) -> Dict[str, FactorICResult]:
        """计算截面IC

        对每个交易日，将所有股票的因子值与未来N日收益做Spearman秩相关。
        """
        # 收集所有日期
        all_dates = set()
        for df in factor_data.values():
            all_dates.update(df.index)
        sorted_dates = sorted(all_dates)

        results = {f: FactorICResult(factor_name=f, forward_days=forward_days)
                   for f in factor_names}
        ic_series = {f: [] for f in factor_names}

        # 对每个日期计算截面IC
        for date in sorted_dates:
            # 收集这一天所有股票的因子值和未来收益
            cross_section = []
            for code, df in factor_data.items():
                if date not in df.index:
                    continue
                # 计算未来N日收益
                date_loc = df.index.get_loc(date)
                if date_loc + forward_days >= len(df):
                    continue
                future_price = df.iloc[date_loc + forward_days]['close']
                current_price = df.loc[date, 'close']
                if current_price <= 0:
                    continue
                future_return = (future_price - current_price) / current_price

                row = {'code': code, 'future_return': future_return}
                for fname in factor_names:
                    val = df.loc[date, fname] if fname in df.columns else np.nan
                    row[fname] = val
                cross_section.append(row)

            if len(cross_section) < min_stocks:
                continue

            cs_df = pd.DataFrame(cross_section)

            # 对每个因子计算与未来收益的Spearman相关
            import warnings
            for fname in factor_names:
                if fname not in cs_df.columns:
                    continue
                valid = cs_df[[fname, 'future_return']].dropna()
                if len(valid) < min_stocks:
                    continue
                # 跳过常数列（避免大量ConstantInputWarning和无效计算）
                if valid[fname].nunique() < 2 or valid['future_return'].nunique() < 2:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ic, _ = stats.spearmanr(valid[fname], valid['future_return'])
                    if not np.isnan(ic):
                        ic_series[fname].append(ic)
                except Exception:
                    pass

        # 汇总结果
        for fname in factor_names:
            series = ic_series[fname]
            if len(series) < 10:
                continue
            ic_mean = np.mean(series)
            ic_std = np.std(series) if np.std(series) > 0 else 1e-10
            ic_ir = ic_mean / ic_std
            ic_positive = np.mean([1 if x > 0 else 0 for x in series])

            # 有效性评级
            abs_ir = abs(ic_ir)
            if abs_ir > 0.5:
                effectiveness = "强有效"
            elif abs_ir > 0.3:
                effectiveness = "中等有效"
            elif abs_ir > 0.1:
                effectiveness = "弱有效"
            else:
                effectiveness = "无效"

            results[fname] = FactorICResult(
                factor_name=fname,
                ic_mean=round(ic_mean, 4),
                ic_std=round(ic_std, 4),
                ic_ir=round(ic_ir, 4),
                ic_series=series,
                ic_positive_ratio=round(ic_positive, 4),
                effectiveness=effectiveness,
                forward_days=forward_days,
            )

        return results

    def _compute_factor_correlation(
        self, factor_data: Dict[str, pd.DataFrame],
        factor_names: List[str]
    ) -> pd.DataFrame:
        """计算因子间相关性矩阵（检测冗余因子）"""
        # 取最近一个截面的因子值
        all_factor_values = []
        for code, df in factor_data.items():
            if df.empty:
                continue
            latest = df.iloc[-1]
            row = {}
            for fname in factor_names:
                row[fname] = latest.get(fname, np.nan)
            all_factor_values.append(row)

        if len(all_factor_values) < 10:
            return pd.DataFrame()

        factor_df = pd.DataFrame(all_factor_values)
        return factor_df[factor_names].corr(method='spearman').round(3)

    def generate_summary(self, report: FactorValidationReport,
                         forward_days: int = 10) -> pd.DataFrame:
        """生成因子有效性摘要表

        Args:
            report: 验证报告
            forward_days: 选择哪个预测周期的结果

        Returns:
            DataFrame，按|IC_IR|降序排列
        """
        rows = []
        suffix = f"_fwd{forward_days}"
        for key, result in report.factor_results.items():
            if not key.endswith(suffix):
                continue
            rows.append({
                '因子': result.factor_name,
                'IC均值': result.ic_mean,
                'IC标准差': result.ic_std,
                'IC_IR': result.ic_ir,
                '|IC_IR|': abs(result.ic_ir),
                'IC正比例': result.ic_positive_ratio,
                '有效性': result.effectiveness,
                '预测周期': f"{forward_days}日",
                'IC样本数': len(result.ic_series),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('|IC_IR|', ascending=False).reset_index(drop=True)
        return df
