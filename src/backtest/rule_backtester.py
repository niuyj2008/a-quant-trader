"""
决策规则回测器

阈值网格搜索、附加条件消融实验、自适应评分区间，
用数据验证策略中的硬编码决策规则是否有效。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ThresholdResult:
    """阈值搜索单组结果"""
    buy_threshold: float
    sell_threshold: float
    sharpe: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    profit_factor: float = 0.0


@dataclass
class AblationResult:
    """消融实验结果"""
    condition_name: str
    with_condition_sharpe: float = 0.0
    without_condition_sharpe: float = 0.0
    delta_sharpe: float = 0.0
    with_condition_winrate: float = 0.0
    without_condition_winrate: float = 0.0
    is_beneficial: bool = False


class RuleBacktester:
    """决策规则回测器

    用于系统性测试策略参数：
    1. grid_search_thresholds: 遍历买卖阈值组合，找最优搭配
    2. ablation_study: 有/无附加条件的消融对比
    3. adaptive_score_range: 滚动分位数替代固定评分区间
    """

    def __init__(self, hold_days: int = 20, commission: float = 0.001):
        """
        Args:
            hold_days: 信号触发后默认持有天数
            commission: 单边交易成本（佣金+滑点）
        """
        self.hold_days = hold_days
        self.commission = commission

    def grid_search_thresholds(
        self,
        data_dict: Dict[str, pd.DataFrame],
        strategy,
        buy_range: Tuple[float, float] = (50, 80),
        sell_range: Tuple[float, float] = (20, 50),
        step: float = 5.0,
    ) -> List[ThresholdResult]:
        """阈值网格搜索

        对每组(buy_threshold, sell_threshold)，用策略在历史数据上模拟交易，
        统计夏普比率、年化收益、最大回撤、胜率等指标。

        Args:
            data_dict: {股票代码: DataFrame} 含OHLCV和因子的数据
            strategy: 策略实例（BaseInterpretableStrategy子类）
            buy_range: 买入阈值搜索范围 (min, max)
            sell_range: 卖出阈值搜索范围 (min, max)
            step: 搜索步长

        Returns:
            各阈值组合的结果列表，按夏普降序
        """
        buy_values = np.arange(buy_range[0], buy_range[1] + step, step)
        sell_values = np.arange(sell_range[0], sell_range[1] + step, step)

        results = []
        total = len(buy_values) * len(sell_values)
        tested = 0

        for buy_th in buy_values:
            for sell_th in sell_values:
                if sell_th >= buy_th:
                    continue  # 卖出阈值必须低于买入阈值

                tested += 1
                if tested % 10 == 0:
                    logger.debug(f"阈值搜索进度: {tested}/{total}")

                # 临时修改策略参数
                original_params = strategy.params.copy()
                strategy.params['buy_threshold'] = buy_th
                strategy.params['sell_threshold'] = sell_th

                # 模拟交易
                result = self._simulate_threshold(
                    data_dict, strategy, buy_th, sell_th
                )
                results.append(result)

                # 恢复原始参数
                strategy.params = original_params

        # 按夏普排序
        results.sort(key=lambda r: r.sharpe, reverse=True)

        if results:
            best = results[0]
            logger.info(
                f"阈值网格搜索完成: 最优 buy={best.buy_threshold}, "
                f"sell={best.sell_threshold}, sharpe={best.sharpe:.3f}, "
                f"共测试 {len(results)} 组"
            )

        return results

    def _simulate_threshold(
        self,
        data_dict: Dict[str, pd.DataFrame],
        strategy,
        buy_th: float,
        sell_th: float,
    ) -> ThresholdResult:
        """用指定阈值模拟策略在多只股票上的交易表现"""
        all_returns = []
        trade_count = 0
        wins = 0

        for code, df in data_dict.items():
            if df.empty or len(df) < 120:
                continue

            try:
                # 用后半段数据作为测试集
                test_df = df.iloc[len(df) // 2:]
                if len(test_df) < 60:
                    continue

                report = strategy.analyze_stock(code, test_df)
                if report is None:
                    continue

                score = report.score
                current_price = report.current_price
                if current_price is None or current_price <= 0:
                    continue

                # 买入信号
                if score >= buy_th:
                    ret = self._calc_forward_return(test_df, self.hold_days)
                    if ret is not None:
                        net_ret = ret - 2 * self.commission  # 双边成本
                        all_returns.append(net_ret)
                        trade_count += 1
                        if net_ret > 0:
                            wins += 1

                # 也统计卖出信号（做空等价：未买入=避免损失）
                # 此处仅统计买入信号表现，卖出阈值通过控制不买入间接影响

            except Exception:
                pass

        result = ThresholdResult(
            buy_threshold=buy_th,
            sell_threshold=sell_th,
            trade_count=trade_count,
        )

        if len(all_returns) < 3:
            return result

        returns = np.array(all_returns)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        result.annual_return = mean_ret * (252 / self.hold_days)
        result.max_drawdown = self._estimate_max_drawdown(returns)
        result.win_rate = wins / trade_count if trade_count > 0 else 0

        if std_ret > 1e-10:
            result.sharpe = mean_ret / std_ret * np.sqrt(252 / self.hold_days)

        avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
        avg_loss = abs(np.mean(returns[returns < 0])) if np.any(returns < 0) else 1e-10
        result.profit_factor = avg_win / avg_loss

        return result

    def _calc_forward_return(self, df: pd.DataFrame, days: int) -> Optional[float]:
        """计算最近一行的未来N日收益"""
        if len(df) < days + 1:
            return None
        buy_price = df.iloc[-days - 1]['close']
        sell_price = df.iloc[-1]['close']
        if buy_price <= 0:
            return None
        return (sell_price - buy_price) / buy_price

    def _estimate_max_drawdown(self, returns: np.ndarray) -> float:
        """从一组收益估计最大回撤"""
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

    def ablation_study(
        self,
        data_dict: Dict[str, pd.DataFrame],
        strategy,
        conditions: Optional[Dict[str, callable]] = None,
    ) -> List[AblationResult]:
        """消融实验：逐一移除附加条件，测量对策略效果的影响

        对每个附加条件（如"三段动量一致"、"ADX>25"等），分别测试：
        - 有该条件时的策略表现
        - 去掉该条件时的策略表现
        差值即为该条件的增量价值。

        Args:
            data_dict: 股票数据
            strategy: 策略实例
            conditions: {条件名: 判断函数(df)->bool} 如果为None，使用默认条件集

        Returns:
            各条件的消融结果
        """
        if conditions is None:
            conditions = self._get_default_conditions()

        # 基准：有所有条件时的表现
        baseline_sharpe, baseline_winrate = self._evaluate_with_filter(
            data_dict, strategy, filter_fn=None
        )

        results = []
        for cond_name, cond_fn in conditions.items():
            # 测试移除该条件后的表现（即不执行该过滤）
            without_sharpe, without_winrate = self._evaluate_with_filter(
                data_dict, strategy, filter_fn=None, skip_condition=cond_name
            )

            # 测试仅使用该条件的表现
            with_sharpe, with_winrate = self._evaluate_with_filter(
                data_dict, strategy, filter_fn=cond_fn
            )

            delta = baseline_sharpe - without_sharpe

            result = AblationResult(
                condition_name=cond_name,
                with_condition_sharpe=baseline_sharpe,
                without_condition_sharpe=without_sharpe,
                delta_sharpe=delta,
                with_condition_winrate=baseline_winrate,
                without_condition_winrate=without_winrate,
                is_beneficial=delta > 0.05,  # 夏普提升>0.05算有效
            )
            results.append(result)

            logger.info(
                f"消融实验 [{cond_name}]: "
                f"有={baseline_sharpe:.3f}, 无={without_sharpe:.3f}, "
                f"Δ={delta:+.3f}, {'有效' if result.is_beneficial else '无效'}"
            )

        return results

    def _evaluate_with_filter(
        self,
        data_dict: Dict[str, pd.DataFrame],
        strategy,
        filter_fn: Optional[callable] = None,
        skip_condition: Optional[str] = None,
    ) -> Tuple[float, float]:
        """在可选过滤条件下评估策略表现

        Returns:
            (sharpe, win_rate)
        """
        returns = []
        wins = 0

        for code, df in data_dict.items():
            if df.empty or len(df) < 120:
                continue

            try:
                test_df = df.iloc[len(df) // 2:]
                if len(test_df) < 60:
                    continue

                # 如果有过滤函数，且不通过则跳过
                if filter_fn is not None and not filter_fn(test_df):
                    continue

                report = strategy.analyze_stock(code, test_df)
                if report is None or report.action != "buy":
                    continue

                ret = self._calc_forward_return(test_df, self.hold_days)
                if ret is not None:
                    net_ret = ret - 2 * self.commission
                    returns.append(net_ret)
                    if net_ret > 0:
                        wins += 1
            except Exception:
                pass

        if len(returns) < 3:
            return 0.0, 0.0

        arr = np.array(returns)
        mean_ret = np.mean(arr)
        std_ret = np.std(arr)
        sharpe = mean_ret / std_ret * np.sqrt(252 / self.hold_days) if std_ret > 1e-10 else 0.0
        win_rate = wins / len(returns)

        return sharpe, win_rate

    def _get_default_conditions(self) -> Dict[str, callable]:
        """默认消融条件集"""
        return {
            "ADX趋势确认(>25)": lambda df: (
                df.iloc[-1].get('adx', 0) > 25 if 'adx' in df.columns else True
            ),
            "量价配合(量比>1.2)": lambda df: (
                df.iloc[-1].get('volume_ratio', 1) > 1.2
                if 'volume_ratio' in df.columns else True
            ),
            "RSI非超买(<70)": lambda df: (
                df.iloc[-1].get('rsi_14', 50) < 70
                if 'rsi_14' in df.columns else True
            ),
            "波动率可控(<4%)": lambda df: (
                df.iloc[-1].get('volatility_20', 0.03) < 0.04
                if 'volatility_20' in df.columns else True
            ),
            "MACD金叉": lambda df: (
                df.iloc[-1].get('macd_hist', 0) > 0
                if 'macd_hist' in df.columns else True
            ),
        }

    def adaptive_score_range(
        self,
        data_dict: Dict[str, pd.DataFrame],
        factor_name: str,
        window: int = 250,
        lower_pct: float = 10,
        upper_pct: float = 90,
    ) -> Dict[str, Tuple[float, float]]:
        """自适应评分区间：用滚动分位数替代固定映射范围

        对每个因子，统计其在股票池中最近window天的分位数分布，
        用P10和P90作为评分映射的(low_bad, high_good)，
        替代硬编码的固定值。

        Args:
            data_dict: 股票数据（需已计算因子）
            factor_name: 因子名
            window: 滚动窗口（交易日数）
            lower_pct: 下分位数（默认P10）
            upper_pct: 上分位数（默认P90）

        Returns:
            {股票代码: (low_bound, high_bound)} 各股票的自适应区间
        """
        result = {}

        for code, df in data_dict.items():
            if factor_name not in df.columns or len(df) < window:
                continue

            recent = df[factor_name].iloc[-window:]
            valid = recent.dropna()
            if len(valid) < 50:
                continue

            low = float(np.percentile(valid, lower_pct))
            high = float(np.percentile(valid, upper_pct))

            # 确保low != high
            if abs(high - low) < 1e-10:
                low = float(valid.min())
                high = float(valid.max())

            result[code] = (round(low, 6), round(high, 6))

        return result

    def compute_pooled_score_range(
        self,
        data_dict: Dict[str, pd.DataFrame],
        factor_names: List[str],
        window: int = 250,
        lower_pct: float = 10,
        upper_pct: float = 90,
    ) -> Dict[str, Tuple[float, float]]:
        """计算全池统一的自适应评分区间

        将所有股票的因子值汇聚，计算统一的分位数区间。
        适合用于策略配置文件中的score_ranges。

        Args:
            data_dict: 股票数据
            factor_names: 需要计算的因子列表
            window: 窗口天数
            lower_pct: 下分位
            upper_pct: 上分位

        Returns:
            {因子名: (low_bound, high_bound)}
        """
        result = {}

        for fname in factor_names:
            all_values = []
            for df in data_dict.values():
                if fname not in df.columns:
                    continue
                recent = df[fname].iloc[-window:] if len(df) >= window else df[fname]
                valid = recent.dropna().tolist()
                all_values.extend(valid)

            if len(all_values) < 100:
                logger.debug(f"因子 {fname} 有效数据不足: {len(all_values)}")
                continue

            arr = np.array(all_values)
            low = float(np.percentile(arr, lower_pct))
            high = float(np.percentile(arr, upper_pct))

            if abs(high - low) < 1e-10:
                low = float(arr.min())
                high = float(arr.max())

            result[fname] = (round(low, 6), round(high, 6))

        logger.info(f"计算完成 {len(result)} 个因子的自适应评分区间")
        return result

    def generate_heatmap_data(
        self, results: List[ThresholdResult]
    ) -> pd.DataFrame:
        """将网格搜索结果转为热力图数据（买入阈值×卖出阈值→夏普）

        Returns:
            DataFrame，行=buy_threshold, 列=sell_threshold, 值=sharpe
        """
        if not results:
            return pd.DataFrame()

        rows = []
        for r in results:
            rows.append({
                'buy_threshold': r.buy_threshold,
                'sell_threshold': r.sell_threshold,
                'sharpe': r.sharpe,
            })

        df = pd.DataFrame(rows)
        pivot = df.pivot_table(
            index='buy_threshold',
            columns='sell_threshold',
            values='sharpe',
            aggfunc='first'
        )
        return pivot.sort_index(ascending=False)

    def generate_report(self, results: List[ThresholdResult]) -> pd.DataFrame:
        """生成阈值搜索结果汇总表"""
        rows = []
        for r in results:
            rows.append({
                '买入阈值': r.buy_threshold,
                '卖出阈值': r.sell_threshold,
                '夏普比率': round(r.sharpe, 3),
                '年化收益': f"{r.annual_return:.2%}",
                '最大回撤': f"{r.max_drawdown:.2%}",
                '胜率': f"{r.win_rate:.2%}",
                '交易次数': r.trade_count,
                '盈亏比': round(r.profit_factor, 2),
            })
        return pd.DataFrame(rows)
