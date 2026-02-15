"""
专业回测报告 - Phase 9.5

对标Quantopian/QuantConnect的专业级回测报告
包含30+指标、月度/年度收益表、因子暴露分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger
from datetime import datetime


class ProfessionalBacktestReport:
    """
    专业级回测报告生成器

    对标业界标准,提供完整的回测分析
    """

    def __init__(self, backtest_result: Dict, benchmark_data: Optional[pd.Series] = None):
        """
        Args:
            backtest_result: 回测结果
                - 'equity_curve': 权益曲线(pd.Series)
                - 'trades': 交易记录(List[Dict])
                - 'holdings': 持仓记录(pd.DataFrame)
                - 'metrics': 基础指标(Dict)
            benchmark_data: 基准收益率序列(可选)
        """
        self.result = backtest_result
        self.benchmark = benchmark_data
        self.equity_curve = backtest_result.get('equity_curve')
        self.trades = backtest_result.get('trades', [])

        logger.info("专业回测报告初始化")

    # ========== 核心指标计算(30+个) ==========

    def calculate_all_metrics(self) -> Dict:
        """
        计算30+个专业指标

        Returns:
            指标字典
        """
        metrics = {}

        # 1. 收益指标
        metrics.update(self._calculate_return_metrics())

        # 2. 风险指标
        metrics.update(self._calculate_risk_metrics())

        # 3. 风险调整收益
        metrics.update(self._calculate_risk_adjusted_metrics())

        # 4. 交易指标
        metrics.update(self._calculate_trade_metrics())

        # 5. 相对基准指标
        if self.benchmark is not None:
            metrics.update(self._calculate_benchmark_metrics())

        # 6. 稳定性指标
        metrics.update(self._calculate_stability_metrics())

        return metrics

    def _calculate_return_metrics(self) -> Dict:
        """收益指标"""
        returns = self.equity_curve.pct_change().dropna()

        # 总收益率
        total_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1

        # 年化收益率
        trading_days = len(self.equity_curve)
        years = trading_days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # CAGR(复合年增长率)
        cagr = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) ** (252 / trading_days) - 1

        # 累计最大收益
        cummax_return = (self.equity_curve / self.equity_curve.expanding().min()).max() - 1

        return {
            '总收益率': total_return,
            '年化收益率': annual_return,
            'CAGR': cagr,
            '累计最大收益': cummax_return,
            '日均收益率': returns.mean(),
        }

    def _calculate_risk_metrics(self) -> Dict:
        """风险指标"""
        returns = self.equity_curve.pct_change().dropna()

        # 年化波动率
        volatility = returns.std() * np.sqrt(252)

        # 下行波动率(只考虑负收益)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0

        # 最大回撤
        cummax = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - cummax) / cummax
        max_drawdown = drawdown.min()

        # 最长回撤期(天数)
        max_dd_duration = self._calculate_max_drawdown_duration(drawdown)

        # VaR和CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()

        # 最大单日涨跌幅
        max_daily_gain = returns.max()
        max_daily_loss = returns.min()

        return {
            '年化波动率': volatility,
            '下行波动率': downside_deviation,
            '最大回撤': max_drawdown,
            '最长回撤期(天)': max_dd_duration,
            'VaR(95%)': var_95,
            'CVaR(95%)': cvar_95,
            '最大单日涨幅': max_daily_gain,
            '最大单日跌幅': max_daily_loss,
        }

    def _calculate_risk_adjusted_metrics(self) -> Dict:
        """风险调整收益指标"""
        returns = self.equity_curve.pct_change().dropna()
        annual_return = self._calculate_return_metrics()['年化收益率']
        volatility = self._calculate_risk_metrics()['年化波动率']
        max_drawdown = self._calculate_risk_metrics()['最大回撤']
        downside_deviation = self._calculate_risk_metrics()['下行波动率']

        # 夏普比率(假设无风险利率3%)
        risk_free_rate = 0.03
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

        # Sortino比率(只考虑下行风险)
        sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

        # Calmar比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Omega比率
        omega_ratio = self._calculate_omega_ratio(returns, threshold=0)

        # 信息比率(相对基准)
        if self.benchmark is not None:
            excess_return = annual_return - self.benchmark.mean() * 252
            tracking_error = (returns - self.benchmark.pct_change()).std() * np.sqrt(252)
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
        else:
            information_ratio = 0

        return {
            '夏普比率': sharpe_ratio,
            'Sortino比率': sortino_ratio,
            'Calmar比率': calmar_ratio,
            'Omega比率': omega_ratio,
            '信息比率': information_ratio,
        }

    def _calculate_trade_metrics(self) -> Dict:
        """交易指标"""
        if not self.trades:
            return {
                '交易次数': 0,
                '胜率': 0,
                '盈亏比': 0,
                '平均持仓天数': 0,
                '换手率': 0,
            }

        # 交易次数
        n_trades = len(self.trades)

        # 胜率
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / n_trades if n_trades > 0 else 0

        # 盈亏比
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]
        avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

        # 平均持仓天数
        holding_days = []
        for trade in self.trades:
            if 'entry_date' in trade and 'exit_date' in trade:
                entry = pd.to_datetime(trade['entry_date'])
                exit_date = pd.to_datetime(trade['exit_date'])
                days = (exit_date - entry).days
                holding_days.append(days)

        avg_holding_days = np.mean(holding_days) if holding_days else 0

        # 换手率(年化)
        total_traded_value = sum([abs(t.get('amount', 0)) for t in self.trades])
        avg_capital = self.equity_curve.mean()
        years = len(self.equity_curve) / 252
        turnover_rate = total_traded_value / (avg_capital * years) if avg_capital > 0 and years > 0 else 0

        return {
            '交易次数': n_trades,
            '胜率': win_rate,
            '盈亏比': profit_factor,
            '平均持仓天数': avg_holding_days,
            '换手率(年化)': turnover_rate,
            '最大连续盈利': self._calculate_max_consecutive_wins(),
            '最大连续亏损': self._calculate_max_consecutive_losses(),
        }

    def _calculate_benchmark_metrics(self) -> Dict:
        """相对基准指标"""
        if self.benchmark is None:
            return {}

        returns = self.equity_curve.pct_change().dropna()
        benchmark_returns = self.benchmark.pct_change().dropna()

        # 对齐长度
        common_idx = returns.index.intersection(benchmark_returns.index)
        if len(common_idx) < 2:
            return {}
        returns = returns.loc[common_idx]
        benchmark_returns = benchmark_returns.loc[common_idx]

        # Alpha和Beta
        cov = np.cov(returns, benchmark_returns)[0, 1]
        var_benchmark = np.var(benchmark_returns)
        beta = cov / var_benchmark if var_benchmark > 0 else 1.0

        annual_return = self._calculate_return_metrics()['年化收益率']
        benchmark_annual_return = (1 + benchmark_returns.mean()) ** 252 - 1
        risk_free_rate = 0.03

        alpha = annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))

        # 跟踪误差
        tracking_error = (returns - benchmark_returns).std() * np.sqrt(252)

        # 超额收益率
        excess_return = annual_return - benchmark_annual_return

        return {
            'Alpha': alpha,
            'Beta': beta,
            '跟踪误差': tracking_error,
            '超额收益率': excess_return,
        }

    def _calculate_stability_metrics(self) -> Dict:
        """稳定性指标"""
        # 月度收益表
        monthly_returns = self._calculate_monthly_returns()

        if len(monthly_returns) == 0:
            return {
                '收益稳定性': 0,
                '正收益月份占比': 0,
                '最佳月份': 0,
                '最差月份': 0,
            }

        # 收益稳定性(月度收益的变异系数倒数)
        monthly_mean = monthly_returns.mean()
        monthly_std = monthly_returns.std()
        stability = abs(monthly_mean) / monthly_std if monthly_std > 0 else 0

        # 正收益月份占比
        positive_months_ratio = (monthly_returns > 0).sum() / len(monthly_returns)

        # 最佳/最差月份
        best_month = monthly_returns.max()
        worst_month = monthly_returns.min()

        return {
            '收益稳定性': stability,
            '正收益月份占比': positive_months_ratio,
            '最佳月份': best_month,
            '最差月份': worst_month,
        }

    # ========== 辅助计算函数 ==========

    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """计算最长回撤期"""
        underwater = drawdown < 0
        periods = []
        current_period = 0

        for is_underwater in underwater:
            if is_underwater:
                current_period += 1
            else:
                if current_period > 0:
                    periods.append(current_period)
                current_period = 0

        return max(periods) if periods else 0

    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """计算Omega比率"""
        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = abs(excess_returns[excess_returns < 0].sum())

        omega = gains / losses if losses > 0 else 0
        return omega

    def _calculate_max_consecutive_wins(self) -> int:
        """最大连续盈利次数"""
        if not self.trades:
            return 0

        max_wins = 0
        current_wins = 0

        for trade in self.trades:
            if trade.get('pnl', 0) > 0:
                current_wins += 1
                max_wins = max(max_wins, current_wins)
            else:
                current_wins = 0

        return max_wins

    def _calculate_max_consecutive_losses(self) -> int:
        """最大连续亏损次数"""
        if not self.trades:
            return 0

        max_losses = 0
        current_losses = 0

        for trade in self.trades:
            if trade.get('pnl', 0) < 0:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0

        return max_losses

    def _calculate_monthly_returns(self) -> pd.Series:
        """计算月度收益率"""
        if self.equity_curve is None or len(self.equity_curve) == 0:
            return pd.Series()

        # 重采样到月末(使用'ME'而不是已弃用的'M')
        monthly_equity = self.equity_curve.resample('ME').last()
        monthly_returns = monthly_equity.pct_change().dropna()

        return monthly_returns

    # ========== 月度/年度收益表 ==========

    def generate_monthly_returns_table(self) -> pd.DataFrame:
        """
        生成月度收益表(类似Quantopian格式)

        Returns:
            月度收益DataFrame,行=年份,列=月份
        """
        monthly_returns = self._calculate_monthly_returns()

        if len(monthly_returns) == 0:
            return pd.DataFrame()

        # 构建透视表
        monthly_returns.index = pd.to_datetime(monthly_returns.index)

        table = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values,
        })

        pivot_table = table.pivot(index='year', columns='month', values='return')

        # 重命名列为月份名称
        month_names = ['1月', '2月', '3月', '4月', '5月', '6月',
                      '7月', '8月', '9月', '10月', '11月', '12月']
        pivot_table.columns = [month_names[i-1] for i in pivot_table.columns]

        # 添加年度总收益列
        pivot_table['年度收益'] = pivot_table.sum(axis=1)

        return pivot_table

    def generate_yearly_returns_table(self) -> pd.DataFrame:
        """
        生成年度收益表

        Returns:
            年度收益DataFrame
        """
        if self.equity_curve is None or len(self.equity_curve) == 0:
            return pd.DataFrame()

        yearly_equity = self.equity_curve.resample('Y').last()
        yearly_returns = yearly_equity.pct_change().dropna()

        table = pd.DataFrame({
            '年份': yearly_returns.index.year,
            '收益率': yearly_returns.values,
        })

        return table

    # ========== 因子暴露分析 ==========

    def analyze_factor_exposure(self, factor_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        因子暴露分析

        Args:
            factor_data: 因子数据DataFrame
                - index: 日期
                - columns: 因子名称
                - 如果为None,尝试从回测结果中提取

        Returns:
            因子暴露分析结果
        """
        if factor_data is None:
            factor_data = self.result.get('factor_exposures')

        if factor_data is None or factor_data.empty:
            return {
                'average_exposure': {},
                'exposure_volatility': {},
                'max_exposure': {},
                'min_exposure': {},
            }

        # 平均暴露
        average_exposure = factor_data.mean().to_dict()

        # 暴露波动率
        exposure_volatility = factor_data.std().to_dict()

        # 最大/最小暴露
        max_exposure = factor_data.max().to_dict()
        min_exposure = factor_data.min().to_dict()

        return {
            'average_exposure': average_exposure,
            'exposure_volatility': exposure_volatility,
            'max_exposure': max_exposure,
            'min_exposure': min_exposure,
        }

    # ========== 滚动分析 ==========

    def calculate_rolling_metrics(self, window: int = 63) -> pd.DataFrame:
        """
        滚动指标分析(如滚动夏普比率)

        Args:
            window: 滚动窗口(默认63天=3个月)

        Returns:
            滚动指标DataFrame
        """
        if self.equity_curve is None or len(self.equity_curve) < window:
            return pd.DataFrame()

        returns = self.equity_curve.pct_change().dropna()

        rolling_metrics = pd.DataFrame(index=returns.index)

        # 滚动夏普比率
        rolling_sharpe = (
            returns.rolling(window).mean() / returns.rolling(window).std()
        ) * np.sqrt(252)
        rolling_metrics['夏普比率'] = rolling_sharpe

        # 滚动波动率
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_metrics['波动率'] = rolling_vol

        # 滚动最大回撤(使用returns的index)
        rolling_dd = []
        for i in range(len(returns)):
            if i < window:
                rolling_dd.append(np.nan)
            else:
                window_equity = self.equity_curve.iloc[i-window+1:i+2]  # +1因为returns比equity_curve少1行
                cummax = window_equity.expanding().max()
                dd = ((window_equity - cummax) / cummax).min()
                rolling_dd.append(dd)

        rolling_metrics['最大回撤'] = rolling_dd

        return rolling_metrics

    # ========== 回撤详细分析 ==========

    def analyze_drawdowns(self) -> List[Dict]:
        """
        回撤详细分析

        Returns:
            回撤列表,每个回撤包含起止日期、幅度、恢复时间
        """
        if self.equity_curve is None or len(self.equity_curve) == 0:
            return []

        cummax = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - cummax) / cummax

        drawdowns = []
        in_drawdown = False
        start_date = None
        start_value = None
        min_value = None
        min_date = None

        for date, dd_value in drawdown.items():
            if dd_value < 0 and not in_drawdown:
                # 开始回撤
                in_drawdown = True
                start_date = date
                start_value = self.equity_curve.loc[date]
                min_value = self.equity_curve.loc[date]
                min_date = date
            elif dd_value < 0 and in_drawdown:
                # 回撤继续
                if self.equity_curve.loc[date] < min_value:
                    min_value = self.equity_curve.loc[date]
                    min_date = date
            elif dd_value == 0 and in_drawdown:
                # 回撤结束
                end_date = date
                end_value = self.equity_curve.loc[date]

                dd_depth = (min_value - start_value) / start_value

                drawdowns.append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'min_date': min_date,
                    'depth': dd_depth,
                    'duration': (end_date - start_date).days,
                    'recovery_time': (end_date - min_date).days,
                })

                in_drawdown = False

        # 按深度排序
        drawdowns.sort(key=lambda x: x['depth'])

        return drawdowns

    # ========== 完整报告生成 ==========

    def generate_full_report(self) -> str:
        """
        生成完整的专业回测报告

        Returns:
            格式化的文本报告
        """
        report = []
        report.append("=" * 80)
        report.append("专业回测报告 (Professional Backtest Report)")
        report.append("=" * 80)

        # 1. 执行摘要
        report.append("\n📊 执行摘要")
        report.append("-" * 80)
        metrics = self.calculate_all_metrics()

        report.append(f"  回测期间: {self.equity_curve.index[0]} ~ {self.equity_curve.index[-1]}")
        report.append(f"  交易天数: {len(self.equity_curve)}天")
        report.append(f"  初始资金: {self.equity_curve.iloc[0]:,.2f}")
        report.append(f"  最终资金: {self.equity_curve.iloc[-1]:,.2f}")
        report.append(f"  总收益率: {metrics['总收益率']:.2%}")
        report.append(f"  年化收益率: {metrics['年化收益率']:.2%}")
        report.append(f"  夏普比率: {metrics['夏普比率']:.2f}")
        report.append(f"  最大回撤: {metrics['最大回撤']:.2%}")

        # 2. 核心指标
        report.append("\n\n📈 核心指标 (30+个)")
        report.append("-" * 80)

        # 收益指标
        report.append("\n[收益指标]")
        for key in ['总收益率', '年化收益率', 'CAGR', '累计最大收益', '日均收益率']:
            value = metrics[key]
            report.append(f"  {key:20s}: {value:>10.2%}")

        # 风险指标
        report.append("\n[风险指标]")
        for key in ['年化波动率', '下行波动率', '最大回撤', '最长回撤期(天)', 'VaR(95%)', 'CVaR(95%)']:
            value = metrics[key]
            if '天' in key:
                report.append(f"  {key:20s}: {value:>10.0f}天")
            else:
                report.append(f"  {key:20s}: {value:>10.2%}")

        # 风险调整收益
        report.append("\n[风险调整收益]")
        for key in ['夏普比率', 'Sortino比率', 'Calmar比率', 'Omega比率', '信息比率']:
            value = metrics[key]
            report.append(f"  {key:20s}: {value:>10.2f}")

        # 交易指标
        report.append("\n[交易指标]")
        for key in ['交易次数', '胜率', '盈亏比', '平均持仓天数', '换手率(年化)']:
            value = metrics[key]
            if key == '交易次数':
                report.append(f"  {key:20s}: {value:>10.0f}")
            elif key == '平均持仓天数':
                report.append(f"  {key:20s}: {value:>10.1f}天")
            elif key in ['胜率', '换手率(年化)']:
                report.append(f"  {key:20s}: {value:>10.2%}")
            else:
                report.append(f"  {key:20s}: {value:>10.2f}")

        # 3. 月度收益表
        monthly_table = self.generate_monthly_returns_table()
        if not monthly_table.empty:
            report.append("\n\n📅 月度收益表")
            report.append("-" * 80)
            report.append(monthly_table.to_string())

        # 4. 回撤分析
        drawdowns = self.analyze_drawdowns()
        if drawdowns:
            report.append("\n\n📉 Top 5 回撤分析")
            report.append("-" * 80)
            for i, dd in enumerate(drawdowns[:5], 1):
                report.append(f"\n  #{i} 回撤:")
                report.append(f"    开始日期: {dd['start_date']}")
                report.append(f"    谷底日期: {dd['min_date']}")
                report.append(f"    恢复日期: {dd['end_date']}")
                report.append(f"    回撤幅度: {dd['depth']:.2%}")
                report.append(f"    持续时间: {dd['duration']}天")
                report.append(f"    恢复时间: {dd['recovery_time']}天")

        # 5. 基准对比
        if self.benchmark is not None:
            report.append("\n\n🎯 vs 基准对比")
            report.append("-" * 80)
            report.append(f"  Alpha: {metrics['Alpha']:.2%}")
            report.append(f"  Beta: {metrics['Beta']:.2f}")
            report.append(f"  跟踪误差: {metrics['跟踪误差']:.2%}")
            report.append(f"  超额收益: {metrics['超额收益率']:.2%}")

        report.append("\n" + "=" * 80)
        report.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)

        return "\n".join(report)


# 便捷函数
def generate_professional_report(backtest_result: Dict,
                                 benchmark_data: Optional[pd.Series] = None) -> Dict:
    """
    快速生成专业回测报告

    Args:
        backtest_result: 回测结果
        benchmark_data: 基准数据(可选)

    Returns:
        {
            'metrics': 所有指标,
            'monthly_table': 月度收益表,
            'drawdowns': 回撤分析,
            'report': 完整报告文本,
        }
    """
    reporter = ProfessionalBacktestReport(backtest_result, benchmark_data)

    metrics = reporter.calculate_all_metrics()
    monthly_table = reporter.generate_monthly_returns_table()
    drawdowns = reporter.analyze_drawdowns()
    report_text = reporter.generate_full_report()

    return {
        'metrics': metrics,
        'monthly_table': monthly_table,
        'drawdowns': drawdowns,
        'report': report_text,
    }


# 示例
if __name__ == "__main__":
    # 创建模拟回测结果
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')

    # 模拟权益曲线(带波动)
    np.random.seed(42)
    equity = 100000 * (1 + np.random.randn(252).cumsum() * 0.01)
    equity_curve = pd.Series(equity, index=dates)

    # 模拟交易记录
    trades = [
        {'entry_date': '2023-01-15', 'exit_date': '2023-02-01', 'pnl': 500, 'amount': 10000},
        {'entry_date': '2023-02-10', 'exit_date': '2023-02-25', 'pnl': -200, 'amount': 8000},
        {'entry_date': '2023-03-05', 'exit_date': '2023-03-20', 'pnl': 800, 'amount': 12000},
    ]

    backtest_result = {
        'equity_curve': equity_curve,
        'trades': trades,
    }

    # 生成报告
    result = generate_professional_report(backtest_result)

    print(result['report'])
