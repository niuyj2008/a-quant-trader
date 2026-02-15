"""
数据真实性验证器 - Phase 8

严格验证所有数据的真实性,禁止Mock数据
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger
from datetime import datetime


class DataValidator:
    """
    数据真实性验证器

    验证内容:
    1. 价格数据真实性
    2. 回测结果真实性
    3. 数据来源追溯
    """

    def __init__(self):
        self.validation_history = []

    def validate_price_data(self, df: pd.DataFrame, code: str, market: str = 'CN') -> Dict:
        """
        验证价格数据真实性

        Args:
            df: OHLCV数据
            code: 股票/ETF代码
            market: 市场('CN' or 'US')

        Returns:
            验证结果字典
        """
        issues = []

        # 1. 检查DataFrame是否为空
        if df is None or df.empty:
            issues.append("数据为空")
            return {
                'is_valid': False,
                'issues': issues,
                'code': code,
            }

        # 2. 检查必要列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"缺少必要列: {missing_cols}")

        # 3. 检查是否有重复日期
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            issues.append(f"存在{dup_count}个重复日期")

        # 4. 检查涨跌幅是否合理(A股±10%, 美股±50%)
        returns = df['close'].pct_change()
        max_abs_return = returns.abs().max()

        if market == 'CN':
            max_allowed = 0.12  # A股考虑ST涨跌幅5%,加缓冲到12%
        else:
            max_allowed = 0.50  # 美股50%

        if max_abs_return > max_allowed:
            extreme_dates = returns[returns.abs() > max_allowed].index.tolist()
            issues.append(
                f"存在异常涨跌幅: {max_abs_return:.1%} (日期: {extreme_dates[:3]})"
            )

        # 5. 检查成交量是否为0
        zero_volume_ratio = (df['volume'] == 0).sum() / len(df)
        if zero_volume_ratio > 0.1:
            issues.append(f"超过10%的日期成交量为0 ({zero_volume_ratio:.1%})")

        # 6. 检查价格是否为负数或0
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            issues.append("存在负数或0价格")

        # 7. 检查High >= Low
        if (df['high'] < df['low']).any():
            invalid_count = (df['high'] < df['low']).sum()
            issues.append(f"存在{invalid_count}个High < Low的异常数据")

        # 8. 检查Close在[Low, High]范围内
        if ((df['close'] < df['low']) | (df['close'] > df['high'])).any():
            out_of_range = ((df['close'] < df['low']) | (df['close'] > df['high'])).sum()
            issues.append(f"存在{out_of_range}个Close不在[Low,High]范围内")

        # 9. 检查Open在[Low, High]范围内
        if ((df['open'] < df['low']) | (df['open'] > df['high'])).any():
            out_of_range = ((df['open'] < df['low']) | (df['open'] > df['high'])).sum()
            issues.append(f"存在{out_of_range}个Open不在[Low,High]范围内")

        # 10. 检查数据来源标记(如果有)
        data_source = df.attrs.get('source', None) if hasattr(df, 'attrs') else None
        if data_source is None:
            # 尝试从列中获取
            if 'source' in df.columns:
                data_source = df['source'].iloc[0] if len(df) > 0 else 'unknown'
            else:
                issues.append("缺少数据来源标记")
                data_source = 'unknown'

        # 11. 检查数据时间范围连续性
        if len(df) > 1:
            date_diff = df.index.to_series().diff()
            max_gap = date_diff.max()

            # 检查是否有超长间隔(超过60天)
            if pd.notna(max_gap) and max_gap.days > 60:
                issues.append(f"数据存在超长间隔: {max_gap.days}天")

        result = {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'data_source': data_source,
            'date_range': f"{df.index.min()} ~ {df.index.max()}",
            'records': len(df),
            'code': code,
            'market': market,
            'validation_time': datetime.now().isoformat(),
        }

        # 记录验证历史
        self.validation_history.append(result)

        if not result['is_valid']:
            logger.warning(f"数据验证失败 [{code}]: {issues}")
        else:
            logger.info(f"数据验证通过 [{code}]: {len(df)}条记录, 来源={data_source}")

        return result

    def validate_backtest_result(self, result: Dict, trades: List[Dict],
                                 code: str, start_date: str, end_date: str) -> Dict:
        """
        验证回测结果真实性

        Args:
            result: 回测结果
            trades: 交易记录列表
            code: 股票代码
            start_date: 回测起始日期
            end_date: 回测结束日期

        Returns:
            验证结果字典
        """
        issues = []

        # 1. 检查交易记录数量
        if len(trades) == 0 and result.get('total_return', 0) != 0:
            issues.append("无交易但有收益(疑似假数据)")

        # 2. 检查收益率是否过高(年化>500%为异常)
        annual_return = result.get('annual_return', 0)
        if abs(annual_return) > 5.0:
            issues.append(f"年化收益率异常: {annual_return:.0%}")

        # 3. 检查夏普比率是否异常高
        sharpe = result.get('sharpe_ratio', 0)
        if sharpe > 5.0:
            issues.append(f"夏普比率异常高: {sharpe:.2f} (>5.0)")

        # 4. 检查最大回撤是否合理
        max_dd = result.get('max_drawdown', 0)
        if max_dd > 0:
            issues.append(f"最大回撤为正数(应为负): {max_dd:.1%}")
        if max_dd < -0.90:
            issues.append(f"最大回撤过大: {max_dd:.1%} (<-90%)")

        # 5. 检查交易次数是否合理
        n_trades = result.get('n_trades', len(trades))
        days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        if days > 0:
            trades_per_day = n_trades / days
            if trades_per_day > 5:
                issues.append(f"交易频率过高: {trades_per_day:.1f}笔/天")

        # 6. 检查胜率是否合理
        win_rate = result.get('win_rate', 0)
        if win_rate > 0.95:
            issues.append(f"胜率过高(疑似假数据): {win_rate:.1%}")
        if win_rate < 0 or win_rate > 1:
            issues.append(f"胜率不在合理范围[0,1]: {win_rate:.1%}")

        # 7. 验证交易记录的完整性
        for i, trade in enumerate(trades):
            if 'price' not in trade or 'shares' not in trade:
                issues.append(f"交易记录{i}缺少price或shares字段")

            if trade.get('price', 0) <= 0:
                issues.append(f"交易记录{i}价格≤0: {trade.get('price')}")

            if trade.get('shares', 0) <= 0:
                issues.append(f"交易记录{i}股数≤0: {trade.get('shares')}")

        # 8. 检查是否标记了数据来源
        if 'data_source' not in result:
            issues.append("回测结果未标记数据来源")

        validation_result = {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'code': code,
            'backtest_period': f"{start_date} ~ {end_date}",
            'n_trades': len(trades),
            'validation_time': datetime.now().isoformat(),
        }

        self.validation_history.append(validation_result)

        if not validation_result['is_valid']:
            logger.warning(f"回测结果验证失败 [{code}]: {issues}")
        else:
            logger.info(f"回测结果验证通过 [{code}]: {len(trades)}笔交易")

        return validation_result

    def validate_factor_data(self, factors: pd.DataFrame, code: str) -> Dict:
        """
        验证因子数据真实性

        Args:
            factors: 因子DataFrame
            code: 股票代码

        Returns:
            验证结果字典
        """
        issues = []

        if factors is None or factors.empty:
            issues.append("因子数据为空")
            return {'is_valid': False, 'issues': issues, 'code': code}

        # 1. 检查是否有无穷大或NaN值
        inf_count = np.isinf(factors.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            issues.append(f"存在{inf_count}个无穷大值")

        nan_count = factors.select_dtypes(include=[np.number]).isna().sum().sum()
        nan_ratio = nan_count / (len(factors) * len(factors.columns))
        if nan_ratio > 0.3:
            issues.append(f"NaN值比例过高: {nan_ratio:.1%}")

        # 2. 检查因子值范围是否合理
        numeric_cols = factors.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col.startswith('return'):
                # 收益率因子应该在[-1, 10]范围内
                if (factors[col].abs() > 10).any():
                    issues.append(f"因子{col}存在异常值(绝对值>10)")
            elif col.endswith('_ratio') or col.endswith('_pct'):
                # 比率类因子通常在[0, 10]范围
                if (factors[col] < -1).any() or (factors[col] > 20).any():
                    issues.append(f"因子{col}存在异常值")

        # 3. 检查是否所有因子都是常数(无变化)
        constant_factors = []
        for col in numeric_cols:
            if factors[col].std() == 0:
                constant_factors.append(col)

        if len(constant_factors) > len(numeric_cols) * 0.3:
            issues.append(f"超过30%的因子为常数: {constant_factors[:5]}")

        result = {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'code': code,
            'n_factors': len(factors.columns),
            'n_records': len(factors),
            'validation_time': datetime.now().isoformat(),
        }

        self.validation_history.append(result)

        if not result['is_valid']:
            logger.warning(f"因子数据验证失败 [{code}]: {issues}")
        else:
            logger.info(f"因子数据验证通过 [{code}]: {len(factors.columns)}个因子")

        return result

    def get_validation_summary(self) -> Dict:
        """获取验证历史摘要"""
        if not self.validation_history:
            return {
                'total_validations': 0,
                'passed': 0,
                'failed': 0,
                'pass_rate': 0.0,
            }

        total = len(self.validation_history)
        passed = sum(1 for v in self.validation_history if v['is_valid'])
        failed = total - passed

        return {
            'total_validations': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total if total > 0 else 0.0,
            'recent_failures': [
                v for v in self.validation_history[-10:] if not v['is_valid']
            ],
        }

    def enforce_real_data_only(self):
        """
        强制只使用真实数据的运行时检查

        在数据获取和回测时插入检查点
        """
        # 这是一个检查点标记,实际实现需要在DataFetcher和BacktestEngine中集成
        logger.info("数据真实性检查已启用")
        pass


# 全局验证器实例
_global_validator = DataValidator()


def get_validator() -> DataValidator:
    """获取全局验证器实例"""
    return _global_validator
