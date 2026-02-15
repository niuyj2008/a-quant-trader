"""
测试ETF定投策略 - Phase 6
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.strategy.etf_strategies import (
    ETFDollarCostAveraging,
    ETFValueAveraging,
    ETFSmartRebalancing,
)
from src.backtest.engine import BacktestEngine
from src.data.fetcher import DataFetcher
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_dca_strategy():
    """测试1: 定期定额策略"""
    print("=" * 60)
    print("测试1: 定期定额策略 (DCA)")
    print("=" * 60)

    # 创建模拟数据(3年周频数据)
    dates = pd.date_range(start='2021-01-04', end='2024-01-01', freq='W-MON')
    np.random.seed(42)

    # 模拟ETF价格(初始100元,有波动)
    returns = np.random.normal(0.001, 0.02, len(dates))  # 周均涨0.1%,波动2%
    prices = 100 * (1 + returns).cumprod()

    df = pd.DataFrame({
        'close': prices,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'volume': 1000000,
    }, index=dates)

    print(f"\n▶ 模拟数据:")
    print(f"  时间范围: {df.index[0]} ~ {df.index[-1]}")
    print(f"  数据点数: {len(df)}")
    print(f"  起始价格: {df['close'].iloc[0]:.2f}")
    print(f"  结束价格: {df['close'].iloc[-1]:.2f}")
    print(f"  期间涨幅: {(df['close'].iloc[-1] / df['close'].iloc[0] - 1):.2%}")

    # 创建DCA策略(使用美股市场,可买任意股数)
    strategy = ETFDollarCostAveraging(frequency='weekly', invest_amount=1000, market='US')

    # 回测
    engine = BacktestEngine()
    result = engine.run_dca_backtest(
        etf_code='510300',
        df=df,
        strategy=strategy,
        start_date='2021-01-04',
        end_date='2024-01-01'
    )

    print(f"\n📊 DCA回测结果:")
    print(f"  总投入: {result['总投入']:.2f} 元")
    print(f"  最终市值: {result['最终市值']:.2f} 元")
    print(f"  绝对收益: {result['绝对收益']:.2f} 元")
    print(f"  绝对收益率: {result['绝对收益率']:.2%}")
    print(f"  IRR(年化): {result['IRR(年化)']:.2%}")
    print(f"  最大回撤: {result['最大回撤']:.2%}")
    print(f"  定投次数: {result['定投次数']}")

    # 验证
    assert result['status'] == 'success', "回测应该成功"
    assert result['定投次数'] > 100, "3年周频应该有100+次定投"
    assert result['总投入'] > 100000, "总投入应该>10万"
    assert result['IRR(年化)'] != 0, "IRR应该计算出结果"

    print("\n✅ DCA策略测试通过!")
    return True


def test_value_averaging():
    """测试2: 价值平均策略"""
    print("\n" + "=" * 60)
    print("测试2: 价值平均策略 (VA)")
    print("=" * 60)

    # 创建模拟数据(波动较大的市场,价格更低方便测试)
    dates = pd.date_range(start='2021-01-04', end='2024-01-01', freq='W-MON')
    np.random.seed(42)

    # 先跌后涨的市场,起始价格10元
    prices_part1 = 10 * (1 + np.random.normal(-0.002, 0.03, len(dates)//2)).cumprod()
    prices_part2 = prices_part1[-1] * (1 + np.random.normal(0.005, 0.03, len(dates) - len(dates)//2)).cumprod()
    prices = np.concatenate([prices_part1, prices_part2])

    df = pd.DataFrame({
        'close': prices,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'volume': 1000000,
    }, index=dates)

    print(f"\n▶ 模拟数据(先跌后涨):")
    print(f"  起始价格: {df['close'].iloc[0]:.2f}")
    print(f"  中期价格: {df['close'].iloc[len(df)//2]:.2f}")
    print(f"  结束价格: {df['close'].iloc[-1]:.2f}")
    print(f"  前半程涨幅: {(df['close'].iloc[len(df)//2] / df['close'].iloc[0] - 1):.2%}")
    print(f"  后半程涨幅: {(df['close'].iloc[-1] / df['close'].iloc[len(df)//2] - 1):.2%}")

    # 创建VA策略
    strategy = ETFValueAveraging(target_growth_rate=0.01, base_amount=1000)

    # 回测
    engine = BacktestEngine()
    result = engine.run_dca_backtest(
        etf_code='QQQ',
        df=df,
        strategy=strategy,
        start_date='2021-01-04',
        end_date='2024-01-01'
    )

    print(f"\n📊 VA回测结果:")
    print(f"  总投入: {result['总投入']:.2f} 元")
    print(f"  最终市值: {result['最终市值']:.2f} 元")
    print(f"  绝对收益: {result['绝对收益']:.2f} 元")
    print(f"  绝对收益率: {result['绝对收益率']:.2%}")
    print(f"  IRR(年化): {result['IRR(年化)']:.2%}")
    print(f"  最大回撤: {result['最大回撤']:.2%}")
    print(f"  交易次数: {result['交易次数']}")

    # 验证
    assert result['status'] == 'success', "回测应该成功"
    assert result['交易次数'] > 0, "应该有交易发生"

    print("\n✅ VA策略测试通过!")
    return True


def test_smart_rebalancing():
    """测试3: 智能再平衡策略"""
    print("\n" + "=" * 60)
    print("测试3: 智能再平衡策略")
    print("=" * 60)

    # 创建多ETF组合
    etf_weights = {
        '510300': 0.4,  # 沪深300 40%
        'QQQ': 0.3,     # 纳斯达克 30%
        'TLT': 0.3,     # 债券 30%
    }

    strategy = ETFSmartRebalancing(
        etf_weights=etf_weights,
        rebalance_frequency='quarterly',
        deviation_threshold=0.05
    )

    print(f"\n▶ 组合配置:")
    for code, weight in etf_weights.items():
        print(f"  {code}: {weight:.0%}")

    # 模拟当前持仓(权重偏离目标)
    current_holdings = {
        '510300': 1000,  # 假设价格100元 → 市值10万
        'QQQ': 200,      # 假设价格200元 → 市值4万
        'TLT': 500,      # 假设价格120元 → 市值6万
    }

    prices = {
        '510300': 100,
        'QQQ': 200,
        'TLT': 120,
    }

    total_value = sum(current_holdings[code] * prices[code] for code in current_holdings)

    print(f"\n  当前总市值: {total_value:.0f} 元")
    print(f"\n  当前权重分布:")
    for code in etf_weights:
        current_weight = (current_holdings[code] * prices[code]) / total_value
        target_weight = etf_weights[code]
        deviation = current_weight - target_weight
        print(f"    {code}: 当前{current_weight:.1%}, 目标{target_weight:.1%}, 偏差{deviation:+.1%}")

    # 生成再平衡计划
    orders = strategy.generate_rebalance_plan(current_holdings, total_value, prices)

    print(f"\n📊 再平衡计划:")
    if orders:
        print(f"  需要调仓 {len(orders)} 笔:")
        for order in orders:
            action_cn = "买入" if order['action'] == 'buy' else "卖出"
            print(f"    {action_cn} {order['code']}: {order['shares']}股, "
                  f"金额{order['amount']:.0f}元, {order['reason']}")
    else:
        print("  无需调仓(偏差在阈值内)")

    # 验证
    assert len(orders) > 0, "应该有调仓指令(因为当前偏离目标)"

    # 验证再平衡日判断
    assert strategy.is_rebalance_day('2024-01-08') == True, "1月应该是季度再平衡日"
    assert strategy.is_rebalance_day('2024-02-05', '2024-01-08') == False, "2月不是再平衡日"
    assert strategy.is_rebalance_day('2024-04-01', '2024-01-08') == True, "4月是季度再平衡日"

    print("\n✅ 智能再平衡测试通过!")
    return True


def test_real_data_dca():
    """测试4: 真实数据DCA回测(沪深300ETF)"""
    print("\n" + "=" * 60)
    print("测试4: 真实数据DCA回测 (沪深300ETF)")
    print("=" * 60)

    # 获取真实数据
    fetcher = DataFetcher()

    print(f"\n▶ 获取510300历史数据...")
    try:
        df = fetcher.get_daily_data(
            code='510300',
            start_date='2021-01-01',
            end_date='2024-01-01',
            market='CN'
        )
    except Exception as e:
        print(f"⚠️  数据获取失败: {e}")
        print("⚠️  跳过真实数据测试")
        return True

    if df is None or df.empty:
        print("⚠️  无法获取数据,跳过真实数据测试")
        return True

    print(f"  数据范围: {df.index[0]} ~ {df.index[-1]}")
    print(f"  数据点数: {len(df)}")
    print(f"  起始价格: {df['close'].iloc[0]:.2f}")
    print(f"  结束价格: {df['close'].iloc[-1]:.2f}")
    print(f"  期间涨幅: {(df['close'].iloc[-1] / df['close'].iloc[0] - 1):.2%}")

    # 创建月频DCA策略(A股市场,10000元足够买100股)
    strategy = ETFDollarCostAveraging(frequency='monthly', invest_amount=10000, market='CN')

    # 回测
    engine = BacktestEngine()
    result = engine.run_dca_backtest(
        etf_code='510300',
        df=df,
        strategy=strategy,
        start_date='2021-01-01',
        end_date='2024-01-01'
    )

    print(f"\n📊 真实数据DCA回测结果:")
    print(f"  总投入: {result['总投入']:.2f} 元")
    print(f"  最终市值: {result['最终市值']:.2f} 元")
    print(f"  绝对收益: {result['绝对收益']:.2f} 元")
    print(f"  绝对收益率: {result['绝对收益率']:.2%}")
    print(f"  IRR(年化): {result['IRR(年化)']:.2%}")
    print(f"  最大回撤: {result['最大回撤']:.2%}")
    print(f"  定投次数: {result['定投次数']}")

    # 验证
    assert result['status'] == 'success', "真实数据回测应该成功"
    assert result['定投次数'] >= 30, "3年月频至少30次定投"

    # 对比Buy & Hold策略
    buy_hold_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
    print(f"\n📈 策略对比:")
    print(f"  DCA年化收益: {result['IRR(年化)']:.2%}")
    print(f"  Buy&Hold收益: {buy_hold_return:.2%}")

    print("\n✅ 真实数据测试通过!")
    return True


if __name__ == "__main__":
    try:
        # 运行所有测试
        test_dca_strategy()
        test_value_averaging()
        test_smart_rebalancing()
        test_real_data_dca()

        print("\n" + "=" * 60)
        print("🎉 所有Phase 6 ETF定投测试通过!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
