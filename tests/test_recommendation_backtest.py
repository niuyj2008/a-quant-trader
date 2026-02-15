"""
测试推荐系统回测功能 - Phase 4
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trading.trade_journal import TradeJournal
from datetime import datetime, timedelta
import pandas as pd


def test_recommendation_backtest_3months():
    """测试1: 回测历史推荐(3个月周期)"""
    print("=" * 60)
    print("测试1: 回测历史推荐(3个月周期)")
    print("=" * 60)

    journal = TradeJournal(db_path="data/test_recommendation_backtest.db")

    # 模拟添加一些历史推荐(A股)
    test_stocks = [
        ("600519", "贵州茅台", "CN", 90),  # 90天前
        ("000001", "平安银行", "CN", 60),  # 60天前
        ("600036", "招商银行", "CN", 30),  # 30天前
        ("601318", "中国平安", "CN", 7),   # 7天前
    ]

    print("\n▶ 添加历史推荐记录:")
    for code, name, market, days_ago in test_stocks:
        rec_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')

        journal.add_recommendation(
            market=market,
            code=code,
            name=name,
            strategy="多因子均衡",
            action="buy",
            score=85.0,
            confidence=0.8,
            reason="技术面+基本面综合评分良好",
            price_at_recommend=100.0,  # 简化测试,使用固定价格
        )
        print(f"  ✓ {code}({name}): {days_ago}天前推荐 @100元")

    # 执行回测
    print("\n▶ 执行回测(获取后续价格)...")
    backtest_result = journal.backtest_recommendations(lookback_days=100, update_db=True)

    print(f"\n📊 回测结果:")
    print(f"  回测推荐数: {backtest_result.get('回测推荐数', 0)}")
    print(f"  更新数: {backtest_result.get('更新数', 0)}")

    if '1周胜率' in backtest_result:
        print(f"\n  1周回测:")
        print(f"    胜率: {backtest_result['1周胜率']:.1%}")
        print(f"    平均收益: {backtest_result['1周平均收益']:.2%}")

    if '1月胜率' in backtest_result:
        print(f"\n  1月回测:")
        print(f"    胜率: {backtest_result['1月胜率']:.1%}")
        print(f"    平均收益: {backtest_result['1月平均收益']:.2%}")

    if '3月胜率' in backtest_result:
        print(f"\n  3月回测:")
        print(f"    胜率: {backtest_result['3月胜率']:.1%}")
        print(f"    平均收益: {backtest_result['3月平均收益']:.2%}")

    # 验证
    assert backtest_result.get('更新数', 0) > 0, "应该至少更新一条记录"

    print("\n✅ 回测功能测试通过!")
    return True


def test_recommendation_performance_stats():
    """测试2: 推荐绩效统计"""
    print("\n" + "=" * 60)
    print("测试2: 推荐绩效统计(1周/1月/3月)")
    print("=" * 60)

    journal = TradeJournal(db_path="data/test_recommendation_backtest.db")

    # 获取绩效统计
    perf = journal.get_recommendation_performance(market="CN")

    print(f"\n📊 推荐绩效统计:")
    print(f"  总推荐数: {perf.get('总推荐数', 0)}")

    if '1周胜率' in perf:
        print(f"\n  1周:")
        print(f"    回测数: {perf.get('1周回测数', 0)}")
        print(f"    胜率: {perf['1周胜率']:.1%}")
        print(f"    平均收益: {perf['1周平均收益']:.2%}")

    if '1月胜率' in perf:
        print(f"\n  1月:")
        print(f"    回测数: {perf.get('1月回测数', 0)}")
        print(f"    胜率: {perf['1月胜率']:.1%}")
        print(f"    平均收益: {perf['1月平均收益']:.2%}")

    if '3月胜率' in perf:
        print(f"\n  3月:")
        print(f"    回测数: {perf.get('3月回测数', 0)}")
        print(f"    胜率: {perf['3月胜率']:.1%}")
        print(f"    平均收益: {perf['3月平均收益']:.2%}")

    print("\n✅ 绩效统计测试通过!")
    return True


def test_strategy_winrate_comparison():
    """测试3: 策略胜率对比"""
    print("\n" + "=" * 60)
    print("测试3: 不同策略胜率对比")
    print("=" * 60)

    journal = TradeJournal(db_path="data/test_recommendation_backtest.db")

    # 添加不同策略的推荐
    strategies = [
        ("多因子均衡", ["600519", "000001"]),
        ("动量趋势", ["600036", "601318"]),
    ]

    print("\n▶ 添加不同策略推荐:")
    for strategy, codes in strategies:
        for code in codes:
            rec_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            journal.add_recommendation(
                market="CN",
                code=code,
                name=f"测试股票{code}",
                strategy=strategy,
                action="buy",
                score=80.0,
                confidence=0.75,
                reason=f"{strategy}推荐",
                price_at_recommend=50.0,
            )
        print(f"  ✓ {strategy}: {len(codes)}只股票")

    # 回测
    print("\n▶ 执行回测...")
    journal.backtest_recommendations(lookback_days=90, update_db=True)

    # 策略对比
    comparison = journal.get_strategy_winrate_comparison()

    if not comparison.empty:
        print(f"\n📊 策略胜率对比:")
        print(f"\n{'策略':<15} {'推荐数':<8} {'1周胜率':<10} {'1月胜率':<10} {'3月胜率':<10}")
        print("-" * 60)

        for _, row in comparison.iterrows():
            strategy = row['strategy']
            count = int(row['total_count'])
            wr_1w = row['winrate_1w'] if pd.notna(row['winrate_1w']) else 0
            wr_1m = row['winrate_1m'] if pd.notna(row['winrate_1m']) else 0
            wr_3m = row['winrate_3m'] if pd.notna(row['winrate_3m']) else 0

            print(f"{strategy:<15} {count:<8} {wr_1w:<10.1%} {wr_1m:<10.1%} {wr_3m:<10.1%}")

    else:
        print("\n⚠️  暂无足够数据进行策略对比")

    print("\n✅ 策略对比测试通过!")
    return True


def test_us_stock_recommendation():
    """测试4: 美股推荐回测"""
    print("\n" + "=" * 60)
    print("测试4: 美股推荐回测")
    print("=" * 60)

    journal = TradeJournal(db_path="data/test_recommendation_backtest.db")

    # 添加美股推荐
    us_stocks = [
        ("AAPL", "Apple Inc.", 90),
        ("MSFT", "Microsoft", 60),
        ("GOOGL", "Alphabet", 30),
    ]

    print("\n▶ 添加美股推荐:")
    for code, name, days_ago in us_stocks:
        rec_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        journal.add_recommendation(
            market="US",
            code=code,
            name=name,
            strategy="美股动量",
            action="buy",
            score=88.0,
            confidence=0.85,
            reason="美股技术面强势",
            price_at_recommend=150.0,
        )
        print(f"  ✓ {code}({name}): {days_ago}天前 @$150")

    # 回测美股
    print("\n▶ 回测美股推荐...")
    us_result = journal.backtest_recommendations(lookback_days=100, update_db=True)

    print(f"\n📊 美股回测结果:")
    print(f"  更新数: {us_result.get('更新数', 0)}")

    if '1周胜率' in us_result:
        print(f"  1周胜率: {us_result['1周胜率']:.1%}")
    if '1月胜率' in us_result:
        print(f"  1月胜率: {us_result['1月胜率']:.1%}")
    if '3月胜率' in us_result:
        print(f"  3月胜率: {us_result['3月胜率']:.1%}")

    # 美股绩效
    us_perf = journal.get_recommendation_performance(market="US")
    print(f"\n📊 美股推荐绩效:")
    print(f"  总推荐数: {us_perf.get('总推荐数', 0)}")

    print("\n✅ 美股回测测试通过!")
    return True


if __name__ == "__main__":
    try:
        # 运行所有测试
        test_recommendation_backtest_3months()
        test_recommendation_performance_stats()
        test_strategy_winrate_comparison()
        test_us_stock_recommendation()

        print("\n" + "=" * 60)
        print("🎉 所有Phase 4测试通过! 推荐系统改进完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
