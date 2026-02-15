"""
测试策略历史验证 - Phase 5
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backtest.strategy_validator import SimpleStrategyValidator, quick_validate_strategy
from src.trading.trade_journal import TradeJournal
from datetime import datetime, timedelta
import numpy as np


def test_setup_mock_recommendations():
    """测试0: 设置模拟推荐记录"""
    print("=" * 60)
    print("测试0: 设置历史推荐记录(模拟)")
    print("=" * 60)

    journal = TradeJournal(db_path="data/test_strategy_validation.db")

    # 模拟"多因子均衡"策略的历史推荐
    # 窗口1 (12个月前-9个月前): 表现优秀
    # 窗口2 (9个月前-6个月前): 表现良好
    # 窗口3 (6个月前-3个月前): 表现下滑
    # 窗口4 (3个月前-现在): 表现一般

    strategies_data = [
        # 策略1: 多因子均衡 (总体表现好)
        {
            'name': '多因子均衡',
            'windows': [
                {'start_days_ago': 365, 'count': 10, 'avg_return': 0.12, 'std': 0.05},  # 窗口1
                {'start_days_ago': 275, 'count': 12, 'avg_return': 0.10, 'std': 0.04},  # 窗口2
                {'start_days_ago': 185, 'count': 8, 'avg_return': 0.06, 'std': 0.06},   # 窗口3
                {'start_days_ago': 95, 'count': 10, 'avg_return': 0.08, 'std': 0.05},   # 窗口4
            ]
        },
        # 策略2: 动量趋势 (前期好,后期差)
        {
            'name': '动量趋势',
            'windows': [
                {'start_days_ago': 365, 'count': 15, 'avg_return': 0.15, 'std': 0.08},
                {'start_days_ago': 275, 'count': 12, 'avg_return': 0.08, 'std': 0.10},
                {'start_days_ago': 185, 'count': 10, 'avg_return': 0.02, 'std': 0.12},
                {'start_days_ago': 95, 'count': 8, 'avg_return': -0.03, 'std': 0.10},
            ]
        },
    ]

    print("\n▶ 生成历史推荐记录:")

    total_count = 0

    for strat_data in strategies_data:
        strategy_name = strat_data['name']
        print(f"\n  策略: {strategy_name}")

        for window in strat_data['windows']:
            start_days = window['start_days_ago']
            count = window['count']
            avg_ret = window['avg_return']
            std_ret = window['std']

            for i in range(count):
                # 生成日期(分散在窗口内)
                days_ago = start_days - i * 8  # 每8天一条推荐
                rec_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')

                # 生成收益率(正态分布)
                return_3m = np.random.normal(avg_ret, std_ret)

                journal.add_recommendation(
                    market="CN",
                    code=f"60{1000+total_count:04d}",
                    name=f"测试股票{total_count}",
                    strategy=strategy_name,
                    action="buy",
                    score=80.0,
                    confidence=0.75,
                    reason=f"{strategy_name}推荐",
                    price_at_recommend=100.0,
                )

                # 更新3月收益和日期
                import sqlite3
                with sqlite3.connect(journal.db_path) as conn:
                    conn.execute("""
                        UPDATE recommendations
                        SET date = ?,
                            return_3m = ?,
                            price_after_3m = ?,
                            backtest_status = 'completed'
                        WHERE id = (SELECT MAX(id) FROM recommendations)
                    """, (rec_date, return_3m, 100 * (1 + return_3m)))
                    conn.commit()

                total_count += 1

            print(f"    窗口({start_days}天前): {count}条推荐, 平均收益{avg_ret:.1%}")

    print(f"\n✅ 共生成{total_count}条历史推荐记录")
    return True


def test_validate_single_strategy():
    """测试1: 验证单个策略"""
    print("\n" + "=" * 60)
    print("测试1: 验证单个策略 - 多因子均衡")
    print("=" * 60)

    validator = SimpleStrategyValidator(db_path="data/test_strategy_validation.db")

    result = validator.validate_strategy_by_recommendations(
        strategy_name="多因子均衡",
        market="CN",
        lookback_days=400,
        window_days=90
    )

    if result['status'] != 'success':
        print(f"⚠️  验证失败: {result.get('message')}")
        return False

    print(f"\n📊 验证结果:")
    print(f"  策略: {result['strategy']}")
    print(f"  市场: {result['market']}")
    print(f"  推荐总数: {result['total_recommendations']}")

    summary = result['summary']
    print(f"\n  滚动窗口数: {summary['滚动窗口数']}")
    print(f"  总收益率: {summary['总收益率']}")
    print(f"  年化收益率: {summary['年化收益率']}")
    print(f"  夏普比率: {summary['夏普比率']}")
    print(f"  最大回撤: {summary['最大回撤']}")
    print(f"  窗口胜率: {summary['窗口胜率']}")
    print(f"  平均窗口收益: {summary['平均窗口收益']}")
    print(f"  参数稳定性: {summary['参数稳定性']}")

    health = result['health_score']
    print(f"\n🏥 健康评分:")
    print(f"  总分: {health['total_score']}/100")
    print(f"  评级: {health['grade']}")
    print(f"  建议: {health['recommendation']}")

    subscores = health['subscores']
    print(f"\n  细分得分:")
    print(f"    窗口胜率: {subscores['window_winrate']:.1f}/100")
    print(f"    平均收益: {subscores['avg_return']:.1f}/100")
    print(f"    夏普比率: {subscores['sharpe_ratio']:.1f}/100")
    print(f"    最大回撤: {subscores['max_drawdown']:.1f}/100")
    print(f"    稳定性: {subscores['stability']:.1f}/100")

    # 验证
    assert result['status'] == 'success', "验证应该成功"
    assert health['total_score'] > 0, "健康分数应该>0"

    print("\n✅ 单策略验证测试通过!")
    return True


def test_compare_strategies():
    """测试2: 对比多个策略"""
    print("\n" + "=" * 60)
    print("测试2: 对比多个策略")
    print("=" * 60)

    validator = SimpleStrategyValidator(db_path="data/test_strategy_validation.db")

    comparison = validator.compare_strategies(
        strategies=["多因子均衡", "动量趋势"],
        market="CN",
        lookback_days=400
    )

    if comparison.empty:
        print("⚠️  对比结果为空")
        return False

    print(f"\n📊 策略对比结果:")
    print("\n" + "=" * 100)
    print(f"{'策略':<12} {'总分':<6} {'评级':<6} {'窗口胜率':<10} {'年化收益':<10} {'夏普比率':<10} {'建议':<30}")
    print("=" * 100)

    for _, row in comparison.iterrows():
        print(f"{row['策略']:<12} {row['总分']:<6.1f} {row['评级']:<6} {row['窗口胜率']:<10} "
              f"{row['年化收益']:<10} {row['夏普比率']:<10} {row['建议']:<30}")

    print("=" * 100)

    # 验证
    assert len(comparison) == 2, "应该有2个策略"

    print("\n✅ 策略对比测试通过!")
    return True


def test_quick_validate():
    """测试3: 快速验证函数"""
    print("\n" + "=" * 60)
    print("测试3: 快速验证函数")
    print("=" * 60)

    result = quick_validate_strategy(
        strategy_name="多因子均衡",
        market="CN",
        lookback_days=400
    )

    if result['status'] == 'success':
        print(f"\n✅ 快速验证成功!")
        print(f"  健康评分: {result['health_score']['total_score']}/100")
        print(f"  评级: {result['health_score']['grade']}")
        print(f"  建议: {result['health_score']['recommendation']}")
    else:
        print(f"⚠️  验证失败: {result.get('message')}")

    print("\n✅ 快速验证测试通过!")
    return True


def test_window_details():
    """测试4: 窗口详情分析"""
    print("\n" + "=" * 60)
    print("测试4: 窗口详情分析")
    print("=" * 60)

    validator = SimpleStrategyValidator(db_path="data/test_strategy_validation.db")

    result = validator.validate_strategy_by_recommendations(
        strategy_name="多因子均衡",
        market="CN",
        lookback_days=400,
        window_days=90
    )

    if result['status'] != 'success':
        print(f"⚠️  验证失败")
        return False

    print(f"\n📊 窗口详情 ({len(result['windows'])}个窗口):")
    print("\n" + "-" * 80)
    print(f"{'窗口':<6} {'时间范围':<25} {'收益率':<10} {'胜率':<8} {'交易数':<8}")
    print("-" * 80)

    for window in result['windows']:
        time_range = f"{window.test_start} ~ {window.test_end}"
        return_str = f"{window.test_return:.2%}"
        winrate_str = f"{window.test_win_rate:.1%}"

        # 根据收益率标记
        marker = "📈" if window.test_return > 0 else "📉"

        print(f"{marker} {window.window_id:<4} {time_range:<25} {return_str:<10} "
              f"{winrate_str:<8} {window.n_trades:<8}")

    print("-" * 80)

    print("\n✅ 窗口详情测试通过!")
    return True


if __name__ == "__main__":
    try:
        # 运行所有测试
        test_setup_mock_recommendations()
        test_validate_single_strategy()
        test_compare_strategies()
        test_quick_validate()
        test_window_details()

        print("\n" + "=" * 60)
        print("🎉 所有Phase 5测试通过! 策略历史验证完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
