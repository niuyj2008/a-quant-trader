"""
测试专业回测报告 - Phase 9.5
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backtest.professional_report import (
    ProfessionalBacktestReport,
    generate_professional_report
)
import pandas as pd
import numpy as np


def create_mock_backtest_result(days=252, initial_capital=100000):
    """创建模拟回测结果"""
    np.random.seed(42)

    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')

    # 模拟权益曲线(有波动和回撤)
    returns = np.random.normal(0.001, 0.02, days)
    equity = initial_capital * (1 + returns).cumprod()
    equity_curve = pd.Series(equity, index=dates)

    # 模拟交易记录
    trades = []
    for i in range(20):
        entry_idx = np.random.randint(0, days-20)
        exit_idx = entry_idx + np.random.randint(5, 15)

        if exit_idx < days:
            pnl = np.random.normal(0, 1000)
            trades.append({
                'entry_date': dates[entry_idx].strftime('%Y-%m-%d'),
                'exit_date': dates[exit_idx].strftime('%Y-%m-%d'),
                'pnl': pnl,
                'amount': 10000,
            })

    return {
        'equity_curve': equity_curve,
        'trades': trades,
    }


def test_calculate_return_metrics():
    """测试1: 收益指标计算"""
    print("=" * 60)
    print("测试1: 收益指标计算")
    print("=" * 60)

    result = create_mock_backtest_result(days=252)
    reporter = ProfessionalBacktestReport(result)

    metrics = reporter._calculate_return_metrics()

    print(f"\n收益指标:")
    print(f"  总收益率: {metrics['总收益率']:.2%}")
    print(f"  年化收益率: {metrics['年化收益率']:.2%}")
    print(f"  CAGR: {metrics['CAGR']:.2%}")
    print(f"  累计最大收益: {metrics['累计最大收益']:.2%}")

    # 验证
    assert '总收益率' in metrics
    assert '年化收益率' in metrics
    assert 'CAGR' in metrics
    assert isinstance(metrics['总收益率'], (int, float))

    print("\n✅ 测试1通过!")
    return True


def test_calculate_risk_metrics():
    """测试2: 风险指标计算"""
    print("\n" + "=" * 60)
    print("测试2: 风险指标计算")
    print("=" * 60)

    result = create_mock_backtest_result(days=252)
    reporter = ProfessionalBacktestReport(result)

    metrics = reporter._calculate_risk_metrics()

    print(f"\n风险指标:")
    print(f"  年化波动率: {metrics['年化波动率']:.2%}")
    print(f"  最大回撤: {metrics['最大回撤']:.2%}")
    print(f"  VaR(95%): {metrics['VaR(95%)']:.2%}")
    print(f"  最长回撤期: {metrics['最长回撤期(天)']}天")

    # 验证
    assert metrics['年化波动率'] > 0, "波动率应>0"
    assert metrics['最大回撤'] < 0, "最大回撤应<0"
    assert metrics['最长回撤期(天)'] >= 0

    print("\n✅ 测试2通过!")
    return True


def test_calculate_risk_adjusted_metrics():
    """测试3: 风险调整收益指标"""
    print("\n" + "=" * 60)
    print("测试3: 风险调整收益指标")
    print("=" * 60)

    result = create_mock_backtest_result(days=252)
    reporter = ProfessionalBacktestReport(result)

    metrics = reporter._calculate_risk_adjusted_metrics()

    print(f"\n风险调整收益:")
    print(f"  夏普比率: {metrics['夏普比率']:.2f}")
    print(f"  Sortino比率: {metrics['Sortino比率']:.2f}")
    print(f"  Calmar比率: {metrics['Calmar比率']:.2f}")
    print(f"  Omega比率: {metrics['Omega比率']:.2f}")

    # 验证
    assert '夏普比率' in metrics
    assert 'Sortino比率' in metrics
    assert isinstance(metrics['夏普比率'], (int, float))

    print("\n✅ 测试3通过!")
    return True


def test_calculate_trade_metrics():
    """测试4: 交易指标计算"""
    print("\n" + "=" * 60)
    print("测试4: 交易指标计算")
    print("=" * 60)

    result = create_mock_backtest_result(days=252)
    reporter = ProfessionalBacktestReport(result)

    metrics = reporter._calculate_trade_metrics()

    print(f"\n交易指标:")
    print(f"  交易次数: {metrics['交易次数']}")
    print(f"  胜率: {metrics['胜率']:.2%}")
    print(f"  盈亏比: {metrics['盈亏比']:.2f}")
    print(f"  平均持仓天数: {metrics['平均持仓天数']:.1f}天")

    # 验证
    assert metrics['交易次数'] > 0, "应该有交易记录"
    assert 0 <= metrics['胜率'] <= 1, "胜率应在0-1之间"
    assert metrics['平均持仓天数'] > 0

    print("\n✅ 测试4通过!")
    return True


def test_calculate_all_metrics():
    """测试5: 计算所有30+个指标"""
    print("\n" + "=" * 60)
    print("测试5: 计算所有30+个指标")
    print("=" * 60)

    result = create_mock_backtest_result(days=252)
    reporter = ProfessionalBacktestReport(result)

    metrics = reporter.calculate_all_metrics()

    print(f"\n总指标数: {len(metrics)}")
    print(f"指标列表: {list(metrics.keys())[:10]}...")  # 显示前10个

    # 验证至少有25个指标
    assert len(metrics) >= 25, f"应该至少有25个指标,实际{len(metrics)}个"

    # 验证关键指标存在
    key_metrics = ['总收益率', '夏普比率', '最大回撤', '胜率', '交易次数']
    for key in key_metrics:
        assert key in metrics, f"缺少关键指标: {key}"

    print("\n✅ 测试5通过!")
    return True


def test_generate_monthly_returns_table():
    """测试6: 月度收益表"""
    print("\n" + "=" * 60)
    print("测试6: 月度收益表")
    print("=" * 60)

    # 创建1年半的数据
    result = create_mock_backtest_result(days=500)
    reporter = ProfessionalBacktestReport(result)

    monthly_table = reporter.generate_monthly_returns_table()

    print(f"\n月度收益表:")
    print(monthly_table)

    # 验证
    assert not monthly_table.empty, "月度收益表不应为空"
    assert '年度收益' in monthly_table.columns, "应包含年度收益列"

    print("\n✅ 测试6通过!")
    return True


def test_analyze_drawdowns():
    """测试7: 回撤分析"""
    print("\n" + "=" * 60)
    print("测试7: 回撤分析")
    print("=" * 60)

    result = create_mock_backtest_result(days=252)
    reporter = ProfessionalBacktestReport(result)

    drawdowns = reporter.analyze_drawdowns()

    print(f"\n回撤次数: {len(drawdowns)}")

    if drawdowns:
        print(f"\nTop 3 回撤:")
        for i, dd in enumerate(drawdowns[:3], 1):
            print(f"  #{i}: 幅度={dd['depth']:.2%}, 持续={dd['duration']}天")

    # 验证
    assert isinstance(drawdowns, list)
    if drawdowns:
        assert 'depth' in drawdowns[0]
        assert 'duration' in drawdowns[0]
        assert drawdowns[0]['depth'] < 0, "回撤深度应<0"

    print("\n✅ 测试7通过!")
    return True


def test_calculate_rolling_metrics():
    """测试8: 滚动指标"""
    print("\n" + "=" * 60)
    print("测试8: 滚动指标")
    print("=" * 60)

    result = create_mock_backtest_result(days=252)
    reporter = ProfessionalBacktestReport(result)

    rolling_metrics = reporter.calculate_rolling_metrics(window=63)

    print(f"\n滚动指标:")
    print(f"  指标列: {rolling_metrics.columns.tolist()}")
    print(f"  数据行数: {len(rolling_metrics)}")

    # 验证
    assert not rolling_metrics.empty
    assert '夏普比率' in rolling_metrics.columns
    assert '波动率' in rolling_metrics.columns

    print("\n✅ 测试8通过!")
    return True


def test_with_benchmark():
    """测试9: 基准对比"""
    print("\n" + "=" * 60)
    print("测试9: 基准对比")
    print("=" * 60)

    result = create_mock_backtest_result(days=252)

    # 创建基准数据
    dates = result['equity_curve'].index
    benchmark = pd.Series(
        100000 * (1 + np.random.normal(0.0005, 0.015, 252)).cumprod(),
        index=dates
    )

    reporter = ProfessionalBacktestReport(result, benchmark)

    metrics = reporter.calculate_all_metrics()

    print(f"\n基准对比指标:")
    print(f"  Alpha: {metrics['Alpha']:.2%}")
    print(f"  Beta: {metrics['Beta']:.2f}")
    print(f"  跟踪误差: {metrics['跟踪误差']:.2%}")
    print(f"  超额收益: {metrics['超额收益率']:.2%}")

    # 验证
    assert 'Alpha' in metrics
    assert 'Beta' in metrics
    assert isinstance(metrics['Beta'], (int, float))

    print("\n✅ 测试9通过!")
    return True


def test_generate_full_report():
    """测试10: 生成完整报告"""
    print("\n" + "=" * 60)
    print("测试10: 生成完整报告")
    print("=" * 60)

    result = create_mock_backtest_result(days=252)
    reporter = ProfessionalBacktestReport(result)

    report = reporter.generate_full_report()

    print("\n生成的报告:")
    print(report[:500] + "...")  # 显示前500字符

    # 验证
    assert len(report) > 0, "报告应非空"
    assert "专业回测报告" in report
    assert "执行摘要" in report
    assert "核心指标" in report
    assert "月度收益表" in report

    print("\n✅ 测试10通过!")
    return True


def test_quick_function():
    """测试11: 快速函数"""
    print("\n" + "=" * 60)
    print("测试11: 快速生成函数")
    print("=" * 60)

    result = create_mock_backtest_result(days=252)

    # 使用快速函数
    output = generate_professional_report(result)

    print(f"\n返回键: {list(output.keys())}")

    # 验证
    assert 'metrics' in output
    assert 'monthly_table' in output
    assert 'drawdowns' in output
    assert 'report' in output

    print(f"\n指标数: {len(output['metrics'])}")
    print(f"回撤次数: {len(output['drawdowns'])}")

    print("\n✅ 测试11通过!")
    return True


if __name__ == "__main__":
    try:
        # 设置随机种子
        np.random.seed(42)

        # 运行所有测试
        test_calculate_return_metrics()
        test_calculate_risk_metrics()
        test_calculate_risk_adjusted_metrics()
        test_calculate_trade_metrics()
        test_calculate_all_metrics()
        test_generate_monthly_returns_table()
        test_analyze_drawdowns()
        test_calculate_rolling_metrics()
        test_with_benchmark()
        test_generate_full_report()
        test_quick_function()

        print("\n" + "=" * 60)
        print("🎉 所有Phase 9.5 专业回测报告测试通过!")
        print("专业回测报告功能已准备就绪!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
