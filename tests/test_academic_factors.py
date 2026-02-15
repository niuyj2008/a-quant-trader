"""
测试业界标杆因子 - Phase 9.4
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.factors.academic_factors import AcademicFactors, quick_academic_analysis
import pandas as pd
import numpy as np


def create_mock_stock_data(days=300, start_price=100):
    """创建模拟股票数据"""
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')

    # 生成价格(带趋势)
    trend = np.linspace(0, 20, days)
    noise = np.random.randn(days) * 2
    prices = start_price + trend + noise.cumsum()

    df = pd.DataFrame({
        'close': prices,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'volume': np.random.randint(1000000, 10000000, days),
    }, index=dates)

    return df


def create_mock_market_data(n_stocks=100):
    """创建模拟市场数据"""
    df = pd.DataFrame({
        'market_cap': np.random.uniform(50, 500, n_stocks),  # 市值50-500亿
        'pb': np.random.uniform(1, 10, n_stocks),             # PB 1-10
        'return': np.random.normal(0.001, 0.02, n_stocks),    # 收益率
    })
    return df


def test_fama_french_smb_factor():
    """测试1: SMB规模因子"""
    print("=" * 60)
    print("测试1: SMB规模因子")
    print("=" * 60)

    analyzer = AcademicFactors()

    # 市值分位数(30%=50亿, 70%=200亿)
    percentiles = {'30%': 50, '70%': 200}

    # 测试小盘股
    smb_small = analyzer.calculate_smb_factor(30, percentiles)
    print(f"\n小盘股(30亿): SMB={smb_small}")
    assert smb_small == 1.0, "小盘股SMB应为1.0"

    # 测试大盘股
    smb_large = analyzer.calculate_smb_factor(300, percentiles)
    print(f"大盘股(300亿): SMB={smb_large}")
    assert smb_large == -1.0, "大盘股SMB应为-1.0"

    # 测试中盘股
    smb_mid = analyzer.calculate_smb_factor(100, percentiles)
    print(f"中盘股(100亿): SMB={smb_mid}")
    assert smb_mid == 0.0, "中盘股SMB应为0.0"

    print("\n✅ 测试1通过!")
    return True


def test_fama_french_hml_factor():
    """测试2: HML价值因子"""
    print("\n" + "=" * 60)
    print("测试2: HML价值因子")
    print("=" * 60)

    analyzer = AcademicFactors()

    # PB分位数(30%=2.0, 70%=5.0)
    percentiles = {'30%': 2.0, '70%': 5.0}

    # 测试价值股(低PB)
    hml_value = analyzer.calculate_hml_factor(1.5, percentiles)
    print(f"\n价值股(PB=1.5): HML={hml_value}")
    assert hml_value == 1.0, "低PB股票HML应为1.0"

    # 测试成长股(高PB)
    hml_growth = analyzer.calculate_hml_factor(8.0, percentiles)
    print(f"成长股(PB=8.0): HML={hml_growth}")
    assert hml_growth == -1.0, "高PB股票HML应为-1.0"

    # 测试中性股
    hml_neutral = analyzer.calculate_hml_factor(3.5, percentiles)
    print(f"中性股(PB=3.5): HML={hml_neutral}")
    assert hml_neutral == 0.0, "中等PB股票HML应为0.0"

    print("\n✅ 测试2通过!")
    return True


def test_fama_french_three_factors():
    """测试3: Fama-French三因子组合"""
    print("\n" + "=" * 60)
    print("测试3: Fama-French三因子组合")
    print("=" * 60)

    analyzer = AcademicFactors()

    # 准备数据
    stock_data = pd.Series({
        'return': 0.02,      # 2%收益
        'market_cap': 80,    # 80亿市值(小盘)
        'pb': 2.5,           # 2.5倍PB(价值)
    })

    market_data = create_mock_market_data(100)

    # 计算三因子
    ff3 = analyzer.calculate_fama_french_three_factors(stock_data, market_data)

    print(f"\n三因子结果:")
    print(f"  MKT(市场因子): {ff3['MKT']:.4f}")
    print(f"  SMB(规模因子): {ff3['SMB']:.2f}")
    print(f"  HML(价值因子): {ff3['HML']:.2f}")
    print(f"  综合得分: {ff3['ff3_score']:.2f}")

    # 验证
    assert 'MKT' in ff3, "应包含MKT因子"
    assert 'SMB' in ff3, "应包含SMB因子"
    assert 'HML' in ff3, "应包含HML因子"
    assert 'ff3_score' in ff3, "应包含综合得分"

    print("\n✅ 测试3通过!")
    return True


def test_momentum_factor():
    """测试4: 动量因子"""
    print("\n" + "=" * 60)
    print("测试4: 动量因子")
    print("=" * 60)

    analyzer = AcademicFactors()

    # 创建上涨趋势的股票
    df_uptrend = create_mock_stock_data(days=300, start_price=100)
    df_uptrend['close'] = 100 + np.linspace(0, 50, 300)  # 线性上涨

    momentum = analyzer.calculate_momentum_factor(df_uptrend, lookback=252, skip=21)

    print(f"\n上涨趋势股票:")
    print(f"  起始价: {df_uptrend['close'].iloc[0]:.2f}")
    print(f"  最新价: {df_uptrend['close'].iloc[-1]:.2f}")
    print(f"  动量因子: {momentum.iloc[-1]:.4f}")

    # 验证: 上涨股票应有正动量
    assert momentum.iloc[-1] > 0, "上涨趋势应有正动量"

    # 创建下跌趋势的股票
    df_downtrend = create_mock_stock_data(days=300, start_price=100)
    df_downtrend['close'] = 100 - np.linspace(0, 30, 300)  # 线性下跌

    momentum_down = analyzer.calculate_momentum_factor(df_downtrend, lookback=252, skip=21)

    print(f"\n下跌趋势股票:")
    print(f"  起始价: {df_downtrend['close'].iloc[0]:.2f}")
    print(f"  最新价: {df_downtrend['close'].iloc[-1]:.2f}")
    print(f"  动量因子: {momentum_down.iloc[-1]:.4f}")

    # 验证: 下跌股票应有负动量
    assert momentum_down.iloc[-1] < 0, "下跌趋势应有负动量"

    print("\n✅ 测试4通过!")
    return True


def test_reversal_factor():
    """测试5: 短期反转因子"""
    print("\n" + "=" * 60)
    print("测试5: 短期反转因子")
    print("=" * 60)

    analyzer = AcademicFactors()

    # 创建短期快速上涨的股票
    df = create_mock_stock_data(days=100)
    df['close'].iloc[-21:] = df['close'].iloc[-22] * np.linspace(1, 1.2, 21)  # 最近1个月涨20%

    reversal = analyzer.calculate_reversal_factor(df, lookback=21)

    print(f"\n最近1个月涨幅: {(df['close'].iloc[-1] / df['close'].iloc[-22] - 1):.2%}")
    print(f"反转因子: {reversal.iloc[-1]:.4f}")

    # 验证: 快速上涨后,反转因子为负(预期回调)
    assert reversal.iloc[-1] < 0, "快速上涨后应预期反转(负值)"

    print("\n✅ 测试5通过!")
    return True


def test_quality_factor():
    """测试6: 质量因子"""
    print("\n" + "=" * 60)
    print("测试6: 质量因子")
    print("=" * 60)

    analyzer = AcademicFactors()

    # 高质量公司
    high_quality = {
        'roe': 0.20,           # 高ROE
        'roe_std': 0.02,       # 低波动
        'asset_growth': 0.10,  # 适度增长
        'gross_margin': 0.40,  # 高毛利
    }

    quality_high = analyzer.calculate_quality_factor(high_quality)
    print(f"\n高质量公司:")
    print(f"  ROE: {high_quality['roe']:.1%}")
    print(f"  质量得分: {quality_high:.1f}/100")

    assert quality_high >= 70, "高质量公司得分应>=70"

    # 低质量公司
    low_quality = {
        'roe': 0.05,           # 低ROE
        'roe_std': 0.08,       # 高波动
        'asset_growth': 0.40,  # 激进扩张
        'gross_margin': 0.15,
    }

    quality_low = analyzer.calculate_quality_factor(low_quality)
    print(f"\n低质量公司:")
    print(f"  ROE: {low_quality['roe']:.1%}")
    print(f"  质量得分: {quality_low:.1f}/100")

    assert quality_low < 50, "低质量公司得分应<50"

    print("\n✅ 测试6通过!")
    return True


def test_low_volatility_factor():
    """测试7: 低波动率因子"""
    print("\n" + "=" * 60)
    print("测试7: 低波动率因子")
    print("=" * 60)

    analyzer = AcademicFactors()

    # 低波动股票(稳定)
    df_stable = pd.DataFrame({
        'close': 100 + np.random.randn(100) * 0.5,  # 低波动
    }, index=pd.date_range(start='2024-01-01', periods=100, freq='D'))

    low_vol = analyzer.calculate_low_volatility_factor(df_stable, period=60)

    print(f"\n低波动股票:")
    print(f"  波动率: {df_stable['close'].pct_change().std():.4f}")
    print(f"  低波因子: {low_vol.iloc[-1]:.4f}")

    # 高波动股票
    df_volatile = pd.DataFrame({
        'close': 100 + np.random.randn(100) * 5,  # 高波动
    }, index=pd.date_range(start='2024-01-01', periods=100, freq='D'))

    high_vol = analyzer.calculate_low_volatility_factor(df_volatile, period=60)

    print(f"\n高波动股票:")
    print(f"  波动率: {df_volatile['close'].pct_change().std():.4f}")
    print(f"  低波因子: {high_vol.iloc[-1]:.4f}")

    # 验证: 低波动股票的低波因子应更高(负的负值)
    assert low_vol.iloc[-1] > high_vol.iloc[-1], "低波动股票的低波因子应高于高波动股票"

    print("\n✅ 测试7通过!")
    return True


def test_beta_calculation():
    """测试8: Beta系数"""
    print("\n" + "=" * 60)
    print("测试8: Beta系数")
    print("=" * 60)

    analyzer = AcademicFactors()

    # 创建市场收益
    np.random.seed(42)
    market_returns = pd.Series(np.random.normal(0.001, 0.02, 252))

    # 高Beta股票(波动是市场的1.5倍)
    stock_high_beta = market_returns * 1.5 + np.random.normal(0, 0.01, 252)

    beta_high = analyzer.calculate_beta(stock_high_beta, market_returns)

    print(f"\n高Beta股票:")
    print(f"  Beta: {beta_high:.2f}")
    print(f"  预期: ~1.5")

    assert beta_high > 1.0, "高Beta股票Beta应>1"

    # 低Beta股票(波动是市场的0.5倍)
    stock_low_beta = market_returns * 0.5 + np.random.normal(0, 0.005, 252)

    beta_low = analyzer.calculate_beta(stock_low_beta, market_returns)

    print(f"\n低Beta股票:")
    print(f"  Beta: {beta_low:.2f}")
    print(f"  预期: ~0.5")

    assert beta_low < 1.0, "低Beta股票Beta应<1"

    print("\n✅ 测试8通过!")
    return True


def test_comprehensive_score():
    """测试9: 综合评分"""
    print("\n" + "=" * 60)
    print("测试9: 综合学术因子评分")
    print("=" * 60)

    analyzer = AcademicFactors()

    # 准备数据
    stock_data = create_mock_stock_data(days=300, start_price=100)
    market_data = create_mock_market_data(100)

    financial_data = {
        'market_cap': 150,
        'pb': 3.5,
        'roe': 0.15,
        'roe_std': 0.03,
        'asset_growth': 0.10,
        'gross_margin': 0.35,
    }

    # 计算综合评分
    scores = analyzer.calculate_comprehensive_score(
        stock_data, market_data, financial_data
    )

    print(f"\n综合评分结果:")
    print(f"  Fama-French得分: {scores['ff3_score']:.1f}/30")
    print(f"  动量得分: {scores['momentum_score']:.1f}/20")
    print(f"  质量得分: {scores['quality_score']:.1f}/30")
    print(f"  低波得分: {scores['low_vol_score']:.1f}/20")
    print(f"  总分: {scores['total_score']:.1f}/100")
    print(f"  评级: {scores['rank']}")

    # 验证
    assert 'total_score' in scores, "应包含总分"
    assert 'rank' in scores, "应包含评级"
    assert 0 <= scores['total_score'] <= 100, "总分应在0-100之间"
    assert scores['rank'] in ['A+', 'A', 'B', 'C', 'D'], "评级应为A+/A/B/C/D"

    print("\n✅ 测试9通过!")
    return True


def test_generate_factor_report():
    """测试10: 生成因子报告"""
    print("\n" + "=" * 60)
    print("测试10: 生成因子报告")
    print("=" * 60)

    analyzer = AcademicFactors()

    stock_data = create_mock_stock_data(days=300)
    market_data = create_mock_market_data(100)

    financial_data = {
        'market_cap': 150,
        'pb': 3.5,
        'roe': 0.15,
        'roe_std': 0.03,
        'asset_growth': 0.10,
        'gross_margin': 0.35,
    }

    scores = analyzer.calculate_comprehensive_score(
        stock_data, market_data, financial_data
    )

    # 生成报告
    report = analyzer.generate_factor_report(scores)

    print("\n生成的报告:")
    print(report)

    # 验证
    assert len(report) > 0, "报告应非空"
    assert "学术因子分析报告" in report, "应包含标题"
    assert "Fama-French三因子" in report, "应包含FF三因子"
    assert "动量因子" in report, "应包含动量因子"
    assert "质量因子" in report, "应包含质量因子"
    assert "综合评分" in report, "应包含综合评分"

    print("\n✅ 测试10通过!")
    return True


def test_quick_academic_analysis():
    """测试11: 快速分析函数"""
    print("\n" + "=" * 60)
    print("测试11: 快速学术因子分析")
    print("=" * 60)

    stock_data = create_mock_stock_data(days=300)
    market_data = create_mock_market_data(100)

    financial_data = {
        'market_cap': 150,
        'pb': 3.5,
        'roe': 0.15,
        'roe_std': 0.03,
        'asset_growth': 0.10,
        'gross_margin': 0.35,
    }

    # 使用快速分析函数
    result = quick_academic_analysis(stock_data, market_data, financial_data)

    print(f"\n快速分析完成!")
    print(f"  返回键: {list(result.keys())[:8]}...")  # 显示前8个键

    # 验证
    assert 'total_score' in result, "应返回总分"
    assert 'rank' in result, "应返回评级"
    assert 'report' in result, "应返回报告"

    print("\n" + result['report'])

    print("\n✅ 测试11通过!")
    return True


if __name__ == "__main__":
    try:
        # 设置随机种子确保可复现
        np.random.seed(42)

        # 运行所有测试
        test_fama_french_smb_factor()
        test_fama_french_hml_factor()
        test_fama_french_three_factors()
        test_momentum_factor()
        test_reversal_factor()
        test_quality_factor()
        test_low_volatility_factor()
        test_beta_calculation()
        test_comprehensive_score()
        test_generate_factor_report()
        test_quick_academic_analysis()

        print("\n" + "=" * 60)
        print("🎉 所有Phase 9.4 业界标杆因子测试通过!")
        print("学术因子库已准备就绪!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
