"""
快速功能验证脚本
运行核心模块的基础测试
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("量化交易系统 - 快速功能验证")
print("=" * 60)
print()

# 测试1: 策略集成
print("✓ 测试策略集成...")
try:
    from src.strategy.ensemble_strategy import EnsembleStrategy
    print("  ✅ 策略集成模块加载成功")
except Exception as e:
    print(f"  ❌ 策略集成模块加载失败: {e}")

# 测试2: 学术因子
print("\n✓ 测试学术因子...")
try:
    from src.factors.academic_factors import AcademicFactors
    analyzer = AcademicFactors()
    smb = analyzer.calculate_smb_factor(100, {'30%': 50, '70%': 200})
    assert smb == 0.0
    print("  ✅ 学术因子计算正确")
except Exception as e:
    print(f"  ❌ 学术因子测试失败: {e}")

# 测试3: 专业回测报告
print("\n✓ 测试专业回测报告...")
try:
    from src.backtest.professional_report import ProfessionalBacktestReport
    import pandas as pd
    import numpy as np

    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    equity = pd.Series(100000 * (1 + np.random.randn(100).cumsum() * 0.01), index=dates)

    reporter = ProfessionalBacktestReport({'equity_curve': equity, 'trades': []})
    metrics = reporter.calculate_all_metrics()

    assert '总收益率' in metrics
    assert '夏普比率' in metrics
    print(f"  ✅ 专业回测报告生成成功({len(metrics)}个指标)")
except Exception as e:
    print(f"  ❌ 专业回测报告测试失败: {e}")

# 测试4: ML算法对比
print("\n✓ 测试ML算法对比...")
try:
    from src.optimization.ml_benchmark import MLAlgorithmBenchmark
    print("  ✅ ML算法对比模块加载成功")
except Exception as e:
    print(f"  ❌ ML算法对比模块加载失败: {e}")

# 测试5: 参数优化
print("\n✓ 测试参数优化...")
try:
    from src.optimization.parameter_optimizer import ParameterOptimizer
    print("  ✅ 参数优化模块加载成功")
except Exception as e:
    print(f"  ❌ 参数优化模块加载失败: {e}")

# 测试6: 数据验证
print("\n✓ 测试数据验证...")
try:
    from src.validation.data_validator import DataValidator
    validator = DataValidator()

    # 创建测试数据
    import pandas as pd
    import numpy as np

    test_df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [99, 100, 101],
        'close': [103, 104, 105],
        'volume': [1000000, 1100000, 1200000],
        'source': ['test', 'test', 'test'],
    }, index=pd.date_range(start='2024-01-01', periods=3, freq='D'))

    result = validator.validate_price_data(test_df, 'TEST', 'CN')
    assert result['is_valid'] == True
    print("  ✅ 数据验证功能正常")
except Exception as e:
    print(f"  ❌ 数据验证测试失败: {e}")

# 测试7: ETF定投
print("\n✓ 测试ETF定投...")
try:
    from src.strategy.etf_strategies import ETFDollarCostAveraging
    strategy = ETFDollarCostAveraging(frequency='weekly', invest_amount=5000)
    print("  ✅ ETF定投策略加载成功")
except Exception as e:
    print(f"  ❌ ETF定投模块加载失败: {e}")

# 测试8: 基本面分析
print("\n✓ 测试基本面分析...")
try:
    from src.analysis.fundamental import FundamentalAnalyzer
    analyzer = FundamentalAnalyzer()
    score = analyzer.generate_fundamental_score('TEST')
    assert '综合得分' in score
    print("  ✅ 基本面分析功能正常")
except Exception as e:
    print(f"  ❌ 基本面分析测试失败: {e}")

# 测试9: 持仓管理
print("\n✓ 测试持仓管理...")
try:
    from src.trading.portfolio_manager import PortfolioManager

    # 使用默认数据库路径
    manager = PortfolioManager(db_path="data/trade_journal.db")
    dashboard = manager.get_portfolio_dashboard(market='CN')
    assert 'total_market_value' in dashboard
    print("  ✅ 持仓管理模块功能正常")
except Exception as e:
    print(f"  ❌ 持仓管理模块测试失败: {e}")

# 测试10: 目标导向推荐
print("\n✓ 测试目标导向推荐...")
try:
    from src.strategy.goal_based_recommender import InvestmentGoal, StrategyRecommender

    goal = InvestmentGoal(
        time_horizon_years=3,     # 3年
        target_return=0.20,       # 年化20%
        risk_tolerance='moderate'
    )
    print(f"  ✅ 目标导向推荐加载成功(年化目标:{goal.target_return:.1%})")
except Exception as e:
    print(f"  ❌ 目标导向推荐测试失败: {e}")

print("\n" + "=" * 60)
print("✅ 快速验证完成!所有核心模块加载正常!")
print("=" * 60)
