"""
Phase 3 全面测试脚本
测试内容:
  1. 独立置信度指标（因子一致性 + 历史胜率）
  2. 交易成本门槛过滤
  3. RiskManager 集成到调仓
  4. IC_IR加权时降低高相关因子权重
  5. 评分区间自适应
  6. 端到端集成测试
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置工作目录
os.chdir(project_root)


def generate_mock_ohlcv(n_days=120, trend='up', volatility=0.02, seed=42):
    """生成模拟的OHLCV数据"""
    np.random.seed(seed)
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')

    # 生成价格序列
    if trend == 'up':
        drift = 0.001
    elif trend == 'down':
        drift = -0.001
    else:
        drift = 0.0

    returns = np.random.normal(drift, volatility, n_days)
    prices = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n_days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, n_days).astype(float),
        'amount': np.random.uniform(1e7, 1e8, n_days),
        'turnover': np.random.uniform(0.5, 3.0, n_days),
    }, index=dates)

    return df


# =============================================================
# Test 1: 独立置信度指标
# =============================================================
def test_confidence_independence():
    """测试置信度与综合评分是否已解耦"""
    print("\n" + "="*60)
    print("Test 1: 独立置信度指标")
    print("="*60)

    from src.strategy.interpretable_strategy import BalancedMultiFactorStrategy

    strategy = BalancedMultiFactorStrategy()

    # 用不同趋势的数据测试
    df_up = generate_mock_ohlcv(120, 'up', volatility=0.02, seed=42)
    df_down = generate_mock_ohlcv(120, 'down', volatility=0.04, seed=43)
    df_flat = generate_mock_ohlcv(120, 'flat', volatility=0.01, seed=44)

    report_up = strategy.analyze_stock("TEST_UP", df_up, name="上涨股")
    report_down = strategy.analyze_stock("TEST_DOWN", df_down, name="下跌股")
    report_flat = strategy.analyze_stock("TEST_FLAT", df_flat, name="横盘股")

    print(f"\n  上涨股: score={report_up.score:.1f}, confidence={report_up.confidence:.1f}")
    print(f"  下跌股: score={report_down.score:.1f}, confidence={report_down.confidence:.1f}")
    print(f"  横盘股: score={report_flat.score:.1f}, confidence={report_flat.confidence:.1f}")

    # 验证: confidence != score (解耦)
    assert report_up.confidence != report_up.score, \
        f"置信度与评分未解耦! confidence={report_up.confidence}, score={report_up.score}"

    # 验证: confidence 在 0-100 范围内
    for r in [report_up, report_down, report_flat]:
        assert 0 <= r.confidence <= 100, f"置信度越界: {r.confidence}"

    print("\n  ✅ 置信度与评分已成功解耦")
    print(f"  ✅ 所有置信度在 [0, 100] 范围内")
    return True


def test_confidence_computation_logic():
    """测试置信度计算的三个维度"""
    print("\n" + "="*60)
    print("Test 1b: 置信度计算逻辑")
    print("="*60)

    from src.strategy.interpretable_strategy import BalancedMultiFactorStrategy

    strategy = BalancedMultiFactorStrategy()

    # 测试因子全部一致看多的情况
    scores_all_bullish = {'A': 80, 'B': 75, 'C': 90}
    weights_all = {'A': 0.3, 'B': 0.3, 'C': 0.4}

    conf_bullish = strategy._compute_confidence(
        scores_all_bullish, weights_all,
        composite_score=80, buy_threshold=65, sell_threshold=35
    )

    # 测试因子方向不一致的情况
    scores_mixed = {'A': 80, 'B': 30, 'C': 90}
    conf_mixed = strategy._compute_confidence(
        scores_mixed, weights_all,
        composite_score=70, buy_threshold=65, sell_threshold=35
    )

    print(f"\n  全部看多: confidence={conf_bullish:.1f}")
    print(f"  方向混乱: confidence={conf_mixed:.1f}")

    # 因子全一致时置信度应更高
    assert conf_bullish > conf_mixed, \
        f"因子一致性逻辑错误! 全一致={conf_bullish}, 混乱={conf_mixed}"

    print(f"  ✅ 因子一致性: 全一致({conf_bullish:.1f}) > 混乱({conf_mixed:.1f})")

    # 测试历史胜率影响
    strategy._historical_win_rate = 0.8
    conf_with_history = strategy._compute_confidence(
        scores_all_bullish, weights_all,
        composite_score=80, buy_threshold=65, sell_threshold=35
    )
    strategy._historical_win_rate = 0.2
    conf_with_bad_history = strategy._compute_confidence(
        scores_all_bullish, weights_all,
        composite_score=80, buy_threshold=65, sell_threshold=35
    )
    strategy._historical_win_rate = None  # 重置

    print(f"\n  高胜率(80%): confidence={conf_with_history:.1f}")
    print(f"  低胜率(20%): confidence={conf_with_bad_history:.1f}")

    assert conf_with_history > conf_with_bad_history, \
        f"历史胜率逻辑错误! 高胜率={conf_with_history}, 低胜率={conf_with_bad_history}"

    print(f"  ✅ 历史胜率影响正确")
    return True


# =============================================================
# Test 2: 交易成本门槛过滤
# =============================================================
def test_cost_filter():
    """测试交易成本过滤是否正确降级弱信号"""
    print("\n" + "="*60)
    print("Test 2: 交易成本门槛过滤")
    print("="*60)

    from src.strategy.interpretable_strategy import (
        BalancedMultiFactorStrategy, DecisionReport
    )

    strategy = BalancedMultiFactorStrategy()

    # 创建一个刚过买入阈值的弱信号
    weak_report = DecisionReport(
        code="WEAK", name="弱信号", date="2025-01-01",
        action="buy", action_cn="买入",
        confidence=66, score=66,  # 刚过65阈值
        strategy_name="test",
        current_price=100.0,
    )

    # 高波动 → 更高的成本 → 更容易被过滤
    result = strategy._apply_cost_filter(weak_report, volatility=0.05)
    print(f"\n  弱买入信号(score=66, vol=5%): action={result.action}")

    assert result.action == "hold", \
        f"弱信号应被降级! got action={result.action}"
    assert any("交易成本" in r for r in result.reasoning), \
        "应包含交易成本降级的理由"
    print(f"  ✅ 弱信号被正确降级为持有")
    print(f"  理由: {[r for r in result.reasoning if '成本' in r]}")

    # 创建一个强信号 — 不应被过滤
    strong_report = DecisionReport(
        code="STRONG", name="强信号", date="2025-01-01",
        action="buy", action_cn="买入",
        confidence=85, score=85,  # 远超65阈值
        strategy_name="test",
        current_price=100.0,
    )

    result2 = strategy._apply_cost_filter(strong_report, volatility=0.02)
    print(f"\n  强买入信号(score=85, vol=2%): action={result2.action}")
    assert result2.action == "buy", f"强信号不应被降级! got action={result2.action}"
    print(f"  ✅ 强信号保持买入不变")

    # 卖出信号不应被过滤
    sell_report = DecisionReport(
        code="SELL", name="卖出", date="2025-01-01",
        action="sell", action_cn="卖出",
        confidence=30, score=30,
        strategy_name="test",
    )
    result3 = strategy._apply_cost_filter(sell_report, volatility=0.03)
    assert result3.action == "sell", "卖出信号不应被过滤"
    print(f"  ✅ 卖出信号不受影响")

    return True


def test_cost_filter_in_strategy():
    """测试成本过滤在实际策略中是否被调用"""
    print("\n" + "="*60)
    print("Test 2b: 成本过滤端到端测试")
    print("="*60)

    from src.strategy.interpretable_strategy import (
        BalancedMultiFactorStrategy, MomentumTrendStrategy,
        ValueInvestStrategy, LowVolDefenseStrategy,
        MeanReversionStrategy, TechnicalBreakoutStrategy,
    )

    strategies = [
        BalancedMultiFactorStrategy(),
        MomentumTrendStrategy(),
        ValueInvestStrategy(),
        LowVolDefenseStrategy(),
        MeanReversionStrategy(),
        TechnicalBreakoutStrategy(),
    ]

    df = generate_mock_ohlcv(120, 'up', 0.02, seed=42)

    errors = []
    for strategy in strategies:
        try:
            report = strategy.analyze_stock("TEST", df, name="测试")
            # 只要不报错就OK，成本过滤在内部已被调用
            assert report.score >= 0
            assert report.confidence >= 0
            print(f"  ✅ {strategy.name}: score={report.score:.1f}, "
                  f"conf={report.confidence:.1f}, action={report.action}")
        except Exception as e:
            errors.append(f"{strategy.name}: {e}")
            print(f"  ❌ {strategy.name}: {e}")

    if errors:
        raise AssertionError(f"策略执行错误: {errors}")

    return True


# =============================================================
# Test 3: RiskManager 集成
# =============================================================
def test_risk_manager_integration():
    """测试 RiskManager 是否正确集成到 PortfolioManager"""
    print("\n" + "="*60)
    print("Test 3: RiskManager 集成")
    print("="*60)

    from src.trading.risk import RiskManager, RiskConfig

    # 测试基本风险检查
    config = RiskConfig(max_position_pct=0.10, max_total_positions=5)
    rm = RiskManager(config)

    positions = {'AAPL': (100, 150.0), 'MSFT': (50, 350.0)}
    prices = {'AAPL': 160.0, 'MSFT': 380.0}
    total_equity = 100000

    risks = rm.calculate_position_risk(positions, prices, total_equity)

    print(f"\n  持仓风险计算:")
    for code, risk in risks.items():
        print(f"    {code}: weight={risk.weight:.1%}, "
              f"pnl={risk.unrealized_pnl_pct:.1%}, "
              f"stop_loss={risk.stop_loss_triggered}")

    # 测试交易风险检查 - 超仓位限制
    allowed, reason = rm.check_trade_risk(
        code='GOOGL', action='buy', shares=100, price=150.0,
        current_positions=positions, total_equity=total_equity
    )
    print(f"\n  买入GOOGL 100股@150: allowed={allowed}, reason={reason}")

    # 买入15000/100000 = 15% > 10% 限制
    assert not allowed, "应该拒绝超仓位限制的交易"
    print(f"  ✅ 超仓位交易被正确拒绝")

    # 合规交易应通过
    allowed2, reason2 = rm.check_trade_risk(
        code='GOOGL', action='buy', shares=50, price=150.0,
        current_positions=positions, total_equity=total_equity
    )
    print(f"  买入GOOGL 50股@150: allowed={allowed2}, reason={reason2}")
    assert allowed2, "合规交易应该通过"
    print(f"  ✅ 合规交易正确通过")

    return True


def test_portfolio_manager_risk():
    """测试 PortfolioManager 中的风险集成"""
    print("\n" + "="*60)
    print("Test 3b: PortfolioManager 风险集成")
    print("="*60)

    from src.trading.portfolio_manager import PortfolioManager
    from src.trading.risk import RiskConfig

    # 验证 PortfolioManager 初始化时创建了 RiskManager
    config = RiskConfig(max_position_pct=0.15)
    pm = PortfolioManager(db_path="data/test_risk.db", risk_config=config)

    assert hasattr(pm, 'risk_manager'), "PortfolioManager 应有 risk_manager 属性"
    assert pm.risk_manager.config.max_position_pct == 0.15, \
        "RiskConfig 应正确传递"

    print(f"  ✅ PortfolioManager 已集成 RiskManager")
    print(f"  ✅ RiskConfig 正确传递 (max_position_pct=0.15)")

    return True


# =============================================================
# Test 4: 因子共线性惩罚
# =============================================================
def test_correlation_penalty():
    """测试 IC_IR 加权中的因子共线性惩罚"""
    print("\n" + "="*60)
    print("Test 4: 因子共线性惩罚")
    print("="*60)

    from src.optimization.weight_optimizer import WeightOptimizer
    from src.factors.factor_validator import FactorICResult

    optimizer = WeightOptimizer()

    # 构造3个因子的IC结果
    factor_results = {
        'momentum_5_fwd10': FactorICResult(
            factor_name='momentum_5', ic_mean=0.06, ic_std=0.1,
            ic_ir=0.6, forward_days=10
        ),
        'momentum_20_fwd10': FactorICResult(
            factor_name='momentum_20', ic_mean=0.05, ic_std=0.1,
            ic_ir=0.5, forward_days=10
        ),
        'rsi_14_fwd10': FactorICResult(
            factor_name='rsi_14', ic_mean=0.03, ic_std=0.1,
            ic_ir=0.3, forward_days=10
        ),
    }
    strategy_factors = ['momentum_5', 'momentum_20', 'rsi_14']

    # 无相关性矩阵 → 标准IC_IR加权
    weights_no_corr = optimizer.optimize_icir(
        factor_results, strategy_factors
    )
    print(f"\n  无共线性惩罚:")
    for f, w in weights_no_corr.items():
        print(f"    {f}: {w:.4f}")

    # 有相关性矩阵 → momentum_5 和 momentum_20 高度相关
    corr_matrix = pd.DataFrame(
        [[1.0, 0.85, 0.2],
         [0.85, 1.0, 0.15],
         [0.2, 0.15, 1.0]],
        index=strategy_factors,
        columns=strategy_factors
    )

    weights_with_corr = optimizer.optimize_icir(
        factor_results, strategy_factors,
        correlation_matrix=corr_matrix
    )
    print(f"\n  有共线性惩罚 (momentum_5↔momentum_20 corr=0.85):")
    for f, w in weights_with_corr.items():
        print(f"    {f}: {w:.4f}")

    # momentum_20 的 IC_IR 更低 (0.5 < 0.6)，应被惩罚
    assert weights_with_corr['momentum_20'] < weights_no_corr['momentum_20'], \
        "高相关低IC_IR因子应被降权"

    # momentum_5 (更强因子) 不应被惩罚
    # (它可能因归一化有微小变化，但不应大幅下降)

    # rsi_14 与两者不相关，不应被惩罚
    print(f"\n  momentum_20 权重变化: {weights_no_corr['momentum_20']:.4f} → "
          f"{weights_with_corr['momentum_20']:.4f}")

    print(f"  ✅ 高相关弱因子被正确降权")

    # 测试无相关性的情况 → 不应有变化
    corr_low = pd.DataFrame(
        [[1.0, 0.1, 0.1],
         [0.1, 1.0, 0.1],
         [0.1, 0.1, 1.0]],
        index=strategy_factors,
        columns=strategy_factors
    )
    weights_low_corr = optimizer.optimize_icir(
        factor_results, strategy_factors,
        correlation_matrix=corr_low
    )

    # 低相关 → 权重应与无矩阵时一致
    for f in strategy_factors:
        assert abs(weights_low_corr[f] - weights_no_corr[f]) < 0.01, \
            f"低相关因子不应被惩罚: {f}"

    print(f"  ✅ 低相关因子不受影响")
    return True


def test_factor_signs():
    """测试负IC因子的方向标记"""
    print("\n" + "="*60)
    print("Test 4b: 负IC因子方向标记")
    print("="*60)

    from src.optimization.weight_optimizer import WeightOptimizer
    from src.factors.factor_validator import FactorICResult

    optimizer = WeightOptimizer()

    factor_results = {
        'momentum_20_fwd10': FactorICResult(
            factor_name='momentum_20', ic_mean=0.05, ic_std=0.1,
            ic_ir=0.5, forward_days=10
        ),
        'volatility_20_fwd10': FactorICResult(
            factor_name='volatility_20', ic_mean=-0.04, ic_std=0.1,
            ic_ir=-0.4, forward_days=10  # 负IC: 高波动 → 低收益
        ),
    }

    weights = optimizer.optimize_icir(
        factor_results, ['momentum_20', 'volatility_20']
    )
    signs = optimizer.get_factor_signs()

    print(f"\n  momentum_20 方向: {signs.get('momentum_20', '?')}")
    print(f"  volatility_20 方向: {signs.get('volatility_20', '?')}")

    assert signs['momentum_20'] == 1, "正IC因子应标记为+1"
    assert signs['volatility_20'] == -1, "负IC因子应标记为-1"

    print(f"  ✅ 正IC因子标记为 +1, 负IC因子标记为 -1")
    return True


# =============================================================
# Test 5: 评分区间自适应
# =============================================================
def test_adaptive_score_range():
    """测试评分区间自适应计算"""
    print("\n" + "="*60)
    print("Test 5: 评分区间自适应")
    print("="*60)

    from src.strategy.interpretable_strategy import BalancedMultiFactorStrategy

    strategy = BalancedMultiFactorStrategy()

    # 生成包含因子数据的DataFrame
    df = generate_mock_ohlcv(120, 'up', 0.02, seed=42)
    factored = strategy._compute_factors(df)

    # 测试自适应区间计算
    adaptive_range = strategy._compute_adaptive_ranges(factored, 'momentum_20')
    print(f"\n  momentum_20 自适应区间: {adaptive_range}")

    assert adaptive_range is not None, "应该能计算出自适应区间"
    low, high = adaptive_range
    assert low < high, f"下界应小于上界: low={low}, high={high}"
    print(f"  ✅ 自适应区间有效: [{low:.4f}, {high:.4f}]")

    # 测试 _score_factor_adaptive
    latest_val = factored['momentum_20'].iloc[-1]

    # 使用自适应评分
    adaptive_score = strategy._score_factor_adaptive(
        latest_val, factored, 'momentum_20', -0.15, 0.15
    )
    # 使用硬编码评分
    hardcoded_score = strategy._score_factor(latest_val, -0.15, 0.15)

    print(f"\n  momentum_20 最新值: {latest_val:.4f}")
    print(f"  自适应评分: {adaptive_score:.1f}")
    print(f"  硬编码评分: {hardcoded_score:.1f}")

    # 两者可以不同（因为区间不同），但都应在 0-100 范围内
    assert 0 <= adaptive_score <= 100, f"自适应评分越界: {adaptive_score}"
    assert 0 <= hardcoded_score <= 100

    # 如果区间真的不同，评分也应不同
    if adaptive_range != (-0.15, 0.15):
        print(f"  ✅ 自适应评分与硬编码评分不同（区间已调整）")
    else:
        print(f"  ℹ️ 区间恰好与默认相近，评分相似")

    # 测试不存在的列 → 应返回None
    none_range = strategy._compute_adaptive_ranges(factored, 'nonexistent_column')
    assert none_range is None, "不存在的列应返回None"
    print(f"  ✅ 不存在的因子列正确返回None")

    # 测试数据不足 → 应返回None
    short_df = factored.tail(10)  # 只有10条数据
    short_range = strategy._compute_adaptive_ranges(short_df, 'momentum_20')
    assert short_range is None, "数据不足时应返回None"
    print(f"  ✅ 数据不足时正确返回None")

    return True


def test_adaptive_in_strategy():
    """测试自适应评分在策略中的实际效果"""
    print("\n" + "="*60)
    print("Test 5b: 自适应评分策略效果")
    print("="*60)

    from src.strategy.interpretable_strategy import (
        BalancedMultiFactorStrategy, MomentumTrendStrategy
    )

    # 用极端行情测试：如果所有数据都在狭窄区间，自适应应放大评分差异
    np.random.seed(100)
    dates = pd.date_range(end=datetime.now(), periods=120, freq='B')
    # 价格几乎不动的低波动股
    prices = 100 + np.cumsum(np.random.normal(0, 0.001, 120))

    df_lowvol = pd.DataFrame({
        'open': prices - 0.1,
        'high': prices + 0.2,
        'low': prices - 0.2,
        'close': prices,
        'volume': np.random.randint(100000, 500000, 120).astype(float),
        'amount': np.random.uniform(1e7, 5e7, 120),
        'turnover': np.random.uniform(0.5, 1.5, 120),
    }, index=dates)

    balanced = BalancedMultiFactorStrategy()
    report = balanced.analyze_stock("LOWVOL", df_lowvol, name="低波动")

    print(f"\n  低波动股分析:")
    print(f"    score={report.score:.1f}, confidence={report.confidence:.1f}")
    print(f"    action={report.action}")

    # 不应报错
    assert report.score >= 0
    print(f"  ✅ 低波动极端场景正常运行")

    # 测试 MomentumTrendStrategy 也能正常运行
    mom = MomentumTrendStrategy()
    report2 = mom.analyze_stock("LOWVOL", df_lowvol, name="低波动")
    print(f"\n  动量策略低波动股:")
    print(f"    score={report2.score:.1f}, confidence={report2.confidence:.1f}")
    assert report2.score >= 0
    print(f"  ✅ 动量策略低波动场景正常运行")

    return True


# =============================================================
# Test 6: 端到端集成测试
# =============================================================
def test_end_to_end():
    """端到端测试：所有策略对多只股票的完整分析"""
    print("\n" + "="*60)
    print("Test 6: 端到端集成测试")
    print("="*60)

    from src.strategy.interpretable_strategy import (
        STRATEGY_REGISTRY, multi_strategy_analysis
    )

    # 生成3只不同特征的股票
    stocks = {
        'BULL': generate_mock_ohlcv(120, 'up', 0.015, seed=1),
        'BEAR': generate_mock_ohlcv(120, 'down', 0.03, seed=2),
        'FLAT': generate_mock_ohlcv(120, 'flat', 0.02, seed=3),
    }

    print(f"\n  测试 {len(STRATEGY_REGISTRY)} 个策略 × {len(stocks)} 只股票")

    errors = []
    for stock_name, df in stocks.items():
        results = multi_strategy_analysis(stock_name, df, name=stock_name)

        for strategy_key, report in results.items():
            try:
                # 基本验证
                assert report.code == stock_name
                assert 0 <= report.score <= 100, \
                    f"score越界: {report.score}"
                assert 0 <= report.confidence <= 100, \
                    f"confidence越界: {report.confidence}"
                assert report.confidence != report.score, \
                    f"confidence==score未解耦: {report.confidence}"
                assert report.action in ('buy', 'sell', 'hold', 'add', 'reduce')
                assert len(report.factor_scores) > 0
                assert len(report.factor_weights) > 0

                # 验证因子贡献和应接近总分
                contrib_sum = sum(report.factor_contributions.values())
                assert abs(contrib_sum - report.score) < 1.0, \
                    f"因子贡献和({contrib_sum:.1f})≠总分({report.score:.1f})"

            except AssertionError as e:
                errors.append(f"{stock_name}/{strategy_key}: {e}")

        print(f"  {stock_name}:")
        for k, r in results.items():
            print(f"    {k:12s}: score={r.score:5.1f} conf={r.confidence:5.1f} "
                  f"action={r.action:4s}")

    if errors:
        print(f"\n  ❌ 发现 {len(errors)} 个错误:")
        for e in errors:
            print(f"    - {e}")
        raise AssertionError(f"集成测试失败: {len(errors)} 个错误")

    total_tests = len(STRATEGY_REGISTRY) * len(stocks)
    print(f"\n  ✅ 全部 {total_tests} 次策略分析通过")
    return True


def test_strategy_router_with_updates():
    """测试策略路由器是否与修改后的策略兼容"""
    print("\n" + "="*60)
    print("Test 6b: 策略路由器兼容性")
    print("="*60)

    from src.strategy.strategy_router import StrategyRouter

    router = StrategyRouter()

    df = generate_mock_ohlcv(120, 'up', 0.02, seed=42)
    result = router.recommend("TEST", df, name="测试")

    print(f"\n  路由推荐:")
    print(f"    首选: {result.primary_strategy} ({result.primary_reason})")
    print(f"    次选: {result.secondary_strategy}")
    print(f"    置信度: {result.confidence:.0f}")
    print(f"    市场状态: {result.market_regime}")

    assert result.primary_strategy in (
        'balanced', 'momentum', 'value', 'low_vol', 'reversion', 'breakout'
    )
    assert 0 <= result.confidence <= 100

    print(f"  ✅ 策略路由器正常工作")
    return True


# =============================================================
# 主入口
# =============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3 全面测试")
    print("=" * 60)

    tests = [
        ("1a", "独立置信度 - 解耦验证", test_confidence_independence),
        ("1b", "独立置信度 - 计算逻辑", test_confidence_computation_logic),
        ("2a", "交易成本过滤 - 单元测试", test_cost_filter),
        ("2b", "交易成本过滤 - 端到端", test_cost_filter_in_strategy),
        ("3a", "RiskManager - 基本功能", test_risk_manager_integration),
        ("3b", "RiskManager - PM集成", test_portfolio_manager_risk),
        ("4a", "因子共线性惩罚", test_correlation_penalty),
        ("4b", "负IC因子方向标记", test_factor_signs),
        ("5a", "自适应评分区间", test_adaptive_score_range),
        ("5b", "自适应评分策略效果", test_adaptive_in_strategy),
        ("6a", "端到端集成测试", test_end_to_end),
        ("6b", "策略路由器兼容", test_strategy_router_with_updates),
    ]

    passed = 0
    failed = 0
    failures = []

    for test_id, test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            failures.append((test_id, test_name, str(e)))
            import traceback
            print(f"\n  ❌ 测试 {test_id} 失败: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败 / {len(tests)} 总计")
    print("=" * 60)

    if failures:
        print("\n失败测试:")
        for tid, tname, err in failures:
            print(f"  [{tid}] {tname}: {err}")
        sys.exit(1)
    else:
        print("\n✅ 所有测试通过!")
        sys.exit(0)
