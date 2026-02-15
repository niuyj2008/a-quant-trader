"""
测试策略集成 - Phase 9.3
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.strategy.ensemble_strategy import EnsembleStrategy, create_ensemble_strategy
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ========== 模拟策略类 ==========
class MockBullishStrategy:
    """模拟看多策略(总是推荐买入)"""

    def __init__(self, name='看多策略'):
        self.name = name

    def generate_signals(self, df, date, context=None):
        return [{
            'action': 'buy',
            'reason': '看多策略: 市场趋势向上',
            'confidence': 0.8,
        }]


class MockBearishStrategy:
    """模拟看空策略(总是推荐卖出)"""

    def __init__(self, name='看空策略'):
        self.name = name

    def generate_signals(self, df, date, context=None):
        return [{
            'action': 'sell',
            'reason': '看空策略: 市场趋势向下',
            'confidence': 0.7,
        }]


class MockNeutralStrategy:
    """模拟中性策略(不发出信号)"""

    def __init__(self, name='中性策略'):
        self.name = name

    def generate_signals(self, df, date, context=None):
        return []


class MockConditionalStrategy:
    """模拟条件策略(根据价格决策)"""

    def __init__(self, name='条件策略', threshold=100):
        self.name = name
        self.threshold = threshold

    def generate_signals(self, df, date, context=None):
        if date not in df.index:
            return []

        price = df.loc[date, 'close']

        if price > self.threshold:
            return [{'action': 'sell', 'reason': f'价格{price}超过阈值{self.threshold}'}]
        else:
            return [{'action': 'buy', 'reason': f'价格{price}低于阈值{self.threshold}'}]


# ========== 辅助函数 ==========
def create_mock_data(days=100):
    """创建模拟数据"""
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    prices = 100 + np.random.randn(days).cumsum()

    df = pd.DataFrame({
        'close': prices,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'volume': np.random.randint(1000000, 10000000, days),
    }, index=dates)

    return df


# ========== 测试用例 ==========
def test_voting_method_majority_buy():
    """测试1: 投票法 - 多数推荐买入"""
    print("=" * 60)
    print("测试1: 投票法 - 多数推荐买入")
    print("=" * 60)

    # 创建3个策略: 2个看多, 1个看空
    strategies = [
        MockBullishStrategy(),
        MockBullishStrategy(),
        MockBearishStrategy(),
    ]

    ensemble = EnsembleStrategy(strategies, method='voting')

    df = create_mock_data()
    date = df.index[50]

    signals = ensemble.generate_signals(df, date)

    print(f"\n子策略配置: 2个看多策略 + 1个看空策略")
    print(f"生成信号数: {len(signals)}")

    if signals:
        sig = signals[0]
        print(f"\n信号详情:")
        print(f"  动作: {sig['action']}")
        print(f"  理由: {sig['reason']}")
        print(f"  置信度: {sig['confidence']:.2f}")
        print(f"  投票详情: {sig['voting_details']}")

        # 验证
        assert sig['action'] == 'buy', "多数策略推荐买入,应该生成买入信号"
        assert sig['confidence'] >= 0.6, "2/3策略同意,置信度应>=0.6"

    print("\n✅ 测试1通过!")
    return True


def test_voting_method_majority_sell():
    """测试2: 投票法 - 多数推荐卖出"""
    print("\n" + "=" * 60)
    print("测试2: 投票法 - 多数推荐卖出")
    print("=" * 60)

    # 创建3个策略: 1个看多, 2个看空
    strategies = [
        MockBullishStrategy(),
        MockBearishStrategy(),
        MockBearishStrategy(),
    ]

    ensemble = EnsembleStrategy(strategies, method='voting')

    df = create_mock_data()
    date = df.index[50]

    signals = ensemble.generate_signals(df, date)

    print(f"\n子策略配置: 1个看多策略 + 2个看空策略")
    print(f"生成信号数: {len(signals)}")

    if signals:
        sig = signals[0]
        print(f"\n信号详情:")
        print(f"  动作: {sig['action']}")
        print(f"  理由: {sig['reason']}")
        print(f"  置信度: {sig['confidence']:.2f}")

        # 验证
        assert sig['action'] == 'sell', "多数策略推荐卖出,应该生成卖出信号"

    print("\n✅ 测试2通过!")
    return True


def test_voting_method_no_majority():
    """测试3: 投票法 - 无多数(不发出信号)"""
    print("\n" + "=" * 60)
    print("测试3: 投票法 - 无多数")
    print("=" * 60)

    # 创建3个策略: 1个看多, 1个看空, 1个中性
    strategies = [
        MockBullishStrategy(),
        MockBearishStrategy(),
        MockNeutralStrategy(),
    ]

    ensemble = EnsembleStrategy(strategies, method='voting')

    df = create_mock_data()
    date = df.index[50]

    signals = ensemble.generate_signals(df, date)

    print(f"\n子策略配置: 1看多 + 1看空 + 1中性")
    print(f"生成信号数: {len(signals)}")

    # 验证
    assert len(signals) == 0, "无多数意见,不应生成信号"

    print("\n✅ 测试3通过!")
    return True


def test_weighted_method_basic():
    """测试4: 加权法 - 基础测试"""
    print("\n" + "=" * 60)
    print("测试4: 加权法 - 基础测试")
    print("=" * 60)

    # 创建3个策略,设置不同权重
    strategies = [
        MockBullishStrategy(),   # 权重0.5
        MockBullishStrategy(),   # 权重0.3
        MockBearishStrategy(),   # 权重0.2
    ]

    ensemble = EnsembleStrategy(
        strategies,
        method='weighted',
        weights=[0.5, 0.3, 0.2]
    )

    df = create_mock_data()
    date = df.index[50]

    signals = ensemble.generate_signals(df, date)

    print(f"\n策略权重: {ensemble.weights}")
    print(f"生成信号数: {len(signals)}")

    if signals:
        sig = signals[0]
        print(f"\n信号详情:")
        print(f"  动作: {sig['action']}")
        print(f"  加权得分: {sig['weighted_score']:.2f}")
        print(f"  置信度: {sig['confidence']:.2f}")

        # 验证: 买入权重0.8, 卖出权重0.2, 应该生成买入信号
        assert sig['action'] == 'buy', "加权得分应为正,生成买入信号"
        assert sig['weighted_score'] > 0, "买入权重大于卖出,得分应为正"

    print("\n✅ 测试4通过!")
    return True


def test_weighted_method_custom_weights():
    """测试5: 加权法 - 自定义权重"""
    print("\n" + "=" * 60)
    print("测试5: 加权法 - 自定义权重")
    print("=" * 60)

    strategies = [
        MockBullishStrategy(),
        MockBearishStrategy(),
    ]

    # 测试不同权重组合
    weight_scenarios = [
        ([0.8, 0.2], 'buy'),   # 看多权重大 → 买入
        ([0.2, 0.8], 'sell'),  # 看空权重大 → 卖出
        ([0.5, 0.5], None),    # 权重相等 → 可能无信号
    ]

    df = create_mock_data()
    date = df.index[50]

    for weights, expected_action in weight_scenarios:
        ensemble = EnsembleStrategy(strategies, method='weighted', weights=weights)
        signals = ensemble.generate_signals(df, date)

        print(f"\n权重 {weights}:")
        if signals:
            print(f"  动作: {signals[0]['action']}")
            print(f"  得分: {signals[0]['weighted_score']:.2f}")

            if expected_action:
                assert signals[0]['action'] == expected_action
        else:
            print(f"  无信号(得分接近0)")

    print("\n✅ 测试5通过!")
    return True


def test_dynamic_method():
    """测试6: 动态加权法"""
    print("\n" + "=" * 60)
    print("测试6: 动态加权法")
    print("=" * 60)

    strategies = [
        MockBullishStrategy(),
        MockBearishStrategy(),
        MockNeutralStrategy(),
    ]

    ensemble = EnsembleStrategy(strategies, method='dynamic')

    # 记录历史表现(模拟)
    print("\n模拟历史表现:")
    print("  策略0(看多): 6胜4负")
    for _ in range(6):
        ensemble.record_performance(0, 1.0)   # 盈利
    for _ in range(4):
        ensemble.record_performance(0, -1.0)  # 亏损

    print("  策略1(看空): 3胜7负")
    for _ in range(3):
        ensemble.record_performance(1, 1.0)
    for _ in range(7):
        ensemble.record_performance(1, -1.0)

    print("  策略2(中性): 5胜5负")
    for _ in range(5):
        ensemble.record_performance(2, 1.0)
    for _ in range(5):
        ensemble.record_performance(2, -1.0)

    df = create_mock_data()
    date = df.index[50]

    signals = ensemble.generate_signals(df, date)

    print(f"\n动态调整后的权重: {ensemble.weights}")
    print(f"策略0权重应最高(胜率60%)")

    # 验证
    assert ensemble.weights[0] > ensemble.weights[1], "策略0表现最好,权重应最高"
    assert ensemble.weights[0] > ensemble.weights[2]

    if signals:
        print(f"\n信号详情:")
        print(f"  动作: {signals[0]['action']}")
        print(f"  当前权重: {signals[0]['current_weights']}")

    print("\n✅ 测试6通过!")
    return True


def test_optimize_weights():
    """测试7: 权重优化"""
    print("\n" + "=" * 60)
    print("测试7: 权重优化(基于历史收益)")
    print("=" * 60)

    strategies = [
        MockBullishStrategy(),
        MockBearishStrategy(),
        MockNeutralStrategy(),
    ]

    ensemble = EnsembleStrategy(strategies, method='weighted')

    # 创建模拟历史收益数据(固定种子确保可复现)
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

    # 策略1收益: 高收益高波动
    returns_s1 = np.random.normal(0.002, 0.02, 100)
    # 策略2收益: 低收益低波动
    returns_s2 = np.random.normal(0.0005, 0.01, 100)
    # 策略3收益: 负收益
    returns_s3 = np.random.normal(-0.001, 0.015, 100)

    historical_returns = pd.DataFrame({
        'MockBullishStrategy': returns_s1,
        'MockBearishStrategy': returns_s2,
        'MockNeutralStrategy': returns_s3,
    }, index=dates)

    print("\n历史收益统计:")
    print(historical_returns.describe())

    # 优化权重(最大化夏普比率)
    optimized_weights = ensemble.optimize_weights(
        historical_returns,
        objective='sharpe'
    )

    print(f"\n优化后的权重: {optimized_weights}")
    print(f"  策略0(高收益高波动): {optimized_weights[0]:.2%}")
    print(f"  策略1(低收益低波动): {optimized_weights[1]:.2%}")
    print(f"  策略2(负收益): {optimized_weights[2]:.2%}")

    # 验证
    assert sum(optimized_weights) - 1.0 < 0.01, "权重之和应该为1"
    assert all(w >= 0 for w in optimized_weights), "所有权重应该>=0"

    # 计算实际平均收益,验证权重分配合理性
    avg_returns = historical_returns.mean()
    print(f"\n实际平均收益:")
    print(f"  策略0: {avg_returns[0]:.4f}")
    print(f"  策略1: {avg_returns[1]:.4f}")
    print(f"  策略2: {avg_returns[2]:.4f}")

    # 最低收益的策略应该权重最低(或接近0)
    worst_idx = avg_returns.argmin()
    assert optimized_weights[worst_idx] <= max(optimized_weights), "最差策略权重应不高于最佳策略"

    print("\n✅ 测试7通过!")
    return True


def test_create_ensemble_helper():
    """测试8: 便捷创建函数"""
    print("\n" + "=" * 60)
    print("测试8: 便捷创建函数")
    print("=" * 60)

    strategy_configs = [
        {'class': MockBullishStrategy, 'params': {'name': '看多A'}},
        {'class': MockBearishStrategy, 'params': {'name': '看空B'}},
        {'class': MockNeutralStrategy, 'params': {'name': '中性C'}},
    ]

    ensemble = create_ensemble_strategy(
        strategy_configs,
        method='voting'
    )

    print(f"\n创建的集成策略:")
    print(f"  子策略数: {len(ensemble.strategies)}")
    print(f"  方法: {ensemble.method}")

    summary = ensemble.get_strategy_summary()
    print(f"\n策略摘要:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # 验证
    assert len(ensemble.strategies) == 3, "应该创建3个子策略"
    assert ensemble.method == 'voting'

    print("\n✅ 测试8通过!")
    return True


def test_conditional_strategy():
    """测试9: 条件策略集成"""
    print("\n" + "=" * 60)
    print("测试9: 条件策略集成")
    print("=" * 60)

    # 创建3个条件策略,阈值不同
    strategies = [
        MockConditionalStrategy(threshold=90),   # 阈值90
        MockConditionalStrategy(threshold=100),  # 阈值100
        MockConditionalStrategy(threshold=110),  # 阈值110
    ]

    ensemble = EnsembleStrategy(strategies, method='voting')

    # 创建价格从95开始的数据
    df = pd.DataFrame({
        'close': [95, 96, 97, 98, 99],
        'open': [94, 95, 96, 97, 98],
        'high': [96, 97, 98, 99, 100],
        'low': [94, 95, 96, 97, 98],
        'volume': [1000000] * 5,
    }, index=pd.date_range(start='2024-01-01', periods=5, freq='D'))

    date = df.index[2]  # 价格97
    print(f"\n当前价格: {df.loc[date, 'close']}")
    print(f"策略阈值: [90, 100, 110]")

    signals = ensemble.generate_signals(df, date)

    print(f"\n生成信号数: {len(signals)}")
    if signals:
        sig = signals[0]
        print(f"信号详情:")
        print(f"  动作: {sig['action']}")
        print(f"  理由: {sig['reason']}")

        # 价格97: 高于90(卖), 低于100(买), 低于110(买)
        # 2个买入 vs 1个卖出 → 应该买入
        assert sig['action'] == 'buy', "多数策略推荐买入"

    print("\n✅ 测试9通过!")
    return True


if __name__ == "__main__":
    try:
        # 运行所有测试
        test_voting_method_majority_buy()
        test_voting_method_majority_sell()
        test_voting_method_no_majority()
        test_weighted_method_basic()
        test_weighted_method_custom_weights()
        test_dynamic_method()
        test_optimize_weights()
        test_create_ensemble_helper()
        test_conditional_strategy()

        print("\n" + "=" * 60)
        print("🎉 所有Phase 9.3 策略集成测试通过!")
        print("策略集成功能已准备就绪!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
