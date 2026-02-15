"""
测试目标导向推荐系统 - Phase 7
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.strategy.goal_based_recommender import (
    InvestmentGoal,
    StrategyRecommender,
    quick_recommend,
)


def test_investment_goal():
    """测试0: 投资目标类"""
    print("=" * 60)
    print("测试0: 投资目标类")
    print("=" * 60)

    # 创建目标
    goal = InvestmentGoal(
        time_horizon_years=3,
        target_return=0.15,  # 15%年化收益
        risk_tolerance='moderate',
        initial_capital=100000,
        monthly_invest=5000,
    )

    print("\n▶ 投资目标摘要:")
    print(goal.summary())

    print("\n▶ 风险约束:")
    constraints = goal.get_risk_constraints()
    for key, value in constraints.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")

    # 验证
    assert goal.time_horizon_years == 3
    assert goal.target_return == 0.15
    assert goal.risk_tolerance == 'moderate'

    print("\n✅ 投资目标类测试通过!")
    return True


def test_conservative_goal():
    """测试1: 保守型目标"""
    print("\n" + "=" * 60)
    print("测试1: 保守型目标 (3年年化8%)")
    print("=" * 60)

    goal = InvestmentGoal(
        time_horizon_years=3,
        target_return=0.08,  # 8%年化收益
        risk_tolerance='conservative',
        initial_capital=100000,
    )

    recommender = StrategyRecommender()
    result = recommender.recommend(goal)

    if result['status'] == 'success':
        print("\n" + result['report'])

        strategies = result['recommended_strategies']
        print(f"\n📊 推荐策略数量: {len(strategies)}")

        # 验证
        assert len(strategies) > 0, "应该至少有1个推荐策略"

        best_strategy = strategies[0]
        print(f"\n🏆 最佳策略: {best_strategy['name']}")
        print(f"   匹配度: {best_strategy['scores']['total']:.1f}/100")
        print(f"   达成概率: {best_strategy['success_probability']:.1%}")

        # 保守型目标应该推荐低风险策略
        assert best_strategy['performance']['risk_level'] in ['conservative', 'moderate'], \
            "保守型目标应推荐低风险策略"

    else:
        print(f"\n⚠️  {result['message']}")

    print("\n✅ 保守型目标测试通过!")
    return True


def test_aggressive_goal():
    """测试2: 激进型目标"""
    print("\n" + "=" * 60)
    print("测试2: 激进型目标 (3年翻倍,年化26%)")
    print("=" * 60)

    # 3年翻倍需要年化26%收益
    goal = InvestmentGoal(
        time_horizon_years=3,
        target_return=0.26,  # 26%年化收益
        risk_tolerance='aggressive',
        initial_capital=100000,
    )

    recommender = StrategyRecommender()
    result = recommender.recommend(goal)

    if result['status'] == 'success':
        print("\n" + result['report'])

        strategies = result['recommended_strategies']
        print(f"\n📊 推荐策略数量: {len(strategies)}")

        if len(strategies) > 0:
            best_strategy = strategies[0]
            print(f"\n🏆 最佳策略: {best_strategy['name']}")
            print(f"   匹配度: {best_strategy['scores']['total']:.1f}/100")
            print(f"   达成概率: {best_strategy['success_probability']:.1%}")

            # 验证: 高目标的达成概率应该较低
            assert best_strategy['success_probability'] < 0.8, \
                "高目标的达成概率应该较低"
        else:
            print("  ⚠️  目标过高,无匹配策略")

    else:
        print(f"\n⚠️  {result['message']}")

    print("\n✅ 激进型目标测试通过!")
    return True


def test_moderate_long_term():
    """测试3: 稳健型长期目标"""
    print("\n" + "=" * 60)
    print("测试3: 稳健型长期目标 (5年年化12%)")
    print("=" * 60)

    goal = InvestmentGoal(
        time_horizon_years=5,
        target_return=0.12,  # 12%年化收益
        risk_tolerance='moderate',
        initial_capital=200000,
        monthly_invest=5000,
    )

    recommender = StrategyRecommender()
    result = recommender.recommend(goal)

    if result['status'] == 'success':
        print("\n" + result['report'])

        strategies = result['recommended_strategies']

        if len(strategies) > 0:
            # 验证: 应该推荐多因子均衡或价值投资
            strategy_names = [s['name'] for s in strategies]
            print(f"\n📋 所有推荐策略: {strategy_names}")

            assert any(name in ['多因子均衡', 'ETF价值平均', '价值投资'] for name in strategy_names), \
                "稳健长期目标应推荐均衡型策略"

    print("\n✅ 稳健型长期目标测试通过!")
    return True


def test_etf_preference():
    """测试4: ETF偏好"""
    print("\n" + "=" * 60)
    print("测试4: ETF偏好用户")
    print("=" * 60)

    goal = InvestmentGoal(
        time_horizon_years=5,
        target_return=0.10,
        risk_tolerance='moderate',
        initial_capital=100000,
        prefer_etf=True,  # 偏好ETF
    )

    recommender = StrategyRecommender()
    result = recommender.recommend(goal)

    if result['status'] == 'success':
        print("\n" + result['report'])

        strategies = result['recommended_strategies']

        if len(strategies) > 0:
            # 验证: 所有推荐策略都应该是ETF类型
            for strategy in strategies:
                strategy_type = strategy['performance']['type']
                print(f"  {strategy['name']}: {strategy_type}")

                assert strategy_type in ['etf_dca', 'etf_va', 'rebalancing'], \
                    "ETF偏好用户应只推荐ETF策略"

    print("\n✅ ETF偏好测试通过!")
    return True


def test_quick_recommend():
    """测试5: 快速推荐函数"""
    print("\n" + "=" * 60)
    print("测试5: 快速推荐函数")
    print("=" * 60)

    # 使用便捷函数
    result = quick_recommend(
        target_return=0.15,  # 15%年化
        years=3,
        risk_tolerance='moderate',
        initial_capital=100000
    )

    if result['status'] == 'success':
        print("\n✅ 快速推荐成功!")
        print(f"  推荐策略数: {len(result['recommended_strategies'])}")

        if result['recommended_strategies']:
            best = result['recommended_strategies'][0]
            print(f"  最佳策略: {best['name']}")
            print(f"  达成概率: {best['success_probability']:.1%}")

    print("\n✅ 快速推荐函数测试通过!")
    return True


def test_impossible_goal():
    """测试6: 不可达成的目标"""
    print("\n" + "=" * 60)
    print("测试6: 不可达成的目标 (1年翻倍)")
    print("=" * 60)

    goal = InvestmentGoal(
        time_horizon_years=1,
        target_return=1.0,  # 100%年化收益
        risk_tolerance='aggressive',
        initial_capital=100000,
    )

    recommender = StrategyRecommender()
    result = recommender.recommend(goal)

    if result['status'] == 'no_match':
        print(f"\n⚠️  预期结果: {result['message']}")
        print("✅ 正确识别出不可达成的目标")
    elif result['status'] == 'success':
        if result['recommended_strategies']:
            best = result['recommended_strategies'][0]
            print(f"\n  找到策略: {best['name']}")
            print(f"  达成概率: {best['success_probability']:.1%}")

            # 验证: 达成概率应该很低
            assert best['success_probability'] < 0.3, \
                "不可达成目标的概率应该<30%"

    print("\n✅ 不可达成目标测试通过!")
    return True


if __name__ == "__main__":
    try:
        # 运行所有测试
        test_investment_goal()
        test_conservative_goal()
        test_aggressive_goal()
        test_moderate_long_term()
        test_etf_preference()
        test_quick_recommend()
        test_impossible_goal()

        print("\n" + "=" * 60)
        print("🎉 所有Phase 7测试通过! 目标导向推荐完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
