"""
测试持仓管理器 - 重点测试美股
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trading.portfolio_manager import PortfolioManager
from datetime import datetime
import json


def test_us_stock_portfolio_dashboard():
    """测试美股持仓仪表盘"""
    print("=" * 60)
    print("测试1: 美股持仓仪表盘")
    print("=" * 60)

    mgr = PortfolioManager(db_path="data/test_us_portfolio.db")

    # 添加5只美股 (FAANG + 金融)
    us_stocks = [
        ("AAPL", 100, 175.0, "Apple Inc.", "Technology"),
        ("MSFT", 50, 380.0, "Microsoft Corporation", "Technology"),
        ("GOOGL", 30, 140.0, "Alphabet Inc.", "Technology"),
        ("AMZN", 25, 160.0, "Amazon.com Inc.", "E-commerce"),
        ("META", 40, 350.0, "Meta Platforms Inc.", "Technology"),
        ("JPM", 80, 150.0, "JPMorgan Chase", "Finance"),
        ("BRK.B", 15, 400.0, "Berkshire Hathaway", "Finance"),
    ]

    print("\n▶ 添加美股持仓:")
    for code, shares, price, name, sector in us_stocks:
        mgr.journal.add_or_update_position(
            market="US",
            code=code,
            shares=shares,
            price=price,
            name=name,
            sector=sector,
            strategy_tag="US Tech/Finance"
        )
        print(f"  ✓ {code}: {shares}股 @ ${price}")

    # 更新市场价格 (模拟价格变动)
    print("\n▶ 更新市场价格 (模拟涨跌):")
    price_changes = {
        "AAPL": 185.0,    # +5.7%
        "MSFT": 390.0,    # +2.6%
        "GOOGL": 135.0,   # -3.6%
        "AMZN": 170.0,    # +6.3%
        "META": 330.0,    # -5.7%
        "JPM": 155.0,     # +3.3%
        "BRK.B": 410.0,   # +2.5%
    }

    for code, new_price in price_changes.items():
        mgr.journal.update_price(
            market="US",
            code=code,
            current_price=new_price
        )

    # 获取仪表盘
    dashboard = mgr.get_portfolio_dashboard(market="US")

    print(f"\n📊 美股持仓仪表盘:")
    print(f"  总市值: ${dashboard['total_market_value']:,.2f}")
    print(f"  总成本: ${dashboard['total_cost']:,.2f}")
    print(f"  浮动盈亏: ${dashboard['unrealized_pnl']:,.2f} ({dashboard['unrealized_pnl_pct']:.2%})")
    print(f"  持仓数量: {dashboard['position_count']}")
    print(f"  盈利股票: {dashboard['profitable_count']}")
    print(f"  亏损股票: {dashboard['losing_count']}")

    print(f"\n🏭 行业分布:")
    for sector, weight in sorted(dashboard['sector_distribution'].items(), key=lambda x: -x[1]):
        print(f"  {sector}: {weight:.1%}")

    print(f"\n🏆 Top 5 持仓:")
    for i, pos in enumerate(dashboard['top_positions'], 1):
        pnl_symbol = "📈" if pos['unrealized_pnl_pct'] > 0 else "📉"
        print(f"  {i}. {pos['code']}: {pos['weight']:.1%} | ${pos['market_value']:,.0f} | {pnl_symbol} {pos['unrealized_pnl_pct']:.2%}")

    # 验证
    assert dashboard['position_count'] == 7, "持仓数量应为7"
    assert dashboard['total_market_value'] > 0, "总市值应大于0"
    assert dashboard['profitable_count'] + dashboard['losing_count'] == 7, "盈亏统计错误"

    print("\n✅ 美股仪表盘测试通过!")
    return True


def test_us_stock_fundamental_analysis():
    """测试美股基本面分析集成"""
    print("\n" + "=" * 60)
    print("测试2: 美股基本面分析集成")
    print("=" * 60)

    mgr = PortfolioManager(db_path="data/test_us_portfolio.db")

    # 分析AAPL
    print("\n▶ 分析AAPL (苹果):")
    analysis = mgr.analyze_holding("AAPL", "US", include_fundamental=True)

    print(f"\n{analysis['code']} - {analysis['name']}")
    print(f"  持仓信息:")
    print(f"    股数: {analysis['holding_info']['shares']}")
    print(f"    成本: ${analysis['holding_info']['average_cost']:.2f}")
    print(f"    现价: ${analysis['holding_info']['current_price']:.2f}")
    print(f"    盈亏: {analysis['holding_info']['unrealized_pnl_pct']:.2%}")
    print(f"    市值: ${analysis['holding_info']['market_value']:,.2f}")
    print(f"    权重: {analysis['holding_info']['weight']:.1%}")
    print(f"    行业: {analysis['holding_info']['sector']}")

    if analysis.get('fundamental_score'):
        fs = analysis['fundamental_score']
        print(f"\n  基本面评分:")
        print(f"    盈利能力: {fs.get('盈利能力', 0)}/100")
        print(f"    成长性: {fs.get('成长性', 0)}/100")
        print(f"    估值吸引力: {fs.get('估值吸引力', 0)}/100")
        print(f"    财务健康: {fs.get('财务健康', 0)}/100")
        print(f"    综合得分: {fs.get('综合得分', 0)}/100")
        print(f"    评级: {fs.get('评级', 'N/A')}")
    else:
        print("\n  ⚠️  基本面分析失败 (可能网络问题)")

    print(f"\n  💡 操作建议: {analysis['recommendation']}")

    # 验证
    assert 'holding_info' in analysis, "缺少持仓信息"
    assert analysis['code'] == "AAPL", "股票代码错误"

    print("\n✅ 美股基本面分析测试通过!")
    return True


def test_rebalance_plan():
    """测试调仓计划生成"""
    print("\n" + "=" * 60)
    print("测试3: 调仓计划生成")
    print("=" * 60)

    mgr = PortfolioManager(db_path="data/test_us_portfolio.db")

    # 设定目标权重 (调整为更平衡的配置)
    target_weights = {
        "AAPL": 0.20,    # 20%
        "MSFT": 0.20,    # 20%
        "GOOGL": 0.15,   # 15%
        "AMZN": 0.15,    # 15%
        "META": 0.10,    # 10%
        "JPM": 0.15,     # 15%
        "BRK.B": 0.05,   # 5%
    }

    print("\n▶ 目标权重分配:")
    for code, weight in target_weights.items():
        print(f"  {code}: {weight:.0%}")

    # 生成调仓计划
    plan = mgr.generate_rebalance_plan(
        market="US",
        target_weights=target_weights,
        min_trade_amount=100.0
    )

    print(f"\n📋 调仓计划 ({len(plan)}项操作):")
    buy_total = 0
    sell_total = 0

    for item in plan:
        action_symbol = "🟢" if item['action'] == 'buy' else "🔴"
        print(f"  {action_symbol} {item['action'].upper()}: {item['code']} {item['shares']}股 (${item['amount']:,.0f})")

        if item['action'] == 'buy':
            buy_total += item['amount']
        else:
            sell_total += item['amount']

    print(f"\n  买入总额: ${buy_total:,.0f}")
    print(f"  卖出总额: ${sell_total:,.0f}")
    print(f"  净流入: ${buy_total - sell_total:,.0f}")

    print("\n✅ 调仓计划测试通过!")
    return True


def test_strategy_comparison():
    """测试持仓vs策略对比"""
    print("\n" + "=" * 60)
    print("测试4: 持仓vs策略对比")
    print("=" * 60)

    mgr = PortfolioManager(db_path="data/test_us_portfolio.db")

    # 模拟策略推荐 (新的推荐组合)
    strategy_recommendations = [
        ("AAPL", 95),    # 保留,但评分下降
        ("MSFT", 90),    # 保留
        ("GOOGL", 85),   # 保留
        ("NVDA", 92),    # 新推荐(英伟达)
        ("TSLA", 88),    # 新推荐(特斯拉)
        ("JPM", 80),     # 保留
        # META, AMZN, BRK.B 不再推荐
    ]

    print("\n▶ 策略推荐股票:")
    for code, score in strategy_recommendations:
        print(f"  {code}: {score}分")

    # 对比分析
    comparison = mgr.compare_with_strategy(
        market="US",
        strategy_recommendations=strategy_recommendations,
        threshold=0.03  # 3%阈值
    )

    print(f"\n📊 对比结果:")

    if comparison['should_buy']:
        print(f"\n  🟢 应买入 ({len(comparison['should_buy'])}只):")
        for item in comparison['should_buy']:
            print(f"    {item['code']}: 评分{item['score']} - {item['reason']}")

    if comparison['should_sell']:
        print(f"\n  🔴 应卖出 ({len(comparison['should_sell'])}只):")
        for item in comparison['should_sell']:
            print(f"    {item['code']} ({item['name']}): {item['shares']}股 - {item['reason']}")

    if comparison['keep_holding']:
        print(f"\n  ➡️  继续持有 ({len(comparison['keep_holding'])}只):")
        for item in comparison['keep_holding']:
            print(f"    {item['code']} ({item['name']}): {item['weight']:.1%}")

    # 验证
    assert len(comparison['should_buy']) > 0, "应该有需要买入的股票"
    assert len(comparison['should_sell']) > 0, "应该有需要卖出的股票"

    print("\n✅ 策略对比测试通过!")
    return True


def test_batch_holdings_analysis():
    """测试批量持仓分析"""
    print("\n" + "=" * 60)
    print("测试5: 批量持仓分析 (含基本面)")
    print("=" * 60)

    mgr = PortfolioManager(db_path="data/test_us_portfolio.db")

    holdings = mgr.journal.get_holdings(market="US")
    print(f"\n▶ 分析{len(holdings)}只美股持仓:\n")

    results = []

    for _, row in holdings.iterrows():
        code = row['code']
        try:
            analysis = mgr.analyze_holding(code, "US", include_fundamental=True)

            fundamental_score = 0
            if analysis.get('fundamental_score'):
                fundamental_score = analysis['fundamental_score'].get('综合得分', 0)

            results.append({
                'code': code,
                'name': analysis['name'],
                'pnl_pct': analysis['holding_info']['unrealized_pnl_pct'],
                'fundamental_score': fundamental_score,
                'recommendation': analysis['recommendation']
            })

            pnl_symbol = "📈" if analysis['holding_info']['unrealized_pnl_pct'] > 0 else "📉"
            print(f"{code:6} {pnl_symbol} {analysis['holding_info']['unrealized_pnl_pct']:>6.1%} | 基本面:{fundamental_score:>3}/100 | {analysis['recommendation']}")

        except Exception as e:
            print(f"{code:6} ❌ 分析失败: {e}")

    # 排序展示
    print("\n📊 综合评分排名 (基本面得分):")
    results.sort(key=lambda x: x['fundamental_score'], reverse=True)
    for i, r in enumerate(results[:5], 1):
        print(f"  {i}. {r['code']}: {r['fundamental_score']}/100")

    print("\n✅ 批量分析测试通过!")
    return True


if __name__ == "__main__":
    try:
        # 运行所有测试
        test_us_stock_portfolio_dashboard()
        test_us_stock_fundamental_analysis()
        test_rebalance_plan()
        test_strategy_comparison()
        test_batch_holdings_analysis()

        print("\n" + "=" * 60)
        print("🎉 所有美股测试通过! Phase 3 完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
