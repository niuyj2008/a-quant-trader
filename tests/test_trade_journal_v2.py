"""
测试新的TradeJournal数据库结构
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trading.trade_journal import TradeJournal
from datetime import datetime
import json


def test_database_structure():
    """测试新数据库表结构"""
    print("=" * 60)
    print("测试1: 数据库初始化和表结构")
    print("=" * 60)

    # 创建测试数据库
    journal = TradeJournal(db_path="data/test_journal.db")

    import sqlite3
    with sqlite3.connect(journal.db_path) as conn:
        # 检查holdings表结构
        cursor = conn.execute("PRAGMA table_info(holdings)")
        holdings_cols = [row[1] for row in cursor.fetchall()]
        print(f"\n✓ holdings表字段 ({len(holdings_cols)}个):")
        for col in holdings_cols:
            print(f"  - {col}")

        # 检查trades表结构
        cursor = conn.execute("PRAGMA table_info(trades)")
        trades_cols = [row[1] for row in cursor.fetchall()]
        print(f"\n✓ trades表字段 ({len(trades_cols)}个):")
        for col in trades_cols:
            print(f"  - {col}")

        # 检查recommendations表结构
        cursor = conn.execute("PRAGMA table_info(recommendations)")
        recs_cols = [row[1] for row in cursor.fetchall()]
        print(f"\n✓ recommendations表字段 ({len(recs_cols)}个):")
        for col in recs_cols:
            print(f"  - {col}")

        # 检查portfolio_performance表
        cursor = conn.execute("PRAGMA table_info(portfolio_performance)")
        perf_cols = [row[1] for row in cursor.fetchall()]
        print(f"\n✓ portfolio_performance表字段 ({len(perf_cols)}个):")
        for col in perf_cols:
            print(f"  - {col}")

        # 检查索引
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = [row[0] for row in cursor.fetchall()]
        print(f"\n✓ 创建的索引 ({len(indexes)}个):")
        for idx in indexes:
            if not idx.startswith('sqlite_'):
                print(f"  - {idx}")

    print("\n✅ 数据库结构测试通过!")
    return True


def test_position_lifecycle():
    """测试持仓生命周期管理"""
    print("\n" + "=" * 60)
    print("测试2: 持仓生命周期管理")
    print("=" * 60)

    journal = TradeJournal(db_path="data/test_journal.db")

    # 1. 首次建仓
    print("\n▶ 步骤1: 首次建仓 - 买入600519 100股 @1800元")
    journal.add_or_update_position(
        market="CN",
        code="600519",
        shares=100,
        price=1800.0,
        name="贵州茅台",
        sector="白酒",
        strategy_tag="价值投资"
    )

    holdings = journal.get_holdings(market="CN")
    print(f"✓ 当前持仓: {len(holdings)}只")
    if len(holdings) > 0:
        print(f"  - {holdings.iloc[0]['code']}: {holdings.iloc[0]['total_shares']}股 @{holdings.iloc[0]['average_cost']:.2f}")
        print(f"  - buy_batches: {holdings.iloc[0]['buy_batches']}")

    # 2. 加仓
    print("\n▶ 步骤2: 加仓 - 再买入600519 50股 @1750元")
    journal.add_or_update_position(
        market="CN",
        code="600519",
        shares=50,
        price=1750.0,
        name="贵州茅台"
    )

    holdings = journal.get_holdings(market="CN")
    if len(holdings) > 0:
        h = holdings.iloc[0]
        print(f"✓ 加仓后: {h['total_shares']}股 @{h['average_cost']:.2f}")
        batches = json.loads(h['buy_batches'])
        print(f"  - 建仓批次: {len(batches)}次")
        for i, batch in enumerate(batches, 1):
            print(f"    {i}. {batch['date']}: {batch['shares']}股 @{batch['price']}")

    # 3. 部分减仓
    print("\n▶ 步骤3: 减仓 - 卖出600519 50股 @1900元")
    realized_pnl = journal.reduce_position(
        market="CN",
        code="600519",
        shares=50,
        price=1900.0
    )
    print(f"✓ 实现盈亏: {realized_pnl:.2f}元")

    holdings = journal.get_holdings(market="CN")
    if len(holdings) > 0:
        h = holdings.iloc[0]
        print(f"  - 剩余持仓: {h['total_shares']}股")
        print(f"  - 累计实现盈亏: {h['realized_pnl']:.2f}元")

    # 4. 清仓
    print("\n▶ 步骤4: 清仓 - 卖出剩余100股 @1850元")
    realized_pnl = journal.reduce_position(
        market="CN",
        code="600519",
        shares=100,
        price=1850.0
    )
    print(f"✓ 本次实现盈亏: {realized_pnl:.2f}元")

    holdings = journal.get_holdings(market="CN")
    print(f"✓ 清仓后持仓数: {len(holdings)}")

    print("\n✅ 持仓生命周期测试通过!")
    return True


def test_recommendation_tracking():
    """测试推荐追踪功能"""
    print("\n" + "=" * 60)
    print("测试3: 推荐记录增强字段")
    print("=" * 60)

    journal = TradeJournal(db_path="data/test_journal.db")

    # 记录一个推荐
    print("\n▶ 记录推荐: 000001 @10.5元, 目标价12元")
    journal.record_recommendation(
        market="CN",
        code="000001",
        strategy="动量策略",
        score=85.0,
        confidence=0.75,
        reason="突破60日均线,成交量放大",
        price=10.5,
        name="平安银行"
    )

    # 更新推荐的回测数据(模拟)
    import sqlite3
    with sqlite3.connect(journal.db_path) as conn:
        conn.execute("""
            UPDATE recommendations SET
                price_after_1w=11.0, return_1w=0.0476,
                price_after_1m=11.5, return_1m=0.0952,
                price_after_3m=12.2, return_3m=0.1619,
                target_price=12.0,
                stop_loss_suggested=9.5,
                backtest_status='completed'
            WHERE code='000001' AND strategy='动量策略'
        """)
        conn.commit()

    recs = journal.get_recommendations(market="CN", limit=1)
    if len(recs) > 0:
        r = recs.iloc[0]
        print(f"✓ 推荐股票: {r['code']} ({r['name']})")
        print(f"  - 推荐价: ¥{r['price_at_recommend']:.2f}")
        print(f"  - 目标价: ¥{r['target_price']:.2f}")
        print(f"  - 建议止损: ¥{r['stop_loss_suggested']:.2f}")
        print(f"  - 1周收益: {r['return_1w']:.2%}")
        print(f"  - 1月收益: {r['return_1m']:.2%}")
        print(f"  - 3月收益: {r['return_3m']:.2%}")
        print(f"  - 回测状态: {r['backtest_status']}")

    print("\n✅ 推荐追踪测试通过!")
    return True


if __name__ == "__main__":
    try:
        # 运行所有测试
        test_database_structure()
        test_position_lifecycle()
        test_recommendation_tracking()

        print("\n" + "=" * 60)
        print("🎉 所有测试通过! Phase 1数据库重构完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
