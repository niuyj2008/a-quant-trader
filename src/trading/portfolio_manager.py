"""
持仓管理器 - PortfolioManager

提供持仓仪表盘、持仓vs策略对比、调仓计划生成等功能
重点支持美股市场
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from loguru import logger

import sys
from pathlib import Path

# 添加项目根目录到路径
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

from src.trading.trade_journal import TradeJournal
from src.analysis.fundamental import FundamentalAnalyzer


class PortfolioManager:
    """持仓管理器"""

    def __init__(self, db_path: str = "data/trade_journal.db"):
        """
        初始化持仓管理器

        Args:
            db_path: 交易日志数据库路径
        """
        self.journal = TradeJournal(db_path=db_path)
        self.fundamental_analyzer = FundamentalAnalyzer()

    def get_portfolio_dashboard(self, market: str = "US") -> Dict:
        """
        获取持仓仪表盘

        Args:
            market: 'CN' (A股) 或 'US' (美股)

        Returns:
            {
                'total_market_value': 总市值,
                'total_cost': 总成本,
                'unrealized_pnl': 浮动盈亏,
                'unrealized_pnl_pct': 浮动盈亏率,
                'realized_pnl': 已实现盈亏,
                'today_pnl': 今日盈亏,
                'position_count': 持仓数量,
                'sector_distribution': 行业分布,
                'top_positions': Top5持仓,
                'profitable_count': 盈利股票数,
                'losing_count': 亏损股票数,
            }
        """
        holdings = self.journal.get_holdings(market=market)

        if holdings.empty:
            return {
                'total_market_value': 0,
                'total_cost': 0,
                'unrealized_pnl': 0,
                'unrealized_pnl_pct': 0,
                'realized_pnl': 0,
                'today_pnl': 0,
                'position_count': 0,
                'sector_distribution': {},
                'top_positions': [],
                'profitable_count': 0,
                'losing_count': 0,
            }

        # 计算总市值和总成本
        total_market_value = holdings['market_value'].sum()
        total_cost = holdings['total_invested'].sum()
        unrealized_pnl = holdings['unrealized_pnl'].sum()
        unrealized_pnl_pct = unrealized_pnl / total_cost if total_cost > 0 else 0

        # 已实现盈亏
        realized_pnl = holdings['realized_pnl'].sum()

        # 今日盈亏 (需要获取昨日收盘价计算)
        today_pnl = self._calculate_today_pnl(holdings)

        # 行业分布
        sector_distribution = self._calculate_sector_distribution(holdings)

        # Top5持仓
        holdings_sorted = holdings.sort_values('weight', ascending=False)
        top_positions = []
        for _, row in holdings_sorted.head(5).iterrows():
            top_positions.append({
                'code': row['code'],
                'name': row['name'],
                'weight': row['weight'],
                'market_value': row['market_value'],
                'unrealized_pnl_pct': row['unrealized_pnl_pct'],
            })

        # 盈亏统计
        profitable_count = len(holdings[holdings['unrealized_pnl'] > 0])
        losing_count = len(holdings[holdings['unrealized_pnl'] < 0])

        return {
            'total_market_value': total_market_value,
            'total_cost': total_cost,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'realized_pnl': realized_pnl,
            'today_pnl': today_pnl,
            'position_count': len(holdings),
            'sector_distribution': sector_distribution,
            'top_positions': top_positions,
            'profitable_count': profitable_count,
            'losing_count': losing_count,
        }

    def _calculate_today_pnl(self, holdings: pd.DataFrame) -> float:
        """计算今日盈亏"""
        # 简化实现: 假设持仓不变,只计算价格变动
        # TODO: 需要实时价格数据API
        return 0.0

    def _calculate_sector_distribution(self, holdings: pd.DataFrame) -> Dict[str, float]:
        """
        计算行业分布

        Returns:
            {'Technology': 0.30, 'Finance': 0.25, ...}
        """
        total_value = holdings['market_value'].sum()
        if total_value == 0:
            return {}

        sector_values = holdings.groupby('sector')['market_value'].sum()
        sector_distribution = (sector_values / total_value).to_dict()

        return sector_distribution

    def compare_with_strategy(
        self,
        market: str,
        strategy_recommendations: List[Tuple[str, float]],
        threshold: float = 0.05
    ) -> Dict:
        """
        持仓vs策略推荐对比

        Args:
            market: 市场代码
            strategy_recommendations: 策略推荐 [(code, score), ...]
            threshold: 仓位偏差阈值 (默认5%)

        Returns:
            {
                'should_buy': 应买入的股票,
                'should_sell': 应卖出的股票,
                'should_add': 应加仓的股票,
                'should_reduce': 应减仓的股票,
                'keep_holding': 持有不变的股票,
            }
        """
        current_holdings = self.journal.get_holdings(market=market)

        # 当前持仓股票代码集合
        current_codes = set(current_holdings['code'].tolist())

        # 策略推荐股票代码
        recommended_codes = {code for code, _ in strategy_recommendations}

        # 应买入: 策略推荐但未持有
        should_buy = []
        for code, score in strategy_recommendations:
            if code not in current_codes:
                should_buy.append({
                    'code': code,
                    'score': score,
                    'reason': '策略推荐但未持有',
                })

        # 应卖出: 持有但策略不再推荐
        should_sell = []
        for _, row in current_holdings.iterrows():
            if row['code'] not in recommended_codes:
                should_sell.append({
                    'code': row['code'],
                    'name': row['name'],
                    'shares': row['total_shares'],
                    'unrealized_pnl_pct': row['unrealized_pnl_pct'],
                    'reason': '策略不再推荐',
                })

        # 应加仓/减仓 (基于目标权重vs当前权重)
        should_add = []
        should_reduce = []

        # 计算策略推荐的目标权重
        total_score = sum(score for _, score in strategy_recommendations)
        target_weights = {code: score / total_score for code, score in strategy_recommendations}

        for _, row in current_holdings.iterrows():
            code = row['code']
            current_weight = row['weight']

            if code in target_weights:
                target_weight = target_weights[code]
                weight_diff = target_weight - current_weight

                if weight_diff > threshold:
                    should_add.append({
                        'code': code,
                        'name': row['name'],
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'diff': weight_diff,
                    })
                elif weight_diff < -threshold:
                    should_reduce.append({
                        'code': code,
                        'name': row['name'],
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'diff': weight_diff,
                    })

        # 持有不变
        keep_holding = []
        for _, row in current_holdings.iterrows():
            code = row['code']
            if code in recommended_codes:
                if code not in [x['code'] for x in should_add + should_reduce]:
                    keep_holding.append({
                        'code': code,
                        'name': row['name'],
                        'weight': row['weight'],
                    })

        return {
            'should_buy': should_buy,
            'should_sell': should_sell,
            'should_add': should_add,
            'should_reduce': should_reduce,
            'keep_holding': keep_holding,
        }

    def generate_rebalance_plan(
        self,
        market: str,
        target_weights: Dict[str, float],
        total_portfolio_value: Optional[float] = None,
        min_trade_amount: float = 100.0
    ) -> List[Dict]:
        """
        生成调仓计划

        Args:
            market: 市场代码
            target_weights: 目标权重 {'AAPL': 0.15, 'MSFT': 0.12, ...}
            total_portfolio_value: 总组合价值 (None=使用当前市值)
            min_trade_amount: 最小交易金额 (避免小额调仓)

        Returns:
            [
                {'action': 'buy', 'code': 'AAPL', 'shares': 10, 'amount': 1500},
                {'action': 'sell', 'code': 'TSLA', 'shares': 5, 'amount': 1000},
            ]
        """
        current_holdings = self.journal.get_holdings(market=market)

        # 获取总组合价值
        if total_portfolio_value is None:
            total_portfolio_value = current_holdings['market_value'].sum()

        if total_portfolio_value == 0:
            logger.warning("组合总价值为0,无法生成调仓计划")
            return []

        rebalance_plan = []

        # 当前持仓字典
        current_positions = {}
        for _, row in current_holdings.iterrows():
            current_positions[row['code']] = {
                'shares': row['total_shares'],
                'value': row['market_value'],
                'price': row['current_price'],
            }

        # 处理每个目标股票
        all_codes = set(list(target_weights.keys()) + list(current_positions.keys()))

        for code in all_codes:
            target_weight = target_weights.get(code, 0)
            target_value = total_portfolio_value * target_weight

            current_value = current_positions.get(code, {}).get('value', 0)
            current_price = current_positions.get(code, {}).get('price', 0)

            # 计算价值差异
            value_diff = target_value - current_value

            # 如果差异小于最小交易金额,跳过
            if abs(value_diff) < min_trade_amount:
                continue

            # 需要获取当前价格 (如果不是现有持仓)
            if code not in current_positions:
                # TODO: 获取实时价格
                current_price = self._get_current_price(code, market)
                if current_price == 0:
                    logger.warning(f"无法获取{code}的当前价格,跳过")
                    continue

            # 计算股数差异
            if current_price > 0:
                shares_diff = int(value_diff / current_price)

                # 美股可以买1股,A股需要100股整数倍
                if market == "CN":
                    shares_diff = (shares_diff // 100) * 100

                if shares_diff > 0:
                    rebalance_plan.append({
                        'action': 'buy',
                        'code': code,
                        'price': current_price,
                        'shares': shares_diff,
                        'amount': shares_diff * current_price,
                        'reason': f'调整至目标权重{target_weight:.1%}',
                    })
                elif shares_diff < 0:
                    rebalance_plan.append({
                        'action': 'sell',
                        'code': code,
                        'price': current_price,
                        'shares': abs(shares_diff),
                        'amount': abs(shares_diff) * current_price,
                        'reason': f'调整至目标权重{target_weight:.1%}',
                    })

        return rebalance_plan

    def _get_current_price(self, code: str, market: str) -> float:
        """获取当前价格"""
        # TODO: 集成实时价格API
        # A股: ak.stock_zh_a_spot_em()
        # 美股: yfinance
        return 0.0

    def analyze_holding(
        self,
        code: str,
        market: str,
        include_fundamental: bool = True
    ) -> Dict:
        """
        分析单个持仓 (技术面 + 基本面)

        Args:
            code: 股票代码
            market: 市场
            include_fundamental: 是否包含基本面分析

        Returns:
            {
                'code': 股票代码,
                'name': 股票名称,
                'holding_info': 持仓信息,
                'fundamental_score': 基本面评分 (可选),
                'combined_score': 综合评分,
                'recommendation': 操作建议,
            }
        """
        # 获取持仓信息
        holdings = self.journal.get_holdings(market=market)
        holding = holdings[holdings['code'] == code]

        if holding.empty:
            return {'error': f'未找到{code}的持仓信息'}

        holding_row = holding.iloc[0]

        result = {
            'code': code,
            'name': holding_row['name'],
            'holding_info': {
                'shares': holding_row['total_shares'],
                'average_cost': holding_row['average_cost'],
                'current_price': holding_row['current_price'],
                'unrealized_pnl_pct': holding_row['unrealized_pnl_pct'],
                'market_value': holding_row['market_value'],
                'weight': holding_row['weight'],
                'holding_days': holding_row['holding_days'],
                'sector': holding_row['sector'],
            }
        }

        # 基本面分析
        if include_fundamental:
            try:
                fundamental_result = self.fundamental_analyzer.generate_fundamental_score(
                    code=code,
                    market=market,
                    sector=holding_row['sector']
                )
                result['fundamental_score'] = fundamental_result
            except Exception as e:
                logger.error(f"基本面分析失败 {code}: {e}")
                result['fundamental_score'] = None

        # 生成操作建议
        result['recommendation'] = self._generate_holding_recommendation(
            holding_row,
            result.get('fundamental_score')
        )

        return result

    def _generate_holding_recommendation(
        self,
        holding: pd.Series,
        fundamental_score: Optional[Dict]
    ) -> str:
        """
        生成持仓操作建议

        逻辑:
        - 盈亏超过止损/止盈线 → 建议卖出/部分止盈
        - 基本面恶化 → 建议减仓
        - 基本面优秀+浮亏 → 建议加仓
        """
        unrealized_pnl_pct = holding['unrealized_pnl_pct']
        stop_loss_price = holding['stop_loss_price']
        take_profit_price = holding['take_profit_price']
        current_price = holding['current_price']

        # 止损
        if stop_loss_price > 0 and current_price <= stop_loss_price:
            return "⚠️  触发止损,建议卖出"

        # 止盈
        if take_profit_price > 0 and current_price >= take_profit_price:
            return "🎯 达到止盈目标,建议部分止盈"

        # 基于浮盈浮亏
        if unrealized_pnl_pct < -0.10:
            if fundamental_score and fundamental_score.get('综合得分', 0) >= 75:
                return "💎 浮亏但基本面优秀,可考虑加仓摊低成本"
            else:
                return "⚠️  浮亏较大,建议止损或观察"
        elif unrealized_pnl_pct > 0.30:
            return "🎉 盈利丰厚,建议部分止盈锁定利润"

        # 基于基本面评分
        if fundamental_score:
            score = fundamental_score.get('综合得分', 0)
            rating = fundamental_score.get('评级', '')

            if score >= 80:
                return f"✅ 基本面优秀({rating}),建议继续持有"
            elif score < 60:
                return f"⚠️  基本面较差({rating}),建议减仓或卖出"

        return "➡️  持有观察"

    def get_portfolio_performance_summary(
        self,
        market: str,
        period_days: int = 30
    ) -> Dict:
        """
        获取组合绩效摘要

        Args:
            market: 市场
            period_days: 统计周期(天)

        Returns:
            {
                'total_return': 总收益率,
                'win_rate': 胜率,
                'avg_holding_days': 平均持仓天数,
                'turnover_rate': 换手率,
                'best_stock': 最佳股票,
                'worst_stock': 最差股票,
            }
        """
        # TODO: 实现绩效统计逻辑
        # 需要从trades表统计历史交易数据
        return {
            'total_return': 0,
            'win_rate': 0,
            'avg_holding_days': 0,
            'turnover_rate': 0,
            'best_stock': None,
            'worst_stock': None,
        }


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("测试: PortfolioManager (重点美股)")
    print("=" * 60)

    # 创建管理器
    mgr = PortfolioManager(db_path="data/test_portfolio.db")

    # 测试1: 添加美股持仓
    print("\n▶ 步骤1: 添加美股持仓")

    # AAPL - 苹果
    mgr.journal.add_or_update_position(
        market="US",
        code="AAPL",
        shares=100,
        price=175.0,
        name="Apple Inc.",
        sector="Technology",
        strategy_tag="Tech Giants"
    )

    # MSFT - 微软
    mgr.journal.add_or_update_position(
        market="US",
        code="MSFT",
        shares=50,
        price=380.0,
        name="Microsoft Corporation",
        sector="Technology",
        strategy_tag="Tech Giants"
    )

    # GOOGL - 谷歌
    mgr.journal.add_or_update_position(
        market="US",
        code="GOOGL",
        shares=30,
        price=140.0,
        name="Alphabet Inc.",
        sector="Technology",
        strategy_tag="Tech Giants"
    )

    # JPM - 摩根大通
    mgr.journal.add_or_update_position(
        market="US",
        code="JPM",
        shares=80,
        price=150.0,
        name="JPMorgan Chase & Co.",
        sector="Finance",
        strategy_tag="Blue Chip"
    )

    # JNJ - 强生
    mgr.journal.add_or_update_position(
        market="US",
        code="JNJ",
        shares=60,
        price=160.0,
        name="Johnson & Johnson",
        sector="Healthcare",
        strategy_tag="Dividend"
    )

    print("✓ 已添加5只美股持仓")

    # 测试2: 获取仪表盘
    print("\n▶ 步骤2: 获取美股持仓仪表盘")
    dashboard = mgr.get_portfolio_dashboard(market="US")

    print(f"\n持仓概览:")
    print(f"  总市值: ${dashboard['total_market_value']:,.2f}")
    print(f"  总成本: ${dashboard['total_cost']:,.2f}")
    print(f"  浮动盈亏: ${dashboard['unrealized_pnl']:,.2f} ({dashboard['unrealized_pnl_pct']:.2%})")
    print(f"  持仓数量: {dashboard['position_count']}")
    print(f"  盈利/亏损: {dashboard['profitable_count']}/{dashboard['losing_count']}")

    print(f"\n行业分布:")
    for sector, weight in dashboard['sector_distribution'].items():
        print(f"  {sector}: {weight:.1%}")

    print(f"\nTop 5持仓:")
    for i, pos in enumerate(dashboard['top_positions'], 1):
        print(f"  {i}. {pos['code']}: {pos['weight']:.1%} (${pos['market_value']:,.2f})")

    # 测试3: 持仓详情分析
    print("\n▶ 步骤3: 分析单个持仓 (AAPL)")
    analysis = mgr.analyze_holding("AAPL", "US", include_fundamental=True)

    print(f"\n{analysis['code']} - {analysis['name']}")
    print(f"  持仓: {analysis['holding_info']['shares']}股 @ ${analysis['holding_info']['average_cost']:.2f}")
    print(f"  市值: ${analysis['holding_info']['market_value']:,.2f}")
    print(f"  盈亏: {analysis['holding_info']['unrealized_pnl_pct']:.2%}")
    print(f"  行业: {analysis['holding_info']['sector']}")

    if analysis.get('fundamental_score'):
        fs = analysis['fundamental_score']
        print(f"\n  基本面评分:")
        print(f"    综合得分: {fs.get('综合得分', 0)}/100")
        print(f"    评级: {fs.get('评级', 'N/A')}")
        print(f"    盈利能力: {fs.get('盈利能力', 0)}")
        print(f"    成长性: {fs.get('成长性', 0)}")

    print(f"\n  操作建议: {analysis['recommendation']}")

    print("\n✅ PortfolioManager测试完成!")
