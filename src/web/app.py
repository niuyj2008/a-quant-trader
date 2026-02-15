"""
股票量化策略决策支持系统 - Web界面（完整版 v3.0）

核心功能面板:
  1. 📊 个股交易策略 — 输入任意股票，生成买/卖/持有建议 + 基本面评分
  2. 💼 当前持仓策略 — 持仓仪表盘、行业分布、调仓计划
  3. 🎯 个股推荐 — 全市场扫描 + 策略胜率对比 + 有效性预警
  4. 📈 行情分析 — K线图、技术指标
  5. 🔬 因子研究 — 因子计算与分析 + 学术因子雷达图
  6. 🧪 策略回测 — 历史回测 + 健康评分 + 专业指标
  7. 📝 交易记录 — 交易明细与统计
  8. 💰 ETF定投 — DCA/VA/再平衡策略回测
  9. 🎯 目标规划 — 目标导向策略推荐 + 蒙特卡洛模拟
  10. ⚡ 策略优化 — ML算法对比 + 策略集成
  11. 📋 专业报告 — 30+核心指标 + 月度收益表 + 回撤分析
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path
import copy

# 添加项目根目录到path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.fetcher import DataFetcher
from src.data.market import MARKET_CN, MARKET_US, get_market, get_stock_pool
from src.factors.factor_engine import FactorEngine, FACTOR_CATEGORIES
from src.strategy.interpretable_strategy import (
    get_strategy, get_all_strategies, multi_strategy_analysis,
    STRATEGY_NAMES, STRATEGY_DESCRIPTIONS, STRATEGY_RISK_LEVELS,
    DecisionReport,
)
from src.trading.trade_journal import TradeJournal

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="📊 量化策略决策支持系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 缓存初始化 ====================
@st.cache_resource
def get_fetcher_v4():
    import src.data.fetcher
    import importlib
    importlib.reload(src.data.fetcher)
    from src.data.fetcher import DataFetcher
    return DataFetcher(source="akshare", use_cache=True)

@st.cache_resource
def get_factor_engine():
    return FactorEngine()

@st.cache_resource
def get_journal():
    return TradeJournal()

@st.cache_resource
def get_portfolio_manager():
    from src.trading.portfolio_manager import PortfolioManager
    return PortfolioManager()

@st.cache_data(ttl=300)
def fetch_stock_data(code: str, start_date: str, market: str = "CN"):
    fetcher = get_fetcher_v4()
    return fetcher.get_daily_data(code, start_date=start_date, market=market)

@st.cache_data(ttl=600)
def fetch_financial_data(code: str, market: str = "CN"):
    fetcher = get_fetcher_v4()
    return fetcher.get_financial_data(code, market=market)

@st.cache_data(ttl=3600)
def fetch_macro_data():
    fetcher = get_fetcher_v4()
    return fetcher.get_macro_data()

@st.cache_data(ttl=3600)
def fetch_sentiment_data(market: str = "CN"):
    fetcher = get_fetcher_v4()
    return fetcher.get_sentiment_data(market)

@st.cache_data(ttl=3600)
def fetch_stock_name(code: str, market: str = "CN") -> str:
    """获取股票名称"""
    try:
        fetcher = get_fetcher_v4()
        stock_list = fetcher.get_stock_list(market)
        match = stock_list[stock_list['code'] == code]
        if not match.empty:
            return match.iloc[0]['name']
    except Exception:
        pass
    return ""


# ==================== 侧边栏 ====================
def render_sidebar():
    with st.sidebar:
        st.title("⚙️ 系统设置")

        # 市场选择
        market = st.selectbox("🌍 市场", ["A股 (CN)", "美股 (US)"],
                             help="选择交易市场")
        market_code = "CN" if "CN" in market else "US"

        # 日期范围
        st.subheader("📅 数据范围")
        years = st.slider("历史数据年数", 1, 10, 3)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        st.markdown("---")

        # 量化术语
        with st.expander("📚 量化术语通俗解释"):
            st.markdown("""
            - **因子**: 影响股价的特征指标（如动量、PE、波动率）
            - **RSI**: 相对强弱指标，>70超买，<30超卖
            - **MACD**: 趋势跟踪指标，金叉买入、死叉卖出
            - **夏普比率**: 风险调整后收益，>1.0表示优秀
            - **回撤**: 从最高点回落的幅度
            - **IC值**: 因子预测力，>0.03有效
            - **Walk-Forward**: 滚动训练+验证，验证策略真实有效性
            - **IRR**: 内部收益率，定投的真实年化收益
            - **蒙特卡洛模拟**: 随机模拟数千种可能结果，评估概率
            """)

        st.markdown("---")
        st.caption(f"系统版本 v3.0 | {datetime.now().strftime('%Y-%m-%d')}")

    return market_code, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


# ==================== Tab1: 个股交易策略 ====================
def render_stock_strategy(market_code, start_date):
    st.header("📊 个股交易策略分析")
    st.markdown("输入任意股票代码，系统将使用**6种策略模型**自动分析并给出买卖建议")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        if market_code == "CN":
            code = st.text_input("🔍 输入A股代码", value="000001", placeholder="如: 000001, 600519")
        else:
            code = st.text_input("🔍 输入美股代码", value="AAPL", placeholder="如: AAPL, MSFT")
    with col2:
        strategy_keys = list(STRATEGY_NAMES.keys())
        selected_strategies = st.multiselect(
            "📋 选择策略",
            strategy_keys,
            default=strategy_keys,
            format_func=lambda x: f"{STRATEGY_NAMES[x]} ({STRATEGY_RISK_LEVELS[x]}风险)"
        )
    with col3:
        analyze_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)

    if analyze_btn and code:
        with st.spinner(f"正在分析 {code}..."):
            try:
                df = fetch_stock_data(code, start_date, market_code)
                if df.empty:
                    st.error(f"❌ 无法获取 {code} 的数据")
                    return

                financial = fetch_financial_data(code, market_code)

                # 各策略分析
                results = {}
                for key in selected_strategies:
                    try:
                        strategy = get_strategy(key)
                        results[key] = strategy.analyze_stock(code, df, financial, name=code)
                    except Exception as e:
                        st.warning(f"策略 {STRATEGY_NAMES[key]} 分析失败: {e}")

                if not results:
                    st.error("所有策略分析均失败")
                    return

                # ---- 当前价格和基本信息 ----
                latest = df.iloc[-1]
                prev_close = df.iloc[-2]['close'] if len(df) > 1 else latest['close']
                change_pct = (latest['close'] - prev_close) / prev_close * 100

                stock_name = fetch_stock_name(code, market_code)
                if stock_name:
                    st.subheader(f"{stock_name}（{code}）")
                else:
                    st.subheader(code)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("当前价格", f"{'$' if market_code == 'US' else '¥'}{latest['close']:.2f}",
                          f"{change_pct:+.2f}%")
                m2.metric("最高价", f"{latest['high']:.2f}")
                m3.metric("最低价", f"{latest['low']:.2f}")
                m4.metric("成交量", f"{latest['volume']:,.0f}")

                st.markdown("---")

                # ---- 基本面评分卡片 (Phase 2/3) ----
                if financial:
                    st.subheader("💎 基本面评分")
                    _render_fundamental_scorecard(financial, market_code)
                    st.markdown("---")

                # ---- 策略结果概览 ----
                st.subheader("📋 策略信号概览")
                overview_data = []
                for key, report in results.items():
                    action_emoji = {"买入": "🟢", "卖出": "🔴", "持有": "🟡",
                                    "加仓": "🔵", "减仓": "🟠", "清仓": "⛔"}.get(report.action_cn, "⚪")
                    overview_data.append({
                        "策略": STRATEGY_NAMES[key],
                        "信号": f"{action_emoji} {report.action_cn}",
                        "信号强度": f"{report.confidence:.0f}/100",
                        "风险等级": STRATEGY_RISK_LEVELS[key],
                        "止损价": f"{report.stop_loss_price:.2f}" if report.stop_loss_price else "-",
                    })
                st.dataframe(pd.DataFrame(overview_data), use_container_width=True, hide_index=True)

                # ---- 综合建议 ----
                buy_count = sum(1 for r in results.values() if r.action in ('buy', 'add'))
                sell_count = sum(1 for r in results.values() if r.action in ('sell', 'reduce'))
                total = len(results)

                if buy_count > total * 0.6:
                    st.success(f"✅ **综合建议: 买入** — {buy_count}/{total}个策略看多")
                elif sell_count > total * 0.6:
                    st.error(f"🔴 **综合建议: 卖出** — {sell_count}/{total}个策略看空")
                else:
                    st.info(f"🟡 **综合建议: 观望** — 多空分歧较大({buy_count}看多, {sell_count}看空)")

                st.markdown("---")

                # ---- 各策略详细分析 ----
                st.subheader("🔍 策略详细分析")
                tabs = st.tabs([STRATEGY_NAMES[k] for k in results.keys()])

                for tab, (key, report) in zip(tabs, results.items()):
                    with tab:
                        _render_strategy_detail(report, market_code)

                # ---- K线图 ----
                st.subheader("📈 行情走势")
                _render_candlestick(df, code)

            except Exception as e:
                st.error(f"分析失败: {e}")
                import traceback
                st.code(traceback.format_exc())


def _render_fundamental_scorecard(financial: dict, market_code: str):
    """渲染基本面四维度评分卡片"""
    # 计算四个维度得分
    scores = {}

    # 盈利能力 (ROE, 净利率)
    roe = financial.get('roe', 0)
    net_margin = financial.get('net_profit_margin', financial.get('net_margin', 0))
    if isinstance(roe, (int, float)) and roe > 0:
        scores['盈利能力'] = min(100, max(0, roe * 3 + (net_margin or 0) * 2))
    else:
        scores['盈利能力'] = 50

    # 成长性 (营收增长, 利润增长)
    rev_growth = financial.get('revenue_growth', financial.get('revenue_yoy', 0))
    profit_growth = financial.get('profit_growth', financial.get('net_profit_yoy', 0))
    if isinstance(rev_growth, (int, float)):
        scores['成长性'] = min(100, max(0, 50 + rev_growth * 1.5 + (profit_growth or 0) * 1.0))
    else:
        scores['成长性'] = 50

    # 估值吸引力 (PE, PB — 越低越好)
    pe = financial.get('pe', financial.get('pe_ttm', 30))
    pb = financial.get('pb', 3)
    if isinstance(pe, (int, float)) and pe > 0:
        pe_score = max(0, min(100, 100 - pe * 1.5))
        pb_score = max(0, min(100, 100 - (pb or 3) * 15))
        scores['估值吸引力'] = (pe_score + pb_score) / 2
    else:
        scores['估值吸引力'] = 50

    # 财务健康 (资产负债率, 现金流)
    debt_ratio = financial.get('debt_ratio', financial.get('asset_liability_ratio', 50))
    if isinstance(debt_ratio, (int, float)):
        scores['财务健康'] = max(0, min(100, 100 - debt_ratio * 0.8))
    else:
        scores['财务健康'] = 50

    # 综合得分
    total_score = sum(scores.values()) / len(scores) if scores else 50
    grade = 'A+' if total_score >= 85 else 'A' if total_score >= 75 else 'B' if total_score >= 60 else 'C' if total_score >= 40 else 'D'

    # 展示
    cols = st.columns(5)
    cols[0].metric("综合评级", grade, f"{total_score:.0f}分")
    for i, (name, score) in enumerate(scores.items()):
        cols[i + 1].metric(name, f"{score:.0f}/100")


def _render_strategy_detail(report: DecisionReport, market_code: str):
    """渲染单个策略的详细分析"""
    col1, col2 = st.columns([1, 1])

    with col1:
        # 因子雷达图
        if report.factor_scores:
            fig = go.Figure()
            names = list(report.factor_scores.keys())
            values = list(report.factor_scores.values())
            values_closed = values + [values[0]]
            names_closed = names + [names[0]]

            fig.add_trace(go.Scatterpolar(
                r=values_closed, theta=names_closed,
                fill='toself', name='因子得分',
                fillcolor='rgba(99, 110, 250, 0.2)',
                line=dict(color='rgb(99, 110, 250)')
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="因子得分雷达图",
                height=400, showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 因子贡献条形图
        if report.factor_contributions:
            contrib_df = pd.DataFrame({
                '因子': list(report.factor_contributions.keys()),
                '贡献度': list(report.factor_contributions.values()),
            }).sort_values('贡献度', ascending=True)

            fig = go.Figure(go.Bar(
                x=contrib_df['贡献度'], y=contrib_df['因子'],
                orientation='h',
                marker_color=['#2ecc71' if v > 0 else '#e74c3c' for v in contrib_df['贡献度']]
            ))
            fig.update_layout(title="因子贡献度分解", height=400, xaxis_title="贡献度")
            st.plotly_chart(fig, use_container_width=True)

    # 决策理由
    st.markdown(f"**操作建议:** {report.action_cn} | **信号强度:** {report.confidence:.0f}/100")

    if report.reasoning:
        st.markdown("**📝 决策理由:**")
        for r in report.reasoning:
            st.markdown(f"- {r}")

    if report.risk_warnings:
        st.markdown("**⚠️ 风险提示:**")
        for w in report.risk_warnings:
            st.warning(w)

    # 关键价位
    cols = st.columns(4)
    currency = "$" if market_code == "US" else "¥"
    if report.current_price:
        cols[0].metric("当前价", f"{currency}{report.current_price:.2f}")
    if report.stop_loss_price:
        cols[1].metric("止损价", f"{currency}{report.stop_loss_price:.2f}")
    if report.support_price:
        cols[2].metric("支撑位", f"{currency}{report.support_price:.2f}")
    if report.resistance_price:
        cols[3].metric("阻力位", f"{currency}{report.resistance_price:.2f}")


# ==================== Tab2: 持仓策略 (增强版) ====================
def render_holding_strategy(market_code, start_date):
    st.header("💼 当前持仓策略")
    st.markdown("管理您的持仓，系统自动分析每只持仓并给出操作建议")

    journal = get_journal()

    # 添加持仓
    with st.expander("➕ 添加/管理持仓", expanded=False):
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        with col1:
            h_code = st.text_input("股票代码", key="h_code", placeholder="如: 000001")
        with col2:
            h_price = st.number_input("买入价格", min_value=0.01, value=10.0, key="h_price")
        with col3:
            h_shares = st.number_input("持仓数量", min_value=1, value=100, key="h_shares")
        with col4:
            if st.button("添加", key="add_holding"):
                journal.add_holding(market_code, h_code, int(h_shares), h_price, name=h_code)
                st.success(f"✅ 已添加 {h_code}")
                st.rerun()

    # 显示持仓
    holdings_df = journal.get_holdings(market_code)

    if holdings_df.empty:
        st.info("📭 暂无持仓。请在上方添加您的持仓信息。")
        st.markdown("**示例持仓（A股）：**")
        demo_data = pd.DataFrame({
            '代码': ['000001', '600519', '300750'],
            '名称': ['平安银行', '贵州茅台', '宁德时代'],
            '建议买入价': [11.50, 1550.0, 180.0],
            '建议数量': [1000, 100, 500],
        })
        st.dataframe(demo_data, hide_index=True)
        return

    # ---- 持仓仪表盘 (Phase 3) ----
    st.subheader("📊 持仓仪表盘")
    try:
        pm = get_portfolio_manager()
        dashboard = pm.get_portfolio_dashboard(market=market_code)

        currency = "$" if market_code == "US" else "¥"
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("总市值", f"{currency}{dashboard.get('total_market_value', 0):,.0f}")
        m2.metric("总成本", f"{currency}{dashboard.get('total_cost', 0):,.0f}")

        unrealized_pnl = dashboard.get('unrealized_pnl', 0)
        unrealized_pnl_pct = dashboard.get('unrealized_pnl_pct', 0)
        m3.metric("浮动盈亏", f"{currency}{unrealized_pnl:+,.0f}",
                  f"{unrealized_pnl_pct:+.2%}")

        m4.metric("持仓数量", f"{dashboard.get('position_count', 0)}只")

        profitable = dashboard.get('profitable_count', 0)
        losing = dashboard.get('losing_count', 0)
        m5.metric("盈/亏", f"{profitable}盈 / {losing}亏")

        # ---- 行业分布图 (Phase 3) ----
        sector_dist = dashboard.get('sector_distribution', {})
        top_positions = dashboard.get('top_positions', [])

        if sector_dist or top_positions:
            col_left, col_right = st.columns(2)

            with col_left:
                if sector_dist:
                    st.markdown("**行业分布**")
                    fig = go.Figure(data=[go.Pie(
                        labels=list(sector_dist.keys()),
                        values=list(sector_dist.values()),
                        hole=0.4
                    )])
                    fig.update_layout(height=300, margin=dict(t=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)

            with col_right:
                if top_positions:
                    st.markdown("**Top 5 持仓**")
                    top_data = []
                    for pos in top_positions[:5]:
                        pnl_pct = pos.get('unrealized_pnl_pct', pos.get('pnl_pct', 0))
                        emoji = "📈" if pnl_pct >= 0 else "📉"
                        top_data.append({
                            "代码": pos.get('code', ''),
                            "名称": pos.get('name', ''),
                            "权重": f"{pos.get('weight', 0):.1%}",
                            "盈亏": f"{emoji} {pnl_pct:+.1%}" if isinstance(pnl_pct, (int, float)) else "-",
                        })
                    st.dataframe(pd.DataFrame(top_data), hide_index=True, use_container_width=True)

    except Exception as e:
        st.warning(f"持仓仪表盘加载失败: {e}")

    st.markdown("---")

    # ---- 持仓分析 ----
    st.subheader("📋 持仓策略分析")

    strategy_key = st.selectbox("分析策略", list(STRATEGY_NAMES.keys()),
                                format_func=lambda x: STRATEGY_NAMES[x], key="hold_strat")
    strategy = get_strategy(strategy_key)

    holdings_info = {}
    for _, row in holdings_df.iterrows():
        holdings_info[row['code']] = {
            'name': row.get('name', row['code']),
            'shares': row['shares'],
            'cost_price': row['cost_price'],
        }

    if st.button("🔄 分析持仓建议", type="primary"):
        with st.spinner("正在分析持仓..."):
            data_dict = {}
            for code in holdings_info:
                try:
                    df = fetch_stock_data(code, start_date, market_code)
                    if not df.empty:
                        data_dict[code] = df
                except Exception:
                    pass

            if data_dict:
                reports = strategy.analyze_portfolio(holdings_info, data_dict)

                # 汇总表
                summary_data = []
                for report in reports:
                    info = holdings_info.get(report.code, {})
                    cost = info.get('cost_price', 0)
                    pnl = (report.current_price - cost) / cost * 100 if cost > 0 and report.current_price else 0
                    action_emoji = {"买入": "🟢", "卖出": "🔴", "持有": "🟡",
                                    "加仓": "🔵", "减仓": "🟠", "清仓": "⛔"}.get(report.action_cn, "⚪")

                    # 止损/止盈提醒 (Phase 3)
                    alert = ""
                    if report.stop_loss_price and report.current_price:
                        if report.current_price <= report.stop_loss_price:
                            alert = "⚠️ 触发止损"
                    if pnl >= 30:
                        alert = "🎯 建议止盈"

                    summary_data.append({
                        "代码": report.code,
                        "成本价": f"{cost:.2f}",
                        "现价": f"{report.current_price:.2f}" if report.current_price else "-",
                        "盈亏": f"{pnl:+.1f}%",
                        "建议": f"{action_emoji} {report.action_cn}",
                        "信号强度": f"{report.confidence:.0f}",
                        "预警": alert,
                        "理由": report.reasoning[0] if report.reasoning else "",
                    })

                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

                # 详细建议
                for report in reports:
                    with st.expander(f"{report.code} - {report.action_cn}"):
                        st.markdown(report.get_reasoning_text())
                        if report.risk_warnings:
                            for w in report.risk_warnings:
                                st.warning(w)

    # ---- 调仓计划 (Phase 3) ----
    st.markdown("---")
    st.subheader("📝 调仓计划生成")
    st.markdown("基于策略推荐，自动生成调仓清单")

    if st.button("生成调仓计划", key="rebalance_plan"):
        with st.spinner("生成调仓计划..."):
            try:
                pm = get_portfolio_manager()
                # 使用等权目标
                codes = list(holdings_info.keys())
                if codes:
                    equal_weight = 1.0 / len(codes)
                    target_weights = {c: equal_weight for c in codes}
                    plan = pm.generate_rebalance_plan(market_code, target_weights)
                    if plan:
                        buy_ops = [p for p in plan if p.get('action') == 'buy']
                        sell_ops = [p for p in plan if p.get('action') == 'sell']

                        if sell_ops:
                            st.markdown("**🔴 卖出操作:**")
                            sell_data = [{"代码": p['code'], "股数": p['shares'],
                                         "金额": f"{p['amount']:.0f}", "原因": p.get('reason', '')}
                                        for p in sell_ops]
                            st.dataframe(pd.DataFrame(sell_data), hide_index=True)

                        if buy_ops:
                            st.markdown("**🟢 买入操作:**")
                            buy_data = [{"代码": p['code'], "股数": p['shares'],
                                        "金额": f"{p['amount']:.0f}", "原因": p.get('reason', '')}
                                       for p in buy_ops]
                            st.dataframe(pd.DataFrame(buy_data), hide_index=True)

                        if not buy_ops and not sell_ops:
                            st.info("当前持仓已接近目标权重，无需调仓")
                    else:
                        st.info("无调仓建议")
            except Exception as e:
                st.warning(f"调仓计划生成失败: {e}")


# ==================== Tab3: 个股推荐 (增强版) ====================
def render_recommendations(market_code, start_date):
    st.header("🎯 个股推荐")
    st.markdown("系统自动扫描市场，推荐综合评分最高的投资标的")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        rec_strategy = st.selectbox("推荐策略", list(STRATEGY_NAMES.keys()),
                                     format_func=lambda x: STRATEGY_NAMES[x], key="rec_strat")
    with col2:
        top_n = st.slider("推荐数量", 5, 30, 10)
    with col3:
        scan_btn = st.button("🔎 开始扫描", type="primary", use_container_width=True)

    if scan_btn:
        stock_pool = get_stock_pool(market_code)
        strategy = get_strategy(rec_strategy)

        progress = st.progress(0)
        status = st.empty()

        data_dict = {}
        financial_dict = {}
        for i, code in enumerate(stock_pool):
            status.text(f"正在获取 {code} ({i+1}/{len(stock_pool)})...")
            progress.progress((i + 1) / len(stock_pool))
            try:
                df = fetch_stock_data(code, start_date, market_code)
                if not df.empty and len(df) > 30:
                    data_dict[code] = df
                    fin = fetch_financial_data(code, market_code)
                    if fin:
                        financial_dict[code] = fin
            except Exception:
                pass

        status.text("正在生成推荐...")
        try:
            recommendations = strategy.scan_market(data_dict, financial_dict, top_n=top_n)
        except Exception as e:
            st.error(f"扫描失败: {e}")
            return

        progress.empty()
        status.empty()

        if not recommendations:
            st.info("本次扫描未产生推荐信号")
            return

        journal = get_journal()

        # 推荐列表
        st.subheader(f"📋 {STRATEGY_NAMES[rec_strategy]} — Top {len(recommendations)} 推荐")

        rec_data = []
        for i, report in enumerate(recommendations):
            action_emoji = {"买入": "🟢", "加仓": "🔵"}.get(report.action_cn, "⚪")
            rec_data.append({
                "排名": i + 1,
                "代码": report.code,
                "信号": f"{action_emoji} {report.action_cn}",
                "综合得分": f"{report.score:.1f}",
                "信号强度": f"{report.confidence:.0f}/100",
                "现价": f"{report.current_price:.2f}" if report.current_price else "-",
                "止损价": f"{report.stop_loss_price:.2f}" if report.stop_loss_price else "-",
                "核心理由": report.reasoning[0] if report.reasoning else "",
            })

            # 记录推荐
            try:
                journal.record_recommendation(
                    market_code, report.code, rec_strategy,
                    report.score, report.confidence,
                    report.reasoning[0] if report.reasoning else "",
                    report.current_price or 0, name=report.name
                )
            except Exception:
                pass

        st.dataframe(pd.DataFrame(rec_data), use_container_width=True, hide_index=True)

        # 详细分析
        st.subheader("📝 详细分析")
        for report in recommendations[:5]:
            with st.expander(f"#{rec_data[recommendations.index(report)]['排名']} {report.code} — {report.action_cn}({report.confidence:.0f}分)"):
                _render_strategy_detail(report, market_code)

    # ---- 策略胜率对比 (Phase 4) ----
    st.markdown("---")
    st.subheader("📊 策略胜率对比分析")

    journal = get_journal()
    try:
        winrate_df = journal.get_strategy_winrate_comparison()
        if winrate_df is not None and not winrate_df.empty:
            # 策略有效性预警 (Phase 4)
            for _, row in winrate_df.iterrows():
                winrate_3m = row.get('winrate_3m', 0.5)
                if isinstance(winrate_3m, (int, float)) and winrate_3m < 0.45:
                    st.warning(f"⚠️ {row.get('strategy', '未知')}策略 3月胜率<45%({winrate_3m:.1%})，策略可能失效!")

            # 对比表格
            display_data = []
            for _, row in winrate_df.iterrows():
                display_data.append({
                    "策略": row.get('strategy', ''),
                    "推荐数": row.get('total_count', 0),
                    "1周胜率": f"{row.get('winrate_1w', 0):.1%}" if pd.notna(row.get('winrate_1w')) else "N/A",
                    "1周均收益": f"{row.get('avg_return_1w', 0):+.2%}" if pd.notna(row.get('avg_return_1w')) else "N/A",
                    "1月胜率": f"{row.get('winrate_1m', 0):.1%}" if pd.notna(row.get('winrate_1m')) else "N/A",
                    "1月均收益": f"{row.get('avg_return_1m', 0):+.2%}" if pd.notna(row.get('avg_return_1m')) else "N/A",
                    "3月胜率": f"{row.get('winrate_3m', 0):.1%}" if pd.notna(row.get('winrate_3m')) else "N/A",
                    "3月均收益": f"{row.get('avg_return_3m', 0):+.2%}" if pd.notna(row.get('avg_return_3m')) else "N/A",
                })
            st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)

            # 策略胜率柱状图 (Phase 4)
            if len(winrate_df) > 0:
                fig = go.Figure()
                strategies = winrate_df.get('strategy', winrate_df.index).tolist()
                for period, label in [('winrate_1w', '1周'), ('winrate_1m', '1月'), ('winrate_3m', '3月')]:
                    vals = winrate_df.get(period, pd.Series([0] * len(winrate_df))).tolist()
                    fig.add_trace(go.Bar(name=f'{label}胜率', x=strategies,
                                         y=[v * 100 if isinstance(v, (int, float)) and pd.notna(v) else 0 for v in vals]))
                fig.add_hline(y=50, line_dash='dash', line_color='gray', annotation_text='50%基准线')
                fig.update_layout(barmode='group', title="各策略多周期胜率对比",
                                  yaxis_title="胜率(%)", height=400)
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("暂无足够的推荐数据进行胜率统计")
    except Exception as e:
        st.caption(f"胜率分析暂不可用: {e}")

    # 历史推荐回顾
    st.markdown("---")
    st.subheader("📜 历史推荐回顾")
    hist_recs = journal.get_recommendations(market_code, limit=20)
    if not hist_recs.empty:
        display_cols = ['date', 'code', 'name', 'strategy', 'score', 'confidence',
                        'price_at_recommend', 'return_1w', 'reason']
        available_cols = [c for c in display_cols if c in hist_recs.columns]
        st.dataframe(hist_recs[available_cols], use_container_width=True, hide_index=True)

        perf = journal.get_recommendation_performance(market_code)
        if perf.get('已回测数', 0) > 0:
            st.metric("推荐胜率", perf.get('胜率', 'N/A'))
    else:
        st.info("暂无历史推荐记录")


# ==================== Tab4: 行情分析 ====================
def render_market_analysis(market_code, start_date):
    st.header("📈 行情分析")

    col1, col2 = st.columns([3, 1])
    with col1:
        if market_code == "CN":
            code = st.text_input("股票代码", value="000001", key="ma_code")
        else:
            code = st.text_input("股票代码", value="AAPL", key="ma_code")
    with col2:
        if st.button("查询", key="ma_query"):
            pass

    if code:
        try:
            df = fetch_stock_data(code, start_date, market_code)
            if df.empty:
                st.warning("无数据")
                return
            _render_candlestick(df, code)

            # 技术指标
            engine = get_factor_engine()
            factored = engine.compute(df, ['rsi_14', 'macd', 'bollinger', 'ma_5', 'ma_20', 'ma_60'])

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("RSI")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=factored.index, y=factored['rsi_14'], name='RSI(14)'))
                fig.add_hline(y=70, line_dash='dash', line_color='red', annotation_text='超买')
                fig.add_hline(y=30, line_dash='dash', line_color='green', annotation_text='超卖')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("MACD")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=factored.index, y=factored['macd'], name='MACD'))
                fig.add_trace(go.Scatter(x=factored.index, y=factored['macd_signal'], name='Signal'))
                colors = ['green' if v >= 0 else 'red' for v in factored['macd_hist']]
                fig.add_trace(go.Bar(x=factored.index, y=factored['macd_hist'],
                                     name='Histogram', marker_color=colors))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"获取数据失败: {e}")


# ==================== Tab5: 因子研究 (增强版) ====================
def render_factor_research(market_code, start_date):
    st.header("🔬 因子研究")

    code = st.text_input("股票代码", value="000001" if market_code == "CN" else "AAPL", key="fr_code")

    if code:
        try:
            df = fetch_stock_data(code, start_date, market_code)
            if df.empty:
                st.warning("无数据")
                return

            engine = get_factor_engine()
            factored = engine.compute_all_core_factors(df)

            # 获取其他维度数据
            financial = fetch_financial_data(code, market_code)
            macro = fetch_macro_data()
            sentiment = fetch_sentiment_data(market_code)

            # 合并最新因子值
            latest_tech = factored.iloc[-1].to_dict()
            combined_factors = latest_tech.copy()

            if financial:
                combined_factors.update(financial)
            if macro:
                for k, v in macro.items():
                    if isinstance(v, pd.Series) and not v.empty:
                         combined_factors[k] = v.iloc[-1]
            if sentiment:
                _merge_sentiment_data(combined_factors, sentiment)

            # ---- 学术因子雷达图 (Phase 9.4) ----
            st.subheader("🎓 学术因子评分")
            try:
                from src.factors.academic_factors import AcademicFactors
                academic = AcademicFactors()
                # 尝试获取市场数据作为基准
                market_idx = "000300" if market_code == "CN" else "SPY"
                try:
                    market_data = fetch_stock_data(market_idx, start_date, market_code)
                except Exception:
                    market_data = df  # fallback

                academic_scores = academic.calculate_comprehensive_score(df, market_data, financial or {})

                if academic_scores:
                    total_score = academic_scores.get('total_score', 50)
                    rank = academic_scores.get('rank', 'C')

                    # 评级卡片
                    cols = st.columns(6)
                    cols[0].metric("综合评级", rank, f"{total_score:.0f}分")
                    cols[1].metric("FF3因子", f"{academic_scores.get('ff3_score', 0):.0f}/30")
                    cols[2].metric("动量", f"{academic_scores.get('momentum_score', 0):.0f}/20")
                    cols[3].metric("质量", f"{academic_scores.get('quality_score', 0):.0f}/30")
                    cols[4].metric("低波动", f"{academic_scores.get('low_vol_score', 0):.0f}/20")
                    cols[5].metric("Beta", f"{academic_scores.get('fama_french', {}).get('MKT', 1.0):.2f}")

                    # 雷达图
                    radar_names = ['FF3因子', '动量', '质量', '低波动']
                    radar_values = [
                        academic_scores.get('ff3_score', 0) / 30 * 100,
                        academic_scores.get('momentum_score', 0) / 20 * 100,
                        academic_scores.get('quality_score', 0) / 30 * 100,
                        academic_scores.get('low_vol_score', 0) / 20 * 100,
                    ]
                    radar_values_closed = radar_values + [radar_values[0]]
                    radar_names_closed = radar_names + [radar_names[0]]

                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=radar_values_closed, theta=radar_names_closed,
                        fill='toself', name='学术因子',
                        fillcolor='rgba(255, 165, 0, 0.2)',
                        line=dict(color='orange')
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        title="学术因子雷达图 (Fama-French + 动量 + 质量 + 低波)",
                        height=400, showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.caption(f"学术因子分析暂不可用: {e}")

            st.markdown("---")

            # 因子分类展示 (原有功能)
            st.subheader("📊 因子值一览")

            if 'gdp' in combined_factors: combined_factors['gdp_growth'] = combined_factors['gdp']
            if 'm2' in combined_factors: combined_factors['m2_growth'] = combined_factors['m2']

            display_categories = copy.deepcopy(FACTOR_CATEGORIES)
            if market_code == "US":
                display_categories["宏观经济"]["factors"] = ["interest_rate"]
                display_categories["市场情绪"]["factors"] = ["vix"]
                display_categories["宏观经济"]["description"] = "美联储利率/国债收益率"
                display_categories["市场情绪"]["description"] = "恐慌指数 (VIX)"

            for cat_name, cat_info in display_categories.items():
                with st.expander(f"**{cat_name}** — {cat_info['description']}"):
                    data = []
                    for f in cat_info['factors']:
                        val = combined_factors.get(f)
                        if val is not None and pd.notna(val):
                            if isinstance(val, (int, float)):
                                if f == 'northbound_flow':
                                    val_str = f"{val:.2f}亿"
                                else:
                                    val_str = f"{val:.4f}"
                            else:
                                val_str = str(val)
                            data.append({"因子": f, "当前值": val_str})
                    if data:
                        st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)
                    else:
                        st.caption("暂无数据")

            # 因子相关性
            st.subheader("📉 因子相关性矩阵")
            numerical_cols = [c for c in factored.columns
                              if c not in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turnover']
                              and factored[c].dtype in ['float64', 'float32']]
            if numerical_cols:
                corr = factored[numerical_cols[:10]].corr()
                fig = go.Figure(data=go.Heatmap(
                    z=corr.values, x=corr.columns, y=corr.index,
                    colorscale='RdBu_r', zmid=0
                ))
                fig.update_layout(height=500, title="因子相关性")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"因子研究失败: {e}")
            import traceback
            st.code(traceback.format_exc())


def _merge_sentiment_data(combined_factors: dict, sentiment: dict):
    """合并市场情绪数据到因子字典"""
    if 'margin_balance' in sentiment and isinstance(sentiment['margin_balance'], pd.DataFrame):
        mb = sentiment['margin_balance']
        if not mb.empty:
            combined_factors['margin_balance_change'] = mb.iloc[-1].get('融资买入额')

    if 'northbound_flow' in sentiment and isinstance(sentiment['northbound_flow'], pd.DataFrame):
        nf = sentiment['northbound_flow']
        if not nf.empty:
            val = nf.iloc[-1].get('north_money')
            if val is not None:
                combined_factors['northbound_flow'] = val / 1e8

    if 'vix' in sentiment and isinstance(sentiment['vix'], pd.DataFrame):
        vix_df = sentiment['vix']
        if not vix_df.empty:
            try:
                if isinstance(vix_df.columns, pd.MultiIndex):
                    if 'Close' in vix_df.columns.get_level_values(0):
                        val_s = vix_df['Close'].iloc[-1]
                        val = val_s.iloc[0] if isinstance(val_s, pd.Series) else val_s
                        combined_factors['vix'] = val
                elif 'Close' in vix_df.columns:
                    combined_factors['vix'] = vix_df['Close'].iloc[-1]
            except Exception:
                pass

    if 'us_yield' in sentiment and isinstance(sentiment['us_yield'], pd.DataFrame):
        us_yield_df = sentiment['us_yield']
        if not us_yield_df.empty:
            try:
                if isinstance(us_yield_df.columns, pd.MultiIndex):
                    if 'Close' in us_yield_df.columns.get_level_values(0):
                        val_s = us_yield_df['Close'].iloc[-1]
                        val = val_s.iloc[0] if isinstance(val_s, pd.Series) else val_s
                        combined_factors['interest_rate'] = val
                elif 'Close' in us_yield_df.columns:
                    combined_factors['interest_rate'] = us_yield_df['Close'].iloc[-1]
            except Exception:
                pass


# ==================== Tab6: 策略回测 (增强版) ====================
def render_backtest(market_code, start_date):
    st.header("🧪 策略回测")
    st.markdown("使用历史数据验证策略表现")

    col1, col2, col3 = st.columns(3)
    with col1:
        code = st.text_input("回测标的", value="000001" if market_code == "CN" else "AAPL", key="bt_code")
    with col2:
        bt_start = st.date_input("开始日期", datetime(2020, 1, 1), key="bt_start")
    with col3:
        bt_end = st.date_input("结束日期", datetime.now(), key="bt_end")

    bt_strategy = st.selectbox("回测策略", list(STRATEGY_NAMES.keys()),
                                format_func=lambda x: STRATEGY_NAMES[x], key="bt_strategy")

    if st.button("开始回测", type="primary", key="bt_run"):
        with st.spinner("回测中..."):
            try:
                df = fetch_stock_data(code, str(bt_start), market_code)
                if df.empty:
                    st.error("无数据")
                    return

                df = df[df.index <= str(bt_end)]
                strategy = get_strategy(bt_strategy)

                # 简化回测: 逐周分析
                weekly = DataFetcher.aggregate_to_weekly(df)
                results = []
                cumulative = 1.0
                max_cum = 1.0
                max_drawdown = 0.0

                for i in range(20, len(weekly)):
                    window = df[df.index <= weekly.index[i]]
                    try:
                        report = strategy.analyze_stock(code, window, name=code)
                        week_return = (weekly.iloc[i]['close'] / weekly.iloc[i-1]['close'] - 1) if i > 0 else 0

                        if report.action in ('buy', 'add') and report.confidence >= 60:
                            cumulative *= (1 + week_return)
                            position = "持有"
                        elif report.action in ('sell', 'reduce'):
                            position = "空仓"
                        else:
                            position = "观望"

                        # 追踪最大回撤
                        max_cum = max(max_cum, cumulative)
                        dd = (cumulative - max_cum) / max_cum
                        max_drawdown = min(max_drawdown, dd)

                        results.append({
                            'date': weekly.index[i],
                            'action': report.action_cn,
                            'confidence': report.confidence,
                            'week_return': week_return,
                            'cumulative': cumulative,
                            'position': position,
                        })
                    except Exception:
                        pass

                if results:
                    results_df = pd.DataFrame(results)

                    # 收益曲线
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=results_df['date'], y=results_df['cumulative'],
                                             name='策略收益', line=dict(width=2)))
                    bm_cum = (1 + weekly['close'].pct_change()).cumprod().iloc[20:]
                    fig.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum.values,
                                             name='买入持有', line=dict(dash='dash')))
                    fig.update_layout(title="策略收益 vs 买入持有", height=400,
                                      yaxis_title="累计收益倍数")
                    st.plotly_chart(fig, use_container_width=True)

                    # ---- 增强绩效指标 (Phase 5/9.5) ----
                    total_ret = cumulative - 1
                    n_weeks = len(results_df)
                    n_years = n_weeks / 52
                    annualized_ret = (cumulative ** (1 / n_years) - 1) if n_years > 0 else 0

                    # 夏普比率
                    hold_returns = results_df[results_df['position'] == '持有']['week_return']
                    if len(hold_returns) > 1:
                        sharpe = (hold_returns.mean() / hold_returns.std()) * np.sqrt(52) if hold_returns.std() > 0 else 0
                    else:
                        sharpe = 0

                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    col1.metric("总收益", f"{total_ret:.2%}")
                    col2.metric("年化收益", f"{annualized_ret:.2%}")
                    col3.metric("夏普比率", f"{sharpe:.2f}")
                    col4.metric("最大回撤", f"{max_drawdown:.2%}")
                    buy_weeks = len(results_df[results_df['action'].isin(['买入', '加仓'])])
                    col5.metric("买入周数", f"{buy_weeks}/{n_weeks}")
                    win_weeks = len(results_df[(results_df['position'] == '持有') & (results_df['week_return'] > 0)])
                    total_hold = len(results_df[results_df['position'] == '持有'])
                    col6.metric("持仓胜率", f"{win_weeks/total_hold:.1%}" if total_hold > 0 else "N/A")

                    # ---- 策略健康评分 (Phase 5) ----
                    st.markdown("---")
                    st.subheader("🏥 策略健康评分")
                    try:
                        from src.backtest.walk_forward import StrategyHealthScorer, WalkForwardResult, WalkForwardWindow
                        # 构建简化的WalkForwardResult
                        mock_window = WalkForwardWindow(
                            window_id=1,
                            train_start=str(bt_start),
                            train_end=str(bt_end),
                            test_start=str(bt_start),
                            test_end=str(bt_end),
                            test_return=total_ret,
                            test_sharpe=sharpe,
                            test_max_drawdown=abs(max_drawdown),
                            test_win_rate=win_weeks / total_hold if total_hold > 0 else 0,
                            n_trades=buy_weeks,
                        )
                        wf_result = WalkForwardResult(
                            windows=[mock_window],
                            total_return=total_ret,
                            annualized_return=annualized_ret,
                            overall_sharpe=sharpe,
                            overall_max_drawdown=abs(max_drawdown),
                        )
                        scorer = StrategyHealthScorer()
                        health = scorer.score(wf_result)

                        h_cols = st.columns(6)
                        h_cols[0].metric("健康评分", f"{health.get('total_score', 0):.0f}/100")
                        h_cols[1].metric("评级", health.get('grade', 'N/A'))
                        subscores = health.get('subscores', {})
                        h_cols[2].metric("收益", f"{subscores.get('avg_return', 0):.0f}")
                        h_cols[3].metric("夏普", f"{subscores.get('sharpe_ratio', 0):.0f}")
                        h_cols[4].metric("回撤控制", f"{subscores.get('max_drawdown', 0):.0f}")
                        h_cols[5].metric("稳定性", f"{subscores.get('stability', 0):.0f}")

                        rec = health.get('recommendation', '')
                        if rec:
                            st.info(f"💡 {rec}")

                    except Exception as e:
                        st.caption(f"健康评分暂不可用: {e}")

            except Exception as e:
                st.error(f"回测失败: {e}")
                import traceback
                st.code(traceback.format_exc())


# ==================== Tab7: 交易记录 ====================
def render_trade_records(market_code):
    st.header("📝 交易记录")

    journal = get_journal()

    tab1, tab2 = st.tabs(["交易明细", "绩效统计"])

    with tab1:
        trades = journal.get_trades(market=market_code, limit=50)
        if trades.empty:
            st.info("暂无交易记录")
        else:
            display_cols = ['date', 'code', 'name', 'action', 'price', 'shares', 'amount', 'strategy', 'reason']
            available = [c for c in display_cols if c in trades.columns]
            st.dataframe(trades[available], use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("交易统计将在有足够交易记录后自动生成")


# ==================== Tab8: ETF定投 (Phase 6) ====================
def render_etf_dip(market_code, start_date):
    st.header("💰 ETF定投策略")
    st.markdown("系统支持**定期定额(DCA)**、**价值平均(VA)**、**智能再平衡**三种定投策略")

    col1, col2, col3 = st.columns(3)
    with col1:
        if market_code == "CN":
            etf_code = st.text_input("ETF代码", value="510300", placeholder="如: 510300, 510500",
                                     key="etf_code")
        else:
            etf_code = st.text_input("ETF代码", value="SPY", placeholder="如: SPY, QQQ",
                                     key="etf_code")
    with col2:
        dip_strategy = st.selectbox("定投策略", ["定期定额(DCA)", "价值平均(VA)", "智能再平衡"],
                                     key="dip_strategy")
    with col3:
        frequency = st.selectbox("定投频率", ["weekly", "biweekly", "monthly"],
                                  format_func=lambda x: {"weekly": "每周", "biweekly": "每两周", "monthly": "每月"}[x],
                                  key="dip_freq")

    col4, col5 = st.columns(2)
    with col4:
        invest_amount = st.number_input("每次投入金额", min_value=100, value=5000, step=500, key="dip_amount")
    with col5:
        dip_years = st.slider("回测年数", 1, 10, 3, key="dip_years")

    if st.button("开始定投回测", type="primary", key="dip_run"):
        with st.spinner("定投回测中..."):
            try:
                dip_start = (datetime.now() - timedelta(days=dip_years * 365)).strftime('%Y-%m-%d')
                df = fetch_stock_data(etf_code, dip_start, market_code)
                if df.empty:
                    st.error(f"无法获取 {etf_code} 的数据")
                    return

                from src.strategy.etf_strategies import ETFDollarCostAveraging, ETFValueAveraging

                # 执行定投回测
                total_invested = 0
                total_shares = 0
                invest_records = []

                if "DCA" in dip_strategy:
                    dca = ETFDollarCostAveraging(frequency=frequency, invest_amount=invest_amount, market=market_code)
                    for date_idx in df.index:
                        date_str = str(date_idx)
                        signals = dca.generate_signals(df.loc[:date_idx], date_str)
                        for sig in signals:
                            total_invested += sig['amount']
                            total_shares += sig['shares']
                            invest_records.append({
                                'date': date_idx,
                                'price': sig['price'],
                                'shares': sig['shares'],
                                'amount': sig['amount'],
                            })

                elif "VA" in dip_strategy:
                    va = ETFValueAveraging(frequency=frequency, target_growth=invest_amount, market=market_code)
                    for date_idx in df.index:
                        date_str = str(date_idx)
                        signals = va.generate_signals(df.loc[:date_idx], date_str)
                        for sig in signals:
                            total_invested += sig['amount']
                            total_shares += sig['shares']
                            invest_records.append({
                                'date': date_idx,
                                'price': sig['price'],
                                'shares': sig['shares'],
                                'amount': sig['amount'],
                            })

                if invest_records:
                    records_df = pd.DataFrame(invest_records)
                    final_price = df.iloc[-1]['close']
                    final_value = total_shares * final_price
                    total_return = (final_value - total_invested) / total_invested if total_invested > 0 else 0

                    # 简化IRR计算
                    n_years_actual = (df.index[-1] - df.index[0]).days / 365.25
                    irr = (final_value / total_invested) ** (1 / n_years_actual) - 1 if n_years_actual > 0 and total_invested > 0 else 0

                    # 结果展示
                    currency = "$" if market_code == "US" else "¥"
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("总投入", f"{currency}{total_invested:,.0f}")
                    m2.metric("最终市值", f"{currency}{final_value:,.0f}")
                    m3.metric("总收益", f"{total_return:+.2%}")
                    m4.metric("IRR年化", f"{irr:+.2%}")
                    m5.metric("定投次数", f"{len(invest_records)}次")

                    # 定投曲线
                    records_df['cumulative_invested'] = records_df['amount'].cumsum()
                    records_df['cumulative_shares'] = records_df['shares'].cumsum()
                    records_df['market_value'] = records_df['cumulative_shares'] * records_df['price']

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=records_df['date'], y=records_df['cumulative_invested'],
                                             name='累计投入', fill='tozeroy'))
                    fig.add_trace(go.Scatter(x=records_df['date'], y=records_df['market_value'],
                                             name='市值', line=dict(width=2, color='orange')))
                    fig.update_layout(title="定投累计投入 vs 市值", height=400,
                                      yaxis_title=f"金额({currency})")
                    st.plotly_chart(fig, use_container_width=True)

                    # 买入价格分布
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=records_df['date'], y=records_df['price'],
                                              mode='markers', name='买入价格',
                                              marker=dict(size=5, color='blue')))
                    avg_price = total_invested / total_shares if total_shares > 0 else 0
                    fig2.add_hline(y=avg_price, line_dash='dash', line_color='red',
                                   annotation_text=f'平均成本 {currency}{avg_price:.2f}')
                    fig2.update_layout(title="定投买入价格记录", height=300)
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("回测期间未产生定投信号")

            except Exception as e:
                st.error(f"定投回测失败: {e}")
                import traceback
                st.code(traceback.format_exc())


# ==================== Tab9: 目标导向推荐 (Phase 7) ====================
def render_goal_planning(market_code, start_date):
    st.header("🎯 目标导向策略推荐")
    st.markdown("设定您的投资目标，系统自动匹配最优策略组合")

    col1, col2, col3 = st.columns(3)
    with col1:
        time_horizon = st.slider("投资期限(年)", 1, 10, 3, key="goal_horizon")
    with col2:
        target_return = st.slider("目标年化收益(%)", 5, 50, 15, key="goal_return") / 100
    with col3:
        risk_tolerance = st.selectbox("风险承受能力",
                                       ["conservative", "moderate", "aggressive"],
                                       format_func=lambda x: {"conservative": "保守型", "moderate": "稳健型", "aggressive": "激进型"}[x],
                                       index=1, key="goal_risk")

    col4, col5 = st.columns(2)
    with col4:
        initial_capital = st.number_input("初始资金", min_value=10000, value=100000, step=10000, key="goal_capital")
    with col5:
        monthly_invest = st.number_input("每月定投(0=一次性投入)", min_value=0, value=0, step=1000, key="goal_monthly")

    if st.button("生成推荐方案", type="primary", key="goal_run"):
        with st.spinner("分析中..."):
            try:
                from src.strategy.goal_based_recommender import quick_recommend

                result = quick_recommend(
                    target_return=target_return,
                    years=time_horizon,
                    risk_tolerance=risk_tolerance,
                    initial_capital=initial_capital,
                )

                if result.get('status') == 'success':
                    strategies = result.get('recommended_strategies', [])

                    if strategies:
                        st.subheader("📋 推荐策略列表")

                        for i, strat in enumerate(strategies):
                            scores = strat.get('scores', {})
                            perf = strat.get('performance', {})
                            prob = strat.get('success_probability', 0)

                            with st.expander(f"#{i+1} {strat['name']} — 达成概率 {prob:.1%}", expanded=(i == 0)):
                                # 评分
                                s_cols = st.columns(5)
                                s_cols[0].metric("总分", f"{scores.get('total', 0):.0f}/100")
                                s_cols[1].metric("收益匹配", f"{scores.get('return_match', 0):.0f}")
                                s_cols[2].metric("风险控制", f"{scores.get('risk_control', 0):.0f}")
                                s_cols[3].metric("稳定性", f"{scores.get('stability', 0):.0f}")
                                s_cols[4].metric("期限匹配", f"{scores.get('horizon_match', 0):.0f}")

                                # 历史表现
                                p_cols = st.columns(4)
                                p_cols[0].metric("年化收益", f"{perf.get('annual_return', 0):.1%}")
                                p_cols[1].metric("夏普比率", f"{perf.get('sharpe_ratio', 0):.2f}")
                                p_cols[2].metric("最大回撤", f"{perf.get('max_drawdown', 0):.1%}")
                                p_cols[3].metric("达成概率", f"{prob:.1%}")

                        # 蒙特卡洛模拟可视化
                        st.markdown("---")
                        st.subheader("📊 蒙特卡洛模拟")

                        best_strat = strategies[0]
                        best_perf = best_strat.get('performance', {})
                        ann_ret = best_perf.get('annual_return', target_return)
                        ann_vol = best_perf.get('volatility', 0.2)

                        # 运行模拟
                        n_sim = 1000
                        n_months = int(time_horizon * 12)
                        monthly_ret = ann_ret / 12
                        monthly_vol = ann_vol / np.sqrt(12)

                        np.random.seed(42)
                        simulations = np.zeros((n_sim, n_months + 1))
                        simulations[:, 0] = initial_capital

                        for t in range(1, n_months + 1):
                            rand_returns = np.random.normal(monthly_ret, monthly_vol, n_sim)
                            simulations[:, t] = simulations[:, t - 1] * (1 + rand_returns) + monthly_invest

                        final_values = simulations[:, -1]
                        target_value = initial_capital * (1 + target_return) ** time_horizon
                        success_count = np.sum(final_values >= target_value)

                        # 模拟结果图
                        fig = go.Figure()
                        # 画部分路径
                        for j in range(min(100, n_sim)):
                            fig.add_trace(go.Scatter(
                                x=list(range(n_months + 1)), y=simulations[j],
                                mode='lines', line=dict(width=0.3, color='rgba(100,149,237,0.15)'),
                                showlegend=False
                            ))
                        # 中位数和分位数
                        median_path = np.median(simulations, axis=0)
                        p10 = np.percentile(simulations, 10, axis=0)
                        p90 = np.percentile(simulations, 90, axis=0)
                        months_range = list(range(n_months + 1))
                        fig.add_trace(go.Scatter(x=months_range, y=median_path,
                                                  name='中位数', line=dict(width=3, color='blue')))
                        fig.add_trace(go.Scatter(x=months_range, y=p90,
                                                  name='90%分位', line=dict(width=1, dash='dot', color='green')))
                        fig.add_trace(go.Scatter(x=months_range, y=p10,
                                                  name='10%分位', line=dict(width=1, dash='dot', color='red')))
                        fig.add_hline(y=target_value, line_dash='dash', line_color='orange',
                                       annotation_text=f'目标: ¥{target_value:,.0f}')
                        fig.update_layout(title=f"蒙特卡洛模拟 ({n_sim}次) — 达成概率 {success_count/n_sim:.1%}",
                                          height=500, xaxis_title="月份", yaxis_title="资产价值")
                        st.plotly_chart(fig, use_container_width=True)

                        # 分布直方图
                        fig2 = go.Figure()
                        fig2.add_trace(go.Histogram(x=final_values, nbinsx=50,
                                                     marker_color='steelblue', name='最终价值分布'))
                        fig2.add_vline(x=target_value, line_dash='dash', line_color='red',
                                        annotation_text='目标值')
                        fig2.add_vline(x=np.median(final_values), line_dash='dash', line_color='blue',
                                        annotation_text='中位数')
                        fig2.update_layout(title="最终资产价值分布", height=350,
                                          xaxis_title="资产价值", yaxis_title="频次")
                        st.plotly_chart(fig2, use_container_width=True)

                    # 推荐报告
                    report_text = result.get('report', '')
                    if report_text:
                        with st.expander("📄 详细推荐报告"):
                            st.markdown(report_text)

                elif result.get('status') == 'no_match':
                    st.warning("未找到匹配您目标的策略。请尝试调整目标参数。")
                else:
                    st.error(f"推荐失败: {result}")

            except Exception as e:
                st.error(f"目标推荐失败: {e}")
                import traceback
                st.code(traceback.format_exc())


# ==================== Tab10: 策略优化 (Phase 9) ====================
def render_strategy_optimization(market_code, start_date):
    st.header("⚡ 策略优化")

    opt_tab1, opt_tab2 = st.tabs(["ML算法对比", "策略集成"])

    # ---- ML算法对比 (Phase 9.1) ----
    with opt_tab1:
        st.subheader("🤖 ML算法性能对比")
        st.markdown("对比 LightGBM、XGBoost、RandomForest、Ridge 四种算法的预测表现")

        code = st.text_input("分析标的", value="000001" if market_code == "CN" else "AAPL", key="ml_code")

        if st.button("运行ML对比", type="primary", key="ml_run"):
            with st.spinner("ML算法对比中（可能需要较长时间）..."):
                try:
                    df = fetch_stock_data(code, start_date, market_code)
                    if df.empty:
                        st.error("无数据")
                        return

                    engine = get_factor_engine()
                    factored = engine.compute_all_core_factors(df)
                    factored = factored.dropna()

                    # 构造目标列
                    factored['return_5d'] = factored['close'].shift(-5) / factored['close'] - 1
                    factored = factored.dropna()

                    factor_cols = [c for c in factored.columns
                                   if c not in ['open', 'high', 'low', 'close', 'volume', 'amount',
                                               'turnover', 'return_5d']
                                   and factored[c].dtype in ['float64', 'float32']]

                    if len(factor_cols) < 3:
                        st.warning("因子数量不足以进行ML对比")
                        return

                    from src.optimization.ml_benchmark import MLAlgorithmBenchmark
                    benchmark = MLAlgorithmBenchmark(factored, factor_cols, target_column='return_5d')
                    comparison = benchmark.run_walk_forward_comparison(n_splits=3)

                    if comparison is not None and not comparison.empty:
                        st.dataframe(comparison, use_container_width=True)

                        # IC对比柱状图
                        fig = go.Figure()
                        ic_col = 'IC均值' if 'IC均值' in comparison.columns else comparison.columns[0]
                        algorithms = comparison.index.tolist()
                        ic_values = comparison[ic_col].tolist()

                        colors = ['#2ecc71' if v == max(ic_values) else '#3498db' for v in ic_values]
                        fig.add_trace(go.Bar(x=algorithms, y=ic_values, marker_color=colors))
                        fig.update_layout(title="ML算法 IC均值对比", yaxis_title="IC均值", height=400)
                        st.plotly_chart(fig, use_container_width=True)

                        # 推荐最优算法
                        best_algo = algorithms[ic_values.index(max(ic_values))]
                        st.success(f"🏆 推荐算法: **{best_algo}** (IC均值最高)")
                    else:
                        st.warning("ML对比未返回有效结果")

                except Exception as e:
                    st.error(f"ML对比失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # ---- 策略集成 (Phase 9.3) ----
    with opt_tab2:
        st.subheader("🔗 策略集成配置")
        st.markdown("组合多个子策略，提升信号稳定性")

        selected_subs = st.multiselect(
            "选择子策略", list(STRATEGY_NAMES.keys()),
            default=list(STRATEGY_NAMES.keys())[:3],
            format_func=lambda x: STRATEGY_NAMES[x],
            key="ensemble_strategies"
        )

        ensemble_method = st.radio("集成方法",
                                    ["voting", "weighted", "dynamic"],
                                    format_func=lambda x: {"voting": "投票法", "weighted": "加权法", "dynamic": "动态加权法"}[x],
                                    horizontal=True, key="ensemble_method")

        ens_code = st.text_input("测试标的", value="000001" if market_code == "CN" else "AAPL", key="ens_code")

        if st.button("运行集成测试", type="primary", key="ens_run"):
            with st.spinner("策略集成测试中..."):
                try:
                    df = fetch_stock_data(ens_code, start_date, market_code)
                    if df.empty:
                        st.error("无数据")
                        return

                    # 适配可解释策略 → EnsembleStrategy 所需的 generate_signals 接口
                    class _StrategyAdapter:
                        def __init__(self, strategy, code, financial=None):
                            self._strategy = strategy
                            self._code = code
                            self._financial = financial
                            self.__class__.__name__ = strategy.__class__.__name__

                        def generate_signals(self, df, date, context=None):
                            report = self._strategy.analyze_stock(self._code, df, self._financial)
                            return [{"action": report.action, "confidence": report.confidence / 100.0,
                                     "reason": "; ".join(report.reasoning[:2])}]

                    financial = fetch_financial_data(ens_code, market_code)
                    strategies = [_StrategyAdapter(get_strategy(k), ens_code, financial) for k in selected_subs]

                    from src.strategy.ensemble_strategy import EnsembleStrategy
                    ensemble = EnsembleStrategy(strategies, method=ensemble_method)

                    date_str = str(df.index[-1])
                    signals = ensemble.generate_signals(df, date_str)

                    if signals:
                        st.subheader("集成信号结果")
                        for sig in signals:
                            action = sig.get('action', 'hold')
                            confidence = sig.get('confidence', 0)
                            reason = sig.get('reason', '')
                            emoji = {"buy": "🟢", "sell": "🔴", "hold": "🟡"}.get(action, "⚪")

                            st.markdown(f"**{emoji} {action.upper()}** | 置信度: {confidence:.1%} | {reason}")

                            # 投票细节
                            if sig.get('voting_details'):
                                st.markdown("**投票详情:**")
                                for detail in sig['voting_details']:
                                    st.markdown(f"  - {detail}")

                            # 当前权重
                            if sig.get('current_weights'):
                                st.markdown("**策略权重:**")
                                weight_data = [{"策略": k, "权重": f"{v:.2%}"} for k, v in sig['current_weights'].items()]
                                st.dataframe(pd.DataFrame(weight_data), hide_index=True)
                    else:
                        st.info("集成策略未产生信号")

                except Exception as e:
                    st.error(f"策略集成失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())


# ==================== Tab11: 专业回测报告 (Phase 9.5) ====================
def render_professional_report(market_code, start_date):
    st.header("📋 专业回测报告")
    st.markdown("生成包含**30+核心指标**的机构级回测报告")

    col1, col2, col3 = st.columns(3)
    with col1:
        code = st.text_input("回测标的", value="000001" if market_code == "CN" else "AAPL", key="pr_code")
    with col2:
        pr_start = st.date_input("开始日期", datetime(2020, 1, 1), key="pr_start")
    with col3:
        pr_end = st.date_input("结束日期", datetime.now(), key="pr_end")

    pr_strategy = st.selectbox("回测策略", list(STRATEGY_NAMES.keys()),
                                format_func=lambda x: STRATEGY_NAMES[x], key="pr_strategy")

    if st.button("生成专业报告", type="primary", key="pr_run"):
        with st.spinner("生成专业报告中..."):
            try:
                df = fetch_stock_data(code, str(pr_start), market_code)
                if df.empty:
                    st.error("无数据")
                    return

                df = df[df.index <= str(pr_end)]
                strategy = get_strategy(pr_strategy)

                # 执行回测生成权益曲线
                weekly = DataFetcher.aggregate_to_weekly(df)
                equity = [1.0]
                trades_list = []

                for i in range(20, len(weekly)):
                    window = df[df.index <= weekly.index[i]]
                    try:
                        report = strategy.analyze_stock(code, window, name=code)
                        week_return = (weekly.iloc[i]['close'] / weekly.iloc[i-1]['close'] - 1) if i > 0 else 0

                        if report.action in ('buy', 'add') and report.confidence >= 60:
                            equity.append(equity[-1] * (1 + week_return))
                            trades_list.append({
                                'date': weekly.index[i],
                                'action': report.action,
                                'price': weekly.iloc[i]['close'],
                            })
                        else:
                            equity.append(equity[-1])
                    except Exception:
                        equity.append(equity[-1])

                equity_series = pd.Series(equity[1:], index=weekly.index[20:])

                # 使用专业回测报告引擎
                from src.backtest.professional_report import ProfessionalBacktestReport

                backtest_result = {
                    'equity_curve': equity_series,
                    'trades': trades_list,
                    'holdings': pd.DataFrame(),
                    'metrics': {},
                }

                # 基准（传入价格序列，与equity_curve对齐，报告内部会自行计算收益率）
                benchmark = weekly['close'].iloc[20:]

                pro_report = ProfessionalBacktestReport(backtest_result, benchmark_data=benchmark)
                metrics = pro_report.calculate_all_metrics()

                # ---- 核心指标面板 ----
                st.subheader("📊 核心指标")

                # 收益指标
                st.markdown("**收益指标**")
                r_cols = st.columns(5)
                r_cols[0].metric("总收益率", f"{metrics.get('总收益率', 0):.2%}")
                r_cols[1].metric("年化收益率", f"{metrics.get('年化收益率', 0):.2%}")
                r_cols[2].metric("CAGR", f"{metrics.get('CAGR', 0):.2%}")
                r_cols[3].metric("日均收益率", f"{metrics.get('日均收益率', 0):.4%}")
                r_cols[4].metric("正收益月占比", f"{metrics.get('正收益月份占比', 0):.1%}")

                # 风险指标
                st.markdown("**风险指标**")
                k_cols = st.columns(5)
                k_cols[0].metric("年化波动率", f"{metrics.get('年化波动率', 0):.2%}")
                k_cols[1].metric("最大回撤", f"{metrics.get('最大回撤', 0):.2%}")
                k_cols[2].metric("VaR(95%)", f"{metrics.get('VaR(95%)', 0):.2%}")
                k_cols[3].metric("CVaR(95%)", f"{metrics.get('CVaR(95%)', 0):.2%}")
                k_cols[4].metric("下行波动率", f"{metrics.get('下行波动率', 0):.2%}")

                # 风险调整收益
                st.markdown("**风险调整收益**")
                s_cols = st.columns(5)
                s_cols[0].metric("夏普比率", f"{metrics.get('夏普比率', 0):.2f}")
                s_cols[1].metric("Sortino比率", f"{metrics.get('Sortino比率', 0):.2f}")
                s_cols[2].metric("Calmar比率", f"{metrics.get('Calmar比率', 0):.2f}")
                s_cols[3].metric("Alpha", f"{metrics.get('Alpha', 0):.2%}")
                s_cols[4].metric("Beta", f"{metrics.get('Beta', 0):.2f}")

                # 交易指标
                st.markdown("**交易指标**")
                t_cols = st.columns(5)
                t_cols[0].metric("交易次数", f"{metrics.get('交易次数', 0)}")
                t_cols[1].metric("胜率", f"{metrics.get('胜率', 0):.1%}")
                t_cols[2].metric("盈亏比", f"{metrics.get('盈亏比', 0):.2f}")
                t_cols[3].metric("最大连续盈利", f"{metrics.get('最大连续盈利', 0)}")
                t_cols[4].metric("最大连续亏损", f"{metrics.get('最大连续亏损', 0)}")

                st.markdown("---")

                # ---- 权益曲线 ----
                st.subheader("📈 权益曲线")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=equity_series.index, y=equity_series.values,
                                         name='策略净值', line=dict(width=2)))
                fig.update_layout(height=400, yaxis_title="净值")
                st.plotly_chart(fig, use_container_width=True)

                # ---- 月度收益表 ----
                st.subheader("📅 月度收益表")
                try:
                    monthly_table = pro_report.generate_monthly_returns_table()
                    if monthly_table is not None and not monthly_table.empty:
                        # 热力表格: 正绿负红
                        styled = monthly_table.style.map(
                            lambda v: f"background-color: {'#d4edda' if isinstance(v, (int, float)) and v > 0 else '#f8d7da' if isinstance(v, (int, float)) and v < 0 else ''}"
                        ).format("{:.2%}", na_rep="-")
                        st.dataframe(styled, use_container_width=True)
                except Exception as e:
                    st.caption(f"月度收益表生成失败: {e}")

                # ---- 回撤分析 ----
                st.subheader("📉 Top 回撤事件")
                try:
                    drawdowns = pro_report.analyze_drawdowns()
                    if drawdowns:
                        dd_data = []
                        for j, dd in enumerate(drawdowns[:5]):
                            dd_data.append({
                                "排名": j + 1,
                                "开始": str(dd.get('start_date', ''))[:10],
                                "谷底": str(dd.get('min_date', ''))[:10],
                                "恢复": str(dd.get('end_date', ''))[:10],
                                "深度": f"{dd.get('depth', 0):.2%}",
                                "持续(天)": dd.get('duration', 0),
                                "恢复(天)": dd.get('recovery_time', 0),
                            })
                        st.dataframe(pd.DataFrame(dd_data), hide_index=True, use_container_width=True)
                except Exception as e:
                    st.caption(f"回撤分析失败: {e}")

            except Exception as e:
                st.error(f"报告生成失败: {e}")
                import traceback
                st.code(traceback.format_exc())


# ==================== K线图渲染 ====================
def _render_candlestick(df, title=""):
    """渲染K线图"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='K线'
    ), row=1, col=1)

    # 均线
    for period, color in [(5, '#f39c12'), (20, '#3498db'), (60, '#e74c3c')]:
        ma = df['close'].rolling(period).mean()
        fig.add_trace(go.Scatter(x=df.index, y=ma, name=f'MA{period}',
                                  line=dict(width=1, color=color)), row=1, col=1)

    # 成交量
    colors = ['#e74c3c' if df['close'].iloc[i] >= df['open'].iloc[i] else '#2ecc71'
              for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='成交量',
                          marker_color=colors), row=2, col=1)

    fig.update_layout(
        title=f"📈 {title}", height=600,
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)


# ==================== 主入口 ====================
def main():
    market_code, start_date, end_date = render_sidebar()

    # 存储到session
    st.session_state['market'] = market_code

    # 主标签页 (11个Tab)
    tabs = st.tabs([
        "📊 个股策略", "💼 持仓策略", "🎯 个股推荐",
        "📈 行情分析", "🔬 因子研究", "🧪 策略回测",
        "📝 交易记录",
        "💰 ETF定投", "🎯 目标规划", "⚡ 策略优化",
        "📋 专业报告",
    ])

    with tabs[0]:
        render_stock_strategy(market_code, start_date)
    with tabs[1]:
        render_holding_strategy(market_code, start_date)
    with tabs[2]:
        render_recommendations(market_code, start_date)
    with tabs[3]:
        render_market_analysis(market_code, start_date)
    with tabs[4]:
        render_factor_research(market_code, start_date)
    with tabs[5]:
        render_backtest(market_code, start_date)
    with tabs[6]:
        render_trade_records(market_code)
    with tabs[7]:
        render_etf_dip(market_code, start_date)
    with tabs[8]:
        render_goal_planning(market_code, start_date)
    with tabs[9]:
        render_strategy_optimization(market_code, start_date)
    with tabs[10]:
        render_professional_report(market_code, start_date)


if __name__ == "__main__":
    main()
