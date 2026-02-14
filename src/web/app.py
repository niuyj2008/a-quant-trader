"""
股票量化策略决策支持系统 - Web界面（增强版）

核心功能面板:
  1. 📊 个股交易策略 — 输入任意股票，生成买/卖/持有建议
  2. 💼 当前持仓策略 — 管理持仓，自动给出调仓建议
  3. 🎯 个股推荐 — 全市场扫描Top推荐
  4. 📈 行情分析 — K线图、技术指标
  5. 🔬 因子研究 — 因子计算与分析
  6. 🧪 策略回测 — 历史回测验证
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

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
            """)

        st.markdown("---")
        st.caption(f"系统版本 v2.1 | {datetime.now().strftime('%Y-%m-%d')}")

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

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("当前价格", f"{'$' if market_code == 'US' else '¥'}{latest['close']:.2f}",
                          f"{change_pct:+.2f}%")
                m2.metric("最高价", f"{latest['high']:.2f}")
                m3.metric("最低价", f"{latest['low']:.2f}")
                m4.metric("成交量", f"{latest['volume']:,.0f}")

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
    currency = "$" if "US" in str(st.session_state.get('market', 'CN')) else "¥"
    if report.current_price:
        cols[0].metric("当前价", f"{currency}{report.current_price:.2f}")
    if report.stop_loss_price:
        cols[1].metric("止损价", f"{currency}{report.stop_loss_price:.2f}")
    if report.support_price:
        cols[2].metric("支撑位", f"{currency}{report.support_price:.2f}")
    if report.resistance_price:
        cols[3].metric("阻力位", f"{currency}{report.resistance_price:.2f}")


# ==================== Tab2: 持仓策略 ====================
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

    # 分析每只持仓
    st.subheader("📊 持仓分析")

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
                    summary_data.append({
                        "代码": report.code,
                        "成本价": f"{cost:.2f}",
                        "现价": f"{report.current_price:.2f}" if report.current_price else "-",
                        "盈亏": f"{pnl:+.1f}%",
                        "建议": f"{action_emoji} {report.action_cn}",
                        "信号强度": f"{report.confidence:.0f}",
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


# ==================== Tab3: 个股推荐 ====================
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
        for report in recommendations[:5]:  # 显示Top5的详细分析
            with st.expander(f"#{rec_data[recommendations.index(report)]['排名']} {report.code} — {report.action_cn}({report.confidence:.0f}分)"):
                _render_strategy_detail(report, market_code)

    # 历史推荐回顾
    st.markdown("---")
    st.subheader("📜 历史推荐回顾")
    journal = get_journal()
    hist_recs = journal.get_recommendations(market_code, limit=20)
    if not hist_recs.empty:
        display_cols = ['date', 'code', 'name', 'strategy', 'score', 'confidence',
                        'price_at_recommend', 'return_1w', 'reason']
        available_cols = [c for c in display_cols if c in hist_recs.columns]
        st.dataframe(hist_recs[available_cols], use_container_width=True, hide_index=True)

        # 推荐绩效
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


# ==================== Tab5: 因子研究 ====================
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
            latest_idx = df.index[-1]
            latest_tech = factored.iloc[-1].to_dict()
            
            combined_factors = latest_tech.copy()
            
            # 合并基本面
            if financial:
                combined_factors.update(financial)
            
            # 合并宏观 (取最新值)
            if macro:
                for k, v in macro.items():
                    if isinstance(v, pd.Series) and not v.empty:
                         combined_factors[k] = v.iloc[-1]
            
            # 合并情绪
            if sentiment:
                # 处理 A股情绪数据 DataFrame
                if 'margin_balance' in sentiment and isinstance(sentiment['margin_balance'], pd.DataFrame):
                     mb = sentiment['margin_balance']
                     if not mb.empty:
                         # 融资买入额
                         combined_factors['margin_balance_change'] = mb.iloc[-1].get('融资买入额')
                
                # 处理 北向资金
                if 'northbound_flow' in sentiment and isinstance(sentiment['northbound_flow'], pd.DataFrame):
                    nf = sentiment['northbound_flow']
                    if not nf.empty:
                        # 获取最新一天的净买入额
                        # 注意：north_money 单位通常是元，可能需要格式化为亿元
                        val = nf.iloc[-1].get('north_money')
                        if val is not None:
                            combined_factors['northbound_flow'] = val / 1e8  # 转换为亿元
                
                # 处理 VIX
                if 'vix' in sentiment and isinstance(sentiment['vix'], pd.DataFrame):
                    vix_df = sentiment['vix']
                    if not vix_df.empty:
                        try:
                            # Handle MultiIndex columns (Price, Ticker) or simple DataFrame
                            if isinstance(vix_df.columns, pd.MultiIndex):
                                # Extract 'Close' level
                                if 'Close' in vix_df.columns.get_level_values(0):
                                    val_s = vix_df['Close'].iloc[-1]
                                    # If multiple tickers (unexpected), take first
                                    if isinstance(val_s, pd.Series):
                                        val = val_s.iloc[0]
                                    else:
                                        val = val_s
                                    combined_factors['vix'] = val
                            else:
                                # Normal DataFrame
                                if 'Close' in vix_df.columns:
                                    val = vix_df['Close'].iloc[-1]
                                    combined_factors['vix'] = val
                        except Exception:
                            pass
                
                # 处理 美股 10Y Yield
                if 'us_yield' in sentiment and isinstance(sentiment['us_yield'], pd.DataFrame):
                    us_yield_df = sentiment['us_yield']
                    if not us_yield_df.empty:
                        try:
                            if isinstance(us_yield_df.columns, pd.MultiIndex):
                                if 'Close' in us_yield_df.columns.get_level_values(0):
                                    val_s = us_yield_df['Close'].iloc[-1]
                                    if isinstance(val_s, pd.Series):
                                        val = val_s.iloc[0]
                                    else:
                                        val = val_s
                                    combined_factors['interest_rate'] = val
                            else:
                                if 'Close' in us_yield_df.columns:
                                    val = us_yield_df['Close'].iloc[-1]
                                    combined_factors['interest_rate'] = val
                        except Exception:
                            pass



            # 因子分类展示
            st.subheader("📊 因子值一览")
            
            # 手动映射一些别名以匹配 FACTOR_CATEGORIES 中的键
            if 'gdp' in combined_factors: combined_factors['gdp_growth'] = combined_factors['gdp']
            if 'm2' in combined_factors: combined_factors['m2_growth'] = combined_factors['m2']

            # 根据市场调整展示类别
            import copy
            display_categories = copy.deepcopy(FACTOR_CATEGORIES)
            if market_code == "US":
                # 美股展示调整
                display_categories["宏观经济"]["factors"] = ["interest_rate"] # 仅展示利率
                display_categories["市场情绪"]["factors"] = ["vix"] # 仅展示VIX
                display_categories["宏观经济"]["description"] = "美联储利率/国债收益率"
                display_categories["市场情绪"]["description"] = "恐慌指数 (VIX)"

            for cat_name, cat_info in display_categories.items():
                with st.expander(f"**{cat_name}** — {cat_info['description']}"):
                    data = []
                    for f in cat_info['factors']:
                        val = combined_factors.get(f)
                        if val is not None and pd.notna(val):
                            # 格式化数值
                            if isinstance(val, (int, float)):
                                if f == 'northbound_flow':
                                    val_str = f"{val:.2f}亿"
                                else:
                                    val_str = f"{val:.4f}"
                            else:
                                val_str = str(val)
                            data.append({"因子": f, "当前值": val_str})
                        else:
                            # 尝试模糊匹配 (比如 'pe' 在 financial 中可能是 'pe' 或 '市盈率')
                            pass
                            
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


# ==================== Tab6: 策略回测 ====================
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
                    # 基准
                    bm_cum = (1 + weekly['close'].pct_change()).cumprod().iloc[20:]
                    fig.add_trace(go.Scatter(x=bm_cum.index, y=bm_cum.values,
                                             name='买入持有', line=dict(dash='dash')))
                    fig.update_layout(title="策略收益 vs 买入持有", height=400,
                                      yaxis_title="累计收益倍数")
                    st.plotly_chart(fig, use_container_width=True)

                    # 绩效
                    total_ret = cumulative - 1
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("总收益", f"{total_ret:.2%}")
                    buy_weeks = len(results_df[results_df['action'].isin(['买入', '加仓'])])
                    col2.metric("买入周数", f"{buy_weeks}/{len(results_df)}")
                    win_weeks = len(results_df[(results_df['position'] == '持有') & (results_df['week_return'] > 0)])
                    total_hold = len(results_df[results_df['position'] == '持有'])
                    col3.metric("持仓胜率", f"{win_weeks/total_hold:.1%}" if total_hold > 0 else "N/A")
                    col4.metric("回测周数", len(results_df))

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

    # 主标签页
    tabs = st.tabs([
        "📊 个股策略", "💼 持仓策略", "🎯 个股推荐",
        "📈 行情分析", "🔬 因子研究", "🧪 策略回测",
        "📝 交易记录"
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


if __name__ == "__main__":
    main()
