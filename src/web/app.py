"""
股票量化策略决策支持系统 - Web界面（v4.0 整合版）

核心功能面板（7个Tab）:
  1. 📊 个股分析 — 策略信号 + 行情走势 + 因子研究（一站式）
  2. 💼 持仓管理 — 持仓仪表盘、策略分析、交易记录
  3. 🎯 市场扫描 — 全市场推荐 + 策略胜率对比 + 有效性预警
  4. 🧪 策略回测 — 标准/专业回测报告（可切换）
  5. 💰 ETF定投 — DCA/VA/再平衡策略回测
  6. 🎯 目标规划 — 目标导向策略推荐 + 蒙特卡洛模拟
  7. ⚡ 策略实验室 — ML算法对比 + 策略集成
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
from src.strategy.strategy_router import StrategyRouter
from loguru import logger

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
    """获取基本面数据（PE/PB/ROE/营收增长等）

    自动从yfinance(美股)或akshare(A股)获取真实数据。
    如果获取失败返回空dict，策略会自动降级到纯技术面分析。
    """
    try:
        fetcher = get_fetcher_v4()
        data = fetcher.get_financial_data(code, market=market)
        # 过滤掉值为None/0的字段
        if data:
            data = {k: v for k, v in data.items() if v is not None}
        return data or {}
    except Exception as e:
        logger.debug(f"获取基本面数据失败 {code}: {e}")
        return {}

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
        st.caption(f"系统版本 v4.0 | {datetime.now().strftime('%Y-%m-%d')}")

    return market_code, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


# ==================== 共享工具函数 ====================

def _render_fundamental_scorecard(financial: dict, market_code: str):
    """渲染基本面四维度评分卡片"""
    scores = {}

    # 盈利能力
    roe = financial.get('roe', 0)
    net_margin = financial.get('net_profit_margin', financial.get('net_margin', 0))
    if isinstance(roe, (int, float)) and roe > 0:
        scores['盈利能力'] = min(100, max(0, roe * 3 + (net_margin or 0) * 2))
    else:
        scores['盈利能力'] = 50

    # 成长性
    rev_growth = financial.get('revenue_growth', financial.get('revenue_yoy', 0))
    profit_growth = financial.get('profit_growth', financial.get('net_profit_yoy', 0))
    if isinstance(rev_growth, (int, float)):
        scores['成长性'] = min(100, max(0, 50 + rev_growth * 1.5 + (profit_growth or 0) * 1.0))
    else:
        scores['成长性'] = 50

    # 估值吸引力
    pe = financial.get('pe', financial.get('pe_ttm', 30))
    pb = financial.get('pb', 3)
    if isinstance(pe, (int, float)) and pe > 0:
        pe_score = max(0, min(100, 100 - pe * 1.5))
        pb_score = max(0, min(100, 100 - (pb or 3) * 15))
        scores['估值吸引力'] = (pe_score + pb_score) / 2
    else:
        scores['估值吸引力'] = 50

    # 财务健康
    debt_ratio = financial.get('debt_ratio', financial.get('asset_liability_ratio', 50))
    if isinstance(debt_ratio, (int, float)):
        scores['财务健康'] = max(0, min(100, 100 - debt_ratio * 0.8))
    else:
        scores['财务健康'] = 50

    total_score = sum(scores.values()) / len(scores) if scores else 50
    grade = 'A+' if total_score >= 85 else 'A' if total_score >= 75 else 'B' if total_score >= 60 else 'C' if total_score >= 40 else 'D'

    cols = st.columns(5)
    cols[0].metric("综合评级", grade, f"{total_score:.0f}分")
    for i, (name, score) in enumerate(scores.items()):
        cols[i + 1].metric(name, f"{score:.0f}/100")


def _render_strategy_detail(report: DecisionReport, market_code: str):
    """渲染单个策略的详细分析"""
    col1, col2 = st.columns([1, 1])

    with col1:
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


def _render_candlestick(df, title=""):
    """渲染K线图"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='K线'
    ), row=1, col=1)

    for period, color in [(5, '#f39c12'), (20, '#3498db'), (60, '#e74c3c')]:
        ma = df['close'].rolling(period).mean()
        fig.add_trace(go.Scatter(x=df.index, y=ma, name=f'MA{period}',
                                  line=dict(width=1, color=color)), row=1, col=1)

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


def _run_weekly_backtest(code, df, strategy, bt_start, bt_end):
    """共享的逐周回测引擎，返回 (results_df, equity_series, weekly, stats)"""
    df_filtered = df[df.index <= str(bt_end)]
    weekly = DataFetcher.aggregate_to_weekly(df_filtered)

    results = []
    equity = [1.0]
    trades_list = []
    cumulative = 1.0
    max_cum = 1.0
    max_drawdown = 0.0

    for i in range(20, len(weekly)):
        window = df_filtered[df_filtered.index <= weekly.index[i]]
        try:
            report = strategy.analyze_stock(code, window, name=code)
            week_return = (weekly.iloc[i]['close'] / weekly.iloc[i-1]['close'] - 1) if i > 0 else 0

            if report.action in ('buy', 'add') and report.confidence >= 60:
                cumulative *= (1 + week_return)
                equity.append(equity[-1] * (1 + week_return))
                position = "持有"
                trades_list.append({
                    'date': weekly.index[i],
                    'action': report.action,
                    'price': weekly.iloc[i]['close'],
                })
            elif report.action in ('sell', 'reduce'):
                equity.append(equity[-1])
                position = "空仓"
            else:
                equity.append(equity[-1])
                position = "观望"

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
            equity.append(equity[-1])

    results_df = pd.DataFrame(results) if results else pd.DataFrame()
    equity_series = pd.Series(equity[1:], index=weekly.index[20:]) if len(equity) > 1 else pd.Series()

    n_weeks = len(results_df)
    n_years = n_weeks / 52 if n_weeks > 0 else 0
    total_ret = cumulative - 1
    annualized_ret = (cumulative ** (1 / n_years) - 1) if n_years > 0 else 0

    hold_returns = results_df[results_df['position'] == '持有']['week_return'] if not results_df.empty else pd.Series()
    if len(hold_returns) > 1 and hold_returns.std() > 0:
        sharpe = (hold_returns.mean() / hold_returns.std()) * np.sqrt(52)
    else:
        sharpe = 0

    buy_weeks = len(results_df[results_df['action'].isin(['买入', '加仓'])]) if not results_df.empty else 0
    win_weeks = len(results_df[(results_df['position'] == '持有') & (results_df['week_return'] > 0)]) if not results_df.empty else 0
    total_hold = len(results_df[results_df['position'] == '持有']) if not results_df.empty else 0

    stats = {
        'total_return': total_ret,
        'annualized_return': annualized_ret,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'n_weeks': n_weeks,
        'n_years': n_years,
        'buy_weeks': buy_weeks,
        'win_weeks': win_weeks,
        'total_hold': total_hold,
        'trades_list': trades_list,
    }

    return results_df, equity_series, weekly, stats


# ==================== Tab A: 个股分析（合并原Tab1+Tab4+Tab5） ====================

def render_stock_analysis(market_code, start_date):
    """个股综合分析 — 策略信号 + 行情走势 + 因子研究"""
    st.header("📊 个股综合分析")
    st.markdown("输入任意股票代码，一站式查看**策略信号、行情走势、因子研究**")

    # === 顶部: 共享的股票代码输入 + 策略选择 ===
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        if market_code == "CN":
            code = st.text_input("🔍 输入A股代码", value="000001", placeholder="如: 000001, 600519", key="sa_code")
        else:
            code = st.text_input("🔍 输入美股代码", value="AAPL", placeholder="如: AAPL, MSFT", key="sa_code")
    with col2:
        strategy_keys = list(STRATEGY_NAMES.keys())
        strategy_mode = st.radio(
            "选择模式", ["🤖 智能推荐", "📋 手动选择"],
            horizontal=True, key="sa_mode"
        )
        if strategy_mode == "📋 手动选择":
            selected_strategies = st.multiselect(
                "📋 选择策略",
                strategy_keys,
                default=strategy_keys,
                format_func=lambda x: f"{STRATEGY_NAMES[x]} ({STRATEGY_RISK_LEVELS[x]}风险)",
                key="sa_strategies"
            )
        else:
            selected_strategies = strategy_keys  # 智能推荐模式下仍跑全部，但高亮推荐的
    with col3:
        analyze_btn = st.button("🚀 开始分析", type="primary", use_container_width=True, key="sa_analyze")

    # 策略介绍
    with st.expander("💡 了解6种策略", expanded=False):
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("""
**📊 多因子均衡策略**（中等风险）
综合考虑估值、成长性、动量、波动率等多个维度，像"全科医生"一样给股票做全面体检。
不偏重任何单一指标，追求稳健收益，适合大多数投资者的"默认选择"。

**🚀 动量趋势策略**（中高风险）
"强者恒强"——买入近期涨势好的股票，回避下跌趋势的标的。
类似冲浪，顺着浪的方向走，适合趋势行情中追求超额收益。

**💰 价值投资策略**（低风险）
寻找被市场低估的"便宜好货"——低市盈率、高ROE、财务稳健的公司。
像巴菲特一样用打折价买优质公司，适合有耐心的长线投资者。
""")
        with s2:
            st.markdown("""
**🛡️ 低波动防御策略**（低风险）
专挑股价波动小、走势平稳的股票。行情好时可能跑不赢大盘，
但下跌时回撤更小，适合风险厌恶型投资者，"少赚但也少亏"。

**🔄 均值回归策略**（高风险）
逆向思维——当股价跌得过狠时买入，涨得过高时卖出，赌它"物极必反"。
类似"抄底逃顶"，在震荡市中表现好，但趋势市中有较大风险。

**⚡ 技术突破策略**（中高风险）
捕捉股价放量突破关键价位（前高、箱体上沿）的信号。
突破往往意味着新趋势的开始，适合中短线操作和活跃交易者。
""")

    if not (analyze_btn and code):
        return

    with st.spinner(f"正在分析 {code}..."):
        try:
            df = fetch_stock_data(code, start_date, market_code)
            if df.empty:
                st.error(f"❌ 无法获取 {code} 的数据")
                return

            financial = fetch_financial_data(code, market_code)

            # 股票基本信息
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

            # 智能推荐模式：显示路由建议
            if strategy_mode == "🤖 智能推荐":
                try:
                    router = StrategyRouter()
                    router.load_feedback(market=market_code)
                    routing = router.recommend(code, df, financial_data=financial)
                    rec_col1, rec_col2 = st.columns([3, 2])
                    with rec_col1:
                        st.info(
                            f"🤖 **智能推荐：{STRATEGY_NAMES.get(routing.primary_strategy, routing.primary_strategy)}** "
                            f"(置信度 {routing.confidence:.0f})\n\n"
                            f"理由：{routing.primary_reason}"
                        )
                    with rec_col2:
                        if routing.secondary_strategy:
                            st.caption(
                                f"次选：{STRATEGY_NAMES.get(routing.secondary_strategy, routing.secondary_strategy)} — "
                                f"{routing.secondary_reason}"
                            )
                        if routing.excluded_strategies:
                            st.caption(f"不推荐：{', '.join(STRATEGY_NAMES.get(s, s) for s in routing.excluded_strategies)}")
                        st.caption(f"市场状态：{routing.market_regime}")
                except Exception as e:
                    logger.debug(f"智能推荐失败: {e}")

            # === 四个子Tab (Phase 10: 新增AI分析) ===
            sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
                "📋 策略信号", "📈 行情走势", "🔬 因子研究", "🤖 AI分析"
            ])

            with sub_tab1:
                _render_strategy_signals_panel(code, df, financial, selected_strategies, market_code)

            with sub_tab2:
                _render_market_charts_panel(code, df, market_code)

            with sub_tab3:
                _render_factor_research_panel(code, df, financial, market_code, start_date)

            with sub_tab4:
                _render_ai_analysis_panel(code, df, financial, market_code, stock_name)

        except Exception as e:
            st.error(f"分析失败: {e}")
            import traceback
            st.code(traceback.format_exc())


def _render_strategy_signals_panel(code, df, financial, selected_strategies, market_code):
    """策略信号子Tab — 基本面评分 + 策略信号概览 + 综合建议 + 详细分析"""
    # 基本面评分
    if financial and any(v for v in financial.values()):
        st.subheader("💎 基本面评分")
        _render_fundamental_scorecard(financial, market_code)
        st.markdown("---")
    else:
        st.caption("基本面数据不可用，策略将使用纯技术面分析（value策略可能不准确）")

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

    # 自动记录信号到signal_log
    try:
        from src.data.data_cache import DataCache
        cache = DataCache()
        for key, report in results.items():
            strategy = get_strategy(key)
            cache.save_signal(
                date=report.date if hasattr(report, 'date') and report.date else datetime.now().strftime('%Y-%m-%d'),
                code=code,
                strategy=key,
                action=report.action,
                confidence=report.confidence,
                composite_score=report.score,
                factor_scores=report.factor_scores if hasattr(report, 'factor_scores') else None,
                weights_version=strategy.config_version if hasattr(strategy, 'config_version') else "default",
                price_at_signal=report.current_price if hasattr(report, 'current_price') and report.current_price else 0.0,
                market=market_code,
            )
    except Exception as sig_e:
        logger.debug(f"信号记录失败: {sig_e}")

    # 策略结果概览
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

    # 综合建议
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

    # 各策略详细分析
    st.subheader("🔍 策略详细分析")
    tabs = st.tabs([STRATEGY_NAMES[k] for k in results.keys()])

    for tab, (key, report) in zip(tabs, results.items()):
        with tab:
            _render_strategy_detail(report, market_code)


def _render_market_charts_panel(code, df, market_code):
    """行情走势子Tab — K线图 + RSI + MACD"""
    _render_candlestick(df, code)

    # 技术指标
    try:
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
        st.error(f"技术指标计算失败: {e}")


def _render_factor_research_panel(code, df, financial, market_code, start_date):
    """因子研究子Tab — 学术因子 + 因子值一览 + 相关性矩阵"""
    try:
        engine = get_factor_engine()
        factored = engine.compute_all_core_factors(df)

        macro = fetch_macro_data()
        sentiment = fetch_sentiment_data(market_code)

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

        # 学术因子雷达图
        st.subheader("🎓 学术因子评分")
        try:
            from src.factors.academic_factors import AcademicFactors
            academic = AcademicFactors()
            market_idx = "000300" if market_code == "CN" else "SPY"
            try:
                market_data = fetch_stock_data(market_idx, start_date, market_code)
            except Exception:
                market_data = df

            academic_scores = academic.calculate_comprehensive_score(df, market_data, financial or {})

            if academic_scores:
                total_score = academic_scores.get('total_score', 50)
                rank = academic_scores.get('rank', 'C')

                cols = st.columns(6)
                cols[0].metric("综合评级", rank, f"{total_score:.0f}分")
                cols[1].metric("FF3因子", f"{academic_scores.get('ff3_score', 0):.0f}/30")
                cols[2].metric("动量", f"{academic_scores.get('momentum_score', 0):.0f}/20")
                cols[3].metric("质量", f"{academic_scores.get('quality_score', 0):.0f}/30")
                cols[4].metric("低波动", f"{academic_scores.get('low_vol_score', 0):.0f}/20")
                cols[5].metric("Beta", f"{academic_scores.get('fama_french', {}).get('MKT', 1.0):.2f}")

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

        # 因子分类展示
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


def _render_ai_analysis_panel(code, df, financial, market_code, stock_name):
    """AI分析子Tab — Phase 10新增功能集成面板"""
    st.subheader("🤖 AI增强分析 (Phase 10)")
    st.markdown("整合Grok AI、行研共识、HMM市场状态、行业轮动、智能风控、DL过滤器")

    try:
        from src.web.ai_panel_utils import (
            render_grok_sentiment_panel,
            render_research_consensus_panel,
            render_market_regime_panel,
            render_industry_rotation_panel,
            render_dl_filter_status,
            render_risk_panel
        )

        # 1. 获取Grok AI数据 (可选，取决于配置)
        grok_data = None
        try:
            from src.external.grok_client import GrokClient
            grok_client = GrokClient()
            if grok_client.is_available():
                with st.spinner("🤖 Grok AI分析中..."):
                    grok_sentiment = grok_client.analyze_stock_sentiment(code, market_code)
                    grok_market = grok_client.analyze_market_regime(market_code)
                    if grok_sentiment or grok_market:
                        grok_data = {
                            'sentiment': grok_sentiment,
                            'market': grok_market
                        }
        except Exception as e:
            st.caption(f"Grok AI未启用或不可用: {e}")

        # 渲染Grok面板
        render_grok_sentiment_panel(grok_data)
        st.markdown("---")

        # 2. 行研报告共识
        research_data = None
        try:
            fetcher = get_fetcher_v4()
            research_data = fetcher.get_research_data(code, market_code)
        except Exception as e:
            logger.debug(f"获取行研数据失败: {e}")

        render_research_consensus_panel(research_data, code)
        st.markdown("---")

        # 3. HMM市场状态识别
        regime_info = None
        try:
            from src.factors.macro_factors import MarketRegimeHMM
            detector = MarketRegimeHMM()

            # 获取指数数据
            index_code = "000300" if market_code == "CN" else "^GSPC"
            index_df = fetch_stock_data(index_code,
                                       (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                                       market_code)

            if not index_df.empty:
                regime, confidence, description = detector.detect_regime(index_df)
                regime_info = {
                    'regime': regime,
                    'confidence': confidence,
                    'description': description
                }
        except Exception as e:
            logger.debug(f"HMM市场状态识别失败: {e}")

        if regime_info:
            render_market_regime_panel(regime_info)
            st.markdown("---")

        # 4. 行业轮动
        industry_scores = None
        try:
            from src.factors.industry_factors import IndustryRotationFactor
            industry_factor = IndustryRotationFactor()
            industry_scores = industry_factor.compute_industry_scores(market_code, lookback_days=20)
        except Exception as e:
            logger.debug(f"行业轮动分析失败: {e}")

        if industry_scores:
            render_industry_rotation_panel(industry_scores, top_n=10)
            st.markdown("---")

        # 5. DL信号过滤器状态
        try:
            from src.models.dl_signal_filter import DLSignalFilter
            dl_filter = DLSignalFilter()
            dl_info = dl_filter.get_model_info()
            render_dl_filter_status(dl_info)
            st.markdown("---")
        except Exception as e:
            logger.debug(f"DL过滤器状态获取失败: {e}")

        # 6. 智能风控面板
        risk_alerts = []
        atr_info = None
        correlation_warnings = None
        black_swan = None

        try:
            from src.trading.risk import ATRStopLoss, BlackSwanDetector

            # ATR止损信息
            atr_stop = ATRStopLoss()
            current_price = df.iloc[-1]['close']
            atr_info = atr_stop.get_stop_info(df, current_price)

            # 黑天鹅检测 (使用指数数据)
            index_code = "000300" if market_code == "CN" else "^GSPC"
            try:
                index_df = fetch_stock_data(index_code,
                                           (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                                           market_code)
                if not index_df.empty:
                    detector = BlackSwanDetector()
                    black_swan = detector.check(index_df)
            except:
                pass

            # 基础风险告警
            if financial:
                pe = financial.get('pe')
                if pe and pe > 50:
                    risk_alerts.append(f"⚠️ 市盈率过高 (PE={pe:.1f})")
                if pe and pe < 0:
                    risk_alerts.append(f"⚠️ 公司亏损 (PE={pe:.1f})")

            render_risk_panel(risk_alerts, atr_info, correlation_warnings, black_swan)

        except Exception as e:
            logger.debug(f"智能风控面板渲染失败: {e}")

        # 提示信息
        st.markdown("---")
        st.info("""
        💡 **如何启用完整AI功能**:

        1. **Grok AI分析**:
           - 获取xAI API Key: https://console.x.ai/
           - 设置环境变量: `export XAI_API_KEY="xai-xxxxx"`
           - 修改 `config/settings.yaml` 中 `grok.enabled: true`

        2. **行研报告**: 美股自动启用(yfinance)，A股使用AKShare免费数据

        3. **HMM/行业轮动/智能风控**: 已自动启用，基于免费数据源

        4. **DL信号过滤**: 需要先在"策略实验室"中训练LSTM/Transformer模型
        """)

    except ImportError as e:
        st.error(f"AI分析模块导入失败: {e}")
        st.info("请确认已安装Phase 10相关依赖")
    except Exception as e:
        st.error(f"AI分析面板渲染失败: {e}")
        import traceback
        with st.expander("查看错误详情"):
            st.code(traceback.format_exc())


# ==================== Tab B: 持仓管理（合并原Tab2+Tab7） ====================

def render_portfolio(market_code, start_date):
    """持仓管理 — 持仓总览、策略分析、交易记录"""
    st.header("💼 持仓管理")

    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "📊 持仓总览", "📋 策略分析", "📝 交易记录", "📡 信号日志"
    ])

    with sub_tab1:
        _render_portfolio_dashboard(market_code, start_date)

    with sub_tab2:
        _render_portfolio_strategy(market_code, start_date)

    with sub_tab3:
        _render_trade_records(market_code)

    with sub_tab4:
        _render_signal_log_panel(market_code)


def _render_signal_log_panel(market_code):
    """信号日志面板 — 历史信号记录、统计、胜率分析"""
    st.subheader("📡 信号日志")
    st.markdown("自动记录的所有策略信号，包括操作、信号强度和实际收益。")

    try:
        from src.data.data_cache import DataCache
        cache = DataCache()
        signals = cache.load_signals(market=market_code, limit=5000)

        if signals.empty:
            st.info("暂无信号记录。在「个股分析」中分析股票时会自动记录信号。")
            return

        # 筛选器
        col1, col2, col3 = st.columns(3)
        with col1:
            strategies = signals['strategy'].unique().tolist()
            sel_strategy = st.selectbox("筛选策略", ["全部"] + strategies, key="sl_strategy")
        with col2:
            codes = signals['code'].unique().tolist()
            sel_code = st.selectbox("筛选股票", ["全部"] + codes, key="sl_code")
        with col3:
            sel_action = st.selectbox("筛选操作", ["全部", "buy", "sell", "hold", "add", "reduce"],
                                      key="sl_action")

        filtered = signals.copy()
        if sel_strategy != "全部":
            filtered = filtered[filtered['strategy'] == sel_strategy]
        if sel_code != "全部":
            filtered = filtered[filtered['code'] == sel_code]
        if sel_action != "全部":
            filtered = filtered[filtered['action'] == sel_action]

        # 统计概览
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("信号总数", len(filtered))
        with col2:
            buy_signals = len(filtered[filtered['action'].isin(['buy', 'add'])])
            st.metric("买入信号", buy_signals)
        with col3:
            sell_signals = len(filtered[filtered['action'].isin(['sell', 'reduce'])])
            st.metric("卖出信号", sell_signals)
        with col4:
            if 'return_5d' in filtered.columns:
                filled = filtered[filtered['return_5d'].notna()]
                if len(filled) > 0:
                    buy_filled = filled[filled['action'].isin(['buy', 'add'])]
                    if len(buy_filled) > 0:
                        win_rate = (buy_filled['return_5d'] > 0).mean()
                        st.metric("买入5日胜率", f"{win_rate:.1%}")
                    else:
                        st.metric("买入5日胜率", "-")
                else:
                    st.metric("买入5日胜率", "待回填")
            else:
                st.metric("买入5日胜率", "-")

        # 各策略信号分布
        st.markdown("**各策略信号分布**")
        dist = filtered.groupby(['strategy', 'action']).size().unstack(fill_value=0)
        st.dataframe(dist, use_container_width=True)

        # 信号明细表
        st.markdown("**信号明细 (最近200条)**")
        display_cols = ['date', 'code', 'strategy', 'action', 'confidence',
                        'composite_score', 'price_at_signal']
        for rc in ['return_5d', 'return_10d', 'return_20d']:
            if rc in filtered.columns:
                display_cols.append(rc)

        available_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(
            filtered[available_cols].sort_values('date', ascending=False).head(200),
            use_container_width=True, hide_index=True
        )

        # 待回填 + 回填按钮
        pending = cache.get_pending_backfill_signals(market=market_code)
        if not pending.empty:
            col_bf1, col_bf2 = st.columns([3, 1])
            with col_bf1:
                st.warning(f"有 {len(pending)} 条信号待收益回填")
            with col_bf2:
                if st.button("📊 回填收益", key="sl_backfill"):
                    with st.spinner("正在回填信号收益..."):
                        try:
                            fetcher = get_fetcher_v4()
                            result = cache.batch_backfill_returns(fetcher, market=market_code)
                            st.success(
                                f"回填完成: 总计{result['total']}条, "
                                f"已填{result['filled']}条, 跳过{result['skipped']}条"
                            )
                            st.rerun()
                        except Exception as bf_e:
                            st.error(f"回填失败: {bf_e}")

    except Exception as e:
        st.error(f"信号日志加载失败: {e}")
        import traceback
        st.code(traceback.format_exc())


def _render_portfolio_dashboard(market_code, start_date):
    """持仓总览: 添加持仓 + 仪表盘 + 行业分布 + 调仓计划"""
    st.markdown("管理您的持仓，查看仪表盘和调仓计划")

    journal = get_journal()

    market_label = "美股" if market_code == "US" else "A股"

    # 添加持仓
    with st.expander("➕ 添加/管理持仓", expanded=False):
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        with col1:
            h_code = st.text_input("股票代码", key="h_code",
                                   placeholder="如: AAPL" if market_code == "US" else "如: 000001")
        with col2:
            h_price = st.number_input("买入价格", min_value=0.01, value=10.0, key="h_price")
        with col3:
            h_shares = st.number_input("持仓数量", min_value=1, value=100, key="h_shares")
        with col4:
            if st.button("添加", key="add_holding"):
                if not h_code or not h_code.strip():
                    st.error("请输入股票代码")
                else:
                    h_code_clean = h_code.strip().upper() if market_code == "US" else h_code.strip()
                    journal.add_holding(market_code, h_code_clean, int(h_shares), h_price, name=h_code_clean)
                    st.success(f"✅ 已添加 {h_code_clean}（{market_label}）")
                    st.rerun()

        # 显示当前持仓列表（含删除功能）
        existing_holdings = journal.get_holdings(market_code)
        if not existing_holdings.empty:
            st.markdown(f"**当前{market_label}持仓（{len(existing_holdings)}只）：**")
            for idx, row in existing_holdings.iterrows():
                hcol1, hcol2, hcol3, hcol4 = st.columns([2, 2, 2, 1])
                hcol1.write(f"**{row['code']}** {row.get('name', '')}")
                hcol2.write(f"成本: {row['average_cost']:.2f}")
                hcol3.write(f"数量: {row['total_shares']}")
                if hcol4.button("删除", key=f"del_{row['code']}_{idx}"):
                    journal.remove_holding(market_code, row['code'])
                    st.rerun()

    # 显示持仓
    holdings_df = journal.get_holdings(market_code)

    if holdings_df.empty:
        # 检查另一个市场是否有持仓
        other_market = "CN" if market_code == "US" else "US"
        other_label = "A股" if other_market == "CN" else "美股"
        other_holdings = journal.get_holdings(other_market)

        st.info(f"📭 当前{market_label}市场暂无持仓。请在上方添加您的持仓信息。")
        if not other_holdings.empty:
            st.warning(f"💡 您在 **{other_label}** 市场有 {len(other_holdings)} 只持仓。请在侧边栏切换市场查看。")
        else:
            placeholder_code = "000001" if market_code == "CN" else "AAPL"
            st.markdown(f"**提示**：在上方输入股票代码（如 `{placeholder_code}`）、买入价格和数量后点击「添加」。")
        return

    # 刷新持仓实时价格和行业信息
    with st.spinner("正在获取最新行情..."):
        try:
            fetcher = get_fetcher_v4()
            import sqlite3 as _sqlite3
            for _, row in holdings_df.iterrows():
                code = row['code']
                if not code:
                    continue
                try:
                    df = fetcher.get_daily_data(code, market=market_code)
                    if df is not None and not df.empty:
                        latest_price = float(df['close'].iloc[-1])
                        journal.update_price(market_code, code, latest_price)
                except Exception:
                    pass
                # 尝试填充行业信息（仅当sector为空时）
                if not row.get('sector'):
                    try:
                        if market_code == "US":
                            import yfinance as yf
                            ticker = yf.Ticker(code)
                            info = ticker.info
                            sector = info.get('sector', '')
                            if sector:
                                with _sqlite3.connect(journal.db_path) as conn:
                                    conn.execute(
                                        "UPDATE holdings SET sector=? WHERE market=? AND code=?",
                                        (sector, market_code, code)
                                    )
                                    conn.commit()
                    except Exception:
                        pass
            # 重新加载更新后的持仓（含最新价格）
            holdings_df = journal.get_holdings(market_code)
            # 计算并更新权重
            if not holdings_df.empty:
                total_mv = holdings_df['market_value'].sum()
                if total_mv > 0:
                    with _sqlite3.connect(journal.db_path) as conn:
                        for _, row in holdings_df.iterrows():
                            w = row['market_value'] / total_mv
                            conn.execute(
                                "UPDATE holdings SET weight=? WHERE market=? AND code=?",
                                (w, market_code, row['code'])
                            )
                        conn.commit()
        except Exception as e:
            logger.debug(f"刷新持仓价格失败: {e}")

    # 持仓仪表盘
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

        # 行业分布图 + Top5持仓
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

    # 调仓计划
    st.markdown("---")
    st.subheader("📝 调仓计划生成")
    st.markdown("基于策略推荐，自动生成调仓清单")

    holdings_info = {}
    for _, row in holdings_df.iterrows():
        holdings_info[row['code']] = {
            'name': row.get('name', row['code']),
            'shares': row.get('total_shares', row.get('shares', 0)),
            'cost_price': row.get('average_cost', row.get('cost_price', 0)),
        }

    if st.button("生成调仓计划", key="rebalance_plan"):
        with st.spinner("生成调仓计划..."):
            try:
                pm = get_portfolio_manager()
                codes = list(holdings_info.keys())
                if codes:
                    equal_weight = 1.0 / len(codes)
                    target_weights = {c: equal_weight for c in codes}
                    plan = pm.generate_rebalance_plan(market_code, target_weights)
                    if plan:
                        buy_ops = [p for p in plan if p.get('action') == 'buy']
                        sell_ops = [p for p in plan if p.get('action') == 'sell']

                        _cur = "$" if market_code == "US" else "¥"
                        if sell_ops:
                            st.markdown("**🔴 卖出操作:**")
                            sell_data = [{
                                "代码": p['code'],
                                "当前价格": f"{_cur}{p.get('price', 0):.2f}",
                                "卖出股数": p['shares'],
                                "卖出金额": f"{_cur}{p['amount']:,.0f}",
                                "原因": p.get('reason', ''),
                            } for p in sell_ops]
                            st.dataframe(pd.DataFrame(sell_data), hide_index=True)

                        if buy_ops:
                            st.markdown("**🟢 买入操作:**")
                            buy_data = [{
                                "代码": p['code'],
                                "当前价格": f"{_cur}{p.get('price', 0):.2f}",
                                "买入股数": p['shares'],
                                "买入金额": f"{_cur}{p['amount']:,.0f}",
                                "原因": p.get('reason', ''),
                            } for p in buy_ops]
                            st.dataframe(pd.DataFrame(buy_data), hide_index=True)

                        if not buy_ops and not sell_ops:
                            st.info("当前持仓已接近目标权重，无需调仓")
                    else:
                        st.info("无调仓建议")
            except Exception as e:
                st.warning(f"调仓计划生成失败: {e}")


def _render_portfolio_strategy(market_code, start_date):
    """持仓策略分析: 选策略 → 分析每只持仓 → 建议表"""
    st.markdown("选择策略分析每只持仓，获取操作建议和止损止盈提醒")

    journal = get_journal()
    holdings_df = journal.get_holdings(market_code)

    if holdings_df.empty:
        st.info("📭 暂无持仓，请先在「持仓总览」中添加持仓。")
        return

    holdings_info = {}
    for _, row in holdings_df.iterrows():
        holdings_info[row['code']] = {
            'name': row.get('name', row['code']),
            'shares': row.get('total_shares', row.get('shares', 0)),
            'cost_price': row.get('average_cost', row.get('cost_price', 0)),
        }

    strategy_key = st.selectbox("分析策略", list(STRATEGY_NAMES.keys()),
                                format_func=lambda x: STRATEGY_NAMES[x], key="hold_strat")
    strategy = get_strategy(strategy_key)

    if st.button("🔄 分析持仓建议", type="primary", key="analyze_holdings"):
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

                summary_data = []
                for report in reports:
                    info = holdings_info.get(report.code, {})
                    cost = info.get('cost_price', 0)
                    pnl = (report.current_price - cost) / cost * 100 if cost > 0 and report.current_price else 0
                    action_emoji = {"买入": "🟢", "卖出": "🔴", "持有": "🟡",
                                    "加仓": "🔵", "减仓": "🟠", "清仓": "⛔"}.get(report.action_cn, "⚪")

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

                for report in reports:
                    with st.expander(f"{report.code} - {report.action_cn}"):
                        st.markdown(report.get_reasoning_text())
                        if report.risk_warnings:
                            for w in report.risk_warnings:
                                st.warning(w)


def _render_trade_records(market_code):
    """交易记录 + 绩效统计"""
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
        trades = journal.get_trades(market=market_code, limit=200)
        if trades.empty:
            st.info("暂无交易记录，交易统计将在有足够记录后自动生成")
        else:
            total_trades = len(trades)
            st.metric("总交易次数", total_trades)

            if 'action' in trades.columns:
                buy_trades = len(trades[trades['action'].isin(['buy', '买入', 'add', '加仓'])])
                sell_trades = len(trades[trades['action'].isin(['sell', '卖出', 'reduce', '减仓'])])
                col1, col2 = st.columns(2)
                col1.metric("买入次数", buy_trades)
                col2.metric("卖出次数", sell_trades)

            if 'amount' in trades.columns:
                total_amount = trades['amount'].sum()
                st.metric("总交易金额", f"{total_amount:,.0f}")


# ==================== Tab C: 市场扫描（原Tab3） ====================

def render_market_scan(market_code, start_date):
    """市场扫描推荐 — 全市场扫描 + 策略胜率对比 + 有效性预警"""
    st.header("🎯 市场扫描推荐")
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

        st.subheader("📝 详细分析")
        for report in recommendations[:5]:
            with st.expander(f"#{rec_data[recommendations.index(report)]['排名']} {report.code} — {report.action_cn}({report.confidence:.0f}分)"):
                _render_strategy_detail(report, market_code)

    # 策略胜率对比
    st.markdown("---")
    st.subheader("📊 策略胜率对比分析")

    journal = get_journal()
    try:
        winrate_df = journal.get_strategy_winrate_comparison()
        if winrate_df is not None and not winrate_df.empty:
            for _, row in winrate_df.iterrows():
                winrate_3m = row.get('winrate_3m', 0.5)
                if isinstance(winrate_3m, (int, float)) and winrate_3m < 0.45:
                    st.warning(f"⚠️ {row.get('strategy', '未知')}策略 3月胜率<45%({winrate_3m:.1%})，策略可能失效!")

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


# ==================== Tab D: 策略回测（合并原Tab6+Tab11） ====================

def render_backtest(market_code, start_date):
    """策略回测 — 标准/专业报告"""
    st.header("🧪 策略回测")
    st.markdown("使用历史数据验证策略表现，支持**标准报告**和**专业报告(30+指标)**")

    col1, col2, col3 = st.columns(3)
    with col1:
        code = st.text_input("回测标的", value="000001" if market_code == "CN" else "AAPL", key="bt_code")
    with col2:
        bt_start = st.date_input("开始日期", datetime(2020, 1, 1), key="bt_start")
    with col3:
        bt_end = st.date_input("结束日期", datetime.now(), key="bt_end")

    col4, col5 = st.columns([2, 1])
    with col4:
        bt_strategy = st.selectbox("回测策略", list(STRATEGY_NAMES.keys()),
                                    format_func=lambda x: STRATEGY_NAMES[x], key="bt_strategy")
    with col5:
        report_level = st.radio(
            "报告详细程度",
            ["standard", "professional"],
            format_func=lambda x: {"standard": "标准报告", "professional": "专业报告(30+指标)"}[x],
            horizontal=True, key="bt_report_level"
        )

    if st.button("开始回测", type="primary", key="bt_run"):
        with st.spinner("回测中..."):
            try:
                df = fetch_stock_data(code, str(bt_start), market_code)
                if df.empty:
                    st.error("无数据")
                    return

                strategy = get_strategy(bt_strategy)
                results_df, equity_series, weekly, stats = _run_weekly_backtest(
                    code, df, strategy, bt_start, bt_end
                )

                if results_df.empty:
                    st.warning("回测未产生有效结果")
                    return

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

                # 基础绩效指标
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                col1.metric("总收益", f"{stats['total_return']:.2%}")
                col2.metric("年化收益", f"{stats['annualized_return']:.2%}")
                col3.metric("夏普比率", f"{stats['sharpe']:.2f}")
                col4.metric("最大回撤", f"{stats['max_drawdown']:.2%}")
                col5.metric("买入周数", f"{stats['buy_weeks']}/{stats['n_weeks']}")
                col6.metric("持仓胜率", f"{stats['win_weeks']/stats['total_hold']:.1%}" if stats['total_hold'] > 0 else "N/A")

                # 策略健康评分
                st.markdown("---")
                st.subheader("🏥 策略健康评分")
                try:
                    from src.backtest.walk_forward import StrategyHealthScorer, WalkForwardResult, WalkForwardWindow
                    mock_window = WalkForwardWindow(
                        window_id=1,
                        train_start=str(bt_start),
                        train_end=str(bt_end),
                        test_start=str(bt_start),
                        test_end=str(bt_end),
                        test_return=stats['total_return'],
                        test_sharpe=stats['sharpe'],
                        test_max_drawdown=abs(stats['max_drawdown']),
                        test_win_rate=stats['win_weeks'] / stats['total_hold'] if stats['total_hold'] > 0 else 0,
                        n_trades=stats['buy_weeks'],
                    )
                    wf_result = WalkForwardResult(
                        windows=[mock_window],
                        total_return=stats['total_return'],
                        annualized_return=stats['annualized_return'],
                        overall_sharpe=stats['sharpe'],
                        overall_max_drawdown=abs(stats['max_drawdown']),
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

                # 专业报告内容
                if report_level == "professional":
                    st.markdown("---")
                    st.subheader("📋 专业回测报告")
                    _render_professional_report_content(equity_series, weekly, stats['trades_list'])

            except Exception as e:
                st.error(f"回测失败: {e}")
                import traceback
                st.code(traceback.format_exc())


def _render_professional_report_content(equity_series, weekly, trades_list):
    """专业报告内容: 30+指标 + 月度收益表 + 回撤分析"""
    try:
        from src.backtest.professional_report import ProfessionalBacktestReport

        backtest_result = {
            'equity_curve': equity_series,
            'trades': trades_list,
            'holdings': pd.DataFrame(),
            'metrics': {},
        }

        benchmark = weekly['close'].iloc[20:]

        pro_report = ProfessionalBacktestReport(backtest_result, benchmark_data=benchmark)
        metrics = pro_report.calculate_all_metrics()

        # 核心指标面板
        st.subheader("📊 核心指标（30+）")

        st.markdown("**收益指标**")
        r_cols = st.columns(5)
        r_cols[0].metric("总收益率", f"{metrics.get('总收益率', 0):.2%}")
        r_cols[1].metric("年化收益率", f"{metrics.get('年化收益率', 0):.2%}")
        r_cols[2].metric("CAGR", f"{metrics.get('CAGR', 0):.2%}")
        r_cols[3].metric("日均收益率", f"{metrics.get('日均收益率', 0):.4%}")
        r_cols[4].metric("正收益月占比", f"{metrics.get('正收益月份占比', 0):.1%}")

        st.markdown("**风险指标**")
        k_cols = st.columns(5)
        k_cols[0].metric("年化波动率", f"{metrics.get('年化波动率', 0):.2%}")
        k_cols[1].metric("最大回撤", f"{metrics.get('最大回撤', 0):.2%}")
        k_cols[2].metric("VaR(95%)", f"{metrics.get('VaR(95%)', 0):.2%}")
        k_cols[3].metric("CVaR(95%)", f"{metrics.get('CVaR(95%)', 0):.2%}")
        k_cols[4].metric("下行波动率", f"{metrics.get('下行波动率', 0):.2%}")

        st.markdown("**风险调整收益**")
        s_cols = st.columns(5)
        s_cols[0].metric("夏普比率", f"{metrics.get('夏普比率', 0):.2f}")
        s_cols[1].metric("Sortino比率", f"{metrics.get('Sortino比率', 0):.2f}")
        s_cols[2].metric("Calmar比率", f"{metrics.get('Calmar比率', 0):.2f}")
        s_cols[3].metric("Alpha", f"{metrics.get('Alpha', 0):.2%}")
        s_cols[4].metric("Beta", f"{metrics.get('Beta', 0):.2f}")

        st.markdown("**交易指标**")
        t_cols = st.columns(5)
        t_cols[0].metric("交易次数", f"{metrics.get('交易次数', 0)}")
        t_cols[1].metric("胜率", f"{metrics.get('胜率', 0):.1%}")
        t_cols[2].metric("盈亏比", f"{metrics.get('盈亏比', 0):.2f}")
        t_cols[3].metric("最大连续盈利", f"{metrics.get('最大连续盈利', 0)}")
        t_cols[4].metric("最大连续亏损", f"{metrics.get('最大连续亏损', 0)}")

        st.markdown("---")

        # 权益曲线
        st.subheader("📈 权益曲线")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity_series.index, y=equity_series.values,
                                 name='策略净值', line=dict(width=2)))
        fig.update_layout(height=400, yaxis_title="净值")
        st.plotly_chart(fig, use_container_width=True)

        # 月度收益表
        st.subheader("📅 月度收益表")
        try:
            monthly_table = pro_report.generate_monthly_returns_table()
            if monthly_table is not None and not monthly_table.empty:
                styled = monthly_table.style.map(
                    lambda v: f"background-color: {'#d4edda' if isinstance(v, (int, float)) and v > 0 else '#f8d7da' if isinstance(v, (int, float)) and v < 0 else ''}"
                ).format("{:.2%}", na_rep="-")
                st.dataframe(styled, use_container_width=True)
        except Exception as e:
            st.caption(f"月度收益表生成失败: {e}")

        # 回撤分析
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
        st.error(f"专业报告生成失败: {e}")
        import traceback
        st.code(traceback.format_exc())


# ==================== Tab E: ETF定投（原Tab8，不变） ====================

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
                    va = ETFValueAveraging(target_growth_rate=0.01, base_amount=invest_amount)
                    current_value = 0.0
                    for date_idx in df.index:
                        date_str = str(date_idx)
                        current_price = df.loc[date_idx, 'close']
                        current_value = total_shares * current_price
                        signals = va.generate_signals(df.loc[:date_idx], date_str, current_value)
                        for sig in signals:
                            if sig['action'] == 'buy':
                                total_invested += sig['amount']
                                total_shares += sig['shares']
                            else:
                                total_shares -= sig['shares']
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

                    n_years_actual = (df.index[-1] - df.index[0]).days / 365.25
                    irr = (final_value / total_invested) ** (1 / n_years_actual) - 1 if n_years_actual > 0 and total_invested > 0 else 0

                    currency = "$" if market_code == "US" else "¥"
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("总投入", f"{currency}{total_invested:,.0f}")
                    m2.metric("最终市值", f"{currency}{final_value:,.0f}")
                    m3.metric("总收益", f"{total_return:+.2%}")
                    m4.metric("IRR年化", f"{irr:+.2%}")
                    m5.metric("定投次数", f"{len(invest_records)}次")

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


# ==================== Tab F: 目标规划（原Tab9，不变） ====================

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
                                s_cols = st.columns(5)
                                s_cols[0].metric("总分", f"{scores.get('total', 0):.0f}/100")
                                s_cols[1].metric("收益匹配", f"{scores.get('return_match', 0):.0f}")
                                s_cols[2].metric("风险控制", f"{scores.get('risk_control', 0):.0f}")
                                s_cols[3].metric("稳定性", f"{scores.get('stability', 0):.0f}")
                                s_cols[4].metric("期限匹配", f"{scores.get('horizon_match', 0):.0f}")

                                p_cols = st.columns(4)
                                p_cols[0].metric("年化收益", f"{perf.get('annual_return', 0):.1%}")
                                p_cols[1].metric("夏普比率", f"{perf.get('sharpe_ratio', 0):.2f}")
                                p_cols[2].metric("最大回撤", f"{perf.get('max_drawdown', 0):.1%}")
                                p_cols[3].metric("达成概率", f"{prob:.1%}")

                        # 蒙特卡洛模拟
                        st.markdown("---")
                        st.subheader("📊 蒙特卡洛模拟")

                        best_strat = strategies[0]
                        best_perf = best_strat.get('performance', {})
                        ann_ret = best_perf.get('annual_return', target_return)
                        ann_vol = best_perf.get('volatility', 0.2)

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

                        fig = go.Figure()
                        for j in range(min(100, n_sim)):
                            fig.add_trace(go.Scatter(
                                x=list(range(n_months + 1)), y=simulations[j],
                                mode='lines', line=dict(width=0.3, color='rgba(100,149,237,0.15)'),
                                showlegend=False
                            ))
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


# ==================== Tab G: 策略实验室（原Tab10，不变） ====================

def render_strategy_lab(market_code, start_date):
    st.header("⚡ 策略实验室")

    opt_tab1, opt_tab2, opt_tab3, opt_tab4, opt_tab5 = st.tabs([
        "ML算法对比", "策略集成", "🔬 因子验证", "📊 策略冗余度分析", "🔄 数据驱动工作台"
    ])

    # ML算法对比
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

                        fig = go.Figure()
                        ic_col = 'IC均值' if 'IC均值' in comparison.columns else comparison.columns[0]
                        algorithms = comparison.index.tolist()
                        ic_values = comparison[ic_col].tolist()

                        colors = ['#2ecc71' if v == max(ic_values) else '#3498db' for v in ic_values]
                        fig.add_trace(go.Bar(x=algorithms, y=ic_values, marker_color=colors))
                        fig.update_layout(title="ML算法 IC均值对比", yaxis_title="IC均值", height=400)
                        st.plotly_chart(fig, use_container_width=True)

                        best_algo = algorithms[ic_values.index(max(ic_values))]
                        st.success(f"🏆 推荐算法: **{best_algo}** (IC均值最高)")
                    else:
                        st.warning("ML对比未返回有效结果")

                except Exception as e:
                    st.error(f"ML对比失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # 策略集成
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

                            if sig.get('voting_details'):
                                st.markdown("**投票详情:**")
                                for detail in sig['voting_details']:
                                    st.markdown(f"  - {detail}")

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

    # 因子验证
    with opt_tab3:
        st.subheader("🔬 因子有效性验证")
        st.markdown(
            "对股票池中所有股票计算截面IC/IC_IR，评估每个因子的预测能力。\n\n"
            "- **|IC_IR| > 0.5** = 强有效因子\n"
            "- **|IC_IR| > 0.3** = 中等有效\n"
            "- **|IC_IR| > 0.1** = 弱有效\n"
            "- **|IC_IR| ≤ 0.1** = 无效（可考虑剔除）"
        )

        fv_pool_size = st.selectbox(
            "股票池", ["默认精选池(30只)", "S&P 500训练池(美股)"],
            key="fv_pool"
        )

        if st.button("运行因子验证", type="primary", key="fv_run"):
            with st.spinner("正在计算因子IC（可能需要几分钟）..."):
                try:
                    from src.factors.factor_validator import FactorValidator
                    from src.data.market import get_stock_pool

                    if "S&P 500" in fv_pool_size:
                        pool = get_stock_pool("US", size="sp500")[:50]  # 取前50只加速
                        fv_market = "US"
                    else:
                        pool = get_stock_pool(market_code)
                        fv_market = market_code

                    data_dict = {}
                    progress = st.progress(0)
                    for i, sym in enumerate(pool):
                        try:
                            d = fetch_stock_data(sym, start_date, fv_market)
                            if not d.empty and len(d) >= 60:
                                data_dict[sym] = d
                        except Exception:
                            pass
                        progress.progress((i + 1) / len(pool))

                    if len(data_dict) < 10:
                        st.warning(f"有效数据不足（仅{len(data_dict)}只），请检查数据源")
                    else:
                        validator = FactorValidator()
                        report = validator.validate(data_dict, min_stocks=min(20, max(5, len(data_dict) // 2)))

                        for fwd in [5, 10, 20]:
                            summary = validator.generate_summary(report, forward_days=fwd)
                            if not summary.empty:
                                st.markdown(f"**预测周期: {fwd}日**")
                                st.dataframe(summary, use_container_width=True, hide_index=True)

                        if report.correlation_matrix is not None and not report.correlation_matrix.empty:
                            st.markdown("**因子间相关性矩阵**")
                            fig = px.imshow(
                                report.correlation_matrix,
                                text_auto=True, aspect="auto",
                                color_continuous_scale="RdBu_r",
                                zmin=-1, zmax=1,
                            )
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"因子验证失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # 策略冗余度分析
    with opt_tab4:
        st.subheader("📊 策略冗余度分析")
        st.markdown(
            "分析6个策略之间的信号相关性和收益相关性，识别冗余与互补关系。\n\n"
            "- **信号相关>0.8** → 两个策略冗余，建议合并\n"
            "- **增量夏普≤0** → 该策略对集成无贡献，可移除"
        )

        if st.button("运行冗余度分析", type="primary", key="rd_run"):
            with st.spinner("正在分析策略冗余度..."):
                try:
                    from src.optimization.strategy_redundancy import StrategyRedundancyAnalyzer
                    from src.data.market import get_stock_pool

                    pool = get_stock_pool(market_code)[:20]
                    data_dict = {}
                    progress = st.progress(0)
                    for i, sym in enumerate(pool):
                        try:
                            d = fetch_stock_data(sym, start_date, market_code)
                            if not d.empty and len(d) >= 120:
                                data_dict[sym] = d
                        except Exception:
                            pass
                        progress.progress((i + 1) / len(pool))

                    if len(data_dict) < 5:
                        st.warning("有效数据不足")
                    else:
                        analyzer = StrategyRedundancyAnalyzer()
                        rd_report = analyzer.analyze(data_dict)

                        if rd_report.signal_correlation is not None and not rd_report.signal_correlation.empty:
                            st.markdown("**信号相关性矩阵**")
                            fig = px.imshow(
                                rd_report.signal_correlation,
                                text_auto=True, aspect="auto",
                                color_continuous_scale="RdBu_r",
                                zmin=-1, zmax=1,
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)

                        if rd_report.incremental_value:
                            st.markdown("**增量贡献 (Δ夏普)**")
                            inc_df = pd.DataFrame([
                                {"策略": STRATEGY_NAMES.get(k, k), "增量夏普": v,
                                 "评价": "有效" if v > 0.05 else ("边际" if v > 0 else "冗余")}
                                for k, v in sorted(rd_report.incremental_value.items(),
                                                    key=lambda x: x[1], reverse=True)
                            ])
                            st.dataframe(inc_df, use_container_width=True, hide_index=True)

                        if rd_report.redundant_pairs:
                            st.warning(
                                f"发现 {len(rd_report.redundant_pairs)} 对冗余策略: " +
                                ", ".join(f"{STRATEGY_NAMES.get(k1,k1)}↔{STRATEGY_NAMES.get(k2,k2)}({c:.2f})"
                                          for k1, k2, c in rd_report.redundant_pairs)
                            )

                        if rd_report.recommended_removals:
                            st.error(f"建议移除: {', '.join(STRATEGY_NAMES.get(s, s) for s in rd_report.recommended_removals)}")
                        else:
                            st.success("所有策略均有独立贡献，无冗余")

                except Exception as e:
                    st.error(f"冗余度分析失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # 数据驱动工作台
    with opt_tab5:
        _render_data_driven_workbench(market_code, start_date)


def _render_data_driven_workbench(market_code: str, start_date: str):
    """数据驱动工作台 — 5个步骤组成的可视化流水线"""
    st.subheader("🔄 数据驱动工作台")
    st.markdown(
        "完整的数据驱动闭环：**批量下载** → **因子验证** → **权重优化** → "
        "**阈值搜索** → **配置管理与再训练**"
    )

    # ===== Step 1: 批量下载训练数据 =====
    with st.expander("📥 Step 1: 批量下载训练数据", expanded=False):
        st.markdown("下载股票池的历史数据，构建训练数据集。")

        # --- 已有数据历史 ---
        import sqlite3
        from src.data.data_cache import DataCache
        try:
            _cache = DataCache()
            with sqlite3.connect(_cache.db_path) as _conn:
                _dl_history = pd.read_sql_query("""
                    SELECT market as 市场,
                           COUNT(DISTINCT code) as 股票数,
                           SUM(cnt) as 总行数,
                           MIN(earliest) as 最早日期,
                           MAX(latest) as 最新日期,
                           MAX(last_upd) as 最后更新
                    FROM (
                        SELECT d.market, d.code, COUNT(*) as cnt,
                               MIN(d.date) as earliest, MAX(d.date) as latest,
                               m.last_update as last_upd
                        FROM daily_ohlcv d
                        LEFT JOIN cache_meta m ON d.market = m.market AND d.code = m.code
                        GROUP BY d.market, d.code
                    ) GROUP BY market
                """, _conn)

            if not _dl_history.empty:
                st.markdown("**已缓存数据概览**")
                st.dataframe(_dl_history, use_container_width=True, hide_index=True)

                # 详细列表（可折叠）
                with st.expander("查看各股票详情", expanded=False):
                    _detail = pd.read_sql_query("""
                        SELECT d.market as 市场, d.code as 代码, COUNT(*) as 数据行数,
                               MIN(d.date) as 起始, MAX(d.date) as 截止,
                               m.last_update as 最后更新
                        FROM daily_ohlcv d
                        LEFT JOIN cache_meta m ON d.market = m.market AND d.code = m.code
                        GROUP BY d.market, d.code
                        ORDER BY d.market, d.code
                    """, sqlite3.connect(_cache.db_path))
                    st.dataframe(_detail, use_container_width=True, hide_index=True,
                                 height=300)
            else:
                st.info("暂无缓存数据，请先下载。")
        except Exception:
            pass

        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            dl_market = st.selectbox("市场", ["US", "CN"], index=0 if market_code == "US" else 1,
                                     key="wb_dl_market")
        with col2:
            # 股票池选项根据市场动态变化
            if dl_market == "US":
                _pool_options = ["S&P 500(前100只)", "S&P 500(全量~500只)", "默认精选池(30只)"]
            else:
                _pool_options = ["A股精选池(40只)", "默认精选池(30只)"]
            dl_pool = st.selectbox("股票池", _pool_options, key="wb_dl_pool")
        with col3:
            dl_years = st.slider("历史年数", 3, 15, 10, key="wb_dl_years")

        # 预估数据量 & 已有数据检测
        pool_size_map = {
            "S&P 500(前100只)": 100, "S&P 500(全量~500只)": 500,
            "默认精选池(30只)": 30, "A股精选池(40只)": 40,
        }
        est_stocks = pool_size_map.get(dl_pool, 30)
        est_rows = est_stocks * dl_years * 252

        # 检查已缓存的数量，提示是否需要重新下载
        try:
            with sqlite3.connect(_cache.db_path) as _conn:
                _cached_count = _conn.execute(
                    "SELECT COUNT(DISTINCT code) FROM daily_ohlcv WHERE market=?",
                    [dl_market]
                ).fetchone()[0]
                _cached_rows = _conn.execute(
                    "SELECT COUNT(*) FROM daily_ohlcv WHERE market=?",
                    [dl_market]
                ).fetchone()[0]
        except Exception:
            _cached_count = 0
            _cached_rows = 0

        if _cached_count > 0 and _cached_count >= est_stocks * 0.8:
            st.success(
                f"已有缓存: {dl_market}市场 {_cached_count}只股票 / {_cached_rows:,}行。"
                f"再次下载将自动增量更新（仅补充缺失数据），不会重复下载。"
            )
        elif _cached_count > 0:
            st.info(
                f"已有缓存: {dl_market}市场 {_cached_count}只/{_cached_rows:,}行，"
                f"目标 ~{est_stocks}只。点击下载将增量补充剩余数据。"
            )
        else:
            st.info(f"预估: ~{est_stocks}只股票 × {dl_years}年 ≈ {est_rows:,}行日线数据")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            do_download = st.button("开始下载", type="primary", key="wb_dl_run")
        with col_btn2:
            load_cached = st.button("直接加载已有缓存", key="wb_dl_load",
                                     disabled=(_cached_count == 0))

        if do_download:
            with st.spinner("批量下载中..."):
                try:
                    from src.data.market import get_stock_pool

                    if "S&P 500(全量" in dl_pool:
                        pool = get_stock_pool("US", size="sp500")
                    elif "S&P 500(前100" in dl_pool:
                        pool = get_stock_pool("US", size="sp500")[:100]
                    elif "A股精选池" in dl_pool:
                        pool = get_stock_pool("CN")
                    else:
                        pool = get_stock_pool(dl_market)

                    fetcher = DataFetcher()
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    stats = {"success": 0, "failed": 0, "skipped": 0}

                    def on_progress(current, total, code, status):
                        progress_bar.progress(current / total)
                        stats[status] = stats.get(status, 0) + 1
                        status_text.text(
                            f"[{current}/{total}] {code} - {status} | "
                            f"成功:{stats['success']} 失败:{stats['failed']} 跳过:{stats['skipped']}"
                        )

                    results = fetcher.batch_download(
                        stock_list=pool, years=dl_years,
                        market=dl_market, include_financial=True,
                        progress_callback=on_progress
                    )

                    # 提取data_dict和financial_dict
                    data_dict = {}
                    financial_dict = {}
                    for code, info in results.items():
                        if info.get("status") == "success" and "daily" in info:
                            d = info["daily"]
                            if not d.empty and len(d) >= 60:
                                data_dict[code] = d
                            if "financial" in info and info["financial"]:
                                financial_dict[code] = info["financial"]

                    st.session_state['training_data'] = data_dict
                    st.session_state['training_financial'] = financial_dict

                    st.success(
                        f"下载完成! 有效数据: {len(data_dict)}只股票, "
                        f"财务数据: {len(financial_dict)}只"
                    )

                    total_rows = sum(len(d) for d in data_dict.values())
                    st.metric("总数据行数", f"{total_rows:,}")

                except Exception as e:
                    st.error(f"批量下载失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        if load_cached:
            with st.spinner("加载缓存数据中..."):
                try:
                    with sqlite3.connect(_cache.db_path) as _conn:
                        _codes = [r[0] for r in _conn.execute(
                            "SELECT DISTINCT code FROM daily_ohlcv WHERE market=?",
                            [dl_market]
                        ).fetchall()]

                    data_dict = {}
                    for _code in _codes:
                        try:
                            d = fetch_stock_data(_code, start_date, dl_market)
                            if not d.empty and len(d) >= 60:
                                data_dict[_code] = d
                        except Exception:
                            pass

                    st.session_state['training_data'] = data_dict
                    total_rows = sum(len(d) for d in data_dict.values())
                    st.success(f"已从缓存加载 {len(data_dict)} 只股票, {total_rows:,} 行数据")
                except Exception as e:
                    st.error(f"加载缓存失败: {e}")

        if 'training_data' in st.session_state:
            n = len(st.session_state['training_data'])
            st.success(f"已有训练数据: {n}只股票 (可直接进入Step 2)")

    # ===== Step 2: 因子有效性验证 =====
    with st.expander("🔬 Step 2: 因子有效性验证", expanded=False):
        st.markdown("计算截面IC/IC_IR，评估各因子的预测能力。")

        # --- 历史记录 ---
        try:
            _fv_history = _cache.load_training_history(step="factor_validation", limit=5)
            if not _fv_history.empty:
                st.markdown("**历史验证记录**")
                _fv_display = _fv_history[['created_at', 'market', 'stock_count', 'method', 'result_summary']].copy()
                _fv_display.columns = ['时间', '市场', '股票数', '方法', '结果摘要']
                st.dataframe(_fv_display, use_container_width=True, hide_index=True)
        except Exception:
            pass

        st.markdown("---")
        use_training = st.checkbox(
            "使用Step 1下载的训练数据", value='training_data' in st.session_state,
            key="wb_fv_use_training"
        )

        if st.button("运行因子验证", type="primary", key="wb_fv_run"):
            with st.spinner("正在计算因子IC（可能需要几分钟）..."):
                try:
                    from src.factors.factor_validator import FactorValidator

                    if use_training and 'training_data' in st.session_state:
                        data_dict = st.session_state['training_data']
                    else:
                        from src.data.market import get_stock_pool
                        pool = get_stock_pool(market_code)
                        data_dict = {}
                        progress = st.progress(0)
                        for i, sym in enumerate(pool):
                            try:
                                d = fetch_stock_data(sym, start_date, market_code)
                                if not d.empty and len(d) >= 60:
                                    data_dict[sym] = d
                            except Exception:
                                pass
                            progress.progress((i + 1) / len(pool))

                    if len(data_dict) < 10:
                        st.warning(f"有效数据不足（仅{len(data_dict)}只）")
                    else:
                        validator = FactorValidator()
                        report = validator.validate(data_dict, min_stocks=min(20, max(5, len(data_dict) // 2)))

                        st.session_state['factor_results'] = report

                        # 生成摘要用于历史记录
                        _fv_summaries = []
                        for fwd in [5, 10, 20]:
                            summary = validator.generate_summary(report, forward_days=fwd)
                            if not summary.empty:
                                st.markdown(f"**预测周期: {fwd}日**")
                                st.dataframe(summary, use_container_width=True, hide_index=True)
                                # 取top3强有效因子名
                                strong = summary[summary['有效性'].isin(['强有效', '中等有效'])].head(3)
                                if not strong.empty:
                                    _fv_summaries.append(f"{fwd}日Top: {', '.join(strong['因子'].tolist())}")

                        if report.correlation_matrix is not None and not report.correlation_matrix.empty:
                            st.markdown("**因子间相关性矩阵**")
                            fig = px.imshow(
                                report.correlation_matrix,
                                text_auto=True, aspect="auto",
                                color_continuous_scale="RdBu_r",
                                zmin=-1, zmax=1,
                            )
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)

                        # 保存历史记录
                        _result_summary = f"因子数:{len(report.factor_results)}"
                        if _fv_summaries:
                            _result_summary += " | " + "; ".join(_fv_summaries)
                        try:
                            _cache.save_training_history(
                                step="factor_validation",
                                market=market_code,
                                method="IC/IC_IR",
                                stock_count=len(data_dict),
                                result_summary=_result_summary,
                            )
                        except Exception:
                            pass

                        st.success("因子验证完成，结果已保存")

                except Exception as e:
                    st.error(f"因子验证失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        if 'factor_results' in st.session_state:
            st.success("已有因子验证结果 (可直接进入Step 3)")

    # ===== Step 3: 权重优化 =====
    with st.expander("⚖️ Step 3: 权重优化", expanded=False):
        st.markdown("基于因子验证结果，优化策略内因子权重。")

        # --- 历史记录 ---
        try:
            _wo_history = _cache.load_training_history(step="weight_optimization", limit=10)
            if not _wo_history.empty:
                st.markdown("**历史优化记录**")
                _wo_display = _wo_history[['created_at', 'market', 'strategy', 'method', 'stock_count', 'result_summary']].copy()
                _wo_display.columns = ['时间', '市场', '策略', '方法', '股票数', '结果摘要']
                st.dataframe(_wo_display, use_container_width=True, hide_index=True)
        except Exception:
            pass

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            wo_strategy = st.selectbox(
                "选择策略", list(STRATEGY_NAMES.keys()),
                format_func=lambda x: STRATEGY_NAMES[x],
                key="wb_wo_strategy"
            )
        with col2:
            wo_method = st.selectbox(
                "优化方法", ["IC_IR加权", "随机搜索最大化夏普"],
                key="wb_wo_method"
            )

        if st.button("优化权重", type="primary", key="wb_wo_run"):
            with st.spinner("正在优化权重..."):
                try:
                    from src.optimization.weight_optimizer import WeightOptimizer, OptimizationResult

                    optimizer = WeightOptimizer()
                    strategy = get_strategy(wo_strategy)

                    factor_names = list(strategy.params.get('optimized_weights', {}).keys())
                    if not factor_names:
                        factor_names = [
                            'rsi_14', 'momentum_5', 'momentum_20', 'ma_cross',
                            'volatility_20', 'adx', 'volume_ratio', 'price_position'
                        ]

                    weights = None
                    _wo_method_name = wo_method

                    if wo_method == "IC_IR加权":
                        if 'factor_results' not in st.session_state:
                            st.warning("请先运行Step 2因子验证")
                        else:
                            factor_report = st.session_state['factor_results']
                            weights = optimizer.optimize_icir(
                                factor_report.factor_results, factor_names,
                                correlation_matrix=factor_report.correlation_matrix,
                            )
                            opt_result = OptimizationResult(
                                strategy_name=wo_strategy, method="ic_ir", weights=weights,
                            )
                            st.session_state['opt_result'] = {wo_strategy: opt_result}

                            st.markdown("**优化后权重**")
                            w_df = pd.DataFrame([
                                {"因子": k, "优化权重": f"{v:.4f}"}
                                for k, v in sorted(weights.items(), key=lambda x: x[1], reverse=True)
                            ])
                            st.dataframe(w_df, use_container_width=True, hide_index=True)

                    else:  # 随机搜索夏普
                        if 'training_data' not in st.session_state:
                            st.warning("请先运行Step 1下载训练数据")
                        else:
                            data_dict = st.session_state['training_data']
                            weights = optimizer.optimize_sharpe(
                                data_dict, strategy, factor_names, n_trials=200
                            )
                            opt_result = OptimizationResult(
                                strategy_name=wo_strategy, method="sharpe", weights=weights,
                            )
                            st.session_state['opt_result'] = {wo_strategy: opt_result}

                            st.markdown("**优化后权重（夏普最大化）**")
                            w_df = pd.DataFrame([
                                {"因子": k, "优化权重": f"{v:.4f}"}
                                for k, v in sorted(weights.items(), key=lambda x: x[1], reverse=True)
                            ])
                            st.dataframe(w_df, use_container_width=True, hide_index=True)

                    # 保存历史记录
                    if weights:
                        _top3 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
                        _wo_summary = "Top权重: " + ", ".join(f"{k}={v:.3f}" for k, v in _top3)
                        _data_count = len(st.session_state.get('training_data', {}))
                        try:
                            _cache.save_training_history(
                                step="weight_optimization",
                                market=market_code,
                                strategy=STRATEGY_NAMES.get(wo_strategy, wo_strategy),
                                method=_wo_method_name,
                                stock_count=_data_count,
                                result_summary=_wo_summary,
                            )
                        except Exception:
                            pass

                    # Walk-Forward验证
                    if 'opt_result' in st.session_state and 'factor_results' in st.session_state and 'training_data' in st.session_state:
                        st.markdown("**Walk-Forward稳定性验证**")
                        try:
                            wf_results = optimizer.walk_forward_validate(
                                st.session_state['training_data'],
                                st.session_state['factor_results'].factor_results,
                                factor_names,
                            )
                            if wf_results:
                                wf_df = pd.DataFrame(wf_results)
                                st.dataframe(wf_df, use_container_width=True, hide_index=True)
                        except Exception as wf_e:
                            st.info(f"Walk-Forward验证跳过: {wf_e}")

                except Exception as e:
                    st.error(f"权重优化失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # 保存配置按钮
        if 'opt_result' in st.session_state:
            st.success("已有优化结果")
            if st.button("💾 保存到配置文件", key="wb_wo_save"):
                try:
                    from src.optimization.weight_optimizer import WeightOptimizer
                    optimizer = WeightOptimizer()
                    optimizer.save_config(st.session_state['opt_result'])
                    st.success("已保存到 config/strategy_weights.json")
                except Exception as e:
                    st.error(f"保存失败: {e}")

    # ===== Step 4: 阈值网格搜索 =====
    with st.expander("🎯 Step 4: 阈值网格搜索", expanded=False):
        st.markdown("遍历(买入阈值, 卖出阈值)组合，找到夏普最优点。")

        # --- 历史记录 ---
        try:
            _gs_history = _cache.load_training_history(step="grid_search", limit=10)
            if not _gs_history.empty:
                st.markdown("**历史搜索记录**")
                _gs_display = _gs_history[['created_at', 'market', 'strategy', 'stock_count', 'result_summary']].copy()
                _gs_display.columns = ['时间', '市场', '策略', '股票数', '最优结果']
                st.dataframe(_gs_display, use_container_width=True, hide_index=True)
        except Exception:
            pass

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            gs_strategy = st.selectbox(
                "选择策略", list(STRATEGY_NAMES.keys()),
                format_func=lambda x: STRATEGY_NAMES[x],
                key="wb_gs_strategy"
            )
        with col2:
            gs_step = st.selectbox("搜索步长", [5.0, 2.5, 10.0], key="wb_gs_step")

        col3, col4 = st.columns(2)
        with col3:
            gs_buy_lo = st.number_input("买入阈值下限", 40, 90, 50, key="wb_gs_buy_lo")
            gs_buy_hi = st.number_input("买入阈值上限", 50, 95, 80, key="wb_gs_buy_hi")
        with col4:
            gs_sell_lo = st.number_input("卖出阈值下限", 10, 50, 20, key="wb_gs_sell_lo")
            gs_sell_hi = st.number_input("卖出阈值上限", 20, 60, 50, key="wb_gs_sell_hi")

        if st.button("开始网格搜索", type="primary", key="wb_gs_run"):
            with st.spinner("网格搜索中（可能需要几分钟）..."):
                try:
                    from src.backtest.rule_backtester import RuleBacktester

                    if 'training_data' not in st.session_state:
                        st.warning("请先运行Step 1下载训练数据")
                    else:
                        data_dict = st.session_state['training_data']
                        strategy = get_strategy(gs_strategy)
                        backtester = RuleBacktester()

                        results = backtester.grid_search_thresholds(
                            data_dict, strategy,
                            buy_range=(gs_buy_lo, gs_buy_hi),
                            sell_range=(gs_sell_lo, gs_sell_hi),
                            step=gs_step,
                        )

                        if results:
                            # 热力图
                            heatmap_df = backtester.generate_heatmap_data(results)
                            if not heatmap_df.empty:
                                st.markdown("**夏普比率热力图 (买入阈值 × 卖出阈值)**")
                                fig = px.imshow(
                                    heatmap_df,
                                    text_auto=".2f", aspect="auto",
                                    color_continuous_scale="RdYlGn",
                                    labels=dict(x="卖出阈值", y="买入阈值", color="夏普"),
                                )
                                fig.update_layout(height=500)
                                st.plotly_chart(fig, use_container_width=True)

                            # 结果排名表
                            st.markdown("**Top 10 参数组合**")
                            sorted_results = sorted(results, key=lambda r: r.sharpe, reverse=True)[:10]
                            res_df = pd.DataFrame([
                                {
                                    "买入阈值": r.buy_threshold,
                                    "卖出阈值": r.sell_threshold,
                                    "夏普": f"{r.sharpe:.3f}",
                                    "年化收益": f"{r.annual_return:.2%}",
                                    "最大回撤": f"{r.max_drawdown:.2%}",
                                    "胜率": f"{r.win_rate:.1%}",
                                    "交易次数": r.trade_count,
                                }
                                for r in sorted_results
                            ])
                            st.dataframe(res_df, use_container_width=True, hide_index=True)

                            # 保存最优
                            best = sorted_results[0]
                            st.session_state['best_thresholds'] = {
                                'strategy': gs_strategy,
                                'buy_threshold': best.buy_threshold,
                                'sell_threshold': best.sell_threshold,
                                'sharpe': best.sharpe,
                            }
                            st.success(
                                f"最优: 买入>{best.buy_threshold} 卖出<{best.sell_threshold} "
                                f"夏普={best.sharpe:.3f}"
                            )

                            # 保存历史记录
                            try:
                                _cache.save_training_history(
                                    step="grid_search",
                                    market=market_code,
                                    strategy=STRATEGY_NAMES.get(gs_strategy, gs_strategy),
                                    method=f"step={gs_step}",
                                    stock_count=len(data_dict),
                                    result_summary=f"买入>{best.buy_threshold} 卖出<{best.sell_threshold} 夏普={best.sharpe:.3f} 胜率={best.win_rate:.1%}",
                                    params={"buy_range": [gs_buy_lo, gs_buy_hi],
                                            "sell_range": [gs_sell_lo, gs_sell_hi],
                                            "step": gs_step},
                                )
                            except Exception:
                                pass

                            # 自适应评分区间
                            st.markdown("---")
                            st.markdown("**自适应评分区间 (P10-P90)**")
                            try:
                                factor_names = [
                                    'rsi_14', 'momentum_5', 'momentum_20', 'ma_cross',
                                    'volatility_20', 'adx', 'volume_ratio', 'price_position'
                                ]
                                ranges = backtester.compute_pooled_score_range(
                                    data_dict, factor_names
                                )
                                if ranges:
                                    range_df = pd.DataFrame([
                                        {"因子": k, "P10(低)": f"{v[0]:.4f}", "P90(高)": f"{v[1]:.4f}"}
                                        for k, v in ranges.items()
                                    ])
                                    st.dataframe(range_df, use_container_width=True, hide_index=True)
                            except Exception as sr_e:
                                st.info(f"评分区间计算跳过: {sr_e}")

                        else:
                            st.warning("网格搜索未返回有效结果")

                except Exception as e:
                    st.error(f"网格搜索失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # 应用最优阈值按钮
        if 'best_thresholds' in st.session_state:
            bt = st.session_state['best_thresholds']
            st.info(f"当前最优阈值: {STRATEGY_NAMES[bt['strategy']]} 买入>{bt['buy_threshold']} 卖出<{bt['sell_threshold']}")
            if st.button("📝 应用最优阈值到配置", key="wb_gs_apply"):
                try:
                    import json
                    from pathlib import Path
                    config_path = Path("config/strategy_weights.json")
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    config = {}
                    if config_path.exists():
                        with open(config_path) as f:
                            config = json.load(f)
                    sk = bt['strategy']
                    if sk not in config:
                        config[sk] = {}
                    config[sk]['buy_threshold'] = bt['buy_threshold']
                    config[sk]['sell_threshold'] = bt['sell_threshold']
                    config[sk]['threshold_source'] = 'grid_search'
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2, ensure_ascii=False)
                    st.success(f"已更新 {config_path}")
                except Exception as e:
                    st.error(f"保存失败: {e}")

    # ===== Step 5: 配置状态与再训练 =====
    with st.expander("📋 Step 5: 配置状态与再训练", expanded=False):
        st.markdown("查看当前配置文件状态、信号日志统计，支持一键再训练。")

        # --- 全流程训练历史汇总 ---
        try:
            _all_history = _cache.load_training_history(limit=30)
            if not _all_history.empty:
                st.markdown("**全流程训练历史**")
                _ah_display = _all_history[['created_at', 'step', 'market', 'strategy', 'method', 'stock_count', 'result_summary']].copy()
                _ah_display.columns = ['时间', '步骤', '市场', '策略', '方法', '股票数', '结果摘要']
                # 步骤名映射
                _step_map = {
                    'factor_validation': '🔬 因子验证',
                    'weight_optimization': '⚖️ 权重优化',
                    'grid_search': '🎯 阈值搜索',
                    'retrain': '🔄 再训练',
                }
                _ah_display['步骤'] = _ah_display['步骤'].map(lambda x: _step_map.get(x, x))
                st.dataframe(_ah_display, use_container_width=True, hide_index=True, height=250)
        except Exception:
            pass

        st.markdown("---")

        # 当前配置文件
        import json
        from pathlib import Path
        config_path = Path("config/strategy_weights.json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            st.markdown("**当前 `config/strategy_weights.json`**")
            st.json(config)

            # 各策略参数来源
            st.markdown("**各策略参数来源**")
            source_rows = []
            for key in STRATEGY_NAMES:
                if key in config:
                    src = config[key].get('threshold_source', config[key].get('method', '优化配置'))
                    source_rows.append({"策略": STRATEGY_NAMES[key], "参数来源": f"优化配置 ({src})"})
                else:
                    source_rows.append({"策略": STRATEGY_NAMES[key], "参数来源": "默认硬编码"})
            st.dataframe(pd.DataFrame(source_rows), use_container_width=True, hide_index=True)
        else:
            st.info("尚无优化配置文件 (config/strategy_weights.json)，使用默认硬编码参数")

        # 信号日志统计
        st.markdown("---")
        st.markdown("**信号日志统计**")
        try:
            from src.data.data_cache import DataCache
            cache = DataCache()
            signals = cache.load_signals(market=market_code, limit=5000)
            if signals.empty:
                st.info("暂无信号记录。在「个股分析」中分析股票后会自动记录信号。")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总信号数", len(signals))
                with col2:
                    recent = signals[signals['date'] >= (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')]
                    st.metric("近30天信号", len(recent))
                with col3:
                    # 胜率：return_5d > 0 的比例
                    if 'return_5d' in signals.columns:
                        filled = signals[signals['return_5d'].notna()]
                        if len(filled) > 0:
                            win_rate = (filled['return_5d'] > 0).mean()
                            st.metric("5日胜率", f"{win_rate:.1%}")
                        else:
                            st.metric("5日胜率", "待回填")
                    else:
                        st.metric("5日胜率", "-")

                # 各策略信号分布
                st.markdown("**各策略信号分布**")
                dist = signals.groupby(['strategy', 'action']).size().unstack(fill_value=0)
                st.dataframe(dist, use_container_width=True)

                # 待回填信号 + 回填按钮
                pending = cache.get_pending_backfill_signals(market=market_code)
                if not pending.empty:
                    col_p1, col_p2 = st.columns([3, 1])
                    with col_p1:
                        st.warning(f"有 {len(pending)} 条信号待收益回填")
                    with col_p2:
                        if st.button("📊 回填收益", key="wb_backfill"):
                            with st.spinner("正在回填信号收益..."):
                                try:
                                    fetcher = get_fetcher_v4()
                                    result = cache.batch_backfill_returns(fetcher, market=market_code)
                                    st.success(
                                        f"回填完成: 已填{result['filled']}条, 跳过{result['skipped']}条"
                                    )
                                    st.rerun()
                                except Exception as bf_e:
                                    st.error(f"回填失败: {bf_e}")

        except Exception as e:
            st.warning(f"信号日志读取失败: {e}")

        # 一键再训练
        st.markdown("---")
        if st.button("🔄 一键再训练 (Step 2→3→4 自动执行)", type="primary", key="wb_retrain"):
            if 'training_data' not in st.session_state:
                st.warning("请先运行Step 1下载训练数据")
            else:
                data_dict = st.session_state['training_data']
                retrain_strategy = st.session_state.get('wb_wo_strategy', 'balanced')
                progress_text = st.empty()
                retrain_progress = st.progress(0)

                try:
                    # === Train/Test隔离：自动留出最近6个月数据作为样本外测试集 ===
                    from datetime import timedelta
                    test_cutoff = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
                    train_dict = {}
                    test_dict = {}
                    for code, df in data_dict.items():
                        if df.empty:
                            continue
                        # 确保索引可比较
                        if isinstance(df.index, pd.DatetimeIndex):
                            train_mask = df.index < pd.Timestamp(test_cutoff)
                        else:
                            train_mask = pd.Series([True] * len(df), index=df.index)
                        train_df = df[train_mask]
                        test_df = df[~train_mask]
                        if len(train_df) >= 60:
                            train_dict[code] = train_df
                        if len(test_df) >= 5:
                            test_dict[code] = test_df

                    if not train_dict:
                        train_dict = data_dict  # 降级：数据不足时用全量

                    progress_text.text(
                        f"数据切分完成: 训练集{len(train_dict)}只, 测试集{len(test_dict)}只 "
                        f"(截止{test_cutoff})"
                    )

                    # Step 2: 因子验证（仅用训练集）
                    progress_text.text("Step 2/4: 因子验证中（训练集）...")
                    retrain_progress.progress(0.1)
                    from src.factors.factor_validator import FactorValidator
                    validator = FactorValidator()
                    factor_report = validator.validate(train_dict, min_stocks=min(20, max(5, len(train_dict) // 2)))
                    st.session_state['factor_results'] = factor_report
                    retrain_progress.progress(0.25)

                    # Step 3: 权重优化（仅用训练集）
                    progress_text.text("Step 3/4: 权重优化中（训练集）...")
                    from src.optimization.weight_optimizer import WeightOptimizer, OptimizationResult
                    optimizer = WeightOptimizer()
                    strategy = get_strategy(retrain_strategy)
                    factor_names = list(strategy.params.get('optimized_weights', {}).keys())
                    if not factor_names:
                        factor_names = [
                            'rsi_14', 'momentum_5', 'momentum_20', 'ma_cross',
                            'volatility_20', 'adx', 'volume_ratio', 'price_position'
                        ]

                    weights = optimizer.optimize_icir(
                        factor_report.factor_results, factor_names,
                        correlation_matrix=factor_report.correlation_matrix,
                    )
                    opt_result = OptimizationResult(
                        strategy_name=retrain_strategy,
                        method="ic_ir",
                        weights=weights,
                    )
                    st.session_state['opt_result'] = {retrain_strategy: opt_result}
                    retrain_progress.progress(0.5)

                    # Step 4: 阈值网格搜索（仅用训练集）
                    progress_text.text("Step 4/4: 阈值网格搜索中（训练集）...")
                    from src.backtest.rule_backtester import RuleBacktester
                    backtester = RuleBacktester()
                    gs_results = backtester.grid_search_thresholds(
                        train_dict, strategy,
                        buy_range=(50, 80), sell_range=(20, 50), step=5.0,
                    )
                    retrain_progress.progress(0.75)

                    # === 样本外验证 ===
                    oos_sharpe = None
                    in_sample_sharpe = None
                    if gs_results:
                        best = max(gs_results, key=lambda r: r.sharpe)
                        in_sample_sharpe = best.sharpe

                        if test_dict:
                            progress_text.text("样本外验证中...")
                            # 用最优阈值在测试集上评估
                            strategy.params['buy_threshold'] = best.buy_threshold
                            strategy.params['sell_threshold'] = best.sell_threshold
                            oos_results = backtester.grid_search_thresholds(
                                test_dict, strategy,
                                buy_range=(best.buy_threshold, best.buy_threshold),
                                sell_range=(best.sell_threshold, best.sell_threshold),
                                step=1.0,
                            )
                            if oos_results:
                                oos_sharpe = oos_results[0].sharpe

                    retrain_progress.progress(0.9)

                    # 保存配置
                    optimizer.save_config({retrain_strategy: opt_result})
                    if gs_results:
                        best = max(gs_results, key=lambda r: r.sharpe)
                        import json
                        cp = Path("config/strategy_weights.json")
                        config = {}
                        if cp.exists():
                            with open(cp) as f:
                                config = json.load(f)
                        if retrain_strategy not in config:
                            config[retrain_strategy] = {}
                        config[retrain_strategy]['buy_threshold'] = best.buy_threshold
                        config[retrain_strategy]['sell_threshold'] = best.sell_threshold
                        config[retrain_strategy]['threshold_source'] = 'grid_search_retrain'
                        with open(cp, 'w') as f:
                            json.dump(config, f, indent=2, ensure_ascii=False)

                    retrain_progress.progress(1.0)
                    progress_text.text("再训练完成!")

                    # 保存再训练历史
                    _retrain_summary = f"策略={STRATEGY_NAMES[retrain_strategy]}, 权重+阈值已更新"
                    if in_sample_sharpe is not None:
                        _retrain_summary += f", 样本内夏普={in_sample_sharpe:.3f}"
                    if oos_sharpe is not None:
                        _retrain_summary += f", 样本外夏普={oos_sharpe:.3f}"
                    try:
                        _cache.save_training_history(
                            step="retrain",
                            market=market_code,
                            strategy=STRATEGY_NAMES.get(retrain_strategy, retrain_strategy),
                            method="auto(IC_IR+grid_search)+OOS验证",
                            stock_count=len(data_dict),
                            result_summary=_retrain_summary,
                        )
                    except Exception:
                        pass

                    # 显示样本内/样本外对比
                    st.success(
                        f"再训练完成! 策略 {STRATEGY_NAMES[retrain_strategy]} 的权重和阈值已更新。"
                    )
                    if in_sample_sharpe is not None or oos_sharpe is not None:
                        oos_col1, oos_col2, oos_col3 = st.columns(3)
                        with oos_col1:
                            st.metric("训练集股票数", len(train_dict))
                        with oos_col2:
                            if in_sample_sharpe is not None:
                                st.metric("样本内夏普", f"{in_sample_sharpe:.3f}")
                        with oos_col3:
                            if oos_sharpe is not None:
                                delta = oos_sharpe - in_sample_sharpe if in_sample_sharpe else 0
                                st.metric("样本外夏普", f"{oos_sharpe:.3f}",
                                          delta=f"{delta:+.3f}",
                                          delta_color="normal")
                            else:
                                st.metric("样本外夏普", "测试集不足")

                        if oos_sharpe is not None and in_sample_sharpe is not None:
                            decay = 1 - oos_sharpe / in_sample_sharpe if in_sample_sharpe != 0 else 0
                            if decay > 0.5:
                                st.warning(f"样本外衰减{decay:.0%}，可能存在过拟合")
                            elif decay > 0.3:
                                st.info(f"样本外衰减{decay:.0%}，策略泛化能力一般")
                            else:
                                st.success(f"样本外衰减{decay:.0%}，策略泛化良好")

                    st.balloons()

                except Exception as e:
                    st.error(f"再训练失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())


# ==================== 主入口 ====================
def main():
    market_code, start_date, end_date = render_sidebar()

    st.session_state['market'] = market_code

    # 主标签页（7个Tab）
    tabs = st.tabs([
        "📊 个股分析", "💼 持仓管理", "🎯 市场扫描",
        "🧪 策略回测",
        "💰 ETF定投", "🎯 目标规划", "⚡ 策略实验室",
    ])

    with tabs[0]:
        render_stock_analysis(market_code, start_date)
    with tabs[1]:
        render_portfolio(market_code, start_date)
    with tabs[2]:
        render_market_scan(market_code, start_date)
    with tabs[3]:
        render_backtest(market_code, start_date)
    with tabs[4]:
        render_etf_dip(market_code, start_date)
    with tabs[5]:
        render_goal_planning(market_code, start_date)
    with tabs[6]:
        render_strategy_lab(market_code, start_date)


if __name__ == "__main__":
    main()
