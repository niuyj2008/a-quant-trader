"""
A股量化交易系统 - Web可视化界面

基于Streamlit的交互式分析平台
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

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import DataFetcher
from src.factors import FactorEngine
from src.backtest import BacktestEngine, BacktestConfig
from src.strategy import MACrossStrategy, MomentumStrategy
from src.models import AlphaFactorModel, StockPredictor, LIGHTGBM_AVAILABLE, XGBOOST_AVAILABLE
from src.train_pipeline import TrainingPipeline

# ============================================================
# 量化术语通俗解释（面向零基础用户）
# ============================================================
QUANT_GLOSSARY = {
    "K线": "K线图是股票价格的一种图形表示法。每根K线代表一段时间（如1天），展示开盘价、收盘价、最高价、最低价四个价格。红色代表涨，绿色代表跌。",
    "均线": "均线是过去一段时间收盘价的平均值连成的线。例如'20日均线'就是过去20天的平均价格。均线向上说明趋势向好，向下说明趋势走弱。",
    "RSI": "RSI（相对强弱指数）衡量股票是否被'过度买入'或'过度卖出'。数值0-100，超过70可能涨太多了（超买），低于30可能跌太多了（超卖）。",
    "动量": "动量表示股价变化的速度和力量。动量为正说明股价在上涨，为负说明在下跌。数值越大，涨跌力度越强。",
    "成交量": "成交量是一段时间内交易的股票数量。成交量大说明买卖活跃，通常价格变动也会更剧烈。",
    "因子": "因子是用来预测股票表现的'体检指标'。比如'动量因子'看股票跑得快不快，'波动率因子'看股票震荡厉不厉害。好的因子能帮助我们选出优质股票。",
    "回测": "回测是用历史数据模拟交易，看看策略在过去表现如何。就像'穿越时空'验证你的方法是否靠谱，避免用真金白银试错。",
    "收益率": "收益率表示投资赚了百分之多少。例如10%收益率意味着投入100元赚了10元。年化收益率是把短期收益换算成一年能赚多少的标准化指标。",
    "最大回撤": "最大回撤是账户从最高点下跌到最低点的幅度。比如账户从100万跌到70万，回撤就是30%。回撤越小，说明风险控制越好，亏损时心理压力越小。",
    "夏普比率": "夏普比率衡量'每承担一份风险能赚多少'。数值越高，策略性价比越好。一般大于1就算不错，大于2是优秀。",
    "AI选股": "让人工智能分析海量数据，找出最可能上涨的股票。AI比人更客观，不会因为情绪乱下单，但也不是100%准确。",
    "模型训练": "把历史数据'喂'给AI，让它学习规律。就像教练训练运动员，练习越多，预测越准。训练好的模型可以用来预测未来。",
    "波动率": "波动率衡量股价的震荡程度。波动率高的股票涨跌都很剧烈，风险较大但机会也多；波动率低的股票走势平稳，相对安全。",
    "均线交叉": "当短期均线从下往上穿过长期均线时叫'金叉'，通常是买入信号；反之叫'死叉'，通常是卖出信号。这是最经典的技术分析方法之一。",
}

# 页面配置
st.set_page_config(
    page_title="全球量化交易系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def fetch_stock_data(code: str, start_date: str, market: str = "CN") -> pd.DataFrame:
    """缓存股票数据获取"""
    fetcher = DataFetcher()
    return fetcher.get_daily_data(code, start_date=start_date, market=market)


@st.cache_data(ttl=300)
def fetch_stock_list() -> pd.DataFrame:
    """缓存股票列表"""
    fetcher = DataFetcher()
    return fetcher.get_stock_list()


def create_candlestick_chart(df: pd.DataFrame, title: str = "") -> go.Figure:
    """创建K线图"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )
    
    # K线
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线',
            increasing_line_color='#ef5350',
            decreasing_line_color='#26a69a'
        ),
        row=1, col=1
    )
    
    # 均线
    if 'ma_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['ma_20'], name='MA20', line=dict(color='#FFA726', width=1)),
            row=1, col=1
        )
    if 'ma_60' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['ma_60'], name='MA60', line=dict(color='#42A5F5', width=1)),
            row=1, col=1
        )
    
    # 成交量
    colors = ['#ef5350' if df['close'].iloc[i] >= df['open'].iloc[i] else '#26a69a' 
              for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], name='成交量', marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=600,
        template='plotly_dark',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_equity_curve(result) -> go.Figure:
    """创建收益曲线图"""
    if result.equity_curve is None or len(result.equity_curve) == 0:
        return go.Figure()
    
    returns = result.equity_curve / result.equity_curve.iloc[0] - 1
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=['累计收益', '回撤']
    )
    
    # 收益曲线
    fig.add_trace(
        go.Scatter(
            x=returns.index, y=returns * 100,
            fill='tozeroy',
            name='策略收益',
            line=dict(color='#667eea')
        ),
        row=1, col=1
    )
    
    # 回撤
    rolling_max = result.equity_curve.cummax()
    drawdown = (result.equity_curve - rolling_max) / rolling_max * 100
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, y=drawdown,
            fill='tozeroy',
            name='回撤',
            line=dict(color='#ef5350')
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        template='plotly_dark',
        showlegend=True
    )
    fig.update_yaxes(title_text="收益率 (%)", row=1, col=1)
    fig.update_yaxes(title_text="回撤 (%)", row=2, col=1)
    
    return fig


def main():
    # 标题
    st.markdown('<p class="main-header">📈 全球量化交易系统</p>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 设置")
        
        # 数据设置
        st.subheader("📊 数据设置")
        
        market = st.selectbox(
            "选择市场",
            ["CN (A股)", "US (美股)"],
            index=0,
            help="💡 A股是中国大陆的股票市场，美股是美国股票市场。新手建议先从A股开始熟悉。"
        )
        market_code = "CN" if "CN" in market else "US"
        
        default_code = "000001" if market_code == "CN" else "AAPL"
        if market_code == "CN":
            help_text = "💡 股票代码是股票的'身份证号'。A股代码是6位数字，例如：000001（平安银行）、600519（贵州茅台）、300750（宁德时代）"
        else:
            help_text = "💡 美股代码是公司名称的缩写，例如：AAPL（苹果）、NVDA（英伟达）、TSLA（特斯拉）、MSFT（微软）"
        
        stock_code = st.text_input("股票代码", value=default_code, help=help_text)
        
        date_range = st.date_input(
            "日期范围",
            value=(datetime.now() - timedelta(days=365), datetime.now()),
            help="💡 选择要分析的时间段。建议至少选择3个月以上的数据，时间越长，分析越准确。1年是比较理想的分析周期。"
        )
        
        if st.button("🔄 获取数据", type="primary", use_container_width=True):
            st.session_state['refresh'] = True
    
    # 主内容区
    tabs = st.tabs(["📊 行情分析", "🔬 因子研究", "📈 策略回测", "🤖 AI选股", "🧠 模型训练"])
    
    # 词汇表/概念解释浮窗配置 (通用工具函数)
    def concept_help(title: str, content: str):
        with st.popover(f"❓什么是{title}？"):
            st.write(content)

    # Tab 1: 行情分析
    with tabs[0]:
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.header("行情分析")
        with col2:
            concept_help("行情分析", "通过观察股票的价格走势（K线图）和成交量，来判断市场的热度和买卖力量。这是最基础的看盘方法。")
        
        # 新手入门指南
        with st.expander("📖 新手入门：如何看懂行情分析？", expanded=False):
            st.markdown("""
            **行情分析** 是投资的第一步，让你了解一只股票的价格变化情况。
            
            **图表说明：**
            - 📊 **K线图**：每根柱子代表一天的价格变化。**红色代表涨**（收盘价高于开盘价），**绿色代表跌**。柱子越长说明当天涨跌幅度越大。
            - 📈 **均线**：图中的曲线是均线，橙色是20日均线（短期趋势），蓝色是60日均线（长期趋势）。均线向上说明整体趋势向好。
            - 📉 **成交量**：底部的柱状图，柱子越高说明当天交易越活跃。
            
            **操作步骤：**
            1. 在左侧边栏输入股票代码（例如：000001）
            2. 选择要分析的日期范围
            3. 点击"🔄 获取数据"按钮
            4. 观察图表，红色K线多说明最近涨势较好
            
            **小贴士：** 不要只看一两天的涨跌，要结合较长时间的趋势来判断。
            """)
        
        try:
            with st.spinner("加载数据中..."):
                start_date = date_range[0].strftime("%Y-%m-%d") if isinstance(date_range, tuple) else "2024-01-01"
                df = fetch_stock_data(stock_code, start_date, market=market_code)
                
                # 计算因子
                factor_engine = FactorEngine()
                df = factor_engine.compute(df, ['ma_20', 'ma_60', 'rsi_14', 'momentum_20'])
            
            # 指标卡片
            col1, col2, col3, col4 = st.columns(4)
            
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            change = (latest['close'] - prev['close']) / prev['close'] * 100
            
            currency_symbol = "¥" if market_code == "CN" else "$"
            
            col1.metric("最新价", f"{currency_symbol}{latest['close']:.2f}", f"{change:+.2f}%")
            col2.metric("最高价", f"{currency_symbol}{latest['high']:.2f}")
            col3.metric("最低价", f"{currency_symbol}{latest['low']:.2f}")
            col4.metric("成交量", f"{latest['volume']/10000:.0f}万" if market_code == "CN" else f"{latest['volume']:,}")
            
            # K线图
            st.plotly_chart(create_candlestick_chart(df, f"{stock_code} K线图"), use_container_width=True)
            
            # 技术指标
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("RSI指标")
                st.caption("💡 RSI衡量股票是否'超买'或'超卖'，帮助判断买卖时机")
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df.index, y=df['rsi_14'], name='RSI(14)', line=dict(color='#9C27B0')))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买区")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖区")
                fig_rsi.update_layout(height=300, template='plotly_dark')
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                # RSI 解读
                latest_rsi = df['rsi_14'].iloc[-1]
                if latest_rsi > 70:
                    st.warning(f"⚠️ 当前RSI={latest_rsi:.1f}，处于超买区间。股票可能涨太多了，短期有回调风险，不建议追高。")
                elif latest_rsi < 30:
                    st.success(f"✅ 当前RSI={latest_rsi:.1f}，处于超卖区间。股票可能跌太多了，可关注反弹机会。")
                else:
                    st.info(f"ℹ️ 当前RSI={latest_rsi:.1f}，在正常区间（30-70），无明显超买超卖信号。")
            
            with col2:
                st.subheader("动量指标")
                st.caption("💡 动量反映股价变化的速度和方向，柱子越高涨势越强")
                fig_mom = go.Figure()
                fig_mom.add_trace(go.Bar(x=df.index, y=df['momentum_20'] * 100, name='20日动量'))
                fig_mom.update_layout(height=300, template='plotly_dark')
                st.plotly_chart(fig_mom, use_container_width=True)
                
                # 动量解读
                latest_momentum = df['momentum_20'].iloc[-1] * 100
                if latest_momentum > 10:
                    st.success(f"📈 当前动量={latest_momentum:.1f}%，股价上涨力度较强，趋势向好。")
                elif latest_momentum < -10:
                    st.warning(f"📉 当前动量={latest_momentum:.1f}%，股价下跌力度较强，注意风险。")
                else:
                    st.info(f"➡️ 当前动量={latest_momentum:.1f}%，股价走势平稳，无明显趋势。")
                
        except Exception as e:
            st.error(f"数据获取失败: {e}")
            st.info("请检查股票代码是否正确，或稍后重试")
    
    # Tab 2: 因子研究
    with tabs[1]:
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.header("因子研究")
        with col2:
            concept_help("因子", "因子就像是股票的『体检指标』。比如『动量』代表股票最近跑得快不快，『波动率』代表股票跳动得厉不厉害。研究因子就是找哪些指标能预示股票未来赚钱。")
        
        # 新手入门指南
        with st.expander("📖 新手入门：什么是因子研究？", expanded=False):
            st.markdown("""
            **因子研究** 就是寻找"选股密码"。我们想找到一些指标，能帮助预测哪些股票未来会涨。
            
            **常见因子解释：**
            - **动量因子 (momentum)**: 看股票最近涨得快不快。数字越大表示最近涨幅越大。
            - **波动率因子 (volatility)**: 看股票震荡大不大。数字越大说明价格起伏越剧烈。
            - **RSI**: 判断股票是否涨太多或跌太多。
            - **均线 (ma)**: 过去一段时间的平均价格。
            
            **操作步骤：**
            1. 在左侧选择你感兴趣的因子
            2. 点击"📊 计算因子"
            3. 查看因子数值和相关性热力图
            
            **小贴士：** 好因子之间相关性应该较低（热力图颜色浅），这样组合效果更好。
            """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("因子配置")
            # 因子选择帮助映射
            factor_names = {
                'momentum_5': '5日动量（短期涨跌幅）',
                'momentum_10': '10日动量',
                'momentum_20': '20日动量（中期涨跌幅）',
                'volatility_10': '10日波动率（短期震荡）',
                'volatility_20': '20日波动率',
                'rsi_14': 'RSI相对强弱指数',
                'ma_20': '20日均线'
            }
            selected_factors = st.multiselect(
                "选择因子",
                list(factor_names.keys()),
                default=['momentum_20', 'rsi_14'],
                format_func=lambda x: factor_names.get(x, x),
                help="💡 可以同时选择多个因子进行对比分析。建议选择2-4个进行组合。"
            )
            
            st.subheader("多因子模型")
            model_type = st.selectbox(
                "模型类型",
                ['均衡模型', '动量模型', '价值模型', '质量模型'],
                help="💡 不同模型侧重不同风格：动量追涨杀跌，价值寻找低估，均衡则兼顾多方面。"
            )
        
        with col2:
            if st.button("📊 计算因子", use_container_width=True):
                with st.spinner("计算中..."):
                    try:
                        df = fetch_stock_data(stock_code, "2024-01-01", market=market_code)
                        factor_engine = FactorEngine()
                        df_factors = factor_engine.compute(df, selected_factors)
                        
                        st.subheader("因子值")
                        st.dataframe(df_factors[['close'] + selected_factors].tail(20), use_container_width=True)
                        
                        # 因子相关性
                        if len(selected_factors) > 1:
                            st.subheader("因子相关性")
                            corr = df_factors[selected_factors].corr()
                            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
                            fig_corr.update_layout(height=400)
                            st.plotly_chart(fig_corr, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"计算失败: {e}")
    
    # Tab 3: 策略回测
    with tabs[2]:
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.header("策略回测")
        with col2:
            concept_help("回测", "回测就是『穿越时空』。假设你在几年前用某套方法买卖，看看现在能赚多少钱。它可以帮你验证你的方法是否真的靠谱，而不是靠运气。")
        
        # 新手入门指南
        with st.expander("📖 新手入门：什么是策略回测？", expanded=False):
            st.markdown("""
            **策略回测** 就像"开上帝视角玩游戏"——我们假装回到过去，用某套买卖规则操作，看看能赚多少钱。
            
            **为什么要回测？**
            - 验证策略是否真的有效，而非靠运气
            - 了解策略在不同市场环境下的表现
            - 发现策略的风险点（比如最大亏损多少）
            
            **策略说明：**
            - **均线交叉**：当短期均线上穿长期均线时买入（金叉），下穿时卖出（死叉）
            - **动量策略**：买入最近涨得最好的股票，卖出涨势减弱的股票
            
            **关键指标解读：**
            - **总收益率**：整个回测期间赚了多少
            - **年化收益率**：换算成每年赚多少，便于与银行存款对比
            - **最大回撤**：账户从最高点跌到最低点的幅度，越小越好
            - **夏普比率**：收益与风险的性价比，一般>1算不错
            """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("回测配置")
            
            strategy_type = st.selectbox(
                "策略类型",
                ['均线交叉', '动量策略'],
                help="💡 均线交叉适合趋势明显的市场；动量策略适合强者恒强的行情。"
            )
            
            initial_capital = st.number_input(
                "初始资金", 
                value=1000000, 
                step=100000,
                help="💡 设定模拟投资的初始本金。建议设置与你实际计划投入的金额相近。"
            )
            
            if strategy_type == '均线交叉':
                short_period = st.slider(
                    "短期均线", 5, 20, 5,
                    help="💡 短期均线越短，对价格变化越敏感，但可能产生更多假信号。"
                )
                st.caption(f"计算最近 {short_period} 天的平均价格")
                long_period = st.slider(
                    "长期均线", 10, 60, 20,
                    help="💡 长期均线代表大趋势，通常设为短期的3-4倍。"
                )
                st.caption(f"计算最近 {long_period} 天的平均价格")
            else:
                lookback = st.slider(
                    "动量周期", 5, 60, 20,
                    help="💡 看过去多少天的涨幅来判断动量强弱。"
                )
                st.caption(f"根据过去 {lookback} 天的涨幅排名选股")
                top_n = st.slider(
                    "持仓数量", 1, 10, 3,
                    help="💡 同时持有几只股票。数量越多，风险越分散。"
                )
                st.caption(f"同时持有动量最强的 {top_n} 只股票")
            
            run_backtest = st.button("🚀 运行回测", type="primary", use_container_width=True)
        
        with col2:
            if run_backtest:
                with st.spinner("回测中..."):
                    try:
                        # 准备数据
                        if market_code == "CN":
                            codes = ['000001', '000002', '600000', '600036', '601398']
                        else:
                            codes = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
                            
                        data = {}
                        for code in codes:
                            data[code] = fetch_stock_data(code, "2024-01-01", market=market_code)
                        
                        # 配置
                        config = BacktestConfig(initial_capital=initial_capital)
                        engine = BacktestEngine(config)
                        
                        # 选择策略
                        if strategy_type == '均线交叉':
                            strategy = MACrossStrategy(short_period, long_period)
                        else:
                            strategy = MomentumStrategy(lookback=lookback, top_n=top_n)
                        
                        # 运行回测
                        result = engine.run(data, strategy)
                        
                        # 显示结果
                        st.subheader("📊 回测结果")
                        
                        metrics = result.summary()
                        col_a, col_b, col_c, col_d = st.columns(4)
                        
                        with col_a:
                            st.metric("总收益率", metrics['总收益率'])
                            st.caption("回测期间总共赚了百分之几")
                        with col_b:
                            st.metric("年化收益率", metrics['年化收益率'])
                            st.caption("平均每年赚多少")
                        with col_c:
                            st.metric("最大回撤", metrics['最大回撤'])
                            with st.popover("？什么是回撤"):
                                st.write("回撤代表你的账户从最高点掉下来多少。最大回撤越小，说明你的心理压力越小，风险控制越好。")
                        with col_d:
                            st.metric("夏普比率", metrics['夏普比率'])
                            with st.popover("？什么是夏普"):
                                st.write("夏普比率代表『每承担一份风险能赚多少超额收益』。这个数值越高，说明你的策略性价比越高。")
                        
                        # 收益曲线
                        st.plotly_chart(create_equity_curve(result), use_container_width=True)
                        
                        # 交易记录
                        if result.trades:
                            st.subheader("📝 交易记录")
                            trades_df = pd.DataFrame([
                                {
                                    '日期': t.date.strftime('%Y-%m-%d'),
                                    '股票': t.code,
                                    '方向': '买入' if t.direction == 'buy' else '卖出',
                                    '价格': f"{currency_symbol}{t.price:.2f}",
                                    '数量': t.shares,
                                    '金额': f"{currency_symbol}{t.amount:.0f}"
                                }
                                for t in result.trades[-20:]
                            ])
                            st.dataframe(trades_df, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"回测失败: {e}")
    
    
    # Tab 4: AI选股
    with tabs[3]:
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.header("AI智能选股")
        with col2:
            concept_help("AI选股", "这就是『人工智能当军师』。让电脑分析成千上万条数据，找出它认为明天最可能涨的股票。它比人类更客观，不会因为心情不好乱下单。")
        
        # 新手入门指南
        with st.expander("📖 新手入门：AI选股是什么？怎么用？", expanded=False):
            st.markdown("""
            **AI选股** 就是让人工智能帮你"海选"股票。它会分析大量数据，找出最值得关注的股票。
            
            **选股模型说明：**
            - **多因子模型**：综合多个指标（如动量、波动率等）打分排名
            - **机器学习模型**：用历史数据训练AI，让它自己学会选股规律
            
            **如何理解结果？**
            - **综合得分**：分数越高，AI越看好这只股票
            - 得分 > 0 表示可能跑赢大盘，< 0 表示可能落后
            - 排名前几的股票是AI认为最有潜力的
            
            **重要提醒：**
            - AI选股仅供**参考**，不代表一定会涨
            - 建议结合自己的判断和风险承受能力做决定
            - 任何投资都有风险，请谨慎操作
            """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("模型配置")
            
            ai_model = st.selectbox(
                "选股模型",
                ['多因子模型', '机器学习模型']
            )
            
            if ai_model == '多因子模型':
                factor_model = st.selectbox(
                    "因子组合",
                    ['均衡模型', '动量模型', '价值模型']
                )
            else:
                ml_options = ['RandomForest']
                if LIGHTGBM_AVAILABLE:
                    ml_options.append('LightGBM')
                if XGBOOST_AVAILABLE:
                    ml_options.append('XGBoost')
                    
                ml_model = st.selectbox("ML模型", ml_options)
                
                if not LIGHTGBM_AVAILABLE or not XGBOOST_AVAILABLE:
                    missing = []
                    if not LIGHTGBM_AVAILABLE: missing.append("LightGBM")
                    if not XGBOOST_AVAILABLE: missing.append("XGBoost")
                    st.warning(f"注意: {', '.join(missing)} 未安装 (正在后台安装依赖)，当前仅显示可用模型")
            
            top_k = st.slider("选股数量", 3, 20, 10)
            
            run_selection = st.button("🤖 开始选股", type="primary", use_container_width=True)
        
        with col2:
            if run_selection:
                with st.spinner("AI选股中..."):
                    try:
                        # 准备数据
                        # 默认使用前20只股票作为演示池，避免全市场遍历耗时过长
                        if market_code == "CN":
                             st.info("正在获取实时股票列表...")
                             stock_list_df = fetch_stock_list()
                             default_pool_size = 20
                             codes = stock_list_df['code'].head(default_pool_size).tolist()
                        else:
                             # 美股暂不支持全市场扫描，使用精选列表
                             st.info("使用美股精选列表...")
                             codes = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD', 'INTC', 'NFLX']
                        
                        st.write(f"正在分析 {len(codes)} 只股票 (来自实时市场列表)...")
                        
                        factor_engine = FactorEngine()
                        data = {}
                        
                        progress = st.progress(0)
                        for i, code in enumerate(codes):
                            try:
                                df = fetch_stock_data(code, "2024-01-01", market=market_code)
                                df = factor_engine.compute(df)
                                data[code] = df
                            except:
                                pass
                            progress.progress((i + 1) / len(codes))
                        
                        # 运行选股
                        if ai_model == '多因子模型':
                            model_map = {
                                '均衡模型': AlphaFactorModel.balanced_model,
                                '动量模型': AlphaFactorModel.momentum_model,
                                '价值模型': AlphaFactorModel.value_model,
                            }
                            model = model_map[factor_model]()
                            selected = model.select_stocks(data, top_n=top_k)
                        else:
                            model_map = {
                                'RandomForest': 'random_forest',
                                'LightGBM': 'lightgbm',
                                'XGBoost': 'xgboost'
                            }
                            predictor = StockPredictor(model_type=model_map[ml_model])
                            feature_cols = ['momentum_5', 'momentum_20', 'rsi_14', 'volatility_20']
                            predictor.train(data, feature_cols)
                            selected = predictor.select_stocks(data, top_n=top_k)
                        
                        # 显示结果
                        st.subheader("🎯 推荐股票")
                        
                        result_df = pd.DataFrame(selected, columns=['股票代码', '综合得分'])
                        result_df['排名'] = range(1, len(result_df) + 1)
                        result_df = result_df[['排名', '股票代码', '综合得分']]
                        result_df['综合得分'] = result_df['综合得分'].apply(lambda x: f"{x:.4f}")
                        
                        st.dataframe(result_df, use_container_width=True, hide_index=True)
                        
                        # 可视化
                        fig = px.bar(
                            x=[s[0] for s in selected],
                            y=[s[1] for s in selected],
                            labels={'x': '股票代码', 'y': '得分'},
                            title='选股得分排名'
                        )
                        fig.update_layout(template='plotly_dark')
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"选股失败: {e}")

    # Tab 5: 模型训练
    with tabs[4]:
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.header("🧠 模型训练")
        with col2:
            concept_help("模型训练", "就像在『教练带运动员』。我们把历史数据和正确答案给AI看，让它学会总结规律。训练得越好，它在面对未来新行情时的预测就越准。")
        
        # 新手入门指南
        with st.expander("📖 新手入门：什么是模型训练？", expanded=False):
            st.markdown("""
            **模型训练** 就是"教AI学会选股"。就像老师教学生，我们用历史数据告诉AI过去什么情况下股票会涨，让它学会总结规律。
            
            **模型类型说明：**
            - **LSTM**：长短期记忆网络，擅长学习时间序列规律（如股价走势）
            - **Transformer**：注意力机制模型，能捕捉复杂的市场关联
            - **RandomForest/LightGBM**：传统机器学习模型，训练速度快，适合入门
            
            **参数说明：**
            - **训练轮数 (Epochs)**：让AI学习多少遍，次数越多学得越深（但也可能"死记硬背"）
            - **序列长度 (Lookback)**：AI每次看多少天的历史数据来预测
            - **学习率**：AI每次学习的"步子大小"，太大容易跳过正确答案，太小学得慢
            
            **训练完成后：**
            - 模型会自动保存，下次可以直接使用
            - 可以在"AI选股"中使用训练好的模型
            
            **小贴士：** 新手建议从 RandomForest 开始，训练快且效果稳定。
            """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("配置训练参数")
            
            train_options = ['LSTM', 'Transformer', 'RandomForest']
            if LIGHTGBM_AVAILABLE:
                train_options.append('LightGBM')
            if XGBOOST_AVAILABLE:
                train_options.append('XGBoost')
            
            train_model_type = st.selectbox("模型类型", train_options)
            
            if not LIGHTGBM_AVAILABLE or not XGBOOST_AVAILABLE:
                st.caption("安装完成后请点击下方按钮刷新")
                if st.button("🔄 刷新依赖状态"):
                    st.rerun()

            epochs = st.number_input("训练轮数 (Epochs)", min_value=1, max_value=1000, value=10)
            seq_len = st.number_input("序列长度 (Lookback)", min_value=1, max_value=60, value=10)
            lr = st.number_input("学习率", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
            
            st.subheader("特征选择")
            feature_options = ['ma_5', 'ma_20', 'rsi_14', 'momentum_5', 'momentum_20', 'volatility_20']
            selected_features = st.multiselect("训练特征", feature_options, default=['ma_5', 'ma_20', 'rsi_14'])
            
            start_train = st.button("🚀 开始训练", type="primary", use_container_width=True)
            
        with col2:
            if start_train:
                with st.spinner(f"正在训练 {train_model_type} 模型..."):
                    try:
                        # 1. 准备数据
                        st.info("正在获取训练数据...")
                        codes = ['000001', '000002', '600000', '600036', '601398', '601988']
                        fetcher = DataFetcher()
                        engine = FactorEngine()
                        data_dict = {}
                        
                        progress_bar = st.progress(0)
                        for i, code in enumerate(codes):
                            try:
                                df = fetcher.get_daily_data(code, start_date='2023-01-01')
                                if not df.empty:
                                    df = engine.compute(df, selected_features)
                                    data_dict[code] = df
                            except:
                                pass
                            progress_bar.progress((i + 1) / len(codes))
                        
                        if not data_dict:
                            st.error("没有可用训练数据")
                            st.stop()
                            
                        # 2. 训练管道
                        st.info(f"开始训练流程 (Samples: {sum(len(df) for df in data_dict.values())})...")
                        pipeline = TrainingPipeline(data_dir="data/models")
                        
                        if train_model_type in ['LSTM', 'Transformer']:
                            metrics, path = pipeline.train_dl_model(
                                data=data_dict, 
                                feature_cols=selected_features,
                                model_type=train_model_type.lower(),
                                epochs=epochs,
                                seq_len=seq_len
                            )
                        else:
                            metrics, path = pipeline.train_ml_model(
                                data=data_dict,
                                feature_cols=selected_features,
                                model_type=train_model_type.lower()
                            )
                        
                        st.success("✅ 训练完成!")
                        st.json({
                            "模型路径": path,
                            "状态": metrics,
                            "参数": {
                                "Epochs": epochs,
                                "Seq Len": seq_len,
                                "Features": len(selected_features)
                            }
                        })
                        
                    except Exception as e:
                        st.error(f"训练失败: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    
    # 页脚
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888;'>"
        "<div style='text-align: center; color: #888;'>"
        "全球量化交易系统 v0.2.0 | Powered by AKShare + yfinance + Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
