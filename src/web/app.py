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

# 页面配置
st.set_page_config(
    page_title="A股量化交易系统",
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
def fetch_stock_data(code: str, start_date: str) -> pd.DataFrame:
    """缓存股票数据获取"""
    fetcher = DataFetcher()
    return fetcher.get_daily_data(code, start_date=start_date)


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
    st.markdown('<p class="main-header">📈 A股量化交易系统</p>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 设置")
        
        # 数据设置
        st.subheader("📊 数据设置")
        stock_code = st.text_input("股票代码", value="000001", help="输入6位股票代码")
        
        date_range = st.date_input(
            "日期范围",
            value=(datetime.now() - timedelta(days=365), datetime.now()),
            help="选择数据时间范围"
        )
        
        if st.button("🔄 获取数据", type="primary", use_container_width=True):
            st.session_state['refresh'] = True
    
    # 主内容区
    tabs = st.tabs(["📊 行情分析", "🔬 因子研究", "📈 策略回测", "🤖 AI选股", "🧠 模型训练"])
    
    # Tab 1: 行情分析
    with tabs[0]:
        st.header("行情分析")
        
        try:
            with st.spinner("加载数据中..."):
                start_date = date_range[0].strftime("%Y-%m-%d") if isinstance(date_range, tuple) else "2024-01-01"
                df = fetch_stock_data(stock_code, start_date)
                
                # 计算因子
                factor_engine = FactorEngine()
                df = factor_engine.compute(df, ['ma_20', 'ma_60', 'rsi_14', 'momentum_20'])
            
            # 指标卡片
            col1, col2, col3, col4 = st.columns(4)
            
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            change = (latest['close'] - prev['close']) / prev['close'] * 100
            
            col1.metric("最新价", f"¥{latest['close']:.2f}", f"{change:+.2f}%")
            col2.metric("最高价", f"¥{latest['high']:.2f}")
            col3.metric("最低价", f"¥{latest['low']:.2f}")
            col4.metric("成交量", f"{latest['volume']/10000:.0f}万")
            
            # K线图
            st.plotly_chart(create_candlestick_chart(df, f"{stock_code} K线图"), use_container_width=True)
            
            # 技术指标
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("RSI指标")
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df.index, y=df['rsi_14'], name='RSI(14)', line=dict(color='#9C27B0')))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖")
                fig_rsi.update_layout(height=300, template='plotly_dark')
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            with col2:
                st.subheader("动量指标")
                fig_mom = go.Figure()
                fig_mom.add_trace(go.Bar(x=df.index, y=df['momentum_20'] * 100, name='20日动量'))
                fig_mom.update_layout(height=300, template='plotly_dark')
                st.plotly_chart(fig_mom, use_container_width=True)
                
        except Exception as e:
            st.error(f"数据获取失败: {e}")
            st.info("请检查股票代码是否正确，或稍后重试")
    
    # Tab 2: 因子研究
    with tabs[1]:
        st.header("因子研究")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("因子配置")
            selected_factors = st.multiselect(
                "选择因子",
                ['momentum_5', 'momentum_10', 'momentum_20', 'volatility_10', 'volatility_20', 'rsi_14', 'ma_20'],
                default=['momentum_20', 'rsi_14']
            )
            
            st.subheader("多因子模型")
            model_type = st.selectbox(
                "模型类型",
                ['均衡模型', '动量模型', '价值模型', '质量模型']
            )
        
        with col2:
            if st.button("📊 计算因子", use_container_width=True):
                with st.spinner("计算中..."):
                    try:
                        df = fetch_stock_data(stock_code, "2024-01-01")
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
        st.header("策略回测")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("回测配置")
            
            strategy_type = st.selectbox(
                "策略类型",
                ['均线交叉', '动量策略']
            )
            
            initial_capital = st.number_input("初始资金", value=1000000, step=100000)
            
            if strategy_type == '均线交叉':
                short_period = st.slider("短期均线", 5, 20, 5)
                long_period = st.slider("长期均线", 10, 60, 20)
            else:
                lookback = st.slider("动量周期", 5, 60, 20)
                top_n = st.slider("持仓数量", 1, 10, 3)
            
            run_backtest = st.button("🚀 运行回测", type="primary", use_container_width=True)
        
        with col2:
            if run_backtest:
                with st.spinner("回测中..."):
                    try:
                        # 准备数据
                        codes = ['000001', '000002', '600000', '600036', '601398']
                        data = {}
                        for code in codes:
                            data[code] = fetch_stock_data(code, "2024-01-01")
                        
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
                        col_a.metric("总收益率", metrics['总收益率'])
                        col_b.metric("年化收益率", metrics['年化收益率'])
                        col_c.metric("最大回撤", metrics['最大回撤'])
                        col_d.metric("夏普比率", metrics['夏普比率'])
                        
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
                                    '价格': f"¥{t.price:.2f}",
                                    '数量': t.shares,
                                    '金额': f"¥{t.amount:.0f}"
                                }
                                for t in result.trades[-20:]
                            ])
                            st.dataframe(trades_df, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"回测失败: {e}")
    
    
    # Tab 4: AI选股
    with tabs[3]:
        # ... (AI选股内容保持不变) ...
        st.header("AI智能选股")
        
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
                        st.info("正在获取实时股票列表...")
                        stock_list_df = fetch_stock_list()
                        # 默认使用前20只股票作为演示池，避免全市场遍历耗时过长
                        default_pool_size = 20
                        codes = stock_list_df['code'].head(default_pool_size).tolist()
                        
                        st.write(f"正在分析 {len(codes)} 只股票 (来自实时市场列表)...")
                        
                        factor_engine = FactorEngine()
                        data = {}
                        
                        progress = st.progress(0)
                        for i, code in enumerate(codes):
                            try:
                                df = fetch_stock_data(code, "2024-01-01")
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
        st.header("🧠 模型训练")
        
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
        "A股量化交易系统 v0.1.0 | Powered by AKShare + Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
