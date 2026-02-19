"""
Web界面AI分析面板工具函数 - Phase 10.3

提供在Streamlit中展示AI分析结果的组件函数。
"""

import pandas as pd
import streamlit as st
from typing import Dict, Optional, List


def render_grok_sentiment_panel(grok_data: Optional[Dict]):
    """渲染Grok AI情绪分析面板

    Args:
        grok_data: GrokClient返回的数据
            {'sentiment': {...}, 'market': {...}, 'analysis': {...}}
    """
    if not grok_data:
        st.info("💡 Grok AI分析未启用或无数据")
        return

    sentiment = grok_data.get('sentiment')
    market = grok_data.get('market')

    # 个股情绪
    if sentiment:
        st.subheader("🤖 个股社交情绪")

        col1, col2, col3 = st.columns(3)
        with col1:
            score = sentiment.get('sentiment_score', 0)
            st.metric("情绪得分", f"{score:.2f}",
                     delta="看多" if score > 0 else "看空")
        with col2:
            conf = sentiment.get('confidence', 0)
            st.metric("置信度", f"{conf:.0%}")
        with col3:
            posts = sentiment.get('post_count', 0)
            st.metric("讨论量", posts)

        # 关键话题
        topics = sentiment.get('key_topics', [])
        if topics:
            st.write("**热门话题:**", ", ".join(topics[:5]))

        # 看多/看空信号
        col_bull, col_bear = st.columns(2)
        with col_bull:
            st.write("**看多信号:**")
            for sig in sentiment.get('bullish_signals', [])[:3]:
                st.write(f"- {sig}")
        with col_bear:
            st.write("**看空信号:**")
            for sig in sentiment.get('bearish_signals', [])[:3]:
                st.write(f"- {sig}")

        # 事件风险
        risk = sentiment.get('event_risk', 'none')
        if risk != 'none':
            st.warning(f"⚠️ 事件风险级别: {risk}")

    # 市场情绪
    if market:
        st.subheader("🌍 市场整体情绪")

        col1, col2 = st.columns(2)
        with col1:
            mood = market.get('market_mood', 'neutral')
            mood_cn = {
                'euphoria': '🚀 狂热', 'optimistic': '😊 乐观',
                'neutral': '😐 中性', 'anxious': '😰 焦虑',
                'panic': '😱 恐慌'
            }.get(mood, mood)
            st.metric("市场情绪", mood_cn)
        with col2:
            fg = market.get('fear_greed_estimate', 50)
            st.metric("恐慌贪婪指数", fg,
                     delta="贪婪" if fg > 50 else "恐慌")

        # 关键事件
        events = market.get('key_events', [])
        if events:
            st.write("**24h重大事件:**")
            for evt in events[:3]:
                st.write(f"- {evt}")

        # 板块轮动
        rotation = market.get('sector_rotation', {})
        if rotation:
            col_hot, col_cold = st.columns(2)
            with col_hot:
                st.write("**🔥 热门板块:**")
                for sec in rotation.get('hot_sectors', [])[:3]:
                    st.write(f"- {sec}")
            with col_cold:
                st.write("**❄️ 冷门板块:**")
                for sec in rotation.get('cold_sectors', [])[:3]:
                    st.write(f"- {sec}")


def render_research_consensus_panel(research_data: Optional[Dict], code: str):
    """渲染行研共识面板

    Args:
        research_data: DataFetcher.get_research_data()返回值
        code: 股票代码
    """
    if not research_data:
        st.info("💡 行研报告数据暂无")
        return

    st.subheader("📊 华尔街/机构共识")

    recommendations = research_data.get('recommendations')
    price_targets = research_data.get('price_targets') or research_data.get('analyst_price_targets')

    # 评级分布
    if recommendations is not None and not recommendations.empty:
        try:
            from src.factors.research_factors import RATING_MAP

            # 找到评级列 (可能是 'To Grade' 或 'to_grade')
            grade_col = None
            for col in ['To Grade', 'to_grade', 'toGrade']:
                if col in recommendations.columns:
                    grade_col = col
                    break

            if grade_col:
                # 统计评级分布
                grades = recommendations[grade_col].map(RATING_MAP).dropna()
                if len(grades) > 0:
                    rating_dist = grades.value_counts().sort_index(ascending=False)

                    # 映射到中文
                    rating_names = {5: '强力买入', 4: '买入', 3: '持有', 2: '减持', 1: '卖出'}
                    rating_dist.index = [rating_names.get(i, str(i)) for i in rating_dist.index]

                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.bar_chart(rating_dist)
                    with col2:
                        avg_rating = grades.mean()
                        st.metric("平均评级", f"{avg_rating:.2f}/5")

                        # 统计覆盖机构 (可能是 'Firm' 或 'firm')
                        firm_col = 'Firm' if 'Firm' in recommendations.columns else 'firm'
                        if firm_col in recommendations.columns:
                            st.metric("覆盖机构", len(recommendations[firm_col].unique()))
                else:
                    st.caption("评级数据格式不支持")
            else:
                st.caption("未找到评级列")
        except Exception as e:
            st.caption(f"评级数据解析失败: {e}")

    # 目标价
    if price_targets:
        st.write("**一致目标价:**")
        try:
            if isinstance(price_targets, dict):
                for key, val in price_targets.items():
                    if val is not None:
                        st.write(f"- {key}: ${val:.2f}" if isinstance(val, (int, float)) else f"- {key}: {val}")
            elif isinstance(price_targets, pd.DataFrame) and not price_targets.empty:
                # 提取目标价范围
                if 'targetMean' in price_targets.columns:
                    mean_target = price_targets['targetMean'].iloc[0]
                    if mean_target and pd.notna(mean_target):
                        st.metric("平均目标价", f"${mean_target:.2f}")
        except Exception as e:
            st.caption(f"目标价数据解析失败: {e}")


def render_market_regime_panel(regime_info: Dict):
    """渲染市场状态识别面板

    Args:
        regime_info: {
            'regime': MarketRegime枚举,
            'confidence': float,
            'description': str
        }
    """
    st.subheader("🎯 市场状态识别 (HMM)")

    regime = regime_info.get('regime')
    conf = regime_info.get('confidence', 0)
    desc = regime_info.get('description', '')

    if regime:
        regime_cn = {
            'bull': '🐂 牛市', 'bear': '🐻 熊市',
            'sideways': '↔️ 震荡', 'crisis': '⚠️ 危机'
        }.get(regime.value if hasattr(regime, 'value') else str(regime), str(regime))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("当前状态", regime_cn)
        with col2:
            st.metric("识别置信度", f"{conf:.0%}")

        st.caption(desc)


def render_industry_rotation_panel(industry_scores: Dict[str, float],
                                   top_n: int = 10):
    """渲染行业轮动面板

    Args:
        industry_scores: {行业名称: 0-100得分}
        top_n: 显示前N名
    """
    if not industry_scores:
        st.info("💡 行业轮动数据暂无")
        return

    st.subheader("🔄 行业轮动 (动量排名)")

    # 排序
    sorted_industries = sorted(industry_scores.items(),
                              key=lambda x: x[1], reverse=True)

    # 分为强势和弱势
    top_industries = sorted_industries[:top_n]
    bottom_industries = sorted_industries[-top_n:][::-1]

    col_strong, col_weak = st.columns(2)

    with col_strong:
        st.write("**💪 强势行业 TOP", top_n, "**")
        for industry, score in top_industries:
            st.progress(score / 100, text=f"{industry}: {score:.0f}")

    with col_weak:
        st.write("**📉 弱势行业 BOTTOM", top_n, "**")
        for industry, score in bottom_industries:
            st.progress(score / 100, text=f"{industry}: {score:.0f}")


def render_dl_filter_status(dl_filter_info: Dict):
    """渲染DL信号过滤器状态

    Args:
        dl_filter_info: DLSignalFilter.get_model_info()返回值
    """
    if not dl_filter_info.get('enabled'):
        st.caption("🤖 DL信号过滤: 未启用")
        return

    with st.expander("🤖 深度学习过滤器"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**模型类型:** {dl_filter_info.get('model_type', 'N/A').upper()}")
            st.write(f"**特征数量:** {dl_filter_info.get('feature_count', 0)}")
        with col2:
            threshold = dl_filter_info.get('confidence_threshold', 0.6)
            st.write(f"**置信阈值:** {threshold:.0%}")
            st.write("**状态:** ✅ 运行中")


def render_risk_panel(risk_alerts: List[str],
                     atr_info: Optional[Dict] = None,
                     correlation_warnings: Optional[List[str]] = None,
                     black_swan: Optional[Dict] = None):
    """渲染增强风控面板

    Args:
        risk_alerts: 基础风险告警列表
        atr_info: ATR止损信息
        correlation_warnings: 相关性预警
        black_swan: 黑天鹅检测结果
    """
    st.subheader("🛡️ 智能风控")

    # 黑天鹅预警（最高优先级）
    if black_swan and black_swan.get('triggered'):
        severity = black_swan.get('severity', 'warning')
        msg = black_swan.get('message', '')

        if severity == 'critical':
            st.error(f"🚨 {msg}")
        else:
            st.warning(f"⚠️ {msg}")

    # 基础风险告警
    if risk_alerts:
        with st.expander("⚠️ 风险告警", expanded=True):
            for alert in risk_alerts:
                st.write(f"- {alert}")

    # ATR自适应止损
    if atr_info:
        with st.expander("📏 ATR自适应止损"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("ATR值", f"{atr_info.get('atr', 0):.4f}")
            with col2:
                st.metric("止损价", f"¥{atr_info.get('stop_price', 0):.2f}")
            with col3:
                st.metric("止损幅度", f"{atr_info.get('stop_pct', 0):.2%}")

    # 相关性预警
    if correlation_warnings:
        with st.expander("🔗 持仓相关性监控"):
            for warn in correlation_warnings:
                st.write(f"- {warn}")


# ==================== 使用示例 ====================

def example_usage():
    """使用示例（在app.py中调用）

    # 在个股分析页面添加AI面板Tab
    with st.tabs(["策略信号", "行情走势", "AI分析"]):
        with tabs[2]:
            # Grok分析
            grok_data = grok_client.analyze_stock_sentiment(code)
            render_grok_sentiment_panel({'sentiment': grok_data})

            # 行研共识
            research_data = fetcher.get_research_data(code, market)
            render_research_consensus_panel(research_data, code)

            # 市场状态
            regime_info = router.get_current_regime(index_df)
            render_market_regime_panel(regime_info)

            # 行业轮动
            industry_factor = IndustryRotationFactor()
            scores = industry_factor.compute_industry_scores(market)
            render_industry_rotation_panel(scores)
    """
    pass
