"""
股票量化策略决策支持系统 - 市场情绪因子模块

将 DataFetcher.get_sentiment_data() 返回的原始情绪数据转化为 0-100 因子得分。

数据来源:
  - A股: 融资融券余额(margin_balance), 北向资金(northbound_flow)
  - 美股: VIX恐慌指数(vix), 10年期国债收益率(us_yield)

设计原则:
  - 所有得分归一化到 0-100，与 InterpretableStrategy 因子体系兼容
  - 数据不足时返回 None，策略层自动跳过
  - 情绪因子作为辅助因子，建议权重 5-8%
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from loguru import logger


class MarketSentimentFactor:
    """市场情绪因子计算器

    将融资融券、北向资金、VIX等情绪数据转化为因子得分。
    """

    def compute_all_scores(self, sentiment_data: Dict,
                           market: str = "CN") -> Dict[str, float]:
        """一次性计算所有情绪因子得分

        Args:
            sentiment_data: DataFetcher.get_sentiment_data() 的返回值
            market: "CN" 或 "US"

        Returns:
            {因子名: 0-100得分}，数据不足的因子不包含
        """
        if not sentiment_data:
            return {}

        scores = {}

        if market == "CN":
            # A股情绪因子
            score = self.margin_sentiment(sentiment_data.get('margin_balance'))
            if score is not None:
                scores['融资情绪'] = score

            score = self.northbound_momentum(sentiment_data.get('northbound_flow'))
            if score is not None:
                scores['北向资金动量'] = score
        else:
            # 美股情绪因子
            score = self.vix_regime(sentiment_data.get('vix'))
            if score is not None:
                scores['VIX情绪'] = score

            score = self.yield_signal(sentiment_data.get('us_yield'))
            if score is not None:
                scores['利率信号'] = score

        return scores

    # ==================== A股情绪因子 ====================

    def margin_sentiment(self, margin_data) -> Optional[float]:
        """融资融券情绪得分

        逻辑:
        - 融资买入额的20日动量 > 0 表示杠杆资金流入（看多情绪）
        - 融资买入额持续下降表示杠杆资金撤离（看空情绪）

        Returns:
            0-100, 高分=乐观情绪, 50=中性
        """
        if margin_data is None:
            return None

        try:
            df = margin_data if isinstance(margin_data, pd.DataFrame) else None
            if df is None or df.empty:
                return None

            # 取融资买入额列
            col = '融资买入额' if '融资买入额' in df.columns else df.columns[0]
            series = df[col].dropna().astype(float)

            if len(series) < 20:
                return None

            # 计算20日动量（最近vs20日前的变化率）
            recent = series.iloc[-5:].mean()
            past = series.iloc[-25:-20].mean() if len(series) >= 25 else series.iloc[:5].mean()

            if past == 0:
                return 50.0

            momentum = (recent - past) / abs(past)

            # 映射到0-100: momentum在[-0.3, 0.3]区间
            score = 50 + momentum / 0.3 * 50
            return max(0, min(100, score))

        except Exception as e:
            logger.debug(f"融资情绪因子计算失败: {e}")
            return None

    def northbound_momentum(self, northbound_data) -> Optional[float]:
        """北向资金动量得分

        逻辑:
        - 近5日净买入累计 > 0 表示外资看多
        - 近20日净买入趋势上升 = 强信号

        Returns:
            0-100, 高分=外资看多
        """
        if northbound_data is None:
            return None

        try:
            df = northbound_data if isinstance(northbound_data, pd.DataFrame) else None
            if df is None or df.empty:
                return None

            col = 'north_money' if 'north_money' in df.columns else df.columns[0]
            series = df[col].dropna().astype(float)

            if len(series) < 10:
                return None

            # 近5日累计净买入
            recent_5d = series.iloc[-5:].sum()
            # 近20日累计净买入
            recent_20d = series.iloc[-20:].sum() if len(series) >= 20 else series.sum()

            # 归一化：北向日均净买入一般在-200亿~200亿之间
            # 5日累计在-1000亿~1000亿
            norm_5d = recent_5d / 500  # 归一到 [-2, 2] 大致范围
            norm_20d = recent_20d / 2000

            # 短期权重60% + 中期权重40%
            combined = norm_5d * 0.6 + norm_20d * 0.4
            score = 50 + combined * 25  # 映射到 [0, 100]
            return max(0, min(100, score))

        except Exception as e:
            logger.debug(f"北向资金因子计算失败: {e}")
            return None

    # ==================== 美股情绪因子 ====================

    def vix_regime(self, vix_data) -> Optional[float]:
        """VIX恐慌指数情绪得分

        逻辑（逆向思维）:
        - VIX > 30 = 极度恐惧 → 高分（恐惧中贪婪）
        - VIX 20-30 = 较高恐惧 → 偏高
        - VIX 12-20 = 正常 → 中性
        - VIX < 12 = 过度乐观 → 低分（贪婪中恐惧）

        注意: 这里使用逆向情绪因子，与Warren Buffett的"别人恐惧时贪婪"一致。
        如果用户偏好顺势情绪（高VIX=看空），需要在策略层反转。

        Returns:
            0-100, 高分=市场恐惧（逆向看多机会）
        """
        if vix_data is None:
            return None

        try:
            df = vix_data if isinstance(vix_data, pd.DataFrame) else None
            if df is None or df.empty:
                return None

            # 取Close列
            if 'Close' in df.columns:
                vix_close = df['Close']
            elif 'close' in df.columns:
                vix_close = df['close']
            else:
                vix_close = df.iloc[:, -1]  # 最后一列

            vix_close = vix_close.dropna().astype(float)
            if len(vix_close) == 0:
                return None

            current_vix = float(vix_close.iloc[-1])

            # 分段映射（逆向）
            if current_vix > 35:
                score = 90  # 极度恐惧 = 逆向看多
            elif current_vix > 25:
                score = 75
            elif current_vix > 20:
                score = 60
            elif current_vix > 15:
                score = 50  # 正常
            elif current_vix > 12:
                score = 35
            else:
                score = 20  # 过度贪婪 = 逆向看空

            # VIX变化趋势微调: VIX快速上升 +5, 快速下降 -5
            if len(vix_close) >= 5:
                vix_change_5d = (current_vix - float(vix_close.iloc[-5])) / float(vix_close.iloc[-5])
                if vix_change_5d > 0.2:
                    score = min(100, score + 5)  # VIX急升
                elif vix_change_5d < -0.2:
                    score = max(0, score - 5)  # VIX急降

            return float(score)

        except Exception as e:
            logger.debug(f"VIX情绪因子计算失败: {e}")
            return None

    def yield_signal(self, yield_data) -> Optional[float]:
        """10年期国债收益率信号

        逻辑:
        - 收益率上升 → 资金从股市流向债市（偏空）→ 低分
        - 收益率下降 → 资金从债市流向股市（偏多）→ 高分
        - 基于20日变化趋势

        Returns:
            0-100, 高分=利率下降利好股市
        """
        if yield_data is None:
            return None

        try:
            df = yield_data if isinstance(yield_data, pd.DataFrame) else None
            if df is None or df.empty:
                return None

            if 'Close' in df.columns:
                yields = df['Close']
            elif 'close' in df.columns:
                yields = df['close']
            else:
                yields = df.iloc[:, -1]

            yields = yields.dropna().astype(float)
            if len(yields) < 20:
                return None

            current = float(yields.iloc[-1])
            past_20d = float(yields.iloc[-20])

            if past_20d == 0:
                return 50.0

            # 收益率变化（负变化=利好股市）
            change = (current - past_20d) / abs(past_20d)

            # 映射: change在[-0.1, 0.1]区间
            # change < 0 (利率下降) → 高分
            score = 50 - change / 0.1 * 30
            return max(0, min(100, score))

        except Exception as e:
            logger.debug(f"利率信号因子计算失败: {e}")
            return None


class GrokSentimentFactor:
    """Grok AI 情绪因子计算器

    将 GrokClient 返回的结构化情绪数据转化为 0-100 因子得分，
    可直接接入 InterpretableStrategy 的因子打分体系。

    设计原则:
      - sentiment_score(-1~1) → 0-100 线性映射
      - 低置信度(confidence<0.3)或低样本量(post_count<5)时向50衰减
      - event_risk 高时额外输出风险预警（不改变得分，由策略层处理）
    """

    def compute_all_scores(self, grok_sentiment: Optional[Dict] = None,
                           grok_market: Optional[Dict] = None) -> Dict[str, Any]:
        """计算所有 Grok 相关因子得分

        Args:
            grok_sentiment: GrokClient.analyze_stock_sentiment() 的返回值
            grok_market: GrokClient.analyze_market_regime() 的返回值

        Returns:
            {
                'AI舆情': float,           # 0-100 个股舆情得分
                'AI市场情绪': float,        # 0-100 市场整体情绪
                'AI事件风险': str,          # "none"/"low"/"medium"/"high"
                'AI热门板块': List[str],    # 热门板块列表
            }
            数据不足的因子不包含
        """
        scores = {}

        if grok_sentiment:
            score = self.stock_sentiment_score(grok_sentiment)
            if score is not None:
                scores['AI舆情'] = score

            event_risk = grok_sentiment.get('event_risk', 'none')
            if event_risk != 'none':
                scores['AI事件风险'] = event_risk

        if grok_market:
            score = self.market_mood_score(grok_market)
            if score is not None:
                scores['AI市场情绪'] = score

            hot = grok_market.get('sector_rotation', {}).get('hot_sectors', [])
            if hot:
                scores['AI热门板块'] = hot

        return scores

    def stock_sentiment_score(self, grok_result: Dict) -> Optional[float]:
        """个股舆情得分

        将 Grok 返回的 sentiment_score(-1~1) 转化为 0-100 因子得分。
        低置信度或低讨论量时向50（中性）衰减。

        Returns:
            0-100, 高分=社交媒体看多
        """
        if not grok_result:
            return None

        try:
            raw_score = float(grok_result.get('sentiment_score', 0))
            confidence = float(grok_result.get('confidence', 0))
            post_count = int(grok_result.get('post_count', 0))

            # 基础映射: -1~1 → 0~100
            base_score = (raw_score + 1) / 2 * 100

            # 置信度衰减: confidence < 0.3 时大幅向50衰减
            if confidence < 0.3:
                decay = 0.3  # 低置信度只保留30%偏离
            elif confidence < 0.5:
                decay = 0.6
            elif confidence < 0.7:
                decay = 0.8
            else:
                decay = 1.0

            # 样本量衰减: post_count < 5 时进一步衰减
            if post_count < 3:
                sample_decay = 0.2
            elif post_count < 5:
                sample_decay = 0.5
            elif post_count < 10:
                sample_decay = 0.8
            else:
                sample_decay = 1.0

            # 综合衰减: score = 50 + (base - 50) * decay * sample_decay
            combined_decay = decay * sample_decay
            score = 50 + (base_score - 50) * combined_decay

            return max(0, min(100, score))

        except (TypeError, ValueError) as e:
            logger.debug(f"Grok个股情绪因子计算失败: {e}")
            return None

    def market_mood_score(self, grok_market: Dict) -> Optional[float]:
        """市场整体情绪得分

        将 Grok 的 market_mood 和 fear_greed_estimate 转化为因子得分。
        使用逆向思维（与VIX因子一致）: 恐慌=逆向看多机会。

        Returns:
            0-100, 使用逆向逻辑: 高分=市场恐慌（买入机会）
        """
        if not grok_market:
            return None

        try:
            mood = grok_market.get('market_mood', 'neutral')
            fear_greed = int(grok_market.get('fear_greed_estimate', 50))

            # 逆向映射: fear_greed 0(极度恐惧)→90, 100(极度贪婪)→10
            contrarian_score = 100 - fear_greed

            # mood 微调
            mood_adjustment = {
                'panic': 10,
                'anxious': 5,
                'neutral': 0,
                'optimistic': -5,
                'euphoria': -10,
            }
            adjustment = mood_adjustment.get(mood, 0)
            score = contrarian_score + adjustment

            return max(0, min(100, float(score)))

        except (TypeError, ValueError) as e:
            logger.debug(f"Grok市场情绪因子计算失败: {e}")
            return None

    def extract_risk_warnings(self, grok_sentiment: Optional[Dict] = None,
                              grok_market: Optional[Dict] = None) -> list:
        """从 Grok 结果中提取风险预警

        供策略层在 DecisionReport.risk_warnings 中使用。

        Returns:
            风险预警文本列表
        """
        warnings = []

        if grok_sentiment:
            event_risk = grok_sentiment.get('event_risk', 'none')
            if event_risk == 'high':
                warnings.append("AI检测到重大事件风险（财报/监管/诉讼等），建议谨慎")
            elif event_risk == 'medium':
                warnings.append("AI检测到中等事件风险，注意仓位控制")

            bearish = grok_sentiment.get('bearish_signals', [])
            if len(bearish) >= 3:
                warnings.append(f"社交媒体存在多个看空信号: {'; '.join(bearish[:2])}")

        if grok_market:
            risk_alerts = grok_market.get('risk_alerts', [])
            for alert in risk_alerts[:2]:  # 最多取2条
                warnings.append(f"市场风险: {alert}")

        return warnings
