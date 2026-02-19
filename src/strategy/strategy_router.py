"""
策略自动路由器

根据股票当前的技术状态（趋势强度、波动率、估值水平等）
自动推荐最匹配的策略，替代用户手动选择。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

from src.factors.factor_engine import FactorEngine
from src.factors.macro_factors import MarketRegimeHMM, MacroRegimeDetector, MarketRegime


@dataclass
class RoutingResult:
    """路由推荐结果"""
    code: str
    primary_strategy: str       # 首选策略key
    primary_reason: str         # 推荐理由
    secondary_strategy: str = ""  # 次选策略key
    secondary_reason: str = ""
    excluded_strategies: List[str] = field(default_factory=list)  # 不适用策略
    market_regime: str = ""     # 市场状态描述
    confidence: float = 0.0     # 路由置信度(0-100)
    factor_snapshot: Dict[str, float] = field(default_factory=dict)  # 关键因子快照


class StrategyRouter:
    """策略自动路由器

    基于已有因子判断当前市场状态，将股票路由到最匹配的策略：
    - ADX>25 + 动量一致 → 趋势市 → momentum
    - ADX<20 + 低波动 → 震荡市 → reversion
    - 价格接近阻力位 + 放量 → 突破前夕 → breakout
    - 波动率极低 + 趋势稳定 → 防御 → low_vol
    - 有财务数据 + PE低位 → value
    - 难以判断 → balanced（兜底）

    支持反馈机制：从历史信号日志中学习各策略在不同市场状态下的实际表现，
    用历史胜率调整未来推荐权重。
    """

    def __init__(self):
        self.factor_engine = FactorEngine()
        self._strategy_performance = {}  # {strategy: {win_rate, avg_return, n_signals}}
        self._regime_detector = MarketRegimeHMM()
        self._macro_detector = MacroRegimeDetector()

    def load_feedback(self, market: str = "US"):
        """从signal_log加载各策略的历史表现，用于调整推荐权重

        Args:
            market: 市场代码
        """
        try:
            from src.data.data_cache import DataCache
            cache = DataCache()
            signals = cache.load_signals(market=market, limit=10000)
            if signals.empty:
                return

            # 只看有回填收益的信号
            filled = signals[signals['return_5d'].notna()].copy()
            if filled.empty:
                return

            for strategy, group in filled.groupby('strategy'):
                buy_signals = group[group['action'].isin(['buy', 'add'])]
                if len(buy_signals) < 3:
                    continue

                win_rate = (buy_signals['return_5d'] > 0).mean()
                avg_return = buy_signals['return_5d'].mean()

                self._strategy_performance[strategy] = {
                    'win_rate': float(win_rate),
                    'avg_return': float(avg_return),
                    'n_signals': len(buy_signals),
                }

            if self._strategy_performance:
                logger.info(f"路由器反馈已加载: {len(self._strategy_performance)} 个策略的历史表现")
        except Exception as e:
            logger.debug(f"加载路由器反馈失败: {e}")

    def get_performance_summary(self) -> Dict[str, Dict]:
        """获取各策略历史表现摘要"""
        return self._strategy_performance

    def recommend(self, code: str, df: pd.DataFrame,
                  financial_data: Optional[Dict] = None,
                  name: str = "") -> RoutingResult:
        """为单只股票推荐最佳策略

        Args:
            code: 股票代码
            df: 日线DataFrame（含OHLCV）
            financial_data: 基本面数据（PE/PB/ROE等）
            name: 股票名称

        Returns:
            RoutingResult
        """
        result = RoutingResult(code=code, primary_strategy="balanced",
                               primary_reason="默认兜底策略")

        if df.empty or len(df) < 60:
            result.primary_reason = "数据不足，使用均衡策略"
            return result

        try:
            factored = self.factor_engine.compute_all_core_factors(df)
            latest = factored.iloc[-1]
        except Exception as e:
            logger.debug(f"因子计算失败 {code}: {e}")
            return result

        # 提取关键指标
        adx = latest.get('adx', 20)
        rsi = latest.get('rsi_14', 50)
        m5 = latest.get('momentum_5', 0)
        m20 = latest.get('momentum_20', 0)
        ma_cross = latest.get('ma_cross', 0)
        vol20 = latest.get('volatility_20', 0.03)
        vol_ratio = latest.get('volume_ratio', 1.0)
        price_pos = latest.get('price_position', 0.5)

        result.factor_snapshot = {
            'adx': float(adx) if pd.notna(adx) else 20,
            'rsi': float(rsi) if pd.notna(rsi) else 50,
            'm5': float(m5) if pd.notna(m5) else 0,
            'm20': float(m20) if pd.notna(m20) else 0,
            'volatility': float(vol20) if pd.notna(vol20) else 0.03,
            'volume_ratio': float(vol_ratio) if pd.notna(vol_ratio) else 1.0,
            'price_position': float(price_pos) if pd.notna(price_pos) else 0.5,
        }

        # 安全获取数值
        adx = result.factor_snapshot['adx']
        rsi = result.factor_snapshot['rsi']
        m5 = result.factor_snapshot['m5']
        m20 = result.factor_snapshot['m20']
        vol20 = result.factor_snapshot['volatility']
        vol_ratio = result.factor_snapshot['volume_ratio']
        price_pos = result.factor_snapshot['price_position']
        ma_cross_val = float(ma_cross) if pd.notna(ma_cross) else 0

        # ===== 路由规则 =====

        scores = {}  # {策略key: (得分, 理由)}

        # 1. 趋势市判断 → momentum
        trend_aligned = (m5 > 0 and m20 > 0 and ma_cross_val > 0)
        if adx > 25 and trend_aligned:
            scores['momentum'] = (85, f"强趋势市(ADX={adx:.0f}，动量一致向上)")
        elif adx > 20 and m20 > 0.03:
            scores['momentum'] = (60, f"中等趋势(ADX={adx:.0f}，20日动量{m20:.1%})")

        # 2. 震荡市/超卖 → reversion
        if adx < 20 and rsi < 35:
            scores['reversion'] = (85, f"震荡超卖(ADX={adx:.0f}，RSI={rsi:.0f})")
        elif rsi < 30:
            scores['reversion'] = (75, f"严重超卖(RSI={rsi:.0f})")
        elif adx < 18 and 0.2 < price_pos < 0.5:
            scores['reversion'] = (55, f"低位震荡(ADX={adx:.0f}，价格位置{price_pos:.0%})")

        # 3. 突破前夕 → breakout
        if price_pos > 0.85 and vol_ratio > 1.5:
            scores['breakout'] = (80, f"接近阻力位且放量(位置{price_pos:.0%}，量比{vol_ratio:.1f})")
        elif price_pos > 0.8 and vol_ratio > 1.2 and m5 > 0.02:
            scores['breakout'] = (65, f"突破蓄势(位置{price_pos:.0%}，短期动量{m5:.1%})")

        # 4. 低波动 → low_vol
        if vol20 < 0.015 and adx < 25:
            scores['low_vol'] = (80, f"极低波动({vol20:.2%})，适合防御配置")
        elif vol20 < 0.02 and 0 < m20 < 0.05:
            scores['low_vol'] = (65, f"低波动稳健({vol20:.2%})，温和上涨")

        # 5. 价值 → value（需基本面数据）
        if financial_data:
            pe = financial_data.get('pe')
            pb = financial_data.get('pb')
            roe = financial_data.get('roe')
            if pe and 0 < pe < 15 and roe and roe > 0.10:
                scores['value'] = (85, f"低估值高质量(PE={pe:.1f}，ROE={roe:.1%})")
            elif pe and 0 < pe < 20 and pb and 0 < pb < 2:
                scores['value'] = (65, f"估值偏低(PE={pe:.1f}，PB={pb:.1f})")

        # 6. balanced 作为兜底
        scores['balanced'] = (50, "综合多因子均衡分析")

        # ===== 反馈调整：用历史胜率修正策略得分 =====
        if self._strategy_performance:
            for key in list(scores.keys()):
                if key in self._strategy_performance:
                    perf = self._strategy_performance[key]
                    win_rate = perf['win_rate']
                    n = perf['n_signals']
                    # 只在有足够样本(>=10)时调整，调整幅度 ±15分
                    if n >= 10:
                        # win_rate=0.5 → 不调整, >0.5 → 加分, <0.5 → 减分
                        adjustment = (win_rate - 0.5) * 30  # 最大±15分
                        old_score, reason = scores[key]
                        new_score = max(10, min(100, old_score + adjustment))
                        scores[key] = (new_score, f"{reason} [历史胜率{win_rate:.0%}({n}次)]")

        # ===== 市场状态识别（HMM优先，规则引擎降级）=====
        regime, regime_conf, regime_desc = self._regime_detector.detect_regime(
            df, adx=adx, vol20=vol20, m20=m20, rsi=rsi
        )

        # 市场状态修正: 根据 regime 调整各策略得分
        for key in list(scores.keys()):
            modifier = self._regime_detector.get_strategy_modifier(regime, key)
            if modifier != 1.0:
                old_score, reason = scores[key]
                new_score = max(10, min(100, old_score * modifier))
                if modifier > 1.0:
                    scores[key] = (new_score, f"{reason} [市场状态利好×{modifier:.2f}]")
                else:
                    scores[key] = (new_score, f"{reason} [市场状态不利×{modifier:.2f}]")

        # ===== 排名 =====
        ranked = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)

        result.primary_strategy = ranked[0][0]
        result.primary_reason = ranked[0][1][1]
        result.confidence = ranked[0][1][0]

        if len(ranked) > 1:
            result.secondary_strategy = ranked[1][0]
            result.secondary_reason = ranked[1][1][1]

        # 识别不适用策略
        recommended = {result.primary_strategy, result.secondary_strategy}
        excluded = []

        if adx > 30 and trend_aligned:
            excluded.append('reversion')
        if vol20 > 0.05:
            excluded.append('low_vol')
        if rsi > 75:
            excluded.append('momentum')
        if not financial_data:
            excluded.append('value')

        result.excluded_strategies = [s for s in excluded if s not in recommended]
        result.market_regime = regime_desc

        return result

    def recommend_batch(self, data_dict: Dict[str, pd.DataFrame],
                        financial_dict: Optional[Dict[str, Dict]] = None
                        ) -> Dict[str, RoutingResult]:
        """批量推荐

        Args:
            data_dict: {代码: DataFrame}
            financial_dict: {代码: 基本面Dict}

        Returns:
            {代码: RoutingResult}
        """
        results = {}
        for code, df in data_dict.items():
            fin = financial_dict.get(code) if financial_dict else None
            results[code] = self.recommend(code, df, financial_data=fin)
        return results

    def generate_summary(self, routing_results: Dict[str, RoutingResult]) -> pd.DataFrame:
        """生成路由推荐汇总表"""
        rows = []
        for code, r in routing_results.items():
            rows.append({
                '股票代码': code,
                '首选策略': r.primary_strategy,
                '推荐理由': r.primary_reason,
                '置信度': f"{r.confidence:.0f}",
                '次选策略': r.secondary_strategy,
                '市场状态': r.market_regime,
                '不推荐策略': ', '.join(r.excluded_strategies) if r.excluded_strategies else '-',
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('置信度', ascending=False).reset_index(drop=True)
        return df

    def strategy_distribution(self, routing_results: Dict[str, RoutingResult]) -> Dict[str, int]:
        """统计各策略被推荐的次数"""
        counts = {}
        for r in routing_results.values():
            counts[r.primary_strategy] = counts.get(r.primary_strategy, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def get_current_regime(self, df: pd.DataFrame) -> Tuple[MarketRegime, float, str]:
        """获取当前市场状态（供外部调用）

        Returns:
            (MarketRegime, confidence, description)
        """
        return self._regime_detector.detect_regime(df)
