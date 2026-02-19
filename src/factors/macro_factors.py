"""
股票量化策略决策支持系统 - 宏观因子与市场状态识别模块

包含:
  1. MacroRegimeDetector: 基于宏观经济指标判断经济周期
  2. MarketRegimeHMM: 基于隐马尔可夫模型的市场状态识别

替代 StrategyRouter._describe_regime() 现有的简单阈值逻辑，
输出更精确的市场状态（Bull/Bear/Sideways/Crisis）。

设计原则:
  - hmmlearn 未安装时自动降级到规则引擎
  - 所有状态输出标准化为 MarketRegime 枚举
  - 提供策略增益/惩罚系数，供 StrategyRouter 使用
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, Optional, Tuple
from loguru import logger


class MarketRegime(Enum):
    """市场状态枚举"""
    BULL = "bull"           # 牛市
    BEAR = "bear"           # 熊市
    SIDEWAYS = "sideways"   # 震荡
    CRISIS = "crisis"       # 危机/恐慌

    @property
    def cn_name(self) -> str:
        names = {
            "bull": "牛市（趋势向上）",
            "bear": "熊市（趋势向下）",
            "sideways": "震荡（无明确方向）",
            "crisis": "危机（极端波动）",
        }
        return names.get(self.value, self.value)


class MacroCycle(Enum):
    """宏观经济周期"""
    EXPANSION = "expansion"     # 扩张期
    PEAK = "peak"               # 顶部
    CONTRACTION = "contraction" # 收缩期
    TROUGH = "trough"           # 底部

    @property
    def cn_name(self) -> str:
        names = {
            "expansion": "扩张期",
            "peak": "高点",
            "contraction": "收缩期",
            "trough": "低谷",
        }
        return names.get(self.value, self.value)


# ==================== 策略增益系数 ====================

# 不同市场状态下各策略的增益/惩罚系数
# > 1.0 表示该策略在此状态下有优势
# < 1.0 表示该策略在此状态下需要降低信号强度
REGIME_STRATEGY_MODIFIERS = {
    MarketRegime.BULL: {
        "balanced": 1.0,
        "momentum": 1.2,    # 牛市动量策略有优势
        "value": 0.9,
        "low_vol": 0.8,
        "reversion": 0.7,   # 牛市反转策略风险高
        "breakout": 1.15,
    },
    MarketRegime.BEAR: {
        "balanced": 0.9,
        "momentum": 0.7,    # 熊市追涨危险
        "value": 1.15,      # 熊市估值洼地
        "low_vol": 1.2,     # 熊市防御策略有优势
        "reversion": 1.1,
        "breakout": 0.7,
    },
    MarketRegime.SIDEWAYS: {
        "balanced": 1.1,
        "momentum": 0.8,
        "value": 1.0,
        "low_vol": 1.05,
        "reversion": 1.2,   # 震荡市反转策略最强
        "breakout": 0.8,
    },
    MarketRegime.CRISIS: {
        "balanced": 0.7,
        "momentum": 0.5,    # 危机时所有方向策略减弱
        "value": 0.8,
        "low_vol": 1.3,     # 危机时防御策略最强
        "reversion": 0.6,
        "breakout": 0.5,
    },
}

# 不同宏观周期对策略的修正
MACRO_STRATEGY_MODIFIERS = {
    MacroCycle.EXPANSION: {
        "momentum": 1.1,
        "value": 0.95,
        "low_vol": 0.9,
    },
    MacroCycle.PEAK: {
        "momentum": 0.9,
        "low_vol": 1.1,
    },
    MacroCycle.CONTRACTION: {
        "momentum": 0.8,
        "value": 1.1,
        "low_vol": 1.2,
    },
    MacroCycle.TROUGH: {
        "value": 1.2,      # 底部区域价值策略最强
        "reversion": 1.15,
    },
}


class MarketRegimeHMM:
    """基于隐马尔可夫模型的市场状态识别

    使用指数/个股的收益率、波动率、资金流等特征，
    通过 HMM 识别市场处于 Bull/Bear/Sideways/Crisis 中的哪个状态。

    当 hmmlearn 未安装时，自动降级到规则引擎。
    """

    def __init__(self, n_states: int = 4, lookback: int = 120):
        """
        Args:
            n_states: 隐状态数量（默认4=Bull/Bear/Sideways/Crisis）
            lookback: 训练窗口（交易日数）
        """
        self.n_states = n_states
        self.lookback = lookback
        self._hmm_model = None
        self._hmm_available = self._check_hmmlearn()

    def _check_hmmlearn(self) -> bool:
        try:
            import hmmlearn
            return True
        except ImportError:
            logger.info("hmmlearn 未安装，市场状态识别将使用规则引擎降级模式")
            return False

    def detect_regime(self, df: pd.DataFrame,
                      adx: Optional[float] = None,
                      vol20: Optional[float] = None,
                      m20: Optional[float] = None,
                      rsi: Optional[float] = None) -> Tuple[MarketRegime, float, str]:
        """识别当前市场状态

        Args:
            df: 指数或个股日线数据（需包含 close, volume 列）
            adx, vol20, m20, rsi: 可选的预计算技术指标（用于规则引擎降级）

        Returns:
            (MarketRegime, confidence, description)
            - regime: 市场状态枚举
            - confidence: 识别置信度 0-1
            - description: 中文描述
        """
        if self._hmm_available and len(df) >= self.lookback:
            return self._detect_hmm(df)
        else:
            return self._detect_rules(df, adx, vol20, m20, rsi)

    def _detect_hmm(self, df: pd.DataFrame) -> Tuple[MarketRegime, float, str]:
        """HMM 方式识别市场状态"""
        try:
            from hmmlearn.hmm import GaussianHMM

            # 特征工程
            close = df['close'].values[-self.lookback:]
            returns = np.diff(np.log(close))
            vol_5d = pd.Series(returns).rolling(5).std().values
            vol_20d = pd.Series(returns).rolling(20).std().values

            # 构建特征矩阵: [收益率, 5日波动率, 20日波动率]
            valid_start = 20  # 跳过 NaN
            features = np.column_stack([
                returns[valid_start:],
                vol_5d[valid_start:],
                vol_20d[valid_start:],
            ])

            # 去除 NaN
            mask = ~np.isnan(features).any(axis=1)
            features = features[mask]

            if len(features) < 30:
                return self._detect_rules(df)

            # 训练 HMM
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42,
            )
            model.fit(features)
            self._hmm_model = model

            # 预测当前状态
            states = model.predict(features)
            current_state = states[-1]
            proba = model.predict_proba(features)
            current_confidence = float(proba[-1, current_state])

            # 将 HMM 状态映射到有意义的市场状态
            # 策略: 根据每个隐状态的均值收益率和波动率来分配标签
            regime = self._map_hmm_state(model, current_state)

            desc = f"{regime.cn_name} (HMM置信度{current_confidence:.0%})"
            return regime, current_confidence, desc

        except Exception as e:
            logger.debug(f"HMM 市场状态识别失败，降级到规则引擎: {e}")
            return self._detect_rules(df)

    def _map_hmm_state(self, model, current_state: int) -> MarketRegime:
        """将 HMM 隐状态映射到市场状态枚举

        根据每个状态的均值收益率和波动率自动分配标签:
        - 高收益 + 低波动 → Bull
        - 低收益 + 低波动 → Sideways
        - 低收益 + 高波动 → Bear
        - 极端波动 → Crisis
        """
        means = model.means_
        # means[:, 0] = 收益率均值, means[:, 1]/[:, 2] = 波动率

        state_features = []
        for i in range(self.n_states):
            ret_mean = means[i, 0]
            vol_mean = means[i, 1] if means.shape[1] > 1 else 0
            state_features.append((i, ret_mean, vol_mean))

        # 按波动率排序，最高波动的为 Crisis
        state_features.sort(key=lambda x: x[2], reverse=True)

        # 最高波动 → Crisis
        crisis_state = state_features[0][0]

        # 剩余按收益率排序
        remaining = [s for s in state_features if s[0] != crisis_state]
        remaining.sort(key=lambda x: x[1], reverse=True)

        # 最高收益 → Bull, 最低收益 → Bear, 中间 → Sideways
        bull_state = remaining[0][0] if remaining else -1
        bear_state = remaining[-1][0] if remaining else -1
        sideways_state = remaining[1][0] if len(remaining) > 2 else remaining[-1][0]

        state_map = {
            crisis_state: MarketRegime.CRISIS,
            bull_state: MarketRegime.BULL,
            bear_state: MarketRegime.BEAR,
            sideways_state: MarketRegime.SIDEWAYS,
        }

        return state_map.get(current_state, MarketRegime.SIDEWAYS)

    def _detect_rules(self, df: pd.DataFrame,
                      adx: Optional[float] = None,
                      vol20: Optional[float] = None,
                      m20: Optional[float] = None,
                      rsi: Optional[float] = None) -> Tuple[MarketRegime, float, str]:
        """规则引擎降级模式

        升级版的 _describe_regime，输出标准化的 MarketRegime。
        """
        # 如果指标未预计算，从 df 推算
        if m20 is None and len(df) >= 20:
            close = df['close'].values
            m20 = (close[-1] / close[-20] - 1)

        if vol20 is None and len(df) >= 20:
            returns = np.diff(np.log(df['close'].values[-21:]))
            vol20 = float(np.std(returns))

        if rsi is None:
            rsi = 50.0  # 默认中性

        if adx is None:
            adx = 20.0  # 默认中等

        m20 = m20 or 0.0
        vol20 = vol20 or 0.03

        # 规则判断
        # Crisis: 高波动 + 大幅下跌
        if vol20 > 0.05 and m20 < -0.10:
            regime = MarketRegime.CRISIS
            confidence = min(0.9, 0.5 + vol20 * 5)
        # Bull: 上涨趋势 + 趋势明确
        elif m20 > 0.05 and adx > 20:
            regime = MarketRegime.BULL
            confidence = min(0.85, 0.5 + m20 * 3)
        # Bear: 下跌趋势
        elif m20 < -0.05:
            regime = MarketRegime.BEAR
            confidence = min(0.85, 0.5 + abs(m20) * 3)
        # Sideways: 其他
        else:
            regime = MarketRegime.SIDEWAYS
            confidence = 0.6

        parts = []
        if adx > 25:
            parts.append("趋势强")
        elif adx < 15:
            parts.append("无趋势")

        if vol20 > 0.04:
            parts.append("高波动")
        elif vol20 < 0.02:
            parts.append("低波动")

        if rsi > 70:
            parts.append("超买")
        elif rsi < 30:
            parts.append("超卖")

        detail = " | ".join(parts) if parts else ""
        desc = f"{regime.cn_name}"
        if detail:
            desc += f" ({detail})"
        desc += " [规则引擎]"

        return regime, confidence, desc

    def get_strategy_modifier(self, regime: MarketRegime,
                              strategy_key: str) -> float:
        """获取特定市场状态下策略的增益系数

        Args:
            regime: 当前市场状态
            strategy_key: 策略标识 (balanced/momentum/value/low_vol/reversion/breakout)

        Returns:
            增益系数 (>1 表示有利, <1 表示不利)
        """
        modifiers = REGIME_STRATEGY_MODIFIERS.get(regime, {})
        return modifiers.get(strategy_key, 1.0)


class MacroRegimeDetector:
    """宏观经济周期识别

    基于 PMI、CPI、利率等宏观指标判断经济处于
    扩张/高点/收缩/低谷中的哪个阶段。

    当宏观数据不可用时返回 None。
    """

    def detect_cycle(self, macro_data: Optional[Dict]) -> Optional[Tuple[MacroCycle, str]]:
        """识别宏观经济周期

        Args:
            macro_data: DataFetcher.get_macro_data() 的返回值

        Returns:
            (MacroCycle, description) 或 None
        """
        if not macro_data:
            return None

        try:
            pmi = self._extract_pmi(macro_data)
            cpi = self._extract_cpi(macro_data)

            if pmi is None:
                return None

            # PMI > 50 = 扩张, < 50 = 收缩
            # PMI 趋势决定是高点还是低谷
            pmi_trend = self._get_trend(pmi) if isinstance(pmi, pd.Series) else None
            pmi_latest = float(pmi.iloc[-1]) if isinstance(pmi, pd.Series) else float(pmi)

            if pmi_latest > 52:
                if pmi_trend and pmi_trend < 0:
                    cycle = MacroCycle.PEAK
                    desc = f"经济高点(PMI={pmi_latest:.1f}但已开始回落)"
                else:
                    cycle = MacroCycle.EXPANSION
                    desc = f"经济扩张期(PMI={pmi_latest:.1f})"
            elif pmi_latest > 50:
                cycle = MacroCycle.EXPANSION
                desc = f"经济温和扩张(PMI={pmi_latest:.1f})"
            elif pmi_latest > 48:
                if pmi_trend and pmi_trend > 0:
                    cycle = MacroCycle.TROUGH
                    desc = f"经济接近低谷(PMI={pmi_latest:.1f}但已企稳回升)"
                else:
                    cycle = MacroCycle.CONTRACTION
                    desc = f"经济轻度收缩(PMI={pmi_latest:.1f})"
            else:
                if pmi_trend and pmi_trend > 0:
                    cycle = MacroCycle.TROUGH
                    desc = f"经济低谷(PMI={pmi_latest:.1f}，正在回升)"
                else:
                    cycle = MacroCycle.CONTRACTION
                    desc = f"经济收缩期(PMI={pmi_latest:.1f})"

            return cycle, desc

        except Exception as e:
            logger.debug(f"宏观周期识别失败: {e}")
            return None

    def get_strategy_modifier(self, cycle: MacroCycle,
                              strategy_key: str) -> float:
        """获取宏观周期对策略的修正系数"""
        modifiers = MACRO_STRATEGY_MODIFIERS.get(cycle, {})
        return modifiers.get(strategy_key, 1.0)

    def _extract_pmi(self, macro_data: Dict) -> Optional[pd.Series]:
        """从宏观数据中提取 PMI"""
        pmi_data = macro_data.get('pmi')
        if pmi_data is None:
            return None
        if isinstance(pmi_data, pd.DataFrame):
            # 取第一个数值列
            for col in pmi_data.columns:
                if pmi_data[col].dtype in ('float64', 'int64'):
                    return pmi_data[col].dropna()
        if isinstance(pmi_data, pd.Series):
            return pmi_data.dropna()
        return None

    def _extract_cpi(self, macro_data: Dict):
        """从宏观数据中提取 CPI"""
        return macro_data.get('cpi')

    def _get_trend(self, series: pd.Series, window: int = 3) -> Optional[float]:
        """计算序列的短期趋势"""
        if len(series) < window + 1:
            return None
        recent = series.iloc[-window:].mean()
        prior = series.iloc[-(window * 2):-window].mean() if len(series) >= window * 2 else series.iloc[0]
        return recent - prior
