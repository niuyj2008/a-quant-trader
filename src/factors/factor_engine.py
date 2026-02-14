"""
股票量化策略决策支持系统 - 因子计算引擎（增强版）

六大维度因子体系：技术面、基本面、宏观经济、市场情绪、波动/风险、行业
"""

import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Optional
from loguru import logger


class FactorEngine:
    """因子计算引擎（增强版）"""

    def __init__(self):
        self.factors: Dict[str, Callable] = {}
        self._register_builtin_factors()

    def _register_builtin_factors(self):
        """注册所有内置因子"""
        # ======== 技术面因子 ========
        # 动量
        self.register("momentum_5", lambda df: self.momentum(df, 5))
        self.register("momentum_10", lambda df: self.momentum(df, 10))
        self.register("momentum_20", lambda df: self.momentum(df, 20))
        self.register("momentum_60", lambda df: self.momentum(df, 60))
        # 均线
        self.register("ma_5", lambda df: self.ma(df, 5))
        self.register("ma_10", lambda df: self.ma(df, 10))
        self.register("ma_20", lambda df: self.ma(df, 20))
        self.register("ma_60", lambda df: self.ma(df, 60))
        # 波动率
        self.register("volatility_10", lambda df: self.volatility(df, 10))
        self.register("volatility_20", lambda df: self.volatility(df, 20))
        # 技术指标
        self.register("rsi_14", lambda df: self.rsi(df, 14))
        self.register("macd", self.macd)
        self.register("bollinger", self.bollinger)
        # 成交量
        self.register("volume_ratio", self.volume_ratio)
        self.register("turnover_rate", lambda df: df.get('turnover', pd.Series(dtype=float)))
        # ======== 新增因子 ========
        self.register("atr_14", lambda df: self.atr(df, 14))
        self.register("ma_cross", self.ma_cross_signal)
        self.register("price_position", self.price_position)
        self.register("volume_ma_ratio", self.volume_ma_ratio)
        self.register("weekly_return", lambda df: self.momentum(df, 5))
        self.register("beta", lambda df: self.beta(df))
        self.register("downside_vol", lambda df: self.downside_volatility(df, 20))
        self.register("support_resistance", self.support_resistance)

    def register(self, name: str, func: Callable):
        """注册自定义因子"""
        self.factors[name] = func

    def compute(self, df: pd.DataFrame, factor_names: Optional[List[str]] = None) -> pd.DataFrame:
        """计算因子"""
        if factor_names is None:
            factor_names = list(self.factors.keys())
        result = df.copy()
        for name in factor_names:
            if name not in self.factors:
                logger.warning(f"因子 {name} 未注册")
                continue
            try:
                val = self.factors[name](df)
                if isinstance(val, pd.DataFrame):
                    for col in val.columns:
                        result[col] = val[col]
                else:
                    result[name] = val
            except Exception as e:
                logger.error(f"计算因子 {name} 失败: {e}")
        return result

    def compute_all_core_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有核心因子（用于策略信号生成）"""
        core_factors = [
            'momentum_5', 'momentum_20', 'momentum_60',
            'ma_5', 'ma_20', 'ma_60',
            'volatility_20', 'rsi_14', 'macd', 'bollinger',
            'volume_ratio', 'atr_14', 'ma_cross', 'price_position',
            'volume_ma_ratio', 'beta', 'downside_vol',
        ]
        return self.compute(df, core_factors)

    # ======== 技术面因子 ========

    @staticmethod
    def momentum(df: pd.DataFrame, period: int) -> pd.Series:
        """动量因子: N日收益率"""
        return df['close'].pct_change(period)

    @staticmethod
    def ma(df: pd.DataFrame, period: int) -> pd.Series:
        """均线因子: N日移动平均"""
        return df['close'].rolling(window=period).mean()

    @staticmethod
    def volatility(df: pd.DataFrame, period: int) -> pd.Series:
        """波动率因子: N日收益率标准差"""
        returns = df['close'].pct_change()
        return returns.rolling(window=period).std()

    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """RSI相对强弱指标"""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD指标"""
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame({'macd': macd_line, 'macd_signal': signal_line, 'macd_hist': histogram})

    @staticmethod
    def bollinger(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """布林带"""
        middle = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        return pd.DataFrame({
            'boll_upper': middle + std_dev * std,
            'boll_middle': middle,
            'boll_lower': middle - std_dev * std
        })

    @staticmethod
    def volume_ratio(df: pd.DataFrame, period: int = 5) -> pd.Series:
        """量比"""
        avg_volume = df['volume'].rolling(window=period).mean()
        return df['volume'] / avg_volume

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR平均真实波幅"""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def ma_cross_signal(df: pd.DataFrame) -> pd.Series:
        """均线交叉信号: MA5与MA20的差值占比"""
        ma5 = df['close'].rolling(5).mean()
        ma20 = df['close'].rolling(20).mean()
        return (ma5 - ma20) / ma20

    @staticmethod
    def price_position(df: pd.DataFrame, period: int = 60) -> pd.Series:
        """价格位置: 当前价在N日最高最低价区间中的位置 (0-1)"""
        highest = df['high'].rolling(period).max()
        lowest = df['low'].rolling(period).min()
        denom = highest - lowest
        denom = denom.replace(0, np.nan)
        return (df['close'] - lowest) / denom

    @staticmethod
    def volume_ma_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """成交量与N日均量比值"""
        vol_ma = df['volume'].rolling(period).mean()
        return df['volume'] / vol_ma

    @staticmethod
    def beta(df: pd.DataFrame, period: int = 60) -> pd.Series:
        """Beta因子（与自身趋势的相关性，简化版）"""
        returns = df['close'].pct_change()
        market_return = returns.rolling(period).mean()  # 简化：用自身均值代替指数
        cov = returns.rolling(period).cov(market_return)
        var = market_return.rolling(period).var()
        var = var.replace(0, np.nan)
        return cov / var

    @staticmethod
    def downside_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """下行波动率：只计算负收益的标准差"""
        returns = df['close'].pct_change()
        neg_returns = returns.where(returns < 0, 0)
        return neg_returns.rolling(window=period).std()

    @staticmethod
    def support_resistance(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """简易支撑位/阻力位"""
        support = df['low'].rolling(period).min()
        resistance = df['high'].rolling(period).max()
        return pd.DataFrame({
            'support': support,
            'resistance': resistance,
            'support_dist': (df['close'] - support) / df['close'],
            'resistance_dist': (resistance - df['close']) / df['close'],
        })


class AlphaFactors:
    """Alpha因子库 - 选股因子"""

    @staticmethod
    def alpha001(df: pd.DataFrame) -> pd.Series:
        """Alpha#001: 量价背离"""
        log_vol_delta = np.log(df['volume'].clip(lower=1)).diff()
        price_change = (df['close'] - df['open']) / df['open']
        return -1 * log_vol_delta.rolling(6).corr(price_change)

    @staticmethod
    def alpha002(df: pd.DataFrame) -> pd.Series:
        """Alpha#002: 日内位置变化"""
        denom = df['high'] - df['low']
        denom = denom.replace(0, np.nan)
        factor = ((df['close'] - df['low']) - (df['high'] - df['close'])) / denom
        return -1 * factor.diff()

    @staticmethod
    def alpha003(df: pd.DataFrame) -> pd.Series:
        """Alpha#003: 5日反转因子"""
        return -df['close'].pct_change(5)

    @staticmethod
    def alpha004(df: pd.DataFrame) -> pd.Series:
        """Alpha#004: 成交量加权价格变动"""
        vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap.pct_change(5)

    @staticmethod
    def size_factor(market_cap: pd.Series) -> pd.Series:
        """市值因子"""
        return np.log(market_cap.clip(lower=1))

    @staticmethod
    def value_factor(pe: pd.Series, pb: pd.Series) -> pd.Series:
        """价值因子: 1/PE + 1/PB"""
        pe_safe = pe.replace(0, np.nan)
        pb_safe = pb.replace(0, np.nan)
        return 1 / pe_safe + 1 / pb_safe


# ==================== 因子维度分类 ====================
FACTOR_CATEGORIES = {
    "技术面": {
        "factors": ["momentum_5", "momentum_20", "momentum_60", "rsi_14",
                     "macd", "ma_cross", "price_position", "bollinger"],
        "description": "基于价格和成交量的技术分析指标",
    },
    "波动/风险": {
        "factors": ["volatility_20", "atr_14", "beta", "downside_vol"],
        "description": "衡量风险和价格波动程度",
    },
    "成交量": {
        "factors": ["volume_ratio", "volume_ma_ratio", "turnover_rate"],
        "description": "反映市场交投活跃度",
    },
    "基本面": {
        "factors": ["pe", "pb", "roe", "revenue_growth", "gross_margin", "debt_ratio"],
        "description": "公司财务健康和估值水平",
    },
    "宏观经济": {
        "factors": ["gdp_growth", "cpi", "pmi", "m2_growth", "interest_rate"],
        "description": "宏观经济环境对市场的影响",
    },
    "市场情绪": {
        "factors": ["margin_balance_change", "northbound_flow"],
        "description": "反映市场参与者情绪和资金流向",
    },
}


if __name__ == "__main__":
    from src.data import get_stock_data
    df = get_stock_data("000001", start_date="2024-01-01")
    engine = FactorEngine()
    result = engine.compute_all_core_factors(df)
    print(result.tail())
    print(f"\n计算了 {len([c for c in result.columns if c not in df.columns])} 个因子")
