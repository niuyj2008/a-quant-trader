"""
A股量化交易系统 - 因子计算引擎

提供常用技术因子和财务因子的计算
"""

import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Optional, Union
from loguru import logger


class FactorEngine:
    """因子计算引擎"""
    
    def __init__(self):
        """初始化因子引擎"""
        self.factors: Dict[str, Callable] = {}
        self._register_builtin_factors()
    
    def _register_builtin_factors(self):
        """注册内置因子"""
        # 动量因子
        self.register("momentum_5", lambda df: self.momentum(df, 5))
        self.register("momentum_10", lambda df: self.momentum(df, 10))
        self.register("momentum_20", lambda df: self.momentum(df, 20))
        self.register("momentum_60", lambda df: self.momentum(df, 60))
        
        # 均线因子
        self.register("ma_5", lambda df: self.ma(df, 5))
        self.register("ma_10", lambda df: self.ma(df, 10))
        self.register("ma_20", lambda df: self.ma(df, 20))
        self.register("ma_60", lambda df: self.ma(df, 60))
        
        # 波动率因子
        self.register("volatility_10", lambda df: self.volatility(df, 10))
        self.register("volatility_20", lambda df: self.volatility(df, 20))
        
        # 技术指标
        self.register("rsi_14", lambda df: self.rsi(df, 14))
        self.register("macd", self.macd)
        self.register("bollinger", self.bollinger)
        
        # 成交量因子
        self.register("volume_ratio", self.volume_ratio)
        self.register("turnover_rate", lambda df: df.get('turnover', pd.Series()))
    
    def register(self, name: str, func: Callable):
        """
        注册自定义因子
        
        Args:
            name: 因子名称
            func: 因子计算函数，接收DataFrame返回Series
        """
        self.factors[name] = func
        logger.debug(f"注册因子: {name}")
    
    def compute(self, df: pd.DataFrame, factor_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        计算因子
        
        Args:
            df: OHLCV数据
            factor_names: 要计算的因子列表，为None时计算所有因子
            
        Returns:
            包含因子值的DataFrame
        """
        if factor_names is None:
            factor_names = list(self.factors.keys())
        
        result = df.copy()
        
        for name in factor_names:
            if name not in self.factors:
                logger.warning(f"因子 {name} 未注册")
                continue
            
            try:
                result[name] = self.factors[name](df)
            except Exception as e:
                logger.error(f"计算因子 {name} 失败: {e}")
        
        return result
    
    # ============ 内置因子计算方法 ============
    
    @staticmethod
    def momentum(df: pd.DataFrame, period: int) -> pd.Series:
        """
        动量因子: N日收益率
        
        Args:
            df: OHLCV数据
            period: 周期
        """
        return df['close'].pct_change(period)
    
    @staticmethod
    def ma(df: pd.DataFrame, period: int) -> pd.Series:
        """
        均线因子: N日移动平均
        
        Args:
            df: OHLCV数据
            period: 周期
        """
        return df['close'].rolling(window=period).mean()
    
    @staticmethod
    def volatility(df: pd.DataFrame, period: int) -> pd.Series:
        """
        波动率因子: N日收益率标准差
        
        Args:
            df: OHLCV数据
            period: 周期
        """
        returns = df['close'].pct_change()
        return returns.rolling(window=period).std()
    
    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        RSI相对强弱指标
        
        Args:
            df: OHLCV数据
            period: 周期，默认14
        """
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        MACD指标
        
        Args:
            df: OHLCV数据
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期
            
        Returns:
            DataFrame with macd, signal, histogram
        """
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_hist': histogram
        })
    
    @staticmethod
    def bollinger(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        布林带
        
        Args:
            df: OHLCV数据
            period: 周期
            std_dev: 标准差倍数
            
        Returns:
            DataFrame with upper, middle, lower bands
        """
        middle = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        return pd.DataFrame({
            'boll_upper': upper,
            'boll_middle': middle,
            'boll_lower': lower
        })
    
    @staticmethod
    def volume_ratio(df: pd.DataFrame, period: int = 5) -> pd.Series:
        """
        量比: 当日成交量 / N日平均成交量
        
        Args:
            df: OHLCV数据
            period: 周期
        """
        avg_volume = df['volume'].rolling(window=period).mean()
        return df['volume'] / avg_volume


class AlphaFactors:
    """Alpha因子库 - 选股因子"""
    
    @staticmethod
    def alpha001(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#001: (-1 * corr(rank(delta(log(volume), 1)), rank(((close - open) / open)), 6))
        """
        log_vol_delta = np.log(df['volume']).diff()
        price_change = (df['close'] - df['open']) / df['open']
        
        return -1 * log_vol_delta.rolling(6).corr(price_change)
    
    @staticmethod
    def alpha002(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#002: -1 * delta((((close - low) - (high - close)) / (high - low)), 1)
        """
        factor = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        return -1 * factor.diff()
    
    @staticmethod
    def alpha003(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#003: 5日反转因子
        """
        return -df['close'].pct_change(5)
    
    @staticmethod
    def alpha004(df: pd.DataFrame) -> pd.Series:
        """
        Alpha#004: 成交量加权价格变动
        """
        vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap.pct_change(5)
    
    @staticmethod
    def size_factor(market_cap: pd.Series) -> pd.Series:
        """
        市值因子: log(市值)
        """
        return np.log(market_cap)
    
    @staticmethod
    def value_factor(pe: pd.Series, pb: pd.Series) -> pd.Series:
        """
        价值因子: 1/PE + 1/PB
        """
        return 1/pe + 1/pb


if __name__ == "__main__":
    # 测试代码
    from src.data import get_stock_data
    
    # 获取测试数据
    df = get_stock_data("000001", start_date="2024-01-01")
    
    # 初始化因子引擎
    engine = FactorEngine()
    
    # 计算因子
    result = engine.compute(df, ["momentum_5", "momentum_20", "rsi_14", "ma_20"])
    print(result.tail())
