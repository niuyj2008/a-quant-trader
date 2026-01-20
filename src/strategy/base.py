"""
A股量化交易系统 - 策略基类

提供策略开发的基础框架
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Signal:
    """交易信号"""
    code: str
    action: str  # "buy", "sell", "hold"
    shares: int = 100
    price: Optional[float] = None
    reason: str = ""
    confidence: float = 1.0


class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, name: str = "BaseStrategy", params: Optional[Dict] = None):
        """
        初始化策略
        
        Args:
            name: 策略名称
            params: 策略参数
        """
        self.name = name
        self.params = params or {}
        self.positions: Dict[str, int] = {}  # 当前持仓
        
    @abstractmethod
    def generate_signals(
        self,
        date: datetime,
        data: Dict[str, pd.DataFrame]
    ) -> List[Signal]:
        """
        生成交易信号
        
        Args:
            date: 当前日期
            data: 股票历史数据
            
        Returns:
            交易信号列表
        """
        pass
    
    def on_trade(self, signal: Signal, success: bool):
        """
        交易回调
        
        Args:
            signal: 交易信号
            success: 是否成功
        """
        if success:
            if signal.action == "buy":
                self.positions[signal.code] = self.positions.get(signal.code, 0) + signal.shares
            elif signal.action == "sell":
                self.positions[signal.code] = self.positions.get(signal.code, 0) - signal.shares
                if self.positions[signal.code] <= 0:
                    del self.positions[signal.code]
    
    def __call__(self, engine, date: datetime, data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """使策略可以作为回测引擎的策略函数调用"""
        signals = self.generate_signals(date, data)
        return [
            {'code': s.code, 'action': s.action, 'shares': s.shares}
            for s in signals
        ]


class MomentumStrategy(BaseStrategy):
    """动量策略"""
    
    def __init__(self, lookback: int = 20, top_n: int = 5):
        """
        Args:
            lookback: 动量计算周期
            top_n: 持有股票数量
        """
        super().__init__(
            name="MomentumStrategy",
            params={'lookback': lookback, 'top_n': top_n}
        )
        self.lookback = lookback
        self.top_n = top_n
    
    def generate_signals(self, date: datetime, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        signals = []
        
        # 计算动量
        momentums = {}
        for code, df in data.items():
            if len(df) >= self.lookback:
                momentum = df['close'].iloc[-1] / df['close'].iloc[-self.lookback] - 1
                momentums[code] = momentum
        
        if not momentums:
            return signals
        
        # 排序选择Top N
        sorted_codes = sorted(momentums.items(), key=lambda x: x[1], reverse=True)
        top_codes = [code for code, _ in sorted_codes[:self.top_n]]
        
        # 卖出不在Top N的持仓
        for code in list(self.positions.keys()):
            if code not in top_codes:
                signals.append(Signal(
                    code=code,
                    action="sell",
                    shares=self.positions[code],
                    reason="动量排名下降"
                ))
        
        # 买入新进入Top N的股票
        for code in top_codes:
            if code not in self.positions:
                signals.append(Signal(
                    code=code,
                    action="buy",
                    shares=100,
                    reason=f"动量排名前{self.top_n}"
                ))
        
        return signals


class MACrossStrategy(BaseStrategy):
    """均线交叉策略"""
    
    def __init__(self, short_period: int = 5, long_period: int = 20):
        """
        Args:
            short_period: 短期均线周期
            long_period: 长期均线周期
        """
        super().__init__(
            name="MACrossStrategy",
            params={'short_period': short_period, 'long_period': long_period}
        )
        self.short_period = short_period
        self.long_period = long_period
    
    def generate_signals(self, date: datetime, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        signals = []
        
        for code, df in data.items():
            if len(df) < self.long_period + 1:
                continue
            
            # 计算均线
            ma_short = df['close'].rolling(self.short_period).mean()
            ma_long = df['close'].rolling(self.long_period).mean()
            
            # 检测交叉
            prev_short, curr_short = ma_short.iloc[-2], ma_short.iloc[-1]
            prev_long, curr_long = ma_long.iloc[-2], ma_long.iloc[-1]
            
            # 金叉买入
            if prev_short < prev_long and curr_short >= curr_long:
                if code not in self.positions:
                    signals.append(Signal(
                        code=code,
                        action="buy",
                        shares=100,
                        reason="金叉"
                    ))
            
            # 死叉卖出
            elif prev_short > prev_long and curr_short <= curr_long:
                if code in self.positions:
                    signals.append(Signal(
                        code=code,
                        action="sell",
                        shares=self.positions[code],
                        reason="死叉"
                    ))
        
        return signals


class DualThrustStrategy(BaseStrategy):
    """Dual Thrust策略"""
    
    def __init__(self, lookback: int = 4, k1: float = 0.5, k2: float = 0.5):
        """
        Args:
            lookback: 回看周期
            k1: 上轨系数
            k2: 下轨系数
        """
        super().__init__(
            name="DualThrustStrategy",
            params={'lookback': lookback, 'k1': k1, 'k2': k2}
        )
        self.lookback = lookback
        self.k1 = k1
        self.k2 = k2
    
    def generate_signals(self, date: datetime, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        signals = []
        
        for code, df in data.items():
            if len(df) < self.lookback + 1:
                continue
            
            # 计算Range
            hh = df['high'].iloc[-self.lookback-1:-1].max()
            hc = df['close'].iloc[-self.lookback-1:-1].max()
            lc = df['close'].iloc[-self.lookback-1:-1].min()
            ll = df['low'].iloc[-self.lookback-1:-1].min()
            
            range_val = max(hh - lc, hc - ll)
            
            # 计算上下轨
            open_price = df['open'].iloc[-1]
            upper = open_price + self.k1 * range_val
            lower = open_price - self.k2 * range_val
            
            current_price = df['close'].iloc[-1]
            
            # 突破上轨买入
            if current_price > upper and code not in self.positions:
                signals.append(Signal(
                    code=code,
                    action="buy",
                    shares=100,
                    reason="突破上轨"
                ))
            
            # 突破下轨卖出
            elif current_price < lower and code in self.positions:
                signals.append(Signal(
                    code=code,
                    action="sell",
                    shares=self.positions[code],
                    reason="突破下轨"
                ))
        
        return signals


if __name__ == "__main__":
    # 测试策略
    from src.data import DataFetcher
    from src.backtest import BacktestEngine
    
    # 获取数据
    fetcher = DataFetcher()
    codes = ["000001", "000002", "600000", "600036", "601398"]
    data = {code: fetcher.get_daily_data(code, start_date="2024-01-01") for code in codes}
    
    # 测试均线策略
    strategy = MACrossStrategy(short_period=5, long_period=20)
    engine = BacktestEngine()
    result = engine.run(data, strategy)
    
    print("均线交叉策略回测结果:")
    for k, v in result.summary().items():
        print(f"  {k}: {v}")
