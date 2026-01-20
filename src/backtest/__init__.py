"""
A股量化交易系统 - 回测模块
"""

from .engine import BacktestEngine, BacktestConfig, BacktestResult, Position, Trade

__all__ = ["BacktestEngine", "BacktestConfig", "BacktestResult", "Position", "Trade"]
