"""
股票量化策略决策支持系统 - 回测模块
"""

from .engine import BacktestEngine, BacktestConfig, BacktestResult, Position, Trade
from .walk_forward import WalkForwardValidator, WalkForwardResult, WalkForwardWindow

__all__ = [
    "BacktestEngine", "BacktestConfig", "BacktestResult", "Position", "Trade",
    "WalkForwardValidator", "WalkForwardResult", "WalkForwardWindow",
]
