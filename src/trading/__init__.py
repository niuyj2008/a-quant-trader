"""
股票量化策略决策支持系统 - 交易模块
"""

from .paper_trading import PaperTradingEngine, Order, Position
from .risk import RiskManager, RiskConfig, PositionRisk, PositionSizer
from .trade_journal import TradeJournal

# vnpy_broker 是可选依赖，不影响核心功能
try:
    from .vnpy_broker import (
        BaseBroker, VnpyBroker, SimulatedBroker, LiveTrader,
        BrokerConfig, OrderInfo, create_broker
    )
except Exception:
    pass

__all__ = [
    "PaperTradingEngine", "Order", "Position",
    "RiskManager", "RiskConfig", "PositionRisk", "PositionSizer",
    "TradeJournal",
]

