"""
A股量化交易系统 - 交易模块
"""

from .paper_trading import PaperTradingEngine, Order, Position
from .risk import RiskManager, RiskConfig, PositionRisk, PositionSizer
from .vnpy_broker import (
    BaseBroker, VnpyBroker, SimulatedBroker, LiveTrader,
    BrokerConfig, OrderInfo, create_broker
)

__all__ = [
    # Paper Trading
    "PaperTradingEngine",
    "Order",
    "Position",
    # Risk Management
    "RiskManager",
    "RiskConfig",
    "PositionRisk",
    "PositionSizer",
    # VeighNa Broker
    "BaseBroker",
    "VnpyBroker",
    "SimulatedBroker",
    "LiveTrader",
    "BrokerConfig",
    "OrderInfo",
    "create_broker",
]

