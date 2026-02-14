"""
股票量化策略决策支持系统 - 策略模块
"""

from .base import BaseStrategy, Signal, MomentumStrategy, MACrossStrategy, DualThrustStrategy
from .interpretable_strategy import (
    BaseInterpretableStrategy, DecisionReport, StockSignal,
    BalancedMultiFactorStrategy, MomentumTrendStrategy, ValueInvestStrategy,
    LowVolDefenseStrategy, MeanReversionStrategy, TechnicalBreakoutStrategy,
    STRATEGY_REGISTRY, STRATEGY_NAMES, STRATEGY_DESCRIPTIONS, STRATEGY_RISK_LEVELS,
    get_strategy, get_all_strategies, multi_strategy_analysis,
)

__all__ = [
    "BaseStrategy", "Signal",
    "MomentumStrategy", "MACrossStrategy", "DualThrustStrategy",
    # 可解释策略
    "BaseInterpretableStrategy", "DecisionReport", "StockSignal",
    "BalancedMultiFactorStrategy", "MomentumTrendStrategy", "ValueInvestStrategy",
    "LowVolDefenseStrategy", "MeanReversionStrategy", "TechnicalBreakoutStrategy",
    "STRATEGY_REGISTRY", "STRATEGY_NAMES", "STRATEGY_DESCRIPTIONS", "STRATEGY_RISK_LEVELS",
    "get_strategy", "get_all_strategies", "multi_strategy_analysis",
]
