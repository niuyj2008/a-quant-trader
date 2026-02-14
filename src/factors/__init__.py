"""
股票量化策略决策支持系统 - 因子模块
"""

from .factor_engine import FactorEngine, AlphaFactors, FACTOR_CATEGORIES
from .factor_analyzer import FactorAnalyzer

__all__ = ["FactorEngine", "AlphaFactors", "FACTOR_CATEGORIES", "FactorAnalyzer"]
