"""
A股量化交易系统 - 数据模块
"""

from .fetcher import DataFetcher, get_stock_data, get_all_stocks

__all__ = ["DataFetcher", "get_stock_data", "get_all_stocks"]
