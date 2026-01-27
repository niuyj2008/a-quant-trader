from src.data import DataFetcher
import logging

# Set logging level to see the fallback warnings
import sys
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

def test_fallback():
    fetcher = DataFetcher(source="akshare")
    print("Attempting to fetch data for 600654 (which is failing in AKShare)...")
    try:
        df = fetcher.get_daily_data("600654", start_date="2025-01-01")
        if not df.empty:
            print("Successfully fetched data (likely via yfinance fallback)!")
            print(df.tail())
        else:
            print("Fetched DataFrame is empty.")
    except Exception as e:
        print(f"Fetch failed even with fallback: {e}")

if __name__ == "__main__":
    test_fallback()
