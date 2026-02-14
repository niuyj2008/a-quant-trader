
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.fetcher import DataFetcher
import pandas as pd

def debug_data():
    fetcher = DataFetcher(source="akshare")
    
    print("="*20 + " DEBUG START " + "="*20)
    
    # 1. Macro Data
    print("\n[1] Testing Macro Data...")
    try:
        macro = fetcher.get_macro_data()
        print(f"Keys found: {list(macro.keys())}")
        for k, v in macro.items():
            print(f"--- {k} ---")
            print(f"Type: {type(v)}")
            if isinstance(v, (pd.DataFrame, pd.Series)):
                print(f"Shape: {v.shape}")
                print("Tail:")
                print(v.tail())
            else:
                print(f"Value: {v}")
    except Exception as e:
        print(f"!! Macro Fetch Error: {e}")

    # 2. Sentiment Data
    print("\n[2] Testing Sentiment Data (CN)...")
    try:
        sentiment = fetcher.get_sentiment_data(market="CN")
        print(f"Keys found: {list(sentiment.keys())}")
        for k, v in sentiment.items():
            print(f"--- {k} ---")
            print(f"Type: {type(v)}")
            if isinstance(v, pd.DataFrame):
                print(f"Columns: {v.columns.tolist()}")
                print("Tail:")
                print(v.tail())
            else:
                print(f"Value: {v}")
    except Exception as e:
        print(f"!! Sentiment Fetch Error: {e}")

    print("\n[3] Testing Financial Data (CN - 000001)...")
    try:
        fin = fetcher.get_financial_data("000001", market="CN")
        print(f"Financial Data: {fin}")
    except Exception as e:
        print(f"!! Financial Fetch Error: {e}")

    print("="*20 + " DEBUG END " + "="*20)

if __name__ == "__main__":
    debug_data()
