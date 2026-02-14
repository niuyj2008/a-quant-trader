
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import akshare as ak
import pandas as pd
from src.data.fetcher import DataFetcher

def debug():
    print("=== DEBUG START ===")
    
    # 1. Macro GDP Raw
    print("\n[1] Checking GDP Raw Data...")
    try:
        df = ak.macro_china_gdp()
        print("Columns:", df.columns.tolist())
        print("First 5 rows of Date column:")
        print(df.iloc[:5, 0].values)
    except Exception as e:
        print("GDP Error:", e)

    # 2. CPI Raw
    print("\n[2] Checking CPI Raw Data...")
    try:
        df = ak.macro_china_cpi_monthly()
        print("Columns:", df.columns.tolist())
        print("First 5 rows of Date column:")
        print(df.iloc[:5, 0].values)
    except Exception as e:
        print("CPI Error:", e)

    # 3. Northbound Raw
    print("\n[3] Checking Northbound Raw Data...")
    try:
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
        print("Columns:", df.columns.tolist())
        # Check specific column
        target_col = '当日成交净买额'
        if target_col in df.columns:
            print(f"Sample {target_col}:")
            print(df[target_col].tail(10))
            print(f"Valid count: {df[target_col].notna().sum()}/{len(df)}")
        else:
            print(f"Column {target_col} NOT FOUND")
    except Exception as e:
        print("Northbound Error:", e)

    # 4. Fetcher Logic Test
    print("\n[4] Testing Fetcher Logic...")
    f = DataFetcher()
    macro = f.get_macro_data()
    print("Macro Keys:", macro.keys())
    
    sentiment = f.get_sentiment_data("CN")
    print("Sentiment Keys:", sentiment.keys())
    if 'northbound_flow' in sentiment:
        print("Northbound flow shape:", sentiment['northbound_flow'].shape)
    else:
        print("Northbound flow MISSING in result")

if __name__ == "__main__":
    debug()
