
import akshare as ak
import pandas as pd

def check():
    print("Checking Margin Data Columns...")
    try:
        sh = ak.stock_margin_sse(start_date="20240101")
        print(f"SH Margin Columns: {sh.columns.tolist() if not sh.empty else 'Empty'}")
        if not sh.empty: print(sh.head(2))
        
        sz = ak.stock_margin_szse(start_date="20240101")
        print(f"SZ Margin Columns: {sz.columns.tolist() if not sz.empty else 'Empty'}")
        if not sz.empty: print(sz.head(2))
    except Exception as e:
        print(f"Error: {e}")

    print("\nChecking Northbound Data Columns...")
    try:
        nb = ak.stock_hsgt_hist_em(symbol="北向资金")
        print(f"Northbound Columns: {nb.columns.tolist() if not nb.empty else 'Empty'}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check()
