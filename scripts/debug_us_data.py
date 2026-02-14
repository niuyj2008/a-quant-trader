
import yfinance as yf
import pandas as pd

def debug_us():
    print("Fetching VIX...")
    try:
        vix = yf.download("^VIX", period="1y", progress=False)
        if not vix.empty:
            print("VIX Data Found:")
            print(vix.tail())
        else:
            print("VIX Empty")
    except Exception as e:
        print(f"VIX Error: {e}")

    print("\nFetching US 10Y Yield (^TNX)...")
    try:
        tnx = yf.download("^TNX", period="1y", progress=False)
        if not tnx.empty:
            print("TNX Data Found:")
            print(tnx.tail())
        else:
            print("TNX Empty")
    except Exception as e:
        print(f"TNX Error: {e}")

if __name__ == "__main__":
    debug_us()
