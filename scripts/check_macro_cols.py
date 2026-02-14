
import akshare as ak

def check():
    print("Checking PMI Columns...")
    try:
        df = ak.macro_china_pmi()
        print("PMI Columns:", df.columns.tolist())
    except Exception as e:
        print("PMI Error:", e)

    print("\nChecking M2 Columns...")
    try:
        df = ak.macro_china_money_supply()
        print("M2 Columns:", df.columns.tolist())
    except Exception as e:
        print("M2 Error:", e)

if __name__ == "__main__":
    check()
