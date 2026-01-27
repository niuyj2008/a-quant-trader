import akshare as ak
from datetime import datetime
import sys

try:
    print(f"Testing AKShare. Version: {ak.__version__}")
    df = ak.stock_zh_a_hist(
        symbol="600654",
        period="daily",
        start_date="20250101",
        end_date=datetime.now().strftime("%Y%m%d"),
        adjust="qfq"
    )
    print("Fetch successful!")
    print(df.tail())
except Exception as e:
    print(f"Fetch failed: {e}")
    sys.exit(1)
