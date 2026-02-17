"""
股票量化策略决策支持系统 - 市场抽象层

统一 A股/美股 的交易规则、代码格式和因子差异
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MarketConfig:
    """市场配置"""
    name: str              # 市场名称
    code: str              # 市场代码 CN / US
    currency: str          # 货币
    lot_size: int          # 最小交易单位
    commission_rate: float # 佣金费率
    stamp_duty: float      # 印花税（卖出）
    slippage: float        # 默认滑点
    index_code: str        # 基准指数代码
    index_name: str        # 基准指数名称
    trading_hours: str     # 交易时间
    has_limit_updown: bool # 是否有涨跌停限制
    # 因子可用性
    available_sentiment_factors: List[str] = field(default_factory=list)


# 预定义市场配置
MARKET_CN = MarketConfig(
    name="A股", code="CN", currency="CNY",
    lot_size=100, commission_rate=0.0003, stamp_duty=0.001,
    slippage=0.001, index_code="000300", index_name="沪深300",
    trading_hours="09:30-15:00 CST", has_limit_updown=True,
    available_sentiment_factors=["margin_balance", "northbound_flow", "market_turnover"]
)

MARKET_US = MarketConfig(
    name="美股", code="US", currency="USD",
    lot_size=1, commission_rate=0.0, stamp_duty=0.0,
    slippage=0.0005, index_code="^GSPC", index_name="S&P 500",
    trading_hours="09:30-16:00 ET", has_limit_updown=False,
    available_sentiment_factors=["vix", "put_call_ratio"]
)

MARKETS = {"CN": MARKET_CN, "US": MARKET_US}


def get_market(market_code: str) -> MarketConfig:
    """获取市场配置"""
    market_code = market_code.upper()
    if market_code not in MARKETS:
        raise ValueError(f"不支持的市场: {market_code}，可选: {list(MARKETS.keys())}")
    return MARKETS[market_code]


# A股主流ETF池(10只核心ETF)
CN_ETF_POOL = [
    "510300",  # 沪深300ETF
    "510500",  # 中证500ETF
    "159915",  # 创业板ETF
    "515000",  # 科创50
    "512880",  # 证券ETF
    "515050",  # 5GETF
    "518880",  # 黄金ETF
    "511260",  # 上证10年期国债ETF
    "513100",  # 纳斯达克ETF
    "513500",  # 标普500ETF
]

# 美股主流ETF池(10只核心ETF)
US_ETF_POOL = [
    # 宽基指数
    "SPY",     # S&P 500 ETF Trust
    "QQQ",     # Invesco QQQ (纳斯达克100)
    "IWM",     # iShares Russell 2000 (小盘)
    "VTI",     # Vanguard Total Stock Market

    # 行业/主题
    "XLK",     # Technology Select Sector
    "XLF",     # Financial Select Sector

    # 国际/新兴市场
    "EEM",     # iShares MSCI Emerging Markets

    # 债券
    "TLT",     # iShares 20+ Year Treasury Bond
    "AGG",     # iShares Core U.S. Aggregate Bond

    # 商品
    "GLD",     # SPDR Gold Shares
]


def get_stock_pool(market_code: str, size: str = "default") -> List[str]:
    """获取默认股票池

    Args:
        market_code: 市场代码 CN / US
        size: 股票池大小
            - "default": 默认精选池（30-40只）
            - "sp500": 美股S&P 500训练池（约500只，仅US市场）
    """
    if market_code == "CN":
        return [
            # 沪深300 部分代表性个股
            "000001", "000002", "000063", "000333", "000651",
            "000858", "002230", "002415", "002594", "002714",
            "300750", "600000", "600009", "600016", "600028",
            "600030", "600036", "600050", "600104", "600276",
            "600309", "600519", "600585", "600690", "600887",
            "601006", "601012", "601088", "601166", "601169",
            "601225", "601288", "601318", "601398", "601601",
            "601628", "601668", "601688", "601857", "601988",
        ]
    elif market_code == "US" and size == "sp500":
        return _get_sp500_pool()
    else:
        return [
            # 美股精选 — 大盘科技 + 价值蓝筹
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "TSLA", "META", "BRK-B", "JPM", "V",
            "UNH", "LLY", "MA", "HD", "COST",
            "AVGO", "PG", "JNJ", "ABBV", "WMT",
            "BAC", "CRM", "KO", "PEP", "MRK",
            "AMD", "NFLX", "INTC", "DIS", "CSCO",
        ]


def _get_sp500_pool() -> List[str]:
    """获取S&P 500成分股列表（用于训练数据集）

    优先从Wikipedia动态获取，失败时使用静态列表作为fallback。
    """
    try:
        import pandas as pd
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = table[0]
        symbols = df['Symbol'].str.replace('.', '-', regex=False).tolist()
        if len(symbols) >= 400:
            return symbols
    except Exception:
        pass

    # 静态fallback: S&P 500代表性成分股（按行业分散，约200只覆盖主要行业）
    return [
        # 信息技术 (约50只)
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "CSCO", "ACN",
        "INTC", "IBM", "INTU", "TXN", "QCOM", "AMAT", "NOW", "PANW", "ADI", "LRCX",
        "MU", "KLAC", "SNPS", "CDNS", "MCHP", "MSI", "APH", "FTNT", "ROP", "TEL",
        "NXPI", "IT", "ANSS", "KEYS", "HPQ", "CDW", "ON", "FSLR", "GEN", "MPWR",
        "ZBRA", "TYL", "TRMB", "PTC", "EPAM", "JNPR", "AKAM", "SWKS", "FFIV", "WDC",
        # 通信服务 (约15只)
        "META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR",
        "EA", "TTWO", "WBD", "MTCH", "LYV",
        # 非必需消费品 (约30只)
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "ABNB",
        "CMG", "ORLY", "AZO", "ROST", "DHI", "LEN", "GM", "F", "MAR", "HLT",
        "YUM", "DARDEN", "DPZ", "POOL", "BBY", "ULTA", "EBAY", "ETSY", "APTV", "MGM",
        # 金融 (约40只)
        "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "BLK",
        "C", "AXP", "SCHW", "CB", "MMC", "PGR", "AON", "ICE", "CME", "MCO",
        "USB", "TFC", "PNC", "AIG", "MET", "PRU", "AFL", "TRV", "ALL", "CINF",
        "COF", "DFS", "SYF", "FIS", "FISV", "GPN", "PYPL", "AMP", "BEN", "IVZ",
        # 医疗保健 (约35只)
        "UNH", "LLY", "JNJ", "ABBV", "MRK", "TMO", "ABT", "PFE", "DHR", "BMY",
        "AMGN", "MDT", "ISRG", "ELV", "GILD", "CVS", "CI", "VRTX", "REGN", "SYK",
        "ZTS", "BDX", "BSX", "HCA", "EW", "IDXX", "DXCM", "IQV", "A", "MTD",
        "HOLX", "ALGN", "TECH", "CRL", "CTLT",
        # 工业 (约35只)
        "CAT", "RTX", "UNP", "HON", "DE", "BA", "GE", "LMT", "UPS", "ADP",
        "ETN", "ITW", "EMR", "FDX", "MMM", "CSX", "NSC", "WM", "RSG", "GD",
        "NOC", "TT", "PH", "ROK", "FAST", "VRSK", "CTAS", "PAYX", "CPRT", "ODFL",
        "IR", "WAB", "AME", "DOV", "SWK",
        # 必需消费品 (约20只)
        "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "KMB",
        "GIS", "K", "SYY", "HSY", "ADM", "STZ", "MKC", "CHD", "CAG", "SJM",
        # 能源 (约15只)
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "PXD", "OXY",
        "HES", "DVN", "FANG", "HAL", "BKR",
        # 公用事业 (约10只)
        "NEE", "SO", "DUK", "D", "AEP", "SRE", "EXC", "XEL", "WEC", "ED",
        # 房地产 (约10只)
        "PLD", "AMT", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB",
        # 原材料 (约10只)
        "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "DOW", "DD", "VMC",
    ]


def get_etf_pool(market_code: str) -> List[str]:
    """获取ETF池"""
    if market_code == "CN":
        return CN_ETF_POOL
    elif market_code == "US":
        return US_ETF_POOL
    else:
        raise ValueError(f"不支持的市场: {market_code}")


def is_etf(code: str, market_code: str) -> bool:
    """判断是否为ETF"""
    etf_pool = get_etf_pool(market_code)
    return code in etf_pool
