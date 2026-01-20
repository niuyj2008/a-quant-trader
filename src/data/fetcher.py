"""
A股量化交易系统 - 数据获取模块

提供统一的数据获取接口，支持多数据源切换
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Union
from loguru import logger

try:
    import akshare as ak
except ImportError:
    ak = None
    logger.warning("AKShare未安装，部分数据功能不可用")

try:
    import tushare as ts
except ImportError:
    ts = None
    logger.warning("Tushare未安装，部分数据功能不可用")


class DataFetcher:
    """统一数据获取器"""
    
    def __init__(self, source: str = "akshare", tushare_token: Optional[str] = None):
        """
        初始化数据获取器
        
        Args:
            source: 数据源 ("akshare" 或 "tushare")
            tushare_token: Tushare Pro token
        """
        self.source = source
        
        if source == "tushare" and tushare_token:
            if ts:
                ts.set_token(tushare_token)
                self.pro = ts.pro_api()
            else:
                raise ImportError("Tushare未安装，请运行: pip install tushare")
    
    def get_stock_list(self) -> pd.DataFrame:
        """
        获取A股股票列表
        
        Returns:
            DataFrame with columns: code, name, market, list_date
        """
        if self.source == "akshare":
            return self._get_stock_list_akshare()
        else:
            return self._get_stock_list_tushare()
    
    def _get_stock_list_akshare(self) -> pd.DataFrame:
        """使用AKShare获取股票列表"""
        if ak is None:
            raise ImportError("AKShare未安装")
        
        # 获取沪深A股实时行情，从中提取股票列表
        df = ak.stock_zh_a_spot_em()
        
        result = pd.DataFrame({
            'code': df['代码'],
            'name': df['名称'],
            'market': df['代码'].apply(lambda x: 'SH' if x.startswith('6') else 'SZ'),
        })
        
        return result
    
    def _get_stock_list_tushare(self) -> pd.DataFrame:
        """使用Tushare获取股票列表"""
        df = self.pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,market,list_date'
        )
        
        result = pd.DataFrame({
            'code': df['symbol'],
            'name': df['name'],
            'market': df['market'],
            'list_date': df['list_date']
        })
        
        return result
    
    def get_daily_data(
        self,
        code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = "qfq"
    ) -> pd.DataFrame:
        """
        获取日线数据
        
        Args:
            code: 股票代码 (如 "000001")
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            adjust: 复权类型 ("qfq"前复权, "hfq"后复权, ""不复权)
            
        Returns:
            DataFrame with OHLCV data
        """
        if self.source == "akshare":
            return self._get_daily_akshare(code, start_date, end_date, adjust)
        else:
            return self._get_daily_tushare(code, start_date, end_date, adjust)
    
    def _get_daily_akshare(
        self,
        code: str,
        start_date: Optional[str],
        end_date: Optional[str],
        adjust: str
    ) -> pd.DataFrame:
        """使用AKShare获取日线数据"""
        if ak is None:
            raise ImportError("AKShare未安装")
        
        # 转换复权类型
        adjust_map = {"qfq": "qfq", "hfq": "hfq", "": ""}
        ak_adjust = adjust_map.get(adjust, "qfq")
        
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date.replace("-", "") if start_date else "19900101",
            end_date=end_date.replace("-", "") if end_date else datetime.now().strftime("%Y%m%d"),
            adjust=ak_adjust
        )
        
        # 标准化列名
        result = pd.DataFrame({
            'date': pd.to_datetime(df['日期']),
            'open': df['开盘'],
            'high': df['最高'],
            'low': df['最低'],
            'close': df['收盘'],
            'volume': df['成交量'],
            'amount': df['成交额'],
            'turnover': df.get('换手率', 0),
        })
        
        result.set_index('date', inplace=True)
        return result
    
    def _get_daily_tushare(
        self,
        code: str,
        start_date: Optional[str],
        end_date: Optional[str],
        adjust: str
    ) -> pd.DataFrame:
        """使用Tushare获取日线数据"""
        # 确定交易所后缀
        suffix = ".SH" if code.startswith("6") else ".SZ"
        ts_code = code + suffix
        
        # 转换日期格式
        start = start_date.replace("-", "") if start_date else None
        end = end_date.replace("-", "") if end_date else None
        
        df = ts.pro_bar(
            ts_code=ts_code,
            start_date=start,
            end_date=end,
            adj=adjust if adjust else None
        )
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        result = pd.DataFrame({
            'date': pd.to_datetime(df['trade_date']),
            'open': df['open'],
            'high': df['high'],
            'low': df['low'],
            'close': df['close'],
            'volume': df['vol'],
            'amount': df['amount'],
        })
        
        result.set_index('date', inplace=True)
        result.sort_index(inplace=True)
        
        return result
    
    def get_realtime_quotes(self, codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取实时行情数据
        
        Args:
            codes: 股票代码列表，为None时获取全部A股
            
        Returns:
            DataFrame with realtime quotes
        """
        if self.source == "akshare":
            return self._get_realtime_akshare(codes)
        else:
            raise NotImplementedError("Tushare实时行情需要更高权限")
    
    def _get_realtime_akshare(self, codes: Optional[List[str]]) -> pd.DataFrame:
        """使用AKShare获取实时行情"""
        if ak is None:
            raise ImportError("AKShare未安装")
        
        df = ak.stock_zh_a_spot_em()
        
        result = pd.DataFrame({
            'code': df['代码'],
            'name': df['名称'],
            'price': df['最新价'],
            'change_pct': df['涨跌幅'],
            'change': df['涨跌额'],
            'volume': df['成交量'],
            'amount': df['成交额'],
            'open': df['开盘'],
            'high': df['最高'],
            'low': df['最低'],
            'prev_close': df['昨收'],
            'turnover': df['换手率'],
            'pe': df['市盈率-动态'],
            'pb': df['市净率'],
        })
        
        if codes:
            result = result[result['code'].isin(codes)]
        
        return result
    
    def get_index_data(
        self,
        code: str = "000300",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取指数数据
        
        Args:
            code: 指数代码 (如 "000300" 沪深300)
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame with index data
        """
        if self.source == "akshare":
            if ak is None:
                raise ImportError("AKShare未安装")
            
            df = ak.stock_zh_index_daily(symbol=f"sh{code}")
            
            result = pd.DataFrame({
                'date': pd.to_datetime(df['date']),
                'open': df['open'],
                'high': df['high'],
                'low': df['low'],
                'close': df['close'],
                'volume': df['volume'],
            })
            
            result.set_index('date', inplace=True)
            
            # 过滤日期
            if start_date:
                result = result[result.index >= start_date]
            if end_date:
                result = result[result.index <= end_date]
            
            return result
        else:
            raise NotImplementedError("Tushare指数数据接口待实现")


# 便捷函数
def get_stock_data(code: str, **kwargs) -> pd.DataFrame:
    """获取股票日线数据的快捷函数"""
    fetcher = DataFetcher()
    return fetcher.get_daily_data(code, **kwargs)


def get_all_stocks() -> pd.DataFrame:
    """获取全部A股列表的快捷函数"""
    fetcher = DataFetcher()
    return fetcher.get_stock_list()


if __name__ == "__main__":
    # 测试代码
    fetcher = DataFetcher(source="akshare")
    
    # 测试获取股票列表
    print("获取股票列表...")
    stocks = fetcher.get_stock_list()
    print(f"共 {len(stocks)} 只股票")
    print(stocks.head())
    
    # 测试获取日线数据
    print("\n获取平安银行日线数据...")
    df = fetcher.get_daily_data("000001", start_date="2024-01-01")
    print(df.tail())
