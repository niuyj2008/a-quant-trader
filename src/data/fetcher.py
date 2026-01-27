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

try:
    import yfinance as yf
except ImportError:
    yf = None
    logger.warning("yfinance未安装，美股数据功能不可用")


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
        adjust: str = "qfq",
        market: str = "CN"
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
        if market == "US":
            return self._get_daily_us(code, start_date, end_date)
            
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
        """使用AKShare获取日线数据，支持异常重试和备用源"""
        if ak is None:
            raise ImportError("AKShare未安装")
        
        # 转换复权类型
        adjust_map = {"qfq": "qfq", "hfq": "hfq", "": ""}
        ak_adjust = adjust_map.get(adjust, "qfq")
        
        try:
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
        except Exception as e:
            logger.warning(f"AKShare 获取 {code} 失败: {e}. 尝试使用 yfinance 备用源...")
            
            # 构造 yfinance 格式代码
            if code.startswith('6'):
                yf_code = f"{code}.SS"
            elif code.startswith(('0', '3')):
                yf_code = f"{code}.SZ"
            elif code.startswith(('8', '4')):
                yf_code = f"{code}.BJ"
            else:
                yf_code = code
                
            try:
                # yfinance 默认返回的就是前复权数据 (Close 是复权后的)
                fallback_df = self._get_daily_us(yf_code, start_date, end_date)
                if not fallback_df.empty:
                    logger.info(f"成功使用 yfinance 获取 {code} 数据")
                    # yfinance 数据列名已在 _get_daily_us 中标准化
                    # 添加缺失的 amount 和 turnover 为 0 保持兼容
                    fallback_df['amount'] = 0
                    fallback_df['turnover'] = 0
                    return fallback_df
            except Exception as fe:
                logger.error(f"所有数据源获取 {code} 均失败: {fe}")
            
            raise e
    
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
        if codes and len(codes) > 0 and codes[0].isalpha(): # 假设由字母组成的为美股代码
             return self._get_realtime_us(codes)

        if self.source == "akshare":
            try:
                return self._get_realtime_akshare(codes)
            except Exception as e:
                logger.warning(f"AKShare 实时行情获取失败: {e}. 尝试使用 yfinance 备用源...")
                # 尝试将代码转换为 yfinance 格式并调用美股/全球获取逻辑
                yf_codes = []
                for c in (codes or []):
                    if c.startswith('6'): yf_codes.append(f"{c}.SS")
                    elif c.startswith(('0', '3')): yf_codes.append(f"{c}.SZ")
                    else: yf_codes.append(c)
                
                try:
                    return self._get_realtime_us(yf_codes)
                except:
                    pass
                raise e
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
            
            try:
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
            except Exception as e:
                logger.warning(f"AKShare 指数数据获取失败: {e}. 尝试使用 yfinance 备用源...")
                yf_code = f"{code}.SS" if code.startswith('000') else code # 简单处理常用指数
                try:
                    return self._get_daily_us(yf_code, start_date, end_date)
                except:
                    pass
                raise e
        else:
            raise NotImplementedError("Tushare指数数据接口待实现")

    def _get_daily_us(
        self,
        code: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """使用yfinance获取美股日线数据"""
        if yf is None:
            raise ImportError("yfinance未安装")
            
        df = yf.download(code, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            return pd.DataFrame()
            
        if isinstance(df.columns, pd.MultiIndex):
            # yfinance returns MultiIndex (Price, Ticker) when fetching
            # We just want the Price level (0) if there's only one ticker or we treat it as single
            # But yfinance download(..., group_by='column') or default might vary.
            # Usually for single ticker it might be (Price, Ticker) or just Price.
            # If it is MultiIndex, let's try to flatten or take level 0
            try:
                # If the second level is the ticker, dropping it gives us the price types (Open, Close...)
                if len(df.columns.levels) > 1:
                     df.columns = df.columns.droplevel(1)
            except Exception:
                pass
                
        # 标准化列名
        # Now columns should be strings (hopefully)
        df.columns = [str(c).lower() for c in df.columns]
        df.index.name = 'date'
        
        # 确保只有需要的列
        required_cols = ['open', 'high', 'low', 'close', 'volume']

        # result = df[required_cols].copy() # yfinance might have 'adj close', etc.
        # Let's map explicitly to be safe
        result = pd.DataFrame({
            'open': df['open'],
            'high': df['high'],
            'low': df['low'],
            'close': df['close'], # yfinance close is usually split-adjusted but not dividend-adjusted unless auto_adjust=True (default False? check version). Let's use 'close'. 
            'volume': df['volume']
        })
        
        # Add timestamp if needed or just keep index
        # yfinance index is already datetime
        
        return result

    def _get_realtime_us(self, codes: Optional[List[str]]) -> pd.DataFrame:
        """使用yfinance获取美股实时行情"""
        if yf is None:
            raise ImportError("yfinance未安装")
            
        if not codes:
            return pd.DataFrame()
            
        data_list = []
        for code in codes:
            ticker = yf.Ticker(code)
            try:
                # fast_info is faster for realtime price
                info = ticker.fast_info
                
                # fast_info attributes: last_price, previous_close, open, day_high, day_low, ...
                # volume is not in fast_info directly always, sometimes need info
                
                price = info.last_price
                prev_close = info.previous_close
                if prev_close:
                    change = price - prev_close
                    change_pct = (change / prev_close) * 100
                else:
                    change = 0
                    change_pct = 0
                    
                data_list.append({
                    'code': code,
                    'name': code, # yfinance doesn't easily give short name in fast_info, would need ticker.info which is slow
                    'price': price,
                    'change_pct': change_pct,
                    'change': change,
                    'volume': info.last_volume if hasattr(info, 'last_volume') else 0, # fast_info might not have volume
                    'amount': 0, # not available easily
                    'open': info.open,
                    'high': info.day_high,
                    'low': info.day_low,
                    'prev_close': prev_close,
                })
            except Exception as e:
                logger.error(f"Error fetching {code}: {e}")
                
        return pd.DataFrame(data_list)


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
