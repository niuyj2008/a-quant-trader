"""
股票量化策略决策支持系统 - 数据获取模块（增强版）

支持多数据源切换、周线聚合、基本面/宏观/情绪数据获取
"""

import time
import functools
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union, Callable
from loguru import logger


def rate_limit(min_interval: float = 0.5):
    """限流装饰器：确保两次调用之间至少间隔 min_interval 秒"""
    def decorator(func: Callable) -> Callable:
        last_call_time = [0.0]

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_call_time[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_call_time[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

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

from src.data.data_cache import DataCache

# Phase 8: 导入数据验证器,确保所有数据真实可靠
try:
    from src.validation.data_validator import get_validator
    DATA_VALIDATION_ENABLED = True
except ImportError:
    DATA_VALIDATION_ENABLED = False
    logger.warning("DataValidator未安装,数据验证功能不可用")


class DataFetcher:
    """统一数据获取器（增强版）"""

    def __init__(self, source: str = "akshare", tushare_token: Optional[str] = None,
                 use_cache: bool = True, enable_validation: bool = True):
        """
        初始化数据获取器

        Args:
            source: 数据源 ("akshare" 或 "tushare")
            tushare_token: Tushare Pro token
            use_cache: 是否启用本地缓存
            enable_validation: 是否启用数据验证(Phase 8新增)
        """
        self.source = source
        self.use_cache = use_cache
        self.cache = DataCache() if use_cache else None

        # Phase 8: 数据验证器
        self.enable_validation = enable_validation and DATA_VALIDATION_ENABLED
        if self.enable_validation:
            self.validator = get_validator()
            logger.info("数据验证已启用,将确保所有数据真实可靠")

        if source == "tushare" and tushare_token:
            if ts:
                ts.set_token(tushare_token)
                self.pro = ts.pro_api()
            else:
                raise ImportError("Tushare未安装，请运行: pip install tushare")

    # ==================== 股票列表 ====================

    def get_stock_list(self, market: str = "CN") -> pd.DataFrame:
        """获取股票列表"""
        if market == "US":
            return self._get_us_stock_list()
        if self.source == "akshare":
            return self._get_stock_list_akshare()
        else:
            return self._get_stock_list_tushare()

    def _get_stock_list_akshare(self) -> pd.DataFrame:
        """使用AKShare获取A股股票列表"""
        if ak is None:
            raise ImportError("AKShare未安装")
        df = ak.stock_zh_a_spot_em()
        result = pd.DataFrame({
            'code': df['代码'],
            'name': df['名称'],
            'market': df['代码'].apply(lambda x: 'SH' if x.startswith('6') else 'SZ'),
        })
        return result

    def _get_stock_list_tushare(self) -> pd.DataFrame:
        """使用Tushare获取股票列表"""
        df = self.pro.stock_basic(exchange='', list_status='L',
                                  fields='ts_code,symbol,name,market,list_date')
        result = pd.DataFrame({
            'code': df['symbol'], 'name': df['name'],
            'market': df['market'], 'list_date': df['list_date']
        })
        return result

    def _get_us_stock_list(self) -> pd.DataFrame:
        """美股精选列表"""
        from src.data.market import get_stock_pool
        codes = get_stock_pool("US")
        return pd.DataFrame({'code': codes, 'name': codes, 'market': 'US'})

    # ==================== 日线数据 ====================

    def get_daily_data(self, code: str, start_date: Optional[str] = None,
                       end_date: Optional[str] = None, adjust: str = "qfq",
                       market: str = "CN") -> pd.DataFrame:
        """获取日线数据（带缓存 + 增量更新）"""
        today_str = datetime.now().strftime('%Y-%m-%d')
        req_end = end_date or today_str

        # 增量更新逻辑：检查缓存是否已覆盖所需范围
        if self.use_cache and self.cache:
            cache_range = self.cache.get_cache_range(code, market)
            if cache_range:
                cache_start, cache_end = cache_range
                # 如果缓存已覆盖请求范围（或距今≤1天），直接返回
                if cache_end >= req_end or (
                    not end_date and cache_end >= (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
                ):
                    cached = self.cache.load_daily(code, start_date, end_date, market)
                    if not cached.empty:
                        logger.debug(f"缓存命中: {market}/{code} {len(cached)}条")
                        if self.enable_validation:
                            validation_result = self.validator.validate_price_data(cached, code, market)
                            if not validation_result['is_valid']:
                                logger.warning(f"缓存数据验证失败: {validation_result['issues']}, 重新获取")
                                self.cache.clear_daily(code, market)
                            else:
                                return cached
                        else:
                            return cached

                # 缓存存在但不够新：增量拉取缺失部分
                elif cache_end < req_end:
                    incr_start = (datetime.strptime(cache_end, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                    logger.debug(f"增量更新: {market}/{code} 从 {incr_start} 到 {req_end}")
                    try:
                        if market == "US":
                            incr_df = self._get_daily_us(code, incr_start, req_end)
                        elif self.source == "akshare":
                            incr_df = self._get_daily_akshare(code, incr_start, req_end, adjust)
                        else:
                            incr_df = self._get_daily_tushare(code, incr_start, req_end, adjust)

                        if not incr_df.empty:
                            self.cache.save_daily(code, incr_df, market)
                        # 返回完整范围
                        return self.cache.load_daily(code, start_date, end_date, market)
                    except Exception as e:
                        logger.warning(f"增量更新失败: {e}, 使用已有缓存")
                        cached = self.cache.load_daily(code, start_date, end_date, market)
                        if not cached.empty:
                            return cached

            # 缓存中无此股票数据，但有缓存框架：尝试直接加载
            else:
                cached = self.cache.load_daily(code, start_date, end_date, market)
                if not cached.empty:
                    if self.enable_validation:
                        validation_result = self.validator.validate_price_data(cached, code, market)
                        if not validation_result['is_valid']:
                            self.cache.clear_daily(code, market)
                        else:
                            return cached
                    else:
                        return cached

        # 从数据源全量获取
        if market == "US":
            df = self._get_daily_us(code, start_date, end_date)
        elif self.source == "akshare":
            df = self._get_daily_akshare(code, start_date, end_date, adjust)
        else:
            df = self._get_daily_tushare(code, start_date, end_date, adjust)

        # Phase 8: 验证获取的数据
        if self.enable_validation and not df.empty:
            df.attrs['source'] = self.source if market == 'CN' else 'yfinance'
            df.attrs['fetch_time'] = datetime.now().isoformat()

            validation_result = self.validator.validate_price_data(df, code, market)
            if not validation_result['is_valid']:
                logger.error(f"数据验证失败 [{code}]: {validation_result['issues']}")
                if any('High < Low' in issue or '负数或0价格' in issue
                       for issue in validation_result['issues']):
                    logger.error("数据包含严重错误,拒绝使用")
                    return pd.DataFrame()

        # 写入缓存
        if self.use_cache and self.cache and not df.empty:
            self.cache.save_daily(code, df, market)

        return df

    def _get_daily_akshare(self, code: str, start_date: Optional[str],
                           end_date: Optional[str], adjust: str) -> pd.DataFrame:
        """使用AKShare获取日线数据，支持异常重试和备用源"""
        if ak is None:
            raise ImportError("AKShare未安装")
        adjust_map = {"qfq": "qfq", "hfq": "hfq", "": ""}
        ak_adjust = adjust_map.get(adjust, "qfq")

        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start_date.replace("-", "") if start_date else "19900101",
                end_date=end_date.replace("-", "") if end_date else datetime.now().strftime("%Y%m%d"),
                adjust=ak_adjust
            )
            result = pd.DataFrame({
                'date': pd.to_datetime(df['日期']),
                'open': df['开盘'], 'high': df['最高'],
                'low': df['最低'], 'close': df['收盘'],
                'volume': df['成交量'], 'amount': df['成交额'],
                'turnover': df.get('换手率', 0),
            })
            result.set_index('date', inplace=True)
            return result
        except Exception as e:
            logger.warning(f"AKShare 获取 {code} 失败: {e}. 尝试yfinance备用源...")
            if code.startswith('6'):
                yf_code = f"{code}.SS"
            elif code.startswith(('0', '3')):
                yf_code = f"{code}.SZ"
            else:
                yf_code = code
            try:
                fallback_df = self._get_daily_us(yf_code, start_date, end_date)
                if not fallback_df.empty:
                    fallback_df['amount'] = 0
                    fallback_df['turnover'] = 0
                    return fallback_df
            except Exception as fe:
                logger.error(f"所有数据源获取 {code} 均失败: {fe}")
            raise e

    def _get_daily_tushare(self, code: str, start_date: Optional[str],
                           end_date: Optional[str], adjust: str) -> pd.DataFrame:
        """使用Tushare获取日线数据"""
        suffix = ".SH" if code.startswith("6") else ".SZ"
        ts_code = code + suffix
        start = start_date.replace("-", "") if start_date else None
        end = end_date.replace("-", "") if end_date else None
        df = ts.pro_bar(ts_code=ts_code, start_date=start, end_date=end,
                        adj=adjust if adjust else None)
        if df is None or df.empty:
            return pd.DataFrame()
        result = pd.DataFrame({
            'date': pd.to_datetime(df['trade_date']),
            'open': df['open'], 'high': df['high'],
            'low': df['low'], 'close': df['close'],
            'volume': df['vol'], 'amount': df['amount'],
        })
        result.set_index('date', inplace=True)
        result.sort_index(inplace=True)
        return result

    @rate_limit(min_interval=0.3)
    def _get_daily_us(self, code: str, start_date: Optional[str],
                      end_date: Optional[str]) -> pd.DataFrame:
        """使用yfinance获取美股日线数据（带限流）"""
        if yf is None:
            raise ImportError("yfinance未安装")
        df = yf.download(code, start=start_date, end=end_date, progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            try:
                if len(df.columns.levels) > 1:
                    df.columns = df.columns.droplevel(1)
            except Exception:
                pass
        df.columns = [str(c).lower() for c in df.columns]
        df.index.name = 'date'
        result = pd.DataFrame({
            'open': df['open'], 'high': df['high'],
            'low': df['low'], 'close': df['close'],
            'volume': df['volume']
        })
        return result

    # ==================== 周线数据 ====================

    def get_weekly_data(self, code: str, start_date: Optional[str] = None,
                        end_date: Optional[str] = None, market: str = "CN") -> pd.DataFrame:
        """
        获取周线数据（从日线聚合）

        Returns:
            DataFrame with weekly OHLCV, indexed by week ending date (Friday)
        """
        daily = self.get_daily_data(code, start_date, end_date, market=market)
        if daily.empty:
            return pd.DataFrame()
        return self.aggregate_to_weekly(daily)

    @staticmethod
    def aggregate_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
        """将日线数据聚合为周线"""
        weekly = daily_df.resample('W-FRI').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        })
        # 可选列
        if 'amount' in daily_df.columns:
            weekly['amount'] = daily_df['amount'].resample('W-FRI').sum()
        if 'turnover' in daily_df.columns:
            weekly['turnover'] = daily_df['turnover'].resample('W-FRI').mean()
        weekly.dropna(subset=['open'], inplace=True)
        return weekly

    # ==================== 10年长期数据 ====================

    def get_long_history(self, code: str, years: int = 10,
                         market: str = "CN") -> pd.DataFrame:
        """获取长期历史数据（默认10年）"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
        return self.get_daily_data(code, start_date, end_date, market=market)

    # ==================== 基本面数据 ====================

    def get_financial_data(self, code: str, market: str = "CN") -> Dict:
        """
        获取股票基本面数据（带缓存）

        Returns:
            Dict with pe, pb, roe, revenue_growth, gross_margin, etc.
        """
        # 尝试从缓存加载（7天内有效）
        if self.use_cache and self.cache:
            cached = self.cache.load_financial(code, market)
            if not cached.empty:
                latest = cached.iloc[-1]
                cache_date = str(latest.get('date', ''))
                if cache_date and (datetime.now() - datetime.strptime(cache_date, '%Y-%m-%d')).days < 7:
                    result = {col: latest[col] for col in cached.columns
                              if col not in ('market', 'code', 'date') and pd.notna(latest[col])}
                    if result:
                        logger.debug(f"财务数据缓存命中: {market}/{code}")
                        return result

        if market == "US":
            data = self._get_financial_us(code)
        else:
            data = self._get_financial_cn(code)

        # 写入缓存
        if self.use_cache and self.cache and data:
            self.cache.save_financial(code, data, datetime.now().strftime('%Y-%m-%d'), market)

        return data

    def _get_financial_cn(self, code: str) -> Dict:
        """获取A股基本面数据"""
        result = {}
        if ak is None:
            return result
        try:
            # 实时行情中包含PE/PB
            df = ak.stock_zh_a_spot_em()
            row = df[df['代码'] == code]
            if not row.empty:
                row = row.iloc[0]
                result['pe'] = float(row.get('市盈率-动态', 0)) if pd.notna(row.get('市盈率-动态')) else None
                result['pb'] = float(row.get('市净率', 0)) if pd.notna(row.get('市净率')) else None
                result['market_cap'] = float(row.get('总市值', 0)) if pd.notna(row.get('总市值')) else None
                result['turnover_rate'] = float(row.get('换手率', 0)) if pd.notna(row.get('换手率')) else None
        except Exception as e:
            logger.warning(f"获取A股基本面数据失败: {e}")

        try:
            # 财务指标
            fin_df = ak.stock_financial_abstract_ths(symbol=code, indicator="按年度")
            if fin_df is not None and not fin_df.empty:
                latest = fin_df.iloc[0]
                result['roe'] = self._safe_float(latest, '净资产收益率')
                result['gross_margin'] = self._safe_float(latest, '销售毛利率')
                result['revenue_growth'] = self._safe_float(latest, '营业总收入同比增长率')
                result['net_profit_growth'] = self._safe_float(latest, '净利润同比增长率')
        except Exception as e:
            logger.debug(f"获取财务摘要失败（可忽略）: {e}")

        return result

    def _get_financial_us(self, code: str) -> Dict:
        """获取美股基本面数据"""
        result = {}
        if yf is None:
            return result
        try:
            ticker = yf.Ticker(code)
            info = ticker.info
            result['pe'] = info.get('trailingPE')
            result['pb'] = info.get('priceToBook')
            result['roe'] = info.get('returnOnEquity')
            result['revenue_growth'] = info.get('revenueGrowth')
            result['gross_margin'] = info.get('grossMargins')
            result['market_cap'] = info.get('marketCap')
            result['debt_ratio'] = info.get('debtToEquity')
        except Exception as e:
            logger.warning(f"获取美股基本面数据失败: {e}")
        return result

    # ==================== 行研报告数据 (Phase 10.1) ====================

    def get_research_data(self, code: str, market: str = "CN") -> Dict:
        """获取行研报告结构化数据（带缓存）

        Returns:
            {
                'recommendations': DataFrame,   # 分析师评级历史
                'price_targets': Dict,           # 一致目标价
                'earnings_estimate': DataFrame,  # EPS一致预期
                'revenue_estimate': DataFrame,   # 营收一致预期
            }
        缓存有效期: 24小时
        """
        # 尝试从缓存加载（24小时内有效）
        if self.use_cache and self.cache:
            cached = self.cache.load_research(code, market)
            if cached:
                logger.debug(f"行研数据缓存命中: {market}/{code}")
                return cached

        if market == "US":
            data = self._get_research_us(code)
        else:
            data = self._get_research_cn(code)

        # 写入缓存
        if self.use_cache and self.cache and data:
            self.cache.save_research(code, data, market)

        return data

    @rate_limit(min_interval=0.5)
    def _get_research_us(self, code: str) -> Dict:
        """美股行研数据 - 基于 yfinance 内置 API

        一次创建 Ticker 对象，批量获取所有行研字段。
        yfinance 数据来源于 Yahoo Finance 汇总的华尔街主流投行公开研报结论。
        """
        result = {}
        if yf is None:
            return result

        try:
            ticker = yf.Ticker(code)

            # 1. 分析师评级历史
            #    DataFrame: [date, firm, to_grade, from_grade, action]
            #    action: "upgrade"/"downgrade"/"initiated"/"reiterated"
            try:
                recs = ticker.recommendations
                if recs is not None and isinstance(recs, pd.DataFrame) and not recs.empty:
                    result['recommendations'] = recs
            except Exception as e:
                logger.debug(f"获取 {code} 评级历史失败: {e}")

            # 2. 一致目标价
            #    {current, low, high, mean, median}
            try:
                targets = ticker.analyst_price_targets
                if targets is not None:
                    # 转为标准 dict
                    if isinstance(targets, dict):
                        result['price_targets'] = targets
                    elif hasattr(targets, 'to_dict'):
                        result['price_targets'] = targets.to_dict()
                    else:
                        result['price_targets'] = {
                            'current': getattr(targets, 'current', None),
                            'low': getattr(targets, 'low', None),
                            'high': getattr(targets, 'high', None),
                            'mean': getattr(targets, 'mean', None),
                            'median': getattr(targets, 'median', None),
                        }
            except Exception as e:
                logger.debug(f"获取 {code} 目标价失败: {e}")

            # 3. EPS盈利预测
            try:
                earnings = ticker.earnings_estimate
                if earnings is not None and isinstance(earnings, pd.DataFrame) and not earnings.empty:
                    result['earnings_estimate'] = earnings
            except Exception as e:
                logger.debug(f"获取 {code} 盈利预测失败: {e}")

            # 4. 营收预测
            try:
                revenue = ticker.revenue_estimate
                if revenue is not None and isinstance(revenue, pd.DataFrame) and not revenue.empty:
                    result['revenue_estimate'] = revenue
            except Exception as e:
                logger.debug(f"获取 {code} 营收预测失败: {e}")

        except Exception as e:
            logger.warning(f"获取 {code} 行研数据失败: {e}")

        return result

    @rate_limit(min_interval=1.0)
    def _get_research_cn(self, code: str) -> Dict:
        """A股行研数据 - 基于 AKShare 免费接口

        数据来源:
          - ak.stock_profit_forecast_em(): 盈利预测(机构预测EPS/营收)
          - ak.stock_comment_detail_zlkp_jgcyd_em(): 机构参与度
          - 评级数据通过东方财富接口获取

        Returns:
            {
                'recommendations': DataFrame,  # 模拟评级(基于机构预测方向)
                'price_targets': Dict,         # 一致目标价(如有)
                'earnings_estimate': DataFrame, # 盈利预测
            }
        """
        if ak is None:
            return {}

        result = {}
        # 标准化代码: 去掉后缀 (如 000001.SZ → 000001)
        pure_code = code.split('.')[0] if '.' in code else code

        # 1. 盈利预测数据（机构一致预期）
        try:
            df_forecast = ak.stock_profit_forecast_em(symbol=pure_code)
            if df_forecast is not None and not df_forecast.empty:
                result['earnings_estimate'] = df_forecast
                logger.debug(f"A股 {code} 盈利预测数据获取成功: {len(df_forecast)} 条")
        except Exception as e:
            logger.debug(f"A股 {code} 盈利预测获取失败: {e}")

        # 2. 个股评级汇总（东方财富）
        try:
            df_comment = ak.stock_comment_detail_zlkp_jgcyd_em(symbol=pure_code)
            if df_comment is not None and not df_comment.empty:
                result['institutional_activity'] = df_comment
        except Exception as e:
            logger.debug(f"A股 {code} 机构参与度获取失败: {e}")

        # 3. 将盈利预测转换为类似评级的结构（模拟 recommendations）
        # 通过预测EPS变化方向来推断一致预期方向
        if 'earnings_estimate' in result:
            try:
                recs = self._cn_forecast_to_recommendations(result['earnings_estimate'], pure_code)
                if recs is not None and not recs.empty:
                    result['recommendations'] = recs
            except Exception as e:
                logger.debug(f"A股 {code} 评级转换失败: {e}")

        return result

    def _cn_forecast_to_recommendations(self, forecast_df: pd.DataFrame,
                                         code: str) -> Optional[pd.DataFrame]:
        """将A股盈利预测数据转换为与美股 recommendations 兼容的格式

        AKShare stock_profit_forecast_em 返回列通常包含:
        研究机构, 研究员, 预测年度, 预测指标, 预测值, 发布日期 等

        转换策略:
        - 有盈利增长预测 → "Buy"
        - 盈利持平预测 → "Hold"
        - 盈利下降预测 → "Underperform"
        """
        if forecast_df is None or forecast_df.empty:
            return None

        records = []
        now = datetime.now()

        for _, row in forecast_df.iterrows():
            try:
                # 识别列名（AKShare返回可能有不同列名格式）
                firm = str(row.get('研究机构', row.get('机构名称', '')))
                if not firm:
                    continue

                # 尝试获取发布日期
                date_val = row.get('发布日期', row.get('日期', now))
                try:
                    date = pd.to_datetime(date_val)
                except Exception:
                    date = now

                # 推断评级方向：基于预测指标
                # 不同AKShare版本列名可能不同，做容错处理
                grade = 'Hold'  # 默认中性
                action = 'reiterated'

                # 尝试从"评级"列直接获取
                rating_str = str(row.get('最新评级', row.get('评级', ''))).strip()
                if rating_str:
                    if rating_str in ('买入', '强烈推荐', '推荐', '增持'):
                        grade = 'Buy'
                    elif rating_str in ('中性', '持有', '观望'):
                        grade = 'Hold'
                    elif rating_str in ('减持', '卖出', '回避'):
                        grade = 'Underperform'

                records.append({
                    'Date': date,
                    'firm': firm,
                    'to_grade': grade,
                    'from_grade': '',
                    'action': action,
                })
            except Exception:
                continue

        if not records:
            return None

        return pd.DataFrame(records)

    # ==================== 宏观经济数据 ====================

    def get_macro_data(self) -> Dict[str, pd.Series]:
        """
        获取中国宏观经济指标
        
        Returns:
            Dict of indicator_name -> pd.Series(date -> value)
        """
        result = {}
        if ak is None:
            return result

        # GDP
        try:
            df = ak.macro_china_gdp()
            if df is not None and not df.empty:
                # 过滤并解析日期: "2023年第4季度" -> 2023-12-31
                # 使用列名 '季度' 更安全
                date_col = '季度' if '季度' in df.columns else df.columns[0]
                df['date'] = df[date_col].apply(self._parse_chinese_quarter)
                df.dropna(subset=['date'], inplace=True)
                # 数据通常在第2列 (国内生产总值-绝对值) 或 第3列 (同比增长)
                # 这里取第2列作为 GDP 数值
                s = pd.Series(df.iloc[:, 1].values, index=df['date'])
                result['gdp'] = s.sort_index()
        except Exception as e:
            logger.debug(f"GDP数据获取失败: {e}")

        # CPI
        try:
            df = ak.macro_china_cpi_monthly()
            if df is not None and not df.empty:
                # CPI 日期在 '日期' 列 (第2列)
                date_col = '日期' if '日期' in df.columns else df.columns[1]
                df['date'] = df[date_col].apply(self._parse_chinese_month)
                df.dropna(subset=['date'], inplace=True)
                # 数据在 '今值' (current value)
                val_col = '今值' if '今值' in df.columns else df.columns[2]
                s = pd.Series(df[val_col].values, index=df['date'])
                result['cpi'] = s.sort_index()
        except Exception as e:
            logger.debug(f"CPI数据获取失败: {e}")

        # PMI
        try:
            df = ak.macro_china_pmi()
            if df is not None and not df.empty:
                date_col = '月份' if '月份' in df.columns else df.columns[0]
                df['date'] = df[date_col].apply(self._parse_chinese_month)
                df.dropna(subset=['date'], inplace=True)
                # 制造业数据 usually column 1
                s = pd.Series(df.iloc[:, 1].values, index=df['date'])
                result['pmi'] = s.sort_index()
        except Exception as e:
            logger.debug(f"PMI数据获取失败: {e}")

        # M2
        try:
            df = ak.macro_china_money_supply()
            if df is not None and not df.empty:
                date_col = '月份' if '月份' in df.columns else df.columns[0]
                df['date'] = df[date_col].apply(self._parse_chinese_month)
                df.dropna(subset=['date'], inplace=True)
                for col in df.columns:
                    if 'M2' in str(col) and '同比增长' in str(col):
                         # 优先取同比增长
                         s = pd.Series(df[col].values, index=df['date'])
                         result['m2'] = s.sort_index()
                         break
                if 'm2' not in result:
                    # Fallback to any M2 column
                     for col in df.columns:
                        if 'M2' in str(col):
                            s = pd.Series(df[col].values, index=df['date'])
                            result['m2'] = s.sort_index()
                            break
        except Exception as e:
            logger.debug(f"M2数据获取失败: {e}")

        return result

    # ==================== 市场情绪数据 ====================

    def get_sentiment_data(self, market: str = "CN") -> Dict:
        """获取市场情绪指标"""
        result = {}
        if market == "US":
            return self._get_sentiment_us()
        return self._get_sentiment_cn()

    def _get_sentiment_cn(self) -> Dict:
        """A股情绪数据"""
        result = {}
        if ak is None:
            return result

        # 融资融券余额 (沪深总和)
        try:
            # 分别获取沪市和深市数据并合并
            # 注意: 若AKShare接口返回异常或数据为空，需做容错
            try:
                sh_margin = ak.stock_margin_sse(start_date="20240101")
            except Exception:
                sh_margin = pd.DataFrame()
            
            try:
                sz_margin = ak.stock_margin_szse(start_date="20240101")
            except Exception:
                sz_margin = pd.DataFrame()
            
            if not sh_margin.empty and not sz_margin.empty:
                # 统一列名并按日期索引
                sh_margin['date'] = pd.to_datetime(sh_margin.iloc[:, 0], errors='coerce')
                sz_margin['date'] = pd.to_datetime(sz_margin.iloc[:, 0], errors='coerce')
                
                sh_margin.set_index('date', inplace=True)
                sz_margin.set_index('date', inplace=True)
                
                # 合并
                common_dates = sh_margin.index.intersection(sz_margin.index)
                if not common_dates.empty:
                    # 尝试按位置获取融资买入额 (通常是第2列)
                    sh_buy = sh_margin.iloc[:, 2] if len(sh_margin.columns) > 2 else sh_margin.iloc[:, 1]
                    sz_buy = sz_margin.iloc[:, 1] if len(sz_margin.columns) > 1 else sz_margin.iloc[:, 0]
                    
                    total_buy = sh_buy.reindex(common_dates).fillna(0) + sz_buy.reindex(common_dates).fillna(0)
                    
                    # 构造 DataFrame
                    df = pd.DataFrame({'融资买入额': total_buy})
                    result['margin_balance'] = df
        except Exception as e:
            logger.debug(f"融资融券数据获取失败: {e}")

        # 北向资金
        try:
            df = ak.stock_hsgt_hist_em(symbol="北向资金")
            if df is not None and not df.empty:
                # 列: 日期, 当日成交净买入额 -> 改为 '当日成交净买额'
                df['date'] = pd.to_datetime(df['日期'])
                df.set_index('date', inplace=True)
                
                # 映射列名
                rename_map = {
                    '当日成交净买额': 'north_money',
                    '当日成交净买入额': 'north_money'  # 兼容旧名
                }
                df.rename(columns=rename_map, inplace=True)
                
                if 'north_money' in df.columns:
                    # 过滤掉 NaN 值 (某些接口返回未来日期的空行)
                    df.dropna(subset=['north_money'], inplace=True)
                    if not df.empty:
                        result['northbound_flow'] = df
        except Exception as e:
            logger.debug(f"北向资金数据获取失败: {e}")

        return result

    def _get_sentiment_us(self) -> Dict:
        """美股情绪数据 - VIX & 10Y Yield"""
        result = {}
        try:
            if yf:
                # VIX
                try:
                    vix = yf.download("^VIX", period="1y", progress=False)
                    if not vix.empty:
                        result['vix'] = vix
                    else:
                        result['debug_vix_empty'] = "True"
                except Exception as e:
                    result['debug_vix_error'] = str(e)
                
                # 10Y Treasury Yield
                try:
                    tnx = yf.download("^TNX", period="1y", progress=False)
                    if not tnx.empty:
                        result['us_yield'] = tnx
                    else:
                        result['debug_tnx_empty'] = "True"
                except Exception as e:
                    result['debug_tnx_error'] = str(e)
            else:
                result['debug_yf_missing'] = "True"
        except Exception as e:
            logger.debug(f"美股情绪/宏观数据获取失败: {e}")
            result['debug_global_error'] = str(e)
        return result
    
    # ==================== 日期解析辅助 ====================

    @staticmethod
    def _parse_chinese_quarter(date_str: str):
        """解析 '2025年第1季度' -> datetime"""
        try:
            if not isinstance(date_str, str): return None
            import re
            m = re.match(r'(\d{4})年第(\d)季度', date_str)
            if m:
                year, q = int(m.group(1)), int(m.group(2))
                # 设置为季度末
                month = q * 3
                # 简单处理：3->3.31, 6->6.30, 9->9.30, 12->12.31
                last_days = {3: 31, 6: 30, 9: 30, 12: 31}
                return pd.Timestamp(year=year, month=month, day=last_days.get(month, 30))
            # 尝试处理 '2025年1-4季度' (年度数据) -> 2025-12-31
            m_year = re.search(r'(\d{4})年', date_str) 
            if m_year:
                 return pd.Timestamp(year=int(m_year.group(1)), month=12, day=31)
            return None
        except:
            return None

    @staticmethod
    def _parse_chinese_month(date_str: str):
        """解析 '2025年1月份' -> datetime"""
        try:
            if not isinstance(date_str, str): return None
            import re
            m = re.match(r'(\d{4})年(\d{1,2})月份', date_str)
            if m:
                return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)
            return None
        except:
            return None

    # ==================== 实时行情 ====================

    def get_realtime_quotes(self, codes: Optional[List[str]] = None,
                            market: str = "CN") -> pd.DataFrame:
        """获取实时行情数据"""
        if market == "US" or (codes and len(codes) > 0 and codes[0].isalpha()):
            return self._get_realtime_us(codes)
        if self.source == "akshare":
            try:
                return self._get_realtime_akshare(codes)
            except Exception as e:
                logger.warning(f"AKShare 实时行情获取失败: {e}")
                raise e
        else:
            raise NotImplementedError("Tushare实时行情需要更高权限")

    def _get_realtime_akshare(self, codes: Optional[List[str]]) -> pd.DataFrame:
        """使用AKShare获取A股实时行情"""
        if ak is None:
            raise ImportError("AKShare未安装")
        df = ak.stock_zh_a_spot_em()
        result = pd.DataFrame({
            'code': df['代码'], 'name': df['名称'],
            'price': df['最新价'], 'change_pct': df['涨跌幅'],
            'change': df['涨跌额'], 'volume': df['成交量'],
            'amount': df['成交额'], 'open': df['开盘'],
            'high': df['最高'], 'low': df['最低'],
            'prev_close': df['昨收'], 'turnover': df['换手率'],
            'pe': df['市盈率-动态'], 'pb': df['市净率'],
        })
        if codes:
            result = result[result['code'].isin(codes)]
        return result

    def _get_realtime_us(self, codes: Optional[List[str]]) -> pd.DataFrame:
        """使用yfinance获取美股实时行情"""
        if yf is None:
            raise ImportError("yfinance未安装")
        if not codes:
            return pd.DataFrame()
        data_list = []
        for code in codes:
            try:
                ticker = yf.Ticker(code)
                info = ticker.fast_info
                price = info.last_price
                prev_close = info.previous_close
                change = price - prev_close if prev_close else 0
                change_pct = (change / prev_close) * 100 if prev_close else 0
                data_list.append({
                    'code': code, 'name': code,
                    'price': price, 'change_pct': change_pct,
                    'change': change,
                    'volume': info.last_volume if hasattr(info, 'last_volume') else 0,
                    'amount': 0, 'open': info.open,
                    'high': info.day_high, 'low': info.day_low,
                    'prev_close': prev_close,
                })
            except Exception as e:
                logger.error(f"Error fetching {code}: {e}")
        return pd.DataFrame(data_list)

    # ==================== 指数数据 ====================

    def get_index_data(self, code: str = "000300", start_date: Optional[str] = None,
                       end_date: Optional[str] = None, market: str = "CN") -> pd.DataFrame:
        """获取指数数据"""
        if market == "US":
            return self._get_daily_us(code, start_date, end_date)

        if self.source == "akshare":
            if ak is None:
                raise ImportError("AKShare未安装")
            try:
                df = ak.stock_zh_index_daily(symbol=f"sh{code}")
                result = pd.DataFrame({
                    'date': pd.to_datetime(df['date']),
                    'open': df['open'], 'high': df['high'],
                    'low': df['low'], 'close': df['close'],
                    'volume': df['volume'],
                })
                result.set_index('date', inplace=True)
                if start_date:
                    result = result[result.index >= start_date]
                if end_date:
                    result = result[result.index <= end_date]
                return result
            except Exception as e:
                logger.warning(f"指数数据获取失败: {e}")
                raise e
        else:
            raise NotImplementedError("Tushare指数数据接口待实现")

    # ==================== 批量下载 ====================

    def batch_download(self, stock_list: List[str], years: int = 10,
                       market: str = "US", include_financial: bool = True,
                       progress_callback=None) -> Dict[str, Dict]:
        """批量下载历史数据（用于训练数据集构建）

        智能跳过已有充足缓存的股票，仅下载缺失数据。
        判断标准：缓存数据≥ years*200 行且最新数据距今≤7天 → 直接使用缓存。

        Args:
            stock_list: 股票代码列表
            years: 历史年数
            market: 市场代码
            include_financial: 是否同时获取财务数据
            progress_callback: 进度回调函数 callback(current, total, code, status)

        Returns:
            Dict[code -> {"daily": DataFrame, "financial": Dict, "status": str}]
        """
        results = {}
        total = len(stock_list)
        success = 0
        failed = 0
        cached_hit = 0
        min_rows = years * 200  # 每年约252交易日，200为保守阈值
        fresh_threshold = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        logger.info(f"开始批量下载: {total}只股票, {years}年历史, 市场={market}")

        for i, code in enumerate(stock_list):
            status = "success"
            try:
                # 先检查缓存是否已有充足数据
                cache_used = False
                if self.use_cache and self.cache:
                    cache_range = self.cache.get_cache_range(code, market)
                    if cache_range:
                        cache_start, cache_end = cache_range
                        # 缓存数据够新（7天内）则直接加载
                        if cache_end >= fresh_threshold:
                            cached_df = self.cache.load_daily(code, None, None, market)
                            if not cached_df.empty and len(cached_df) >= min_rows:
                                df = cached_df
                                status = "success"
                                success += 1
                                cached_hit += 1
                                cache_used = True
                                logger.debug(f"[{i+1}/{total}] {code} 缓存命中 ({len(df)}行)")

                if not cache_used:
                    # 缓存不足或不够新，执行下载/增量更新（带超时保护）
                    import signal as _signal
                    import threading

                    df = pd.DataFrame()
                    _download_timeout = 30  # 每只股票最多30秒

                    def _do_download():
                        nonlocal df
                        df = self.get_long_history(code, years=years, market=market)

                    t = threading.Thread(target=_do_download)
                    t.start()
                    t.join(timeout=_download_timeout)
                    if t.is_alive():
                        logger.warning(f"[{i+1}/{total}] {code} 下载超时({_download_timeout}s)，跳过")
                        status = "skipped"
                        failed += 1
                        # 尝试使用已有缓存（即使不完整）
                        if self.use_cache and self.cache:
                            cached_df = self.cache.load_daily(code, None, None, market)
                            if not cached_df.empty:
                                df = cached_df
                                status = "success"
                                success += 1
                                failed -= 1
                    elif df.empty:
                        status = "skipped"
                        failed += 1
                    else:
                        status = "success"
                        success += 1

                result_entry = {"daily": df, "financial": {}, "status": status, "rows": len(df)}

                # 财务数据
                if include_financial and not df.empty:
                    try:
                        fin = self.get_financial_data(code, market=market)
                        result_entry["financial"] = fin
                    except Exception:
                        pass

                results[code] = result_entry

            except Exception as e:
                status = "failed"
                results[code] = {"daily": pd.DataFrame(), "financial": {}, "status": status, "rows": 0}
                failed += 1
                logger.warning(f"[{i+1}/{total}] {code} 失败: {e}")

            if progress_callback:
                progress_callback(i + 1, total, code, status)

            if (i + 1) % 50 == 0:
                logger.info(f"进度: {i+1}/{total}, 成功={success}, 缓存命中={cached_hit}, 失败={failed}")

        logger.info(f"批量下载完成: 成功={success}, 缓存命中={cached_hit}, 失败={failed}, 总计={total}")
        return results

    # ==================== 工具方法 ====================

    @staticmethod
    def _safe_float(row, col_name):
        """安全提取浮点值"""
        try:
            val = row.get(col_name)
            if val is not None and pd.notna(val):
                return float(val)
        except (ValueError, TypeError):
            pass
        return None


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
    fetcher = DataFetcher(source="akshare")
    print("获取股票列表...")
    stocks = fetcher.get_stock_list()
    print(f"共 {len(stocks)} 只股票")

    print("\n获取平安银行周线数据...")
    weekly = fetcher.get_weekly_data("000001", start_date="2024-01-01")
    print(weekly.tail())

    print("\n获取基本面数据...")
    fin = fetcher.get_financial_data("000001")
    print(fin)
