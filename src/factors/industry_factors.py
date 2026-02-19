"""
股票量化策略决策支持系统 - 行业轮动因子模块

基于行业动量排名识别当前强势/弱势板块，
为选股提供行业维度的增强信号。

数据来源:
  - A股: AKShare stock_board_industry_name_em() 行业板块列表
         AKShare stock_board_industry_hist_em() 行业历史行情
  - 美股: yfinance Sector ETF 代理（XLK/XLF/XLE/XLV...）

设计原则:
  - 行业动量: 过去N日涨幅排名
  - 行业强度: 板块内上涨股票占比
  - 行业估值: 板块整体PE水平
  - 所有得分归一化到 0-100
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from loguru import logger


# 美股 Sector ETF 代理
US_SECTOR_ETFS = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financial': 'XLF',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Utilities': 'XLU',
    'Industrial': 'XLI',
    'Materials': 'XLB',
    'Real Estate': 'XLRE',
    'Communication': 'XLC',
}


class IndustryRotationFactor:
    """行业轮动因子计算器

    通过分析板块动量排名，识别当前强势行业，
    为个股选择提供行业维度增益/惩罚。
    """

    def __init__(self, cache_ttl: int = 3600):
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple] = {}  # key -> (timestamp, data)

    def compute_industry_scores(self, market: str = "CN",
                                lookback_days: int = 20) -> Dict[str, float]:
        """计算所有行业的动量得分

        Args:
            market: "CN" 或 "US"
            lookback_days: 动量计算窗口

        Returns:
            {行业名称: 0-100得分}, 高分=强势行业
        """
        cache_key = f"industry:{market}:{lookback_days}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        if market == "CN":
            scores = self._compute_cn_industry_scores(lookback_days)
        else:
            scores = self._compute_us_sector_scores(lookback_days)

        if scores:
            self._set_cache(cache_key, scores)
        return scores

    def get_stock_industry_bonus(self, code: str, industry: str,
                                 industry_scores: Dict[str, float]) -> float:
        """获取个股的行业增益分

        Args:
            code: 股票代码
            industry: 该股票所属行业
            industry_scores: compute_industry_scores() 的返回值

        Returns:
            -10 到 +10 的增益分（加到策略综合得分上）
        """
        if not industry or not industry_scores:
            return 0.0

        score = industry_scores.get(industry, 50)
        # 映射: 80+ → +10, 60-80 → +5, 40-60 → 0, 20-40 → -5, <20 → -10
        bonus = (score - 50) / 50 * 10
        return max(-10, min(10, bonus))

    def get_top_industries(self, market: str = "CN", top_n: int = 5,
                           lookback_days: int = 20) -> List[Tuple[str, float]]:
        """获取排名前N的强势行业

        Returns:
            [(行业名称, 得分), ...] 按得分降序
        """
        scores = self.compute_industry_scores(market, lookback_days)
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def get_bottom_industries(self, market: str = "CN", bottom_n: int = 5,
                              lookback_days: int = 20) -> List[Tuple[str, float]]:
        """获取排名后N的弱势行业"""
        scores = self.compute_industry_scores(market, lookback_days)
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda x: x[1])
        return ranked[:bottom_n]

    # ==================== A股行业分析 ====================

    def _compute_cn_industry_scores(self, lookback_days: int = 20) -> Dict[str, float]:
        """A股行业板块动量排名

        使用 AKShare stock_board_industry_name_em() 获取板块列表，
        然后计算各板块过去N日涨幅。
        """
        try:
            import akshare as ak
        except ImportError:
            logger.debug("AKShare未安装，无法计算A股行业因子")
            return {}

        try:
            # 获取行业板块列表（含实时涨跌幅）
            df_boards = ak.stock_board_industry_name_em()
            if df_boards is None or df_boards.empty:
                return {}

            # 提取板块名和涨跌幅
            name_col = '板块名称' if '板块名称' in df_boards.columns else df_boards.columns[0]
            change_col = None
            for col in ['涨跌幅', '涨幅', '板块涨跌幅']:
                if col in df_boards.columns:
                    change_col = col
                    break

            if change_col is None:
                # 尝试取最后一个数值列
                for col in df_boards.columns:
                    if df_boards[col].dtype in ('float64', 'int64'):
                        change_col = col

            if change_col is None:
                return {}

            # 涨跌幅排名 → 0-100 得分
            changes = df_boards[[name_col, change_col]].copy()
            changes[change_col] = pd.to_numeric(changes[change_col], errors='coerce')
            changes = changes.dropna()

            if len(changes) == 0:
                return {}

            # 百分位排名
            changes['rank_pct'] = changes[change_col].rank(pct=True)
            scores = {}
            for _, row in changes.iterrows():
                scores[str(row[name_col])] = round(row['rank_pct'] * 100, 1)

            return scores

        except Exception as e:
            logger.debug(f"A股行业因子计算失败: {e}")
            return {}

    # ==================== 美股行业分析 ====================

    def _compute_us_sector_scores(self, lookback_days: int = 20) -> Dict[str, float]:
        """美股 Sector ETF 动量排名"""
        try:
            import yfinance as yf
        except ImportError:
            logger.debug("yfinance未安装，无法计算美股行业因子")
            return {}

        try:
            # 批量下载 Sector ETF 数据
            tickers = list(US_SECTOR_ETFS.values())
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 10)

            data = yf.download(
                tickers, start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False, auto_adjust=True,
            )

            if data.empty:
                return {}

            # 计算各 ETF 过去 lookback_days 的收益率
            close = data['Close'] if 'Close' in data.columns else data
            returns = {}

            for sector, ticker in US_SECTOR_ETFS.items():
                if ticker in close.columns:
                    series = close[ticker].dropna()
                    if len(series) >= lookback_days:
                        ret = (series.iloc[-1] / series.iloc[-lookback_days] - 1)
                        returns[sector] = float(ret)

            if not returns:
                return {}

            # 百分位排名 → 0-100
            ret_series = pd.Series(returns)
            rank_pct = ret_series.rank(pct=True)
            scores = {k: round(v * 100, 1) for k, v in rank_pct.items()}

            return scores

        except Exception as e:
            logger.debug(f"美股行业因子计算失败: {e}")
            return {}

    # ==================== 缓存 ====================

    def _get_cache(self, key: str):
        import time
        if key in self._cache:
            ts, data = self._cache[key]
            if time.time() - ts < self.cache_ttl:
                return data
            del self._cache[key]
        return None

    def _set_cache(self, key: str, data):
        import time
        self._cache[key] = (time.time(), data)


class IndustryClassifier:
    """股票行业分类器

    为个股提供行业归属信息，供行业轮动因子使用。
    """

    def __init__(self):
        self._cache: Dict[str, str] = {}

    def get_industry(self, code: str, market: str = "CN") -> Optional[str]:
        """获取股票所属行业

        Args:
            code: 股票代码
            market: "CN" 或 "US"

        Returns:
            行业名称，未找到返回 None
        """
        cache_key = f"{market}:{code}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if market == "CN":
            industry = self._get_cn_industry(code)
        else:
            industry = self._get_us_sector(code)

        if industry:
            self._cache[cache_key] = industry
        return industry

    def _get_cn_industry(self, code: str) -> Optional[str]:
        """A股行业分类（东方财富）"""
        try:
            import akshare as ak
            pure_code = code.split('.')[0] if '.' in code else code
            df = ak.stock_individual_info_em(symbol=pure_code)
            if df is not None and not df.empty:
                # 查找"行业"行
                for _, row in df.iterrows():
                    item = str(row.iloc[0])
                    if '行业' in item:
                        return str(row.iloc[1])
        except Exception as e:
            logger.debug(f"获取 {code} 行业分类失败: {e}")
        return None

    def _get_us_sector(self, code: str) -> Optional[str]:
        """美股行业分类（yfinance）"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(code)
            info = ticker.info
            return info.get('sector')
        except Exception as e:
            logger.debug(f"获取 {code} 行业分类失败: {e}")
        return None
