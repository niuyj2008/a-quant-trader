"""
股票量化策略决策支持系统 - 本地数据缓存模块

使用SQLite存储历史行情数据，避免重复API调用，支持增量更新
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
from loguru import logger


class DataCache:
    """本地数据缓存（SQLite）"""

    def __init__(self, db_path: str = "data/cache/quant_cache.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_ohlcv (
                    market TEXT NOT NULL,
                    code TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL,
                    volume REAL, amount REAL, turnover REAL,
                    PRIMARY KEY (market, code, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS financial_data (
                    market TEXT NOT NULL,
                    code TEXT NOT NULL,
                    date TEXT NOT NULL,
                    pe REAL, pb REAL, ps REAL,
                    roe REAL, roa REAL,
                    revenue_growth REAL, net_profit_growth REAL,
                    gross_margin REAL, debt_ratio REAL,
                    market_cap REAL,
                    PRIMARY KEY (market, code, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS macro_data (
                    indicator TEXT NOT NULL,
                    date TEXT NOT NULL,
                    value REAL,
                    PRIMARY KEY (indicator, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_meta (
                    market TEXT NOT NULL,
                    code TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    last_update TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    PRIMARY KEY (market, code, table_name)
                )
            """)
            conn.commit()

    # ==================== 日线数据缓存 ====================

    def save_daily(self, code: str, df: pd.DataFrame, market: str = "CN"):
        """保存日线数据到缓存"""
        if df.empty:
            return
        with sqlite3.connect(self.db_path) as conn:
            for idx, row in df.iterrows():
                date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
                conn.execute("""
                    INSERT OR REPLACE INTO daily_ohlcv
                    (market, code, date, open, high, low, close, volume, amount, turnover)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (market, code, date_str,
                      float(row.get('open', 0)), float(row.get('high', 0)),
                      float(row.get('low', 0)), float(row.get('close', 0)),
                      float(row.get('volume', 0)), float(row.get('amount', 0)),
                      float(row.get('turnover', 0))))
            # 更新元数据
            dates = df.index
            conn.execute("""
                INSERT OR REPLACE INTO cache_meta (market, code, table_name, last_update, start_date, end_date)
                VALUES (?, ?, 'daily_ohlcv', ?, ?, ?)
            """, (market, code, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                  min(dates).strftime('%Y-%m-%d'), max(dates).strftime('%Y-%m-%d')))
            conn.commit()
        logger.debug(f"缓存 {market}/{code} 日线数据 {len(df)} 条")

    def load_daily(self, code: str, start_date: Optional[str] = None,
                   end_date: Optional[str] = None, market: str = "CN") -> pd.DataFrame:
        """从缓存加载日线数据"""
        query = "SELECT date, open, high, low, close, volume, amount, turnover FROM daily_ohlcv WHERE market=? AND code=?"
        params: list = [market, code]
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return pd.DataFrame()

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df

    def get_cache_range(self, code: str, market: str = "CN") -> Optional[tuple]:
        """获取缓存数据的日期范围"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT start_date, end_date FROM cache_meta WHERE market=? AND code=? AND table_name='daily_ohlcv'",
                (market, code)
            ).fetchone()
        return (row[0], row[1]) if row else None

    # ==================== 基本面数据缓存 ====================

    def save_financial(self, code: str, data: Dict, date: str, market: str = "CN"):
        """保存基本面数据"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO financial_data
                (market, code, date, pe, pb, ps, roe, roa,
                 revenue_growth, net_profit_growth, gross_margin, debt_ratio, market_cap)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (market, code, date,
                  data.get('pe'), data.get('pb'), data.get('ps'),
                  data.get('roe'), data.get('roa'),
                  data.get('revenue_growth'), data.get('net_profit_growth'),
                  data.get('gross_margin'), data.get('debt_ratio'),
                  data.get('market_cap')))
            conn.commit()

    def load_financial(self, code: str, market: str = "CN") -> pd.DataFrame:
        """加载基本面数据"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM financial_data WHERE market=? AND code=? ORDER BY date",
                conn, params=[market, code])
        return df

    # ==================== 宏观数据缓存 ====================

    def save_macro(self, indicator: str, df: pd.DataFrame):
        """保存宏观数据"""
        with sqlite3.connect(self.db_path) as conn:
            for idx, row in df.iterrows():
                date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
                conn.execute(
                    "INSERT OR REPLACE INTO macro_data (indicator, date, value) VALUES (?, ?, ?)",
                    (indicator, date_str, float(row.iloc[0]) if isinstance(row, pd.Series) else float(row)))
            conn.commit()

    def load_macro(self, indicator: str) -> pd.Series:
        """加载宏观数据"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT date, value FROM macro_data WHERE indicator=? ORDER BY date",
                conn, params=[indicator])
        if df.empty:
            return pd.Series()
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date')['value']
