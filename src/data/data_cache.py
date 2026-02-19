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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    code TEXT NOT NULL,
                    market TEXT NOT NULL DEFAULT 'CN',
                    strategy TEXT NOT NULL,
                    action TEXT NOT NULL,
                    confidence REAL,
                    composite_score REAL,
                    factor_scores TEXT,
                    weights_version TEXT,
                    price_at_signal REAL,
                    return_5d REAL,
                    return_10d REAL,
                    return_20d REAL,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signal_log_code_date
                ON signal_log (code, date)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    step TEXT NOT NULL,
                    market TEXT NOT NULL DEFAULT 'US',
                    strategy TEXT,
                    method TEXT,
                    params TEXT,
                    result_summary TEXT,
                    result_detail TEXT,
                    stock_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.commit()

    # ==================== 日线数据缓存 ====================

    def save_daily(self, code: str, df: pd.DataFrame, market: str = "CN"):
        """保存日线数据到缓存（批量写入优化）"""
        if df.empty:
            return
        rows = []
        for idx, row in df.iterrows():
            date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
            rows.append((market, code, date_str,
                         float(row.get('open', 0)), float(row.get('high', 0)),
                         float(row.get('low', 0)), float(row.get('close', 0)),
                         float(row.get('volume', 0)), float(row.get('amount', 0)),
                         float(row.get('turnover', 0))))
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO daily_ohlcv
                (market, code, date, open, high, low, close, volume, amount, turnover)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
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

    def clear_daily(self, code: str, market: str = "CN"):
        """清除指定股票的日线缓存数据"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM daily_ohlcv WHERE market=? AND code=?", (market, code))
            conn.execute("DELETE FROM cache_meta WHERE market=? AND code=? AND table_name='daily_ohlcv'", (market, code))
            conn.commit()
        logger.debug(f"已清除缓存: {market}/{code}")

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

    # ==================== 行研数据缓存 (Phase 10.1) ====================

    def save_research(self, code: str, data: Dict, market: str = "CN"):
        """保存行研数据到缓存（JSON序列化存储）"""
        import json

        # 将 DataFrame 转为 JSON 字符串存储
        serializable = {}
        for key, val in data.items():
            if isinstance(val, pd.DataFrame):
                serializable[key] = val.to_json(date_format='iso')
            elif isinstance(val, dict):
                serializable[key] = json.dumps(val)
            else:
                serializable[key] = str(val)

        json_str = json.dumps(serializable, ensure_ascii=False)
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_data (
                    market TEXT NOT NULL,
                    code TEXT NOT NULL,
                    data_json TEXT,
                    updated_at TEXT,
                    PRIMARY KEY (market, code)
                )
            """)
            conn.execute("""
                INSERT OR REPLACE INTO research_data (market, code, data_json, updated_at)
                VALUES (?, ?, ?, ?)
            """, (market, code, json_str, now))
            conn.commit()
        logger.debug(f"缓存行研数据: {market}/{code}")

    def load_research(self, code: str, market: str = "CN") -> Optional[Dict]:
        """从缓存加载行研数据（24小时有效）"""
        import json

        try:
            with sqlite3.connect(self.db_path) as conn:
                # 检查表是否存在
                table_exists = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='research_data'"
                ).fetchone()
                if not table_exists:
                    return None

                row = conn.execute(
                    "SELECT data_json, updated_at FROM research_data WHERE market=? AND code=?",
                    (market, code)
                ).fetchone()

            if not row:
                return None

            # 检查是否过期（24小时）
            updated_at = datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
            if (datetime.now() - updated_at).total_seconds() > 86400:
                return None

            serialized = json.loads(row[0])
            result = {}
            for key, val_str in serialized.items():
                if key in ('recommendations', 'earnings_estimate', 'revenue_estimate'):
                    try:
                        result[key] = pd.read_json(val_str)
                    except Exception:
                        pass
                elif key == 'price_targets':
                    try:
                        result[key] = json.loads(val_str)
                    except Exception:
                        pass
                else:
                    result[key] = val_str

            return result if result else None

        except Exception as e:
            logger.debug(f"加载行研缓存失败: {e}")
            return None

    # ==================== 信号日志 ====================

    def save_signal(self, date: str, code: str, strategy: str, action: str,
                    confidence: float, composite_score: float,
                    factor_scores: Optional[Dict] = None,
                    weights_version: str = "default",
                    price_at_signal: float = 0.0,
                    market: str = "CN"):
        """记录策略信号到日志"""
        import json
        factor_json = json.dumps(factor_scores, ensure_ascii=False) if factor_scores else None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO signal_log
                (date, code, market, strategy, action, confidence, composite_score,
                 factor_scores, weights_version, price_at_signal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (date, code, market, strategy, action, confidence, composite_score,
                  factor_json, weights_version, price_at_signal))
            conn.commit()

    def backfill_signal_returns(self, signal_id: int, return_5d: float = None,
                                return_10d: float = None, return_20d: float = None):
        """回填信号的实际收益"""
        updates = []
        params = []
        if return_5d is not None:
            updates.append("return_5d = ?")
            params.append(return_5d)
        if return_10d is not None:
            updates.append("return_10d = ?")
            params.append(return_10d)
        if return_20d is not None:
            updates.append("return_20d = ?")
            params.append(return_20d)
        if not updates:
            return
        params.append(signal_id)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"UPDATE signal_log SET {', '.join(updates)} WHERE id = ?", params)
            conn.commit()

    def load_signals(self, strategy: Optional[str] = None, code: Optional[str] = None,
                     market: str = "CN", limit: int = 1000) -> pd.DataFrame:
        """加载信号日志"""
        query = "SELECT * FROM signal_log WHERE market = ?"
        params: list = [market]
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        if code:
            query += " AND code = ?"
            params.append(code)
        query += " ORDER BY date DESC LIMIT ?"
        params.append(limit)
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def get_pending_backfill_signals(self, market: str = "CN") -> pd.DataFrame:
        """获取需要回填收益的信号（return_5d/10d/20d为NULL的记录）"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("""
                SELECT * FROM signal_log
                WHERE market = ? AND (return_5d IS NULL OR return_10d IS NULL OR return_20d IS NULL)
                ORDER BY date
            """, conn, params=[market])
        return df

    def batch_backfill_returns(self, fetcher, market: str = "CN") -> Dict[str, int]:
        """批量回填信号收益

        对所有pending信号，获取信号日之后的价格数据，计算5/10/20日真实收益。

        Args:
            fetcher: DataFetcher实例，用于获取价格数据
            market: 市场代码

        Returns:
            {'total': 待回填数, 'filled': 已回填数, 'skipped': 跳过数}
        """
        pending = self.get_pending_backfill_signals(market=market)
        if pending.empty:
            return {'total': 0, 'filled': 0, 'skipped': 0}

        filled = 0
        skipped = 0
        today = datetime.now()

        # 按股票代码分组，减少API调用
        for code, group in pending.groupby('code'):
            try:
                # 获取该股票的历史数据
                earliest_date = group['date'].min()
                df = fetcher.get_daily_data(code, start_date=earliest_date, market=market)
                if df is None or df.empty:
                    skipped += len(group)
                    continue

                # 确保索引是datetime
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)

                for _, signal in group.iterrows():
                    signal_date = pd.to_datetime(signal['date'])
                    signal_price = signal.get('price_at_signal', 0)

                    if signal_price is None or signal_price <= 0:
                        # 尝试从数据中获取信号日价格
                        if signal_date in df.index:
                            signal_price = df.loc[signal_date, 'close']
                        else:
                            skipped += 1
                            continue

                    # 计算各期限收益
                    r5, r10, r20 = None, None, None
                    future_dates = df.index[df.index > signal_date]

                    if len(future_dates) >= 5 and signal['return_5d'] is None:
                        price_5d = df.loc[future_dates[4], 'close']
                        r5 = (price_5d - signal_price) / signal_price

                    if len(future_dates) >= 10 and signal['return_10d'] is None:
                        price_10d = df.loc[future_dates[9], 'close']
                        r10 = (price_10d - signal_price) / signal_price

                    if len(future_dates) >= 20 and signal['return_20d'] is None:
                        price_20d = df.loc[future_dates[19], 'close']
                        r20 = (price_20d - signal_price) / signal_price

                    if r5 is not None or r10 is not None or r20 is not None:
                        self.backfill_signal_returns(
                            signal_id=signal['id'],
                            return_5d=r5, return_10d=r10, return_20d=r20
                        )
                        filled += 1
                    else:
                        skipped += 1

            except Exception as e:
                logger.debug(f"回填 {code} 信号收益失败: {e}")
                skipped += len(group)

        logger.info(f"信号收益回填完成: 总计{len(pending)}, 已填{filled}, 跳过{skipped}")
        return {'total': len(pending), 'filled': filled, 'skipped': skipped}

    # ==================== 训练历史记录 ====================

    def save_training_history(self, step: str, market: str = "US",
                              strategy: str = None, method: str = None,
                              params: Dict = None, result_summary: str = None,
                              result_detail: str = None, stock_count: int = 0):
        """保存训练步骤的执行记录"""
        import json
        params_json = json.dumps(params, ensure_ascii=False) if params else None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO training_history
                (step, market, strategy, method, params, result_summary, result_detail, stock_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (step, market, strategy, method, params_json,
                  result_summary, result_detail, stock_count))
            conn.commit()

    def load_training_history(self, step: str = None, market: str = None,
                              limit: int = 20) -> pd.DataFrame:
        """加载训练历史记录"""
        query = "SELECT * FROM training_history WHERE 1=1"
        params_list: list = []
        if step:
            query += " AND step = ?"
            params_list.append(step)
        if market:
            query += " AND market = ?"
            params_list.append(market)
        query += " ORDER BY created_at DESC LIMIT ?"
        params_list.append(limit)
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params_list)
