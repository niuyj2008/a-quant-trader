"""
股票量化策略决策支持系统 - 交易日志模块

使用SQLite持久化记录所有交易和持仓变动，支持实际交易与回测对比。
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from loguru import logger


class TradeJournal:
    """交易日志（SQLite持久化）"""

    def __init__(self, db_path: str = "data/trade_journal.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # 交易记录表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT DEFAULT '',
                    date TEXT NOT NULL,
                    action TEXT NOT NULL,
                    price REAL NOT NULL,
                    shares INTEGER NOT NULL,
                    amount REAL NOT NULL,
                    commission REAL DEFAULT 0,
                    strategy TEXT DEFAULT '',
                    reason TEXT DEFAULT '',
                    decision_report TEXT DEFAULT '',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 持仓快照表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS holdings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT DEFAULT '',
                    shares INTEGER NOT NULL,
                    cost_price REAL NOT NULL,
                    current_price REAL,
                    unrealized_pnl REAL DEFAULT 0,
                    weight REAL DEFAULT 0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 推荐记录表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT NOT NULL,
                    date TEXT NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT DEFAULT '',
                    strategy TEXT DEFAULT '',
                    action TEXT DEFAULT 'buy',
                    score REAL DEFAULT 0,
                    confidence REAL DEFAULT 0,
                    reason TEXT DEFAULT '',
                    price_at_recommend REAL DEFAULT 0,
                    price_after_1w REAL,
                    return_1w REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    # ==================== 交易记录 ====================

    def record_trade(self, market: str, code: str, action: str, price: float,
                     shares: int, strategy: str = "", reason: str = "",
                     name: str = "", commission: float = 0,
                     decision_report: str = ""):
        """记录一笔交易"""
        amount = price * shares
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trades (market, code, name, date, action, price, shares,
                                   amount, commission, strategy, reason, decision_report)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (market, code, name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                  action, price, shares, amount, commission, strategy, reason,
                  decision_report))
            conn.commit()
        logger.info(f"交易记录: {action} {code}({name}) @{price} x{shares}")

    def get_trades(self, market: Optional[str] = None,
                   code: Optional[str] = None,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   limit: int = 100) -> pd.DataFrame:
        """查询交易记录"""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        if market:
            query += " AND market=?"
            params.append(market)
        if code:
            query += " AND code=?"
            params.append(code)
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += f" ORDER BY date DESC LIMIT {limit}"

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    # ==================== 持仓管理 ====================

    def update_holding(self, market: str, code: str, shares: int,
                       cost_price: float, current_price: float = 0,
                       name: str = ""):
        """更新持仓"""
        unrealized_pnl = (current_price - cost_price) * shares if current_price > 0 else 0
        with sqlite3.connect(self.db_path) as conn:
            # 先删除旧记录
            conn.execute("DELETE FROM holdings WHERE market=? AND code=?", (market, code))
            if shares > 0:  # 只插入有持仓的
                conn.execute("""
                    INSERT INTO holdings (market, code, name, shares, cost_price,
                                         current_price, unrealized_pnl, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (market, code, name, shares, cost_price, current_price,
                      unrealized_pnl, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()

    def get_holdings(self, market: Optional[str] = None) -> pd.DataFrame:
        """获取持仓"""
        query = "SELECT * FROM holdings"
        params = []
        if market:
            query += " WHERE market=?"
            params.append(market)
        query += " ORDER BY code"
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def add_holding(self, market: str, code: str, shares: int,
                    cost_price: float, name: str = ""):
        """添加持仓（用户手动输入）"""
        self.update_holding(market, code, shares, cost_price, name=name)
        logger.info(f"添加持仓: {code}({name}) {shares}股 @{cost_price}")

    def remove_holding(self, market: str, code: str):
        """移除持仓"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM holdings WHERE market=? AND code=?", (market, code))
            conn.commit()

    # ==================== 推荐记录 ====================

    def record_recommendation(self, market: str, code: str, strategy: str,
                              score: float, confidence: float, reason: str,
                              price: float, name: str = ""):
        """记录推荐"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO recommendations (market, date, code, name, strategy,
                                            score, confidence, reason, price_at_recommend)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (market, datetime.now().strftime('%Y-%m-%d'), code, name,
                  strategy, score, confidence, reason, price))
            conn.commit()

    def update_recommendation_performance(self, rec_id: int, price_after: float):
        """更新推荐的后续表现"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT price_at_recommend FROM recommendations WHERE id=?", (rec_id,)
            ).fetchone()
            if row and row[0] > 0:
                ret = (price_after - row[0]) / row[0]
                conn.execute("""
                    UPDATE recommendations SET price_after_1w=?, return_1w=? WHERE id=?
                """, (price_after, ret, rec_id))
                conn.commit()

    def get_recommendations(self, market: Optional[str] = None,
                            limit: int = 50) -> pd.DataFrame:
        """获取推荐历史"""
        query = "SELECT * FROM recommendations"
        params = []
        if market:
            query += " WHERE market=?"
            params.append(market)
        query += f" ORDER BY date DESC LIMIT {limit}"
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_recommendation_performance(self, market: Optional[str] = None) -> Dict:
        """统计推荐准确率"""
        recs = self.get_recommendations(market, limit=200)
        if recs.empty:
            return {"总推荐数": 0}
        filled = recs[recs['return_1w'].notna()]
        if filled.empty:
            return {"总推荐数": len(recs), "已回测数": 0}
        wins = len(filled[filled['return_1w'] > 0])
        return {
            "总推荐数": len(recs),
            "已回测数": len(filled),
            "胜率": f"{wins / len(filled):.1%}" if len(filled) > 0 else "N/A",
            "平均收益": f"{filled['return_1w'].mean():.2%}",
        }
