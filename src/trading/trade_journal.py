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
            # 启用WAL模式提升并发性能
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            # ========== 交易记录表 (增强版) ==========
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
                    stamp_duty REAL DEFAULT 0,
                    strategy TEXT DEFAULT '',
                    reason TEXT DEFAULT '',
                    decision_report TEXT DEFAULT '',
                    recommendation_id INTEGER DEFAULT NULL,
                    cost_basis REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    pnl_pct REAL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # ========== 持仓表 (完全重构) ==========
            conn.execute("""
                CREATE TABLE IF NOT EXISTS holdings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT DEFAULT '',
                    total_shares INTEGER NOT NULL,
                    average_cost REAL NOT NULL,
                    current_price REAL DEFAULT 0,
                    first_buy_date TEXT DEFAULT '',
                    first_buy_price REAL DEFAULT 0,
                    buy_batches TEXT DEFAULT '[]',
                    unrealized_pnl REAL DEFAULT 0,
                    unrealized_pnl_pct REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    total_invested REAL DEFAULT 0,
                    market_value REAL DEFAULT 0,
                    weight REAL DEFAULT 0,
                    stop_loss_price REAL DEFAULT 0,
                    take_profit_price REAL DEFAULT 0,
                    max_drawdown_pct REAL DEFAULT 0,
                    holding_days INTEGER DEFAULT 0,
                    sector TEXT DEFAULT '',
                    strategy_tag TEXT DEFAULT '',
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(market, code)
                )
            """)

            # ========== 推荐记录表 (增强版) ==========
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
                    price_after_1w REAL DEFAULT NULL,
                    return_1w REAL DEFAULT NULL,
                    price_after_1m REAL DEFAULT NULL,
                    return_1m REAL DEFAULT NULL,
                    price_after_3m REAL DEFAULT NULL,
                    return_3m REAL DEFAULT NULL,
                    stop_loss_suggested REAL DEFAULT 0,
                    target_price REAL DEFAULT 0,
                    backtest_status TEXT DEFAULT 'pending',
                    is_executed BOOLEAN DEFAULT 0,
                    executed_trade_id INTEGER DEFAULT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # ========== 组合绩效表 (新建) ==========
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT NOT NULL,
                    month TEXT NOT NULL,
                    total_return REAL DEFAULT 0,
                    benchmark_return REAL DEFAULT 0,
                    alpha REAL DEFAULT 0,
                    volatility REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    sharpe_ratio REAL DEFAULT 0,
                    avg_position_count REAL DEFAULT 0,
                    turnover_rate REAL DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    sector_exposure TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(market, month)
                )
            """)

            # 创建索引提升查询性能
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_code ON trades(code)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_holdings_market ON holdings(market)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_date ON recommendations(date)")

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

    def update_price(self, market: str, code: str, current_price: float):
        """仅更新持仓价格(不改变股数)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM holdings WHERE market=? AND code=?", (market, code)
            )
            existing = cursor.fetchone()

            if not existing:
                logger.warning(f"未找到持仓 {market}:{code},无法更新价格")
                return

            shares = existing['total_shares']
            cost = existing['average_cost']

            # 重新计算盈亏
            unrealized_pnl = (current_price - cost) * shares
            unrealized_pnl_pct = (current_price - cost) / cost if cost > 0 else 0
            market_value = shares * current_price

            conn.execute("""
                UPDATE holdings SET
                    current_price=?,
                    unrealized_pnl=?,
                    unrealized_pnl_pct=?,
                    market_value=?,
                    updated_at=?
                WHERE market=? AND code=?
            """, (current_price, unrealized_pnl, unrealized_pnl_pct, market_value,
                  datetime.now().strftime('%Y-%m-%d %H:%M:%S'), market, code))
            conn.commit()

            logger.info(f"更新价格: {code} ${current_price:.2f} (盈亏{unrealized_pnl_pct:+.2%})")

    def update_holding(self, market: str, code: str, shares: int,
                       cost_price: float, current_price: float = 0,
                       name: str = "", sector: str = "", strategy_tag: str = ""):
        """更新持仓(兼容旧接口,内部使用新结构)"""
        unrealized_pnl = (current_price - cost_price) * shares if current_price > 0 else 0
        unrealized_pnl_pct = (current_price - cost_price) / cost_price if cost_price > 0 and current_price > 0 else 0
        market_value = shares * current_price if current_price > 0 else shares * cost_price

        with sqlite3.connect(self.db_path) as conn:
            # 检查是否已有持仓
            existing = conn.execute(
                "SELECT * FROM holdings WHERE market=? AND code=?", (market, code)
            ).fetchone()

            if shares <= 0:
                # 清仓
                if existing:
                    conn.execute("DELETE FROM holdings WHERE market=? AND code=?", (market, code))
            elif existing:
                # 更新现有持仓(保留历史信息)
                conn.execute("""
                    UPDATE holdings SET
                        total_shares=?, average_cost=?, current_price=?,
                        unrealized_pnl=?, unrealized_pnl_pct=?, market_value=?,
                        updated_at=?
                    WHERE market=? AND code=?
                """, (shares, cost_price, current_price, unrealized_pnl, unrealized_pnl_pct,
                      market_value, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), market, code))
            else:
                # 新建持仓
                conn.execute("""
                    INSERT INTO holdings (
                        market, code, name, total_shares, average_cost, current_price,
                        first_buy_date, first_buy_price, buy_batches,
                        unrealized_pnl, unrealized_pnl_pct, total_invested, market_value,
                        sector, strategy_tag, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    market, code, name, shares, cost_price, current_price,
                    datetime.now().strftime('%Y-%m-%d'), cost_price,
                    json.dumps([{
                        "date": datetime.now().strftime('%Y-%m-%d'),
                        "price": cost_price,
                        "shares": shares
                    }]),
                    unrealized_pnl, unrealized_pnl_pct, shares * cost_price, market_value,
                    sector, strategy_tag, datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))
            conn.commit()

    def add_or_update_position(self, market: str, code: str, shares: int,
                              price: float, name: str = "", sector: str = "",
                              strategy_tag: str = ""):
        """
        添加或更新持仓(增强版 - 支持加仓)

        Args:
            market: 市场('CN'/'US')
            code: 股票代码
            shares: 新增股数
            price: 成交价
            name: 股票名称
            sector: 所属行业
            strategy_tag: 建仓策略标签
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # 使用Row对象方便访问
            cursor = conn.execute(
                "SELECT * FROM holdings WHERE market=? AND code=?", (market, code)
            )
            existing = cursor.fetchone()

            if existing:
                # 加仓: 更新平均成本和股数
                old_shares = existing['total_shares']
                old_cost = existing['average_cost']
                old_batches_str = existing['buy_batches']
                old_batches = json.loads(old_batches_str) if old_batches_str else []

                new_total_shares = old_shares + shares
                new_avg_cost = (old_shares * old_cost + shares * price) / new_total_shares

                # 记录加仓批次
                old_batches.append({
                    "date": datetime.now().strftime('%Y-%m-%d'),
                    "price": price,
                    "shares": shares
                })

                conn.row_factory = None  # 恢复默认
                conn.execute("""
                    UPDATE holdings SET
                        total_shares=?, average_cost=?, buy_batches=?,
                        total_invested=?, updated_at=?
                    WHERE market=? AND code=?
                """, (new_total_shares, new_avg_cost, json.dumps(old_batches),
                      new_total_shares * new_avg_cost, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      market, code))

                logger.info(f"加仓: {code}({name}) +{shares}股 @{price}, 新均价:{new_avg_cost:.2f}")
            else:
                # 首次建仓
                conn.row_factory = None  # 恢复默认
                self.update_holding(market, code, shares, price, price, name, sector, strategy_tag)
                logger.info(f"建仓: {code}({name}) {shares}股 @{price}")

            conn.commit()

    def reduce_position(self, market: str, code: str, shares: int, price: float) -> float:
        """
        减仓或清仓

        Args:
            market: 市场
            code: 股票代码
            shares: 卖出股数
            price: 卖出价

        Returns:
            realized_pnl: 已实现盈亏
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM holdings WHERE market=? AND code=?", (market, code)
            )
            existing = cursor.fetchone()

            if not existing:
                logger.warning(f"减仓失败: {code} 无持仓记录")
                return 0.0

            old_shares = existing['total_shares']
            old_cost = existing['average_cost']
            old_realized_pnl = existing['realized_pnl']

            if shares > old_shares:
                logger.warning(f"减仓失败: {code} 持仓{old_shares}股,卖出{shares}股超出")
                return 0.0

            # 计算本次实现盈亏
            realized_pnl = (price - old_cost) * shares
            total_realized_pnl = old_realized_pnl + realized_pnl

            new_shares = old_shares - shares

            conn.row_factory = None  # 恢复默认

            if new_shares == 0:
                # 清仓
                conn.execute("DELETE FROM holdings WHERE market=? AND code=?", (market, code))
                logger.info(f"清仓: {code} 卖出{shares}股 @{price}, 实现盈亏:{realized_pnl:.2f}")
            else:
                # 部分减仓
                conn.execute("""
                    UPDATE holdings SET
                        total_shares=?, realized_pnl=?, updated_at=?
                    WHERE market=? AND code=?
                """, (new_shares, total_realized_pnl, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      market, code))
                logger.info(f"减仓: {code} 卖出{shares}股 @{price}, 剩余{new_shares}股, 实现盈亏:{realized_pnl:.2f}")

            conn.commit()
            return realized_pnl

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
                    cost_price: float, name: str = "", sector: str = "",
                    strategy_tag: str = ""):
        """添加持仓（用户手动输入或首次建仓）"""
        self.update_holding(market, code, shares, cost_price, cost_price, name, sector, strategy_tag)
        logger.info(f"添加持仓: {code}({name}) {shares}股 @{cost_price}")

    def remove_holding(self, market: str, code: str):
        """移除持仓"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM holdings WHERE market=? AND code=?", (market, code))
            conn.commit()

    # ==================== 推荐记录 ====================

    def record_recommendation(self, market: str, code: str, strategy: str,
                              score: float, confidence: float, reason: str,
                              price: float, name: str = "", action: str = "buy"):
        """记录推荐"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO recommendations (market, date, code, name, strategy, action,
                                            score, confidence, reason, price_at_recommend)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (market, datetime.now().strftime('%Y-%m-%d'), code, name,
                  strategy, action, score, confidence, reason, price))
            conn.commit()

    def add_recommendation(self, market: str, code: str, name: str, strategy: str,
                          action: str, score: float, confidence: float, reason: str,
                          price_at_recommend: float):
        """添加推荐记录(别名方法,用于测试)"""
        self.record_recommendation(
            market=market,
            code=code,
            strategy=strategy,
            score=score,
            confidence=confidence,
            reason=reason,
            price=price_at_recommend,
            name=name,
            action=action
        )

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
        """统计推荐准确率(1周/1月/3月)"""
        recs = self.get_recommendations(market, limit=200)
        if recs.empty:
            return {"总推荐数": 0}

        result = {"总推荐数": len(recs)}

        # 1周回测
        filled_1w = recs[recs['return_1w'].notna()]
        if not filled_1w.empty:
            wins_1w = len(filled_1w[filled_1w['return_1w'] > 0])
            result["1周回测数"] = len(filled_1w)
            result["1周胜率"] = wins_1w / len(filled_1w)
            result["1周平均收益"] = filled_1w['return_1w'].mean()

        # 1月回测
        filled_1m = recs[recs['return_1m'].notna()]
        if not filled_1m.empty:
            wins_1m = len(filled_1m[filled_1m['return_1m'] > 0])
            result["1月回测数"] = len(filled_1m)
            result["1月胜率"] = wins_1m / len(filled_1m)
            result["1月平均收益"] = filled_1m['return_1m'].mean()

        # 3月回测
        filled_3m = recs[recs['return_3m'].notna()]
        if not filled_3m.empty:
            wins_3m = len(filled_3m[filled_3m['return_3m'] > 0])
            result["3月回测数"] = len(filled_3m)
            result["3月胜率"] = wins_3m / len(filled_3m)
            result["3月平均收益"] = filled_3m['return_3m'].mean()

        return result

    def backtest_recommendations(self, lookback_days: int = 90, update_db: bool = True) -> Dict:
        """
        回测历史推荐的后续表现

        Args:
            lookback_days: 回测多少天前的推荐(默认90天)
            update_db: 是否更新数据库中的return_1w/1m/3m字段

        Returns:
            回测统计结果
        """
        from datetime import timedelta
        from src.data.fetcher import DataFetcher

        fetcher = DataFetcher()

        # 获取历史推荐
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = """
                SELECT * FROM recommendations
                WHERE date >= ?
                ORDER BY date DESC
            """
            recs = pd.read_sql_query(query, conn, params=(cutoff_date,))

        if recs.empty:
            logger.warning(f"未找到{lookback_days}天内的推荐记录")
            return {"回测推荐数": 0}

        logger.info(f"开始回测{len(recs)}条推荐记录...")

        updated_count = 0
        results = []

        for idx, rec in recs.iterrows():
            rec_id = rec['id']
            code = rec['code']
            market = rec['market']
            rec_date = datetime.strptime(rec['date'], '%Y-%m-%d')
            rec_price = rec['price_at_recommend']

            # 跳过已回测的
            if pd.notna(rec['return_3m']):
                continue

            try:
                # 获取推荐后的价格数据
                end_date = datetime.now()
                start_date = rec_date

                df = fetcher.get_daily_data(
                    code,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    market=market
                )

                if df is None or df.empty:
                    logger.warning(f"无法获取{market}:{code}的历史数据")
                    continue

                # 计算1周/1月/3月后的价格
                price_1w = None
                price_1m = None
                price_3m = None

                date_1w = rec_date + timedelta(days=7)
                date_1m = rec_date + timedelta(days=30)
                date_3m = rec_date + timedelta(days=90)

                # 查找最接近的交易日价格
                if date_1w <= datetime.now():
                    price_1w = self._get_closest_price(df, date_1w)
                if date_1m <= datetime.now():
                    price_1m = self._get_closest_price(df, date_1m)
                if date_3m <= datetime.now():
                    price_3m = self._get_closest_price(df, date_3m)

                # 计算收益率
                return_1w = (price_1w - rec_price) / rec_price if price_1w else None
                return_1m = (price_1m - rec_price) / rec_price if price_1m else None
                return_3m = (price_3m - rec_price) / rec_price if price_3m else None

                if update_db:
                    # 更新数据库
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute("""
                            UPDATE recommendations SET
                                price_after_1w = ?,
                                return_1w = ?,
                                price_after_1m = ?,
                                return_1m = ?,
                                price_after_3m = ?,
                                return_3m = ?,
                                backtest_status = 'completed'
                            WHERE id = ?
                        """, (price_1w, return_1w, price_1m, return_1m,
                              price_3m, return_3m, rec_id))
                        conn.commit()
                    updated_count += 1

                results.append({
                    'code': code,
                    'date': rec['date'],
                    'return_1w': return_1w,
                    'return_1m': return_1m,
                    'return_3m': return_3m,
                })

            except Exception as e:
                logger.error(f"回测{code}失败: {e}")
                continue

        logger.info(f"回测完成,更新{updated_count}条记录")

        # 统计结果
        results_df = pd.DataFrame(results)

        if results_df.empty:
            return {"回测推荐数": len(recs), "更新数": 0}

        summary = {
            "回测推荐数": len(recs),
            "更新数": updated_count,
        }

        # 1周统计
        r1w = results_df[results_df['return_1w'].notna()]
        if not r1w.empty:
            summary["1周胜率"] = (r1w['return_1w'] > 0).sum() / len(r1w)
            summary["1周平均收益"] = r1w['return_1w'].mean()

        # 1月统计
        r1m = results_df[results_df['return_1m'].notna()]
        if not r1m.empty:
            summary["1月胜率"] = (r1m['return_1m'] > 0).sum() / len(r1m)
            summary["1月平均收益"] = r1m['return_1m'].mean()

        # 3月统计
        r3m = results_df[results_df['return_3m'].notna()]
        if not r3m.empty:
            summary["3月胜率"] = (r3m['return_3m'] > 0).sum() / len(r3m)
            summary["3月平均收益"] = r3m['return_3m'].mean()

        return summary

    def _get_closest_price(self, df: pd.DataFrame, target_date: datetime) -> Optional[float]:
        """获取最接近目标日期的收盘价"""
        df_copy = df.copy()
        df_copy.index = pd.to_datetime(df_copy.index)

        # 找到最接近且不早于目标日期的交易日
        future_dates = df_copy[df_copy.index >= target_date]

        if future_dates.empty:
            # 如果没有未来数据,返回最后一个价格
            return df_copy['close'].iloc[-1]

        # 返回最接近的交易日收盘价
        return future_dates['close'].iloc[0]

    def get_strategy_winrate_comparison(self) -> pd.DataFrame:
        """对比不同策略的胜率"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT
                    strategy,
                    COUNT(*) as total_count,
                    AVG(CASE WHEN return_1w > 0 THEN 1 ELSE 0 END) as winrate_1w,
                    AVG(return_1w) as avg_return_1w,
                    AVG(CASE WHEN return_1m > 0 THEN 1 ELSE 0 END) as winrate_1m,
                    AVG(return_1m) as avg_return_1m,
                    AVG(CASE WHEN return_3m > 0 THEN 1 ELSE 0 END) as winrate_3m,
                    AVG(return_3m) as avg_return_3m
                FROM recommendations
                WHERE return_1w IS NOT NULL
                GROUP BY strategy
                ORDER BY winrate_3m DESC
            """
            df = pd.read_sql_query(query, conn)

        return df
