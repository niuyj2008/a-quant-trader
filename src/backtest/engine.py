"""
A股量化交易系统 - 回测引擎

提供策略回测框架和评估指标
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 1_000_000.0  # 初始资金
    commission_rate: float = 0.0003       # 佣金费率（万三）
    stamp_duty: float = 0.001             # 印花税（千一，卖出时收取）
    slippage: float = 0.001               # 滑点
    benchmark: str = "000300"             # 基准指数


@dataclass
class Position:
    """持仓信息"""
    code: str
    shares: int
    avg_cost: float
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def profit(self) -> float:
        return (self.current_price - self.avg_cost) * self.shares
    
    @property
    def profit_pct(self) -> float:
        if self.avg_cost == 0:
            return 0
        return (self.current_price - self.avg_cost) / self.avg_cost


@dataclass
class Trade:
    """交易记录"""
    date: datetime
    code: str
    direction: str  # "buy" or "sell"
    price: float
    shares: int
    amount: float
    commission: float
    stamp_duty: float


@dataclass 
class BacktestResult:
    """回测结果"""
    # 收益曲线
    equity_curve: pd.Series = None
    benchmark_curve: pd.Series = None
    
    # 交易记录
    trades: List[Trade] = field(default_factory=list)
    
    # 绩效指标
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # 基准对比
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    
    def summary(self) -> Dict:
        """返回绩效摘要"""
        return {
            "总收益率": f"{self.total_return:.2%}",
            "年化收益率": f"{self.annual_return:.2%}",
            "最大回撤": f"{self.max_drawdown:.2%}",
            "夏普比率": f"{self.sharpe_ratio:.2f}",
            "卡尔玛比率": f"{self.calmar_ratio:.2f}",
            "胜率": f"{self.win_rate:.2%}",
            "盈亏比": f"{self.profit_factor:.2f}",
            "基准收益率": f"{self.benchmark_return:.2%}",
            "Alpha": f"{self.alpha:.2%}",
            "Beta": f"{self.beta:.2f}",
            "交易次数": len(self.trades),
        }


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        初始化回测引擎
        
        Args:
            config: 回测配置
        """
        self.config = config or BacktestConfig()
        self.reset()
    
    def reset(self):
        """重置回测状态"""
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []
    
    @property
    def total_equity(self) -> float:
        """总权益"""
        position_value = sum(p.market_value for p in self.positions.values())
        return self.cash + position_value
    
    def buy(self, code: str, price: float, shares: int, date: datetime) -> bool:
        """
        买入股票
        
        Args:
            code: 股票代码
            price: 买入价格
            shares: 买入数量（手数 * 100）
            date: 交易日期
            
        Returns:
            是否成功
        """
        # 计算滑点后的实际价格
        actual_price = price * (1 + self.config.slippage)
        
        # 计算交易金额和佣金
        amount = actual_price * shares
        commission = max(amount * self.config.commission_rate, 5)  # 最低5元
        total_cost = amount + commission
        
        # 检查资金是否充足
        if total_cost > self.cash:
            logger.warning(f"资金不足: 需要 {total_cost:.2f}, 可用 {self.cash:.2f}")
            return False
        
        # 更新现金
        self.cash -= total_cost
        
        # 更新持仓
        if code in self.positions:
            pos = self.positions[code]
            total_shares = pos.shares + shares
            pos.avg_cost = (pos.avg_cost * pos.shares + actual_price * shares) / total_shares
            pos.shares = total_shares
        else:
            self.positions[code] = Position(
                code=code,
                shares=shares,
                avg_cost=actual_price,
                current_price=actual_price
            )
        
        # 记录交易
        trade = Trade(
            date=date,
            code=code,
            direction="buy",
            price=actual_price,
            shares=shares,
            amount=amount,
            commission=commission,
            stamp_duty=0
        )
        self.trades.append(trade)
        
        logger.debug(f"买入 {code}: {shares}股 @ {actual_price:.2f}, 佣金 {commission:.2f}")
        return True
    
    def sell(self, code: str, price: float, shares: int, date: datetime) -> bool:
        """
        卖出股票
        
        Args:
            code: 股票代码
            price: 卖出价格
            shares: 卖出数量
            date: 交易日期
            
        Returns:
            是否成功
        """
        if code not in self.positions:
            logger.warning(f"没有持有 {code}")
            return False
        
        pos = self.positions[code]
        if pos.shares < shares:
            logger.warning(f"持仓不足: 持有 {pos.shares}, 卖出 {shares}")
            return False
        
        # 计算滑点后的实际价格
        actual_price = price * (1 - self.config.slippage)
        
        # 计算交易金额、佣金和印花税
        amount = actual_price * shares
        commission = max(amount * self.config.commission_rate, 5)
        stamp_duty = amount * self.config.stamp_duty
        total_received = amount - commission - stamp_duty
        
        # 更新现金
        self.cash += total_received
        
        # 更新持仓
        pos.shares -= shares
        if pos.shares == 0:
            del self.positions[code]
        
        # 记录交易
        trade = Trade(
            date=date,
            code=code,
            direction="sell",
            price=actual_price,
            shares=shares,
            amount=amount,
            commission=commission,
            stamp_duty=stamp_duty
        )
        self.trades.append(trade)
        
        logger.debug(f"卖出 {code}: {shares}股 @ {actual_price:.2f}, 佣金 {commission:.2f}, 印花税 {stamp_duty:.2f}")
        return True
    
    def update_prices(self, prices: Dict[str, float], date: datetime):
        """
        更新持仓市值
        
        Args:
            prices: 股票代码 -> 价格
            date: 日期
        """
        for code, pos in self.positions.items():
            if code in prices:
                pos.current_price = prices[code]
        
        # 记录权益
        self.equity_history.append((date, self.total_equity))
    
    def run(
        self,
        data: Dict[str, pd.DataFrame],
        strategy: Callable,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> BacktestResult:
        """
        运行回测
        
        Args:
            data: 股票代码 -> OHLCV DataFrame
            strategy: 策略函数，签名为 strategy(engine, date, data) -> signals
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            回测结果
        """
        self.reset()
        
        # 获取所有交易日期
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index.tolist())
        all_dates = sorted(all_dates)
        
        # 过滤日期范围
        if start_date:
            all_dates = [d for d in all_dates if d >= pd.Timestamp(start_date)]
        if end_date:
            all_dates = [d for d in all_dates if d <= pd.Timestamp(end_date)]
        
        logger.info(f"开始回测: {all_dates[0]} 至 {all_dates[-1]}")
        
        # 逐日回测
        for date in all_dates:
            # 获取当日数据
            daily_data = {}
            daily_prices = {}
            for code, df in data.items():
                if date in df.index:
                    daily_data[code] = df.loc[:date]
                    daily_prices[code] = df.loc[date, 'close']
            
            # 更新持仓价格
            self.update_prices(daily_prices, date)
            
            # 执行策略
            signals = strategy(self, date, daily_data)
            
            # 处理交易信号
            if signals:
                for signal in signals:
                    code = signal['code']
                    action = signal['action']
                    shares = signal.get('shares', 100)
                    
                    if code in daily_prices:
                        price = daily_prices[code]
                        if action == 'buy':
                            self.buy(code, price, shares, date)
                        elif action == 'sell':
                            self.sell(code, price, shares, date)
        
        # 生成回测结果
        result = self._calculate_metrics()
        logger.info(f"回测完成: 总收益率 {result.total_return:.2%}")
        
        return result
    
    def _calculate_metrics(self) -> BacktestResult:
        """计算回测指标"""
        result = BacktestResult()
        result.trades = self.trades
        
        # 生成权益曲线
        dates, equity = zip(*self.equity_history) if self.equity_history else ([], [])
        result.equity_curve = pd.Series(equity, index=pd.DatetimeIndex(dates))
        
        if len(result.equity_curve) < 2:
            return result
        
        # 计算收益率
        initial = self.config.initial_capital
        final = result.equity_curve.iloc[-1]
        result.total_return = (final - initial) / initial
        
        # 年化收益率
        days = (result.equity_curve.index[-1] - result.equity_curve.index[0]).days
        if days > 0:
            result.annual_return = (1 + result.total_return) ** (365 / days) - 1
        
        # 最大回撤
        rolling_max = result.equity_curve.cummax()
        drawdown = (result.equity_curve - rolling_max) / rolling_max
        result.max_drawdown = drawdown.min()
        
        # 夏普比率 (假设无风险利率3%)
        daily_returns = result.equity_curve.pct_change().dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            rf_daily = 0.03 / 252
            excess_returns = daily_returns - rf_daily
            result.sharpe_ratio = np.sqrt(252) * excess_returns.mean() / daily_returns.std()
        
        # 卡尔玛比率
        if result.max_drawdown != 0:
            result.calmar_ratio = -result.annual_return / result.max_drawdown
        
        # 胜率和盈亏比
        if self.trades:
            buy_trades = {t.code: t for t in self.trades if t.direction == 'buy'}
            sell_trades = [t for t in self.trades if t.direction == 'sell']
            
            profits = []
            for sell in sell_trades:
                if sell.code in buy_trades:
                    buy = buy_trades[sell.code]
                    profit = (sell.price - buy.price) * sell.shares
                    profits.append(profit)
            
            if profits:
                wins = [p for p in profits if p > 0]
                losses = [p for p in profits if p < 0]
                result.win_rate = len(wins) / len(profits)
                
                if losses:
                    avg_win = sum(wins) / len(wins) if wins else 0
                    avg_loss = abs(sum(losses) / len(losses))
                    result.profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        return result


# 示例策略
def ma_cross_strategy(engine: BacktestEngine, date: datetime, data: Dict[str, pd.DataFrame]) -> List[Dict]:
    """
    均线交叉策略示例
    
    买入条件: 5日均线上穿20日均线
    卖出条件: 5日均线下穿20日均线
    """
    signals = []
    
    for code, df in data.items():
        if len(df) < 20:
            continue
        
        ma5 = df['close'].rolling(5).mean()
        ma20 = df['close'].rolling(20).mean()
        
        # 检查交叉
        if len(ma5) >= 2 and len(ma20) >= 2:
            # 金叉买入
            if ma5.iloc[-2] < ma20.iloc[-2] and ma5.iloc[-1] >= ma20.iloc[-1]:
                signals.append({'code': code, 'action': 'buy', 'shares': 100})
            # 死叉卖出
            elif ma5.iloc[-2] > ma20.iloc[-2] and ma5.iloc[-1] <= ma20.iloc[-1]:
                if code in engine.positions:
                    signals.append({'code': code, 'action': 'sell', 'shares': engine.positions[code].shares})
    
    return signals


if __name__ == "__main__":
    # 测试代码
    from src.data import DataFetcher
    
    # 获取数据
    fetcher = DataFetcher()
    codes = ["000001", "000002", "600000"]
    data = {}
    for code in codes:
        data[code] = fetcher.get_daily_data(code, start_date="2024-01-01")
    
    # 运行回测
    engine = BacktestEngine()
    result = engine.run(data, ma_cross_strategy)
    
    # 输出结果
    print("\n回测结果:")
    for k, v in result.summary().items():
        print(f"  {k}: {v}")
