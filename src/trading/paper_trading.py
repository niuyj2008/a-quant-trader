"""
A股量化交易系统 - 模拟交易模块

提供策略的模拟执行环境
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from loguru import logger
import json
from pathlib import Path


@dataclass
class Order:
    """订单"""
    order_id: str
    code: str
    direction: str  # "buy" or "sell"
    price: float
    shares: int
    order_type: str = "limit"  # "limit" or "market"
    status: str = "pending"    # "pending", "filled", "cancelled", "rejected"
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_shares: int = 0
    message: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'order_id': self.order_id,
            'code': self.code,
            'direction': self.direction,
            'price': self.price,
            'shares': self.shares,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'filled_price': self.filled_price,
            'filled_shares': self.filled_shares,
        }


@dataclass
class Position:
    """持仓"""
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
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost


class PaperTradingEngine:
    """模拟交易引擎"""
    
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission_rate: float = 0.0003,
        stamp_duty: float = 0.001,
        slippage: float = 0.001,
        data_dir: str = "data/paper_trading"
    ):
        """
        Args:
            initial_capital: 初始资金
            commission_rate: 佣金费率
            stamp_duty: 印花税 (卖出时)
            slippage: 滑点
            data_dir: 数据存储目录
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.stamp_duty = stamp_duty
        self.slippage = slippage
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.order_counter = 0
        
        self.equity_history: List[Dict] = []
        self.trade_history: List[Dict] = []
    
    @property
    def total_equity(self) -> float:
        """总权益"""
        position_value = sum(p.market_value for p in self.positions.values())
        return self.cash + position_value
    
    @property
    def position_value(self) -> float:
        """持仓市值"""
        return sum(p.market_value for p in self.positions.values())
    
    def _generate_order_id(self) -> str:
        """生成订单ID"""
        self.order_counter += 1
        return f"ORD{datetime.now().strftime('%Y%m%d')}{self.order_counter:06d}"
    
    def submit_order(
        self,
        code: str,
        direction: str,
        shares: int,
        price: Optional[float] = None,
        order_type: str = "market"
    ) -> Order:
        """
        提交订单
        
        Args:
            code: 股票代码
            direction: "buy" or "sell"
            shares: 数量
            price: 限价 (市价单可不填)
            order_type: "limit" or "market"
        """
        order = Order(
            order_id=self._generate_order_id(),
            code=code,
            direction=direction,
            price=price or 0,
            shares=shares,
            order_type=order_type
        )
        
        self.orders.append(order)
        logger.info(f"订单提交: {order.order_id} {direction} {code} {shares}股")
        
        return order
    
    def execute_orders(self, prices: Dict[str, float]):
        """
        执行待处理订单
        
        Args:
            prices: 当前价格 {code: price}
        """
        for order in self.orders:
            if order.status != "pending":
                continue
            
            if order.code not in prices:
                order.status = "rejected"
                order.message = "没有价格数据"
                continue
            
            current_price = prices[order.code]
            
            # 计算成交价 (加入滑点)
            if order.direction == "buy":
                fill_price = current_price * (1 + self.slippage)
            else:
                fill_price = current_price * (1 - self.slippage)
            
            # 限价单检查
            if order.order_type == "limit":
                if order.direction == "buy" and fill_price > order.price:
                    continue  # 买单价格高于限价，不成交
                if order.direction == "sell" and fill_price < order.price:
                    continue  # 卖单价格低于限价，不成交
            
            # 执行订单
            if order.direction == "buy":
                success = self._execute_buy(order, fill_price)
            else:
                success = self._execute_sell(order, fill_price)
            
            if success:
                order.status = "filled"
                order.filled_at = datetime.now()
                order.filled_price = fill_price
                order.filled_shares = order.shares
    
    def _execute_buy(self, order: Order, price: float) -> bool:
        """执行买入"""
        amount = price * order.shares
        commission = max(amount * self.commission_rate, 5)
        total_cost = amount + commission
        
        if total_cost > self.cash:
            order.status = "rejected"
            order.message = f"资金不足: 需要 {total_cost:.2f}, 可用 {self.cash:.2f}"
            return False
        
        self.cash -= total_cost
        
        if order.code in self.positions:
            pos = self.positions[order.code]
            total_shares = pos.shares + order.shares
            pos.avg_cost = (pos.avg_cost * pos.shares + price * order.shares) / total_shares
            pos.shares = total_shares
        else:
            self.positions[order.code] = Position(
                code=order.code,
                shares=order.shares,
                avg_cost=price,
                current_price=price
            )
        
        # 记录交易
        self.trade_history.append({
            'time': datetime.now().isoformat(),
            'order_id': order.order_id,
            'code': order.code,
            'direction': 'buy',
            'price': price,
            'shares': order.shares,
            'amount': amount,
            'commission': commission
        })
        
        logger.info(f"买入成交: {order.code} {order.shares}股 @ {price:.2f}")
        return True
    
    def _execute_sell(self, order: Order, price: float) -> bool:
        """执行卖出"""
        if order.code not in self.positions:
            order.status = "rejected"
            order.message = "没有持仓"
            return False
        
        pos = self.positions[order.code]
        if pos.shares < order.shares:
            order.status = "rejected"
            order.message = f"持仓不足: 持有 {pos.shares}, 卖出 {order.shares}"
            return False
        
        amount = price * order.shares
        commission = max(amount * self.commission_rate, 5)
        stamp = amount * self.stamp_duty
        total_received = amount - commission - stamp
        
        self.cash += total_received
        
        pos.shares -= order.shares
        if pos.shares == 0:
            del self.positions[order.code]
        
        # 记录交易
        self.trade_history.append({
            'time': datetime.now().isoformat(),
            'order_id': order.order_id,
            'code': order.code,
            'direction': 'sell',
            'price': price,
            'shares': order.shares,
            'amount': amount,
            'commission': commission,
            'stamp_duty': stamp
        })
        
        logger.info(f"卖出成交: {order.code} {order.shares}股 @ {price:.2f}")
        return True
    
    def update_prices(self, prices: Dict[str, float]):
        """更新持仓价格"""
        for code, pos in self.positions.items():
            if code in prices:
                pos.current_price = prices[code]
        
        # 记录权益
        self.equity_history.append({
            'time': datetime.now().isoformat(),
            'cash': self.cash,
            'position_value': self.position_value,
            'total_equity': self.total_equity
        })
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        for order in self.orders:
            if order.order_id == order_id and order.status == "pending":
                order.status = "cancelled"
                return True
        return False
    
    def get_position_summary(self) -> pd.DataFrame:
        """获取持仓汇总"""
        if not self.positions:
            return pd.DataFrame()
        
        data = []
        for pos in self.positions.values():
            data.append({
                '股票代码': pos.code,
                '持仓数量': pos.shares,
                '成本价': f"¥{pos.avg_cost:.2f}",
                '现价': f"¥{pos.current_price:.2f}",
                '市值': f"¥{pos.market_value:.0f}",
                '浮动盈亏': f"¥{pos.profit:.0f}",
                '盈亏比例': f"{pos.profit_pct:.2%}"
            })
        
        return pd.DataFrame(data)
    
    def get_account_summary(self) -> Dict:
        """获取账户汇总"""
        return {
            '初始资金': f"¥{self.initial_capital:,.0f}",
            '可用现金': f"¥{self.cash:,.0f}",
            '持仓市值': f"¥{self.position_value:,.0f}",
            '总权益': f"¥{self.total_equity:,.0f}",
            '总收益率': f"{(self.total_equity / self.initial_capital - 1):.2%}",
            '持仓数量': len(self.positions),
            '今日成交': len([o for o in self.orders if o.status == 'filled'])
        }
    
    def save_state(self):
        """保存状态"""
        state = {
            'cash': self.cash,
            'positions': {code: {'shares': p.shares, 'avg_cost': p.avg_cost} 
                         for code, p in self.positions.items()},
            'equity_history': self.equity_history[-100:],  # 保留最近100条
            'trade_history': self.trade_history[-100:]
        }
        
        with open(self.data_dir / 'state.json', 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self):
        """加载状态"""
        state_file = self.data_dir / 'state.json'
        if not state_file.exists():
            return
        
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        self.cash = state.get('cash', self.initial_capital)
        
        for code, pos_data in state.get('positions', {}).items():
            self.positions[code] = Position(
                code=code,
                shares=pos_data['shares'],
                avg_cost=pos_data['avg_cost']
            )
        
        self.equity_history = state.get('equity_history', [])
        self.trade_history = state.get('trade_history', [])


if __name__ == "__main__":
    # 测试代码
    engine = PaperTradingEngine(initial_capital=100000)
    
    # 模拟价格
    prices = {
        '000001': 10.5,
        '600000': 8.2,
        '600036': 35.0
    }
    
    # 买入
    order1 = engine.submit_order('000001', 'buy', 1000)
    order2 = engine.submit_order('600000', 'buy', 500)
    
    # 执行订单
    engine.execute_orders(prices)
    
    # 更新价格
    prices['000001'] = 11.0
    prices['600000'] = 8.5
    engine.update_prices(prices)
    
    # 查看账户
    print("账户汇总:")
    for k, v in engine.get_account_summary().items():
        print(f"  {k}: {v}")
    
    print("\n持仓明细:")
    print(engine.get_position_summary())
    
    # 卖出
    order3 = engine.submit_order('000001', 'sell', 500)
    engine.execute_orders(prices)
    
    print("\n卖出后账户:")
    for k, v in engine.get_account_summary().items():
        print(f"  {k}: {v}")
