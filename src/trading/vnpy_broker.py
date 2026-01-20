"""
A股量化交易系统 - VeighNa券商接口集成

提供与VeighNa (vnpy) 交易框架的集成，支持实盘交易
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from loguru import logger
from abc import ABC, abstractmethod

# VeighNa导入（可选依赖）
try:
    from vnpy.event import EventEngine
    from vnpy.trader.engine import MainEngine
    from vnpy.trader.object import (
        TickData, BarData, OrderData, TradeData,
        PositionData, AccountData, ContractData,
        OrderRequest, CancelRequest, SubscribeRequest
    )
    from vnpy.trader.constant import (
        Direction, Offset, OrderType, Exchange, Status
    )
    VNPY_AVAILABLE = True
except ImportError:
    VNPY_AVAILABLE = False
    logger.warning("VeighNa (vnpy) 未安装，实盘交易功能不可用")
    logger.info("安装命令: pip install vnpy vnpy_ctp vnpy_xtp")


@dataclass
class BrokerConfig:
    """券商配置"""
    broker_type: str = "xtp"  # 券商类型: xtp(中泰), ctp(期货)
    user_id: str = ""
    password: str = ""
    td_address: str = ""  # 交易服务器地址
    md_address: str = ""  # 行情服务器地址
    auth_code: str = ""
    app_id: str = ""
    
    # XTP特有配置
    client_id: int = 1
    software_key: str = ""
    
    # 日志
    log_level: str = "INFO"


@dataclass
class OrderInfo:
    """订单信息"""
    order_id: str
    code: str
    exchange: str
    direction: str
    offset: str
    price: float
    volume: int
    traded: int = 0
    status: str = "pending"
    create_time: datetime = field(default_factory=datetime.now)
    update_time: Optional[datetime] = None
    error_msg: str = ""


class BaseBroker(ABC):
    """券商接口基类"""
    
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.connected = False
        self.positions: Dict[str, float] = {}
        self.orders: Dict[str, OrderInfo] = {}
        
        # 回调函数
        self.on_tick: Optional[Callable] = None
        self.on_bar: Optional[Callable] = None
        self.on_order: Optional[Callable] = None
        self.on_trade: Optional[Callable] = None
        self.on_position: Optional[Callable] = None
    
    @abstractmethod
    def connect(self) -> bool:
        """连接券商"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """断开连接"""
        pass
    
    @abstractmethod
    def subscribe(self, codes: List[str]):
        """订阅行情"""
        pass
    
    @abstractmethod
    def buy(self, code: str, price: float, volume: int) -> str:
        """买入"""
        pass
    
    @abstractmethod
    def sell(self, code: str, price: float, volume: int) -> str:
        """卖出"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        pass
    
    @abstractmethod
    def query_position(self) -> Dict[str, float]:
        """查询持仓"""
        pass
    
    @abstractmethod
    def query_account(self) -> Dict:
        """查询账户"""
        pass


class VnpyBroker(BaseBroker):
    """VeighNa券商接口实现"""
    
    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        
        if not VNPY_AVAILABLE:
            raise ImportError("VeighNa未安装，请运行: pip install vnpy")
        
        self.event_engine = None
        self.main_engine = None
        self.gateway_name = ""
    
    def connect(self) -> bool:
        """连接券商"""
        try:
            # 创建事件引擎
            self.event_engine = EventEngine()
            self.main_engine = MainEngine(self.event_engine)
            
            # 根据券商类型加载网关
            if self.config.broker_type == "xtp":
                self._connect_xtp()
            elif self.config.broker_type == "ctp":
                self._connect_ctp()
            else:
                raise ValueError(f"不支持的券商类型: {self.config.broker_type}")
            
            self.connected = True
            logger.info(f"券商连接成功: {self.config.broker_type}")
            return True
            
        except Exception as e:
            logger.error(f"券商连接失败: {e}")
            return False
    
    def _connect_xtp(self):
        """连接中泰XTP"""
        try:
            from vnpy_xtp import XtpGateway
            self.main_engine.add_gateway(XtpGateway)
            self.gateway_name = "XTP"
            
            setting = {
                "账号": self.config.user_id,
                "密码": self.config.password,
                "客户号": self.config.client_id,
                "行情地址": self.config.md_address,
                "行情端口": 0,
                "交易地址": self.config.td_address,
                "交易端口": 0,
                "行情协议": "TCP",
                "授权码": self.config.software_key,
            }
            
            self.main_engine.connect(setting, self.gateway_name)
            
        except ImportError:
            raise ImportError("vnpy_xtp未安装，请运行: pip install vnpy_xtp")
    
    def _connect_ctp(self):
        """连接CTP (期货)"""
        try:
            from vnpy_ctp import CtpGateway
            self.main_engine.add_gateway(CtpGateway)
            self.gateway_name = "CTP"
            
            setting = {
                "用户名": self.config.user_id,
                "密码": self.config.password,
                "经纪商代码": "",
                "交易服务器": self.config.td_address,
                "行情服务器": self.config.md_address,
                "产品名称": self.config.app_id,
                "授权编码": self.config.auth_code,
            }
            
            self.main_engine.connect(setting, self.gateway_name)
            
        except ImportError:
            raise ImportError("vnpy_ctp未安装，请运行: pip install vnpy_ctp")
    
    def disconnect(self):
        """断开连接"""
        if self.main_engine:
            self.main_engine.close()
        self.connected = False
        logger.info("券商已断开连接")
    
    def subscribe(self, codes: List[str]):
        """订阅行情"""
        if not self.connected:
            logger.warning("未连接券商")
            return
        
        for code in codes:
            # 解析代码和交易所
            exchange = self._get_exchange(code)
            
            req = SubscribeRequest(
                symbol=code,
                exchange=exchange
            )
            self.main_engine.subscribe(req, self.gateway_name)
        
        logger.info(f"订阅行情: {codes}")
    
    def _get_exchange(self, code: str) -> Exchange:
        """根据代码获取交易所"""
        if code.startswith("6"):
            return Exchange.SSE  # 上海
        elif code.startswith("0") or code.startswith("3"):
            return Exchange.SZSE  # 深圳
        else:
            return Exchange.SSE
    
    def buy(self, code: str, price: float, volume: int) -> str:
        """买入"""
        return self._send_order(code, Direction.LONG, Offset.OPEN, price, volume)
    
    def sell(self, code: str, price: float, volume: int) -> str:
        """卖出"""
        return self._send_order(code, Direction.SHORT, Offset.CLOSE, price, volume)
    
    def _send_order(
        self,
        code: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: int
    ) -> str:
        """发送订单"""
        if not self.connected:
            logger.warning("未连接券商")
            return ""
        
        exchange = self._get_exchange(code)
        
        req = OrderRequest(
            symbol=code,
            exchange=exchange,
            direction=direction,
            offset=offset,
            type=OrderType.LIMIT,
            price=price,
            volume=volume,
        )
        
        order_id = self.main_engine.send_order(req, self.gateway_name)
        
        if order_id:
            self.orders[order_id] = OrderInfo(
                order_id=order_id,
                code=code,
                exchange=exchange.value,
                direction=direction.value,
                offset=offset.value,
                price=price,
                volume=volume
            )
            logger.info(f"订单已发送: {order_id}")
        
        return order_id or ""
    
    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        if order_id not in self.orders:
            return False
        
        order_info = self.orders[order_id]
        
        req = CancelRequest(
            orderid=order_id,
            symbol=order_info.code,
            exchange=Exchange(order_info.exchange)
        )
        
        self.main_engine.cancel_order(req, self.gateway_name)
        logger.info(f"撤单请求: {order_id}")
        return True
    
    def query_position(self) -> Dict[str, float]:
        """查询持仓"""
        positions = {}
        
        if self.main_engine:
            all_positions = self.main_engine.get_all_positions()
            for pos in all_positions:
                if pos.volume > 0:
                    positions[pos.symbol] = pos.volume
        
        self.positions = positions
        return positions
    
    def query_account(self) -> Dict:
        """查询账户"""
        if not self.main_engine:
            return {}
        
        accounts = self.main_engine.get_all_accounts()
        if accounts:
            acc = accounts[0]
            return {
                "账户ID": acc.accountid,
                "总资产": acc.balance,
                "可用资金": acc.available,
                "冻结资金": acc.frozen,
            }
        return {}


class SimulatedBroker(BaseBroker):
    """模拟券商 - 用于测试"""
    
    def __init__(self, config: BrokerConfig, initial_capital: float = 1_000_000):
        super().__init__(config)
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.positions: Dict[str, Dict] = {}  # code -> {shares, cost}
        self.current_prices: Dict[str, float] = {}
    
    def connect(self) -> bool:
        self.connected = True
        logger.info("模拟券商已连接")
        return True
    
    def disconnect(self):
        self.connected = False
        logger.info("模拟券商已断开")
    
    def subscribe(self, codes: List[str]):
        logger.info(f"模拟订阅: {codes}")
    
    def update_prices(self, prices: Dict[str, float]):
        """更新价格（模拟行情）"""
        self.current_prices.update(prices)
    
    def buy(self, code: str, price: float, volume: int) -> str:
        """买入"""
        amount = price * volume
        commission = max(amount * 0.0003, 5)
        total_cost = amount + commission
        
        if total_cost > self.cash:
            logger.warning(f"资金不足: 需要 {total_cost:.2f}, 可用 {self.cash:.2f}")
            return ""
        
        self.cash -= total_cost
        
        if code in self.positions:
            pos = self.positions[code]
            total_shares = pos['shares'] + volume
            pos['cost'] = (pos['cost'] * pos['shares'] + price * volume) / total_shares
            pos['shares'] = total_shares
        else:
            self.positions[code] = {'shares': volume, 'cost': price}
        
        order_id = f"SIM{datetime.now().strftime('%H%M%S%f')}"
        logger.info(f"模拟买入: {code} {volume}股 @ {price:.2f}")
        return order_id
    
    def sell(self, code: str, price: float, volume: int) -> str:
        """卖出"""
        if code not in self.positions or self.positions[code]['shares'] < volume:
            logger.warning(f"持仓不足")
            return ""
        
        amount = price * volume
        commission = max(amount * 0.0003, 5)
        stamp_duty = amount * 0.001
        net_amount = amount - commission - stamp_duty
        
        self.cash += net_amount
        self.positions[code]['shares'] -= volume
        
        if self.positions[code]['shares'] == 0:
            del self.positions[code]
        
        order_id = f"SIM{datetime.now().strftime('%H%M%S%f')}"
        logger.info(f"模拟卖出: {code} {volume}股 @ {price:.2f}")
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        logger.info(f"模拟撤单: {order_id}")
        return True
    
    def query_position(self) -> Dict[str, float]:
        return {code: pos['shares'] for code, pos in self.positions.items()}
    
    def query_account(self) -> Dict:
        position_value = sum(
            pos['shares'] * self.current_prices.get(code, pos['cost'])
            for code, pos in self.positions.items()
        )
        total = self.cash + position_value
        
        return {
            "总资产": f"¥{total:,.0f}",
            "可用资金": f"¥{self.cash:,.0f}",
            "持仓市值": f"¥{position_value:,.0f}",
            "收益率": f"{(total / self.initial_capital - 1):.2%}"
        }


class LiveTrader:
    """实盘交易执行器"""
    
    def __init__(
        self,
        broker: BaseBroker,
        strategy_func: Callable,
        codes: List[str]
    ):
        """
        Args:
            broker: 券商接口
            strategy_func: 策略函数，签名为 (prices, positions) -> signals
            codes: 关注的股票列表
        """
        self.broker = broker
        self.strategy_func = strategy_func
        self.codes = codes
        self.running = False
    
    def start(self):
        """启动交易"""
        if not self.broker.connect():
            logger.error("券商连接失败")
            return
        
        self.broker.subscribe(self.codes)
        self.running = True
        logger.info("实盘交易已启动")
    
    def stop(self):
        """停止交易"""
        self.running = False
        self.broker.disconnect()
        logger.info("实盘交易已停止")
    
    def on_tick(self, prices: Dict[str, float]):
        """行情回调"""
        if not self.running:
            return
        
        # 获取当前持仓
        positions = self.broker.query_position()
        
        # 执行策略
        signals = self.strategy_func(prices, positions)
        
        # 执行交易信号
        for signal in signals:
            code = signal['code']
            action = signal['action']
            volume = signal.get('volume', 100)
            price = prices.get(code, 0)
            
            if action == 'buy' and price > 0:
                self.broker.buy(code, price, volume)
            elif action == 'sell' and price > 0:
                self.broker.sell(code, price, volume)


# 便捷函数
def create_broker(broker_type: str = "simulated", **kwargs) -> BaseBroker:
    """
    创建券商接口
    
    Args:
        broker_type: "simulated", "xtp", "ctp"
    """
    config = BrokerConfig(broker_type=broker_type, **kwargs)
    
    if broker_type == "simulated":
        return SimulatedBroker(config)
    else:
        return VnpyBroker(config)


if __name__ == "__main__":
    # 测试模拟券商
    broker = create_broker("simulated")
    broker.connect()
    
    # 模拟交易
    broker.buy("000001", 10.5, 1000)
    broker.buy("600000", 8.2, 500)
    
    # 更新价格
    broker.update_prices({"000001": 11.0, "600000": 8.5})
    
    # 查询账户
    print("账户信息:")
    for k, v in broker.query_account().items():
        print(f"  {k}: {v}")
    
    # 卖出
    broker.sell("000001", 11.0, 500)
    
    print("\n卖出后:")
    for k, v in broker.query_account().items():
        print(f"  {k}: {v}")
    
    broker.disconnect()
