"""
A股量化交易系统 - 风险管理模块

提供风险控制和仓位管理功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from loguru import logger


@dataclass
class RiskConfig:
    """风险配置"""
    # 仓位限制
    max_position_pct: float = 0.1       # 单只股票最大仓位 10%
    max_sector_pct: float = 0.3         # 单个行业最大仓位 30%
    max_total_positions: int = 20       # 最大持仓数量
    
    # 止损止盈
    stop_loss_pct: float = -0.1         # 止损 -10%
    take_profit_pct: float = 0.3        # 止盈 30%
    trailing_stop_pct: float = 0.05     # 移动止损 5%
    
    # 风险指标阈值
    max_volatility: float = 0.5         # 最大波动率
    max_drawdown: float = -0.2          # 最大允许回撤 -20%
    max_var: float = 0.05               # VaR阈值 5%


@dataclass
class PositionRisk:
    """持仓风险"""
    code: str
    shares: int
    cost: float
    current_price: float
    market_value: float
    weight: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    
    # 风险信号
    stop_loss_triggered: bool = False
    take_profit_triggered: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "股票代码": self.code,
            "持仓数量": self.shares,
            "成本价": f"¥{self.cost:.2f}",
            "现价": f"¥{self.current_price:.2f}",
            "市值": f"¥{self.market_value:.0f}",
            "仓位": f"{self.weight:.1%}",
            "浮动盈亏": f"¥{self.unrealized_pnl:.0f}",
            "盈亏比例": f"{self.unrealized_pnl_pct:.2%}",
        }


class RiskManager:
    """风险管理器"""
    
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.position_risks: Dict[str, PositionRisk] = {}
        self.alerts: List[str] = []
    
    def calculate_position_risk(
        self,
        positions: Dict[str, Tuple[int, float]],  # code -> (shares, cost)
        prices: Dict[str, float],
        total_equity: float
    ) -> Dict[str, PositionRisk]:
        """
        计算持仓风险
        
        Args:
            positions: 持仓信息 {code: (shares, cost)}
            prices: 当前价格 {code: price}
            total_equity: 总权益
        """
        self.position_risks = {}
        self.alerts = []
        
        for code, (shares, cost) in positions.items():
            current_price = prices.get(code, cost)
            market_value = shares * current_price
            weight = market_value / total_equity if total_equity > 0 else 0
            unrealized_pnl = (current_price - cost) * shares
            unrealized_pnl_pct = (current_price - cost) / cost if cost > 0 else 0
            
            risk = PositionRisk(
                code=code,
                shares=shares,
                cost=cost,
                current_price=current_price,
                market_value=market_value,
                weight=weight,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct
            )
            
            # 检查止损止盈
            if unrealized_pnl_pct <= self.config.stop_loss_pct:
                risk.stop_loss_triggered = True
                self.alerts.append(f"⚠️ {code} 触发止损: {unrealized_pnl_pct:.2%}")
            
            if unrealized_pnl_pct >= self.config.take_profit_pct:
                risk.take_profit_triggered = True
                self.alerts.append(f"🎯 {code} 触发止盈: {unrealized_pnl_pct:.2%}")
            
            # 检查仓位限制
            if weight > self.config.max_position_pct:
                self.alerts.append(f"⚠️ {code} 超过单只仓位限制: {weight:.1%}")
            
            self.position_risks[code] = risk
        
        return self.position_risks
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        计算VaR (Value at Risk)
        
        Args:
            returns: 收益率序列
            confidence: 置信度
        """
        if len(returns) == 0:
            return 0.0
        
        var = np.percentile(returns, (1 - confidence) * 100)
        return var
    
    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        计算CVaR (Conditional VaR)
        """
        var = self.calculate_var(returns, confidence)
        cvar = returns[returns <= var].mean()
        return cvar if not np.isnan(cvar) else var
    
    def calculate_portfolio_risk(
        self,
        equity_curve: pd.Series
    ) -> Dict:
        """
        计算组合风险指标
        """
        if len(equity_curve) < 2:
            return {}
        
        returns = equity_curve.pct_change().dropna()
        
        # 年化波动率
        volatility = returns.std() * np.sqrt(252)
        
        # 最大回撤
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # VaR 和 CVaR
        var_95 = self.calculate_var(returns, 0.95)
        cvar_95 = self.calculate_cvar(returns, 0.95)
        
        # 检查风险阈值
        if volatility > self.config.max_volatility:
            self.alerts.append(f"⚠️ 波动率超标: {volatility:.2%}")
        
        if max_drawdown < self.config.max_drawdown:
            self.alerts.append(f"🚨 回撤超标: {max_drawdown:.2%}")
        
        return {
            "年化波动率": f"{volatility:.2%}",
            "最大回撤": f"{max_drawdown:.2%}",
            "VaR(95%)": f"{var_95:.2%}",
            "CVaR(95%)": f"{cvar_95:.2%}",
            "日均收益": f"{returns.mean():.4%}",
            "收益标准差": f"{returns.std():.4%}",
        }
    
    def check_trade_risk(
        self,
        code: str,
        action: str,
        shares: int,
        price: float,
        current_positions: Dict[str, Tuple[int, float]],
        total_equity: float
    ) -> Tuple[bool, str]:
        """
        检查交易风险
        
        Returns:
            (是否允许交易, 原因)
        """
        if action == "buy":
            # 计算交易后仓位
            trade_value = shares * price
            new_weight = trade_value / total_equity if total_equity > 0 else 0
            
            # 检查单只仓位限制
            if code in current_positions:
                existing_shares, existing_cost = current_positions[code]
                total_value = existing_shares * price + trade_value
                new_weight = total_value / total_equity
            
            if new_weight > self.config.max_position_pct:
                return False, f"超过单只仓位限制 ({self.config.max_position_pct:.0%})"
            
            # 检查持仓数量限制
            if code not in current_positions and len(current_positions) >= self.config.max_total_positions:
                return False, f"超过最大持仓数量 ({self.config.max_total_positions})"
        
        return True, "通过"
    
    def get_stop_loss_orders(self) -> List[Dict]:
        """获取需要执行的止损订单"""
        orders = []
        for code, risk in self.position_risks.items():
            if risk.stop_loss_triggered:
                orders.append({
                    'code': code,
                    'action': 'sell',
                    'shares': risk.shares,
                    'reason': f'止损触发 ({risk.unrealized_pnl_pct:.2%})'
                })
            elif risk.take_profit_triggered:
                # 止盈可以选择部分卖出
                orders.append({
                    'code': code,
                    'action': 'sell',
                    'shares': risk.shares // 2,  # 卖出一半
                    'reason': f'止盈触发 ({risk.unrealized_pnl_pct:.2%})'
                })
        return orders
    
    def get_risk_report(self) -> pd.DataFrame:
        """生成风险报告"""
        if not self.position_risks:
            return pd.DataFrame()
        
        data = [risk.to_dict() for risk in self.position_risks.values()]
        return pd.DataFrame(data)


class PositionSizer:
    """仓位管理器"""
    
    def __init__(self, method: str = "equal"):
        """
        Args:
            method: 仓位计算方法
                - "equal": 等权重
                - "risk_parity": 风险平价
                - "kelly": 凯利公式
        """
        self.method = method
    
    def calculate_weights(
        self,
        codes: List[str],
        returns_data: Optional[Dict[str, pd.Series]] = None,
        win_rates: Optional[Dict[str, float]] = None,
        odds: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        计算目标权重
        
        Args:
            codes: 股票代码列表
            returns_data: 收益率数据 (风险平价需要)
            win_rates: 胜率 (凯利公式需要)
            odds: 赔率 (凯利公式需要)
        """
        n = len(codes)
        if n == 0:
            return {}
        
        if self.method == "equal":
            weight = 1.0 / n
            return {code: weight for code in codes}
        
        elif self.method == "risk_parity" and returns_data:
            return self._risk_parity_weights(codes, returns_data)
        
        elif self.method == "kelly" and win_rates and odds:
            return self._kelly_weights(codes, win_rates, odds)
        
        else:
            # 默认等权重
            weight = 1.0 / n
            return {code: weight for code in codes}
    
    def _risk_parity_weights(
        self,
        codes: List[str],
        returns_data: Dict[str, pd.Series]
    ) -> Dict[str, float]:
        """风险平价权重"""
        volatilities = {}
        for code in codes:
            if code in returns_data and len(returns_data[code]) > 0:
                vol = returns_data[code].std() * np.sqrt(252)
                volatilities[code] = vol if vol > 0 else 0.01
        
        if not volatilities:
            return {code: 1.0/len(codes) for code in codes}
        
        # 风险平价: 权重反比于波动率
        inv_vol = {code: 1.0/vol for code, vol in volatilities.items()}
        total_inv_vol = sum(inv_vol.values())
        
        weights = {code: v/total_inv_vol for code, v in inv_vol.items()}
        
        # 补充未计算的股票
        for code in codes:
            if code not in weights:
                weights[code] = 0.0
        
        return weights
    
    def _kelly_weights(
        self,
        codes: List[str],
        win_rates: Dict[str, float],
        odds: Dict[str, float]
    ) -> Dict[str, float]:
        """凯利公式权重"""
        weights = {}
        
        for code in codes:
            p = win_rates.get(code, 0.5)  # 胜率
            b = odds.get(code, 1.0)       # 赔率
            
            # Kelly = (bp - q) / b = (bp - (1-p)) / b
            kelly = (b * p - (1 - p)) / b
            
            # 限制在0-0.25之间 (使用1/4 Kelly)
            kelly = max(0, min(kelly * 0.25, 0.25))
            weights[code] = kelly
        
        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {code: w/total for code, w in weights.items()}
        
        return weights
    
    def calculate_shares(
        self,
        weights: Dict[str, float],
        prices: Dict[str, float],
        total_capital: float,
        lot_size: int = 100
    ) -> Dict[str, int]:
        """
        计算买入股数
        
        Args:
            weights: 目标权重
            prices: 当前价格
            total_capital: 可用资金
            lot_size: 最小买入单位 (A股为100)
        """
        shares = {}
        
        for code, weight in weights.items():
            if code not in prices or weight <= 0:
                continue
            
            target_value = total_capital * weight
            target_shares = int(target_value / prices[code] / lot_size) * lot_size
            
            if target_shares >= lot_size:
                shares[code] = target_shares
        
        return shares


class ATRStopLoss:
    """波动率自适应止损

    替代固定百分比止损，根据个股ATR动态调整止损距离。
    波动大的股票给予更宽的止损空间，波动小的收紧止损。

    用法:
        atr_sl = ATRStopLoss(multiplier=2.0)
        stop_price = atr_sl.calculate_stop(df, entry_price=100.0)
    """

    def __init__(self, multiplier: float = 2.0, atr_period: int = 14):
        """
        Args:
            multiplier: ATR 倍数，越大止损越宽（推荐1.5-3.0）
            atr_period: ATR计算周期
        """
        self.multiplier = multiplier
        self.atr_period = atr_period

    def calculate_atr(self, df: pd.DataFrame) -> float:
        """计算 Average True Range"""
        if len(df) < self.atr_period + 1:
            return 0.0

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        atr = pd.Series(tr).rolling(self.atr_period).mean().iloc[-1]
        return float(atr) if not np.isnan(atr) else 0.0

    def calculate_stop(self, df: pd.DataFrame, entry_price: float,
                       direction: str = "long") -> Optional[float]:
        """计算自适应止损价

        Args:
            df: 日线数据(需含 high/low/close)
            entry_price: 买入价
            direction: "long"(做多) 或 "short"(做空)

        Returns:
            止损价格，数据不足返回 None
        """
        atr = self.calculate_atr(df)
        if atr <= 0:
            return None

        stop_distance = atr * self.multiplier

        if direction == "long":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def calculate_trailing_stop(self, df: pd.DataFrame,
                                highest_since_entry: float) -> Optional[float]:
        """计算移动止损价（基于ATR）

        Args:
            df: 日线数据
            highest_since_entry: 入场后最高价

        Returns:
            移动止损价格
        """
        atr = self.calculate_atr(df)
        if atr <= 0:
            return None
        return highest_since_entry - atr * self.multiplier

    def get_stop_info(self, df: pd.DataFrame, entry_price: float) -> Dict:
        """获取完整止损信息"""
        atr = self.calculate_atr(df)
        current_price = float(df['close'].iloc[-1])

        if atr <= 0:
            return {
                'atr': 0,
                'stop_price': entry_price * 0.92,  # 降级到8%固定止损
                'stop_pct': -0.08,
                'method': 'fixed_fallback',
            }

        stop_price = entry_price - atr * self.multiplier
        stop_pct = (stop_price - entry_price) / entry_price

        return {
            'atr': round(atr, 4),
            'atr_pct': round(atr / current_price, 4),
            'multiplier': self.multiplier,
            'stop_price': round(stop_price, 2),
            'stop_pct': round(stop_pct, 4),
            'method': f'ATR×{self.multiplier}',
        }


class CorrelationMonitor:
    """持仓相关性监控

    监控组合中各持仓的收益率相关性，
    高相关性意味着分散化不足，增加系统性风险。
    """

    def __init__(self, warning_threshold: float = 0.8,
                 lookback: int = 60):
        """
        Args:
            warning_threshold: 相关性预警阈值
            lookback: 相关性计算窗口（交易日）
        """
        self.warning_threshold = warning_threshold
        self.lookback = lookback

    def calculate_correlation_matrix(
        self, price_data: Dict[str, pd.Series]
    ) -> Optional[pd.DataFrame]:
        """计算持仓之间的收益率相关性矩阵

        Args:
            price_data: {股票代码: 收盘价Series}

        Returns:
            相关性矩阵 DataFrame
        """
        if len(price_data) < 2:
            return None

        returns_dict = {}
        for code, prices in price_data.items():
            if len(prices) >= self.lookback:
                ret = prices.iloc[-self.lookback:].pct_change().dropna()
                returns_dict[code] = ret

        if len(returns_dict) < 2:
            return None

        returns_df = pd.DataFrame(returns_dict)
        return returns_df.corr()

    def get_high_correlation_pairs(
        self, price_data: Dict[str, pd.Series]
    ) -> List[Tuple[str, str, float]]:
        """找出高相关性持仓对

        Returns:
            [(股票A, 股票B, 相关系数), ...] 按相关性降序
        """
        corr_matrix = self.calculate_correlation_matrix(price_data)
        if corr_matrix is None:
            return []

        pairs = []
        codes = corr_matrix.columns.tolist()
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= self.warning_threshold:
                    pairs.append((codes[i], codes[j], round(corr, 3)))

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs

    def get_diversification_score(
        self, price_data: Dict[str, pd.Series]
    ) -> float:
        """计算组合分散化得分

        Returns:
            0-100, 高分=分散化好
        """
        corr_matrix = self.calculate_correlation_matrix(price_data)
        if corr_matrix is None:
            return 50.0  # 无法计算，返回中性

        n = len(corr_matrix)
        if n < 2:
            return 50.0

        # 取上三角矩阵的平均相关系数
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        avg_corr = corr_matrix.where(mask).stack().mean()

        # 映射: avg_corr=0 → 100分, avg_corr=1 → 0分
        score = (1 - avg_corr) * 100
        return max(0, min(100, float(score)))

    def generate_warnings(
        self, price_data: Dict[str, pd.Series]
    ) -> List[str]:
        """生成相关性预警"""
        warnings = []
        high_pairs = self.get_high_correlation_pairs(price_data)

        for code_a, code_b, corr in high_pairs:
            warnings.append(
                f"持仓 {code_a} 与 {code_b} 高度相关(r={corr:.2f})，分散化不足"
            )

        div_score = self.get_diversification_score(price_data)
        if div_score < 30:
            warnings.append(f"组合分散化得分偏低({div_score:.0f}/100)，建议增加非相关资产")

        return warnings


class BlackSwanDetector:
    """黑天鹅检测器

    监控指数级别的极端波动，当触发条件时发出紧急风控信号。
    基于历史收益率分布的 3-sigma 规则。
    """

    def __init__(self, sigma_threshold: float = 3.0,
                 lookback: int = 252):
        """
        Args:
            sigma_threshold: 触发阈值（标准差倍数）
            lookback: 用于计算均值/标准差的历史窗口
        """
        self.sigma_threshold = sigma_threshold
        self.lookback = lookback

    def check(self, index_data: pd.DataFrame) -> Dict:
        """检查是否触发黑天鹅预警

        Args:
            index_data: 指数日线数据（需含 close 列）

        Returns:
            {
                'triggered': bool,
                'severity': str,      # "none"/"warning"/"critical"
                'daily_return': float,
                'z_score': float,
                'message': str,
            }
        """
        result = {
            'triggered': False,
            'severity': 'none',
            'daily_return': 0.0,
            'z_score': 0.0,
            'message': '',
        }

        if index_data is None or len(index_data) < self.lookback + 1:
            return result

        close = index_data['close'].values
        returns = np.diff(np.log(close))

        if len(returns) < self.lookback:
            return result

        # 用历史窗口计算均值和标准差
        hist_returns = returns[-(self.lookback + 1):-1]
        mean_ret = np.mean(hist_returns)
        std_ret = np.std(hist_returns)

        if std_ret == 0:
            return result

        # 今日收益率
        today_return = float(returns[-1])
        z_score = (today_return - mean_ret) / std_ret

        result['daily_return'] = round(today_return, 4)
        result['z_score'] = round(z_score, 2)

        # 判断严重程度
        abs_z = abs(z_score)
        if abs_z >= self.sigma_threshold * 1.5:
            result['triggered'] = True
            result['severity'] = 'critical'
            result['message'] = (
                f"极端波动警报: 指数日收益率{today_return:.2%} "
                f"(Z={z_score:.1f}, 超过{self.sigma_threshold * 1.5:.1f}sigma)"
            )
        elif abs_z >= self.sigma_threshold:
            result['triggered'] = True
            result['severity'] = 'warning'
            result['message'] = (
                f"异常波动预警: 指数日收益率{today_return:.2%} "
                f"(Z={z_score:.1f}, 超过{self.sigma_threshold:.1f}sigma)"
            )

        return result

    def get_emergency_actions(self, severity: str) -> List[str]:
        """获取紧急风控动作建议

        Args:
            severity: "warning" 或 "critical"

        Returns:
            建议动作列表
        """
        if severity == 'critical':
            return [
                "暂停所有买入信号",
                "触发全组合止损检查",
                "高杠杆持仓强制减仓50%",
                "发送紧急通知",
            ]
        elif severity == 'warning':
            return [
                "买入信号强度降低50%",
                "新建仓位规模减半",
                "检查高波动持仓的止损位",
            ]
        return []


if __name__ == "__main__":
    # 测试代码
    config = RiskConfig(
        max_position_pct=0.15,
        stop_loss_pct=-0.08,
        take_profit_pct=0.25
    )
    
    risk_manager = RiskManager(config)
    
    # 模拟持仓
    positions = {
        '000001': (1000, 10.5),
        '600000': (500, 8.2),
        '600036': (200, 35.0),
    }
    
    prices = {
        '000001': 9.8,   # 亏损
        '600000': 10.5,  # 盈利
        '600036': 38.0,  # 盈利
    }
    
    total_equity = 100000
    
    # 计算持仓风险
    risks = risk_manager.calculate_position_risk(positions, prices, total_equity)
    
    print("持仓风险报告:")
    report = risk_manager.get_risk_report()
    print(report)
    
    print("\n风险告警:")
    for alert in risk_manager.alerts:
        print(f"  {alert}")
    
    # 仓位管理
    print("\n仓位管理测试:")
    sizer = PositionSizer(method="equal")
    codes = ['000001', '600000', '600036']
    weights = sizer.calculate_weights(codes)
    shares = sizer.calculate_shares(weights, prices, 100000)
    
    print(f"目标权重: {weights}")
    print(f"建议买入股数: {shares}")
