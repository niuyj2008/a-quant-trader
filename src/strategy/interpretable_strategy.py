"""
股票量化策略决策支持系统 - 可解释策略模块

提供多种交易策略模型，每种策略都生成透明可追溯的决策报告。
支持周频调仓，所有交易信号附带完整的决策理由。

策略列表:
  1. 多因子均衡策略 (BalancedMultiFactorStrategy)
  2. 动量趋势策略 (MomentumTrendStrategy)
  3. 价值投资策略 (ValueInvestStrategy)
  4. 低波动防御策略 (LowVolDefenseStrategy)
  5. 反转策略 (MeanReversionStrategy)
  6. 技术突破策略 (TechnicalBreakoutStrategy)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from loguru import logger
from abc import ABC, abstractmethod

from src.factors.factor_engine import FactorEngine


# ==================== 数据结构 ====================

@dataclass
class DecisionReport:
    """单只股票的决策报告 - 可解释性核心"""
    code: str
    name: str
    date: str
    action: str              # "buy" / "sell" / "hold" / "add" / "reduce"
    action_cn: str           # 中文: "买入" / "卖出" / "持有" / "加仓" / "减仓"
    confidence: float        # 信号强度 0-100
    score: float             # 综合得分
    strategy_name: str       # 策略名称

    # 因子贡献分解
    factor_scores: Dict[str, float] = field(default_factory=dict)       # 因子名 -> 得分
    factor_weights: Dict[str, float] = field(default_factory=dict)      # 因子名 -> 权重
    factor_contributions: Dict[str, float] = field(default_factory=dict) # 因子名 -> 贡献度

    # 决策规则链
    rules_passed: List[str] = field(default_factory=list)    # 通过的规则
    rules_failed: List[str] = field(default_factory=list)    # 未通过的规则

    # 市场分析
    support_price: Optional[float] = None
    resistance_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    target_price: Optional[float] = None
    current_price: Optional[float] = None

    # 文字描述
    summary: str = ""
    reasoning: List[str] = field(default_factory=list)
    risk_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "股票代码": self.code,
            "股票名称": self.name,
            "日期": self.date,
            "操作建议": self.action_cn,
            "信号强度": f"{self.confidence:.0f}/100",
            "综合得分": f"{self.score:.2f}",
            "策略": self.strategy_name,
            "当前价": f"¥{self.current_price:.2f}" if self.current_price else "-",
            "止损价": f"¥{self.stop_loss_price:.2f}" if self.stop_loss_price else "-",
            "支撑位": f"¥{self.support_price:.2f}" if self.support_price else "-",
            "阻力位": f"¥{self.resistance_price:.2f}" if self.resistance_price else "-",
            "理由": self.summary,
        }

    def get_reasoning_text(self) -> str:
        """生成中文决策理由"""
        lines = [f"【{self.strategy_name}】{self.code} {self.name} - {self.action_cn}"]
        lines.append(f"信号强度: {self.confidence:.0f}/100 | 综合得分: {self.score:.2f}")
        if self.reasoning:
            lines.append("决策理由:")
            for r in self.reasoning:
                lines.append(f"  • {r}")
        if self.risk_warnings:
            lines.append("⚠️ 风险提示:")
            for w in self.risk_warnings:
                lines.append(f"  • {w}")
        return "\n".join(lines)


@dataclass
class StockSignal:
    """交易信号"""
    code: str
    action: str         # buy / sell / hold / add / reduce
    confidence: float   # 0-100
    price: float
    shares: int = 0     # 0 = 按仓位管理计算
    report: Optional[DecisionReport] = None


# ==================== 策略基类 ====================

class BaseInterpretableStrategy(ABC):
    """可解释策略基类"""

    # 策略key → 类的映射，子类注册后用于配置加载
    _strategy_key: str = ""

    def __init__(self, name: str, description: str, params: Optional[Dict] = None):
        self.name = name
        self.description = description
        self.params = params or {}
        self.factor_engine = FactorEngine()
        self.config_version = "default"  # 权重版本标识
        # 尝试从配置文件加载优化后的参数
        self._load_config()

    def _load_config(self, config_path: str = "config/strategy_weights.json"):
        """从配置文件加载优化后的参数（权重、阈值、评分区间）

        配置文件由 WeightOptimizer.save_config() 生成。
        如果配置不存在或加载失败，保留硬编码默认值作为fallback。
        """
        import json
        from pathlib import Path

        path = Path(config_path)
        if not path.exists() or not self._strategy_key:
            return

        try:
            with open(path) as f:
                config = json.load(f)

            if self._strategy_key not in config:
                return

            cfg = config[self._strategy_key]

            # 加载权重
            if 'weights' in cfg:
                self.params['optimized_weights'] = cfg['weights']

            # 加载买卖阈值
            if 'buy_threshold' in cfg:
                self.params['buy_threshold'] = cfg['buy_threshold']
            if 'sell_threshold' in cfg:
                self.params['sell_threshold'] = cfg['sell_threshold']

            # 加载评分区间
            if 'score_ranges' in cfg:
                self.params['score_ranges'] = cfg['score_ranges']

            self.config_version = cfg.get('updated_at', 'config')
            logger.debug(f"策略 [{self._strategy_key}] 已加载优化配置 (版本: {self.config_version})")

        except Exception as e:
            logger.debug(f"策略 [{self._strategy_key}] 配置加载失败，使用默认值: {e}")

    def _get_threshold(self, key: str, default: float) -> float:
        """获取阈值，优先使用配置值，fallback到默认值"""
        return self.params.get(key, default)

    def _get_weight(self, factor_name: str, default: float) -> float:
        """获取因子权重，优先使用优化权重"""
        optimized = self.params.get('optimized_weights', {})
        return optimized.get(factor_name, default)

    def _get_score_range(self, factor_name: str, default_low: float,
                         default_high: float) -> tuple:
        """获取评分区间，优先使用自适应区间"""
        ranges = self.params.get('score_ranges', {})
        if factor_name in ranges:
            r = ranges[factor_name]
            if isinstance(r, (list, tuple)) and len(r) == 2:
                return r[0], r[1]
        return default_low, default_high

    @abstractmethod
    def analyze_stock(self, code: str, df: pd.DataFrame,
                      financial_data: Optional[Dict] = None,
                      name: str = "") -> DecisionReport:
        """分析单只股票，生成决策报告"""
        pass

    def analyze_portfolio(self, holdings: Dict[str, Dict],
                          data_dict: Dict[str, pd.DataFrame]) -> List[DecisionReport]:
        """
        分析当前持仓，给出操作建议

        Args:
            holdings: {code: {name, shares, cost_price}}
            data_dict: {code: DataFrame}
        """
        reports = []
        for code, info in holdings.items():
            if code in data_dict:
                report = self.analyze_stock(code, data_dict[code], name=info.get('name', ''))
                # 根据持仓情况调整建议
                report = self._adjust_for_holding(report, info)
                reports.append(report)
        return reports

    def scan_market(self, data_dict: Dict[str, pd.DataFrame],
                    financial_data: Optional[Dict[str, Dict]] = None,
                    top_n: int = 10) -> List[DecisionReport]:
        """
        全市场扫描推荐

        Args:
            data_dict: {code: DataFrame}
            financial_data: {code: {pe, pb, ...}}
            top_n: 推荐数量
        """
        reports = []
        for code, df in data_dict.items():
            try:
                fin = financial_data.get(code) if financial_data else None
                report = self.analyze_stock(code, df, financial_data=fin)
                if report.action in ('buy', 'add') and report.confidence >= 40:
                    reports.append(report)
            except Exception as e:
                logger.debug(f"分析 {code} 失败: {e}")

        # 按综合得分排序
        reports.sort(key=lambda r: r.score, reverse=True)
        return reports[:top_n]

    def _adjust_for_holding(self, report: DecisionReport,
                             holding: Dict) -> DecisionReport:
        """根据已有持仓调整建议"""
        cost_price = holding.get('cost_price', 0)
        if cost_price > 0 and report.current_price:
            pnl_pct = (report.current_price - cost_price) / cost_price

            if report.action == 'buy':
                report.action = 'add'
                report.action_cn = '加仓'
                report.reasoning.append(f"已持仓，当前浮盈{pnl_pct:.1%}，建议加仓")
            elif report.action == 'sell':
                if pnl_pct < -0.08:
                    report.action = 'sell'
                    report.action_cn = '清仓'
                    report.confidence = min(report.confidence + 20, 100)
                    report.reasoning.append(f"浮亏{pnl_pct:.1%}，触发止损线，建议清仓")
                else:
                    report.action = 'reduce'
                    report.action_cn = '减仓'
                    report.reasoning.append(f"信号转弱，建议减仓。当前盈亏: {pnl_pct:.1%}")

            if pnl_pct <= -0.10:
                report.risk_warnings.append(f"⚠️ 浮亏已达 {pnl_pct:.1%}，请注意止损")
        return report

    def _compute_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有核心因子"""
        return self.factor_engine.compute_all_core_factors(df)

    def _score_factor(self, value, low_bad, high_good) -> float:
        """将因子值标准化为0-100分"""
        if pd.isna(value):
            return 50.0
        if high_good > low_bad:
            score = (value - low_bad) / (high_good - low_bad) * 100
        else:
            score = (low_bad - value) / (low_bad - high_good) * 100
        return max(0, min(100, score))


# ==================== 策略1: 多因子均衡策略 ====================

class BalancedMultiFactorStrategy(BaseInterpretableStrategy):
    """
    多因子均衡策略

    综合技术面、动量、波动率等多维因子打分，等权加权。
    适合追求稳健收益的投资者。
    """

    _strategy_key = "balanced"

    def __init__(self):
        super().__init__(
            name="多因子均衡策略",
            description="综合多维因子等权打分，追求稳健收益，适合中长线投资",
            params={"buy_threshold": 65, "sell_threshold": 35}
        )

    def analyze_stock(self, code: str, df: pd.DataFrame,
                      financial_data: Optional[Dict] = None,
                      name: str = "") -> DecisionReport:
        factored = self._compute_factors(df)
        latest = factored.iloc[-1]
        report = DecisionReport(
            code=code, name=name or code,
            date=str(factored.index[-1].date()),
            action="hold", action_cn="持有",
            confidence=50, score=0.0,
            strategy_name=self.name,
            current_price=float(latest['close']),
        )

        # 因子打分
        scores = {}
        weights = {}

        # 动量因子 (20日) — 正向
        scores['动量(20日)'] = self._score_factor(latest.get('momentum_20', 0), -0.15, 0.15)
        weights['动量(20日)'] = 0.15

        # RSI — 中性偏好 (40-60最好)
        rsi_val = latest.get('rsi_14', 50)
        if rsi_val > 70:
            scores['RSI'] = self._score_factor(rsi_val, 100, 50)
        elif rsi_val < 30:
            scores['RSI'] = self._score_factor(rsi_val, 0, 50)
        else:
            scores['RSI'] = 70 + (1 - abs(rsi_val - 50) / 20) * 30
        weights['RSI'] = 0.10

        # 均线趋势
        ma_cross = latest.get('ma_cross', 0)
        scores['均线趋势'] = self._score_factor(ma_cross, -0.05, 0.05)
        weights['均线趋势'] = 0.20

        # 价格位置
        pos = latest.get('price_position', 0.5)
        scores['价格位置'] = self._score_factor(pos, 0, 1) if pos < 0.8 else self._score_factor(pos, 1, 0.5)
        weights['价格位置'] = 0.10

        # MACD
        macd_hist = latest.get('macd_hist', 0)
        scores['MACD'] = self._score_factor(macd_hist, -1, 1)
        weights['MACD'] = 0.15

        # 量能
        vol_ratio = latest.get('volume_ratio', 1)
        scores['量能'] = self._score_factor(vol_ratio, 0.5, 2.0)
        weights['量能'] = 0.10

        # 波动率 — 低波更好
        vol20 = latest.get('volatility_20', 0.03)
        scores['波动率'] = self._score_factor(vol20, 0.06, 0.01)
        weights['波动率'] = 0.10

        # 基本面（如果有数据）
        if financial_data:
            pe = financial_data.get('pe')
            if pe and pe > 0:
                scores['PE估值'] = self._score_factor(pe, 100, 10)
                weights['PE估值'] = 0.10
            else:
                weights['波动率'] += 0.05
                weights['均线趋势'] += 0.05

        # 计算综合得分
        total_weight = sum(weights.values())
        composite_score = sum(scores[k] * weights[k] for k in scores) / total_weight

        # 支撑/阻力
        support = latest.get('support')
        resistance = latest.get('resistance')
        if pd.notna(support):
            report.support_price = float(support)
        if pd.notna(resistance):
            report.resistance_price = float(resistance)
        report.stop_loss_price = report.current_price * 0.92  # 默认8%止损

        # 生成决策
        report.factor_scores = scores
        report.factor_weights = weights
        report.factor_contributions = {k: scores[k] * weights[k] / total_weight
                                       for k in scores}
        report.score = composite_score
        report.confidence = composite_score

        buy_th = self.params['buy_threshold']
        sell_th = self.params['sell_threshold']

        if composite_score >= buy_th:
            report.action = "buy"
            report.action_cn = "买入"
            report.reasoning = self._generate_buy_reasons(scores, buy_th)
        elif composite_score <= sell_th:
            report.action = "sell"
            report.action_cn = "卖出"
            report.reasoning = self._generate_sell_reasons(scores, sell_th)
        else:
            report.action = "hold"
            report.action_cn = "持有"
            report.reasoning = ["综合得分处于中性区间，建议观望"]

        # 风险提示
        if vol20 and vol20 > 0.04:
            report.risk_warnings.append(f"波动率偏高({vol20:.2%})，注意控制仓位")
        if rsi_val and rsi_val > 80:
            report.risk_warnings.append(f"RSI={rsi_val:.0f} 严重超买，谨防回调")
        elif rsi_val and rsi_val < 20:
            report.risk_warnings.append(f"RSI={rsi_val:.0f} 严重超卖，可能继续下探")

        report.summary = f"{self.name}: {report.action_cn}({report.confidence:.0f}分)"
        return report

    def _generate_buy_reasons(self, scores, threshold):
        reasons = []
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        for name, score in top:
            if score >= 60:
                reasons.append(f"{name}表现强势(得分{score:.0f})")
        if not reasons:
            reasons.append(f"综合多因子得分超过买入阈值({threshold})")
        return reasons

    def _generate_sell_reasons(self, scores, threshold):
        reasons = []
        bottom = sorted(scores.items(), key=lambda x: x[1])[:3]
        for name, score in bottom:
            if score <= 40:
                reasons.append(f"{name}表现偏弱(得分{score:.0f})")
        if not reasons:
            reasons.append(f"综合多因子得分低于卖出阈值({threshold})")
        return reasons


# ==================== 策略2: 动量趋势策略 ====================

class MomentumTrendStrategy(BaseInterpretableStrategy):
    """
    动量趋势策略

    跟踪中短期趋势，在趋势确认时入场，趋势减弱时离场。
    适合追求超额收益的投资者。
    """

    _strategy_key = "momentum"

    def __init__(self):
        super().__init__(
            name="动量趋势策略",
            description="跟踪中短期价格趋势，顺势而为，适合追求超额收益",
            params={"momentum_period": 20, "confirm_period": 5,
                    "buy_threshold": 65, "sell_threshold": 35}
        )

    def analyze_stock(self, code: str, df: pd.DataFrame,
                      financial_data: Optional[Dict] = None,
                      name: str = "") -> DecisionReport:
        factored = self._compute_factors(df)
        latest = factored.iloc[-1]
        report = DecisionReport(
            code=code, name=name or code,
            date=str(factored.index[-1].date()),
            action="hold", action_cn="持有",
            confidence=50, score=0.0,
            strategy_name=self.name,
            current_price=float(latest['close']),
        )

        scores = {}
        weights = {}

        # 核心: 多周期动量
        m5 = latest.get('momentum_5', 0)
        m20 = latest.get('momentum_20', 0)
        m60 = latest.get('momentum_60', 0)
        scores['短期动量(5日)'] = self._score_factor(m5, -0.10, 0.10)
        scores['中期动量(20日)'] = self._score_factor(m20, -0.20, 0.20)
        scores['长期动量(60日)'] = self._score_factor(m60, -0.30, 0.30)
        weights['短期动量(5日)'] = 0.20
        weights['中期动量(20日)'] = 0.25
        weights['长期动量(60日)'] = 0.15

        # 趋势确认
        ma_cross = latest.get('ma_cross', 0)
        scores['均线趋势'] = self._score_factor(ma_cross, -0.05, 0.05)
        weights['均线趋势'] = 0.15

        # MACD动能
        macd_hist = latest.get('macd_hist', 0)
        scores['MACD动能'] = self._score_factor(macd_hist, -1, 1)
        weights['MACD动能'] = 0.15

        # 量价配合
        vol_ratio = latest.get('volume_ratio', 1)
        scores['量能配合'] = self._score_factor(vol_ratio, 0.5, 2.5)
        weights['量能配合'] = 0.10

        # 综合得分
        total_w = sum(weights.values())
        composite = sum(scores[k] * weights[k] for k in scores) / total_w

        report.factor_scores = scores
        report.factor_weights = weights
        report.factor_contributions = {k: scores[k] * weights[k] / total_w for k in scores}
        report.score = composite
        report.confidence = composite

        # 趋势一致性检查
        trend_aligned = (m5 > 0 and m20 > 0 and ma_cross > 0)
        trend_broken = (m5 < 0 and m20 < -0.05)

        buy_th = self._get_threshold('buy_threshold', 65)
        sell_th = self._get_threshold('sell_threshold', 35)

        if composite >= buy_th and trend_aligned:
            report.action = "buy"
            report.action_cn = "买入"
            report.reasoning = [
                f"多周期动量一致向上(5日:{m5:.1%}, 20日:{m20:.1%})",
                f"均线多头排列确认趋势",
            ]
        elif composite <= sell_th or trend_broken:
            report.action = "sell"
            report.action_cn = "卖出"
            report.reasoning = ["趋势动量减弱或反转，建议离场"]
        else:
            report.action = "hold"
            report.action_cn = "持有"
            report.reasoning = ["趋势信号不明确，建议观望"]

        # 风险
        if m5 > 0.15:
            report.risk_warnings.append("短期涨幅过大，注意获利回吐风险")

        report.stop_loss_price = report.current_price * 0.90
        support = latest.get('support')
        resistance = latest.get('resistance')
        if pd.notna(support):
            report.support_price = float(support)
        if pd.notna(resistance):
            report.resistance_price = float(resistance)

        report.summary = f"{self.name}: {report.action_cn}({report.confidence:.0f}分)"
        return report


# ==================== 策略3: 价值投资策略 ====================

class ValueInvestStrategy(BaseInterpretableStrategy):
    """
    价值投资策略

    聚焦低估值、高质量的公司，在低估时买入，高估时卖出。
    适合长期持有、注重安全边际的投资者。
    """

    _strategy_key = "value"

    def __init__(self):
        super().__init__(
            name="价值投资策略",
            description="聚焦低估值高质量公司，注重安全边际，适合长线持有",
            params={"buy_threshold": 65, "sell_threshold": 35}
        )

    def analyze_stock(self, code: str, df: pd.DataFrame,
                      financial_data: Optional[Dict] = None,
                      name: str = "") -> DecisionReport:
        factored = self._compute_factors(df)
        latest = factored.iloc[-1]
        report = DecisionReport(
            code=code, name=name or code,
            date=str(factored.index[-1].date()),
            action="hold", action_cn="持有",
            confidence=50, score=0.0,
            strategy_name=self.name,
            current_price=float(latest['close']),
        )

        scores = {}
        weights = {}

        # 基本面权重加大
        if financial_data:
            pe = financial_data.get('pe')
            pb = financial_data.get('pb')
            roe = financial_data.get('roe')
            rev_growth = financial_data.get('revenue_growth')
            gross_margin = financial_data.get('gross_margin')

            if pe and pe > 0:
                scores['PE估值'] = self._score_factor(pe, 80, 8)
                weights['PE估值'] = 0.20
            if pb and pb > 0:
                scores['PB估值'] = self._score_factor(pb, 8, 0.5)
                weights['PB估值'] = 0.15
            if roe:
                scores['ROE盈利能力'] = self._score_factor(roe, 0, 25)
                weights['ROE盈利能力'] = 0.20
            if rev_growth:
                scores['营收增长'] = self._score_factor(rev_growth, -0.1, 0.3)
                weights['营收增长'] = 0.10
            if gross_margin:
                scores['毛利率'] = self._score_factor(gross_margin, 0, 50)
                weights['毛利率'] = 0.10

        # 技术面辅助
        vol20 = latest.get('volatility_20', 0.03)
        scores['波动率(低更好)'] = self._score_factor(vol20, 0.06, 0.01)
        weights['波动率(低更好)'] = 0.10

        pos = latest.get('price_position', 0.5)
        scores['价格位置(低更好)'] = self._score_factor(pos, 1, 0)
        weights['价格位置(低更好)'] = 0.15

        # 如果没有基本面数据，增加技术面权重
        if not financial_data or not scores.get('PE估值'):
            weights['波动率(低更好)'] = 0.30
            weights['价格位置(低更好)'] = 0.30
            scores['动量(反转)'] = self._score_factor(latest.get('momentum_20', 0), 0.10, -0.10)
            weights['动量(反转)'] = 0.40

        total_w = sum(weights.values())
        composite = sum(scores[k] * weights[k] for k in scores) / total_w if total_w > 0 else 50

        report.factor_scores = scores
        report.factor_weights = weights
        report.factor_contributions = {k: scores[k] * weights[k] / total_w for k in scores}
        report.score = composite
        report.confidence = composite

        buy_th = self._get_threshold('buy_threshold', 65)
        sell_th = self._get_threshold('sell_threshold', 35)

        if composite >= buy_th:
            report.action = "buy"
            report.action_cn = "买入"
            report.reasoning = []
            if scores.get('PE估值', 0) >= 70:
                report.reasoning.append(f"估值合理偏低 (PE得分{scores['PE估值']:.0f})")
            if scores.get('ROE盈利能力', 0) >= 70:
                report.reasoning.append(f"盈利能力优秀 (ROE得分{scores['ROE盈利能力']:.0f})")
            if not report.reasoning:
                report.reasoning.append("综合价值评估显示当前处于低估区间")
        elif composite <= sell_th:
            report.action = "sell"
            report.action_cn = "卖出"
            report.reasoning = ["估值偏高或基本面转弱，建议卖出"]
        else:
            report.action = "hold"
            report.action_cn = "持有"
            report.reasoning = ["估值中性，无明显低估机会"]

        report.stop_loss_price = report.current_price * 0.90
        report.summary = f"{self.name}: {report.action_cn}({report.confidence:.0f}分)"
        return report


# ==================== 策略4: 低波动防御策略 ====================

class LowVolDefenseStrategy(BaseInterpretableStrategy):
    """
    低波动防御策略

    选择低波动、下行风险小的股票构建组合。
    适合风险厌恶型投资者，追求稳健收益。
    """

    _strategy_key = "low_vol"

    def __init__(self):
        super().__init__(
            name="低波动防御策略",
            description="选择低波动标的，控制回撤，适合风险厌恶型投资者",
            params={"buy_threshold": 65, "sell_threshold": 35}
        )

    def analyze_stock(self, code: str, df: pd.DataFrame,
                      financial_data: Optional[Dict] = None,
                      name: str = "") -> DecisionReport:
        factored = self._compute_factors(df)
        latest = factored.iloc[-1]
        report = DecisionReport(
            code=code, name=name or code,
            date=str(factored.index[-1].date()),
            action="hold", action_cn="持有",
            confidence=50, score=0.0,
            strategy_name=self.name,
            current_price=float(latest['close']),
        )

        scores = {}
        weights = {}

        # 波动率 — 越低越好
        vol20 = latest.get('volatility_20', 0.03)
        scores['波动率'] = self._score_factor(vol20, 0.05, 0.008)
        weights['波动率'] = 0.25

        # 下行波动率
        dv = latest.get('downside_vol', 0.02)
        scores['下行风险'] = self._score_factor(dv if pd.notna(dv) else 0.02, 0.04, 0.005)
        weights['下行风险'] = 0.20

        # ATR
        atr_val = latest.get('atr_14', 0)
        if pd.notna(atr_val) and report.current_price and report.current_price > 0:
            atr_pct = atr_val / report.current_price
            scores['ATR波幅'] = self._score_factor(atr_pct, 0.05, 0.005)
        else:
            scores['ATR波幅'] = 50
        weights['ATR波幅'] = 0.15

        # 趋势稳定性
        ma_cross = latest.get('ma_cross', 0)
        scores['趋势稳定性'] = self._score_factor(abs(ma_cross), 0.05, 0)
        weights['趋势稳定性'] = 0.15

        # 正向微弱动量 — 不追涨
        m20 = latest.get('momentum_20', 0)
        if 0 <= m20 <= 0.05:
            scores['温和动量'] = 80
        elif m20 < 0:
            scores['温和动量'] = max(30, 50 + m20 * 200)
        else:
            scores['温和动量'] = max(30, 80 - (m20 - 0.05) * 300)
        weights['温和动量'] = 0.15

        # 成交量
        vol_r = latest.get('volume_ratio', 1)
        scores['量能(温和更好)'] = 80 if 0.7 <= vol_r <= 1.5 else 40
        weights['量能(温和更好)'] = 0.10

        total_w = sum(weights.values())
        composite = sum(scores[k] * weights[k] for k in scores) / total_w

        report.factor_scores = scores
        report.factor_weights = weights
        report.factor_contributions = {k: scores[k] * weights[k] / total_w for k in scores}
        report.score = composite
        report.confidence = composite

        buy_th = self._get_threshold('buy_threshold', 65)
        sell_th = self._get_threshold('sell_threshold', 35)

        if composite >= buy_th:
            report.action = "buy"
            report.action_cn = "买入"
            report.reasoning = [f"波动率低({vol20:.2%})，风险可控",
                                f"下行风险小，适合防御配置"]
        elif composite <= sell_th:
            report.action = "sell"
            report.action_cn = "卖出"
            report.reasoning = ["波动率升高或趋势转弱，不再适合防御配置"]
        else:
            report.action = "hold"
            report.action_cn = "持有"
            report.reasoning = ["波动率中等，继续观察"]

        report.stop_loss_price = report.current_price * 0.95  # 5%止损
        report.summary = f"{self.name}: {report.action_cn}({report.confidence:.0f}分)"
        return report


# ==================== 策略5: 均值回归策略 ====================

class MeanReversionStrategy(BaseInterpretableStrategy):
    """
    均值回归(反转)策略

    在超卖时买入，超买时卖出。逆向思维。
    适合震荡市，有较高的纪律要求。
    """

    _strategy_key = "reversion"

    def __init__(self):
        super().__init__(
            name="均值回归策略",
            description="逆向投资，超卖买入超买卖出，适合震荡市",
            params={"buy_threshold": 70, "sell_threshold": 30}
        )

    def analyze_stock(self, code: str, df: pd.DataFrame,
                      financial_data: Optional[Dict] = None,
                      name: str = "") -> DecisionReport:
        factored = self._compute_factors(df)
        latest = factored.iloc[-1]
        report = DecisionReport(
            code=code, name=name or code,
            date=str(factored.index[-1].date()),
            action="hold", action_cn="持有",
            confidence=50, score=0.0,
            strategy_name=self.name,
            current_price=float(latest['close']),
        )

        scores = {}
        weights = {}

        # RSI — 越低越好（反转）
        rsi = latest.get('rsi_14', 50)
        if rsi < 30:
            scores['RSI超卖'] = 90 - rsi  # 越低分越高
        elif rsi > 70:
            scores['RSI超卖'] = max(0, 70 - (rsi - 70) * 3)
        else:
            scores['RSI超卖'] = 50
        weights['RSI超卖'] = 0.25

        # 价格偏离度 — 偏离均线越多越好（回归机会）
        ma_cross = latest.get('ma_cross', 0)
        scores['均线偏离度'] = self._score_factor(ma_cross, 0.05, -0.08)  # 低于均线=高分
        weights['均线偏离度'] = 0.20

        # 布林带位置 — 越接近下轨越好
        pos = latest.get('price_position', 0.5)
        scores['价格位置(低好)'] = self._score_factor(pos, 0.9, 0.1)
        weights['价格位置(低好)'] = 0.15

        # 短期跌幅 — 短期跌越多, 反弹概率越高
        m5 = latest.get('momentum_5', 0)
        scores['短期回调幅度'] = self._score_factor(m5, 0.05, -0.10)
        weights['短期回调幅度'] = 0.20

        # 成交量缩量 — 缩量下跌是好信号
        vol_r = latest.get('volume_ratio', 1)
        if m5 < 0 and vol_r < 0.8:
            scores['缩量下跌'] = 80
        else:
            scores['缩量下跌'] = 50
        weights['缩量下跌'] = 0.10

        # 波动率
        vol20 = latest.get('volatility_20', 0.03)
        scores['波动率'] = self._score_factor(vol20, 0.06, 0.01)
        weights['波动率'] = 0.10

        total_w = sum(weights.values())
        composite = sum(scores[k] * weights[k] for k in scores) / total_w

        report.factor_scores = scores
        report.factor_weights = weights
        report.factor_contributions = {k: scores[k] * weights[k] / total_w for k in scores}
        report.score = composite
        report.confidence = composite

        buy_th = self._get_threshold('buy_threshold', 70)
        sell_th = self._get_threshold('sell_threshold', 30)

        if composite >= buy_th and rsi < 35:
            report.action = "buy"
            report.action_cn = "买入"
            report.reasoning = [f"RSI严重超卖({rsi:.0f})，均值回归概率高",
                                f"价格位于近期底部区域(位置{pos:.0%})"]
        elif composite <= sell_th or rsi > 75:
            report.action = "sell"
            report.action_cn = "卖出"
            report.reasoning = [f"RSI超买({rsi:.0f})或价格过高，反转风险大"]
        else:
            report.action = "hold"
            report.action_cn = "持有"
            report.reasoning = ["未到明确的超买超卖区域，建议观望"]

        report.risk_warnings.append("反转策略风险较高，建议轻仓操作")
        report.stop_loss_price = report.current_price * 0.92
        report.summary = f"{self.name}: {report.action_cn}({report.confidence:.0f}分)"
        return report


# ==================== 策略6: 技术突破策略 ====================

class TechnicalBreakoutStrategy(BaseInterpretableStrategy):
    """
    技术突破策略

    捕捉价格突破关键阻力位/支撑位的机会，配合放量确认。
    适合短中线操作的投资者。
    """

    _strategy_key = "breakout"

    def __init__(self):
        super().__init__(
            name="技术突破策略",
            description="捕捉价格突破和放量确认信号，适合中短线操作",
            params={"buy_threshold": 70, "sell_threshold": 35}
        )

    def analyze_stock(self, code: str, df: pd.DataFrame,
                      financial_data: Optional[Dict] = None,
                      name: str = "") -> DecisionReport:
        factored = self._compute_factors(df)
        latest = factored.iloc[-1]
        report = DecisionReport(
            code=code, name=name or code,
            date=str(factored.index[-1].date()),
            action="hold", action_cn="持有",
            confidence=50, score=0.0,
            strategy_name=self.name,
            current_price=float(latest['close']),
        )

        scores = {}
        weights = {}

        # 突破信号: 接近或突破阻力位
        resistance = latest.get('resistance')
        support = latest.get('support')
        close = latest['close']

        if pd.notna(resistance) and resistance > 0:
            dist_to_resistance = (resistance - close) / close
            if dist_to_resistance <= 0:  # 已突破
                scores['阻力突破'] = 90
            elif dist_to_resistance < 0.02:  # 接近
                scores['阻力突破'] = 70
            else:
                scores['阻力突破'] = max(20, 50 - dist_to_resistance * 300)
            report.resistance_price = float(resistance)
        else:
            scores['阻力突破'] = 50
        weights['阻力突破'] = 0.25

        if pd.notna(support):
            report.support_price = float(support)

        # 放量确认
        vol_ratio = latest.get('volume_ratio', 1)
        vol_ma_r = latest.get('volume_ma_ratio', 1)
        if vol_ratio > 1.5 and vol_ma_r > 1.3:
            scores['放量确认'] = 85
        elif vol_ratio > 1.2:
            scores['放量确认'] = 65
        else:
            scores['放量确认'] = 40
        weights['放量确认'] = 0.20

        # 动量配合
        m5 = latest.get('momentum_5', 0)
        scores['短期动量'] = self._score_factor(m5, -0.05, 0.08)
        weights['短期动量'] = 0.15

        # MACD金叉
        macd_hist = latest.get('macd_hist', 0)
        scores['MACD信号'] = self._score_factor(macd_hist, -0.5, 0.5)
        weights['MACD信号'] = 0.15

        # 均线支撑
        ma_cross = latest.get('ma_cross', 0)
        scores['均线支撑'] = self._score_factor(ma_cross, -0.03, 0.03)
        weights['均线支撑'] = 0.15

        # RSI不能过高
        rsi = latest.get('rsi_14', 50)
        scores['RSI空间'] = self._score_factor(rsi, 85, 40)
        weights['RSI空间'] = 0.10

        total_w = sum(weights.values())
        composite = sum(scores[k] * weights[k] for k in scores) / total_w

        report.factor_scores = scores
        report.factor_weights = weights
        report.factor_contributions = {k: scores[k] * weights[k] / total_w for k in scores}
        report.score = composite
        report.confidence = composite

        buy_th = self._get_threshold('buy_threshold', 70)
        sell_th = self._get_threshold('sell_threshold', 35)

        if composite >= buy_th and scores['阻力突破'] >= 70 and scores['放量确认'] >= 60:
            report.action = "buy"
            report.action_cn = "买入"
            report.reasoning = [
                "价格突破或接近关键阻力位",
                f"成交量放大确认(量比{vol_ratio:.1f})",
            ]
        elif composite <= sell_th:
            report.action = "sell"
            report.action_cn = "卖出"
            report.reasoning = ["突破失败或跌破支撑位"]
        else:
            report.action = "hold"
            report.action_cn = "持有"
            report.reasoning = ["等待突破信号确认"]

        report.stop_loss_price = report.support_price if report.support_price else report.current_price * 0.93
        report.summary = f"{self.name}: {report.action_cn}({report.confidence:.0f}分)"
        return report


# ==================== 策略注册表 ====================

STRATEGY_REGISTRY: Dict[str, type] = {
    "balanced": BalancedMultiFactorStrategy,
    "momentum": MomentumTrendStrategy,
    "value": ValueInvestStrategy,
    "low_vol": LowVolDefenseStrategy,
    "reversion": MeanReversionStrategy,
    "breakout": TechnicalBreakoutStrategy,
}

STRATEGY_NAMES = {
    "balanced": "多因子均衡策略",
    "momentum": "动量趋势策略",
    "value": "价值投资策略",
    "low_vol": "低波动防御策略",
    "reversion": "均值回归策略",
    "breakout": "技术突破策略",
}

STRATEGY_DESCRIPTIONS = {
    "balanced": "综合多维因子等权打分，追求稳健收益，适合中长线投资",
    "momentum": "跟踪中短期价格趋势，顺势而为，适合追求超额收益",
    "value": "聚焦低估值高质量公司，注重安全边际，适合长线持有",
    "low_vol": "选择低波动标的，控制回撤，适合风险厌恶型投资者",
    "reversion": "逆向投资，超卖买入超买卖出，适合震荡市",
    "breakout": "捕捉价格突破和放量确认信号，适合中短线操作",
}

STRATEGY_RISK_LEVELS = {
    "balanced": "中等",
    "momentum": "中高",
    "value": "低",
    "low_vol": "低",
    "reversion": "高",
    "breakout": "中高",
}


def get_strategy(name: str) -> BaseInterpretableStrategy:
    """获取策略实例"""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"未知策略: {name}，可选: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name]()


def get_all_strategies() -> Dict[str, BaseInterpretableStrategy]:
    """获取所有策略实例"""
    return {k: v() for k, v in STRATEGY_REGISTRY.items()}


def multi_strategy_analysis(code: str, df: pd.DataFrame,
                            financial_data: Optional[Dict] = None,
                            name: str = "") -> Dict[str, DecisionReport]:
    """
    使用所有策略分析同一只股票

    Returns:
        Dict[strategy_key -> DecisionReport]
    """
    results = {}
    for key, strategy_cls in STRATEGY_REGISTRY.items():
        try:
            strategy = strategy_cls()
            results[key] = strategy.analyze_stock(code, df, financial_data, name)
        except Exception as e:
            logger.error(f"策略 {key} 分析 {code} 失败: {e}")
    return results
