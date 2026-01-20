"""
A股量化交易系统 - 多因子选股模型

基于多因子的选股策略
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class FactorConfig:
    """因子配置"""
    name: str
    direction: int  # 1: 因子值越大越好, -1: 越小越好
    weight: float = 1.0


class MultiFactorModel:
    """多因子选股模型"""
    
    # 常用因子方向配置
    FACTOR_DIRECTIONS = {
        # 动量因子 - 正向
        'momentum_5': 1,
        'momentum_10': 1,
        'momentum_20': 1,
        'momentum_60': -1,  # 长期动量反转
        
        # 价值因子 - 负向（PE/PB越小越好）
        'pe': -1,
        'pb': -1,
        'ps': -1,
        
        # 质量因子 - 正向
        'roe': 1,
        'roa': 1,
        'gross_margin': 1,
        
        # 波动率因子 - 负向（低波动更好）
        'volatility_10': -1,
        'volatility_20': -1,
        
        # 流动性因子
        'turnover_rate': 1,
        'volume_ratio': 1,
        
        # 技术因子
        'rsi_14': 0,  # 中性，需要特殊处理
    }
    
    def __init__(self, factors: Optional[List[FactorConfig]] = None):
        """
        Args:
            factors: 因子配置列表
        """
        self.factors = factors or []
        self.factor_scores: Dict[str, pd.Series] = {}
    
    def add_factor(self, name: str, direction: int = 1, weight: float = 1.0):
        """添加因子"""
        self.factors.append(FactorConfig(name, direction, weight))
    
    def calculate_factor_score(
        self,
        data: Dict[str, pd.DataFrame],
        factor_name: str,
        direction: int = 1
    ) -> pd.Series:
        """
        计算单个因子的截面得分
        
        Args:
            data: 股票代码 -> DataFrame
            factor_name: 因子名称
            direction: 因子方向
            
        Returns:
            股票代码 -> 因子得分
        """
        # 提取最新因子值
        factor_values = {}
        for code, df in data.items():
            if factor_name in df.columns and len(df) > 0:
                value = df[factor_name].iloc[-1]
                if pd.notna(value) and np.isfinite(value):
                    factor_values[code] = value
        
        if not factor_values:
            return pd.Series()
        
        # 转换为Series
        factor_series = pd.Series(factor_values)
        
        # 去极值 (MAD方法)
        median = factor_series.median()
        mad = (factor_series - median).abs().median()
        upper = median + 5 * 1.4826 * mad
        lower = median - 5 * 1.4826 * mad
        factor_series = factor_series.clip(lower, upper)
        
        # 标准化
        mean = factor_series.mean()
        std = factor_series.std()
        if std > 0:
            factor_series = (factor_series - mean) / std
        
        # 应用方向
        factor_series = factor_series * direction
        
        return factor_series
    
    def calculate_composite_score(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        计算综合因子得分
        
        Args:
            data: 股票代码 -> DataFrame (含因子值)
            
        Returns:
            DataFrame with factor scores and composite score
        """
        if not self.factors:
            raise ValueError("请先添加因子")
        
        scores = {}
        total_weight = sum(f.weight for f in self.factors)
        
        for factor in self.factors:
            direction = factor.direction or self.FACTOR_DIRECTIONS.get(factor.name, 1)
            score = self.calculate_factor_score(data, factor.name, direction)
            
            if not score.empty:
                scores[factor.name] = score * factor.weight
                self.factor_scores[factor.name] = score
        
        if not scores:
            return pd.DataFrame()
        
        # 合并所有因子得分
        score_df = pd.DataFrame(scores)
        
        # 计算综合得分
        score_df['composite_score'] = score_df.sum(axis=1) / total_weight
        
        # 排名
        score_df['rank'] = score_df['composite_score'].rank(ascending=False)
        
        return score_df.sort_values('composite_score', ascending=False)
    
    def select_stocks(
        self,
        data: Dict[str, pd.DataFrame],
        top_n: int = 20,
        exclude_codes: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        选股
        
        Args:
            data: 股票代码 -> DataFrame
            top_n: 选择数量
            exclude_codes: 排除的股票
            
        Returns:
            [(股票代码, 综合得分), ...]
        """
        score_df = self.calculate_composite_score(data)
        
        if score_df.empty:
            return []
        
        # 排除指定股票
        if exclude_codes:
            score_df = score_df.drop(exclude_codes, errors='ignore')
        
        # 选择Top N
        top_stocks = score_df.head(top_n)
        
        return list(zip(top_stocks.index, top_stocks['composite_score']))
    
    def get_factor_exposure(self) -> pd.DataFrame:
        """获取因子暴露报告"""
        if not self.factor_scores:
            return pd.DataFrame()
        
        exposure = pd.DataFrame(self.factor_scores)
        
        # 添加统计信息
        stats = pd.DataFrame({
            'mean': exposure.mean(),
            'std': exposure.std(),
            'min': exposure.min(),
            'max': exposure.max()
        })
        
        return stats


class AlphaFactorModel(MultiFactorModel):
    """Alpha因子模型 - 预设的多因子组合"""
    
    @classmethod
    def momentum_model(cls) -> 'AlphaFactorModel':
        """动量模型"""
        model = cls()
        model.add_factor('momentum_5', direction=1, weight=1.0)
        model.add_factor('momentum_20', direction=1, weight=1.5)
        model.add_factor('volume_ratio', direction=1, weight=0.5)
        return model
    
    @classmethod
    def value_model(cls) -> 'AlphaFactorModel':
        """价值模型"""
        model = cls()
        model.add_factor('pe', direction=-1, weight=1.0)
        model.add_factor('pb', direction=-1, weight=1.0)
        return model
    
    @classmethod
    def quality_model(cls) -> 'AlphaFactorModel':
        """质量模型"""
        model = cls()
        model.add_factor('roe', direction=1, weight=1.5)
        model.add_factor('gross_margin', direction=1, weight=1.0)
        model.add_factor('volatility_20', direction=-1, weight=0.5)
        return model
    
    @classmethod
    def balanced_model(cls) -> 'AlphaFactorModel':
        """均衡模型 - 动量 + 低波动"""
        model = cls()
        model.add_factor('momentum_20', direction=1, weight=1.0)
        model.add_factor('volatility_20', direction=-1, weight=1.0)
        model.add_factor('rsi_14', direction=-1, weight=0.5)  # 超卖更好
        return model


class FactorBacktest:
    """因子回测"""
    
    def __init__(self, model: MultiFactorModel):
        self.model = model
        self.results: List[Dict] = []
    
    def run(
        self,
        data: Dict[str, pd.DataFrame],
        rebalance_freq: int = 20,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        运行因子回测
        
        Args:
            data: 股票代码 -> 完整历史数据
            rebalance_freq: 调仓频率（天）
            top_n: 持仓数量
        """
        # 获取所有交易日
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index.tolist())
        all_dates = sorted(all_dates)
        
        portfolio = {}  # 当前持仓
        results = []
        
        for i, date in enumerate(all_dates):
            # 获取当日数据
            daily_data = {}
            daily_prices = {}
            for code, df in data.items():
                if date in df.index:
                    daily_data[code] = df.loc[:date]
                    daily_prices[code] = df.loc[date, 'close']
            
            # 调仓日
            if i % rebalance_freq == 0:
                try:
                    selected = self.model.select_stocks(daily_data, top_n=top_n)
                    new_portfolio = {code: 1.0/len(selected) for code, _ in selected}
                    portfolio = new_portfolio
                except Exception as e:
                    logger.warning(f"{date} 选股失败: {e}")
            
            # 计算组合收益
            if portfolio:
                portfolio_return = 0.0
                for code, weight in portfolio.items():
                    if code in daily_prices:
                        prev_price = data[code].loc[:date, 'close'].iloc[-2] if len(data[code].loc[:date]) > 1 else daily_prices[code]
                        ret = (daily_prices[code] - prev_price) / prev_price
                        portfolio_return += weight * ret
                
                results.append({
                    'date': date,
                    'return': portfolio_return,
                    'n_stocks': len(portfolio)
                })
        
        self.results = results
        
        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df['cumulative_return'] = (1 + result_df['return']).cumprod() - 1
        
        return result_df


if __name__ == "__main__":
    # 测试代码
    from src.data import DataFetcher
    from src.factors import FactorEngine
    
    # 准备数据
    fetcher = DataFetcher()
    factor_engine = FactorEngine()
    
    codes = ['000001', '000002', '600000', '600036', '601398', '601988', '600519', '000858']
    data = {}
    
    print("获取数据...")
    for code in codes:
        try:
            df = fetcher.get_daily_data(code, start_date='2024-01-01')
            df = factor_engine.compute(df)
            data[code] = df
            print(f"  {code}: {len(df)} 条数据")
        except Exception as e:
            print(f"  {code}: 失败 - {e}")
    
    # 测试多因子选股
    print("\n使用均衡模型选股...")
    model = AlphaFactorModel.balanced_model()
    top_stocks = model.select_stocks(data, top_n=5)
    
    print("Top 5 推荐股票:")
    for code, score in top_stocks:
        print(f"  {code}: {score:.4f}")
    
    # 因子暴露
    print("\n因子暴露:")
    exposure = model.get_factor_exposure()
    print(exposure)
