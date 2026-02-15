"""
业界标杆因子 - Phase 9.4

基于学术研究和业界公认的有效因子
参考文献:
1. Fama & French (1993) - "Common Risk Factors in Returns on Stocks and Bonds"
2. Jegadeesh & Titman (1993) - "Returns to Buying Winners and Selling Losers"
3. Novy-Marx (2013) - "The Quality Dimension of Value Investing"
4. Ang et al. (2006) - "The Cross-Section of Volatility and Expected Returns"
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger


class AcademicFactors:
    """
    学术界/业界公认的有效因子

    包含:
    1. Fama-French三因子 (SMB, HML, MKT)
    2. 动量因子 (Momentum)
    3. 质量因子 (Quality)
    4. 低波动率异常 (Low Volatility Anomaly)
    5. 盈利能力因子 (Profitability)
    """

    def __init__(self):
        logger.info("学术因子库初始化")

    # ========== Fama-French三因子 ==========

    def calculate_market_factor(self, stock_return: float,
                               market_return: float,
                               risk_free_rate: float = 0.03) -> float:
        """
        市场因子 (MKT)

        定义: 市场组合相对无风险利率的超额收益

        Args:
            stock_return: 股票收益率
            market_return: 市场收益率(如沪深300)
            risk_free_rate: 无风险利率(年化,如0.03=3%)

        Returns:
            市场超额收益
        """
        mkt = market_return - risk_free_rate / 252  # 转日频
        return mkt

    def calculate_smb_factor(self, market_cap: float,
                            market_cap_percentiles: Dict[str, float]) -> float:
        """
        规模因子 (SMB - Small Minus Big)

        定义: 小盘股收益 - 大盘股收益

        原理: 小盘股长期跑赢大盘股

        Args:
            market_cap: 当前股票市值(亿元)
            market_cap_percentiles: 市场市值分位数
                {'30%': 50, '70%': 200}  # 30分位=50亿, 70分位=200亿

        Returns:
            SMB因子值
            - >0: 小盘股特征(市值在前30%)
            - <0: 大盘股特征(市值在后30%)
            - 0: 中盘股
        """
        if market_cap < market_cap_percentiles['30%']:
            # 小盘股
            smb = 1.0
        elif market_cap > market_cap_percentiles['70%']:
            # 大盘股
            smb = -1.0
        else:
            # 中盘股
            smb = 0.0

        return smb

    def calculate_hml_factor(self, pb_ratio: float,
                            pb_percentiles: Dict[str, float]) -> float:
        """
        价值因子 (HML - High Minus Low)

        定义: 高账面市值比(低PB)股票收益 - 低账面市值比(高PB)股票收益

        原理: 价值股(低PB)长期跑赢成长股(高PB)

        Args:
            pb_ratio: 市净率(PB)
            pb_percentiles: PB分位数
                {'30%': 2.0, '70%': 5.0}

        Returns:
            HML因子值
            - >0: 价值股特征(PB低于30分位)
            - <0: 成长股特征(PB高于70分位)
        """
        if pb_ratio < pb_percentiles['30%']:
            # 低PB = 价值股
            hml = 1.0
        elif pb_ratio > pb_percentiles['70%']:
            # 高PB = 成长股
            hml = -1.0
        else:
            hml = 0.0

        return hml

    def calculate_fama_french_three_factors(self,
                                           stock_data: pd.Series,
                                           market_data: pd.DataFrame) -> Dict[str, float]:
        """
        一次性计算Fama-French三因子

        Args:
            stock_data: 股票数据
                - 'return': 收益率
                - 'market_cap': 市值(亿元)
                - 'pb': 市净率
            market_data: 市场数据(用于计算分位数)
                - columns: ['market_cap', 'pb', 'return']

        Returns:
            {
                'MKT': 市场因子,
                'SMB': 规模因子,
                'HML': 价值因子,
            }
        """
        # 计算市值和PB的分位数
        market_cap_30 = market_data['market_cap'].quantile(0.3)
        market_cap_70 = market_data['market_cap'].quantile(0.7)

        pb_30 = market_data['pb'].quantile(0.3)
        pb_70 = market_data['pb'].quantile(0.7)

        # 计算三因子
        mkt = self.calculate_market_factor(
            stock_data['return'],
            market_data['return'].mean(),
        )

        smb = self.calculate_smb_factor(
            stock_data['market_cap'],
            {'30%': market_cap_30, '70%': market_cap_70}
        )

        hml = self.calculate_hml_factor(
            stock_data['pb'],
            {'30%': pb_30, '70%': pb_70}
        )

        return {
            'MKT': mkt,
            'SMB': smb,
            'HML': hml,
            'ff3_score': smb + hml,  # 综合得分
        }

    # ========== 动量因子 ==========

    def calculate_momentum_factor(self, df: pd.DataFrame,
                                  lookback: int = 252,
                                  skip: int = 21) -> pd.Series:
        """
        动量因子 (Jegadeesh & Titman, 1993)

        标准定义: 过去12个月收益(跳过最近1个月)

        原理: 过去表现好的股票倾向于继续表现好(短期惯性)

        Args:
            df: 历史数据(必须包含close列)
            lookback: 回看期(默认252=12个月)
            skip: 跳过最近N天(默认21=1个月)

        Returns:
            动量因子序列
        """
        # 计算lookback期收益率
        momentum = df['close'].pct_change(lookback)

        # 跳过最近skip天(避免短期反转)
        momentum = momentum.shift(skip)

        return momentum

    def calculate_reversal_factor(self, df: pd.DataFrame,
                                  lookback: int = 21) -> pd.Series:
        """
        短期反转因子

        定义: 过去1个月收益的负值

        原理: 短期内股价倾向于反转(超跌反弹、超涨回调)

        Args:
            df: 历史数据
            lookback: 回看期(默认21=1个月)

        Returns:
            反转因子序列(负号)
        """
        reversal = -df['close'].pct_change(lookback)
        return reversal

    # ========== 质量因子 ==========

    def calculate_quality_factor(self, financial_data: Dict) -> float:
        """
        质量因子 (Novy-Marx, 2013)

        定义: 高盈利能力 + 低资产增长率 + 稳定盈利

        原理: 高质量公司(高ROE、低扩张、稳定盈利)长期表现优异

        Args:
            financial_data: 财务数据
                - 'roe': ROE(净资产收益率)
                - 'roe_std': ROE标准差(3年)
                - 'asset_growth': 资产增长率(YoY)
                - 'gross_margin': 毛利率

        Returns:
            质量得分(0-100)
        """
        roe = financial_data.get('roe', 0)
        roe_std = financial_data.get('roe_std', 0)
        asset_growth = financial_data.get('asset_growth', 0)
        gross_margin = financial_data.get('gross_margin', 0)

        # 1. 盈利能力(0-40分)
        # ROE 15%=30分, 20%=40分
        profitability_score = min(roe * 200, 40)

        # 2. 盈利稳定性(0-30分)
        # ROE标准差<0.03=满分30, >0.10=0分
        if roe_std < 0.03:
            stability_score = 30
        elif roe_std > 0.10:
            stability_score = 0
        else:
            stability_score = 30 - (roe_std - 0.03) / 0.07 * 30

        # 3. 资产增长(0-30分)
        # 最优增长率5-15%
        if 0.05 <= asset_growth <= 0.15:
            growth_score = 30
        elif asset_growth < 0.05:
            growth_score = 20  # 增长过慢
        elif asset_growth > 0.30:
            growth_score = 0   # 激进扩张
        else:
            # 0.15-0.30之间线性递减
            growth_score = 30 - (asset_growth - 0.15) / 0.15 * 30

        quality_score = profitability_score + stability_score + growth_score

        return min(quality_score, 100)

    def calculate_profitability_factor(self, gross_profit: float,
                                      total_assets: float) -> float:
        """
        盈利能力因子 (Gross Profitability)

        定义: 毛利润 / 总资产

        原理: 单位资产创造的毛利润越高,公司盈利能力越强

        Args:
            gross_profit: 毛利润
            total_assets: 总资产

        Returns:
            盈利能力因子
        """
        if total_assets == 0:
            return 0

        profitability = gross_profit / total_assets

        return profitability

    # ========== 低波动率异常 ==========

    def calculate_low_volatility_factor(self, df: pd.DataFrame,
                                       period: int = 60) -> pd.Series:
        """
        低波动率因子 (Ang et al., 2006)

        原理: 低波动股票长期表现优于高波动股票(违反有效市场假说)

        Args:
            df: 历史数据
            period: 计算波动率的窗口期(默认60=3个月)

        Returns:
            低波动率因子(负号,波动率越低分数越高)
        """
        returns = df['close'].pct_change()
        volatility = returns.rolling(period).std()

        # 取负号: 波动率越低,因子值越高
        low_vol_factor = -volatility

        return low_vol_factor

    def calculate_beta(self, stock_returns: pd.Series,
                      market_returns: pd.Series,
                      period: int = 252) -> float:
        """
        Beta系数

        定义: 股票收益率对市场收益率的敏感度

        Args:
            stock_returns: 股票收益率序列
            market_returns: 市场收益率序列
            period: 计算窗口(默认252=1年)

        Returns:
            Beta值
            - >1: 高风险高收益
            - <1: 低风险低收益
            - <0: 与市场负相关
        """
        # 取最近period天
        stock_ret = stock_returns.iloc[-period:]
        market_ret = market_returns.iloc[-period:]

        # 计算协方差矩阵
        cov = np.cov(stock_ret, market_ret)[0, 1]
        var_market = np.var(market_ret)

        if var_market == 0:
            return 1.0

        beta = cov / var_market

        return beta

    # ========== 综合因子评分 ==========

    def calculate_comprehensive_score(self,
                                     stock_data: pd.DataFrame,
                                     market_data: pd.DataFrame,
                                     financial_data: Dict) -> Dict:
        """
        综合学术因子评分

        整合所有学术因子,给出综合评价

        Args:
            stock_data: 股票历史数据
            market_data: 市场数据(用于计算分位数、Beta)
            financial_data: 财务数据

        Returns:
            {
                'fama_french': {...},
                'momentum': float,
                'quality': float,
                'low_volatility': float,
                'total_score': float,
                'rank': str,
            }
        """
        # 1. Fama-French三因子
        stock_latest = pd.Series({
            'return': stock_data['close'].pct_change().iloc[-1],
            'market_cap': financial_data.get('market_cap', 100),
            'pb': financial_data.get('pb', 3.0),
        })

        market_summary = pd.DataFrame({
            'market_cap': market_data.get('market_cap', []),
            'pb': market_data.get('pb', []),
            'return': market_data.get('return', []),
        })

        ff3 = self.calculate_fama_french_three_factors(stock_latest, market_summary)

        # 2. 动量因子
        momentum = self.calculate_momentum_factor(stock_data).iloc[-1]

        # 3. 质量因子
        quality = self.calculate_quality_factor(financial_data)

        # 4. 低波动率因子
        low_vol = self.calculate_low_volatility_factor(stock_data).iloc[-1]

        # 5. 综合得分(0-100)
        # FF三因子: 30分, 动量: 20分, 质量: 30分, 低波: 20分
        ff3_score = (ff3['ff3_score'] + 2) / 4 * 30  # 归一化到0-30
        momentum_score = (momentum + 1) / 2 * 20 if not np.isnan(momentum) else 10
        quality_score = quality / 100 * 30

        # 低波得分: 归一化处理,避免除零错误
        if np.isnan(low_vol) or low_vol == 0:
            low_vol_score = 10
        else:
            # 简单归一化: 波动率越低,得分越高
            # 假设波动率范围0-0.05,线性映射到0-20分
            vol_abs = abs(low_vol)
            low_vol_score = max(0, min(20, 20 - vol_abs * 400))

        total_score = ff3_score + momentum_score + quality_score + low_vol_score

        # 6. 评级
        if total_score >= 80:
            rank = 'A+'
        elif total_score >= 70:
            rank = 'A'
        elif total_score >= 60:
            rank = 'B'
        elif total_score >= 50:
            rank = 'C'
        else:
            rank = 'D'

        return {
            'fama_french': ff3,
            'momentum': momentum,
            'quality': quality,
            'low_volatility': low_vol,
            'ff3_score': ff3_score,
            'momentum_score': momentum_score,
            'quality_score': quality_score,
            'low_vol_score': low_vol_score,
            'total_score': total_score,
            'rank': rank,
        }

    def generate_factor_report(self, scores: Dict) -> str:
        """
        生成因子分析报告

        Args:
            scores: calculate_comprehensive_score返回的结果

        Returns:
            文本报告
        """
        report = []
        report.append("=" * 60)
        report.append("学术因子分析报告")
        report.append("=" * 60)

        report.append("\n📊 Fama-French三因子:")
        ff3 = scores['fama_french']
        report.append(f"  市场因子(MKT): {ff3['MKT']:.4f}")
        report.append(f"  规模因子(SMB): {ff3['SMB']:.2f} {'(小盘股)' if ff3['SMB'] > 0 else '(大盘股)' if ff3['SMB'] < 0 else '(中盘股)'}")
        report.append(f"  价值因子(HML): {ff3['HML']:.2f} {'(价值股)' if ff3['HML'] > 0 else '(成长股)' if ff3['HML'] < 0 else '(中性)'}")

        report.append(f"\n📈 动量因子:")
        report.append(f"  动量值: {scores['momentum']:.4f}")
        report.append(f"  评价: {'强势股(动量显著)' if scores['momentum'] > 0.1 else '弱势股(负动量)' if scores['momentum'] < -0.1 else '中性'}")

        report.append(f"\n💎 质量因子:")
        report.append(f"  质量得分: {scores['quality']:.1f}/100")
        if scores['quality'] >= 80:
            report.append(f"  评价: 高质量企业")
        elif scores['quality'] >= 60:
            report.append(f"  评价: 中等质量")
        else:
            report.append(f"  评价: 质量一般")

        report.append(f"\n📉 低波动率因子:")
        report.append(f"  波动率: {-scores['low_volatility']:.4f}")
        report.append(f"  评价: {'低波动(防御性)' if scores['low_volatility'] > 0 else '高波动(进攻性)'}")

        report.append(f"\n\n🎯 综合评分:")
        report.append(f"  总分: {scores['total_score']:.1f}/100")
        report.append(f"  评级: {scores['rank']}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


# 便捷函数
def quick_academic_analysis(stock_data: pd.DataFrame,
                           market_data: pd.DataFrame,
                           financial_data: Dict) -> Dict:
    """
    快速学术因子分析

    Args:
        stock_data: 股票历史数据
        market_data: 市场数据
        financial_data: 财务数据

    Returns:
        分析结果 + 报告
    """
    analyzer = AcademicFactors()

    scores = analyzer.calculate_comprehensive_score(
        stock_data, market_data, financial_data
    )

    report = analyzer.generate_factor_report(scores)

    return {
        **scores,
        'report': report,
    }


# 示例
if __name__ == "__main__":
    # 创建模拟数据
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
    stock_data = pd.DataFrame({
        'close': 100 + np.random.randn(300).cumsum(),
        'volume': np.random.randint(1000000, 10000000, 300),
    }, index=dates)

    market_data = pd.DataFrame({
        'market_cap': np.random.uniform(50, 500, 100),
        'pb': np.random.uniform(1, 10, 100),
        'return': np.random.normal(0.001, 0.02, 100),
    })

    financial_data = {
        'market_cap': 150,
        'pb': 3.5,
        'roe': 0.15,
        'roe_std': 0.03,
        'asset_growth': 0.10,
        'gross_margin': 0.35,
    }

    result = quick_academic_analysis(stock_data, market_data, financial_data)

    print(result['report'])
