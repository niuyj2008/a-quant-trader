"""
A股量化交易系统 - 基本面深度分析模块

提供超越PE/PB的深度基本面分析能力
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    logger.warning("AKShare未安装,A股基本面数据功能不可用")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance未安装,美股基本面数据功能不可用")


class FundamentalAnalyzer:
    """基本面深度分析器

    提供:
    1. 盈利能力趋势分析 (ROE/ROA/毛利率)
    2. 增长质量分析 (营收vs利润一致性, 现金流质量)
    3. 相对估值分析 (PE/PB vs 行业, PEG, 历史百分位)
    4. 综合基本面评分 (0-100分)
    """

    def __init__(self):
        self.cache = {}  # 简单缓存机制

    def analyze_profitability_trend(self, code: str, market: str = "CN", years: int = 3) -> Dict:
        """
        盈利能力趋势分析

        Args:
            code: 股票代码
            market: 市场 ('CN'/'US')
            years: 分析年限

        Returns:
            {
                'roe_trend': [0.15, 0.18, 0.20],  # 3年ROE
                'roe_yoy': [None, 0.20, 0.11],    # YoY增长率
                'roa_trend': [...],
                'gross_margin_trend': [...],
                'trend': 'improving',  # 'improving'/'declining'/'stable'
                'inflection_point': '2023',  # 拐点年份(如有)
            }
        """
        try:
            if market == "CN":
                return self._analyze_cn_profitability(code, years)
            elif market == "US":
                return self._analyze_us_profitability(code, years)
            else:
                logger.error(f"不支持的市场: {market}")
                return {}
        except Exception as e:
            logger.error(f"盈利能力分析失败 {code}: {e}")
            return {}

    def _analyze_cn_profitability(self, code: str, years: int) -> Dict:
        """A股盈利能力分析"""
        if not AKSHARE_AVAILABLE:
            return {}

        try:
            # 获取财务摘要数据(过去N年)
            df_finance = ak.stock_financial_abstract_ths(symbol=code, indicator="按年度")

            if df_finance.empty:
                logger.warning(f"{code} 无财务数据")
                return {}

            # 取最近N年数据
            df_recent = df_finance.head(years)

            # 提取ROE (净资产收益率)
            roe_col = None
            for col in df_recent.columns:
                if 'ROE' in col or '净资产收益率' in col:
                    roe_col = col
                    break

            # 提取ROA (总资产收益率)
            roa_col = None
            for col in df_recent.columns:
                if 'ROA' in col or '总资产收益率' in col:
                    roa_col = col
                    break

            # 提取毛利率
            margin_col = None
            for col in df_recent.columns:
                if '毛利率' in col:
                    margin_col = col
                    break

            roe_trend = []
            roa_trend = []
            margin_trend = []

            if roe_col:
                roe_trend = self._extract_numeric_values(df_recent[roe_col])
            if roa_col:
                roa_trend = self._extract_numeric_values(df_recent[roa_col])
            if margin_col:
                margin_trend = self._extract_numeric_values(df_recent[margin_col])

            # 计算YoY增长率
            roe_yoy = self._calculate_yoy(roe_trend)

            # 判断趋势
            trend = self._determine_trend(roe_trend)

            # 寻找拐点
            inflection_point = self._find_inflection_point(roe_trend, df_recent.iloc[:, 0].tolist())

            return {
                'roe_trend': roe_trend,
                'roe_yoy': roe_yoy,
                'roa_trend': roa_trend,
                'gross_margin_trend': margin_trend,
                'trend': trend,
                'inflection_point': inflection_point,
                'years': len(roe_trend),
            }

        except Exception as e:
            logger.error(f"A股盈利能力分析失败 {code}: {e}")
            return {}

    def _analyze_us_profitability(self, code: str, years: int) -> Dict:
        """美股盈利能力分析"""
        if not YFINANCE_AVAILABLE:
            return {}

        try:
            ticker = yf.Ticker(code)

            # 获取财务数据
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet

            if financials.empty or balance_sheet.empty:
                logger.warning(f"{code} 无财务数据")
                return {}

            # 计算ROE = 净利润 / 股东权益
            net_income = financials.loc['Net Income'] if 'Net Income' in financials.index else None
            equity = balance_sheet.loc['Stockholders Equity'] if 'Stockholders Equity' in balance_sheet.index else None

            roe_trend = []
            if net_income is not None and equity is not None:
                # 取最近N年
                for i in range(min(years, len(net_income))):
                    if i < len(equity):
                        roe = (net_income.iloc[i] / equity.iloc[i]) if equity.iloc[i] != 0 else 0
                        roe_trend.append(roe)

            # 计算ROA = 净利润 / 总资产
            total_assets = balance_sheet.loc['Total Assets'] if 'Total Assets' in balance_sheet.index else None
            roa_trend = []
            if net_income is not None and total_assets is not None:
                for i in range(min(years, len(net_income))):
                    if i < len(total_assets):
                        roa = (net_income.iloc[i] / total_assets.iloc[i]) if total_assets.iloc[i] != 0 else 0
                        roa_trend.append(roa)

            roe_yoy = self._calculate_yoy(roe_trend)
            trend = self._determine_trend(roe_trend)

            return {
                'roe_trend': roe_trend,
                'roe_yoy': roe_yoy,
                'roa_trend': roa_trend,
                'gross_margin_trend': [],
                'trend': trend,
                'inflection_point': None,
                'years': len(roe_trend),
            }

        except Exception as e:
            logger.error(f"美股盈利能力分析失败 {code}: {e}")
            return {}

    def analyze_growth_quality(self, code: str, market: str = "CN") -> Dict:
        """
        增长质量分析

        Returns:
            {
                'revenue_growth': 0.15,         # 营收增速
                'profit_growth': 0.12,          # 利润增速
                'consistency': 'good',          # 'good'/'poor'
                'cash_flow_quality': 1.2,       # 现金流/净利润比率
                'receivables_turnover': 6.5,    # 应收账款周转率
                'quality_score': 75,            # 质量评分(0-100)
            }
        """
        try:
            if market == "CN":
                return self._analyze_cn_growth_quality(code)
            elif market == "US":
                return self._analyze_us_growth_quality(code)
            else:
                return {}
        except Exception as e:
            logger.error(f"增长质量分析失败 {code}: {e}")
            return {}

    def _analyze_cn_growth_quality(self, code: str) -> Dict:
        """A股增长质量分析"""
        if not AKSHARE_AVAILABLE:
            return {}

        try:
            # 获取利润表数据
            df_profit = ak.stock_financial_abstract_ths(symbol=code, indicator="按年度")

            if df_profit.empty:
                return {}

            # 营收增长率
            revenue_growth = self._find_growth_rate(df_profit, ['营业总收入', '营业收入'])

            # 净利润增长率
            profit_growth = self._find_growth_rate(df_profit, ['净利润'])

            # 一致性判断: 营收和利润增速差异<5%为good
            consistency = 'good' if abs(revenue_growth - profit_growth) < 0.05 else 'poor'

            # 现金流质量 (经营性现金流 / 净利润)
            cash_flow_quality = self._calculate_cash_flow_quality(df_profit)

            # 质量评分
            quality_score = self._calculate_quality_score(
                revenue_growth, profit_growth, consistency, cash_flow_quality
            )

            return {
                'revenue_growth': revenue_growth,
                'profit_growth': profit_growth,
                'consistency': consistency,
                'cash_flow_quality': cash_flow_quality,
                'receivables_turnover': 0,  # TODO: 需要额外数据源
                'quality_score': quality_score,
            }

        except Exception as e:
            logger.error(f"A股增长质量分析失败 {code}: {e}")
            return {}

    def _analyze_us_growth_quality(self, code: str) -> Dict:
        """美股增长质量分析"""
        if not YFINANCE_AVAILABLE:
            return {}

        try:
            ticker = yf.Ticker(code)
            financials = ticker.financials
            cashflow = ticker.cashflow

            if financials.empty:
                return {}

            # 营收增长率
            revenue = financials.loc['Total Revenue'] if 'Total Revenue' in financials.index else None
            revenue_growth = self._calculate_growth_from_series(revenue)

            # 净利润增长率
            net_income = financials.loc['Net Income'] if 'Net Income' in financials.index else None
            profit_growth = self._calculate_growth_from_series(net_income)

            # 一致性
            consistency = 'good' if abs(revenue_growth - profit_growth) < 0.05 else 'poor'

            # 现金流质量
            operating_cf = cashflow.loc['Operating Cash Flow'] if not cashflow.empty and 'Operating Cash Flow' in cashflow.index else None
            cash_flow_quality = 1.0
            if operating_cf is not None and net_income is not None and len(operating_cf) > 0 and len(net_income) > 0:
                if net_income.iloc[0] != 0:
                    cash_flow_quality = operating_cf.iloc[0] / net_income.iloc[0]

            quality_score = self._calculate_quality_score(
                revenue_growth, profit_growth, consistency, cash_flow_quality
            )

            return {
                'revenue_growth': revenue_growth,
                'profit_growth': profit_growth,
                'consistency': consistency,
                'cash_flow_quality': cash_flow_quality,
                'receivables_turnover': 0,
                'quality_score': quality_score,
            }

        except Exception as e:
            logger.error(f"美股增长质量分析失败 {code}: {e}")
            return {}

    def relative_valuation(self, code: str, market: str = "CN", sector: str = "") -> Dict:
        """
        相对估值分析

        Returns:
            {
                'pe': 25.0,
                'sector_avg_pe': 30.0,
                'pe_percentile': 35,       # 近5年35%分位
                'pb': 3.5,
                'sector_avg_pb': 4.0,
                'peg': 1.2,                # <1低估, >2高估
                'valuation': 'undervalued', # 'undervalued'/'fair'/'overvalued'
            }
        """
        try:
            if market == "CN":
                return self._cn_relative_valuation(code, sector)
            elif market == "US":
                return self._us_relative_valuation(code, sector)
            else:
                return {}
        except Exception as e:
            logger.error(f"相对估值分析失败 {code}: {e}")
            return {}

    def _cn_relative_valuation(self, code: str, sector: str) -> Dict:
        """A股相对估值"""
        if not AKSHARE_AVAILABLE:
            return {}

        try:
            # 获取实时行情数据(包含PE/PB)
            df_spot = ak.stock_zh_a_spot_em()

            # 筛选目标股票
            stock_data = df_spot[df_spot['代码'] == code]

            if stock_data.empty:
                return {}

            # 提取PE和PB
            pe = float(stock_data['市盈率-动态'].values[0]) if '市盈率-动态' in stock_data.columns else 0
            pb = float(stock_data['市净率'].values[0]) if '市净率' in stock_data.columns else 0

            # TODO: 需要行业平均PE/PB数据
            sector_avg_pe = pe * 1.2  # 模拟数据
            sector_avg_pb = pb * 1.15

            # 计算历史百分位(暂时使用简单估算)
            pe_percentile = 50  # 默认50%分位

            # PEG = PE / 利润增长率
            # TODO: 需要利润增长率数据
            peg = 1.5  # 模拟数据

            # 估值判断
            valuation = 'fair'
            if pe < sector_avg_pe * 0.8 and pb < sector_avg_pb * 0.8:
                valuation = 'undervalued'
            elif pe > sector_avg_pe * 1.2 or pb > sector_avg_pb * 1.2:
                valuation = 'overvalued'

            return {
                'pe': pe,
                'sector_avg_pe': sector_avg_pe,
                'pe_percentile': pe_percentile,
                'pb': pb,
                'sector_avg_pb': sector_avg_pb,
                'peg': peg,
                'valuation': valuation,
            }

        except Exception as e:
            logger.error(f"A股相对估值分析失败 {code}: {e}")
            return {}

    def _us_relative_valuation(self, code: str, sector: str) -> Dict:
        """美股相对估值"""
        if not YFINANCE_AVAILABLE:
            return {}

        try:
            ticker = yf.Ticker(code)
            info = ticker.info

            pe = info.get('trailingPE', 0)
            pb = info.get('priceToBook', 0)
            peg = info.get('pegRatio', 0)

            # TODO: 获取行业平均数据
            sector_avg_pe = pe * 1.1
            sector_avg_pb = pb * 1.1

            pe_percentile = 50  # 默认中位数

            valuation = 'fair'
            if peg < 1.0:
                valuation = 'undervalued'
            elif peg > 2.0:
                valuation = 'overvalued'

            return {
                'pe': pe,
                'sector_avg_pe': sector_avg_pe,
                'pe_percentile': pe_percentile,
                'pb': pb,
                'sector_avg_pb': sector_avg_pb,
                'peg': peg,
                'valuation': valuation,
            }

        except Exception as e:
            logger.error(f"美股相对估值分析失败 {code}: {e}")
            return {}

    def generate_fundamental_score(self, code: str, market: str = "CN", sector: str = "") -> Dict:
        """
        综合基本面评分 (0-100)

        评分维度:
        - 盈利能力 (25%): ROE、ROA、毛利率
        - 成长性 (25%): 营收增速、利润增速
        - 估值吸引力 (25%): PE/PB相对位置、PEG
        - 财务健康 (25%): 现金流、负债率

        Returns:
            {
                '盈利能力': 75,
                '成长性': 60,
                '估值吸引力': 80,
                '财务健康': 70,
                '综合得分': 71,
                '评级': 'B+',  # A+/A/A-/B+/B/B-/C+/C/C-/D
            }
        """
        try:
            # 获取各维度数据
            profitability = self.analyze_profitability_trend(code, market)
            growth = self.analyze_growth_quality(code, market)
            valuation = self.relative_valuation(code, market, sector)

            # 计算各维度得分
            profitability_score = self._score_profitability(profitability)
            growth_score = growth.get('quality_score', 50)
            valuation_score = self._score_valuation(valuation)
            health_score = self._score_financial_health(growth)

            # 综合得分 (加权平均)
            total_score = (
                profitability_score * 0.25 +
                growth_score * 0.25 +
                valuation_score * 0.25 +
                health_score * 0.25
            )

            # 评级
            rating = self._get_rating(total_score)

            return {
                '盈利能力': int(profitability_score),
                '成长性': int(growth_score),
                '估值吸引力': int(valuation_score),
                '财务健康': int(health_score),
                '综合得分': int(total_score),
                '评级': rating,
            }

        except Exception as e:
            logger.error(f"综合评分失败 {code}: {e}")
            return {
                '盈利能力': 0,
                '成长性': 0,
                '估值吸引力': 0,
                '财务健康': 0,
                '综合得分': 0,
                '评级': 'N/A',
            }

    # ==================== 辅助方法 ====================

    def _extract_numeric_values(self, series: pd.Series) -> List[float]:
        """从Series中提取数值"""
        values = []
        for val in series:
            try:
                if isinstance(val, str):
                    # 去除%符号等
                    val = val.replace('%', '').replace(',', '')
                num = float(val)
                values.append(num / 100 if abs(num) > 1 else num)  # 百分比转小数
            except:
                continue
        return values

    def _calculate_yoy(self, values: List[float]) -> List[Optional[float]]:
        """计算YoY增长率"""
        yoy = [None]  # 第一年无YoY
        for i in range(1, len(values)):
            if values[i-1] != 0:
                yoy.append((values[i] - values[i-1]) / values[i-1])
            else:
                yoy.append(None)
        return yoy

    def _determine_trend(self, values: List[float]) -> str:
        """判断趋势"""
        if len(values) < 2:
            return 'stable'

        increasing = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
        decreasing = sum(1 for i in range(1, len(values)) if values[i] < values[i-1])

        if increasing > decreasing:
            return 'improving'
        elif decreasing > increasing:
            return 'declining'
        else:
            return 'stable'

    def _find_inflection_point(self, values: List[float], years: List) -> Optional[str]:
        """寻找拐点"""
        if len(values) < 3:
            return None

        for i in range(1, len(values) - 1):
            # 从下降转上升
            if values[i-1] > values[i] < values[i+1]:
                return str(years[i]) if i < len(years) else None
            # 从上升转下降
            if values[i-1] < values[i] > values[i+1]:
                return str(years[i]) if i < len(years) else None

        return None

    def _find_growth_rate(self, df: pd.DataFrame, keywords: List[str]) -> float:
        """在DataFrame中查找增长率(取最新年度数据)"""
        for keyword in keywords:
            for col in df.columns:
                if keyword in col and '增长率' in col:
                    try:
                        # 数据按年份倒序,最后一行是最新数据
                        latest_value = df[col].iloc[-1]

                        # 处理百分比字符串,如"19.55%"
                        if isinstance(latest_value, str):
                            latest_value = latest_value.replace('%', '')
                            return float(latest_value) / 100.0
                        elif isinstance(latest_value, (int, float)):
                            # 如果已经是数值,检查是否需要除以100
                            return latest_value if latest_value < 1 else latest_value / 100.0
                    except:
                        pass
        return 0.0

    def _calculate_cash_flow_quality(self, df: pd.DataFrame) -> float:
        """计算现金流质量(每股经营现金流/每股收益)"""
        try:
            # 使用每股指标避免股本问题
            cf_per_share = None
            eps = None

            # 查找每股经营现金流
            if '每股经营现金流' in df.columns:
                cf_per_share = self._parse_financial_value(df['每股经营现金流'].iloc[-1])

            # 查找每股收益
            if '基本每股收益' in df.columns:
                eps = self._parse_financial_value(df['基本每股收益'].iloc[-1])

            if cf_per_share and eps and eps != 0:
                return cf_per_share / eps

        except Exception as e:
            logger.debug(f"现金流质量计算失败: {e}")

        return 1.0  # 默认值

    def _parse_financial_value(self, value) -> float:
        """解析财务数值(处理"1.47亿"这种格式)"""
        if pd.isna(value) or value is False:
            return 0.0

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            try:
                # 处理"1.47亿"格式
                if '亿' in value:
                    return float(value.replace('亿', '').strip()) * 1e8
                elif '万' in value:
                    return float(value.replace('万', '').strip()) * 1e4
                else:
                    return float(value)
            except:
                return 0.0

        return 0.0

    def _calculate_growth_from_series(self, series: Optional[pd.Series]) -> float:
        """从时间序列计算增长率"""
        if series is None or len(series) < 2:
            return 0.0

        try:
            recent = series.iloc[0]
            previous = series.iloc[1]
            if previous != 0:
                return (recent - previous) / abs(previous)
        except:
            pass

        return 0.0

    def _calculate_quality_score(self, revenue_growth: float, profit_growth: float,
                                consistency: str, cash_flow_quality: float) -> int:
        """计算增长质量评分"""
        score = 50  # 基准分

        # 营收增长加分
        if revenue_growth > 0.20:
            score += 20
        elif revenue_growth > 0.10:
            score += 10

        # 利润增长加分
        if profit_growth > 0.20:
            score += 20
        elif profit_growth > 0.10:
            score += 10

        # 一致性加分
        if consistency == 'good':
            score += 10

        # 现金流质量加分
        if cash_flow_quality > 1.2:
            score += 10
        elif cash_flow_quality > 1.0:
            score += 5

        return min(100, max(0, score))

    def _calculate_percentile(self, series: pd.Series, value: float) -> int:
        """计算百分位"""
        try:
            valid_values = series.dropna()
            if len(valid_values) == 0:
                return 50

            rank = (valid_values < value).sum()
            percentile = int(rank / len(valid_values) * 100)
            return percentile
        except:
            return 50

    def _score_profitability(self, profitability: Dict) -> int:
        """盈利能力评分"""
        if not profitability or 'roe_trend' not in profitability:
            return 50

        score = 50

        roe_trend = profitability.get('roe_trend', [])
        if roe_trend:
            latest_roe = roe_trend[0] if len(roe_trend) > 0 else 0

            # ROE评分
            if latest_roe > 0.20:
                score += 30
            elif latest_roe > 0.15:
                score += 20
            elif latest_roe > 0.10:
                score += 10

            # 趋势加分
            trend = profitability.get('trend', 'stable')
            if trend == 'improving':
                score += 20
            elif trend == 'declining':
                score -= 10

        return min(100, max(0, score))

    def _score_valuation(self, valuation: Dict) -> int:
        """估值吸引力评分"""
        if not valuation:
            return 50

        score = 50

        # 相对PE评分
        pe = valuation.get('pe', 0)
        sector_avg_pe = valuation.get('sector_avg_pe', pe)

        if pe > 0 and sector_avg_pe > 0:
            pe_ratio = pe / sector_avg_pe
            if pe_ratio < 0.8:
                score += 25
            elif pe_ratio < 1.0:
                score += 15
            elif pe_ratio > 1.5:
                score -= 15

        # PEG评分
        peg = valuation.get('peg', 1.5)
        if peg < 1.0:
            score += 25
        elif peg < 1.5:
            score += 10
        elif peg > 2.0:
            score -= 10

        return min(100, max(0, score))

    def _score_financial_health(self, growth: Dict) -> int:
        """财务健康评分"""
        if not growth:
            return 50

        score = 50

        # 现金流质量评分
        cf_quality = growth.get('cash_flow_quality', 1.0)
        if cf_quality > 1.5:
            score += 30
        elif cf_quality > 1.2:
            score += 20
        elif cf_quality > 1.0:
            score += 10
        elif cf_quality < 0.8:
            score -= 20

        # 增长一致性评分
        consistency = growth.get('consistency', 'poor')
        if consistency == 'good':
            score += 20

        return min(100, max(0, score))

    def _get_rating(self, score: int) -> str:
        """根据分数获取评级"""
        if score >= 90:
            return 'A+'
        elif score >= 85:
            return 'A'
        elif score >= 80:
            return 'A-'
        elif score >= 75:
            return 'B+'
        elif score >= 70:
            return 'B'
        elif score >= 65:
            return 'B-'
        elif score >= 60:
            return 'C+'
        elif score >= 55:
            return 'C'
        elif score >= 50:
            return 'C-'
        else:
            return 'D'


if __name__ == "__main__":
    # 测试代码
    analyzer = FundamentalAnalyzer()

    # 测试贵州茅台
    print("=" * 60)
    print("测试: 贵州茅台 (600519)")
    print("=" * 60)

    # 盈利能力
    profitability = analyzer.analyze_profitability_trend("600519", market="CN", years=3)
    print("\n盈利能力趋势:")
    print(f"  ROE趋势: {profitability.get('roe_trend', [])}")
    print(f"  趋势: {profitability.get('trend', 'N/A')}")

    # 增长质量
    growth = analyzer.analyze_growth_quality("600519", market="CN")
    print("\n增长质量:")
    print(f"  营收增速: {growth.get('revenue_growth', 0):.2%}")
    print(f"  利润增速: {growth.get('profit_growth', 0):.2%}")
    print(f"  现金流质量: {growth.get('cash_flow_quality', 0):.2f}")

    # 相对估值
    valuation = analyzer.relative_valuation("600519", market="CN")
    print("\n相对估值:")
    print(f"  PE: {valuation.get('pe', 0):.2f}")
    print(f"  PB: {valuation.get('pb', 0):.2f}")
    print(f"  估值: {valuation.get('valuation', 'N/A')}")

    # 综合评分
    score = analyzer.generate_fundamental_score("600519", market="CN")
    print("\n综合评分:")
    for key, value in score.items():
        print(f"  {key}: {value}")
