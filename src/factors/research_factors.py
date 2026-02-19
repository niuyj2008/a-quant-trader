"""
股票量化策略决策支持系统 - 行研报告因子模块

将华尔街分析师的结构化数据（评级、目标价、盈利预测）转化为 0-100 的因子得分，
可直接接入 InterpretableStrategy 的因子打分体系。

数据来源:
  - 美股: yfinance 内置的 Ticker.recommendations / analyst_price_targets
  - A股: Phase 10.3 通过 Tushare Pro 实现（当前预留接口）

学术依据:
  - Womack (1996): 评级变动后股价存在显著漂移效应
  - Jegadeesh et al. (2004): 一致预期修正与未来收益正相关
"""

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from loguru import logger


# ==================== 评级标准化映射表 ====================

RATING_MAP = {
    # 5分 - 强烈推荐
    'Strong Buy': 5, 'Top Pick': 5, 'Conviction Buy': 5,
    'Strong-Buy': 5,

    # 4分 - 推荐
    'Buy': 4, 'Overweight': 4, 'Outperform': 4,
    'Positive': 4, 'Accumulate': 4, 'Add': 4,
    'Sector Outperform': 4, 'Market Outperform': 4,
    'Long-Term Buy': 4,

    # 3分 - 中性
    'Hold': 3, 'Neutral': 3, 'Equal-Weight': 3,
    'Market Perform': 3, 'Sector Perform': 3,
    'In-Line': 3, 'Peer Perform': 3, 'Fair Value': 3,
    'Equal-weight': 3, 'Sector Weight': 3, 'Mixed': 3,

    # 2分 - 谨慎
    'Underperform': 2, 'Underweight': 2,
    'Sector Underperform': 2, 'Reduce': 2,
    'Negative': 2, 'Market Underperform': 2,
    'Below Average': 2,

    # 1分 - 卖出
    'Sell': 1, 'Strong Sell': 1,
}

# 卖方偏差校正基准线
# 华尔街评级经验分布: Buy约60%, Hold约35%, Sell约5%
# 对应平均分约 3.55（而非理论中性值 3.0）
SELL_SIDE_BIAS = 3.55


class ResearchReportFactor:
    """行研报告因子计算器

    将华尔街分析师的结构化评级和目标价数据转化为量化因子得分，
    设计原则:
    1. 所有得分归一化到 0-100 区间，与现有 InterpretableStrategy 因子体系兼容
    2. 数据不足时返回 None（策略层自动跳过该因子）
    3. 内置卖方偏差修正，避免系统性偏乐观
    4. 每个因子独立计算，策略层决定使用哪些因子及权重
    """

    def __init__(self, cache_ttl: int = 86400):
        """
        Args:
            cache_ttl: 缓存有效时间（秒），默认24小时
        """
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, Dict[str, float]]] = {}

    def compute_all_scores(self, code: str, current_price: float,
                           research_data: Dict) -> Dict[str, float]:
        """一次性计算所有行研因子得分

        Args:
            code: 股票代码
            current_price: 当前价格
            research_data: DataFetcher.get_research_data() 的返回值

        Returns:
            {
                '行研共识评级': float,    # 0-100
                '目标价空间': float,       # 0-100
                '评级变动动量': float,     # 0-100
                '机构覆盖广度': float,     # 0-100
            }
            数据不足的因子不包含在返回字典中
        """
        # 检查缓存
        if code in self._cache:
            ts, cached_scores = self._cache[code]
            if time.time() - ts < self.cache_ttl:
                return cached_scores

        if not research_data:
            return {}

        scores = {}
        recommendations = research_data.get('recommendations')
        price_targets = research_data.get('price_targets')

        # 因子1: 一致预期评级
        if recommendations is not None and not _is_empty_df(recommendations):
            score = self.consensus_rating_score(recommendations)
            if score is not None:
                scores['行研共识评级'] = score

        # 因子2: 目标价上行空间
        if price_targets and current_price > 0:
            score = self.target_price_upside(price_targets, current_price)
            if score is not None:
                scores['目标价空间'] = score

        # 因子3: 评级变动动量
        if recommendations is not None and not _is_empty_df(recommendations):
            score = self.rating_change_momentum(recommendations)
            if score is not None:
                scores['评级变动动量'] = score

        # 因子4: 机构覆盖广度
        if recommendations is not None and not _is_empty_df(recommendations):
            score = self.coverage_breadth(recommendations)
            if score is not None:
                scores['机构覆盖广度'] = score

        # 写入缓存
        self._cache[code] = (time.time(), scores)
        return scores

    # ==================== 因子1: 一致预期评级 ====================

    def consensus_rating_score(self, recommendations: pd.DataFrame) -> Optional[float]:
        """一致预期评级得分

        计算步骤:
        1. 筛选近90天评级 + 时间衰减加权
        2. 每家券商只取最新一条（去重）
        3. 评级文本→数值标准化
        4. 卖方偏差修正
        5. 映射到 0-100

        Returns:
            0-100 的得分，数据不足时返回 None
        """
        try:
            df = self._prepare_recommendations(recommendations)
            if df is None or len(df) < 3:
                return None

            now = datetime.now()

            # 筛选近90天 + 时间衰减权重
            records = []
            for _, row in df.iterrows():
                date = _parse_date(row)
                if date is None:
                    continue
                days_ago = (now - date).days
                if days_ago > 90:
                    continue

                grade = _normalize_rating(row)
                if grade is None:
                    continue

                # 时间衰减
                if days_ago <= 30:
                    weight = 1.0
                elif days_ago <= 60:
                    weight = 0.7
                else:
                    weight = 0.4

                firm = row.get('firm', row.get('Firm', f'unknown_{_}'))
                records.append({
                    'firm': firm,
                    'grade': grade,
                    'weight': weight,
                    'date': date,
                })

            if len(records) < 3:
                return None

            # 每家券商只取最新一条
            records_df = pd.DataFrame(records)
            latest_per_firm = records_df.sort_values('date', ascending=False).drop_duplicates(
                subset='firm', keep='first'
            )

            # 加权平均
            total_weight = latest_per_firm['weight'].sum()
            if total_weight == 0:
                return None
            raw_avg = (latest_per_firm['grade'] * latest_per_firm['weight']).sum() / total_weight

            # 卖方偏差修正
            corrected_avg = raw_avg - (SELL_SIDE_BIAS - 3.0)

            # 映射到 0-100: 1分→0, 3分→50, 5分→100
            score = (corrected_avg - 1.0) / 4.0 * 100
            return max(0, min(100, score))

        except Exception as e:
            logger.debug(f"一致预期评级计算失败: {e}")
            return None

    # ==================== 因子2: 目标价上行空间 ====================

    def target_price_upside(self, price_targets: Dict,
                             current_price: float) -> Optional[float]:
        """目标价上行空间得分

        计算步骤:
        1. 取一致目标价中位数（优先）或均值
        2. 计算 upside%
        3. 分段映射 + 分歧度衰减

        Returns:
            0-100 的得分
        """
        try:
            # 兼容不同返回格式（Dict 或 带属性的对象）
            median_target = _get_target_value(price_targets, 'median')
            mean_target = _get_target_value(price_targets, 'mean')
            low_target = _get_target_value(price_targets, 'low')
            high_target = _get_target_value(price_targets, 'high')

            target = median_target or mean_target
            if target is None or target <= 0 or current_price <= 0:
                return None

            upside = (target - current_price) / current_price

            # 分段映射
            if upside > 0.40:
                score = 95
            elif upside > 0.25:
                score = 85
            elif upside > 0.15:
                score = 75
            elif upside > 0.05:
                score = 60
            elif upside > 0:
                score = 50
            elif upside > -0.10:
                score = 40
            elif upside > -0.20:
                score = 25
            else:
                score = 10

            # 分歧度衰减: high-low差距过大说明分析师观点分散
            if low_target and high_target and high_target > 0:
                dispersion = (high_target - low_target) / current_price
                if dispersion > 1.0:
                    # 分歧极大，得分向50衰减30%
                    score = score * 0.7 + 50 * 0.3

            return max(0, min(100, score))

        except Exception as e:
            logger.debug(f"目标价空间计算失败: {e}")
            return None

    # ==================== 因子3: 评级变动动量 ====================

    def rating_change_momentum(self, recommendations: pd.DataFrame) -> Optional[float]:
        """评级变动动量得分

        学术依据: Womack (1996) 证实评级变动后股价有显著漂移效应

        计算步骤:
        1. 筛选近30天的评级变动记录
        2. 统计 upgrades / downgrades
        3. 7天内变动加权 × 1.5
        4. 分段映射

        Returns:
            0-100 的得分
        """
        try:
            df = self._prepare_recommendations(recommendations)
            if df is None or len(df) == 0:
                return None

            now = datetime.now()
            upgrades = 0.0
            downgrades = 0.0
            initiations = 0

            for _, row in df.iterrows():
                date = _parse_date(row)
                if date is None:
                    continue
                days_ago = (now - date).days
                if days_ago > 30:
                    continue

                action = str(row.get('action', row.get('Action', ''))).lower().strip()

                # 近7天的变动权重更高
                time_weight = 1.5 if days_ago <= 7 else 1.0

                if action in ('upgrade', 'up'):
                    upgrades += time_weight
                elif action in ('downgrade', 'down'):
                    downgrades += time_weight
                elif action in ('initiated', 'init', 'initiate'):
                    initiations += 1

            net_change = upgrades - downgrades

            # 如果30天内完全没有变动记录
            if upgrades == 0 and downgrades == 0:
                if initiations > 0:
                    return 55.0  # 首次覆盖，轻微正面
                return None  # 无数据

            # 分段映射
            if net_change >= 4:
                score = 95
            elif net_change >= 3:
                score = 85
            elif net_change >= 2:
                score = 75
            elif net_change >= 1:
                score = 65
            elif net_change >= 0:
                score = 50
            elif net_change >= -1:
                score = 35
            elif net_change >= -2:
                score = 25
            else:
                score = 10

            return float(score)

        except Exception as e:
            logger.debug(f"评级变动动量计算失败: {e}")
            return None

    # ==================== 因子4: 机构覆盖广度 ====================

    def coverage_breadth(self, recommendations: pd.DataFrame) -> Optional[float]:
        """机构覆盖广度得分

        统计近180天覆盖该股的独立券商数量。

        Returns:
            0-100 的得分
        """
        try:
            df = self._prepare_recommendations(recommendations)
            if df is None or len(df) == 0:
                return None

            now = datetime.now()
            firms = set()

            for _, row in df.iterrows():
                date = _parse_date(row)
                if date is None:
                    continue
                days_ago = (now - date).days
                if days_ago > 180:
                    continue

                firm = row.get('firm', row.get('Firm', ''))
                if firm:
                    firms.add(str(firm).strip())

            n_firms = len(firms)
            if n_firms == 0:
                return None

            # 分段映射
            if n_firms > 25:
                score = 80
            elif n_firms > 20:
                score = 70
            elif n_firms > 15:
                score = 65
            elif n_firms > 10:
                score = 55
            elif n_firms > 5:
                score = 45
            else:
                score = 35

            return float(score)

        except Exception as e:
            logger.debug(f"机构覆盖广度计算失败: {e}")
            return None

    # ==================== 内部工具方法 ====================

    def _prepare_recommendations(self, recommendations: pd.DataFrame) -> Optional[pd.DataFrame]:
        """将 yfinance 返回的 recommendations DataFrame 标准化"""
        if recommendations is None:
            return None
        if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
            df = recommendations.copy()
            # yfinance 有时返回以日期为index的DataFrame
            if df.index.name == 'Date' or (hasattr(df.index, 'dtype') and
                                            pd.api.types.is_datetime64_any_dtype(df.index)):
                df = df.reset_index()
            return df
        return None


# ==================== 模块级工具函数 ====================

def _is_empty_df(obj) -> bool:
    """安全检查 DataFrame 是否为空"""
    if obj is None:
        return True
    if isinstance(obj, pd.DataFrame):
        return obj.empty
    return False


def _parse_date(row) -> Optional[datetime]:
    """从评级记录行中解析日期"""
    for col in ('Date', 'date', 'index'):
        val = row.get(col)
        if val is not None:
            try:
                if isinstance(val, (datetime, pd.Timestamp)):
                    return pd.Timestamp(val).to_pydatetime().replace(tzinfo=None)
                return pd.to_datetime(val).to_pydatetime().replace(tzinfo=None)
            except Exception:
                continue
    return None


def _normalize_rating(row) -> Optional[int]:
    """将评级文本标准化为 1-5 的数值"""
    for col in ('to_grade', 'To Grade', 'toGrade', 'Grade'):
        val = row.get(col)
        if val and str(val).strip():
            grade_str = str(val).strip()
            if grade_str in RATING_MAP:
                return RATING_MAP[grade_str]
            # 模糊匹配: 检查子串
            for key, score in RATING_MAP.items():
                if key.lower() in grade_str.lower() or grade_str.lower() in key.lower():
                    return score
    return None


def _get_target_value(price_targets, key: str) -> Optional[float]:
    """从 price_targets 中安全提取值（兼容 Dict 和对象属性）"""
    if price_targets is None:
        return None
    try:
        # 尝试 dict 方式
        if isinstance(price_targets, dict):
            val = price_targets.get(key)
        else:
            # 尝试属性方式（yfinance 有时返回特殊对象）
            val = getattr(price_targets, key, None)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            return float(val)
    except (TypeError, ValueError):
        pass
    return None
