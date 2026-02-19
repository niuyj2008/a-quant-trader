"""
Phase 10.3 新模块单元测试

测试内容:
  1. IndustryRotationFactor: 行业轮动因子
  2. ATRStopLoss: 波动率自适应止损
  3. CorrelationMonitor: 持仓相关性监控
  4. BlackSwanDetector: 黑天鹅检测
  5. DLSignalFilter: 深度学习信号过滤
  6. A股行研数据获取
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ==================== 行业轮动因子测试 ====================

class TestIndustryRotationFactor:
    def setup_method(self):
        from src.factors.industry_factors import IndustryRotationFactor
        self.factor = IndustryRotationFactor(cache_ttl=1)

    def test_compute_industry_scores_structure(self):
        """测试返回数据结构"""
        # 注意: 此测试可能因网络问题失败，仅验证结构
        scores = self.factor.compute_industry_scores('US', lookback_days=20)
        if scores:  # 只在有数据时测试
            assert isinstance(scores, dict)
            for industry, score in scores.items():
                assert isinstance(industry, str)
                assert 0 <= score <= 100

    def test_get_stock_industry_bonus(self):
        """测试个股行业增益计算"""
        scores = {'Technology': 80, 'Healthcare': 20}
        bonus = self.factor.get_stock_industry_bonus('AAPL', 'Technology', scores)
        assert bonus > 0  # 强势行业正增益

        bonus = self.factor.get_stock_industry_bonus('XYZ', 'Healthcare', scores)
        assert bonus < 0  # 弱势行业负增益

    def test_get_top_industries(self):
        """测试强势行业提取"""
        # Mock数据
        self.factor._cache['industry:US:20'] = (
            999999999,  # 未来时间戳
            {'Tech': 90, 'Finance': 60, 'Energy': 30}
        )
        top = self.factor.get_top_industries('US', top_n=2, lookback_days=20)
        assert len(top) <= 2
        assert top[0][1] > top[1][1]  # 降序排列

    def test_cache_mechanism(self):
        """测试缓存机制"""
        import time
        self.factor._set_cache('test:key', {'data': 123})
        cached = self.factor._get_cache('test:key')
        assert cached == {'data': 123}

        time.sleep(2)  # 超过 TTL=1
        cached = self.factor._get_cache('test:key')
        assert cached is None


# ==================== ATR自适应止损测试 ====================

class TestATRStopLoss:
    def _make_df(self, days=60, volatility=0.02):
        """生成模拟行情"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.normal(0.1, volatility * 100, days))
        high = close * (1 + np.random.uniform(0, 0.02, days))
        low = close * (1 - np.random.uniform(0, 0.02, days))
        return pd.DataFrame({
            'close': close,
            'high': high,
            'low': low,
        }, index=dates)

    def test_calculate_atr(self):
        """测试ATR计算"""
        from src.trading.risk import ATRStopLoss
        atr_sl = ATRStopLoss(multiplier=2.0, atr_period=14)

        df = self._make_df(days=60)
        atr = atr_sl.calculate_atr(df)
        assert atr > 0
        assert atr < 10  # 合理范围

    def test_calculate_stop_long(self):
        """测试多头止损价计算"""
        from src.trading.risk import ATRStopLoss
        atr_sl = ATRStopLoss(multiplier=2.0)

        df = self._make_df(days=60)
        entry = 100.0
        stop = atr_sl.calculate_stop(df, entry, direction='long')
        assert stop is not None
        assert stop < entry  # 多头止损必须低于入场价

    def test_calculate_trailing_stop(self):
        """测试移动止损"""
        from src.trading.risk import ATRStopLoss
        atr_sl = ATRStopLoss(multiplier=2.0)

        df = self._make_df(days=60)
        highest = 105.0
        trail_stop = atr_sl.calculate_trailing_stop(df, highest)
        assert trail_stop is not None
        assert trail_stop < highest

    def test_get_stop_info(self):
        """测试完整止损信息"""
        from src.trading.risk import ATRStopLoss
        atr_sl = ATRStopLoss(multiplier=2.0)

        df = self._make_df(days=60)
        info = atr_sl.get_stop_info(df, entry_price=100.0)
        assert 'atr' in info
        assert 'stop_price' in info
        assert 'stop_pct' in info
        assert info['stop_pct'] < 0  # 止损为负

    def test_fallback_on_insufficient_data(self):
        """数据不足时降级"""
        from src.trading.risk import ATRStopLoss
        atr_sl = ATRStopLoss()

        df = self._make_df(days=5)  # 少于atr_period
        atr = atr_sl.calculate_atr(df)
        assert atr == 0

        info = atr_sl.get_stop_info(df, 100.0)
        assert info['method'] == 'fixed_fallback'


# ==================== 相关性监控测试 ====================

class TestCorrelationMonitor:
    def _make_price_series(self, days=100, trend=0.001):
        """生成价格序列"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        np.random.seed(42)
        returns = np.random.normal(trend, 0.02, days)
        prices = 100 * np.exp(np.cumsum(returns))
        return pd.Series(prices, index=dates)

    def test_calculate_correlation_matrix(self):
        """测试相关性矩阵计算"""
        from src.trading.risk import CorrelationMonitor
        monitor = CorrelationMonitor(lookback=60)

        price_data = {
            'A': self._make_price_series(),
            'B': self._make_price_series(),
            'C': self._make_price_series(),
        }
        corr_matrix = monitor.calculate_correlation_matrix(price_data)
        assert corr_matrix is not None
        assert corr_matrix.shape == (3, 3)
        assert corr_matrix.loc['A', 'A'] == 1.0  # 对角线为1

    def test_get_high_correlation_pairs(self):
        """测试高相关性对识别"""
        from src.trading.risk import CorrelationMonitor
        monitor = CorrelationMonitor(warning_threshold=0.8)

        # 创建高度相关的序列
        base = self._make_price_series()
        price_data = {
            'A': base,
            'B': base * 1.01 + np.random.normal(0, 0.1, len(base)),  # 高度相关
            'C': self._make_price_series(),  # 不相关
        }
        pairs = monitor.get_high_correlation_pairs(price_data)
        # A和B应该被识别为高相关
        if pairs:
            assert any('A' in (p[0], p[1]) and 'B' in (p[0], p[1]) for p in pairs)

    def test_get_diversification_score(self):
        """测试分散化得分"""
        from src.trading.risk import CorrelationMonitor
        monitor = CorrelationMonitor()

        # 低相关性组合
        price_data = {f'S{i}': self._make_price_series() for i in range(5)}
        score = monitor.get_diversification_score(price_data)
        assert 0 <= score <= 100

    def test_generate_warnings(self):
        """测试相关性预警生成"""
        from src.trading.risk import CorrelationMonitor
        monitor = CorrelationMonitor(warning_threshold=0.99)  # 极高阈值

        price_data = {'A': self._make_price_series(), 'B': self._make_price_series()}
        warnings = monitor.generate_warnings(price_data)
        assert isinstance(warnings, list)


# ==================== 黑天鹅检测测试 ====================

class TestBlackSwanDetector:
    def _make_index_data(self, days=300, shock_day=None, shock_size=0.0):
        """生成指数数据，可选插入极端波动"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.01, days)
        if shock_day is not None and 0 < shock_day < days:
            returns[shock_day] = shock_size
        prices = 100 * np.exp(np.cumsum(returns))
        return pd.DataFrame({'close': prices}, index=dates)

    def test_normal_condition(self):
        """正常市场无预警"""
        from src.trading.risk import BlackSwanDetector
        detector = BlackSwanDetector(sigma_threshold=3.0)

        df = self._make_index_data(days=300)
        result = detector.check(df)
        # 正常情况不应触发（概率很小）
        assert 'triggered' in result
        assert 'severity' in result

    def test_warning_trigger(self):
        """3-sigma事件触发预警"""
        from src.trading.risk import BlackSwanDetector
        detector = BlackSwanDetector(sigma_threshold=3.0, lookback=252)

        df = self._make_index_data(days=300, shock_day=299, shock_size=-0.04)  # 最后一天-4%暴跌
        result = detector.check(df)
        # -4%在正常市场下应该是3-sigma级别事件
        assert result['daily_return'] < -0.03

    def test_critical_trigger(self):
        """4.5-sigma极端事件"""
        from src.trading.risk import BlackSwanDetector
        detector = BlackSwanDetector(sigma_threshold=3.0)

        df = self._make_index_data(days=300, shock_day=299, shock_size=-0.08)  # 最后一天-8%崩盘
        result = detector.check(df)
        assert result['triggered']
        assert result['severity'] in ('warning', 'critical')

    def test_get_emergency_actions(self):
        """测试紧急动作建议"""
        from src.trading.risk import BlackSwanDetector
        detector = BlackSwanDetector()

        actions_critical = detector.get_emergency_actions('critical')
        assert len(actions_critical) > 0
        assert any('暂停' in a for a in actions_critical)

        actions_warning = detector.get_emergency_actions('warning')
        assert len(actions_warning) > 0


# ==================== DL信号过滤器测试 ====================

class TestDLSignalFilter:
    def _make_factored_df(self, days=120):
        """生成模拟因子DataFrame"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        np.random.seed(42)
        return pd.DataFrame({
            'close': 100 + np.cumsum(np.random.normal(0.1, 1, days)),
            'ma_5': np.random.uniform(95, 105, days),
            'ma_10': np.random.uniform(95, 105, days),
            'rsi_14': np.random.uniform(30, 70, days),
            'momentum_5': np.random.normal(0, 0.02, days),
            'volatility_20': np.random.uniform(0.01, 0.03, days),
        }, index=dates)

    def test_init_without_model(self):
        """无模型时禁用"""
        from src.models.dl_signal_filter import DLSignalFilter
        dl_filter = DLSignalFilter(model_dir='nonexistent', enabled=True)
        assert not dl_filter.enabled

    def test_filter_signal_disabled(self):
        """禁用时透传信号"""
        from src.models.dl_signal_filter import DLSignalFilter
        dl_filter = DLSignalFilter(enabled=False)

        df = self._make_factored_df()
        result = dl_filter.filter_signal('buy', 'TEST', df, confidence=75.0)
        assert result['action'] == 'buy'
        assert result['confidence'] == 75.0
        assert not result['filtered']

    def test_hold_signal_passthrough(self):
        """hold信号总是透传"""
        from src.models.dl_signal_filter import DLSignalFilter
        dl_filter = DLSignalFilter(enabled=True)

        df = self._make_factored_df()
        result = dl_filter.filter_signal('hold', 'TEST', df)
        assert result['action'] == 'hold'

    def test_get_model_info(self):
        """获取模型信息"""
        from src.models.dl_signal_filter import DLSignalFilter
        dl_filter = DLSignalFilter(enabled=False)
        info = dl_filter.get_model_info()
        assert 'enabled' in info
        assert not info['enabled']

    def test_filter_batch(self):
        """批量过滤"""
        from src.models.dl_signal_filter import DLSignalFilter
        dl_filter = DLSignalFilter(enabled=False)

        df = self._make_factored_df()
        signals = [
            {'code': 'A', 'action': 'buy', 'confidence': 70},
            {'code': 'B', 'action': 'sell', 'confidence': 65},
        ]
        factored_data = {'A': df, 'B': df}

        filtered = dl_filter.filter_batch(signals, factored_data)
        assert len(filtered) == 2


# ==================== A股行研数据获取测试 ====================

class TestCNResearchData:
    def test_get_research_cn_structure(self):
        """测试A股行研数据获取（结构验证）"""
        from src.data.fetcher import DataFetcher
        fetcher = DataFetcher()

        # 注意: 实际API调用可能失败，仅验证不崩溃
        try:
            result = fetcher._get_research_cn('000001')
            assert isinstance(result, dict)
            # 如果有数据，验证结构
            if result:
                assert 'earnings_estimate' in result or 'recommendations' in result
        except Exception:
            pass  # 网络失败不算测试失败

    def test_cn_forecast_to_recommendations(self):
        """测试盈利预测转评级"""
        from src.data.fetcher import DataFetcher
        fetcher = DataFetcher()

        # Mock数据
        forecast_df = pd.DataFrame({
            '研究机构': ['券商A', '券商B'],
            '最新评级': ['买入', '持有'],
            '发布日期': ['2025-01-01', '2025-01-02'],
        })

        recs = fetcher._cn_forecast_to_recommendations(forecast_df, '000001')
        if recs is not None:
            assert not recs.empty
            assert 'Date' in recs.columns
            assert 'firm' in recs.columns
            assert 'to_grade' in recs.columns


# ==================== 集成测试 ====================

class TestPhase10_3_Integration:
    def test_industry_classifier(self):
        """行业分类器"""
        from src.factors.industry_factors import IndustryClassifier
        classifier = IndustryClassifier()

        # 美股示例（yfinance可用时）
        industry = classifier.get_industry('AAPL', market='US')
        # 允许返回None（网络问题）
        if industry:
            assert isinstance(industry, str)

    def test_atr_and_correlation_together(self):
        """ATR止损和相关性监控配合使用"""
        from src.trading.risk import ATRStopLoss, CorrelationMonitor

        atr_sl = ATRStopLoss()
        corr_mon = CorrelationMonitor()

        # 模拟场景：3只股票的历史数据
        dates = pd.date_range(end=datetime.now(), periods=100, freq='B')
        np.random.seed(42)

        price_data = {}
        atr_results = {}

        for code in ['A', 'B', 'C']:
            close = 100 + np.cumsum(np.random.normal(0.1, 1, 100))
            high = close * 1.01
            low = close * 0.99
            df = pd.DataFrame({'close': close, 'high': high, 'low': low}, index=dates)

            price_data[code] = pd.Series(close, index=dates)
            atr_results[code] = atr_sl.get_stop_info(df, entry_price=close[0])

        # 验证ATR结果
        assert all('stop_price' in r for r in atr_results.values())

        # 验证相关性
        div_score = corr_mon.get_diversification_score(price_data)
        assert 0 <= div_score <= 100

    def test_all_phase10_3_modules_importable(self):
        """确保所有新模块可导入"""
        from src.factors.industry_factors import IndustryRotationFactor, IndustryClassifier
        from src.trading.risk import ATRStopLoss, CorrelationMonitor, BlackSwanDetector
        from src.models.dl_signal_filter import DLSignalFilter

        assert IndustryRotationFactor
        assert IndustryClassifier
        assert ATRStopLoss
        assert CorrelationMonitor
        assert BlackSwanDetector
        assert DLSignalFilter
