"""
Phase 10 新因子模块单元测试

测试内容:
  1. GrokSentimentFactor: Grok 情绪因子计算
  2. MarketRegimeHMM: 市场状态识别（规则引擎降级模式）
  3. MacroRegimeDetector: 宏观经济周期识别
  4. InterpretableStrategyAdapter: 策略接口适配
  5. GrokClient: 客户端初始化和缓存逻辑（不调用真实API）
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ==================== Grok 情绪因子测试 ====================

class TestGrokSentimentFactor:
    def setup_method(self):
        from src.factors.sentiment_factors import GrokSentimentFactor
        self.gf = GrokSentimentFactor()

    def test_stock_sentiment_bullish(self):
        """强看多情绪 → 高分"""
        result = self.gf.stock_sentiment_score({
            'sentiment_score': 0.8,
            'confidence': 0.9,
            'post_count': 50,
        })
        assert result is not None
        assert result > 80

    def test_stock_sentiment_bearish(self):
        """强看空情绪 → 低分"""
        result = self.gf.stock_sentiment_score({
            'sentiment_score': -0.7,
            'confidence': 0.8,
            'post_count': 30,
        })
        assert result is not None
        assert result < 25

    def test_stock_sentiment_neutral(self):
        """中性情绪 → 接近50"""
        result = self.gf.stock_sentiment_score({
            'sentiment_score': 0.0,
            'confidence': 0.7,
            'post_count': 20,
        })
        assert result is not None
        assert 45 <= result <= 55

    def test_low_confidence_decay(self):
        """低置信度衰减: 即使看多，得分也应向50靠拢"""
        high_conf = self.gf.stock_sentiment_score({
            'sentiment_score': 0.8,
            'confidence': 0.9,
            'post_count': 50,
        })
        low_conf = self.gf.stock_sentiment_score({
            'sentiment_score': 0.8,
            'confidence': 0.1,
            'post_count': 50,
        })
        assert low_conf < high_conf
        assert abs(low_conf - 50) < abs(high_conf - 50)

    def test_low_post_count_decay(self):
        """低样本量衰减"""
        many_posts = self.gf.stock_sentiment_score({
            'sentiment_score': 0.6,
            'confidence': 0.8,
            'post_count': 100,
        })
        few_posts = self.gf.stock_sentiment_score({
            'sentiment_score': 0.6,
            'confidence': 0.8,
            'post_count': 2,
        })
        assert few_posts < many_posts
        assert abs(few_posts - 50) < abs(many_posts - 50)

    def test_market_mood_contrarian(self):
        """市场情绪逆向: panic → 高分（买入机会）"""
        panic = self.gf.market_mood_score({
            'market_mood': 'panic',
            'fear_greed_estimate': 10,
        })
        euphoria = self.gf.market_mood_score({
            'market_mood': 'euphoria',
            'fear_greed_estimate': 90,
        })
        assert panic is not None
        assert euphoria is not None
        assert panic > 80  # 恐慌 = 逆向看多
        assert euphoria < 20  # 狂热 = 逆向看空

    def test_compute_all_scores_with_data(self):
        """完整计算流程"""
        scores = self.gf.compute_all_scores(
            grok_sentiment={
                'sentiment_score': 0.5,
                'confidence': 0.7,
                'post_count': 20,
                'key_topics': ['earnings', 'AI'],
                'bullish_signals': ['strong guidance'],
                'bearish_signals': [],
                'event_risk': 'medium',
            },
            grok_market={
                'market_mood': 'optimistic',
                'fear_greed_estimate': 65,
                'key_events': ['Fed meeting'],
                'sector_rotation': {'hot_sectors': ['Tech'], 'cold_sectors': ['Energy']},
                'risk_alerts': [],
            },
        )
        assert 'AI舆情' in scores
        assert 'AI市场情绪' in scores
        assert 'AI事件风险' in scores
        assert scores['AI事件风险'] == 'medium'
        assert 'AI热门板块' in scores

    def test_compute_all_scores_empty(self):
        """无数据时返回空字典"""
        scores = self.gf.compute_all_scores(None, None)
        assert scores == {}

    def test_extract_risk_warnings(self):
        """风险预警提取"""
        warnings = self.gf.extract_risk_warnings(
            grok_sentiment={
                'event_risk': 'high',
                'bearish_signals': ['a', 'b', 'c'],
            },
            grok_market={
                'risk_alerts': ['Treasury yields spiking'],
            },
        )
        assert len(warnings) >= 2
        assert any('重大事件风险' in w for w in warnings)

    def test_none_input(self):
        """None 输入不崩溃"""
        assert self.gf.stock_sentiment_score(None) is None
        assert self.gf.stock_sentiment_score({}) is None
        assert self.gf.market_mood_score(None) is None
        assert self.gf.market_mood_score({}) is None


# ==================== 市场状态识别测试 ====================

class TestMarketRegimeHMM:
    def setup_method(self):
        from src.factors.macro_factors import MarketRegimeHMM, MarketRegime
        self.detector = MarketRegimeHMM()
        self.MarketRegime = MarketRegime

    def _make_df(self, trend='up', days=150, volatility=0.02):
        """生成模拟行情数据"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        np.random.seed(42)

        if trend == 'up':
            drift = 0.001
        elif trend == 'down':
            drift = -0.002
        elif trend == 'crisis':
            drift = -0.005
            volatility = 0.06
        else:
            drift = 0.0

        returns = np.random.normal(drift, volatility, days)
        prices = 100 * np.exp(np.cumsum(returns))
        volume = np.random.randint(1000000, 5000000, days)

        return pd.DataFrame({
            'close': prices,
            'open': prices * 0.999,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'volume': volume,
        }, index=dates)

    def test_rules_engine_bull(self):
        """规则引擎: 上涨趋势 → Bull"""
        regime, conf, desc = self.detector._detect_rules(
            self._make_df('up'),
            adx=28, vol20=0.02, m20=0.08, rsi=60,
        )
        assert regime == self.MarketRegime.BULL

    def test_rules_engine_bear(self):
        """规则引擎: 下跌趋势 → Bear"""
        regime, conf, desc = self.detector._detect_rules(
            self._make_df('down'),
            adx=25, vol20=0.03, m20=-0.08, rsi=35,
        )
        assert regime == self.MarketRegime.BEAR

    def test_rules_engine_crisis(self):
        """规则引擎: 高波动+大跌 → Crisis"""
        regime, conf, desc = self.detector._detect_rules(
            self._make_df('crisis'),
            adx=30, vol20=0.06, m20=-0.15, rsi=20,
        )
        assert regime == self.MarketRegime.CRISIS

    def test_rules_engine_sideways(self):
        """规则引擎: 无明显趋势 → Sideways"""
        regime, conf, desc = self.detector._detect_rules(
            self._make_df('sideways'),
            adx=15, vol20=0.02, m20=0.01, rsi=50,
        )
        assert regime == self.MarketRegime.SIDEWAYS

    def test_strategy_modifier(self):
        """策略增益系数"""
        # 牛市: 动量策略应有正增益
        bull_mom = self.detector.get_strategy_modifier(
            self.MarketRegime.BULL, 'momentum'
        )
        assert bull_mom > 1.0

        # 熊市: 动量策略应有负增益
        bear_mom = self.detector.get_strategy_modifier(
            self.MarketRegime.BEAR, 'momentum'
        )
        assert bear_mom < 1.0

        # 危机: 低波动防御策略应有正增益
        crisis_lowvol = self.detector.get_strategy_modifier(
            self.MarketRegime.CRISIS, 'low_vol'
        )
        assert crisis_lowvol > 1.0

    def test_detect_regime_auto(self):
        """完整 detect_regime 流程（自动选择引擎）"""
        df = self._make_df('up', days=150)
        regime, conf, desc = self.detector.detect_regime(df)
        assert isinstance(regime, self.MarketRegime)
        assert 0 <= conf <= 1
        assert len(desc) > 0

    def test_short_data_fallback(self):
        """数据不足时降级到规则引擎"""
        df = self._make_df('up', days=30)
        regime, conf, desc = self.detector.detect_regime(df)
        assert isinstance(regime, self.MarketRegime)
        assert '规则引擎' in desc


# ==================== 宏观周期测试 ====================

class TestMacroRegimeDetector:
    def setup_method(self):
        from src.factors.macro_factors import MacroRegimeDetector, MacroCycle
        self.detector = MacroRegimeDetector()
        self.MacroCycle = MacroCycle

    def test_expansion(self):
        """PMI > 52 → 扩张期"""
        pmi_series = pd.Series([51, 52, 53, 54, 55, 56])
        result = self.detector.detect_cycle({'pmi': pd.DataFrame({'value': pmi_series})})
        assert result is not None
        cycle, desc = result
        assert cycle == self.MacroCycle.EXPANSION

    def test_contraction(self):
        """PMI < 48 + 下降趋势 → 收缩期"""
        pmi_series = pd.Series([50, 49, 48, 47, 46, 45])
        result = self.detector.detect_cycle({'pmi': pd.DataFrame({'value': pmi_series})})
        assert result is not None
        cycle, desc = result
        assert cycle == self.MacroCycle.CONTRACTION

    def test_trough(self):
        """PMI < 48 但回升 → 低谷"""
        pmi_series = pd.Series([44, 43, 42, 43, 45, 47])
        result = self.detector.detect_cycle({'pmi': pd.DataFrame({'value': pmi_series})})
        assert result is not None
        cycle, desc = result
        assert cycle == self.MacroCycle.TROUGH

    def test_no_data(self):
        """无数据 → None"""
        assert self.detector.detect_cycle(None) is None
        assert self.detector.detect_cycle({}) is None

    def test_strategy_modifier(self):
        """宏观周期策略修正"""
        mod = self.detector.get_strategy_modifier(self.MacroCycle.EXPANSION, 'momentum')
        assert mod > 1.0

        mod = self.detector.get_strategy_modifier(self.MacroCycle.CONTRACTION, 'low_vol')
        assert mod > 1.0


# ==================== EnsembleStrategy 适配器测试 ====================

class TestInterpretableStrategyAdapter:
    def _make_df(self, days=120):
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.normal(0.1, 1, days))
        return pd.DataFrame({
            'open': close * 0.999,
            'high': close * 1.01,
            'low': close * 0.99,
            'close': close,
            'volume': np.random.randint(1000000, 5000000, days),
        }, index=dates)

    def test_adapter_basic(self):
        """适配器基本流程: InterpretableStrategy → EnsembleStrategy 信号"""
        from src.strategy.interpretable_strategy import BalancedMultiFactorStrategy
        from src.strategy.ensemble_strategy import InterpretableStrategyAdapter

        strategy = BalancedMultiFactorStrategy()
        adapter = InterpretableStrategyAdapter(strategy)

        df = self._make_df()
        signals = adapter.generate_signals(df, '2025-01-01', context={
            'code': 'TEST',
            'name': 'Test Stock',
        })

        # 应返回列表
        assert isinstance(signals, list)

        # 如果有信号，验证格式
        if signals:
            sig = signals[0]
            assert 'action' in sig
            assert 'confidence' in sig
            assert 0 <= sig['confidence'] <= 1  # Ensemble 格式 0-1
            assert sig['action'] in ('buy', 'sell', 'add', 'reduce')

    def test_adapter_hold_returns_empty(self):
        """持有信号 → 空列表"""
        from src.strategy.ensemble_strategy import InterpretableStrategyAdapter

        class MockHoldStrategy:
            class_name = "MockHold"
            def __init__(self):
                self.__class__.__name__ = "MockHoldStrategy"
            def analyze_stock(self, **kwargs):
                from src.strategy.interpretable_strategy import DecisionReport
                return DecisionReport(
                    code='TEST', name='Test', date='2025-01-01',
                    action='hold', action_cn='持有',
                    confidence=50, score=50, strategy_name='mock',
                )

        adapter = InterpretableStrategyAdapter(MockHoldStrategy())
        signals = adapter.generate_signals(pd.DataFrame(), '2025-01-01', {})
        assert signals == []


# ==================== GrokClient 初始化测试 ====================

class TestGrokClient:
    def test_init_without_key(self):
        """无 API Key → 不可用"""
        from src.external.grok_client import GrokClient
        client = GrokClient(api_key='')
        assert not client.is_available()

    def test_init_with_key(self):
        """有 API Key → 可用"""
        from src.external.grok_client import GrokClient
        client = GrokClient(api_key='test-key-123')
        assert client.is_available()

    def test_cache_mechanism(self):
        """缓存读写"""
        from src.external.grok_client import GrokClient
        client = GrokClient(api_key='test-key')
        client._set_cache('test:key', {'score': 42})
        cached = client._get_cache('test:key', 'sentiment')
        assert cached == {'score': 42}

    def test_budget_tracking(self):
        """每日预算追踪"""
        from src.external.grok_client import GrokClient
        client = GrokClient(api_key='test-key', max_daily_cost=0.01)
        assert client._check_budget()
        client._track_cost(0.02)  # 超出预算
        assert not client._check_budget()

    def test_no_api_call_without_key(self):
        """无 Key 时不发起 API 调用"""
        from src.external.grok_client import GrokClient
        client = GrokClient(api_key='')
        assert client.analyze_stock_sentiment('AAPL') is None
        assert client.analyze_market_regime() is None
        assert client.deep_stock_analysis('AAPL', {}) is None


# ==================== StrategyRouter HMM 集成测试 ====================

class TestStrategyRouterHMM:
    def _make_df(self, days=150):
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.normal(0.1, 1, days))
        return pd.DataFrame({
            'open': close * 0.999,
            'high': close * 1.01,
            'low': close * 0.99,
            'close': close,
            'volume': np.random.randint(1000000, 5000000, days),
        }, index=dates)

    def test_router_with_regime(self):
        """路由器现在输出 HMM/规则引擎 市场状态描述"""
        from src.strategy.strategy_router import StrategyRouter
        router = StrategyRouter()
        df = self._make_df()
        result = router.recommend('TEST', df)
        assert result.market_regime  # 非空
        assert result.primary_strategy in (
            'balanced', 'momentum', 'value', 'low_vol', 'reversion', 'breakout'
        )

    def test_router_short_data(self):
        """数据不足时路由器正常返回"""
        from src.strategy.strategy_router import StrategyRouter
        router = StrategyRouter()
        df = self._make_df(days=30)
        result = router.recommend('TEST', df)
        assert result.primary_strategy == 'balanced'  # 数据不足默认兜底
