"""
Grok AI 客户端 - 基于 xAI 官方 API

利用 Grok 的实时 X/Web 搜索能力获取市场情绪和个股舆情分析。
API 格式兼容 OpenAI，使用 https://api.x.ai/v1 端点。

设计原则:
  - 结构化输出: 所有分析结果强制 JSON Schema 返回，便于因子化
  - 多级缓存: 情绪4h、市场1h、深度分析24h
  - 成本控制: 每日预算上限，批量任务用 grok-3-mini
  - 优雅降级: API不可用时返回 None，策略层自动跳过
"""

import os
import time
import json
import hashlib
from typing import Dict, Optional, List
from loguru import logger


# ==================== 结构化输出 JSON Schema ====================

STOCK_SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment_score": {
            "type": "number",
            "description": "Overall sentiment from -1.0 (extremely bearish) to 1.0 (extremely bullish)"
        },
        "confidence": {
            "type": "number",
            "description": "Confidence level 0-1 based on data quality and consistency"
        },
        "post_count": {
            "type": "integer",
            "description": "Approximate number of relevant posts/discussions found"
        },
        "key_topics": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Top 3-5 discussion topics"
        },
        "bullish_signals": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key bullish arguments found"
        },
        "bearish_signals": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key bearish arguments found"
        },
        "event_risk": {
            "type": "string",
            "enum": ["none", "low", "medium", "high"],
            "description": "Upcoming event risk level (earnings, FDA, lawsuit, etc.)"
        }
    },
    "required": ["sentiment_score", "confidence", "post_count", "key_topics",
                  "bullish_signals", "bearish_signals", "event_risk"],
    "additionalProperties": False,
}

MARKET_REGIME_SCHEMA = {
    "type": "object",
    "properties": {
        "market_mood": {
            "type": "string",
            "enum": ["euphoria", "optimistic", "neutral", "anxious", "panic"],
            "description": "Current overall market mood"
        },
        "fear_greed_estimate": {
            "type": "integer",
            "description": "Estimated fear/greed index 0-100 (0=extreme fear, 100=extreme greed)"
        },
        "key_events": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Top 3-5 market-moving events in the past 24h"
        },
        "sector_rotation": {
            "type": "object",
            "properties": {
                "hot_sectors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Sectors receiving positive attention"
                },
                "cold_sectors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Sectors receiving negative attention"
                }
            },
            "required": ["hot_sectors", "cold_sectors"],
            "additionalProperties": False,
        },
        "risk_alerts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Significant risk warnings if any"
        }
    },
    "required": ["market_mood", "fear_greed_estimate", "key_events",
                  "sector_rotation", "risk_alerts"],
    "additionalProperties": False,
}

DEEP_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "ai_score": {
            "type": "number",
            "description": "AI综合评分 0-100"
        },
        "narrative_summary": {
            "type": "string",
            "description": "一段话总结当前股票的投资逻辑"
        },
        "catalysts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "近期可能的催化剂事件"
        },
        "risks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "主要风险因素"
        },
        "contrarian_view": {
            "type": "string",
            "description": "反向观点 - 如果主流看多则给出看空理由，反之亦然"
        },
        "time_horizon": {
            "type": "string",
            "enum": ["short_term", "medium_term", "long_term"],
            "description": "建议的投资时间框架"
        }
    },
    "required": ["ai_score", "narrative_summary", "catalysts", "risks",
                  "contrarian_view", "time_horizon"],
    "additionalProperties": False,
}


class GrokClient:
    """Grok AI 客户端

    通过 xAI API 获取基于 X/Web 搜索的市场情绪分析。
    所有方法在 API 不可用时返回 None，不影响系统运行。
    """

    BASE_URL = "https://api.x.ai/v1"

    # 模型选择: 批量/简单任务用 mini，深度分析用标准版
    MODEL_MINI = "grok-3-mini"
    MODEL_STANDARD = "grok-4-1"

    # 缓存TTL（秒）
    CACHE_TTL = {
        'sentiment': 4 * 3600,    # 个股情绪: 4小时
        'market': 1 * 3600,       # 市场状态: 1小时
        'analysis': 24 * 3600,    # 深度分析: 24小时
    }

    def __init__(self, api_key: Optional[str] = None,
                 max_daily_cost: float = 5.0,
                 enabled: bool = True):
        """
        Args:
            api_key: xAI API Key，默认从环境变量 XAI_API_KEY 读取
            max_daily_cost: 每日预算上限（美元）
            enabled: 是否启用
        """
        self.api_key = api_key or os.environ.get('XAI_API_KEY', '')
        self.max_daily_cost = max_daily_cost
        self.enabled = enabled and bool(self.api_key)

        # 缓存: key -> (timestamp, result)
        self._cache: Dict[str, tuple] = {}

        # 成本追踪
        self._daily_cost = 0.0
        self._cost_reset_date = ''

        if not self.enabled:
            logger.info("Grok AI 未启用（缺少 XAI_API_KEY 或 enabled=False）")

    def is_available(self) -> bool:
        """检查 Grok API 是否可用"""
        return self.enabled and bool(self.api_key)

    # ==================== 核心API方法 ====================

    def analyze_stock_sentiment(self, ticker: str,
                                market: str = "US") -> Optional[Dict]:
        """分析个股社交媒体情绪

        利用 Grok 的 X 搜索能力分析该股票在社交媒体上的讨论情绪。

        Args:
            ticker: 股票代码 (e.g., "AAPL", "600519")
            market: "US" 或 "CN"

        Returns:
            结构化情绪数据 (符合 STOCK_SENTIMENT_SCHEMA)，失败返回 None
        """
        if not self.is_available():
            return None

        cache_key = f"sentiment:{ticker}:{market}"
        cached = self._get_cache(cache_key, 'sentiment')
        if cached is not None:
            return cached

        if market == "CN":
            prompt = (
                f"搜索X和网络上关于A股 {ticker} 的最新讨论和新闻。"
                f"分析过去48小时内投资者对该股票的情绪倾向。"
                f"注意区分散户情绪和机构观点，重点关注有实质内容的分析帖子。"
                f"如果讨论量很少，请降低 confidence 值。"
            )
        else:
            prompt = (
                f"Search X posts and web for recent discussions about ${ticker} stock. "
                f"Analyze investor sentiment over the past 48 hours. "
                f"Focus on substantive analysis posts, not just price mentions. "
                f"Distinguish between retail and institutional sentiment. "
                f"Check for upcoming catalysts (earnings, FDA decisions, etc). "
                f"If discussion volume is low, reduce confidence accordingly."
            )

        result = self._call_api(
            prompt=prompt,
            schema=STOCK_SENTIMENT_SCHEMA,
            schema_name="stock_sentiment",
            model=self.MODEL_MINI,
            search_enabled=True,
        )

        if result:
            self._set_cache(cache_key, result)

        return result

    def analyze_market_regime(self, market: str = "US") -> Optional[Dict]:
        """分析当前市场整体情绪和状态

        Args:
            market: "US" 或 "CN"

        Returns:
            结构化市场状态数据 (符合 MARKET_REGIME_SCHEMA)
        """
        if not self.is_available():
            return None

        cache_key = f"market:{market}"
        cached = self._get_cache(cache_key, 'market')
        if cached is not None:
            return cached

        if market == "CN":
            prompt = (
                "搜索X和网络上关于A股市场的最新讨论。"
                "分析当前市场整体情绪（恐慌/焦虑/中性/乐观/狂热）。"
                "关注: 1) 过去24小时重大市场事件 2) 板块轮动方向 "
                "3) 政策面消息 4) 外资动向。"
                "给出风险预警（如果有的话）。"
            )
        else:
            prompt = (
                "Search X and web for the latest US stock market discussions. "
                "Analyze overall market sentiment (panic/anxious/neutral/optimistic/euphoria). "
                "Focus on: 1) Major market events in past 24h 2) Sector rotation trends "
                "3) Fed/policy signals 4) Geopolitical risks. "
                "Provide risk alerts if warranted."
            )

        result = self._call_api(
            prompt=prompt,
            schema=MARKET_REGIME_SCHEMA,
            schema_name="market_regime",
            model=self.MODEL_MINI,
            search_enabled=True,
        )

        if result:
            self._set_cache(cache_key, result)

        return result

    def deep_stock_analysis(self, ticker: str, factor_summary: Dict,
                           market: str = "US") -> Optional[Dict]:
        """深度个股分析 - 将量化因子与AI分析结合

        将系统计算的量化因子发送给 Grok，获取AI增强分析。

        Args:
            ticker: 股票代码
            factor_summary: 量化因子摘要 (e.g., {'综合得分': 72, '技术面': 65, ...})
            market: "US" 或 "CN"

        Returns:
            结构化AI分析结果 (符合 DEEP_ANALYSIS_SCHEMA)
        """
        if not self.is_available():
            return None

        cache_key = f"analysis:{ticker}:{market}"
        cached = self._get_cache(cache_key, 'analysis')
        if cached is not None:
            return cached

        factor_str = ", ".join(f"{k}={v}" for k, v in factor_summary.items())

        if market == "CN":
            prompt = (
                f"对A股 {ticker} 进行深度分析。"
                f"我的量化系统给出的因子评分: {factor_str}。"
                f"请结合最新新闻和X上的讨论，从以下角度分析: "
                f"1) 综合AI评分(0-100) 2) 投资逻辑总结 "
                f"3) 近期催化剂 4) 主要风险 5) 反向观点。"
            )
        else:
            prompt = (
                f"Perform deep analysis on ${ticker}. "
                f"My quant system's factor scores: {factor_str}. "
                f"Combine with latest news and X discussions to analyze: "
                f"1) Overall AI score (0-100) 2) Investment thesis summary "
                f"3) Near-term catalysts 4) Key risks 5) Contrarian view. "
                f"Be objective - challenge the quant scores if warranted."
            )

        result = self._call_api(
            prompt=prompt,
            schema=DEEP_ANALYSIS_SCHEMA,
            schema_name="deep_analysis",
            model=self.MODEL_STANDARD,
            search_enabled=True,
        )

        if result:
            self._set_cache(cache_key, result)

        return result

    # ==================== 内部方法 ====================

    def _call_api(self, prompt: str, schema: Dict, schema_name: str,
                  model: str, search_enabled: bool = False) -> Optional[Dict]:
        """调用 xAI API

        Args:
            prompt: 用户提示
            schema: JSON Schema 用于结构化输出
            schema_name: Schema 名称
            model: 模型名称
            search_enabled: 是否启用搜索

        Returns:
            解析后的 JSON 结果，失败返回 None
        """
        if not self._check_budget():
            logger.warning("Grok API 每日预算已用完")
            return None

        try:
            import openai
        except ImportError:
            logger.warning("openai 包未安装，无法使用 Grok API。请运行: pip install openai")
            return None

        try:
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.BASE_URL,
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a financial market analyst with access to real-time X posts and web data. "
                        "Provide objective, data-driven analysis. Always clearly state uncertainty. "
                        "Do NOT provide investment advice - only factual analysis and sentiment assessment."
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            kwargs = {
                "model": model,
                "messages": messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
            }

            # 部分模型支持搜索
            if search_enabled:
                kwargs["search"] = {
                    "mode": "auto",
                    "sources": [
                        {"type": "x", "x_handles": []},
                        {"type": "web"},
                    ],
                }

            response = client.chat.completions.create(**kwargs)

            # 解析结果
            content = response.choices[0].message.content
            result = json.loads(content)

            # 估算成本（粗略）
            usage = getattr(response, 'usage', None)
            if usage:
                input_tokens = getattr(usage, 'prompt_tokens', 0)
                output_tokens = getattr(usage, 'completion_tokens', 0)
                # grok-3-mini: ~$0.30/M input, ~$0.50/M output
                # grok-4-1: ~$3/M input, ~$15/M output
                if 'mini' in model:
                    cost = input_tokens * 0.3e-6 + output_tokens * 0.5e-6
                else:
                    cost = input_tokens * 3e-6 + output_tokens * 15e-6
                self._track_cost(cost)

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Grok API 返回非法JSON: {e}")
            return None
        except Exception as e:
            logger.warning(f"Grok API 调用失败: {e}")
            return None

    def _get_cache(self, key: str, cache_type: str) -> Optional[Dict]:
        """获取缓存"""
        if key in self._cache:
            ts, data = self._cache[key]
            ttl = self.CACHE_TTL.get(cache_type, 3600)
            if time.time() - ts < ttl:
                return data
            del self._cache[key]
        return None

    def _set_cache(self, key: str, data: Dict):
        """设置缓存"""
        self._cache[key] = (time.time(), data)

    def _check_budget(self) -> bool:
        """检查每日预算"""
        today = time.strftime('%Y-%m-%d')
        if self._cost_reset_date != today:
            self._cost_reset_date = today
            self._daily_cost = 0.0
        return self._daily_cost < self.max_daily_cost

    def _track_cost(self, cost: float):
        """记录API成本"""
        today = time.strftime('%Y-%m-%d')
        if self._cost_reset_date != today:
            self._cost_reset_date = today
            self._daily_cost = 0.0
        self._daily_cost += cost
        logger.debug(f"Grok API 本次成本: ${cost:.4f}, 今日累计: ${self._daily_cost:.4f}")

    def get_daily_cost(self) -> float:
        """获取今日累计成本"""
        today = time.strftime('%Y-%m-%d')
        if self._cost_reset_date != today:
            return 0.0
        return self._daily_cost

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
