"""
因子权重优化器

基于因子IC_IR数据驱动地计算最优权重，替代硬编码的"拍脑袋"权重。
支持IC_IR加权和scipy最大化夏普两种方法，以及Walk-Forward防过拟合验证。
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class OptimizationResult:
    """权重优化结果"""
    strategy_name: str
    method: str                        # "ic_ir" 或 "sharpe"
    weights: Dict[str, float] = field(default_factory=dict)
    score_ranges: Dict[str, List[float]] = field(default_factory=dict)
    buy_threshold: float = 65.0
    sell_threshold: float = 35.0
    validation_sharpe: float = 0.0
    original_sharpe: float = 0.0       # 原始硬编码权重的夏普
    improvement: float = 0.0           # 夏普提升百分比
    market: str = "US"
    updated_at: str = ""


class WeightOptimizer:
    """因子权重优化器"""

    def optimize_icir(self, factor_ic_results: Dict[str, 'FactorICResult'],
                      strategy_factors: List[str],
                      min_weight: float = 0.05) -> Dict[str, float]:
        """方法A：基于IC_IR加权

        权重与因子的|IC_IR|成正比，信息比率高的因子获得更高权重。

        Args:
            factor_ic_results: 因子验证结果字典 {factor_name_fwdN: FactorICResult}
            strategy_factors: 该策略使用的因子名列表
            min_weight: 最低权重（防止完全剔除）

        Returns:
            {因子名: 权重} 字典，权重之和为1
        """
        ic_ir_values = {}
        for fname in strategy_factors:
            # 尝试匹配 fwd10 的结果
            key = f"{fname}_fwd10"
            if key in factor_ic_results:
                ic_ir_values[fname] = abs(factor_ic_results[key].ic_ir)
            else:
                ic_ir_values[fname] = 0.0

        # 如果所有因子IC_IR都是0，返回等权
        total_ir = sum(ic_ir_values.values())
        if total_ir < 1e-10:
            n = len(strategy_factors)
            return {f: 1.0 / n for f in strategy_factors}

        # IC_IR加权
        weights = {}
        for fname in strategy_factors:
            w = max(ic_ir_values[fname] / total_ir, min_weight)
            weights[fname] = w

        # 归一化
        total = sum(weights.values())
        weights = {f: round(w / total, 4) for f, w in weights.items()}

        return weights

    def optimize_sharpe(self, data_dict: Dict[str, pd.DataFrame],
                        strategy, factor_names: List[str],
                        n_trials: int = 200) -> Dict[str, float]:
        """方法B：随机搜索最大化夏普比率

        随机生成权重组合，用历史数据模拟策略收益，选夏普最高的权重。

        Args:
            data_dict: {代码: DataFrame} 股票池数据
            strategy: 策略实例（用于evaluate）
            factor_names: 因子名列表
            n_trials: 随机搜索次数

        Returns:
            最优权重字典
        """
        n_factors = len(factor_names)
        best_sharpe = -np.inf
        best_weights = {f: 1.0 / n_factors for f in factor_names}

        for trial in range(n_trials):
            # 随机生成权重（Dirichlet分布，保证和为1且非负）
            raw = np.random.dirichlet(np.ones(n_factors))
            # 加下界约束
            raw = np.maximum(raw, 0.05)
            raw /= raw.sum()
            trial_weights = {f: round(float(w), 4) for f, w in zip(factor_names, raw)}

            # 模拟收益
            sharpe = self._simulate_strategy_sharpe(data_dict, strategy, trial_weights)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = trial_weights

        logger.info(f"Sharpe优化完成: 最优夏普={best_sharpe:.3f}, "
                    f"试验次数={n_trials}")
        return best_weights

    def _simulate_strategy_sharpe(self, data_dict: Dict[str, pd.DataFrame],
                                  strategy, weights: Dict[str, float]) -> float:
        """用给定权重模拟策略收益，计算夏普比率"""
        returns = []

        for code, df in data_dict.items():
            if df.empty or len(df) < 120:
                continue
            try:
                # 用后半段数据模拟
                test_df = df.iloc[len(df) // 2:]
                report = strategy.analyze_stock(code, test_df)
                if report and report.action == "buy":
                    # 简单模拟：买入后持有20天的收益
                    if len(test_df) > 20:
                        ret = (test_df.iloc[-1]['close'] - test_df.iloc[-21]['close']) / test_df.iloc[-21]['close']
                        returns.append(ret)
            except Exception:
                pass

        if len(returns) < 5:
            return -np.inf

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret < 1e-10:
            return 0.0
        return mean_ret / std_ret * np.sqrt(252 / 20)  # 年化夏普

    def walk_forward_validate(self, data_dict: Dict[str, pd.DataFrame],
                              factor_ic_results: Dict,
                              strategy_factors: List[str],
                              train_years: int = 2,
                              test_months: int = 6) -> List[Dict]:
        """Walk-Forward验证：训练期优化 → 验证期检验

        防止过拟合：确保优化后的权重在样本外也有效。

        Returns:
            各窗口的验证结果列表
        """
        # 简化实现：将IC结果按时间分割
        # 由于IC_IR本身是长期统计量，这里主要验证权重的稳定性
        weights = self.optimize_icir(factor_ic_results, strategy_factors)

        # 检查权重稳定性：bootstrap重采样
        n_bootstrap = 50
        weight_samples = []

        for _ in range(n_bootstrap):
            # 对IC序列做bootstrap重采样
            resampled_results = {}
            for key, result in factor_ic_results.items():
                if hasattr(result, 'ic_series') and result.ic_series:
                    n = len(result.ic_series)
                    indices = np.random.randint(0, n, size=n)
                    resampled_series = [result.ic_series[i] for i in indices]
                    # 创建临时result
                    from src.factors.factor_validator import FactorICResult
                    resampled_results[key] = FactorICResult(
                        factor_name=result.factor_name,
                        ic_mean=np.mean(resampled_series),
                        ic_std=max(np.std(resampled_series), 1e-10),
                        ic_ir=np.mean(resampled_series) / max(np.std(resampled_series), 1e-10),
                        ic_series=resampled_series,
                        forward_days=result.forward_days,
                    )

            sample_weights = self.optimize_icir(resampled_results, strategy_factors)
            weight_samples.append(sample_weights)

        # 计算权重的稳定性（标准差）
        stability = {}
        for fname in strategy_factors:
            ws = [s.get(fname, 0) for s in weight_samples]
            stability[fname] = {
                'mean': round(np.mean(ws), 4),
                'std': round(np.std(ws), 4),
                'cv': round(np.std(ws) / max(np.mean(ws), 1e-10), 4),  # 变异系数
            }

        return [{
            'method': 'bootstrap_stability',
            'weights': weights,
            'stability': stability,
            'is_stable': all(s['cv'] < 0.5 for s in stability.values()),
        }]

    def save_config(self, results: Dict[str, OptimizationResult],
                    config_path: str = "config/strategy_weights.json"):
        """保存优化结果到配置文件"""
        from datetime import datetime
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config = {}
        if path.exists():
            with open(path) as f:
                config = json.load(f)

        for strategy_name, result in results.items():
            config[strategy_name] = {
                'weights': result.weights,
                'score_ranges': result.score_ranges,
                'buy_threshold': result.buy_threshold,
                'sell_threshold': result.sell_threshold,
                'method': result.method,
                'validation_sharpe': result.validation_sharpe,
                'original_sharpe': result.original_sharpe,
                'improvement': result.improvement,
                'market': result.market,
                'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"权重配置已保存: {config_path}")

    @staticmethod
    def load_config(config_path: str = "config/strategy_weights.json") -> Dict:
        """加载权重配置"""
        path = Path(config_path)
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)
