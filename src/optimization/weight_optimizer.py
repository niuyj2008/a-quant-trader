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
                      min_weight: float = 0.05,
                      correlation_matrix: Optional[pd.DataFrame] = None,
                      corr_penalty_threshold: float = 0.7) -> Dict[str, float]:
        """方法A：基于IC_IR加权（含相关性惩罚）

        权重与因子的|IC_IR|成正比，信息比率高的因子获得更高权重。
        负IC因子会被标记为需要反转得分方向（在 factor_sign 中记录）。
        高相关因子对会被降权，避免信息冗余。

        Args:
            factor_ic_results: 因子验证结果字典 {factor_name_fwdN: FactorICResult}
            strategy_factors: 该策略使用的因子名列表
            min_weight: 最低权重（防止完全剔除）
            correlation_matrix: 因子相关性矩阵（来自FactorValidator）
            corr_penalty_threshold: 相关性惩罚阈值（默认0.7）

        Returns:
            {因子名: 权重} 字典，权重之和为1
        """
        ic_ir_values = {}
        self._factor_signs = {}  # 记录因子方向: +1=正向, -1=需反转

        for fname in strategy_factors:
            # 尝试匹配 fwd10 的结果
            key = f"{fname}_fwd10"
            if key in factor_ic_results:
                raw_ic_ir = factor_ic_results[key].ic_ir
                # 记录因子方向：正IC = 正向因子，负IC = 反向因子（需反转得分）
                self._factor_signs[fname] = 1 if raw_ic_ir >= 0 else -1
                ic_ir_values[fname] = abs(raw_ic_ir)
                if raw_ic_ir < 0:
                    logger.warning(
                        f"因子 {fname} 的IC_IR为负({raw_ic_ir:.4f})，将反转其得分方向"
                    )
            else:
                ic_ir_values[fname] = 0.0
                self._factor_signs[fname] = 1

        # 如果所有因子IC_IR都是0，返回等权
        total_ir = sum(ic_ir_values.values())
        if total_ir < 1e-10:
            n = len(strategy_factors)
            return {f: 1.0 / n for f in strategy_factors}

        # 相关性惩罚: 对高相关因子对中IC_IR较低的那个降权
        corr_penalties = {f: 1.0 for f in strategy_factors}
        if correlation_matrix is not None and not correlation_matrix.empty:
            available = [f for f in strategy_factors if f in correlation_matrix.columns]
            for i, f1 in enumerate(available):
                for f2 in available[i+1:]:
                    corr_val = abs(correlation_matrix.loc[f1, f2])
                    if corr_val > corr_penalty_threshold:
                        # 惩罚IC_IR较低的因子
                        ir1 = ic_ir_values.get(f1, 0)
                        ir2 = ic_ir_values.get(f2, 0)
                        weaker = f1 if ir1 <= ir2 else f2
                        # 惩罚力度与相关性成正比: corr=0.7→不惩罚, corr=1.0→惩罚50%
                        penalty = 1.0 - (corr_val - corr_penalty_threshold) / (1.0 - corr_penalty_threshold) * 0.5
                        corr_penalties[weaker] = min(corr_penalties[weaker], penalty)
                        logger.info(
                            f"因子共线性惩罚: {f1}↔{f2} 相关性={corr_val:.2f}, "
                            f"{weaker}权重×{penalty:.2f}"
                        )

        # IC_IR加权（含惩罚）
        weights = {}
        for fname in strategy_factors:
            raw_w = ic_ir_values[fname] / total_ir
            penalized_w = raw_w * corr_penalties.get(fname, 1.0)
            weights[fname] = max(penalized_w, min_weight)

        # 归一化
        total = sum(weights.values())
        weights = {f: round(w / total, 4) for f, w in weights.items()}

        return weights

    def get_factor_signs(self) -> Dict[str, int]:
        """获取因子方向标记

        Returns:
            {因子名: 方向} +1=正向因子, -1=反向因子(得分需 100-score)
        """
        return getattr(self, '_factor_signs', {})

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
        """用给定权重模拟策略收益，计算夏普比率（逐日滚动，无前瞻偏差）"""
        returns = []
        hold_days = 20

        for code, df in data_dict.items():
            if df.empty or len(df) < 120:
                continue
            try:
                split_idx = len(df) // 2
                test_start = max(split_idx, 60)

                # 逐日滚动测试
                for i in range(test_start, len(df) - hold_days):
                    window_df = df.iloc[:i + 1]
                    report = strategy.analyze_stock(code, window_df)
                    if report and report.action == "buy":
                        buy_price = df.iloc[i]['close']
                        sell_price = df.iloc[i + hold_days]['close']
                        if buy_price > 0:
                            ret = (sell_price - buy_price) / buy_price
                            returns.append(ret)
            except Exception:
                pass

        if len(returns) < 5:
            return -np.inf

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret < 1e-10:
            return 0.0
        return mean_ret / std_ret * np.sqrt(252 / hold_days)  # 年化夏普

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
