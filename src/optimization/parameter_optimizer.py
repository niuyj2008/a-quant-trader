"""
参数优化引擎 - Phase 9.2

支持Grid Search和Walk-Forward优化,确保参数泛化能力
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from loguru import logger
from itertools import product
from sklearn.model_selection import TimeSeriesSplit
import time


class ParameterOptimizer:
    """
    参数优化器

    支持的优化方法:
    1. Grid Search: 网格搜索,遍历所有参数组合
    2. Walk-Forward Optimization: 滚动窗口优化,避免过拟合
    """

    def __init__(self, objective: str = 'sharpe_ratio'):
        """
        Args:
            objective: 优化目标
                - 'sharpe_ratio': 夏普比率(默认)
                - 'calmar_ratio': Calmar比率
                - 'total_return': 总收益率
                - 'win_rate': 胜率
        """
        self.objective = objective
        self.optimization_history = []

    def grid_search(self,
                   strategy_class: Any,
                   param_grid: Dict[str, List],
                   data: pd.DataFrame,
                   backtest_func: Callable,
                   cv_folds: int = 5) -> Dict:
        """
        网格搜索参数优化

        Args:
            strategy_class: 策略类
            param_grid: 参数网格,如 {'momentum_period': [10,20,30], 'ma_short': [5,10]}
            data: 历史数据
            backtest_func: 回测函数,接受(strategy, data)返回metrics_dict
            cv_folds: K-Fold交叉验证折数

        Returns:
            优化结果字典
        """
        logger.info(f"开始Grid Search: {len(param_grid)}维参数")

        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(product(*param_values))

        total_combinations = len(all_combinations)
        logger.info(f"  总组合数: {total_combinations}")

        results = []

        for idx, combo in enumerate(all_combinations, 1):
            params = dict(zip(param_names, combo))

            if idx % max(1, total_combinations // 10) == 0:
                logger.info(f"  进度: {idx}/{total_combinations} ({idx/total_combinations:.0%})")

            # K-Fold交叉验证
            cv_scores = self._cross_validate(
                strategy_class, params, data, backtest_func, cv_folds
            )

            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)

            results.append({
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'cv_scores': cv_scores,
                'stability': mean_score / std_score if std_score > 0 else 0,
            })

        # 按平均得分排序
        results.sort(key=lambda x: x['mean_score'], reverse=True)

        # 记录历史
        self.optimization_history.append({
            'method': 'grid_search',
            'param_grid': param_grid,
            'best_params': results[0]['params'],
            'best_score': results[0]['mean_score'],
            'n_combinations': total_combinations,
        })

        logger.info(f"Grid Search完成: 最优{self.objective}={results[0]['mean_score']:.4f}")

        return {
            'best_params': results[0]['params'],
            'best_score': results[0]['mean_score'],
            'all_results': results,
            'top_10': results[:10],
        }

    def _cross_validate(self,
                       strategy_class: Any,
                       params: Dict,
                       data: pd.DataFrame,
                       backtest_func: Callable,
                       cv_folds: int) -> List[float]:
        """K-Fold交叉验证"""
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        scores = []

        for train_idx, val_idx in tscv.split(data):
            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]

            # 创建策略实例
            strategy = strategy_class(**params)

            # 回测
            try:
                result = backtest_func(strategy, val_data)
                score = result.get(self.objective, 0)
                scores.append(score)
            except Exception as e:
                logger.warning(f"回测失败: {params}, 错误={e}")
                scores.append(-np.inf)

        return scores

    def walk_forward_optimization(self,
                                  strategy_class: Any,
                                  param_grid: Dict[str, List],
                                  data: pd.DataFrame,
                                  backtest_func: Callable,
                                  train_period: int = 252,
                                  test_period: int = 63) -> Dict:
        """
        Walk-Forward优化(最接近实盘)

        流程:
        1. 在训练期优化参数(Grid Search)
        2. 在验证期使用最优参数回测
        3. 滚动窗口重复

        Args:
            train_period: 训练期天数(默认252天=1年)
            test_period: 测试期天数(默认63天=3个月)

        Returns:
            优化结果字典
        """
        logger.info(f"开始Walk-Forward优化: 训练{train_period}天, 测试{test_period}天")

        dates = data.index
        results = []

        window_idx = 0
        for i in range(0, len(dates) - train_period - test_period, test_period):
            window_idx += 1

            train_start_idx = i
            train_end_idx = i + train_period
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + test_period

            train_data = data.iloc[train_start_idx:train_end_idx]
            test_data = data.iloc[test_start_idx:test_end_idx]

            logger.info(f"  窗口{window_idx}: 训练期{train_data.index[0]} ~ {train_data.index[-1]}")

            # 在训练期优化参数
            opt_result = self.grid_search(
                strategy_class, param_grid, train_data, backtest_func, cv_folds=3
            )

            best_params = opt_result['best_params']
            train_score = opt_result['best_score']

            # 在测试期验证
            strategy = strategy_class(**best_params)
            test_result = backtest_func(strategy, test_data)
            test_score = test_result.get(self.objective, 0)

            logger.info(f"    训练期{self.objective}={train_score:.4f}, 测试期={test_score:.4f}")

            results.append({
                'window': window_idx,
                'train_period': f"{train_data.index[0]} ~ {train_data.index[-1]}",
                'test_period': f"{test_data.index[0]} ~ {test_data.index[-1]}",
                'best_params': best_params,
                'train_score': train_score,
                'test_score': test_score,
                'performance_ratio': test_score / train_score if train_score > 0 else 0,
            })

        # 分析参数稳定性
        param_stability = self._analyze_param_stability(results)

        # 推荐参数(出现频率最高的参数组合)
        recommended_params = self._get_most_frequent_params(results)

        logger.info(f"Walk-Forward优化完成: {len(results)}个窗口")
        logger.info(f"  平均测试期{self.objective}={np.mean([r['test_score'] for r in results]):.4f}")
        logger.info(f"  推荐参数: {recommended_params}")

        return {
            'results': results,
            'n_windows': len(results),
            'avg_train_score': np.mean([r['train_score'] for r in results]),
            'avg_test_score': np.mean([r['test_score'] for r in results]),
            'param_stability': param_stability,
            'recommended_params': recommended_params,
        }

    def _analyze_param_stability(self, results: List[Dict]) -> Dict:
        """分析参数稳定性"""
        # 统计每个参数的值分布
        all_params = {}
        for r in results:
            for param_name, param_value in r['best_params'].items():
                if param_name not in all_params:
                    all_params[param_name] = []
                all_params[param_name].append(param_value)

        # 计算每个参数的标准差(归一化)
        stability = {}
        for param_name, values in all_params.items():
            unique_values = len(set(values))
            total_values = len(values)

            # 稳定性 = 1 - (唯一值数 / 总数)
            # 如果所有窗口都用同一个值,稳定性=1.0
            # 如果每个窗口都不同,稳定性=0.0
            stability_score = 1.0 - (unique_values - 1) / total_values

            stability[param_name] = {
                'score': stability_score,
                'unique_values': unique_values,
                'total_values': total_values,
                'value_distribution': dict(pd.Series(values).value_counts()),
            }

        return stability

    def _get_most_frequent_params(self, results: List[Dict]) -> Dict:
        """获取出现频率最高的参数组合"""
        # 统计每个参数值的出现次数
        param_counts = {}

        for r in results:
            for param_name, param_value in r['best_params'].items():
                if param_name not in param_counts:
                    param_counts[param_name] = {}

                if param_value not in param_counts[param_name]:
                    param_counts[param_name][param_value] = 0

                param_counts[param_name][param_value] += 1

        # 选择每个参数出现最频繁的值
        most_frequent = {}
        for param_name, value_counts in param_counts.items():
            most_frequent[param_name] = max(value_counts, key=value_counts.get)

        return most_frequent

    def random_search(self,
                     strategy_class: Any,
                     param_distributions: Dict[str, Tuple],
                     data: pd.DataFrame,
                     backtest_func: Callable,
                     n_iter: int = 50,
                     cv_folds: int = 5) -> Dict:
        """
        随机搜索参数优化(适合高维参数空间)

        Args:
            param_distributions: 参数分布,如 {'momentum_period': (10, 60), 'ma_short': (5, 20)}
            n_iter: 随机采样次数

        Returns:
            优化结果字典
        """
        logger.info(f"开始Random Search: {n_iter}次随机采样")

        results = []

        for idx in range(n_iter):
            # 随机采样参数
            params = {}
            for param_name, (min_val, max_val) in param_distributions.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    params[param_name] = np.random.uniform(min_val, max_val)

            if (idx + 1) % max(1, n_iter // 10) == 0:
                logger.info(f"  进度: {idx+1}/{n_iter} ({(idx+1)/n_iter:.0%})")

            # K-Fold交叉验证
            cv_scores = self._cross_validate(
                strategy_class, params, data, backtest_func, cv_folds
            )

            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)

            results.append({
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'cv_scores': cv_scores,
            })

        # 排序
        results.sort(key=lambda x: x['mean_score'], reverse=True)

        logger.info(f"Random Search完成: 最优{self.objective}={results[0]['mean_score']:.4f}")

        return {
            'best_params': results[0]['params'],
            'best_score': results[0]['mean_score'],
            'all_results': results,
            'top_10': results[:10],
        }

    def generate_optimization_report(self, optimization_result: Dict,
                                    method: str = 'grid_search') -> str:
        """
        生成参数优化报告

        Args:
            optimization_result: 优化结果
            method: 优化方法('grid_search', 'walk_forward', 'random_search')

        Returns:
            报告字符串
        """
        report = []
        report.append("=" * 70)
        report.append(f"参数优化报告 - {method.upper()}")
        report.append("=" * 70)

        if method == 'walk_forward':
            report.append(f"\n📊 Walk-Forward优化结果:")
            report.append(f"  窗口数: {optimization_result['n_windows']}")
            report.append(f"  平均训练期{self.objective}: {optimization_result['avg_train_score']:.4f}")
            report.append(f"  平均测试期{self.objective}: {optimization_result['avg_test_score']:.4f}")
            report.append(f"  泛化能力: {optimization_result['avg_test_score']/optimization_result['avg_train_score']:.2%}")

            report.append(f"\n\n🎯 推荐参数:")
            for param_name, param_value in optimization_result['recommended_params'].items():
                report.append(f"  {param_name}: {param_value}")

            report.append(f"\n\n📈 参数稳定性:")
            for param_name, stability_info in optimization_result['param_stability'].items():
                report.append(f"  {param_name}:")
                report.append(f"    稳定性得分: {stability_info['score']:.2f}")
                report.append(f"    唯一值数: {stability_info['unique_values']}/{stability_info['total_values']}")
                report.append(f"    值分布: {stability_info['value_distribution']}")

        else:  # grid_search or random_search
            report.append(f"\n🏆 最优参数:")
            for param_name, param_value in optimization_result['best_params'].items():
                report.append(f"  {param_name}: {param_value}")

            report.append(f"\n  {self.objective}: {optimization_result['best_score']:.4f}")

            if 'top_10' in optimization_result:
                report.append(f"\n\n📊 Top 10参数组合:")
                for i, result in enumerate(optimization_result['top_10'][:10], 1):
                    report.append(f"\n  {i}. {self.objective}={result['mean_score']:.4f}")
                    report.append(f"     参数: {result['params']}")

        report.append("\n" + "=" * 70)

        return "\n".join(report)


def quick_optimize(strategy_class: Any,
                  param_grid: Dict[str, List],
                  data: pd.DataFrame,
                  backtest_func: Callable,
                  method: str = 'grid_search',
                  objective: str = 'sharpe_ratio') -> Dict:
    """
    快速参数优化(便捷函数)

    Args:
        method: 优化方法('grid_search', 'walk_forward', 'random_search')

    Returns:
        优化结果
    """
    optimizer = ParameterOptimizer(objective=objective)

    if method == 'walk_forward':
        result = optimizer.walk_forward_optimization(
            strategy_class, param_grid, data, backtest_func
        )
    elif method == 'random_search':
        # 将param_grid转为param_distributions
        param_distributions = {
            name: (min(values), max(values))
            for name, values in param_grid.items()
        }
        result = optimizer.random_search(
            strategy_class, param_distributions, data, backtest_func, n_iter=50
        )
    else:  # grid_search
        result = optimizer.grid_search(
            strategy_class, param_grid, data, backtest_func, cv_folds=5
        )

    # 生成报告
    report = optimizer.generate_optimization_report(result, method)

    return {
        **result,
        'report': report,
    }
