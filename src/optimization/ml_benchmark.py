"""
ML算法性能对比 - Phase 9.1

对比LightGBM、XGBoost、RandomForest在因子预测任务上的表现
使用Walk-Forward交叉验证,确保结果可靠
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import ttest_rel
import time


class MLAlgorithmBenchmark:
    """
    ML算法性能对比实验

    对比算法:
    - LightGBM (主力)
    - XGBoost (对照)
    - RandomForest (辅助)
    - Ridge (基准)
    """

    def __init__(self, data: pd.DataFrame, factor_columns: List[str],
                 target_column: str = 'return_5d'):
        """
        Args:
            data: 包含因子和目标的DataFrame,按日期索引
            factor_columns: 因子列名列表
            target_column: 目标列名(如'return_5d'表示未来5日收益)
        """
        self.data = data
        self.factor_columns = factor_columns
        self.target_column = target_column

        # 准备数据
        self.X = data[factor_columns].fillna(0)
        self.y = data[target_column].fillna(0)

        logger.info(f"ML Benchmark初始化: {len(data)}样本, {len(factor_columns)}因子")

        # 待对比的模型
        self.models = self._init_models()

    def _init_models(self) -> Dict:
        """初始化待对比的模型"""
        models = {}

        # 1. LightGBM
        try:
            import lightgbm as lgb
            models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
            )
            logger.info("✅ LightGBM已加载")
        except ImportError:
            logger.warning("⚠️  LightGBM未安装,将跳过")

        # 2. XGBoost
        try:
            import xgboost as xgb
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbosity=0,
            )
            logger.info("✅ XGBoost已加载")
        except ImportError:
            logger.warning("⚠️  XGBoost未安装,将跳过")

        # 3. RandomForest
        try:
            from sklearn.ensemble import RandomForestRegressor
            models['RandomForest'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
            )
            logger.info("✅ RandomForest已加载")
        except ImportError:
            logger.warning("⚠️  RandomForest未安装,将跳过")

        # 4. Ridge (基准)
        try:
            from sklearn.linear_model import Ridge
            models['Ridge'] = Ridge(
                alpha=1.0,
                random_state=42,
            )
            logger.info("✅ Ridge已加载")
        except ImportError:
            logger.warning("⚠️  Ridge未安装,将跳过")

        if not models:
            raise ImportError("所有ML库都未安装,无法运行Benchmark")

        return models

    def run_walk_forward_comparison(self, n_splits: int = 5) -> pd.DataFrame:
        """
        Walk-Forward交叉验证对比

        Args:
            n_splits: 时间序列划分折数

        Returns:
            对比结果DataFrame
        """
        logger.info(f"开始Walk-Forward对比 (n_splits={n_splits})")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = {name: [] for name in self.models}

        # 时间记录
        training_times = {name: [] for name in self.models}

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(self.X), 1):
            logger.info(f"  Fold {fold_idx}/{n_splits}: 训练{len(train_idx)}样本, 测试{len(test_idx)}样本")

            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

            for name, model in self.models.items():
                # 训练
                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start_time
                training_times[name].append(train_time)

                # 预测
                y_pred = model.predict(X_test)

                # 评估
                ic = self._calculate_ic(y_test, y_pred)
                rank_ic = self._calculate_rank_ic(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                results[name].append({
                    'fold': fold_idx,
                    'ic': ic,
                    'rank_ic': rank_ic,
                    'mse': mse,
                    'mae': mae,
                    'train_time': train_time,
                })

        # 汇总统计
        summary = {}
        for name, scores in results.items():
            ic_values = [s['ic'] for s in scores]
            rank_ic_values = [s['rank_ic'] for s in scores]
            mse_values = [s['mse'] for s in scores]

            summary[name] = {
                'IC均值': np.mean(ic_values),
                'IC标准差': np.std(ic_values),
                'Rank_IC均值': np.mean(rank_ic_values),
                'Rank_IC标准差': np.std(rank_ic_values),
                'MSE均值': np.mean(mse_values),
                'MSE标准差': np.std(mse_values),
                '平均训练时间(秒)': np.mean(training_times[name]),
                '详细结果': scores,
            }

        # 转为DataFrame
        comparison_df = pd.DataFrame(summary).T

        logger.info("Walk-Forward对比完成")
        return comparison_df

    def _calculate_ic(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """计算IC (Information Coefficient)"""
        if len(y_true) < 2:
            return 0.0

        # Pearson相关系数
        ic = np.corrcoef(y_true, y_pred)[0, 1]

        return ic if not np.isnan(ic) else 0.0

    def _calculate_rank_ic(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """计算Rank IC (排序相关系数)"""
        if len(y_true) < 2:
            return 0.0

        # Spearman秩相关
        from scipy.stats import spearmanr
        rank_ic, _ = spearmanr(y_true, y_pred)

        return rank_ic if not np.isnan(rank_ic) else 0.0

    def statistical_significance_test(self, comparison_df: pd.DataFrame) -> Dict:
        """
        统计显著性检验(t-test)

        检验最优算法与其他算法的IC差异是否显著

        Args:
            comparison_df: run_walk_forward_comparison返回的结果

        Returns:
            p值字典
        """
        # 找出IC均值最高的算法
        best_model = comparison_df['IC均值'].idxmax()
        best_ic_values = [s['ic'] for s in comparison_df.loc[best_model, '详细结果']]

        logger.info(f"最优算法: {best_model} (IC均值={comparison_df.loc[best_model, 'IC均值']:.4f})")

        p_values = {}
        for name in comparison_df.index:
            if name == best_model:
                continue

            ic_values = [s['ic'] for s in comparison_df.loc[name, '详细结果']]

            # 配对t检验
            t_stat, p_value = ttest_rel(best_ic_values, ic_values)
            p_values[name] = {
                'p_value': p_value,
                'significant': p_value < 0.05,
                't_statistic': t_stat,
            }

            if p_value < 0.05:
                logger.info(f"  {best_model} vs {name}: p={p_value:.4f} (显著优于)")
            else:
                logger.info(f"  {best_model} vs {name}: p={p_value:.4f} (无显著差异)")

        return p_values

    def generate_report(self, comparison_df: pd.DataFrame,
                       p_values: Optional[Dict] = None) -> str:
        """
        生成对比报告

        Args:
            comparison_df: 对比结果
            p_values: 统计检验p值(可选)

        Returns:
            报告字符串
        """
        report = []
        report.append("=" * 70)
        report.append("ML算法性能对比报告 - Phase 9.1")
        report.append("=" * 70)

        report.append(f"\n📊 对比概况:")
        report.append(f"  样本数: {len(self.data)}")
        report.append(f"  因子数: {len(self.factor_columns)}")
        report.append(f"  目标: {self.target_column}")
        report.append(f"  对比算法: {list(comparison_df.index)}")

        report.append(f"\n\n📈 性能对比:")
        report.append("\n" + "-" * 70)
        report.append(f"{'算法':<15} {'IC均值':<12} {'IC标准差':<12} {'Rank_IC均值':<12} {'训练时间(s)':<12}")
        report.append("-" * 70)

        for name in comparison_df.index:
            row = comparison_df.loc[name]
            report.append(
                f"{name:<15} "
                f"{row['IC均值']:>10.4f}  "
                f"{row['IC标准差']:>10.4f}  "
                f"{row['Rank_IC均值']:>12.4f}  "
                f"{row['平均训练时间(秒)']:>12.3f}"
            )

        report.append("-" * 70)

        # 排名
        report.append(f"\n\n🏆 IC均值排名:")
        ranked = comparison_df.sort_values('IC均值', ascending=False)
        for i, (name, row) in enumerate(ranked.iterrows(), 1):
            medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(i, f'{i}.')
            report.append(f"  {medal} {name}: IC={row['IC均值']:.4f} ± {row['IC标准差']:.4f}")

        # 统计检验结果
        if p_values:
            report.append(f"\n\n📊 统计显著性检验:")
            best_model = ranked.index[0]
            report.append(f"  基准: {best_model}")

            for name, result in p_values.items():
                sig_mark = "✅ 显著" if result['significant'] else "❌ 不显著"
                report.append(
                    f"  vs {name}: p={result['p_value']:.4f}, t={result['t_statistic']:.2f} ({sig_mark})"
                )

        # 建议
        report.append(f"\n\n💡 建议:")
        best_model = ranked.index[0]
        best_ic = ranked.loc[best_model, 'IC均值']

        if best_ic > 0.05:
            report.append(f"  ✅ 推荐使用【{best_model}】作为主力算法")
            report.append(f"     IC均值={best_ic:.4f},具有较强预测能力")
        elif best_ic > 0.03:
            report.append(f"  ⚠️  【{best_model}】可作为辅助算法")
            report.append(f"     IC均值={best_ic:.4f},预测能力中等")
        else:
            report.append(f"  ❌ 所有ML算法表现不佳(最高IC仅{best_ic:.4f})")
            report.append(f"     建议: 优化因子质量或使用纯规则策略")

        report.append("\n" + "=" * 70)

        return "\n".join(report)

    def check_overfitting(self, comparison_df: pd.DataFrame) -> Dict:
        """
        检查过拟合情况

        通过IC标准差判断模型稳定性
        标准差过大说明在不同时期表现差异大,可能过拟合

        Args:
            comparison_df: 对比结果

        Returns:
            过拟合检查结果
        """
        results = {}

        for name in comparison_df.index:
            ic_mean = comparison_df.loc[name, 'IC均值']
            ic_std = comparison_df.loc[name, 'IC标准差']

            # 稳定性得分 = IC均值 / IC标准差
            stability_score = abs(ic_mean) / ic_std if ic_std > 0 else 0

            # 判断
            if ic_std > 0.05:
                status = '⚠️  不稳定'
                comment = f'IC标准差过大({ic_std:.4f}),可能过拟合'
            elif stability_score > 2.0:
                status = '✅ 稳定'
                comment = f'稳定性得分={stability_score:.2f},表现稳定'
            else:
                status = '🔶 中等'
                comment = f'稳定性得分={stability_score:.2f}'

            results[name] = {
                'IC均值': ic_mean,
                'IC标准差': ic_std,
                '稳定性得分': stability_score,
                '状态': status,
                '评价': comment,
            }

        return results


def quick_ml_benchmark(data: pd.DataFrame, factor_columns: List[str],
                      target_column: str = 'return_5d', n_splits: int = 5) -> Dict:
    """
    快速ML算法对比(便捷函数)

    Args:
        data: 数据
        factor_columns: 因子列表
        target_column: 目标列
        n_splits: 交叉验证折数

    Returns:
        完整对比结果
    """
    benchmark = MLAlgorithmBenchmark(data, factor_columns, target_column)

    # 运行对比
    comparison_df = benchmark.run_walk_forward_comparison(n_splits=n_splits)

    # 统计检验
    p_values = benchmark.statistical_significance_test(comparison_df)

    # 过拟合检查
    overfitting_check = benchmark.check_overfitting(comparison_df)

    # 生成报告
    report = benchmark.generate_report(comparison_df, p_values)

    return {
        'comparison': comparison_df,
        'p_values': p_values,
        'overfitting_check': overfitting_check,
        'report': report,
    }
