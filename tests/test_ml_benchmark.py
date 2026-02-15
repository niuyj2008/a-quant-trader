"""
测试ML算法性能对比 - Phase 9.1
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimization.ml_benchmark import MLAlgorithmBenchmark, quick_ml_benchmark
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_mock_factor_data(n_samples=500, n_factors=10):
    """
    创建模拟因子数据(用于测试)

    注意: 这是测试用模拟数据,不用于生产
    """
    np.random.seed(42)

    # 生成日期索引
    dates = pd.date_range(start='2021-01-01', periods=n_samples, freq='D')

    # 生成因子
    factors = {}
    for i in range(n_factors):
        # 每个因子是带趋势的随机游走
        trend = np.linspace(0, 0.1, n_samples)
        noise = np.random.normal(0, 0.05, n_samples)
        factors[f'factor_{i}'] = trend + noise

    # 生成目标(未来5日收益)
    # 与因子有一定相关性
    weights = np.random.uniform(-0.1, 0.1, n_factors)
    factor_matrix = np.array([factors[f'factor_{i}'] for i in range(n_factors)]).T
    base_return = factor_matrix @ weights

    # 添加噪声
    noise = np.random.normal(0, 0.02, n_samples)
    target = base_return + noise

    # 构造DataFrame
    data = pd.DataFrame(factors, index=dates)
    data['return_5d'] = target

    return data


def test_ml_benchmark_initialization():
    """测试1: 初始化ML Benchmark"""
    print("=" * 60)
    print("测试1: 初始化ML Benchmark")
    print("=" * 60)

    # 创建测试数据
    data = create_mock_factor_data(n_samples=500, n_factors=10)
    factor_columns = [col for col in data.columns if col.startswith('factor_')]

    print(f"\n数据规模: {len(data)}样本 × {len(factor_columns)}因子")

    # 初始化Benchmark
    benchmark = MLAlgorithmBenchmark(
        data=data,
        factor_columns=factor_columns,
        target_column='return_5d'
    )

    print(f"\n可用模型: {list(benchmark.models.keys())}")

    # 验证
    assert len(benchmark.models) >= 1, "至少应该加载1个模型"
    assert len(benchmark.X) == len(data), "特征矩阵行数应等于样本数"
    assert len(benchmark.y) == len(data), "目标向量长度应等于样本数"

    print("\n✅ 测试1通过!")
    return True


def test_walk_forward_comparison():
    """测试2: Walk-Forward对比"""
    print("\n" + "=" * 60)
    print("测试2: Walk-Forward对比")
    print("=" * 60)

    # 创建测试数据
    data = create_mock_factor_data(n_samples=500, n_factors=10)
    factor_columns = [col for col in data.columns if col.startswith('factor_')]

    benchmark = MLAlgorithmBenchmark(data, factor_columns, 'return_5d')

    # 运行对比(3折节省时间)
    print("\n运行Walk-Forward对比(3折)...")
    comparison_df = benchmark.run_walk_forward_comparison(n_splits=3)

    print("\n对比结果:")
    print(comparison_df[['IC均值', 'IC标准差', 'Rank_IC均值', '平均训练时间(秒)']])

    # 验证
    assert len(comparison_df) >= 1, "至少应该有1个模型的结果"

    for model_name in comparison_df.index:
        ic_mean = comparison_df.loc[model_name, 'IC均值']
        ic_std = comparison_df.loc[model_name, 'IC标准差']

        print(f"\n{model_name}:")
        print(f"  IC均值: {ic_mean:.4f}")
        print(f"  IC标准差: {ic_std:.4f}")

        # IC应该在合理范围
        assert -1 <= ic_mean <= 1, f"{model_name} IC均值应该在[-1,1]"
        assert ic_std >= 0, f"{model_name} IC标准差应该>=0"

    print("\n✅ 测试2通过!")
    return True


def test_statistical_significance():
    """测试3: 统计显著性检验"""
    print("\n" + "=" * 60)
    print("测试3: 统计显著性检验")
    print("=" * 60)

    data = create_mock_factor_data(n_samples=500, n_factors=10)
    factor_columns = [col for col in data.columns if col.startswith('factor_')]

    benchmark = MLAlgorithmBenchmark(data, factor_columns, 'return_5d')
    comparison_df = benchmark.run_walk_forward_comparison(n_splits=3)

    # 统计检验
    print("\n执行统计显著性检验...")
    p_values = benchmark.statistical_significance_test(comparison_df)

    if p_values:
        print("\n检验结果:")
        for name, result in p_values.items():
            print(f"  vs {name}:")
            print(f"    p值: {result['p_value']:.4f}")
            print(f"    t统计量: {result['t_statistic']:.2f}")
            print(f"    显著: {result['significant']}")

        # 验证
        for name, result in p_values.items():
            assert 0 <= result['p_value'] <= 1, "p值应该在[0,1]"
    else:
        print("  只有1个模型,无需检验")

    print("\n✅ 测试3通过!")
    return True


def test_overfitting_check():
    """测试4: 过拟合检查"""
    print("\n" + "=" * 60)
    print("测试4: 过拟合检查")
    print("=" * 60)

    data = create_mock_factor_data(n_samples=500, n_factors=10)
    factor_columns = [col for col in data.columns if col.startswith('factor_')]

    benchmark = MLAlgorithmBenchmark(data, factor_columns, 'return_5d')
    comparison_df = benchmark.run_walk_forward_comparison(n_splits=3)

    # 过拟合检查
    print("\n检查过拟合情况...")
    overfitting_check = benchmark.check_overfitting(comparison_df)

    print("\n过拟合检查结果:")
    for name, result in overfitting_check.items():
        print(f"\n  {name}:")
        print(f"    IC均值: {result['IC均值']:.4f}")
        print(f"    IC标准差: {result['IC标准差']:.4f}")
        print(f"    稳定性得分: {result['稳定性得分']:.2f}")
        print(f"    状态: {result['状态']}")
        print(f"    评价: {result['评价']}")

        # 验证
        assert result['稳定性得分'] >= 0, "稳定性得分应该>=0"

    print("\n✅ 测试4通过!")
    return True


def test_generate_report():
    """测试5: 生成对比报告"""
    print("\n" + "=" * 60)
    print("测试5: 生成对比报告")
    print("=" * 60)

    data = create_mock_factor_data(n_samples=500, n_factors=10)
    factor_columns = [col for col in data.columns if col.startswith('factor_')]

    benchmark = MLAlgorithmBenchmark(data, factor_columns, 'return_5d')
    comparison_df = benchmark.run_walk_forward_comparison(n_splits=3)
    p_values = benchmark.statistical_significance_test(comparison_df)

    # 生成报告
    print("\n生成对比报告...")
    report = benchmark.generate_report(comparison_df, p_values)

    print("\n" + report)

    # 验证
    assert len(report) > 0, "报告应该非空"
    assert "ML算法性能对比报告" in report, "报告应该包含标题"
    assert "IC均值" in report, "报告应该包含IC均值"

    print("\n✅ 测试5通过!")
    return True


def test_quick_ml_benchmark():
    """测试6: 快速对比函数"""
    print("\n" + "=" * 60)
    print("测试6: 快速对比函数")
    print("=" * 60)

    data = create_mock_factor_data(n_samples=500, n_factors=10)
    factor_columns = [col for col in data.columns if col.startswith('factor_')]

    # 使用快速函数
    print("\n使用quick_ml_benchmark...")
    result = quick_ml_benchmark(
        data=data,
        factor_columns=factor_columns,
        target_column='return_5d',
        n_splits=3
    )

    print("\n快速对比完成!")
    print(f"  返回键: {list(result.keys())}")

    # 验证
    assert 'comparison' in result, "应该返回comparison"
    assert 'p_values' in result, "应该返回p_values"
    assert 'overfitting_check' in result, "应该返回overfitting_check"
    assert 'report' in result, "应该返回report"

    print("\n" + result['report'])

    print("\n✅ 测试6通过!")
    return True


if __name__ == "__main__":
    try:
        # 运行所有测试
        test_ml_benchmark_initialization()
        test_walk_forward_comparison()
        test_statistical_significance()
        test_overfitting_check()
        test_generate_report()
        test_quick_ml_benchmark()

        print("\n" + "=" * 60)
        print("🎉 所有Phase 9.1 ML Benchmark测试通过!")
        print("ML算法对比功能已准备就绪!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
