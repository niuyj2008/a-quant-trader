"""
测试数据验证器 - Phase 8
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.validation.data_validator import DataValidator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_valid_price_data():
    """测试1: 验证正常的价格数据"""
    print("=" * 60)
    print("测试1: 验证正常的价格数据")
    print("=" * 60)

    # 创建正常的价格数据
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)

    prices = 100 * (1 + np.random.normal(0.001, 0.015, len(dates))).cumprod()

    df = pd.DataFrame({
        'open': prices * 0.995,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates)),
    }, index=dates)

    # 添加数据来源标记
    df.attrs['source'] = 'akshare'

    validator = DataValidator()
    result = validator.validate_price_data(df, code='600519', market='CN')

    print(f"\n验证结果: {'✅ 通过' if result['is_valid'] else '❌ 失败'}")
    print(f"数据来源: {result['data_source']}")
    print(f"数据范围: {result['date_range']}")
    print(f"记录数: {result['records']}")

    if result['issues']:
        print(f"问题列表:")
        for issue in result['issues']:
            print(f"  - {issue}")

    assert result['is_valid'], "正常数据应该验证通过"

    print("\n✅ 测试1通过!")
    return True


def test_invalid_price_data():
    """测试2: 验证异常的价格数据"""
    print("\n" + "=" * 60)
    print("测试2: 验证异常的价格数据")
    print("=" * 60)

    # 创建有问题的数据
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')

    df = pd.DataFrame({
        'open': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
        'high': [105, 115, 125, 135, 145, 155, 165, 175, 185, 195],
        'low': [95, 105, 115, 125, 135, 145, 155, 165, 175, 185],
        'close': [102, 112, 122, 132, 142, 152, 162, 172, 182, 192],
        'volume': [1000000] * 10,
    }, index=dates)

    # 制造异常1: High < Low
    df.loc[dates[2], 'high'] = 110
    df.loc[dates[2], 'low'] = 125

    # 制造异常2: Close不在[Low, High]范围
    df.loc[dates[5], 'close'] = 200

    # 制造异常3: 成交量为0
    df.loc[dates[7], 'volume'] = 0

    validator = DataValidator()
    result = validator.validate_price_data(df, code='TEST001', market='CN')

    print(f"\n验证结果: {'✅ 通过' if result['is_valid'] else '❌ 失败(预期)'}")
    print(f"问题数量: {len(result['issues'])}")

    if result['issues']:
        print(f"\n检测到的问题:")
        for issue in result['issues']:
            print(f"  - {issue}")

    assert not result['is_valid'], "异常数据应该验证失败"
    assert len(result['issues']) >= 3, "应该检测出至少3个问题"

    print("\n✅ 测试2通过!")
    return True


def test_extreme_returns():
    """测试3: 验证极端涨跌幅检测"""
    print("\n" + "=" * 60)
    print("测试3: 验证极端涨跌幅检测")
    print("=" * 60)

    validator = DataValidator()

    # 测试用例1: A股极端涨幅(超过12%)
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    prices_cn = [100, 105, 110, 125, 128, 130, 132, 135, 137, 140]  # 第4天涨13.6%

    df_cn = pd.DataFrame({
        'open': [p * 0.99 for p in prices_cn],
        'high': [p * 1.01 for p in prices_cn],
        'low': [p * 0.98 for p in prices_cn],
        'close': prices_cn,
        'volume': [1000000] * 10,
    }, index=dates)

    result_cn = validator.validate_price_data(df_cn, code='TEST_CN', market='CN')
    print(f"\nA股市场验证(涨幅13.6%): {'✅ 通过' if result_cn['is_valid'] else '❌ 失败(预期)'}")
    if result_cn['issues']:
        print(f"  问题: {result_cn['issues']}")

    # 测试用例2: 美股合理大涨(30%)
    prices_us = [100, 105, 110, 143, 145, 148, 150, 152, 155, 158]  # 第4天涨30%

    df_us = pd.DataFrame({
        'open': [p * 0.99 for p in prices_us],
        'high': [p * 1.01 for p in prices_us],
        'low': [p * 0.98 for p in prices_us],
        'close': prices_us,
        'volume': [1000000] * 10,
    }, index=dates)
    df_us.attrs['source'] = 'yfinance'

    result_us = validator.validate_price_data(df_us, code='TSLA', market='US')
    print(f"\n美股市场验证(涨幅30%): {'✅ 通过' if result_us['is_valid'] else '❌ 失败'}")
    if result_us['issues']:
        print(f"  问题: {result_us['issues']}")

    assert not result_cn['is_valid'], "A股市场应该检测出异常涨幅(>12%)"
    assert result_us['is_valid'], "美股市场应该允许30%涨幅"

    print("\n✅ 测试3通过!")
    return True


def test_backtest_result_validation():
    """测试4: 验证回测结果"""
    print("\n" + "=" * 60)
    print("测试4: 验证回测结果")
    print("=" * 60)

    validator = DataValidator()

    # 测试正常回测结果
    normal_result = {
        'total_return': 0.25,
        'annual_return': 0.12,
        'sharpe_ratio': 1.5,
        'max_drawdown': -0.15,
        'n_trades': 50,
        'win_rate': 0.65,
        'data_source': 'akshare',
    }

    normal_trades = [
        {'date': '2024-01-01', 'price': 100, 'shares': 100, 'action': 'buy'},
        {'date': '2024-01-15', 'price': 110, 'shares': 100, 'action': 'sell'},
    ]

    result1 = validator.validate_backtest_result(
        normal_result, normal_trades,
        code='600519', start_date='2024-01-01', end_date='2024-12-31'
    )

    print(f"\n正常回测结果验证: {'✅ 通过' if result1['is_valid'] else '❌ 失败'}")
    if result1['issues']:
        print(f"  问题: {result1['issues']}")

    # 测试异常回测结果
    abnormal_result = {
        'total_return': 0.50,
        'annual_return': 10.0,  # 1000%年化收益
        'sharpe_ratio': 8.0,    # 异常高的夏普
        'max_drawdown': 0.05,   # 正数回撤(错误)
        'n_trades': 0,          # 无交易但有收益
        'win_rate': 0.98,       # 98%胜率
    }

    result2 = validator.validate_backtest_result(
        abnormal_result, [],
        code='FAKE001', start_date='2024-01-01', end_date='2024-12-31'
    )

    print(f"\n异常回测结果验证: {'✅ 通过' if result2['is_valid'] else '❌ 失败(预期)'}")
    print(f"检测到的问题数量: {len(result2['issues'])}")
    if result2['issues']:
        print(f"\n问题列表:")
        for issue in result2['issues']:
            print(f"  - {issue}")

    assert result1['is_valid'], "正常回测结果应该通过"
    assert not result2['is_valid'], "异常回测结果应该失败"
    assert len(result2['issues']) >= 3, "应该检测出多个问题"

    print("\n✅ 测试4通过!")
    return True


def test_factor_data_validation():
    """测试5: 验证因子数据"""
    print("\n" + "=" * 60)
    print("测试5: 验证因子数据")
    print("=" * 60)

    validator = DataValidator()

    # 创建正常因子数据
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)

    factors = pd.DataFrame({
        'return_5d': np.random.normal(0.01, 0.05, len(dates)),
        'return_20d': np.random.normal(0.03, 0.10, len(dates)),
        'momentum': np.random.normal(0.5, 0.2, len(dates)),
        'volatility': np.random.uniform(0.1, 0.4, len(dates)),
        'rsi_14': np.random.uniform(20, 80, len(dates)),
    }, index=dates)

    result1 = validator.validate_factor_data(factors, code='600519')

    print(f"\n正常因子数据验证: {'✅ 通过' if result1['is_valid'] else '❌ 失败'}")
    print(f"  因子数: {result1['n_factors']}")
    print(f"  记录数: {result1['n_records']}")

    # 创建异常因子数据
    abnormal_factors = pd.DataFrame({
        'return_5d': [0.5] * 100,  # 常数因子
        'return_20d': [np.inf] * 100,  # 无穷大
        'momentum': [np.nan] * 100,  # 全NaN
    })

    result2 = validator.validate_factor_data(abnormal_factors, code='FAKE001')

    print(f"\n异常因子数据验证: {'✅ 通过' if result2['is_valid'] else '❌ 失败(预期)'}")
    if result2['issues']:
        print(f"\n问题列表:")
        for issue in result2['issues']:
            print(f"  - {issue}")

    assert result1['is_valid'], "正常因子数据应该通过"
    assert not result2['is_valid'], "异常因子数据应该失败"

    print("\n✅ 测试5通过!")
    return True


def test_validation_summary():
    """测试6: 验证历史摘要"""
    print("\n" + "=" * 60)
    print("测试6: 验证历史摘要")
    print("=" * 60)

    validator = DataValidator()

    # 运行几次验证
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    np.random.seed(42)

    for i in range(5):
        prices = 100 * (1 + np.random.normal(0.001, 0.015, len(dates))).cumprod()
        df = pd.DataFrame({
            'open': prices * 0.995,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates)),
        }, index=dates)
        df.attrs['source'] = 'test'

        validator.validate_price_data(df, code=f'TEST{i:03d}', market='CN')

    summary = validator.get_validation_summary()

    print(f"\n验证历史摘要:")
    print(f"  总验证次数: {summary['total_validations']}")
    print(f"  通过次数: {summary['passed']}")
    print(f"  失败次数: {summary['failed']}")
    print(f"  通过率: {summary['pass_rate']:.1%}")

    assert summary['total_validations'] >= 5, "应该有至少5次验证记录"

    print("\n✅ 测试6通过!")
    return True


if __name__ == "__main__":
    try:
        # 运行所有测试
        test_valid_price_data()
        test_invalid_price_data()
        test_extreme_returns()
        test_backtest_result_validation()
        test_factor_data_validation()
        test_validation_summary()

        print("\n" + "=" * 60)
        print("🎉 所有Phase 8 DataValidator测试通过!")
        print("数据验证器已准备就绪,可以确保系统只使用真实数据!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
