#!/bin/bash

# 全面系统测试脚本
# 运行所有测试模块并生成报告

echo "========================================"
echo "量化交易系统 - 全面测试开始"
echo "========================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 测试计数
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# 测试函数
run_test() {
    local test_file=$1
    local test_name=$(basename $test_file .py)

    echo "----------------------------------------"
    echo "运行: $test_name"
    echo "----------------------------------------"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    if python3 $test_file > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $test_name 通过${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}❌ $test_name 失败${NC}"
        echo "详细错误信息:"
        python3 $test_file 2>&1 | tail -20
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# 切换到项目目录
cd /Users/niuyj/Downloads/workspace_Claude/stock/a-quant-trader

echo "📋 Phase 8: 数据验证"
run_test "tests/test_data_validator.py"

echo ""
echo "📋 Phase 9.1: ML算法对比"
run_test "tests/test_ml_benchmark.py"

echo ""
echo "📋 Phase 9.3: 策略集成"
run_test "tests/test_ensemble_strategy.py"

echo ""
echo "📋 Phase 9.4: 学术因子"
run_test "tests/test_academic_factors.py"

echo ""
echo "📋 Phase 9.5: 专业回测报告"
run_test "tests/test_professional_report.py"

echo ""
echo "📋 Phase 6: ETF定投"
run_test "tests/test_etf_dca.py"

echo ""
echo "📋 Phase 2: 基本面分析"
run_test "tests/test_fundamental_analyzer.py"

echo ""
echo "📋 Phase 3: 持仓管理"
run_test "tests/test_trade_journal_v2.py"

echo ""
echo "📋 Phase 4: 推荐系统"
run_test "tests/test_recommendation_backtest.py"

echo ""
echo "📋 Phase 5: 策略验证"
run_test "tests/test_strategy_validator.py"

echo ""
echo "📋 Phase 7: 目标导向推荐"
run_test "tests/test_goal_recommender.py"

echo ""
echo "📋 美股持仓管理"
run_test "tests/test_portfolio_manager_us.py"

echo ""
echo "========================================"
echo "测试总结"
echo "========================================"
echo -e "总测试数: $TOTAL_TESTS"
echo -e "${GREEN}通过: $PASSED_TESTS${NC}"
echo -e "${RED}失败: $FAILED_TESTS${NC}"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}🎉 所有测试通过!系统运行正常!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  有 $FAILED_TESTS 个测试失败,请检查详细日志${NC}"
    exit 1
fi
