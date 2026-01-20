"""
A股量化交易系统 - Qlib深度集成

提供与Microsoft Qlib的集成，支持高级AI量化策略
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

# Qlib导入（可选依赖）
try:
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    logger.warning("Qlib未安装，高级AI功能不可用")
    logger.info("安装命令: pip install pyqlib")


@dataclass
class QlibConfig:
    """Qlib配置"""
    provider_uri: str = "~/.qlib/qlib_data/cn_data"
    region: str = "cn"
    
    # 数据配置
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    train_end: str = "2023-12-31"
    
    # 模型配置
    model_type: str = "lightgbm"  # lightgbm, catboost, mlp, lstm
    
    # 因子配置
    feature_set: str = "Alpha158"  # Alpha158, Alpha360


class QlibManager:
    """Qlib管理器"""
    
    def __init__(self, config: Optional[QlibConfig] = None):
        self.config = config or QlibConfig()
        self.initialized = False
        self.dataset = None
        self.model = None
        
    def initialize(self) -> bool:
        """初始化Qlib"""
        if not QLIB_AVAILABLE:
            logger.error("Qlib未安装")
            return False
        
        try:
            qlib.init(
                provider_uri=self.config.provider_uri,
                region=self.config.region
            )
            self.initialized = True
            logger.info("Qlib初始化成功")
            return True
        except Exception as e:
            logger.error(f"Qlib初始化失败: {e}")
            return False
    
    def download_data(self, target_dir: str = "~/.qlib/qlib_data/cn_data"):
        """下载A股数据"""
        if not QLIB_AVAILABLE:
            return
        
        from qlib.contrib.data.handler import Alpha158
        
        logger.info("开始下载Qlib中国A股数据...")
        logger.info("这可能需要几分钟到几小时，取决于网络速度")
        
        # 使用qlib内置的数据下载工具
        import subprocess
        cmd = f"python -m qlib.contrib.data.handler --target_dir {target_dir} --region cn"
        subprocess.run(cmd, shell=True)
    
    def get_alpha158_handler(self) -> Dict:
        """获取Alpha158因子处理器配置"""
        return {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_date": self.config.start_date,
                "end_date": self.config.end_date,
                "fit_start_date": self.config.start_date,
                "fit_end_date": self.config.train_end,
                "instruments": "csi300",
                "infer_processors": [
                    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}}
                ],
                "learn_processors": [
                    {"class": "DropnaLabel"},
                    {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}
                ],
                "label": ["Ref($close, -2) / Ref($close, -1) - 1"]
            }
        }
    
    def create_dataset(
        self,
        instruments: str = "csi300",
        handler_config: Optional[Dict] = None
    ):
        """创建数据集"""
        if not self.initialized:
            logger.error("请先初始化Qlib")
            return None
        
        if handler_config is None:
            handler_config = self.get_alpha158_handler()
        
        data_handler_config = {
            "start_date": self.config.start_date,
            "end_date": self.config.end_date,
            "fit_start_date": self.config.start_date,
            "fit_end_date": self.config.train_end,
            "instruments": instruments,
        }
        data_handler_config.update(handler_config.get("kwargs", {}))
        
        # 创建数据集
        dataset = DatasetH(
            handler=handler_config,
            segments={
                "train": (self.config.start_date, self.config.train_end),
                "valid": (self.config.train_end, self.config.end_date),
                "test": (self.config.train_end, self.config.end_date),
            }
        )
        
        self.dataset = dataset
        logger.info(f"数据集创建成功: {instruments}")
        return dataset
    
    def get_lightgbm_model_config(self) -> Dict:
        """获取LightGBM模型配置"""
        return {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 20,
            }
        }
    
    def get_mlp_model_config(self) -> Dict:
        """获取MLP模型配置"""
        return {
            "class": "DNNModelPytorch",
            "module_path": "qlib.contrib.model.pytorch_nn",
            "kwargs": {
                "d_feat": 158,
                "hidden_size": 512,
                "num_layers": 3,
                "dropout": 0.1,
                "n_epochs": 200,
                "lr": 1e-3,
                "batch_size": 2000,
                "GPU": 0,
            }
        }
    
    def get_lstm_model_config(self) -> Dict:
        """获取LSTM模型配置"""
        return {
            "class": "LSTMModel",
            "module_path": "qlib.contrib.model.pytorch_lstm",
            "kwargs": {
                "d_feat": 158,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.0,
                "n_epochs": 200,
                "lr": 1e-3,
                "batch_size": 2000,
            }
        }
    
    def train_model(self, model_config: Optional[Dict] = None):
        """训练模型"""
        if self.dataset is None:
            logger.error("请先创建数据集")
            return None
        
        if model_config is None:
            if self.config.model_type == "lightgbm":
                model_config = self.get_lightgbm_model_config()
            elif self.config.model_type == "mlp":
                model_config = self.get_mlp_model_config()
            elif self.config.model_type == "lstm":
                model_config = self.get_lstm_model_config()
            else:
                model_config = self.get_lightgbm_model_config()
        
        # 动态导入模型类
        module_path = model_config["module_path"]
        class_name = model_config["class"]
        
        import importlib
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        
        # 创建并训练模型
        self.model = model_class(**model_config["kwargs"])
        self.model.fit(self.dataset)
        
        logger.info(f"模型训练完成: {class_name}")
        return self.model
    
    def predict(self, segment: str = "test") -> pd.DataFrame:
        """预测"""
        if self.model is None:
            logger.error("请先训练模型")
            return pd.DataFrame()
        
        pred = self.model.predict(self.dataset, segment=segment)
        
        if isinstance(pred, pd.Series):
            pred = pred.to_frame("score")
        
        return pred
    
    def backtest(
        self,
        predictions: pd.DataFrame,
        topk: int = 50,
        n_drop: int = 5,
        benchmark: str = "SH000300"
    ) -> Dict:
        """
        回测预测结果
        
        Args:
            predictions: 预测得分
            topk: 持有股票数量
            n_drop: 每次调仓卖出数量
            benchmark: 基准指数
        """
        try:
            from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
            from qlib.contrib.evaluate import backtest as qlib_backtest
            from qlib.contrib.evaluate import risk_analysis
            
            strategy_config = {
                "topk": topk,
                "n_drop": n_drop,
            }
            
            strategy = TopkDropoutStrategy(**strategy_config)
            
            # 回测
            report_normal, positions_normal = qlib_backtest(
                pred=predictions,
                strategy=strategy,
                benchmark=benchmark,
                account=100000000,
                exchange_kwargs={
                    "limit_threshold": 0.095,
                    "deal_price": "close",
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5,
                },
            )
            
            # 风险分析
            analysis = risk_analysis(report_normal)
            
            return {
                "report": report_normal,
                "positions": positions_normal,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"回测失败: {e}")
            return {}
    
    def get_stock_pool(self, market: str = "csi300") -> List[str]:
        """获取股票池"""
        if not self.initialized:
            return []
        
        instruments = D.instruments(market=market)
        return D.list_instruments(instruments=instruments, as_list=True)


class QlibAlphaStrategy:
    """Qlib Alpha策略"""
    
    def __init__(
        self,
        qlib_manager: QlibManager,
        topk: int = 20,
        rebalance_freq: int = 5
    ):
        self.qlib_manager = qlib_manager
        self.topk = topk
        self.rebalance_freq = rebalance_freq
        self.predictions: Optional[pd.DataFrame] = None
        self.current_positions: Dict[str, float] = {}
        self.last_rebalance_date: Optional[datetime] = None
    
    def update_predictions(self):
        """更新预测"""
        if self.qlib_manager.model is None:
            logger.warning("模型未训练")
            return
        
        self.predictions = self.qlib_manager.predict()
        logger.info(f"预测更新完成: {len(self.predictions)} 条")
    
    def get_target_positions(self, date: str) -> Dict[str, float]:
        """
        获取目标持仓
        
        Args:
            date: 日期字符串 YYYY-MM-DD
            
        Returns:
            {股票代码: 权重}
        """
        if self.predictions is None or self.predictions.empty:
            return {}
        
        # 获取当天的预测得分
        try:
            date_predictions = self.predictions.loc[date]
            
            if isinstance(date_predictions, pd.Series):
                date_predictions = date_predictions.to_frame("score")
            
            # 选择Top K
            top_stocks = date_predictions.nlargest(self.topk, "score")
            
            # 等权重
            weight = 1.0 / len(top_stocks)
            
            return {stock: weight for stock in top_stocks.index}
            
        except KeyError:
            logger.warning(f"无 {date} 的预测数据")
            return {}
    
    def generate_signals(
        self,
        date: str,
        current_positions: Dict[str, float]
    ) -> List[Dict]:
        """
        生成交易信号
        
        Args:
            date: 当前日期
            current_positions: 当前持仓 {code: shares}
            
        Returns:
            交易信号列表
        """
        target = self.get_target_positions(date)
        signals = []
        
        # 需要卖出的持仓
        for code in current_positions:
            if code not in target:
                signals.append({
                    'code': code,
                    'action': 'sell',
                    'reason': '不在目标持仓'
                })
        
        # 需要买入的股票
        for code in target:
            if code not in current_positions:
                signals.append({
                    'code': code,
                    'action': 'buy',
                    'weight': target[code],
                    'reason': f'Qlib预测Top{self.topk}'
                })
        
        return signals


# 简化的因子表达式工具
class FactorExpression:
    """
    因子表达式计算（简化版Qlib表达式）
    
    支持的操作:
    - $close, $open, $high, $low, $volume: 价格数据
    - Ref(x, n): n日前的值
    - Mean(x, n): n日均值
    - Std(x, n): n日标准差
    - Max(x, n): n日最大值
    - Min(x, n): n日最小值
    - Rank(x): 截面排名
    - Corr(x, y, n): n日相关系数
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: OHLCV数据，columns包含 open, high, low, close, volume
        """
        self.df = df
        self.cache = {}
    
    def ref(self, series: pd.Series, n: int) -> pd.Series:
        """N日前的值"""
        return series.shift(n)
    
    def mean(self, series: pd.Series, n: int) -> pd.Series:
        """N日均值"""
        return series.rolling(n).mean()
    
    def std(self, series: pd.Series, n: int) -> pd.Series:
        """N日标准差"""
        return series.rolling(n).std()
    
    def max(self, series: pd.Series, n: int) -> pd.Series:
        """N日最大值"""
        return series.rolling(n).max()
    
    def min(self, series: pd.Series, n: int) -> pd.Series:
        """N日最小值"""
        return series.rolling(n).min()
    
    def delta(self, series: pd.Series, n: int) -> pd.Series:
        """N日变化"""
        return series.diff(n)
    
    def rank(self, series: pd.Series) -> pd.Series:
        """排名"""
        return series.rank()
    
    def corr(self, x: pd.Series, y: pd.Series, n: int) -> pd.Series:
        """相关系数"""
        return x.rolling(n).corr(y)
    
    def calculate_alpha(self, expression: str) -> pd.Series:
        """
        计算Alpha因子表达式
        
        Example expressions:
        - "Ref($close, 5) / $close - 1"  # 5日动量
        - "($close - Mean($close, 20)) / Std($close, 20)"  # Z-Score
        - "Rank(Ref($close, 1) / $close)"  # 反转因子
        """
        # 替换变量
        expr = expression.replace("$close", "self.df['close']")
        expr = expr.replace("$open", "self.df['open']")
        expr = expr.replace("$high", "self.df['high']")
        expr = expr.replace("$low", "self.df['low']")
        expr = expr.replace("$volume", "self.df['volume']")
        
        # 替换函数
        expr = expr.replace("Ref(", "self.ref(")
        expr = expr.replace("Mean(", "self.mean(")
        expr = expr.replace("Std(", "self.std(")
        expr = expr.replace("Max(", "self.max(")
        expr = expr.replace("Min(", "self.min(")
        expr = expr.replace("Delta(", "self.delta(")
        expr = expr.replace("Rank(", "self.rank(")
        expr = expr.replace("Corr(", "self.corr(")
        
        try:
            result = eval(expr)
            return result
        except Exception as e:
            logger.error(f"表达式计算失败: {expression}, 错误: {e}")
            return pd.Series()
    
    def get_alpha_factors(self) -> pd.DataFrame:
        """计算常用Alpha因子"""
        factors = pd.DataFrame(index=self.df.index)
        
        # 动量因子
        factors['mom_5'] = self.ref(self.df['close'], 5) / self.df['close'] - 1
        factors['mom_10'] = self.ref(self.df['close'], 10) / self.df['close'] - 1
        factors['mom_20'] = self.ref(self.df['close'], 20) / self.df['close'] - 1
        
        # 波动率
        factors['volatility'] = self.std(self.df['close'].pct_change(), 20)
        
        # 均线偏离
        ma20 = self.mean(self.df['close'], 20)
        factors['ma_bias'] = (self.df['close'] - ma20) / ma20
        
        # 成交量变化
        factors['volume_ratio'] = self.df['volume'] / self.mean(self.df['volume'], 5)
        
        # 价格振幅
        factors['amplitude'] = (self.df['high'] - self.df['low']) / self.ref(self.df['close'], 1)
        
        # RSI
        delta = self.df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        rs = self.mean(gain, 14) / self.mean(loss, 14)
        factors['rsi'] = 100 - (100 / (1 + rs))
        
        return factors


if __name__ == "__main__":
    # 测试因子表达式（不需要Qlib）
    from src.data import DataFetcher
    
    print("测试因子表达式计算...")
    
    fetcher = DataFetcher()
    df = fetcher.get_daily_data("000001", start_date="2024-01-01")
    
    factor_expr = FactorExpression(df)
    
    # 计算Alpha因子
    factors = factor_expr.get_alpha_factors()
    print("\nAlpha因子:")
    print(factors.tail())
    
    # 自定义表达式
    custom = factor_expr.calculate_alpha("($close - Mean($close, 20)) / Std($close, 20)")
    print("\n自定义因子 (Z-Score):")
    print(custom.tail())
