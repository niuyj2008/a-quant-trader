"""
A股量化交易系统 - 机器学习模型

提供股票预测的机器学习模型
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger
import pickle
from pathlib import Path

try:
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn未安装，部分ML功能不可用")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except Exception as e:
    LIGHTGBM_AVAILABLE = False
    logger.warning(f"LightGBM不可用 (可能是缺少libomp): {e}")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    logger.warning(f"XGBoost不可用 (可能是缺少libomp): {e}")


@dataclass
class ModelMetrics:
    """模型评估指标"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    ic: float = 0.0  # Information Coefficient
    icir: float = 0.0  # IC Information Ratio
    
    def to_dict(self) -> Dict:
        return {
            "准确率": f"{self.accuracy:.4f}",
            "精确率": f"{self.precision:.4f}",
            "召回率": f"{self.recall:.4f}",
            "F1分数": f"{self.f1:.4f}",
            "IC": f"{self.ic:.4f}",
            "ICIR": f"{self.icir:.4f}",
        }


class BaseMLModel:
    """机器学习模型基类"""
    
    def __init__(self, name: str = "BaseModel"):
        self.name = name
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_names: List[str] = []
        self.is_fitted = False
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target",
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备特征和标签
        
        Args:
            df: 包含特征的DataFrame
            feature_cols: 特征列名
            target_col: 目标列名
            normalize: 是否标准化
        """
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        y = df[target_col].values if target_col in df.columns else None
        
        # 处理缺失值
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if normalize and self.scaler:
            if not self.is_fitted:
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
        
        return X, y
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """训练模型"""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        raise NotImplementedError
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        raise NotImplementedError
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """评估模型"""
        if not SKLEARN_AVAILABLE:
            return ModelMetrics()
        
        y_pred = self.predict(X)
        
        metrics = ModelMetrics(
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred, average='weighted', zero_division=0),
            recall=recall_score(y, y_pred, average='weighted', zero_division=0),
            f1=f1_score(y, y_pred, average='weighted', zero_division=0),
        )
        
        return metrics
    
    def save(self, path: str):
        """保存模型"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'name': self.name
            }, f)
        
        logger.info(f"模型已保存: {path}")
    
    def load(self, path: str):
        """加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.name = data['name']
        self.is_fitted = True
        
        logger.info(f"模型已加载: {path}")


class LightGBMModel(BaseMLModel):
    """LightGBM模型"""
    
    def __init__(self, task: str = "classification", **params):
        """
        Args:
            task: "classification" 或 "regression"
            params: LightGBM参数
        """
        super().__init__(name="LightGBM")
        self.task = task
        self.params = {
            'objective': 'binary' if task == 'classification' else 'regression',
            'metric': 'auc' if task == 'classification' else 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'max_depth': 6,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1,
            **params
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM未安装")
        
        # 划分验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        if self.task == 'classification':
            self.model = lgb.LGBMClassifier(**self.params)
        else:
            self.model = lgb.LGBMRegressor(**self.params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        self.is_fitted = True
        logger.info(f"LightGBM训练完成，最佳迭代次数: {self.model.best_iteration_}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task == 'classification':
            return self.model.predict_proba(X)
        return self.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性"""
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


class XGBoostModel(BaseMLModel):
    """XGBoost模型"""
    
    def __init__(self, task: str = "classification", **params):
        super().__init__(name="XGBoost")
        self.task = task
        self.params = {
            'objective': 'binary:logistic' if task == 'classification' else 'reg:squarederror',
            'eval_metric': 'auc' if task == 'classification' else 'rmse',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbosity': 0,
            **params
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost未安装")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        if self.task == 'classification':
            self.model = xgb.XGBClassifier(**self.params)
        else:
            self.model = xgb.XGBRegressor(**self.params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        self.is_fitted = True
        logger.info("XGBoost训练完成")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task == 'classification':
            return self.model.predict_proba(X)
        return self.predict(X)


class RandomForestModel(BaseMLModel):
    """随机森林模型"""
    
    def __init__(self, task: str = "classification", **params):
        super().__init__(name="RandomForest")
        self.task = task
        self.params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'n_jobs': -1,
            **params
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn未安装")
        
        if self.task == 'classification':
            self.model = RandomForestClassifier(**self.params)
        else:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(**self.params)
        
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("RandomForest训练完成")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task == 'classification':
            return self.model.predict_proba(X)
        return self.predict(X)


class StockPredictor:
    """股票预测器 - 集成多个模型"""
    
    def __init__(self, model_type: str = "lightgbm"):
        """
        Args:
            model_type: 模型类型 ("lightgbm", "xgboost", "random_forest")
        """
        self.model_type = model_type
        self.model = self._create_model(model_type)
        self.feature_cols: List[str] = []
    
    def _create_model(self, model_type: str) -> BaseMLModel:
        if model_type == "lightgbm":
            return LightGBMModel()
        elif model_type == "xgboost":
            return XGBoostModel()
        elif model_type in ["random_forest", "randomforest"]:
            return RandomForestModel()
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def create_target(
        self,
        df: pd.DataFrame,
        lookahead: int = 5,
        threshold: float = 0.02
    ) -> pd.DataFrame:
        """
        创建预测目标 - N日后收益率是否超过阈值
        
        Args:
            df: OHLCV数据
            lookahead: 预测天数
            threshold: 收益率阈值
            
        Returns:
            添加target列的DataFrame
        """
        df = df.copy()
        
        # 计算N日后收益率
        future_return = df['close'].shift(-lookahead) / df['close'] - 1
        
        # 二分类目标: 收益率 > threshold
        df['target'] = (future_return > threshold).astype(int)
        
        # 移除未来数据
        df = df.iloc[:-lookahead]
        
        return df
    
    def train(
        self,
        data: Dict[str, pd.DataFrame],
        feature_cols: List[str],
        lookahead: int = 5,
        threshold: float = 0.02
    ) -> ModelMetrics:
        """
        使用多只股票数据训练模型
        
        Args:
            data: 股票代码 -> OHLCV DataFrame (含因子)
            feature_cols: 特征列名
            lookahead: 预测天数
            threshold: 收益率阈值
        """
        self.feature_cols = feature_cols
        
        # 合并所有股票数据
        all_data = []
        for code, df in data.items():
            df = self.create_target(df, lookahead, threshold)
            df['code'] = code
            all_data.append(df)
        
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.dropna(subset=feature_cols + ['target'])
        
        logger.info(f"训练数据量: {len(combined)} 条")
        
        # 准备特征
        X, y = self.model.prepare_features(combined, feature_cols, 'target')
        
        # 时间序列划分
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # 训练
        self.model.fit(X_train, y_train)
        
        # 评估
        metrics = self.model.evaluate(X_test, y_test)
        
        logger.info(f"测试集评估: {metrics.to_dict()}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预测股票
        
        Args:
            df: 包含特征的DataFrame
            
        Returns:
            添加预测结果的DataFrame
        """
        df = df.copy()
        
        X, _ = self.model.prepare_features(df, self.feature_cols, normalize=True)
        
        df['pred_proba'] = self.model.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else self.model.predict(X)
        df['pred_label'] = self.model.predict(X)
        
        return df
    
    def select_stocks(
        self,
        data: Dict[str, pd.DataFrame],
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        选股 - 选择预测概率最高的股票
        
        Args:
            data: 股票代码 -> DataFrame (最新数据)
            top_n: 选择数量
            
        Returns:
            [(股票代码, 预测概率), ...]
        """
        predictions = []
        
        for code, df in data.items():
            if df.empty:
                continue
            
            try:
                result = self.predict(df.tail(1))
                proba = result['pred_proba'].iloc[-1]
                predictions.append((code, proba))
            except Exception as e:
                logger.warning(f"{code} 预测失败: {e}")
        
        # 按概率排序
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:top_n]


if __name__ == "__main__":
    # 测试代码
    from src.data import DataFetcher
    from src.factors import FactorEngine
    
    # 准备数据
    fetcher = DataFetcher()
    factor_engine = FactorEngine()
    
    codes = ['000001', '000002', '600000', '600036', '601398']
    data = {}
    
    for code in codes:
        df = fetcher.get_daily_data(code, start_date='2023-01-01')
        df = factor_engine.compute(df, ['momentum_5', 'momentum_20', 'ma_20', 'rsi_14', 'volatility_20'])
        data[code] = df
    
    # 训练模型
    predictor = StockPredictor(model_type='random_forest')  # 使用sklearn避免依赖问题
    
    feature_cols = ['momentum_5', 'momentum_20', 'rsi_14', 'volatility_20']
    metrics = predictor.train(data, feature_cols, lookahead=5, threshold=0.02)
    
    print("\n模型训练结果:")
    for k, v in metrics.to_dict().items():
        print(f"  {k}: {v}")
    
    # 选股
    top_stocks = predictor.select_stocks(data, top_n=3)
    print("\nTop 3 推荐股票:")
    for code, proba in top_stocks:
        print(f"  {code}: {proba:.4f}")
