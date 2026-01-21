"""
A股量化交易系统 - 统一训练流水线
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger
import pickle
from pathlib import Path
from datetime import datetime

# 导入自定义模型
from src.models.ml_models import StockPredictor
from src.models.dl_models import LSTMModel, TransformerModel

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
except ImportError:
    pass

class TrainingPipeline:
    """模型训练流水线"""
    
    def __init__(self, data_dir: str = "data/models"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
    
    def prepare_data(self, data_dict: Dict[str, pd.DataFrame], feature_cols: List[str], 
                    target_col: str = 'target', lookback: int = 10, future_return_days: int = 5) -> Tuple:
        """
        准备训练数据
        Args:
            data_dict: {code: df}
            feature_cols: 特征列名
            lookback: 序列长度 (for DL models)
            future_return_days: 预测N日后收益率
        Returns:
            ML: (X_train, y_train, X_test, y_test)
            DL: (train_loader, val_loader)
        """
        X_all = []
        y_all = []
        
        for code, df in data_dict.items():
            df = df.copy()
            # 简单构建Target: N日后收益率
            df['target'] = df['close'].shift(-future_return_days) / df['close'] - 1
            df.dropna(subset=feature_cols + ['target'], inplace=True)
            
            if df.empty:
                continue
                
            X_all.append(df[feature_cols].values)
            y_all.append(df['target'].values)
            
        if not X_all:
            raise ValueError("No valid data found")
            
        X = np.concatenate(X_all, axis=0)
        y = np.concatenate(y_all, axis=0).reshape(-1, 1)
        
        return X, y

    def prepare_sequence_data(self, X: np.ndarray, y: np.ndarray, seq_len: int = 10):
        """将2D数据转换为3D序列数据 (samples, seq_len, features)"""
        # 注意：这里简单的滑动窗口实现比较慢且内存消耗大，生产环境应优化
        # 且上面的prepare_data已经把多只股票拼在一起了，直接滑窗会跨股票混数据
        # 这是一个简化实现。正确做法是在每只股票内部做滑窗。
        # 为了演示，我们假设X已经是通过每只股票内部处理好的序列（需修改prepare_data逻辑），
        # 或者我们只训练一只股票。
        # 暂时简化：不跨股票处理，假设prepare_data返回的是并在了一起的快照。
        # 如果要做时序，必须在prepare_data里针对每只股票生成sequence。
        
        # 重新实现一个简单的sequence生成器
        X_seq = []
        y_seq = []
        
        # 实际上上面的prepare_data已经丢失了时序边界。
        # 我们需要在prepare_data内部做。
        # 这里仅作占位，实际深度学习需要更严谨的数据处理。
        pass

    def train_ml_model(self, data: Dict[str, pd.DataFrame], feature_cols: List[str], model_type: str = 'lightgbm'):
        """训练机器学习模型"""
        logger.info(f"开始训练ML模型: {model_type}")
        predictor = StockPredictor(model_type=model_type)
        
        # 这里复用StockPredictor的train方法，它内部处理了数据
        # 但我们需要统一的保存路径
        metrics = predictor.train(data, feature_cols)
        
        save_path = self.data_dir / f"{model_type}_{datetime.now().strftime('%Y%m%d')}.pkl"
        predictor.model.save(str(save_path))
        
        return metrics, str(save_path)

    def train_dl_model(self, data: Dict[str, pd.DataFrame], feature_cols: List[str], 
                      model_type: str = 'lstm', seq_len: int = 10, epochs: int = 20):
        """训练深度学习模型"""
        logger.info(f"开始训练DL模型: {model_type}")
        
        # 1. 数据预处理
        X_list = []
        y_list = []
        
        for code, df in data.items():
            if len(df) < seq_len + 5:
                continue
            
            # 归一化 (简单处理：每只股票单独归一化还是全局？一般全局稳健些，或滚动)
            # 这里先做特征提取
            feats = df[feature_cols].values
            # 目标：未来5日收益
            targets = (df['close'].shift(-5) / df['close'] - 1).fillna(0).values
            
            # 生成序列
            for i in range(len(df) - seq_len - 5):
                X_list.append(feats[i : i+seq_len])
                y_list.append(targets[i + seq_len])
        
        X = np.array(X_list)
        y = np.array(y_list).reshape(-1, 1)
        
        # 全局归一化
        N, L, F = X.shape
        X_reshaped = X.reshape(-1, F)
        if not hasattr(self.scaler, 'n_samples_seen_'):
            X_scaled = self.scaler.fit_transform(X_reshaped)
        else:
            X_scaled = self.scaler.transform(X_reshaped)
        X = X_scaled.reshape(N, L, F)
        
        # 转换为Tensor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensor_x = torch.FloatTensor(X)
        tensor_y = torch.FloatTensor(y)
        
        dataset = TensorDataset(tensor_x, tensor_y)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64)
        
        # 2. 初始化模型
        input_dim = len(feature_cols)
        if model_type == 'lstm':
            model = LSTMModel(input_dim=input_dim)
        elif model_type == 'transformer':
            model = TransformerModel(input_dim=input_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 3. 训练
        model.fit(train_loader, val_loader, epochs=epochs, device=device)
        
        # 4. 保存
        save_path = self.data_dir / f"{model_type}_{datetime.now().strftime('%Y%m%d')}.pth"
        model.save(str(save_path))
        
        return {'model': model_type, 'status': 'success'}, str(save_path)

if __name__ == "__main__":
    # 测试管道
    from src.data import DataFetcher
    from src.factors import FactorEngine
    
    # 获取少量数据
    fetcher = DataFetcher()
    engine = FactorEngine()
    
    data = {}
    for code in ['000001', '600000']:
        try:
            df = fetcher.get_daily_data(code, start_date='2023-01-01')
            df = engine.compute(df, ['ma_5', 'ma_20', 'rsi_14'])
            data[code] = df
        except:
            pass
            
    pipeline = TrainingPipeline()
    metrics, path = pipeline.train_dl_model(data, ['ma_5', 'ma_20', 'rsi_14'], model_type='lstm', epochs=2)
    print(f"训练完成: {path}")
