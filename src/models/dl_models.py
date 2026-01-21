"""
A股量化交易系统 - 深度学习模型 (PyTorch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
import pickle
from pathlib import Path

# 检查PyTorch可用性
try:
    _ = torch.tensor([1.0])
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch未安装，深度学习功能不可用")

class BaseDLModel(nn.Module):
    """深度学习模型基类"""
    def __init__(self, name: str = "BaseDLModel"):
        super().__init__()
        self.name = name
        self.input_dim = 0
        self.feature_names = []
        self.scaler = None
    
    def save(self, path: str):
        """保存模型"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'state_dict': self.state_dict(),
            'feature_names': self.feature_names,
            'input_dim': self.input_dim,
            'name': self.name,
            'scaler': self.scaler
        }
        torch.save(state, save_path)
        logger.info(f"模型已保存: {path}")

    def load(self, path: str):
        """加载模型"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch未安装")
        
        state = torch.load(path)
        self.load_state_dict(state['state_dict'])
        self.feature_names = state['feature_names']
        self.input_dim = state['input_dim']
        self.name = state['name']
        self.scaler = state.get('scaler')
        logger.info(f"模型已加载: {path}")

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 64, lr: float = 0.001, validation_split: float = 0.2):
        """训练模型"""
        raise NotImplementedError

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """预测得分"""
        raise NotImplementedError


class LSTMModel(BaseDLModel):
    """LSTM时序预测模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 1, dropout: float = 0.2):
        super().__init__(name="LSTM")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
    def fit(self, train_loader, val_loader=None, epochs=50, lr=0.001, device='cpu'):
        """
        训练模型
        Args:
            train_loader: DataLoader, batch data (x, y)
                x: (batch, seq_len, input_dim)
                y: (batch, 1)
        """
        self.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        best_loss = float('inf')
        
        logger.info(f"开始训练LSTM模型 (Device: {device})")
        
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            val_loss_str = ""
            if val_loader:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_val, y_val in val_loader:
                        X_val, y_val = X_val.to(device), y_val.to(device)
                        outputs = self(X_val)
                        loss = criterion(outputs, y_val)
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)
                val_loss_str = f", Val Loss: {avg_val_loss:.6f}"
                
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    # 可以在这里保存最佳模型状态
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}{val_loss_str}")
    
    def predict_score(self, X: np.ndarray, device='cpu') -> np.ndarray:
        """预测"""
        self.eval()
        self.to(device)
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            if X_tensor.dim() == 2:
                # 假如输入是 (n_samples, input_dim)，需要unsqueeze增加seq_len维度
                # 但LSTM通常需要明确seq_len。这里假设如果没传seq_len，就是单步
                X_tensor = X_tensor.unsqueeze(1) 
            
            output = self(X_tensor)
            return output.cpu().numpy()


class TransformerModel(BaseDLModel):
    """Transformer Encoder时序预测模型"""
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, output_dim: int = 1, dropout: float = 0.1):
        super().__init__(name="Transformer")
        self.input_dim = input_dim
        
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.embedding(x)
        # x: (batch, seq_len, d_model)
        x = self.transformer_encoder(x)
        # Global Average Pooling or just take last token
        # Using Global Average Pooling over sequence length
        x = x.mean(dim=1) 
        out = self.fc(x)
        return out

    # 使用与LSTM相同的fit和predict方法（可以重构到BaseDLModel如果逻辑完全一致）
    # 这里为了简单直接复用LSTM的逻辑框架，因为BaseDLModel没有实现fit
    
    def fit(self, train_loader, val_loader=None, epochs=50, lr=0.001, device='cpu'):
        self.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        logger.info(f"开始训练Transformer模型 (Device: {device})")
        
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}")

    def predict_score(self, X: np.ndarray, device='cpu') -> np.ndarray:
        self.eval()
        self.to(device)
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            if X_tensor.dim() == 2:
                X_tensor = X_tensor.unsqueeze(1)
            output = self(X_tensor)
            return output.cpu().numpy()

if __name__ == "__main__":
    # 简单测试
    if TORCH_AVAILABLE:
        # 模拟数据: (batch=32, seq_len=10, features=5)
        X = np.random.randn(32, 10, 5).astype(np.float32)
        y = np.random.randn(32, 1).astype(np.float32)
        
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=8)
        
        model = LSTMModel(input_dim=5)
        model.fit(loader, epochs=2)
        
        pred = model.predict_score(X[:2])
        print("Prediction shape:", pred.shape)
        
        t_model = TransformerModel(input_dim=5)
        t_model.fit(loader, epochs=2)
