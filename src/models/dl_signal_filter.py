"""
深度学习信号过滤器 - Phase 10.3

将已训练的LSTM/Transformer模型预测结果作为策略信号的过滤层：
  - 策略说买入 + DL预测上涨 → 保持买入
  - 策略说买入 + DL预测下跌 → 降级为持有
  - 策略说卖出 + DL预测下跌 → 保持卖出
  - 策略说卖出 + DL预测上涨 → 降级为持有

设计原则:
  - DL作为辅助而非主导，只做否决不做生成
  - 预测置信度低时自动跳过过滤
  - 模型不可用时优雅降级（不影响策略运行）
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, List
from pathlib import Path
from loguru import logger


class DLSignalFilter:
    """深度学习信号过滤器

    加载已训练的DL模型，对策略信号进行二次验证过滤。
    """

    def __init__(self, model_dir: str = "data/models",
                 enabled: bool = True,
                 confidence_threshold: float = 0.6):
        """
        Args:
            model_dir: 模型存储目录
            enabled: 是否启用过滤（False时直接透传信号）
            confidence_threshold: DL预测置信度阈值，低于此值时不过滤
        """
        self.model_dir = Path(model_dir)
        self.enabled = enabled
        self.confidence_threshold = confidence_threshold

        self._model = None
        self._scaler = None
        self._model_type = None
        self._feature_cols = None

        if self.enabled:
            self._try_load_model()

    def _try_load_model(self):
        """尝试加载最新的DL模型"""
        try:
            # 查找最新的模型文件
            model_files = list(self.model_dir.glob("lstm_*.pth"))
            model_files += list(self.model_dir.glob("transformer_*.pth"))

            if not model_files:
                logger.info("DL过滤器: 未找到已训练模型，过滤功能禁用")
                self.enabled = False
                return

            # 按修改时间排序，取最新
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            self._model_type = 'lstm' if 'lstm' in latest_model.name else 'transformer'

            # 加载模型
            from src.models.dl_models import LSTMModel, TransformerModel

            checkpoint = torch.load(latest_model, map_location='cpu')

            if self._model_type == 'lstm':
                self._model = LSTMModel(
                    input_dim=checkpoint.get('input_dim', 10),
                    hidden_dim=checkpoint.get('hidden_dim', 64),
                    num_layers=checkpoint.get('num_layers', 2),
                )
            else:
                self._model = TransformerModel(
                    input_dim=checkpoint.get('input_dim', 10),
                    d_model=checkpoint.get('d_model', 64),
                    nhead=checkpoint.get('nhead', 4),
                    num_layers=checkpoint.get('num_layers', 2),
                )

            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.eval()

            # 加载归一化器
            scaler_path = self.model_dir / f"{self._model_type}_scaler.pkl"
            if scaler_path.exists():
                import pickle
                with open(scaler_path, 'rb') as f:
                    self._scaler = pickle.load(f)

            # 加载特征列表
            features_path = self.model_dir / f"{self._model_type}_features.txt"
            if features_path.exists():
                with open(features_path, 'r') as f:
                    self._feature_cols = [line.strip() for line in f]

            logger.info(f"DL过滤器已加载: {self._model_type} 模型 ({latest_model.name})")

        except Exception as e:
            logger.warning(f"DL模型加载失败: {e}，过滤功能禁用")
            self.enabled = False

    def filter_signal(self, action: str, code: str,
                     factored_df: pd.DataFrame,
                     confidence: float = 70.0) -> Dict:
        """过滤单个策略信号

        Args:
            action: 策略原始信号 ("buy"/"sell"/"hold")
            code: 股票代码
            factored_df: 已计算因子的DataFrame
            confidence: 策略信号置信度 (0-100)

        Returns:
            {
                'action': str,          # 过滤后的信号
                'confidence': float,    # 调整后的置信度
                'dl_prediction': float, # DL预测值（涨跌幅）
                'dl_confidence': float, # DL预测置信度
                'filtered': bool,       # 是否被过滤
                'reason': str,          # 过滤原因
            }
        """
        result = {
            'action': action,
            'confidence': confidence,
            'dl_prediction': 0.0,
            'dl_confidence': 0.0,
            'filtered': False,
            'reason': '',
        }

        # 未启用或hold信号直接返回
        if not self.enabled or action == 'hold':
            return result

        # DL预测
        try:
            prediction, dl_conf = self._predict(factored_df, code)
            result['dl_prediction'] = prediction
            result['dl_confidence'] = dl_conf

            # 置信度低于阈值，不过滤
            if dl_conf < self.confidence_threshold:
                result['reason'] = f"DL置信度低({dl_conf:.1%})，不过滤"
                return result

            # 过滤逻辑
            if action == 'buy':
                if prediction < -0.01:  # DL预测下跌>1%
                    result['action'] = 'hold'
                    result['confidence'] = confidence * 0.5
                    result['filtered'] = True
                    result['reason'] = f"DL预测下跌{prediction:.2%}，买入信号被否决"
                elif prediction < 0:  # DL预测小幅下跌
                    result['confidence'] = confidence * 0.8
                    result['reason'] = f"DL预测略微下跌{prediction:.2%}，信号减弱"
                else:  # DL预测上涨，确认买入
                    result['confidence'] = min(100, confidence * 1.1)
                    result['reason'] = f"DL预测上涨{prediction:.2%}，信号确认"

            elif action == 'sell':
                if prediction > 0.01:  # DL预测上涨>1%
                    result['action'] = 'hold'
                    result['confidence'] = confidence * 0.5
                    result['filtered'] = True
                    result['reason'] = f"DL预测上涨{prediction:.2%}，卖出信号被否决"
                elif prediction > 0:  # DL预测小幅上涨
                    result['confidence'] = confidence * 0.8
                    result['reason'] = f"DL预测略微上涨{prediction:.2%}，信号减弱"
                else:  # DL预测下跌，确认卖出
                    result['confidence'] = min(100, confidence * 1.1)
                    result['reason'] = f"DL预测下跌{prediction:.2%}，信号确认"

        except Exception as e:
            logger.debug(f"DL过滤失败 {code}: {e}")
            result['reason'] = f"DL预测异常，保持原信号"

        return result

    def filter_batch(self, signals: List[Dict],
                    factored_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """批量过滤策略信号

        Args:
            signals: 策略信号列表
                [{'code': ..., 'action': ..., 'confidence': ...}, ...]
            factored_data: {股票代码: 因子DataFrame}

        Returns:
            过滤后的信号列表
        """
        filtered_signals = []

        for sig in signals:
            code = sig.get('code')
            action = sig.get('action', 'hold')
            conf = sig.get('confidence', 70.0)

            if code not in factored_data:
                filtered_signals.append(sig)
                continue

            result = self.filter_signal(
                action, code, factored_data[code], conf
            )

            # 更新信号
            sig['action'] = result['action']
            sig['confidence'] = result['confidence']
            sig['dl_prediction'] = result['dl_prediction']
            sig['dl_filtered'] = result['filtered']
            sig['dl_reason'] = result['reason']

            filtered_signals.append(sig)

        return filtered_signals

    def _predict(self, factored_df: pd.DataFrame,
                code: str) -> tuple[float, float]:
        """使用DL模型预测未来收益率

        Returns:
            (预测收益率, 置信度)
        """
        if self._model is None or self._feature_cols is None:
            return 0.0, 0.0

        try:
            # 提取特征
            features = []
            for col in self._feature_cols:
                if col in factored_df.columns:
                    features.append(factored_df[col].values[-10:])  # 最近10天
                else:
                    features.append(np.zeros(10))  # 缺失特征填0

            X = np.column_stack(features).astype(np.float32)

            # 归一化
            if self._scaler is not None:
                X_flat = X.reshape(-1, X.shape[-1])
                X_scaled = self._scaler.transform(X_flat)
                X = X_scaled.reshape(1, *X.shape)  # (1, seq_len, features)
            else:
                X = X.reshape(1, *X.shape)

            # 预测
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                output = self._model(X_tensor)
                prediction = float(output[0, 0].item())

            # 置信度估算（基于预测值的绝对值）
            confidence = min(1.0, abs(prediction) / 0.05)  # 预测>5%时置信度=1

            return prediction, confidence

        except Exception as e:
            logger.debug(f"DL预测失败 {code}: {e}")
            return 0.0, 0.0

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if not self.enabled or self._model is None:
            return {'enabled': False}

        return {
            'enabled': True,
            'model_type': self._model_type,
            'feature_count': len(self._feature_cols) if self._feature_cols else 0,
            'confidence_threshold': self.confidence_threshold,
        }


# ==================== 便捷函数 ====================

def apply_dl_filter_to_report(report, factored_df: pd.DataFrame,
                              dl_filter: Optional[DLSignalFilter] = None):
    """将DL过滤应用到DecisionReport

    Args:
        report: DecisionReport 对象
        factored_df: 因子DataFrame
        dl_filter: DLSignalFilter实例（None时创建新实例）

    Returns:
        修改后的 report（原地修改）
    """
    if dl_filter is None:
        dl_filter = DLSignalFilter()

    if not dl_filter.enabled:
        return report

    result = dl_filter.filter_signal(
        report.action, report.code, factored_df, report.confidence
    )

    # 更新report
    report.action = result['action']
    report.confidence = result['confidence']

    # 添加DL分析到reasoning
    if result['filtered']:
        report.reasoning.insert(0, f"⚠️ {result['reason']}")
    elif result['reason']:
        report.reasoning.append(f"🤖 {result['reason']}")

    return report
