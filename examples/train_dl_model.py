"""
深度学习模型训练示例

演示如何使用 TrainingPipeline 训练 LSTM/Transformer 模型
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataFetcher
from src.factors import FactorEngine
from src.train_pipeline import TrainingPipeline
from loguru import logger

def main():
    logger.info("开始深度学习模型训练示例...")
    
    # 1. 准备数据
    fetcher = DataFetcher()
    engine = FactorEngine()
    
    codes = ['000001', '600000', '601398', '600036']
    data = {}
    
    feature_cols = ['ma_5', 'ma_10', 'ma_20', 'rsi_14', 'momentum_5', 'volatility_20']
    
    logger.info("正在获取数据并计算因子...")
    for code in codes:
        try:
            # 获取最近1年数据
            df = fetcher.get_daily_data(code, start_date='2023-01-01')
            if df.empty:
                logger.warning(f"{code} 数据为空")
                continue
                
            # 计算因子
            df = engine.compute(df, feature_cols)
            data[code] = df
            logger.info(f"{code}: {len(df)} 条数据")
        except Exception as e:
            logger.error(f"{code} 处理失败: {e}")
            
    if not data:
        logger.error("没有可用数据")
        return

    # 2. 初始化训练管道
    pipeline = TrainingPipeline(data_dir="data/models")
    
    # 3. 训练 LSTM 模型
    try:
        logger.info("\n=== 训练 LSTM 模型 ===")
        metrics, path = pipeline.train_dl_model(
            data=data,
            feature_cols=feature_cols,
            model_type='lstm',
            epochs=5,
            seq_len=10
        )
        logger.info(f"LSTM 训练完成! 模型保存至: {path}")
    except ImportError:
        logger.error("PyTorch 未安装，无法训练深度学习模型")
        logger.info("请运行: pip install torch")
    except Exception as e:
        logger.error(f"LSTM 训练失败: {e}")
        
    # 4. 训练 Transformer 模型
    try:
        logger.info("\n=== 训练 Transformer 模型 ===")
        metrics, path = pipeline.train_dl_model(
            data=data,
            feature_cols=feature_cols,
            model_type='transformer',
            epochs=5,
            seq_len=10
        )
        logger.info(f"Transformer 训练完成! 模型保存至: {path}")
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"Transformer 训练失败: {e}")

if __name__ == "__main__":
    main()
