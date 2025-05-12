import os
import sys
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model_load():
    # 获取项目根目录
    root_dir = Path(__file__).parent.parent
    model_dir = root_dir / "models/asr/sherpa-onnx-streaming-zipformer-en-2023-06-26"
    
    logger.info(f"当前工作目录: {os.getcwd()}")
    logger.info(f"项目根目录: {root_dir}")
    logger.info(f"模型目录: {model_dir}")
    
    # 检查文件是否存在
    files = {
        "encoder": "encoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        "decoder": "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        "joiner": "joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
        "tokens": "tokens.txt"
    }
    
    # 验证模型文件
    for key, filename in files.items():
        file_path = model_dir / filename
        logger.info(f"检查{key}文件: {file_path}")
        if not file_path.exists():
            logger.error(f"{key}文件不存在: {file_path}")
            return False
        else:
            # 检查文件大小
            size = file_path.stat().st_size
            logger.info(f"{key}文件大小: {size/1024/1024:.2f} MB")
            
    try:
        # 验证sherpa_onnx安装
        import sherpa_onnx
        logger.info(f"sherpa_onnx版本: {sherpa_onnx.__version__}")
        
        # 创建识别器
        logger.info("开始创建识别器...")
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(model_dir / "tokens.txt"),
            encoder=str(model_dir / "encoder-epoch-99-avg-1-chunk-16-left-128.onnx"),
            decoder=str(model_dir / "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"),
            joiner=str(model_dir / "joiner-epoch-99-avg-1-chunk-16-left-128.onnx"),
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=20.0,
            debug=True  # 启用调试输出
        )
        logger.info("识别器创建成功!")
        return True
        
    except ImportError as e:
        logger.error(f"sherpa_onnx导入失败: {e}")
        return False
    except Exception as e:
        logger.error(f"创建识别器失败: {e}")
        logger.error(f"错误类型: {type(e)}")
        # 打印完整的堆栈跟踪
        import traceback
        logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_model_load()
    sys.exit(0 if success else 1)