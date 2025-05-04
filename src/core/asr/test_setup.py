import sys
import os
from pathlib import Path
import logging

# 获取项目根目录
try:
    project_root = Path(__file__).resolve().parents[3]  # 从 src/core/asr 向上3级到项目根目录
    if not project_root.exists():
        raise FileNotFoundError(f"项目根目录不存在: {project_root}")

    # 添加项目根目录到 sys.path
    sys_path = str(project_root)
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

    # 检查模型目录
    model_dir = project_root / "models" / "asr" / "sherpa-onnx-streaming-zipformer-en-2023-06-26"
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    # 创建日志目录
    log_dir = project_root / "logs"
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    # 导入 SherpaOnnxASR
    from src.core.asr.sherpa_engine import SherpaOnnxASR
    from src.utils.sherpa_logger import SherpaLogger

    # 初始化日志
    logger = SherpaLogger(str(log_dir)).logger
    logger.info(f"项目根目录: {project_root}")
    logger.info(f"模型目录: {model_dir}")

    # 配置模型参数
    model_config = {
        "type": "int8",
        "name": "0626",
        "sample_rate": 16000,
        "num_threads": 4,
        "enable_endpoint": 1,
        "rule1_min_trailing_silence": 3.0,
        "rule2_min_trailing_silence": 1.5,
        "rule3_min_utterance_length": 25
    }

    logger.info("初始化 SherpaOnnxASR...")
    asr_engine = SherpaOnnxASR(str(model_dir), model_config)

    # 调用 setup 方法
    logger.info("开始调用 setup 方法...")
    try:
        setup_result = asr_engine.setup()
        if not setup_result:
            logger.error("setup 方法返回 False")
            raise RuntimeError("初始化 SherpaOnnxASR 失败")
        logger.info("setup 方法成功返回 True")
    except Exception as e:
        logger.error(f"setup 方法抛出异常: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    logger.info("初始化成功")

except Exception as e:
    print(f"错误: {e}")
    import traceback
    print(traceback.format_exc())
    sys.exit(1)