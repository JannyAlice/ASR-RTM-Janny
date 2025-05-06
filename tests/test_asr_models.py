"""
测试ASR模型功能
"""
import os
import sys
import time
import logging
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test_asr_models.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 导入需要测试的模块
from src.core.asr.model_manager import ASRModelManager
from src.utils.config_manager import config_manager

def test_model_loading():
    """测试模型加载"""
    logger.info("=== 测试模型加载 ===")

    # 创建ASRModelManager实例
    model_manager = ASRModelManager()

    # 获取可用的引擎列表
    engines = model_manager.get_available_engines()
    logger.info(f"可用的引擎列表: {engines}")

    # 测试加载每个可用的引擎
    for engine_name, enabled in engines.items():
        if enabled:
            logger.info(f"测试加载引擎: {engine_name}")
            success = model_manager.load_model(engine_name)
            logger.info(f"引擎 {engine_name} 加载结果: {'成功' if success else '失败'}")

            if success:
                # 获取当前引擎类型
                engine_type = model_manager.get_current_engine_type()
                logger.info(f"当前引擎类型: {engine_type}")

                # 创建识别器
                recognizer = model_manager.create_recognizer()
                logger.info(f"识别器类型: {type(recognizer).__name__}")

                # 检查识别器是否有engine_type属性
                if hasattr(recognizer, 'engine_type'):
                    logger.info(f"识别器引擎类型: {recognizer.engine_type}")

                # 重置引擎
                model_manager.reset()
                logger.info(f"引擎 {engine_name} 重置成功")
        else:
            logger.info(f"引擎 {engine_name} 未启用，跳过测试")

    logger.info("=== 模型加载测试完成 ===")

def test_online_transcription():
    """测试在线转录"""
    logger.info("=== 测试在线转录 ===")

    # 创建ASRModelManager实例
    model_manager = ASRModelManager()

    # 获取可用的引擎列表
    engines = model_manager.get_available_engines()
    logger.info(f"可用的引擎列表: {engines}")

    # 测试每个可用的引擎
    for engine_name, enabled in engines.items():
        if enabled:
            logger.info(f"测试引擎: {engine_name}")
            success = model_manager.load_model(engine_name)
            logger.info(f"引擎 {engine_name} 加载结果: {'成功' if success else '失败'}")

            if success:
                # 获取当前引擎类型
                engine_type = model_manager.get_current_engine_type()
                logger.info(f"当前引擎类型: {engine_type}")

                # 创建识别器
                recognizer = model_manager.create_recognizer()
                logger.info(f"识别器类型: {type(recognizer).__name__}")

                # 检查识别器是否有engine_type属性
                if hasattr(recognizer, 'engine_type'):
                    logger.info(f"识别器引擎类型: {recognizer.engine_type}")

                # 生成测试音频数据
                logger.info("生成测试音频数据")
                audio_data = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)

                # 测试AcceptWaveform方法
                logger.info("测试AcceptWaveform方法")
                try:
                    result = recognizer.AcceptWaveform(audio_data)
                    logger.info(f"AcceptWaveform结果: {result}")
                except Exception as e:
                    logger.error(f"AcceptWaveform错误: {e}")

                # 测试Result方法
                logger.info("测试Result方法")
                try:
                    result = recognizer.Result()
                    logger.info(f"Result结果: {result}")
                except Exception as e:
                    logger.error(f"Result错误: {e}")

                # 测试PartialResult方法
                logger.info("测试PartialResult方法")
                try:
                    result = recognizer.PartialResult()
                    logger.info(f"PartialResult结果: {result}")
                except Exception as e:
                    logger.error(f"PartialResult错误: {e}")

                # 测试Reset方法
                logger.info("测试Reset方法")
                try:
                    recognizer.Reset()
                    logger.info("Reset成功")
                except Exception as e:
                    logger.error(f"Reset错误: {e}")
        else:
            logger.info(f"引擎 {engine_name} 未启用，跳过测试")

    logger.info("=== 在线转录测试完成 ===")

def test_file_transcription():
    """测试文件转录"""
    logger.info("=== 测试文件转录 ===")

    # 创建ASRModelManager实例
    model_manager = ASRModelManager()

    # 获取可用的引擎列表
    engines = model_manager.get_available_engines()
    logger.info(f"可用的引擎列表: {engines}")

    # 测试文件路径
    test_file = r"C:\Users\crige\RealtimeTrans\vosk-api-bak-J\mytest.mp4"
    if not os.path.exists(test_file):
        logger.error(f"测试文件不存在: {test_file}")
        return

    # 测试每个可用的引擎
    for engine_name, enabled in engines.items():
        if enabled:
            logger.info(f"测试引擎: {engine_name}")
            success = model_manager.load_model(engine_name)
            logger.info(f"引擎 {engine_name} 加载结果: {'成功' if success else '失败'}")

            if success:
                # 获取当前引擎类型
                engine_type = model_manager.get_current_engine_type()
                logger.info(f"当前引擎类型: {engine_type}")

                # 测试文件转录
                logger.info(f"测试文件转录: {test_file}")
                try:
                    result = model_manager.transcribe_file(test_file)
                    if result:
                        logger.info(f"转录结果: {result[:100]}...")
                    else:
                        logger.warning("转录结果为空")
                except Exception as e:
                    logger.error(f"文件转录错误: {e}")
        else:
            logger.info(f"引擎 {engine_name} 未启用，跳过测试")

    logger.info("=== 文件转录测试完成 ===")

def test_model_switching():
    """测试模型切换"""
    logger.info("=== 测试模型切换 ===")

    # 创建ASRModelManager实例
    model_manager = ASRModelManager()

    # 获取可用的引擎列表
    engines = model_manager.get_available_engines()
    logger.info(f"可用的引擎列表: {engines}")

    # 获取可用的引擎名称列表
    available_engines = [name for name, enabled in engines.items() if enabled]

    if len(available_engines) < 2:
        logger.warning("可用引擎数量不足，无法测试模型切换")
        return

    # 测试模型切换
    for i in range(len(available_engines)):
        engine1 = available_engines[i]
        engine2 = available_engines[(i + 1) % len(available_engines)]

        logger.info(f"测试从 {engine1} 切换到 {engine2}")

        # 加载第一个引擎
        success1 = model_manager.load_model(engine1)
        logger.info(f"引擎 {engine1} 加载结果: {'成功' if success1 else '失败'}")

        if success1:
            # 获取当前引擎类型
            engine_type1 = model_manager.get_current_engine_type()
            logger.info(f"当前引擎类型: {engine_type1}")

            # 加载第二个引擎
            success2 = model_manager.load_model(engine2)
            logger.info(f"引擎 {engine2} 加载结果: {'成功' if success2 else '失败'}")

            if success2:
                # 获取当前引擎类型
                engine_type2 = model_manager.get_current_engine_type()
                logger.info(f"当前引擎类型: {engine_type2}")

                # 检查引擎类型是否正确
                if engine_type2 == engine2:
                    logger.info(f"模型切换成功: {engine1} -> {engine2}")
                else:
                    logger.warning(f"模型切换后引擎类型不匹配: 期望 {engine2}，实际 {engine_type2}")

    logger.info("=== 模型切换测试完成 ===")

def main():
    """主函数"""
    logger.info("开始测试ASR模型功能")

    # 测试模型加载
    test_model_loading()

    # 测试在线转录
    test_online_transcription()

    # 测试文件转录
    test_file_transcription()

    # 测试模型切换
    test_model_switching()

    logger.info("ASR模型功能测试完成")

if __name__ == "__main__":
    main()
