"""
测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的在线转录功能
确保不会自动降级到 Vosk 模型
"""
import os
import sys
import time

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 设置当前工作目录为项目根目录，确保能正确找到配置文件
os.chdir(project_root)

# 导入必要的模块
from src.core.asr.model_manager import ASRModelManager
from src.utils.sherpa_logger import sherpa_logger

def test_sherpa_0626_no_fallback():
    """测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型不会自动降级到 Vosk 模型"""
    print("=" * 80)
    print("测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型不会自动降级到 Vosk 模型")
    print("=" * 80)

    # 初始化日志工具
    sherpa_logger.setup()
    sherpa_log_file = sherpa_logger.get_log_file()
    print(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")

    # 创建模型管理器
    model_manager = ASRModelManager()

    # 加载 sherpa_0626 模型
    print("加载 sherpa_0626 模型...")
    if model_manager.load_model("sherpa_0626"):
        print("成功加载 sherpa_0626 模型")
    else:
        print("加载 sherpa_0626 模型失败")
        return

    # 获取当前引擎类型
    engine_type = model_manager.get_current_engine_type()
    print(f"当前引擎类型: {engine_type}")

    # 创建识别器
    print("创建识别器...")
    recognizer = model_manager.create_recognizer()
    if not recognizer:
        print("创建识别器失败")
        return
    print(f"成功创建识别器: {type(recognizer).__name__}")

    # 检查识别器的引擎类型
    if hasattr(recognizer, 'engine_type'):
        print(f"识别器引擎类型: {recognizer.engine_type}")
        
        # 确保引擎类型是 sherpa_0626，而不是 vosk
        if recognizer.engine_type != "sherpa_0626":
            print(f"错误: 识别器引擎类型应该是 sherpa_0626，但实际是 {recognizer.engine_type}")
            return
        
        print("测试通过: 识别器引擎类型正确")
    else:
        print("错误: 识别器没有 engine_type 属性")
        return

if __name__ == "__main__":
    test_sherpa_0626_no_fallback()
