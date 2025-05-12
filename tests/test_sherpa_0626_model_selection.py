"""
测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型选择功能
"""
import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 设置当前工作目录为项目根目录，确保能正确找到配置文件
os.chdir(project_root)

# 导入必要的模块
from src.core.asr.model_manager import ASRModelManager
from src.utils.sherpa_logger import sherpa_logger

def test_sherpa_0626_model_selection():
    """测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型选择功能"""
    print("=" * 80)
    print("测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型选择功能")
    print("=" * 80)

    # 初始化日志工具
    sherpa_logger.setup()
    sherpa_log_file = sherpa_logger.get_log_file()
    print(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")

    # 创建模型管理器
    model_manager = ASRModelManager()

    # 获取可用的引擎列表
    engines = model_manager.get_available_engines()
    print(f"可用的引擎列表: {engines}")

    # 测试加载 sherpa_0626 模型
    print("\n测试加载 sherpa_0626 模型...")
    if model_manager.load_model("sherpa_0626"):
        print("成功加载 sherpa_0626 模型")
        
        # 获取当前引擎类型
        engine_type = model_manager.get_current_engine_type()
        print(f"当前引擎类型: {engine_type}")
        
        # 检查引擎是否已初始化
        if model_manager.current_engine:
            print(f"引擎已初始化: {type(model_manager.current_engine).__name__}")
            
            # 检查引擎的配置
            if hasattr(model_manager.current_engine, 'config'):
                print(f"引擎配置: {model_manager.current_engine.config}")
            else:
                print("引擎没有配置属性")
        else:
            print("引擎未初始化")
    else:
        print("加载 sherpa_0626 模型失败")

    # 测试创建识别器
    print("\n测试创建识别器...")
    recognizer = model_manager.create_recognizer()
    if recognizer:
        print(f"成功创建识别器: {type(recognizer).__name__}")
        
        # 检查识别器的引擎类型
        if hasattr(recognizer, 'engine_type'):
            print(f"识别器引擎类型: {recognizer.engine_type}")
        else:
            print("识别器没有引擎类型属性")
    else:
        print("创建识别器失败")

    # 测试加载 vosk 模型
    print("\n测试加载 vosk 模型...")
    if model_manager.load_model("vosk"):
        print("成功加载 vosk 模型")
        
        # 获取当前引擎类型
        engine_type = model_manager.get_current_engine_type()
        print(f"当前引擎类型: {engine_type}")
        
        # 检查引擎是否已初始化
        if model_manager.current_engine:
            print(f"引擎已初始化: {type(model_manager.current_engine).__name__}")
        else:
            print("引擎未初始化")
    else:
        print("加载 vosk 模型失败")

    # 测试创建识别器
    print("\n测试创建识别器...")
    recognizer = model_manager.create_recognizer()
    if recognizer:
        print(f"成功创建识别器: {type(recognizer).__name__}")
        
        # 检查识别器的引擎类型
        if hasattr(recognizer, 'engine_type'):
            print(f"识别器引擎类型: {recognizer.engine_type}")
        else:
            print("识别器没有引擎类型属性")
    else:
        print("创建识别器失败")

    # 再次测试加载 sherpa_0626 模型
    print("\n再次测试加载 sherpa_0626 模型...")
    if model_manager.load_model("sherpa_0626"):
        print("成功加载 sherpa_0626 模型")
        
        # 获取当前引擎类型
        engine_type = model_manager.get_current_engine_type()
        print(f"当前引擎类型: {engine_type}")
        
        # 检查引擎是否已初始化
        if model_manager.current_engine:
            print(f"引擎已初始化: {type(model_manager.current_engine).__name__}")
        else:
            print("引擎未初始化")
    else:
        print("加载 sherpa_0626 模型失败")

    # 测试创建识别器
    print("\n测试创建识别器...")
    recognizer = model_manager.create_recognizer()
    if recognizer:
        print(f"成功创建识别器: {type(recognizer).__name__}")
        
        # 检查识别器的引擎类型
        if hasattr(recognizer, 'engine_type'):
            print(f"识别器引擎类型: {recognizer.engine_type}")
        else:
            print("识别器没有引擎类型属性")
    else:
        print("创建识别器失败")

if __name__ == "__main__":
    test_sherpa_0626_model_selection()
