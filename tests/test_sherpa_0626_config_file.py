"""
测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型配置文件
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

def test_sherpa_0626_config_file():
    """测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型配置文件"""
    print("=" * 80)
    print("测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型配置文件")
    print("=" * 80)

    # 初始化日志工具
    sherpa_logger.setup()
    sherpa_log_file = sherpa_logger.get_log_file()
    print(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")

    # 创建模型管理器
    model_manager = ASRModelManager()

    # 获取模型配置
    models_config = model_manager.models_config
    sherpa_0626_config = models_config.get("sherpa_0626", {})
    print(f"sherpa_0626 模型配置: {sherpa_0626_config}")

    # 检查模型路径
    model_path = sherpa_0626_config.get("path", "")
    print(f"模型路径: {model_path}")

    # 检查模型路径是否存在
    if os.path.exists(model_path):
        print(f"模型路径存在: {model_path}")
        
        # 列出目录中的文件
        print("目录中的文件:")
        for file in os.listdir(model_path):
            if file.endswith(".onnx"):
                print(f"  {file}")
    else:
        print(f"模型路径不存在: {model_path}")

    # 加载模型
    print("加载 sherpa_0626 模型...")
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

if __name__ == "__main__":
    test_sherpa_0626_config_file()
