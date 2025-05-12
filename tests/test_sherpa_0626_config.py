"""
测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型配置
"""
import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 设置当前工作目录为项目根目录，确保能正确找到配置文件
os.chdir(project_root)

# 导入必要的模块
from src.utils.config_manager import config_manager
from src.utils.sherpa_logger import sherpa_logger

def test_sherpa_0626_config():
    """测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型配置"""
    print("=" * 80)
    print("测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型配置")
    print("=" * 80)

    # 初始化日志工具
    sherpa_logger.setup()
    sherpa_log_file = sherpa_logger.get_log_file()
    print(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")

    # 获取配置
    # 直接访问config_manager.config属性，这是完整的配置字典
    config = config_manager.config

    # 检查默认模型
    default_model = config.get("transcription", {}).get("default_model", "")
    print(f"默认模型: {default_model}")

    # 检查 sherpa_0626 模型配置
    sherpa_0626_config = config.get("asr", {}).get("models", {}).get("sherpa_0626", {})
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

    # 检查模型配置
    model_config = sherpa_0626_config.get("config", {})
    print(f"模型配置: {model_config}")

    # 检查模型文件
    encoder_file = model_config.get("encoder", "")
    decoder_file = model_config.get("decoder", "")
    joiner_file = model_config.get("joiner", "")
    tokens_file = model_config.get("tokens", "")

    print(f"模型文件:")
    print(f"  encoder: {encoder_file}")
    print(f"  decoder: {decoder_file}")
    print(f"  joiner: {joiner_file}")
    print(f"  tokens: {tokens_file}")

    # 检查模型文件是否存在
    encoder_path = os.path.join(model_path, encoder_file)
    decoder_path = os.path.join(model_path, decoder_file)
    joiner_path = os.path.join(model_path, joiner_file)
    tokens_path = os.path.join(model_path, tokens_file)

    print(f"检查模型文件是否存在:")
    print(f"  encoder: {os.path.exists(encoder_path)}")
    print(f"  decoder: {os.path.exists(decoder_path)}")
    print(f"  joiner: {os.path.exists(joiner_path)}")
    print(f"  tokens: {os.path.exists(tokens_path)}")

if __name__ == "__main__":
    test_sherpa_0626_config()
