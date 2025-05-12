import sys
import os
import sherpa_onnx
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.sherpa_logger import sherpa_logger

# 初始化日志工具
log_dir = os.path.join(project_root, "logs")
sherpa_logger.setup(log_dir=log_dir)

# 示例配置
model_dir = os.path.join(project_root, "models", "asr", "sherpa-onnx-streaming-zipformer-en-2023-06-26")
model_files = {
    "encoder": os.path.join(model_dir, "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
    "decoder": os.path.join(model_dir, "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
    "joiner": os.path.join(model_dir, "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
    "tokens": os.path.join(model_dir, "tokens.txt")
}

def check_model_metadata(model_files):
    sherpa_logger.info("\n=== 开始检查模型文件 ===")
    sherpa_logger.info(f"模型目录: {model_dir}")
    
    for file_type, file_path in model_files.items():
        sherpa_logger.info(f"\n检查{file_type}模型文件: {os.path.basename(file_path)}")
        
        if not os.path.exists(file_path):
            error_msg = f"模型文件不存在: {file_path}"
            sherpa_logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if not os.access(file_path, os.R_OK):
            error_msg = f"无法读取模型文件: {file_path}"
            sherpa_logger.error(error_msg)
            raise PermissionError(error_msg)
        
        if file_path.endswith(".onnx"):
            try:
                # 使用sherpa_onnx检查模型文件
                sherpa_logger.info(f"✔ 模型文件存在且可读")
                sherpa_logger.info(f"✔ 文件大小: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
                
                # 注意：sherpa_onnx会在初始化时自动检查模型文件
                # 如果模型文件有问题，将在初始化时抛出异常
            except Exception as e:
                error_msg = f"无法加载或检查模型文件 {file_path}: {e}"
                sherpa_logger.error(error_msg)
                raise RuntimeError(error_msg)
        else:
            sherpa_logger.info(f"✔ 文件存在且可读")
            sherpa_logger.info(f"✔ 文件大小: {os.path.getsize(file_path) / 1024:.2f} KB")
    
    sherpa_logger.info("\n✅ 所有模型文件检查完成\n")

if __name__ == "__main__":
    check_model_metadata(model_files)