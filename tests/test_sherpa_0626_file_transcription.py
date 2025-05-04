"""
测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的文件转录功能
"""
import os
import sys
import time
import json
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 设置当前工作目录为项目根目录，确保能正确找到配置文件
os.chdir(project_root)

# 导入 Sherpa-ONNX 日志工具
from src.utils.sherpa_logger import sherpa_logger

# 导入模型管理器
from src.core.asr.model_manager import ASRModelManager

def test_sherpa_0626_file_transcription(file_path=None):
    """测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的文件转录功能"""
    print("=" * 80)
    print("测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的文件转录功能")
    print("=" * 80)
    
    # 设置日志
    sherpa_logger.setup()
    sherpa_log_file = sherpa_logger.get_log_file()
    print(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")
    
    # 创建模型管理器
    print("创建模型管理器...")
    model_manager = ASRModelManager()
    
    # 加载 sherpa_0626 模型
    print("加载 sherpa_0626 模型...")
    if not model_manager.load_model("sherpa_0626"):
        print("加载 sherpa_0626 模型失败")
        return
    
    # 如果没有提供文件路径，则使用默认的测试文件
    if not file_path:
        # 查找测试文件
        test_files_dir = os.path.join(project_root, "tests", "test_files")
        if not os.path.exists(test_files_dir):
            os.makedirs(test_files_dir, exist_ok=True)
        
        # 默认测试文件
        default_test_file = os.path.join(test_files_dir, "test_audio.mp3")
        
        # 如果默认测试文件不存在，提示用户提供文件路径
        if not os.path.exists(default_test_file):
            print(f"默认测试文件不存在: {default_test_file}")
            file_path = input("请输入要转录的音频/视频文件路径: ")
        else:
            file_path = default_test_file
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    print(f"转录文件: {file_path}")
    
    # 开始转录
    print("开始转录...")
    try:
        # 转录文件
        result = model_manager.transcribe_file(file_path)
        
        if result:
            print("\n转录结果:")
            print(result)
            
            # 创建保存目录
            save_dir = os.path.join(project_root, "transcripts")
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            txt_path = os.path.join(save_dir, f"file_test_{base_name}_{timestamp}.txt")
            
            # 保存转录结果
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"文件转录测试 - {timestamp}\n")
                f.write(f"文件: {file_path}\n")
                f.write("=" * 50 + "\n\n")
                f.write(result)
            
            print(f"\n转录结果已保存到: {txt_path}")
        else:
            print("转录失败，没有结果")
    
    except Exception as e:
        print(f"转录错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("测试完成")

if __name__ == "__main__":
    # 获取命令行参数
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        test_sherpa_0626_file_transcription(file_path)
    else:
        test_sherpa_0626_file_transcription()
