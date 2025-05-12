"""
测试 Sherpa-ONNX 文件转录
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
from src.core.signals import TranscriptionSignals
from src.core.audio.file_transcriber import FileTranscriber

def test_sherpa_onnx_file_transcription():
    """测试 Sherpa-ONNX 文件转录"""
    print("测试 Sherpa-ONNX 文件转录...")
    
    # 创建信号
    signals = TranscriptionSignals()
    
    # 连接信号
    signals.new_text.connect(lambda text: print(f"转录文本: {text}"))
    signals.status_updated.connect(lambda status: print(f"状态更新: {status}"))
    signals.progress_updated.connect(lambda progress, text: print(f"进度更新: {progress}%, {text}"))
    signals.error_occurred.connect(lambda error: print(f"错误: {error}"))
    signals.transcription_finished.connect(lambda: print("转录完成"))
    
    # 创建模型管理器
    model_manager = ASRModelManager()
    
    # 加载 Sherpa-ONNX 模型
    model_name = "sherpa_int8"  # 或 "sherpa_std"
    if not model_manager.load_model(model_name):
        print(f"加载 {model_name} 模型失败")
        return
    
    # 创建文件转录器
    file_transcriber = FileTranscriber(signals)
    
    # 选择测试文件
    test_file = input("请输入要转录的音频/视频文件路径: ")
    if not os.path.exists(test_file):
        print(f"文件不存在: {test_file}")
        return
    
    # 开始转录
    if not file_transcriber.start_transcription(test_file, model_manager):
        print("开始转录失败")
        return
    
    # 等待转录完成
    try:
        while file_transcriber.is_transcribing:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("用户中断转录")
        file_transcriber.stop_transcription()
    
    print("测试完成")

if __name__ == "__main__":
    test_sherpa_onnx_file_transcription()
