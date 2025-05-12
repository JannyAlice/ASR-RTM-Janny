"""
测试 Sherpa-ONNX 文件转录（带日志）- 修复版
"""
import os
import sys
import time
import numpy as np

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 设置当前工作目录为项目根目录，确保能正确找到配置文件
os.chdir(project_root)

# 导入 sherpa_onnx 模块
try:
    import sherpa_onnx
    HAS_SHERPA_ONNX = True
except ImportError:
    HAS_SHERPA_ONNX = False
    print("未安装 sherpa_onnx 模块，无法测试 Sherpa-ONNX 模型")
    sys.exit(1)

# 导入 Sherpa-ONNX 日志工具
from src.utils.sherpa_logger import sherpa_logger

def test_sherpa_onnx_file():
    """测试 Sherpa-ONNX 文件转录"""
    print("测试 Sherpa-ONNX 文件转录...")
    
    # 初始化日志工具
    sherpa_logger.setup()
    sherpa_log_file = sherpa_logger.get_log_file()
    print(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")
    
    # 模型路径
    model_path = "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\models\\asr\\sherpa-onnx"
    
    # 确定模型文件名
    use_int8 = True
    encoder_file = "encoder-epoch-99-avg-1.int8.onnx" if use_int8 else "encoder-epoch-99-avg-1.onnx"
    decoder_file = "decoder-epoch-99-avg-1.int8.onnx" if use_int8 else "decoder-epoch-99-avg-1.onnx"
    joiner_file = "joiner-epoch-99-avg-1.int8.onnx" if use_int8 else "joiner-epoch-99-avg-1.onnx"
    tokens_file = "tokens.txt"
    
    # 检查模型文件是否存在
    required_files = [
        os.path.join(model_path, encoder_file),
        os.path.join(model_path, decoder_file),
        os.path.join(model_path, joiner_file),
        os.path.join(model_path, tokens_file)
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            sherpa_logger.error(f"模型文件不存在: {file_path}")
            return
    
    # 创建 OnlineRecognizer
    try:
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=os.path.join(model_path, encoder_file),
            decoder=os.path.join(model_path, decoder_file),
            joiner=os.path.join(model_path, joiner_file),
            tokens=os.path.join(model_path, tokens_file),
            num_threads=4,
            decoding_method="greedy_search",
            provider="cpu",
        )
        sherpa_logger.info("成功创建 OnlineRecognizer")
    except Exception as e:
        sherpa_logger.error(f"创建 OnlineRecognizer 失败: {e}")
        import traceback
        sherpa_logger.error(traceback.format_exc())
        return
    
    # 测试文件路径
    test_file = r"C:\Users\crige\RealtimeTrans\vosk-api-bak-J\mytest.mp4"
    sherpa_logger.info(f"使用测试文件: {test_file}")
    if not os.path.exists(test_file):
        sherpa_logger.error(f"文件不存在: {test_file}")
        return
    
    # 使用 ffmpeg 将文件转换为 16kHz 单声道 PCM 格式
    import subprocess
    import tempfile
    import wave
    
    # 创建临时 WAV 文件
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        temp_wav_path = temp_wav.name
    
    # 使用 ffmpeg 转换
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', test_file,
        '-ar', '16000',  # 采样率 16kHz
        '-ac', '1',      # 单声道
        '-f', 'wav',     # WAV 格式
        '-y',            # 覆盖已有文件
        temp_wav_path
    ]
    
    sherpa_logger.info(f"执行命令: {' '.join(ffmpeg_cmd)}")
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 读取 WAV 文件
    with wave.open(temp_wav_path, 'rb') as wav_file:
        # 检查格式
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2 or wav_file.getframerate() != 16000:
            sherpa_logger.error(f"WAV 文件格式不正确: 通道数={wav_file.getnchannels()}, 采样宽度={wav_file.getsampwidth()}, 采样率={wav_file.getframerate()}")
            return
        
        # 读取所有帧
        frames = wav_file.readframes(wav_file.getnframes())
        
        # 转换为 numpy 数组
        audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    
    # 删除临时文件
    try:
        os.unlink(temp_wav_path)
    except:
        pass
    
    # 记录音频数据信息
    sherpa_logger.info(f"音频数据长度: {len(audio_data)} 样本，最大值: {np.max(np.abs(audio_data))}")
    
    # 创建流
    stream = recognizer.create_stream()
    
    # 处理音频数据
    # 分块处理，每次处理 10 秒的数据
    chunk_size = 16000 * 10  # 10 秒的数据
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        
        # 处理这个块
        stream.accept_waveform(16000, chunk)
        
        # 解码
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        
        # 获取部分结果
        partial_result = recognizer.get_result(stream)
        if partial_result:
            # 过滤掉非英文字符
            import re
            filtered_result = re.sub(r'[^\x00-\x7F]+', '', partial_result)
            if filtered_result.strip():
                sherpa_logger.info(f"部分结果: {filtered_result}")
                print(f"部分结果: {filtered_result}")
    
    # 添加尾部填充
    tail_paddings = np.zeros(int(0.2 * 16000), dtype=np.float32)
    stream.accept_waveform(16000, tail_paddings)
    
    # 标记输入结束
    stream.input_finished()
    
    # 解码
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    
    # 获取最终结果
    result = recognizer.get_result(stream)
    
    # 过滤掉非英文字符
    if result:
        import re
        filtered_result = re.sub(r'[^\x00-\x7F]+', '', result)
        sherpa_logger.info(f"最终结果: {filtered_result}")
        print(f"最终结果: {filtered_result}")
    else:
        sherpa_logger.info("没有结果")
        print("没有结果")
    
    sherpa_logger.info(f"测试完成，日志文件: {sherpa_log_file}")
    print(f"测试完成，日志文件: {sherpa_log_file}")

if __name__ == "__main__":
    test_sherpa_onnx_file()
