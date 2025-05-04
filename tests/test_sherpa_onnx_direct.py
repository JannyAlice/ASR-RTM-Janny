"""
直接使用 sherpa-onnx 的 API 进行测试
"""
import os
import sys
import numpy as np
import wave
import time
from typing import Tuple

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入 sherpa_onnx 模块
try:
    import sherpa_onnx
    HAS_SHERPA_ONNX = True
except ImportError:
    HAS_SHERPA_ONNX = False
    print("未安装 sherpa_onnx 模块，无法测试 Sherpa-ONNX 模型")
    sys.exit(1)

def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    """
    读取 wave 文件

    Args:
        wave_filename: wave 文件路径

    Returns:
        Tuple[np.ndarray, int]: 音频数据和采样率
    """
    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)
        samples_float32 = samples_float32 / 32768
        return samples_float32, f.getframerate()

def test_sherpa_onnx_direct():
    """直接使用 sherpa-onnx 的 API 进行测试"""
    print("开始直接测试 sherpa-onnx API...")
    
    # 模型路径
    model_path = "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\models\\asr\\sherpa-onnx"
    
    # 确定模型文件名
    is_int8 = True
    encoder_file = "encoder-epoch-99-avg-1.int8.onnx" if is_int8 else "encoder-epoch-99-avg-1.onnx"
    decoder_file = "decoder-epoch-99-avg-1.int8.onnx" if is_int8 else "decoder-epoch-99-avg-1.onnx"
    joiner_file = "joiner-epoch-99-avg-1.int8.onnx" if is_int8 else "joiner-epoch-99-avg-1.onnx"
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
            print(f"模型文件不存在: {file_path}")
            return
    
    # 创建 OnlineRecognizer
    try:
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=os.path.join(model_path, encoder_file),
            decoder=os.path.join(model_path, decoder_file),
            joiner=os.path.join(model_path, joiner_file),
            tokens=os.path.join(model_path, tokens_file),
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search"
        )
        print("成功创建 OnlineRecognizer")
    except Exception as e:
        print(f"创建 OnlineRecognizer 失败: {e}")
        import traceback
        print(traceback.format_exc())
        return
    
    # 尝试找一个测试音频文件
    test_audio = None
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith(".wav"):
                test_audio = os.path.join(root, file)
                break
        if test_audio:
            break
    
    if test_audio:
        print(f"找到测试音频文件: {test_audio}")
        
        # 读取音频文件
        try:
            samples, sample_rate = read_wave(test_audio)
            
            # 创建流
            stream = recognizer.create_stream()
            
            # 处理音频数据
            print("处理音频数据...")
            
            # 分块处理音频数据，模拟实时处理
            chunk_size = 1600  # 100ms
            for i in range(0, len(samples), chunk_size):
                chunk = samples[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
                
                # 处理这个块
                stream.accept_waveform(sample_rate, chunk)
                recognizer.decode_stream(stream)
                
                # 获取部分结果
                text = recognizer.get_result(stream)
                if text:
                    print(f"部分结果: {text}")
                
                # 模拟实时处理的延迟
                time.sleep(0.1)
            
            # 添加尾部填充
            tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
            stream.accept_waveform(sample_rate, tail_paddings)
            
            # 标记输入结束
            stream.input_finished()
            
            # 解码
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            
            # 获取最终结果
            final_text = recognizer.get_result(stream)
            print(f"最终结果: {final_text}")
            
        except Exception as e:
            print(f"处理音频文件错误: {e}")
            import traceback
            print(traceback.format_exc())
    else:
        print("未找到测试音频文件")
    
    # 生成测试音频数据（包含一些随机噪声，模拟语音）
    print("\n生成测试音频数据...")
    sample_rate = 16000
    duration = 2  # 2秒
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # 生成一个简单的正弦波，模拟语音
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz 正弦波
    
    # 创建流
    stream = recognizer.create_stream()
    
    # 处理音频数据
    print("处理音频数据...")
    
    # 分块处理音频数据，模拟实时处理
    chunk_size = 1600  # 100ms
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
        
        # 处理这个块
        stream.accept_waveform(sample_rate, chunk)
        recognizer.decode_stream(stream)
        
        # 获取部分结果
        text = recognizer.get_result(stream)
        if text:
            print(f"部分结果: {text}")
        
        # 模拟实时处理的延迟
        time.sleep(0.1)
    
    # 添加尾部填充
    tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
    stream.accept_waveform(sample_rate, tail_paddings)
    
    # 标记输入结束
    stream.input_finished()
    
    # 解码
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    
    # 获取最终结果
    final_text = recognizer.get_result(stream)
    print(f"最终结果: {final_text}")
    
    print("测试完成")

if __name__ == "__main__":
    test_sherpa_onnx_direct()
