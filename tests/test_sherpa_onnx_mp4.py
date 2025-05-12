"""
测试 Sherpa-ONNX 模型处理 MP4 文件
"""
import os
import sys
import json
import numpy as np
import subprocess
import tempfile
from pathlib import Path
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

def extract_audio_from_mp4(mp4_file: str) -> str:
    """
    从 MP4 文件中提取音频

    Args:
        mp4_file: MP4 文件路径

    Returns:
        str: 提取的音频文件路径
    """
    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_wav = temp_file.name
    
    # 使用 ffmpeg 提取音频
    cmd = [
        'ffmpeg',
        '-i', mp4_file,
        '-vn',  # 不要视频
        '-acodec', 'pcm_s16le',  # 16位 PCM
        '-ar', '16000',  # 16kHz 采样率
        '-ac', '1',  # 单声道
        '-y',  # 覆盖输出文件
        temp_wav
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"成功从 {mp4_file} 提取音频到 {temp_wav}")
        return temp_wav
    except subprocess.CalledProcessError as e:
        print(f"提取音频失败: {e}")
        print(f"错误输出: {e.stderr.decode('utf-8', errors='ignore')}")
        return None

def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    """
    读取 wave 文件

    Args:
        wave_filename: wave 文件路径

    Returns:
        Tuple[np.ndarray, int]: 音频数据和采样率
    """
    import wave
    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)
        samples_float32 = samples_float32 / 32768
        return samples_float32, f.getframerate()

def test_mp4_transcription():
    """测试 MP4 文件转录"""
    print("测试 MP4 文件转录...")
    
    # MP4 文件路径
    mp4_file = "C:\\Users\\crige\\RealtimeTrans\\vosk-api-bak-J\\mytest.mp4"
    
    if not os.path.exists(mp4_file):
        print(f"MP4 文件不存在: {mp4_file}")
        return
    
    # 提取音频
    wav_file = extract_audio_from_mp4(mp4_file)
    if not wav_file:
        print("提取音频失败")
        return
    
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
        if not Path(file_path).is_file():
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
            decoding_method="greedy_search",
            provider="cpu",
        )
        print("成功创建 OnlineRecognizer")
    except Exception as e:
        print(f"创建 OnlineRecognizer 失败: {e}")
        import traceback
        print(traceback.format_exc())
        return
    
    # 读取音频文件
    try:
        samples, sample_rate = read_wave(wav_file)
        print(f"音频长度: {len(samples) / sample_rate:.2f} 秒")
        
        # 创建流
        s = recognizer.create_stream()
        
        # 处理音频数据
        print("处理音频数据...")
        s.accept_waveform(sample_rate, samples)
        
        # 添加尾部填充
        tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
        s.accept_waveform(sample_rate, tail_paddings)
        
        # 标记输入结束
        s.input_finished()
        
        # 解码
        while recognizer.is_ready(s):
            recognizer.decode_stream(s)
        
        # 获取结果
        result = recognizer.get_result(s)
        print(f"识别结果: {result}")
        
        # 保存结果到文件
        result_file = "mp4_transcription_result.txt"
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"结果已保存到 {result_file}")
        
        # 清理临时文件
        try:
            os.remove(wav_file)
            print(f"已删除临时文件 {wav_file}")
        except Exception as e:
            print(f"删除临时文件失败: {e}")
        
    except Exception as e:
        print(f"处理音频文件错误: {e}")
        import traceback
        print(traceback.format_exc())

def test_mp4_realtime_simulation():
    """模拟实时转录 MP4 文件"""
    print("\n模拟实时转录 MP4 文件...")
    
    # MP4 文件路径
    mp4_file = "C:\\Users\\crige\\RealtimeTrans\\vosk-api-bak-J\\mytest.mp4"
    
    if not os.path.exists(mp4_file):
        print(f"MP4 文件不存在: {mp4_file}")
        return
    
    # 提取音频
    wav_file = extract_audio_from_mp4(mp4_file)
    if not wav_file:
        print("提取音频失败")
        return
    
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
        if not Path(file_path).is_file():
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
            decoding_method="greedy_search",
            provider="cpu",
        )
        print("成功创建 OnlineRecognizer")
    except Exception as e:
        print(f"创建 OnlineRecognizer 失败: {e}")
        import traceback
        print(traceback.format_exc())
        return
    
    # 读取音频文件
    try:
        samples, sample_rate = read_wave(wav_file)
        print(f"音频长度: {len(samples) / sample_rate:.2f} 秒")
        
        # 创建流
        s = recognizer.create_stream()
        
        # 分块处理音频数据，模拟实时处理
        chunk_size = int(0.5 * sample_rate)  # 500ms 的数据
        print(f"分块大小: {chunk_size / sample_rate:.2f} 秒")
        
        all_results = []
        
        for i in range(0, len(samples), chunk_size):
            chunk = samples[i:i+chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
            
            # 处理这个块
            s.accept_waveform(sample_rate, chunk)
            recognizer.decode_stream(s)
            
            # 获取部分结果
            partial_result = recognizer.get_result(s)
            print(f"部分结果 {i//chunk_size + 1}: {partial_result}")
            all_results.append(partial_result)
        
        # 添加尾部填充
        tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
        s.accept_waveform(sample_rate, tail_paddings)
        
        # 标记输入结束
        s.input_finished()
        
        # 解码
        while recognizer.is_ready(s):
            recognizer.decode_stream(s)
        
        # 获取最终结果
        final_result = recognizer.get_result(s)
        print(f"最终结果: {final_result}")
        
        # 保存结果到文件
        result_file = "mp4_realtime_transcription_result.txt"
        with open(result_file, "w", encoding="utf-8") as f:
            f.write("部分结果:\n")
            for i, result in enumerate(all_results):
                f.write(f"{i+1}: {result}\n")
            f.write("\n最终结果:\n")
            f.write(final_result)
        print(f"结果已保存到 {result_file}")
        
        # 清理临时文件
        try:
            os.remove(wav_file)
            print(f"已删除临时文件 {wav_file}")
        except Exception as e:
            print(f"删除临时文件失败: {e}")
        
    except Exception as e:
        print(f"处理音频文件错误: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_mp4_transcription()
    test_mp4_realtime_simulation()
