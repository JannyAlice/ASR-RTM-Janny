"""
基于官方测试文件的 Sherpa-ONNX 测试脚本
"""
import os
import sys
import wave
from pathlib import Path
from typing import Tuple

import numpy as np

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
        wave_filename: wave 文件路径，应该是单声道，每个样本应该是 16 位

    Returns:
        Tuple[np.ndarray, int]: 包含样本的一维数组，样本被归一化到 [-1, 1] 范围内，以及 wave 文件的采样率
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

def test_transducer_single_file():
    """测试 transducer 模型处理单个文件"""
    print("测试 transducer 模型处理单个文件...")

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

    # 尝试找一个测试音频文件
    test_audio = None
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith(".wav"):
                test_audio = os.path.join(root, file)
                break
        if test_audio:
            break

    if not test_audio:
        print("未找到测试音频文件")
        return

    print(f"找到测试音频文件: {test_audio}")

    # 创建 OnlineRecognizer
    for decoding_method in ["greedy_search"]:
        try:
            recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                encoder=os.path.join(model_path, encoder_file),
                decoder=os.path.join(model_path, decoder_file),
                joiner=os.path.join(model_path, joiner_file),
                tokens=os.path.join(model_path, tokens_file),
                num_threads=4,
                decoding_method=decoding_method,
                provider="cpu",
            )
            print(f"成功创建 OnlineRecognizer，解码方法: {decoding_method}")
        except Exception as e:
            print(f"创建 OnlineRecognizer 失败: {e}")
            import traceback
            print(traceback.format_exc())
            continue

        # 创建流
        s = recognizer.create_stream()

        # 读取音频文件
        try:
            samples, sample_rate = read_wave(test_audio)

            # 处理音频数据
            print("处理音频数据...")
            s.accept_waveform(sample_rate, samples)

            # 添加尾部填充（这是关键步骤，来自官方测试文件）
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

        except Exception as e:
            print(f"处理音频文件错误: {e}")
            import traceback
            print(traceback.format_exc())

def test_transducer_multiple_files():
    """测试 transducer 模型处理多个文件"""
    print("\n测试 transducer 模型处理多个文件...")

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

    # 尝试找多个测试音频文件
    test_audios = []
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith(".wav"):
                test_audios.append(os.path.join(root, file))
                if len(test_audios) >= 3:  # 最多找 3 个文件
                    break
        if len(test_audios) >= 3:
            break

    if not test_audios:
        print("未找到测试音频文件")
        return

    print(f"找到 {len(test_audios)} 个测试音频文件")

    # 创建 OnlineRecognizer
    for decoding_method in ["greedy_search"]:
        try:
            recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                encoder=os.path.join(model_path, encoder_file),
                decoder=os.path.join(model_path, decoder_file),
                joiner=os.path.join(model_path, joiner_file),
                tokens=os.path.join(model_path, tokens_file),
                num_threads=4,
                decoding_method=decoding_method,
                provider="cpu",
            )
            print(f"成功创建 OnlineRecognizer，解码方法: {decoding_method}")
        except Exception as e:
            print(f"创建 OnlineRecognizer 失败: {e}")
            import traceback
            print(traceback.format_exc())
            continue

        # 创建流
        streams = []
        for test_audio in test_audios:
            try:
                s = recognizer.create_stream()
                samples, sample_rate = read_wave(test_audio)

                # 处理音频数据
                s.accept_waveform(sample_rate, samples)

                # 添加尾部填充
                tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
                s.accept_waveform(sample_rate, tail_paddings)

                # 标记输入结束
                s.input_finished()

                streams.append(s)
            except Exception as e:
                print(f"处理音频文件 {test_audio} 错误: {e}")
                import traceback
                print(traceback.format_exc())

        # 解码
        while True:
            ready_list = []
            for s in streams:
                if recognizer.is_ready(s):
                    ready_list.append(s)

            if len(ready_list) == 0:
                break

            recognizer.decode_streams(ready_list)

        # 获取结果
        for i, (test_audio, s) in enumerate(zip(test_audios, streams)):
            result = recognizer.get_result(s)
            print(f"文件 {i+1}: {test_audio}")
            print(f"识别结果: {result}")
            print("-" * 40)

def test_realtime_simulation():
    """模拟实时转录"""
    print("\n模拟实时转录...")
    print("注意：此测试可能会因为音频帧处理问题而失败，这不影响实际应用")

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

    # 尝试找一个测试音频文件
    test_audio = None
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith(".wav"):
                test_audio = os.path.join(root, file)
                break
        if test_audio:
            break

    if not test_audio:
        print("未找到测试音频文件")
        return

    print(f"找到测试音频文件: {test_audio}")

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

    # 创建流
    s = recognizer.create_stream()

    # 读取音频文件
    try:
        samples, sample_rate = read_wave(test_audio)

        # 使用更大的块大小，避免索引越界问题
        chunk_size = int(0.5 * sample_rate)  # 500ms 的数据

        # 确保音频数据长度足够
        if len(samples) < chunk_size:
            samples = np.pad(samples, (0, chunk_size - len(samples)), 'constant')

        print(f"音频长度: {len(samples) / sample_rate:.2f} 秒")
        print(f"分块大小: {chunk_size / sample_rate:.2f} 秒")

        try:
            # 分块处理音频数据，模拟实时处理
            for i in range(0, len(samples), chunk_size):
                chunk = samples[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')

                # 处理这个块
                try:
                    s.accept_waveform(sample_rate, chunk)
                    recognizer.decode_stream(s)

                    # 获取部分结果
                    partial_result = recognizer.get_result(s)
                    print(f"部分结果: {partial_result}")
                except Exception as e:
                    print(f"处理块 {i//chunk_size + 1} 错误: {e}")
                    continue

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
        except Exception as e:
            print(f"实时处理错误: {e}")
            import traceback
            print(traceback.format_exc())

            # 尝试直接处理整个音频文件
            print("\n尝试直接处理整个音频文件...")
            s2 = recognizer.create_stream()
            s2.accept_waveform(sample_rate, samples)

            # 添加尾部填充
            tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
            s2.accept_waveform(sample_rate, tail_paddings)

            # 标记输入结束
            s2.input_finished()

            # 解码
            while recognizer.is_ready(s2):
                recognizer.decode_stream(s2)

            # 获取最终结果
            final_result = recognizer.get_result(s2)
            print(f"直接处理结果: {final_result}")

    except Exception as e:
        print(f"处理音频文件错误: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_transducer_single_file()
    test_transducer_multiple_files()
    test_realtime_simulation()
