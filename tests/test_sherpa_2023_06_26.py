#!/usr/bin/env python3
"""
测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的识别准确率

此测试程序用于验证新模型的识别准确率，不影响现有的vosk和sherpa-onnx 2023-02-20模型。
参考了项目中现有的sherpa-onnx引擎实现和官方测试文件。
"""

import os
import sys
import wave
import time
import json
import numpy as np
from typing import Tuple, Dict, Any, Optional, List

# 确保能够导入项目模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import sherpa_onnx
    HAS_SHERPA_ONNX = True
except ImportError:
    HAS_SHERPA_ONNX = False
    print("警告: 未安装 sherpa_onnx 模块，请先安装")
    sys.exit(1)

# 模型路径
MODEL_2023_02_20_PATH = r"C:\Users\crige\RealtimeTrans\vosk-api\models\asr\sherpa-onnx"
MODEL_2023_06_26_PATH = r"C:\Users\crige\RealtimeTrans\vosk-api\models\asr\sherpa-onnx-streaming-zipformer-en-2023-06-26"

# 测试音频路径
TEST_WAVS_PATH = os.path.join(MODEL_2023_06_26_PATH, "test_wavs")


def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    """
    读取WAV文件并返回采样数据和采样率
    
    Args:
        wave_filename: WAV文件路径
        
    Returns:
        Tuple[np.ndarray, int]: 采样数据和采样率
    """
    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # 2 bytes = 16 bits
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)
        
        # 归一化到 [-1, 1] 范围
        samples_float32 = samples_float32 / 32768
        return samples_float32, f.getframerate()


def load_model_2023_06_26(use_int8: bool = True) -> Optional[sherpa_onnx.OnlineRecognizer]:
    """
    加载 2023-06-26 模型
    
    Args:
        use_int8: 是否使用int8量化模型
        
    Returns:
        sherpa_onnx.OnlineRecognizer: 模型实例
    """
    try:
        print(f"加载 2023-06-26 模型 (use_int8={use_int8})...")
        
        # 确定模型文件名
        if use_int8:
            encoder = os.path.join(MODEL_2023_06_26_PATH, "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx")
            decoder = os.path.join(MODEL_2023_06_26_PATH, "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx")
            joiner = os.path.join(MODEL_2023_06_26_PATH, "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx")
        else:
            encoder = os.path.join(MODEL_2023_06_26_PATH, "encoder-epoch-99-avg-1-chunk-16-left-128.onnx")
            decoder = os.path.join(MODEL_2023_06_26_PATH, "decoder-epoch-99-avg-1-chunk-16-left-128.onnx")
            joiner = os.path.join(MODEL_2023_06_26_PATH, "joiner-epoch-99-avg-1-chunk-16-left-128.onnx")
        
        tokens = os.path.join(MODEL_2023_06_26_PATH, "tokens.txt")
        
        # 检查文件是否存在
        for file_path in [encoder, decoder, joiner, tokens]:
            if not os.path.exists(file_path):
                print(f"错误: 文件不存在: {file_path}")
                return None
        
        # 创建模型实例
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search"
        )
        
        print(f"成功加载 2023-06-26 模型 ({'int8量化' if use_int8 else '标准'})")
        return recognizer
    
    except Exception as e:
        print(f"加载 2023-06-26 模型失败: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def load_model_2023_02_20(use_int8: bool = True) -> Optional[sherpa_onnx.OnlineRecognizer]:
    """
    加载 2023-02-20 模型（用于比较）
    
    Args:
        use_int8: 是否使用int8量化模型
        
    Returns:
        sherpa_onnx.OnlineRecognizer: 模型实例
    """
    try:
        print(f"加载 2023-02-20 模型 (use_int8={use_int8})...")
        
        # 确定模型文件名
        if use_int8:
            encoder = os.path.join(MODEL_2023_02_20_PATH, "encoder-epoch-99-avg-1.int8.onnx")
            decoder = os.path.join(MODEL_2023_02_20_PATH, "decoder-epoch-99-avg-1.onnx")
            joiner = os.path.join(MODEL_2023_02_20_PATH, "joiner-epoch-99-avg-1.int8.onnx")
        else:
            encoder = os.path.join(MODEL_2023_02_20_PATH, "encoder-epoch-99-avg-1.onnx")
            decoder = os.path.join(MODEL_2023_02_20_PATH, "decoder-epoch-99-avg-1.onnx")
            joiner = os.path.join(MODEL_2023_02_20_PATH, "joiner-epoch-99-avg-1.onnx")
        
        tokens = os.path.join(MODEL_2023_02_20_PATH, "tokens.txt")
        
        # 检查文件是否存在
        for file_path in [encoder, decoder, joiner, tokens]:
            if not os.path.exists(file_path):
                print(f"错误: 文件不存在: {file_path}")
                return None
        
        # 创建模型实例
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search"
        )
        
        print(f"成功加载 2023-02-20 模型 ({'int8量化' if use_int8 else '标准'})")
        return recognizer
    
    except Exception as e:
        print(f"加载 2023-02-20 模型失败: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def transcribe_audio(recognizer: sherpa_onnx.OnlineRecognizer, audio_file: str) -> str:
    """
    使用给定的模型转录音频文件
    
    Args:
        recognizer: 模型实例
        audio_file: 音频文件路径
        
    Returns:
        str: 转录结果
    """
    try:
        print(f"转录音频文件: {audio_file}")
        
        # 创建流
        stream = recognizer.create_stream()
        
        # 读取音频数据
        samples, sample_rate = read_wave(audio_file)
        
        # 处理音频数据
        start_time = time.time()
        
        # 接受音频数据
        stream.accept_waveform(sample_rate, samples)
        
        # 添加尾部填充（这是关键步骤，来自官方测试文件）
        tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
        stream.accept_waveform(sample_rate, tail_paddings)
        
        # 标记输入结束
        stream.input_finished()
        
        # 解码
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        
        # 获取结果
        result = recognizer.get_result(stream)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"转录完成，耗时: {elapsed:.2f}秒")
        return result
    
    except Exception as e:
        print(f"转录音频文件失败: {e}")
        import traceback
        print(traceback.format_exc())
        return "转录失败"


def main():
    """主函数"""
    print("=" * 80)
    print("测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的识别准确率")
    print("=" * 80)
    
    # 检查测试音频文件
    test_wavs = []
    for file in os.listdir(TEST_WAVS_PATH):
        if file.endswith(".wav"):
            test_wavs.append(os.path.join(TEST_WAVS_PATH, file))
    
    if not test_wavs:
        print("错误: 未找到测试音频文件")
        return
    
    print(f"找到 {len(test_wavs)} 个测试音频文件")
    
    # 加载模型
    model_2023_06_26_int8 = load_model_2023_06_26(use_int8=True)
    model_2023_06_26_std = load_model_2023_06_26(use_int8=False)
    model_2023_02_20_int8 = load_model_2023_02_20(use_int8=True)
    model_2023_02_20_std = load_model_2023_02_20(use_int8=False)
    
    # 测试每个音频文件
    results = {}
    
    for audio_file in test_wavs:
        file_name = os.path.basename(audio_file)
        results[file_name] = {}
        
        print("\n" + "=" * 50)
        print(f"测试音频文件: {file_name}")
        print("=" * 50)
        
        # 使用 2023-06-26 int8 模型
        if model_2023_06_26_int8:
            print("\n使用 2023-06-26 int8 模型:")
            result = transcribe_audio(model_2023_06_26_int8, audio_file)
            results[file_name]["2023-06-26 int8"] = result
            print(f"转录结果: {result}")
        
        # 使用 2023-06-26 标准模型
        if model_2023_06_26_std:
            print("\n使用 2023-06-26 标准模型:")
            result = transcribe_audio(model_2023_06_26_std, audio_file)
            results[file_name]["2023-06-26 std"] = result
            print(f"转录结果: {result}")
        
        # 使用 2023-02-20 int8 模型
        if model_2023_02_20_int8:
            print("\n使用 2023-02-20 int8 模型:")
            result = transcribe_audio(model_2023_02_20_int8, audio_file)
            results[file_name]["2023-02-20 int8"] = result
            print(f"转录结果: {result}")
        
        # 使用 2023-02-20 标准模型
        if model_2023_02_20_std:
            print("\n使用 2023-02-20 标准模型:")
            result = transcribe_audio(model_2023_02_20_std, audio_file)
            results[file_name]["2023-02-20 std"] = result
            print(f"转录结果: {result}")
    
    # 保存结果到文件
    result_file = os.path.join(os.path.dirname(__file__), "sherpa_model_comparison_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {result_file}")
    
    # 打印结果比较
    print("\n" + "=" * 80)
    print("模型比较结果")
    print("=" * 80)
    
    for file_name, file_results in results.items():
        print(f"\n文件: {file_name}")
        for model_name, result in file_results.items():
            print(f"  {model_name}: {result}")


if __name__ == "__main__":
    main()
