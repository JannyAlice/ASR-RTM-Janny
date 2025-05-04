#!/usr/bin/env python3
"""
测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的系统音频捕获和在线转录功能

此测试程序用于验证新模型的系统音频捕获和在线转录功能，可以播放MP4文件，
测试程序会实时捕获系统音频并进行转录。使用控制台输出结果，避免GUI相关问题。
"""

import os
import sys
import time
import json
import threading
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

# 导入音频捕获工具
import soundcard as sc

# 模型路径
MODEL_2023_06_26_PATH = r"C:\Users\crige\RealtimeTrans\vosk-api\models\asr\sherpa-onnx-streaming-zipformer-en-2023-06-26"


def load_model_2023_06_26(use_int8: bool = True):
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


def format_text(text):
    """格式化文本：首字母大写，末尾加句号"""
    if not text:
        return text
    
    text = text[0].upper() + text[1:]
    if not text.endswith(('.', '!', '?')):
        text += '.'
    
    return text


def capture_audio(recognizer, device, sample_rate=16000, buffer_size=4000):
    """
    捕获系统音频并进行转录
    
    Args:
        recognizer: sherpa-onnx 识别器
        device: 音频设备
        sample_rate: 采样率
        buffer_size: 缓冲区大小
    """
    # 创建转录结果文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    transcript_file = os.path.join(project_root, "transcripts", f"transcript_2023_06_26_{timestamp}.txt")
    os.makedirs(os.path.dirname(transcript_file), exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    # 写入文件头
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(f"# Sherpa-ONNX 2023-06-26 模型转录结果\n")
        f.write(f"# 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 设备: {device.name}\n")
        f.write(f"# 采样率: {sample_rate}\n")
        f.write(f"# 缓冲区大小: {buffer_size}\n\n")
    
    print(f"转录结果将保存到: {transcript_file}")
    print("开始捕获音频...")
    print("请播放音频，按 Ctrl+C 停止...")
    
    try:
        with sc.get_microphone(id=str(device.id), include_loopback=True).recorder(samplerate=sample_rate) as mic:
            # 循环捕获音频
            is_running = True
            
            while is_running:
                try:
                    # 捕获音频数据
                    data = mic.record(numframes=buffer_size)
                    
                    # 转换为单声道
                    if data.shape[1] > 1:
                        data = np.mean(data, axis=1)
                    
                    # 检查数据是否有效
                    if np.max(np.abs(data)) < 0.01:
                        print(".", end="", flush=True)
                        continue
                    
                    # 处理音频数据
                    try:
                        # 创建一个新的流
                        stream = recognizer.create_stream()
                        
                        # 接受音频数据
                        stream.accept_waveform(sample_rate, data)
                        
                        # 添加尾部填充
                        tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
                        stream.accept_waveform(sample_rate, tail_paddings)
                        
                        # 标记输入结束
                        stream.input_finished()
                        
                        # 解码
                        while recognizer.is_ready(stream):
                            recognizer.decode_stream(stream)
                        
                        # 获取结果
                        result = recognizer.get_result(stream)
                        
                        if result:
                            # 格式化文本
                            text = format_text(result)
                            
                            # 计算时间戳
                            elapsed = time.time() - start_time
                            timestamp = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
                            
                            # 输出结果
                            print(f"\n[{timestamp}] {text}")
                            
                            # 保存到文件
                            with open(transcript_file, "a", encoding="utf-8") as f:
                                f.write(f"[{timestamp}] {text}\n")
                        else:
                            print(".", end="", flush=True)
                    
                    except Exception as e:
                        print(f"\n处理音频数据错误: {e}")
                        import traceback
                        print(traceback.format_exc())
                    
                    # 等待一段时间
                    time.sleep(0.1)
                
                except KeyboardInterrupt:
                    print("\n用户中断，停止转录...")
                    is_running = False
                
                except Exception as e:
                    print(f"\n捕获音频数据错误: {e}")
                    import traceback
                    print(traceback.format_exc())
                    time.sleep(0.5)  # 出错后等待一段时间再继续
    
    except Exception as e:
        print(f"捕获音频错误: {e}")
        import traceback
        print(traceback.format_exc())
    
    # 写入文件尾
    with open(transcript_file, "a", encoding="utf-8") as f:
        f.write(f"\n# 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 总时长: {time.time() - start_time:.2f} 秒\n")
    
    print(f"\n转录结束，结果已保存到: {transcript_file}")


def test_sherpa_2023_06_26_console():
    """测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的系统音频捕获和在线转录功能"""
    print("=" * 80)
    print("测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的系统音频捕获和在线转录功能")
    print("=" * 80)
    
    # 加载 2023-06-26 模型
    print("加载 2023-06-26 模型...")
    recognizer = load_model_2023_06_26(use_int8=False)  # 使用标准模型
    if not recognizer:
        print("加载 2023-06-26 模型失败")
        return
    
    print(f"识别器类型: {type(recognizer).__name__}")
    
    # 获取音频设备
    print("获取音频设备...")
    speakers = sc.all_speakers()
    for i, speaker in enumerate(speakers):
        print(f"{i}: {speaker.name}")
    
    # 选择默认设备
    default_device = None
    for speaker in speakers:
        if "CABLE" in speaker.name:
            default_device = speaker
            break
    
    if not default_device:
        default_device = speakers[0]
    
    print(f"使用设备: {default_device.name}")
    
    # 设置参数
    sample_rate = 16000
    buffer_size = 4000  # 250ms
    
    # 捕获音频
    try:
        capture_audio(recognizer, default_device, sample_rate, buffer_size)
    except KeyboardInterrupt:
        print("\n用户中断，停止转录...")


if __name__ == "__main__":
    test_sherpa_2023_06_26_console()
