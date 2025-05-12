"""
测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的高级功能
包括：
1. 持久流模式
2. 累积文本功能
3. 时间戳功能
4. 格式化文本功能
5. 文件保存逻辑
"""
import os
import sys
import time
import json
import numpy as np
from datetime import datetime

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

# 导入模型管理器
from src.core.asr.model_manager import ASRModelManager

def test_sherpa_0626_features():
    """测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的高级功能"""
    print("=" * 80)
    print("测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的高级功能")
    print("=" * 80)

    # 初始化日志工具
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
    print("加载 sherpa_0626 模型成功")

    # 创建识别器
    print("创建识别器...")
    recognizer = model_manager.create_recognizer()
    if not recognizer:
        print("创建识别器失败")
        return
    print("创建识别器成功")

    # 测试持久流模式
    test_persistent_stream(recognizer)

    # 测试累积文本功能
    test_accumulated_text(recognizer)

    # 测试时间戳功能
    test_timestamps(recognizer)

    # 测试格式化文本功能
    test_text_formatting(recognizer)

    # 测试文件保存逻辑
    test_file_saving(recognizer)

    # 打印日志文件路径
    print("\n" + "=" * 40)
    print("测试完成，日志和转录文件信息")
    print("=" * 40)
    print(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")

    # 打印转录文件目录
    transcript_dir = os.path.join(project_root, "transcripts")
    print(f"转录文件目录: {transcript_dir}")

    # 列出最近生成的转录文件
    if os.path.exists(transcript_dir):
        files = [f for f in os.listdir(transcript_dir) if f.startswith("test_features_")]
        files.sort(reverse=True)  # 按名称倒序排列，最新的文件在前面

        if files:
            print("\n最近生成的转录文件:")
            for i, file in enumerate(files[:5]):  # 只显示最近的5个文件
                file_path = os.path.join(transcript_dir, file)
                file_size = os.path.getsize(file_path)
                file_time = os.path.getmtime(file_path)
                file_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(file_time))
                print(f"{i+1}. {file} ({file_size} 字节, {file_time_str})")
        else:
            print("\n未找到转录文件，请检查测试是否成功完成")
    else:
        print(f"\n转录文件目录不存在: {transcript_dir}")

    print("\n所有测试完成")

def test_persistent_stream(recognizer):
    """测试持久流模式"""
    print("\n" + "=" * 40)
    print("测试持久流模式")
    print("=" * 40)

    # 创建测试音频数据
    sample_rate = 16000
    duration = 3  # 3秒
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # 生成一个简单的音频信号（440Hz的正弦波）
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)

    # 将音频数据分成多个块
    chunk_size = 1600  # 100ms
    chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]

    print(f"音频数据分成了 {len(chunks)} 个块")

    # 重置识别器
    recognizer.Reset()

    # 处理音频块
    for i, chunk in enumerate(chunks):
        print(f"处理第 {i+1}/{len(chunks)} 个块...")
        recognizer.AcceptWaveform(chunk)

        # 获取部分结果
        partial = recognizer.PartialResult()
        if partial:
            partial_json = json.loads(partial)
            if "partial" in partial_json:
                print(f"部分结果: {partial_json['partial']}")

    # 获取最终结果
    final = recognizer.FinalResult()
    if final:
        final_json = json.loads(final)
        if "text" in final_json:
            print(f"最终结果: {final_json['text']}")

    print("持久流模式测试完成")

def test_accumulated_text(recognizer):
    """测试累积文本功能"""
    print("\n" + "=" * 40)
    print("测试累积文本功能")
    print("=" * 40)

    # 重置识别器
    recognizer.Reset()

    # 创建测试音频数据
    sample_rate = 16000
    duration = 5  # 5秒
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # 生成一个简单的音频信号（440Hz的正弦波）
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)

    # 将音频数据分成多个块
    chunk_size = 1600  # 100ms
    chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]

    # 处理音频块
    for i, chunk in enumerate(chunks):
        if i % 10 == 0:  # 每10个块打印一次
            print(f"处理第 {i+1}/{len(chunks)} 个块...")
        recognizer.AcceptWaveform(chunk)

    # 获取最终结果
    final = recognizer.FinalResult()
    if final:
        final_json = json.loads(final)
        if "text" in final_json:
            print(f"最终结果: {final_json['text']}")

    # 获取累积文本
    accumulated_text = recognizer.get_accumulated_text() if hasattr(recognizer, 'get_accumulated_text') else None
    if accumulated_text:
        print(f"累积文本: {accumulated_text}")
    else:
        print("识别器没有 get_accumulated_text 方法")

    print("累积文本功能测试完成")

def test_timestamps(recognizer):
    """测试时间戳功能"""
    print("\n" + "=" * 40)
    print("测试时间戳功能")
    print("=" * 40)

    # 重置识别器
    recognizer.Reset()

    # 创建测试音频数据
    sample_rate = 16000
    duration = 5  # 5秒
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # 生成一个简单的音频信号（440Hz的正弦波）
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)

    # 将音频数据分成多个块
    chunk_size = 1600  # 100ms
    chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]

    # 处理音频块
    for i, chunk in enumerate(chunks):
        if i % 10 == 0:  # 每10个块打印一次
            print(f"处理第 {i+1}/{len(chunks)} 个块...")
        recognizer.AcceptWaveform(chunk)

    # 获取最终结果
    final = recognizer.FinalResult()
    if final:
        final_json = json.loads(final)
        if "text" in final_json:
            print(f"最终结果: {final_json['text']}")

    # 获取带时间戳的累积文本
    timestamped_text = recognizer.get_accumulated_text_with_timestamps() if hasattr(recognizer, 'get_accumulated_text_with_timestamps') else None
    if timestamped_text:
        print("带时间戳的累积文本:")
        for text, timestamp in timestamped_text:
            print(f"[{timestamp}] {text}")
    else:
        print("识别器没有 get_accumulated_text_with_timestamps 方法")

    print("时间戳功能测试完成")

def test_text_formatting(recognizer):
    """测试格式化文本功能"""
    print("\n" + "=" * 40)
    print("测试格式化文本功能")
    print("=" * 40)

    # 重置识别器
    recognizer.Reset()

    # 创建测试文本
    test_texts = [
        "this is a test",
        "what is your name",
        "hello world",
        "i am a student",
        "how are you doing today"
    ]

    # 测试格式化文本
    if hasattr(recognizer, '_format_text'):
        print("使用识别器的 _format_text 方法格式化文本:")
        for text in test_texts:
            formatted_text = recognizer._format_text(text)
            print(f"原始文本: '{text}'")
            print(f"格式化后: '{formatted_text}'")
            print()
    else:
        print("识别器没有 _format_text 方法")

    print("格式化文本功能测试完成")

def test_file_saving(recognizer):
    """测试文件保存逻辑"""
    print("\n" + "=" * 40)
    print("测试文件保存逻辑")
    print("=" * 40)

    # 重置识别器
    recognizer.Reset()

    # 创建测试音频数据
    sample_rate = 16000
    duration = 5  # 5秒
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # 生成一个简单的音频信号（440Hz的正弦波）
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)

    # 将音频数据分成多个块
    chunk_size = 1600  # 100ms
    chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]

    # 处理音频块
    for i, chunk in enumerate(chunks):
        if i % 10 == 0:  # 每10个块打印一次
            print(f"处理第 {i+1}/{len(chunks)} 个块...")
        recognizer.AcceptWaveform(chunk)

    # 获取最终结果
    final = recognizer.FinalResult()
    if final:
        final_json = json.loads(final)
        if "text" in final_json:
            print(f"最终结果: {final_json['text']}")

    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(project_root, "transcripts")
    os.makedirs(save_dir, exist_ok=True)

    # 获取格式化的转录文本
    formatted_transcript = None
    if hasattr(recognizer, 'get_formatted_transcript'):
        formatted_transcript = recognizer.get_formatted_transcript()
        if formatted_transcript:
            print("格式化的转录文本:")
            print(formatted_transcript)

    # 如果没有格式化的转录文本，但有最终结果，使用最终结果
    if not formatted_transcript and hasattr(recognizer, 'final_result'):
        formatted_transcript = recognizer.final_result
        print("使用最终结果作为格式化的转录文本:")
        print(formatted_transcript)

    # 如果仍然没有文本，创建一个简单的测试文本
    if not formatted_transcript:
        formatted_transcript = "这是一个测试文本，用于验证文件保存功能。\n生成时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("使用测试文本:")
        print(formatted_transcript)

    # 保存TXT格式
    txt_path = os.path.join(save_dir, f"test_features_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(formatted_transcript)
    print(f"已保存TXT格式: {txt_path}")

    # 同时保存一个包含所有数据的调试文件
    debug_path = txt_path.replace('.txt', '_debug.txt')
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write("=== 格式化的转录文本 ===\n")
        f.write(formatted_transcript)
        f.write("\n\n")

        # 写入累积文本
        if hasattr(recognizer, 'get_accumulated_text'):
            accumulated_text = recognizer.get_accumulated_text()
            f.write("=== 累积文本 ===\n")
            if accumulated_text:
                for text in accumulated_text:
                    f.write(f"{text}\n")
            else:
                f.write("(无累积文本)\n")
            f.write("\n\n")
        else:
            f.write("=== 累积文本 ===\n")
            f.write("(识别器没有 get_accumulated_text 方法)\n\n")

        # 写入带时间戳的文本
        if hasattr(recognizer, 'get_accumulated_text_with_timestamps'):
            timestamped_text = recognizer.get_accumulated_text_with_timestamps()
            f.write("=== 带时间戳的文本 ===\n")
            if timestamped_text:
                for text, timestamp in timestamped_text:
                    f.write(f"[{timestamp}] {text}\n")
            else:
                f.write("(无带时间戳的文本)\n")
            f.write("\n\n")
        else:
            f.write("=== 带时间戳的文本 ===\n")
            f.write("(识别器没有 get_accumulated_text_with_timestamps 方法)\n\n")

        # 写入识别器的属性
        f.write("=== 识别器属性 ===\n")
        for attr in dir(recognizer):
            if not attr.startswith('_'):  # 排除私有属性
                try:
                    value = getattr(recognizer, attr)
                    if not callable(value):  # 排除方法
                        f.write(f"{attr}: {value}\n")
                except:
                    pass
        f.write("\n\n")

        # 写入测试信息
        f.write("=== 测试信息 ===\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"识别器类型: {type(recognizer).__name__}\n")
        f.write(f"引擎类型: {getattr(recognizer, 'engine_type', 'unknown')}\n")

    print(f"已保存调试文件: {debug_path}")

    # 保存SRT格式
    srt_path = os.path.join(save_dir, f"test_features_{timestamp}.srt")
    if hasattr(recognizer, 'get_accumulated_text_with_timestamps'):
        timestamped_text = recognizer.get_accumulated_text_with_timestamps()
        if timestamped_text:
            with open(srt_path, "w", encoding="utf-8") as f:
                for i, (text, timestamp) in enumerate(timestamped_text, 1):
                    # 解析时间戳
                    h, m, s = timestamp.split(':')
                    start_time = f"00:{h}:{m},{s}00"

                    # 计算结束时间（假设每段字幕持续5秒）
                    h_end, m_end, s_end = int(h), int(m), int(s) + 5
                    if s_end >= 60:
                        s_end -= 60
                        m_end += 1
                    if m_end >= 60:
                        m_end -= 60
                        h_end += 1
                    end_time = f"00:{h_end:02d}:{m_end:02d},{s_end:02d}00"

                    # 写入SRT格式
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
            print(f"已保存SRT格式: {srt_path}")
        else:
            # 创建一个简单的SRT文件
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write("1\n")
                f.write("00:00:00,000 --> 00:00:05,000\n")
                f.write("这是一个测试字幕，用于验证SRT文件保存功能。\n\n")
                f.write("2\n")
                f.write("00:00:05,000 --> 00:00:10,000\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            print(f"已保存测试SRT格式: {srt_path}")
    else:
        # 创建一个简单的SRT文件
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("1\n")
            f.write("00:00:00,000 --> 00:00:05,000\n")
            f.write("这是一个测试字幕，用于验证SRT文件保存功能。\n\n")
            f.write("2\n")
            f.write("00:00:05,000 --> 00:00:10,000\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        print(f"已保存测试SRT格式: {srt_path}")

    # 打印文件保存信息
    print("\n文件保存信息:")
    print(f"- TXT格式: {txt_path}")
    print(f"- 调试文件: {debug_path}")
    print(f"- SRT格式: {srt_path}")
    print(f"- 保存目录: {save_dir}")

    print("文件保存逻辑测试完成")

if __name__ == "__main__":
    test_sherpa_0626_features()
