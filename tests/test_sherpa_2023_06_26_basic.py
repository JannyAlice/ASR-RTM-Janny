"""
测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 基本模型的在线转录功能
使用控制台输出，不使用GUI界面
"""
import os
import sys
import time
import numpy as np
import soundcard as sc

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

def test_sherpa_2023_06_26_basic():
    """测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 基本模型的在线转录功能"""
    print("=" * 80)
    print("测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 基本模型的在线转录功能")
    print("=" * 80)

    # 初始化日志工具
    sherpa_logger.setup()
    sherpa_log_file = sherpa_logger.get_log_file()
    print(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")

    # 模型路径
    model_path = r"C:\Users\crige\RealtimeTrans\vosk-api\models\asr\sherpa-onnx-streaming-zipformer-en-2023-06-26"

    # 确定模型文件名 - 使用基本版本，不使用int8量化版本
    encoder_file = "encoder-epoch-99-avg-1-chunk-16-left-128.onnx"
    decoder_file = "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"
    joiner_file = "joiner-epoch-99-avg-1-chunk-16-left-128.onnx"
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
            print(f"错误: 模型文件不存在: {file_path}")
            return

    # 创建 OnlineRecognizer
    try:
        print("创建 OnlineRecognizer...")
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=os.path.join(model_path, encoder_file),
            decoder=os.path.join(model_path, decoder_file),
            joiner=os.path.join(model_path, joiner_file),
            tokens=os.path.join(model_path, tokens_file),
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cpu",
        )
        sherpa_logger.info("成功创建 OnlineRecognizer")
        print("成功创建 OnlineRecognizer")
    except Exception as e:
        sherpa_logger.error(f"创建 OnlineRecognizer 失败: {e}")
        print(f"错误: 创建 OnlineRecognizer 失败: {e}")
        import traceback
        sherpa_logger.error(traceback.format_exc())
        print(traceback.format_exc())
        return

    # 获取音频设备
    print("获取音频设备...")
    sherpa_logger.info("可用的音频设备:")
    speakers = sc.all_speakers()
    for i, speaker in enumerate(speakers):
        print(f"{i}: {speaker.name}")
        sherpa_logger.info(f"{i}: {speaker.name}")

    # 选择默认设备
    default_device = None
    for speaker in speakers:
        if "CABLE" in speaker.name:
            default_device = speaker
            break

    if not default_device:
        default_device = speakers[0]

    print(f"使用设备: {default_device.name}")
    sherpa_logger.info(f"使用设备: {default_device.name}")

    # 设置参数
    sample_rate = 16000
    buffer_size = 8000  # 500ms

    # 创建转录结果文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    transcript_file = os.path.join(project_root, "transcripts", f"transcript_2023_06_26_basic_{timestamp}.txt")
    os.makedirs(os.path.dirname(transcript_file), exist_ok=True)
    
    # 写入文件头
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(f"# Sherpa-ONNX 2023-06-26 基本模型转录结果\n")
        f.write(f"# 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 设备: {default_device.name}\n")
        f.write(f"# 采样率: {sample_rate}\n")
        f.write(f"# 缓冲区大小: {buffer_size}\n\n")
    
    print(f"转录结果将保存到: {transcript_file}")

    # 捕获音频
    try:
        with sc.get_microphone(id=str(default_device.id), include_loopback=True).recorder(samplerate=sample_rate) as mic:
            print("开始捕获音频...")
            print("请播放音频，按 Ctrl+C 停止...")
            sherpa_logger.info("开始捕获音频...")
            sherpa_logger.info("请播放音频，按 Ctrl+C 停止...")

            # 记录开始时间
            start_time = time.time()

            # 创建持久的流
            stream = recognizer.create_stream()
            sherpa_logger.info("创建持久的流")
            print("创建持久的流")

            # 用于存储当前识别的文本
            current_text = ""
            
            # 循环捕获音频
            try:
                while True:
                    # 捕获音频数据
                    data = mic.record(numframes=buffer_size)

                    # 转换为单声道
                    if data.shape[1] > 1:
                        data = np.mean(data, axis=1)

                    # 记录音频数据信息
                    sherpa_logger.debug(f"音频数据形状: {data.shape}, 最大值: {np.max(np.abs(data))}")

                    # 检查数据是否有效
                    if np.max(np.abs(data)) < 0.01:
                        sherpa_logger.debug("音频数据几乎是静音，跳过")
                        print(".", end="", flush=True)
                        continue

                    # 处理音频数据
                    try:
                        # 接受音频数据
                        stream.accept_waveform(sample_rate, data)
                        sherpa_logger.debug("接受音频数据成功")

                        # 解码
                        while recognizer.is_ready(stream):
                            recognizer.decode_stream(stream)
                        sherpa_logger.debug("解码完成")

                        # 获取结果
                        text = recognizer.get_result(stream)
                        sherpa_logger.debug(f"获取结果: '{text}'")

                        if text and text != current_text:
                            # 格式化文本：首字母大写，末尾加句号
                            if text:
                                text = text[0].upper() + text[1:]
                                if not text.endswith(('.', '!', '?')):
                                    text += '.'
                            
                            # 计算时间戳
                            elapsed = time.time() - start_time
                            timestamp_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
                            
                            # 输出结果
                            print(f"\n[{timestamp_str}] {text}")
                            sherpa_logger.info(f"转录结果: {text}")
                            
                            # 保存到文件
                            with open(transcript_file, "a", encoding="utf-8") as f:
                                f.write(f"[{timestamp_str}] {text}\n")
                            
                            # 更新当前文本
                            current_text = text
                            
                            # 每次获取到新的文本后，创建新的流
                            # 这样可以避免文本累积问题，每次都是独立的识别
                            stream = recognizer.create_stream()
                            sherpa_logger.debug("创建新的流")
                    except Exception as e:
                        sherpa_logger.error(f"\n处理音频数据错误: {e}")
                        print(f"\n处理音频数据错误: {e}")
                        import traceback
                        sherpa_logger.error(traceback.format_exc())
                        print(traceback.format_exc())
                        
                        # 创建新的流，避免错误累积
                        stream = recognizer.create_stream()
                        sherpa_logger.debug("创建新的流（错误恢复）")

                    # 等待一段时间
                    time.sleep(0.1)

            except KeyboardInterrupt:
                sherpa_logger.info("\n捕获音频已停止")
                print("\n捕获音频已停止")
                
                # 获取最终结果
                try:
                    # 添加尾部填充
                    tail_paddings = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
                    stream.accept_waveform(sample_rate, tail_paddings)
                    sherpa_logger.debug("添加尾部填充成功")
                    
                    # 标记输入结束
                    stream.input_finished()
                    sherpa_logger.debug("标记输入结束成功")
                    
                    # 解码
                    while recognizer.is_ready(stream):
                        recognizer.decode_stream(stream)
                    sherpa_logger.debug("最终解码完成")
                    
                    # 获取最终结果
                    final_text = recognizer.get_result(stream)
                    sherpa_logger.debug(f"最终结果: '{final_text}'")
                    
                    if final_text and final_text != current_text:
                        # 格式化文本
                        if final_text:
                            final_text = final_text[0].upper() + final_text[1:]
                            if not final_text.endswith(('.', '!', '?')):
                                final_text += '.'
                        
                        # 计算时间戳
                        elapsed = time.time() - start_time
                        timestamp_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
                        
                        # 输出结果
                        print(f"\n[{timestamp_str}] 最终结果: {final_text}")
                        sherpa_logger.info(f"最终结果: {final_text}")
                        
                        # 保存到文件
                        with open(transcript_file, "a", encoding="utf-8") as f:
                            f.write(f"[{timestamp_str}] [最终结果] {final_text}\n")
                except Exception as e:
                    sherpa_logger.error(f"获取最终结果错误: {e}")
                    print(f"获取最终结果错误: {e}")
                
                # 写入文件尾
                with open(transcript_file, "a", encoding="utf-8") as f:
                    f.write(f"\n# 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# 总时长: {time.time() - start_time:.2f} 秒\n")

    except Exception as e:
        sherpa_logger.error(f"捕获音频错误: {e}")
        print(f"捕获音频错误: {e}")
        import traceback
        sherpa_logger.error(traceback.format_exc())
        print(traceback.format_exc())

    sherpa_logger.info(f"测试完成，日志文件: {sherpa_log_file}")
    print(f"测试完成，日志文件: {sherpa_log_file}")
    print(f"转录结果已保存到: {transcript_file}")

if __name__ == "__main__":
    test_sherpa_2023_06_26_basic()
