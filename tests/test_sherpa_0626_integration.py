"""
测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的集成
测试文件转录和在线转录功能
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
    import soundcard as sc
    HAS_SHERPA_ONNX = True
except ImportError:
    HAS_SHERPA_ONNX = False
    print("未安装 sherpa_onnx 或 soundcard 模块，无法测试 Sherpa-ONNX 模型")
    sys.exit(1)

# 导入必要的模块
from src.core.asr.model_manager import ASRModelManager
from src.utils.sherpa_logger import sherpa_logger

def test_sherpa_0626_file_transcription():
    """测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的文件转录功能"""
    print("=" * 80)
    print("测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的文件转录功能")
    print("=" * 80)

    # 初始化日志工具
    sherpa_logger.setup()
    sherpa_log_file = sherpa_logger.get_log_file()
    print(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")

    # 创建模型管理器
    model_manager = ASRModelManager()

    # 加载 sherpa_0626 模型
    print("加载 sherpa_0626 模型...")
    if model_manager.load_model("sherpa_0626"):
        print("成功加载 sherpa_0626 模型")
    else:
        print("加载 sherpa_0626 模型失败")
        return

    # 获取当前引擎类型
    engine_type = model_manager.get_current_engine_type()
    print(f"当前引擎类型: {engine_type}")

    # 确保引擎类型是 sherpa_0626，而不是 vosk
    if engine_type != "sherpa_0626":
        print(f"错误: 引擎类型应该是 sherpa_0626，但实际是 {engine_type}")
        print("测试失败: 出现模型混乱情况，可能自动降级到了其他模型")
        return

    print("测试通过: 引擎类型正确")

    # 测试文件路径
    test_file = input("请输入要转录的音频文件路径: ")
    if not os.path.exists(test_file):
        print(f"文件不存在: {test_file}")
        return

    # 转录文件
    print(f"开始转录文件: {test_file}")
    start_time = time.time()
    result = model_manager.transcribe_file(test_file)
    end_time = time.time()

    if result:
        print(f"转录成功，耗时: {end_time - start_time:.2f} 秒")
        print(f"转录结果: {result}")

        # 保存转录结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(project_root, "transcripts", f"transcript_0626_file_{timestamp}.txt")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Sherpa-ONNX 2023-06-26 模型文件转录结果\n")
            f.write(f"# 文件: {test_file}\n")
            f.write(f"# 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
            f.write(f"# 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
            f.write(f"# 耗时: {end_time - start_time:.2f} 秒\n\n")
            f.write(result)
        print(f"转录结果已保存到: {output_file}")
    else:
        print("转录失败")

def test_sherpa_0626_online_transcription():
    """测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的在线转录功能"""
    print("=" * 80)
    print("测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的在线转录功能")
    print("=" * 80)

    # 初始化日志工具
    sherpa_logger.setup()
    sherpa_log_file = sherpa_logger.get_log_file()
    print(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")

    # 创建模型管理器
    model_manager = ASRModelManager()

    # 加载 sherpa_0626 模型
    print("加载 sherpa_0626 模型...")
    if model_manager.load_model("sherpa_0626"):
        print("成功加载 sherpa_0626 模型")
    else:
        print("加载 sherpa_0626 模型失败")
        return

    # 获取当前引擎类型
    engine_type = model_manager.get_current_engine_type()
    print(f"当前引擎类型: {engine_type}")

    # 检查当前引擎实例
    print(f"当前引擎实例: {model_manager.current_engine}")
    print(f"当前引擎类型: {model_manager.model_type}")

    # 检查引擎是否已初始化
    if hasattr(model_manager.current_engine, 'recognizer'):
        print(f"引擎识别器已初始化: {model_manager.current_engine.recognizer}")
    else:
        print("警告: 引擎识别器未初始化")

    # 创建识别器
    print("创建识别器...")
    recognizer = model_manager.create_recognizer()
    if not recognizer:
        print("创建识别器失败")
        # 打印更多调试信息
        print(f"模型管理器状态:")
        print(f"  当前模型: {model_manager.current_model}")
        print(f"  当前引擎: {model_manager.current_engine}")
        print(f"  模型类型: {model_manager.model_type}")
        print(f"  模型路径: {model_manager.model_path}")
        return
    print(f"成功创建识别器: {type(recognizer).__name__}")

    # 检查识别器的引擎类型
    if hasattr(recognizer, 'engine_type'):
        print(f"识别器引擎类型: {recognizer.engine_type}")

        # 确保引擎类型是 sherpa_0626，而不是 vosk
        if recognizer.engine_type != "sherpa_0626":
            print(f"错误: 识别器引擎类型应该是 sherpa_0626，但实际是 {recognizer.engine_type}")
            print("测试失败: 出现模型混乱情况，可能自动降级到了其他模型")
            return

        print("测试通过: 识别器引擎类型正确")
    else:
        print("警告: 识别器没有 engine_type 属性，无法验证模型类型")

    # 检查识别器的其他属性
    print("识别器属性:")
    for attr in dir(recognizer):
        if not attr.startswith('__'):
            try:
                value = getattr(recognizer, attr)
                if not callable(value):
                    print(f"  {attr}: {value}")
            except:
                pass

    # 导入音频处理器
    try:
        from src.core.audio.audio_processor import AudioProcessor
        from src.core.signals import TranscriptionSignals
    except ImportError:
        print("导入音频处理器失败")
        return

    # 创建信号
    signals = TranscriptionSignals()

    # 创建音频处理器
    audio_processor = AudioProcessor(signals)

    # 获取音频设备
    devices = audio_processor.get_audio_devices()
    if not devices:
        print("未找到音频设备")
        return

    # 显示可用设备
    print("可用音频设备:")
    for i, device in enumerate(devices):
        print(f"{i}: {device.name}")

    # 选择设备
    try:
        device_index = int(input("请选择音频设备 (输入序号): "))
        if device_index < 0 or device_index >= len(devices):
            print("无效的设备序号")
            return
        device = devices[device_index]
    except ValueError:
        print("无效的输入")
        return

    # 设置当前设备
    audio_processor.set_current_device(device)
    print(f"已选择设备: {device.name}")

    # 创建转录结果文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(project_root, "transcripts", f"transcript_0626_online_{timestamp}.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Sherpa-ONNX 2023-06-26 模型在线转录结果\n")
        f.write(f"# 设备: {device.name}\n")
        f.write(f"# 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    print(f"转录结果将保存到: {output_file}")

    # 定义文本更新回调
    def on_text_updated(text):
        print(f"转录结果: {text}")
        print(f"转录结果类型: {type(text)}")
        print(f"转录结果长度: {len(text) if text else 0}")

        # 检查文本是否为空
        if not text or len(text.strip()) == 0:
            print("警告: 转录结果为空，不写入文件")
            return

        # 写入文件
        try:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"{text}\n")
                print(f"成功写入文件: {text}")
        except Exception as e:
            print(f"写入文件错误: {e}")

    # 连接信号
    signals.new_text.connect(on_text_updated)

    # 开始捕获音频
    print("开始捕获音频...")
    print("请播放音频，按 Ctrl+C 停止...")
    if not audio_processor.start_capture(recognizer):
        print("开始音频捕获失败")
        return

    try:
        # 等待用户按 Ctrl+C 停止
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n捕获音频已停止")

    # 停止捕获
    audio_processor.stop_capture()

    # 写入文件尾
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"\n# 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"转录结果已保存到: {output_file}")

def test_sherpa_0626_online_transcription_persistent():
    """测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的在线转录功能（持久流方式）"""
    print("=" * 80)
    print("测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的在线转录功能（持久流方式）")
    print("=" * 80)

    # 初始化日志工具
    sherpa_logger.setup()
    sherpa_log_file = sherpa_logger.get_log_file()
    print(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")

    # 模型路径
    model_path = r"C:\Users\crige\RealtimeTrans\vosk-api\models\asr\sherpa-onnx-streaming-zipformer-en-2023-06-26"

    # 确定模型文件名 - 使用高级版本，不使用int8量化版本
    encoder_file = "encoder-epoch-99-avg-1-chunk-16-left-128.onnx"  # 高级模型文件
    decoder_file = "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"  # 高级模型文件
    joiner_file = "joiner-epoch-99-avg-1-chunk-16-left-128.onnx"    # 高级模型文件
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
    transcript_file = os.path.join(project_root, "transcripts", f"transcript_0626_persistent_{timestamp}.txt")
    os.makedirs(os.path.dirname(transcript_file), exist_ok=True)

    # 写入文件头
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(f"# Sherpa-ONNX 2023-06-26 模型转录结果（持久流方式）\n")
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

            # 创建持久的流（只创建一次，整个过程中使用同一个流）
            stream = recognizer.create_stream()
            sherpa_logger.info("创建持久的流")
            print("创建持久的流")

            # 用于存储当前识别的文本
            current_text = ""

            # 累积的文本
            accumulated_text = ""

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

                            # 累积文本
                            accumulated_text += " " + text
                    except Exception as e:
                        sherpa_logger.error(f"\n处理音频数据错误: {e}")
                        print(f"\n处理音频数据错误: {e}")
                        import traceback
                        sherpa_logger.error(traceback.format_exc())
                        print(traceback.format_exc())

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

                        # 累积文本
                        accumulated_text += " " + final_text
                except Exception as e:
                    sherpa_logger.error(f"获取最终结果错误: {e}")
                    print(f"获取最终结果错误: {e}")

                # 写入文件尾
                with open(transcript_file, "a", encoding="utf-8") as f:
                    f.write(f"\n# 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# 总时长: {time.time() - start_time:.2f} 秒\n")

                    # 写入累积的文本
                    if accumulated_text:
                        f.write(f"\n# 累积文本:\n{accumulated_text.strip()}\n")

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
    # 直接运行在线转录测试（持久流方式）
    print("开始测试 sherpa_0626 在线转录功能（持久流方式）...")
    test_sherpa_0626_online_transcription_persistent()
