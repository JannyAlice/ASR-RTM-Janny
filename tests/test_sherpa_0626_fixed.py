#!/usr/bin/env python3
"""
测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的修复版本
包含模型加载、流创建、文件转录和在线转录功能测试
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

# 导入 soundcard 模块（用于捕获系统音频）
try:
    import soundcard as sc
    HAS_SOUNDCARD = True
except ImportError:
    HAS_SOUNDCARD = False
    print("未安装 soundcard 模块，无法测试在线转录功能")

# 导入必要的模块
from src.core.asr.model_manager import ASRModelManager
from src.utils.sherpa_logger import sherpa_logger

def test_sherpa_0626_model_loading():
    """测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的加载"""
    print("=" * 80)
    print("测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的加载")
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
        return False

    # 获取当前引擎类型
    engine_type = model_manager.get_current_engine_type()
    print(f"当前引擎类型: {engine_type}")

    # 确保引擎类型是 sherpa_0626，而不是 vosk
    if engine_type != "sherpa_0626":
        print(f"错误: 引擎类型应该是 sherpa_0626，但实际是 {engine_type}")
        print("测试失败: 出现模型混乱情况，可能自动降级到了其他模型")
        return False

    print("测试通过: 引擎类型正确")
    return True

def test_sherpa_0626_stream_creation():
    """测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的流创建"""
    print("=" * 80)
    print("测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的流创建")
    print("=" * 80)

    # 初始化日志工具
    sherpa_logger.setup()
    sherpa_log_file = sherpa_logger.get_log_file()
    print(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")

    # 创建模型管理器
    model_manager = ASRModelManager()

    # 加载 sherpa_0626 模型
    print("加载 sherpa_0626 模型...")
    if not model_manager.load_model("sherpa_0626"):
        print("加载 sherpa_0626 模型失败")
        return False

    # 初始化引擎
    print("初始化引擎...")
    if not model_manager.initialize_engine("sherpa_0626"):
        print("初始化引擎失败")
        return False

    # 获取当前引擎
    engine = model_manager.current_engine
    if not engine:
        print("获取引擎失败")
        return False

    # 检查引擎是否有 recognizer 属性
    if not hasattr(engine, 'recognizer'):
        print("引擎没有 recognizer 属性")
        return False

    # 检查 recognizer 是否为 None
    if not engine.recognizer:
        print("recognizer 为 None")
        return False

    # 尝试创建流
    try:
        print("尝试创建流...")
        stream = engine.recognizer.create_stream()
        print("成功创建流")
        return True
    except Exception as e:
        print(f"创建流失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False

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
    if not model_manager.load_model("sherpa_0626"):
        print("加载 sherpa_0626 模型失败")
        return False

    # 初始化引擎
    print("初始化引擎...")
    if not model_manager.initialize_engine("sherpa_0626"):
        print("初始化引擎失败")
        return False

    # 测试文件路径
    test_file = input("请输入要转录的音频文件路径: ")
    if not os.path.exists(test_file):
        print(f"文件不存在: {test_file}")
        return False

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
        output_file = os.path.join(project_root, "transcripts", f"transcript_0626_file_fixed_{timestamp}.txt")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Sherpa-ONNX 2023-06-26 模型文件转录结果\n")
            f.write(f"# 文件: {test_file}\n")
            f.write(f"# 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
            f.write(f"# 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
            f.write(f"# 耗时: {end_time - start_time:.2f} 秒\n\n")
            f.write(result)
        print(f"转录结果已保存到: {output_file}")
        return True
    else:
        print("转录失败")
        return False

def test_sherpa_0626_online_transcription():
    """测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的在线转录功能（持久流方式）"""
    print("=" * 80)
    print("测试 sherpa-onnx-streaming-zipformer-en-2023-06-26 模型的在线转录功能（持久流方式）")
    print("=" * 80)

    # 检查是否安装了 soundcard 模块
    if not HAS_SOUNDCARD:
        print("未安装 soundcard 模块，无法测试在线转录功能")
        return False

    # 初始化日志工具
    sherpa_logger.setup()
    sherpa_log_file = sherpa_logger.get_log_file()
    print(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")

    # 创建模型管理器
    model_manager = ASRModelManager()

    # 加载 sherpa_0626 模型
    print("加载 sherpa_0626 模型...")
    if not model_manager.load_model("sherpa_0626"):
        print("加载 sherpa_0626 模型失败")
        return False

    # 初始化引擎
    print("初始化引擎...")
    if not model_manager.initialize_engine("sherpa_0626"):
        print("初始化引擎失败")
        return False

    # 获取当前引擎
    engine = model_manager.current_engine
    if not engine or not engine.recognizer:
        print("获取引擎失败")
        return False

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
    transcript_file = os.path.join(project_root, "transcripts", f"transcript_0626_online_fixed_{timestamp}.txt")
    os.makedirs(os.path.dirname(transcript_file), exist_ok=True)

    # 写入文件头
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(f"# Sherpa-ONNX 2023-06-26 模型在线转录结果（持久流方式）\n")
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
            stream = engine.recognizer.create_stream()
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
                        while engine.recognizer.is_ready(stream):
                            engine.recognizer.decode_stream(stream)
                        sherpa_logger.debug("解码完成")

                        # 获取结果
                        text = engine.recognizer.get_result(stream)
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
                    while engine.recognizer.is_ready(stream):
                        engine.recognizer.decode_stream(stream)
                    sherpa_logger.debug("最终解码完成")

                    # 获取最终结果
                    final_text = engine.recognizer.get_result(stream)
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
        return False

    sherpa_logger.info(f"测试完成，日志文件: {sherpa_log_file}")
    print(f"测试完成，日志文件: {sherpa_log_file}")
    print(f"转录结果已保存到: {transcript_file}")
    return True

def run_all_tests():
    """运行所有测试"""
    tests = [
        ("模型加载测试", test_sherpa_0626_model_loading),
        ("流创建测试", test_sherpa_0626_stream_creation),
        ("文件转录测试", test_sherpa_0626_file_transcription),
        ("在线转录测试", test_sherpa_0626_online_transcription)
    ]

    results = []
    for name, test_func in tests:
        print(f"\n开始 {name}...")
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            import traceback
            print(traceback.format_exc())
            results.append((name, False))

    # 打印测试结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    for name, success in results:
        status = "通过" if success else "失败"
        print(f"{name}: {status}")

if __name__ == "__main__":
    # 运行所有测试
    run_all_tests()
