"""
测试修复后的Sherpa-ONNX在线转录功能（文本拼接问题修复）
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

# 导入 ASRModelManager
from src.core.asr.model_manager import ASRModelManager

# 导入音频捕获工具
import soundcard as sc

def test_sherpa_online_fixed3():
    """测试修复后的Sherpa-ONNX在线转录功能（文本拼接问题修复）"""
    print("测试修复后的Sherpa-ONNX在线转录功能（文本拼接问题修复）...")

    # 初始化日志工具
    sherpa_logger.setup()
    sherpa_log_file = sherpa_logger.get_log_file()
    print(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")

    # 创建 ASRModelManager 实例
    model_manager = ASRModelManager()

    # 加载 Sherpa-ONNX int8 模型
    model_name = "sherpa_int8"
    print(f"加载模型: {model_name}")
    if not model_manager.load_model(model_name):
        print(f"加载模型 {model_name} 失败")
        return

    # 检查模型类型
    print(f"加载后的模型类型: {model_manager.model_type}")

    # 创建识别器
    print("创建识别器...")
    recognizer = model_manager.create_recognizer()
    if not recognizer:
        print("创建识别器失败")
        return

    print(f"识别器类型: {type(recognizer).__name__}")
    print(f"识别器引擎类型: {recognizer.engine_type}")

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

    # 创建转录结果文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    transcript_file = os.path.join(project_root, "transcripts", f"transcript_test_{timestamp}.txt")
    os.makedirs(os.path.dirname(transcript_file), exist_ok=True)

    # 捕获音频
    try:
        with sc.get_microphone(id=str(default_device.id), include_loopback=True).recorder(samplerate=sample_rate) as mic:
            print("开始捕获音频...")
            print("请播放音频，按 Ctrl+C 停止...")

            # 循环捕获音频
            try:
                while True:
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
                        # 使用 recognizer.AcceptWaveform 处理音频数据
                        if recognizer.AcceptWaveform(data):
                            # 获取结果
                            result_json = recognizer.Result()
                            import json
                            result = json.loads(result_json)
                            if "text" in result and result["text"]:
                                print(f"\n转录结果: {result['text']}")
                                # 保存到文件
                                with open(transcript_file, "a", encoding="utf-8") as f:
                                    f.write(f"[完整结果] {result['text']}\n")

                                # 检查 previous_result 变量
                                if hasattr(recognizer, 'previous_result'):
                                    print(f"previous_result: {recognizer.previous_result}")
                                    # 保存到文件
                                    with open(transcript_file, "a", encoding="utf-8") as f:
                                        f.write(f"[previous_result] {recognizer.previous_result}\n")
                        else:
                            # 获取部分结果
                            partial_json = recognizer.PartialResult()
                            import json
                            partial = json.loads(partial_json)
                            if "partial" in partial and partial["partial"]:
                                print(f"\r部分结果: {partial['partial']}", end="", flush=True)
                    except Exception as e:
                        print(f"\n处理音频数据错误: {e}")
                        import traceback
                        print(traceback.format_exc())

                    # 等待一段时间
                    time.sleep(0.1)

            except KeyboardInterrupt:
                print("\n捕获音频已停止")

                # 获取最终结果
                final_result_json = recognizer.FinalResult()
                import json
                final_result = json.loads(final_result_json)
                if "text" in final_result and final_result["text"]:
                    print(f"\n最终转录结果: {final_result['text']}")
                    # 保存到文件
                    with open(transcript_file, "a", encoding="utf-8") as f:
                        f.write(f"[最终结果] {final_result['text']}\n")

    except Exception as e:
        print(f"捕获音频错误: {e}")
        import traceback
        print(traceback.format_exc())

    print(f"测试完成，转录结果保存在: {transcript_file}")
    print(f"日志文件: {sherpa_log_file}")

if __name__ == "__main__":
    test_sherpa_online_fixed3()
