"""
测试 Sherpa-ONNX 在线转录（结合直接使用和 ASRModelManager）
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

# 导入 ASRModelManager
from src.core.asr.model_manager import ASRModelManager

def test_sherpa_onnx_combined():
    """测试 Sherpa-ONNX 在线转录（结合直接使用和 ASRModelManager）"""
    print("测试 Sherpa-ONNX 在线转录（结合直接使用和 ASRModelManager）...")
    
    # 初始化日志工具
    sherpa_logger.setup()
    sherpa_log_file = sherpa_logger.get_log_file()
    print(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")
    
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
        if not os.path.exists(file_path):
            sherpa_logger.error(f"模型文件不存在: {file_path}")
            return
    
    # 方法1：直接使用 sherpa_onnx 模块
    print("\n=== 方法1：直接使用 sherpa_onnx 模块 ===")
    try:
        recognizer1 = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=os.path.join(model_path, encoder_file),
            decoder=os.path.join(model_path, decoder_file),
            joiner=os.path.join(model_path, joiner_file),
            tokens=os.path.join(model_path, tokens_file),
            num_threads=4,
            decoding_method="greedy_search",
            provider="cpu",
        )
        print("成功创建 OnlineRecognizer（方法1）")
    except Exception as e:
        print(f"创建 OnlineRecognizer（方法1）失败: {e}")
        import traceback
        print(traceback.format_exc())
        return
    
    # 方法2：使用 ASRModelManager
    print("\n=== 方法2：使用 ASRModelManager ===")
    model_manager = ASRModelManager()
    model_name = "sherpa_int8"
    print(f"加载模型: {model_name}")
    if not model_manager.load_model(model_name):
        print(f"加载模型 {model_name} 失败")
        return
    
    print("创建识别器...")
    recognizer2 = model_manager.create_recognizer()
    if not recognizer2:
        print("创建识别器失败")
        return
    
    print(f"识别器类型: {type(recognizer2).__name__}")
    
    # 获取音频设备
    print("\n=== 获取音频设备 ===")
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
        with sc.get_microphone(id=str(default_device.id), include_loopback=True).recorder(samplerate=sample_rate) as mic:
            print("\n=== 开始捕获音频 ===")
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
                    
                    # 方法1：直接使用 sherpa_onnx 模块
                    try:
                        # 为每个音频块创建一个新的流
                        stream = recognizer1.create_stream()
                        
                        # 1. 接受音频数据
                        stream.accept_waveform(sample_rate, data)
                        
                        # 2. 添加尾部填充
                        tail_paddings = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
                        stream.accept_waveform(sample_rate, tail_paddings)
                        
                        # 3. 标记输入结束
                        stream.input_finished()
                        
                        # 4. 解码
                        decode_count = 0
                        while recognizer1.is_ready(stream):
                            recognizer1.decode_stream(stream)
                            decode_count += 1
                        
                        # 5. 获取结果
                        text1 = recognizer1.get_result(stream)
                        
                        if text1:
                            # 过滤掉非英文字符
                            import re
                            filtered_text1 = re.sub(r'[^\x00-\x7F]+', '', text1)
                            if filtered_text1.strip():
                                print(f"\n方法1转录结果: {filtered_text1}")
                    except Exception as e:
                        print(f"\n方法1处理音频数据错误: {e}")
                        import traceback
                        print(traceback.format_exc())
                    
                    # 方法2：使用 ASRModelManager
                    try:
                        # 使用 recognizer.AcceptWaveform 处理音频数据
                        if recognizer2.AcceptWaveform(data):
                            # 获取结果
                            result_json = recognizer2.Result()
                            import json
                            result = json.loads(result_json)
                            if "text" in result and result["text"]:
                                print(f"\n方法2转录结果: {result['text']}")
                    except Exception as e:
                        print(f"\n方法2处理音频数据错误: {e}")
                        import traceback
                        print(traceback.format_exc())
                    
                    # 等待一段时间
                    time.sleep(0.1)
            
            except KeyboardInterrupt:
                print("\n捕获音频已停止")
                
                # 方法2：获取最终结果
                final_result_json = recognizer2.FinalResult()
                import json
                final_result = json.loads(final_result_json)
                if "text" in final_result and final_result["text"]:
                    print(f"\n方法2最终转录结果: {final_result['text']}")
    
    except Exception as e:
        print(f"捕获音频错误: {e}")
        import traceback
        print(traceback.format_exc())
    
    print(f"测试完成，日志文件: {sherpa_log_file}")

if __name__ == "__main__":
    test_sherpa_onnx_combined()
