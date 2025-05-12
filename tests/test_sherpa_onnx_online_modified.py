"""
测试修改后的 Sherpa-ONNX 在线转录功能（使用更大的缓冲区和更长的尾部填充）
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

# 导入音频捕获工具
import soundcard as sc

def test_sherpa_onnx_online_modified():
    """测试修改后的 Sherpa-ONNX 在线转录功能"""
    print("测试修改后的 Sherpa-ONNX 在线转录功能（使用更大的缓冲区和更长的尾部填充）...")
    
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
    buffer_size = 8000  # 500ms，更大的缓冲区
    
    # 捕获音频
    try:
        with sc.get_microphone(id=str(default_device.id), include_loopback=True).recorder(samplerate=sample_rate) as mic:
            print("开始捕获音频...")
            print("请播放音频，按 Ctrl+C 停止...")
            
            # 循环捕获音频
            try:
                # 创建一个持久的流
                stream = recognizer.create_stream()
                print("创建持久的流")
                
                while True:
                    # 捕获音频数据
                    data = mic.record(numframes=buffer_size)
                    
                    # 转换为单声道
                    if data.shape[1] > 1:
                        data = np.mean(data, axis=1)
                    
                    # 记录音频数据信息
                    print(f"音频数据形状: {data.shape}, 最大值: {np.max(np.abs(data))}")
                    
                    # 检查数据是否有效
                    if np.max(np.abs(data)) < 0.01:
                        print(".", end="", flush=True)
                        continue
                    
                    # 处理音频数据
                    try:
                        # 接受音频数据
                        stream.accept_waveform(sample_rate, data)
                        print("接受音频数据成功")
                        
                        # 解码
                        decode_count = 0
                        while recognizer.is_ready(stream):
                            recognizer.decode_stream(stream)
                            decode_count += 1
                        print(f"解码完成，解码次数: {decode_count}")
                        
                        # 获取结果
                        text = recognizer.get_result(stream)
                        print(f"获取结果: '{text}'")
                        
                        if text:
                            # 过滤掉非英文字符
                            import re
                            filtered_text = re.sub(r'[^\x00-\x7F]+', '', text)
                            if filtered_text.strip():
                                print(f"\n转录结果: {filtered_text}")
                    except Exception as e:
                        print(f"\n处理音频数据错误: {e}")
                        import traceback
                        print(traceback.format_exc())
                    
                    # 等待一段时间
                    time.sleep(0.1)
            
            except KeyboardInterrupt:
                print("\n捕获音频已停止")
                
                # 添加尾部填充并标记输入结束
                tail_paddings = np.zeros(int(0.5 * sample_rate), dtype=np.float32)  # 更长的尾部填充
                stream.accept_waveform(sample_rate, tail_paddings)
                stream.input_finished()
                
                # 解码
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
                
                # 获取最终结果
                final_text = recognizer.get_result(stream)
                if final_text:
                    # 过滤掉非英文字符
                    import re
                    filtered_text = re.sub(r'[^\x00-\x7F]+', '', final_text)
                    if filtered_text.strip():
                        print(f"\n最终转录结果: {filtered_text}")
    
    except Exception as e:
        print(f"捕获音频错误: {e}")
        import traceback
        print(traceback.format_exc())
    
    print(f"测试完成，日志文件: {sherpa_log_file}")

if __name__ == "__main__":
    test_sherpa_onnx_online_modified()
