"""
测试 Sherpa-ONNX 在线转录
"""
import os
import sys
import numpy as np
import soundcard as sc
import time

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

def test_sherpa_onnx_online():
    """测试 Sherpa-ONNX 在线转录"""
    print("测试 Sherpa-ONNX 在线转录...")
    
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
    print("可用的音频设备:")
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
            print("开始捕获音频...")
            print("请播放音频，按 Ctrl+C 停止...")
            
            # 循环捕获音频
            start_time = time.time()
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
                    
                    # 创建流
                    stream = recognizer.create_stream()
                    
                    # 处理音频数据
                    try:
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
                        text = recognizer.get_result(stream)
                        if text:
                            print(f"\n转录结果: {text}")
                    except Exception as e:
                        print(f"\n处理音频数据错误: {e}")
                        import traceback
                        print(traceback.format_exc())
                    
                    # 等待一段时间
                    time.sleep(0.1)
            
            except KeyboardInterrupt:
                print("\n捕获音频已停止")
    
    except Exception as e:
        print(f"捕获音频错误: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_sherpa_onnx_online()
