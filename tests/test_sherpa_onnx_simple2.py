"""
测试 Sherpa-ONNX 模型的简单加载和识别（使用新的 API）
"""
import os
import sys
import json
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

def test_sherpa_onnx_model():
    """测试 Sherpa-ONNX 模型的加载和识别"""
    print("开始测试 Sherpa-ONNX 模型...")
    
    # 模型路径
    model_path = "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\models\\asr\\sherpa-onnx"
    
    # 测试 int8 模型
    print("\n=== 测试 Sherpa-ONNX int8 模型 ===")
    
    # 确定模型文件名
    encoder_file = "encoder-epoch-99-avg-1.int8.onnx"
    decoder_file = "decoder-epoch-99-avg-1.int8.onnx"
    joiner_file = "joiner-epoch-99-avg-1.int8.onnx"
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
    
    try:
        # 尝试使用 from_transducer 方法创建识别器
        print("尝试使用 from_transducer 方法创建识别器...")
        try:
            recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                encoder=os.path.join(model_path, encoder_file),
                decoder=os.path.join(model_path, decoder_file),
                joiner=os.path.join(model_path, joiner_file),
                tokens=os.path.join(model_path, tokens_file),
                num_threads=4,
                sample_rate=16000,
                feature_dim=80,
                decoding_method="greedy_search"
            )
            print("成功使用 from_transducer 创建识别器")
        except Exception as e:
            print(f"使用 from_transducer 创建识别器失败: {e}")
            
            # 尝试使用 from_conformer 方法创建识别器
            print("\n尝试使用 from_conformer 方法创建识别器...")
            try:
                recognizer = sherpa_onnx.OnlineRecognizer.from_conformer(
                    encoder=os.path.join(model_path, encoder_file),
                    decoder=os.path.join(model_path, decoder_file),
                    joiner=os.path.join(model_path, joiner_file),
                    tokens=os.path.join(model_path, tokens_file),
                    num_threads=4,
                    sample_rate=16000,
                    feature_dim=80,
                    decoding_method="greedy_search"
                )
                print("成功使用 from_conformer 创建识别器")
            except Exception as e:
                print(f"使用 from_conformer 创建识别器失败: {e}")
                
                # 尝试直接创建识别器
                print("\n尝试直接创建识别器...")
                try:
                    recognizer = sherpa_onnx.OnlineRecognizer()
                    print("成功直接创建识别器")
                except Exception as e:
                    print(f"直接创建识别器失败: {e}")
                    return
        
        # 创建流
        print("创建流...")
        stream = recognizer.create_stream()
        print("成功创建流")
        
        # 生成测试音频数据（1秒的静音）
        print("生成测试音频数据...")
        sample_rate = 16000
        audio_data = np.zeros(sample_rate, dtype=np.float32)
        
        # 处理音频数据
        print("处理音频数据...")
        stream.accept_waveform(sample_rate, audio_data)
        recognizer.decode_stream(stream)
        
        # 获取结果
        print("获取结果...")
        result = recognizer.get_result(stream)
        print(f"识别结果: {result}")
        
        # 测试最终结果
        print("获取最终结果...")
        stream.input_finished()
        recognizer.decode_stream(stream)
        final_result = recognizer.get_result(stream)
        print(f"最终结果: {final_result}")
        
        print("测试成功")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_sherpa_onnx_model()
