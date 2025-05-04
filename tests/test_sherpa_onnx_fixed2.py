"""
测试修复后的 Sherpa-ONNX 模型
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

from src.core.asr.sherpa_engine import SherpaOnnxASR

def test_sherpa_onnx_model():
    """测试 Sherpa-ONNX 模型的加载和识别"""
    print("开始测试 Sherpa-ONNX 模型...")
    
    # 模型路径
    model_path = "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\models\\asr\\sherpa-onnx"
    
    # 测试 int8 模型
    print("\n=== 测试 Sherpa-ONNX int8 模型 ===")
    
    # 创建模型配置
    model_config = {
        "type": "int8",
        "config": {
            "num_threads": 4,
            "sample_rate": 16000,
            "feature_dim": 80,
            "decoding_method": "greedy_search"
        }
    }
    
    # 创建 SherpaOnnxASR 实例
    asr = SherpaOnnxASR(model_path, model_config)
    
    # 初始化
    if asr.setup():
        print("成功初始化 Sherpa-ONNX int8 模型")
        
        # 生成测试音频数据（1秒的静音）
        print("生成测试音频数据...")
        sample_rate = 16000
        audio_data = np.zeros(sample_rate, dtype=np.float32)
        
        # 处理音频数据
        print("处理音频数据...")
        result = asr.transcribe(audio_data)
        print(f"识别结果: {result}")
        
        # 测试最终结果
        print("获取最终结果...")
        final_result = asr.get_final_result()
        print(f"最终结果: {final_result}")
        
        print("测试成功")
    else:
        print("初始化 Sherpa-ONNX int8 模型失败")

if __name__ == "__main__":
    test_sherpa_onnx_model()
