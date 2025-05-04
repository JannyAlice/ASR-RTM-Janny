"""
测试 Sherpa-ONNX 模型
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

from src.core.asr.model_manager import ASRModelManager
from src.utils.config_manager import config_manager

def test_sherpa_onnx_model():
    """测试 Sherpa-ONNX 模型加载和识别"""
    print("开始测试 Sherpa-ONNX 模型...")
    
    # 创建模型管理器
    model_manager = ASRModelManager()
    
    # 测试 int8 模型
    print("\n=== 测试 Sherpa-ONNX int8 模型 ===")
    if model_manager.load_model("sherpa_int8"):
        print("成功加载 Sherpa-ONNX int8 模型")
        
        # 创建识别器
        recognizer = model_manager.create_recognizer()
        if recognizer:
            print("成功创建 Sherpa-ONNX int8 识别器")
            
            # 生成测试音频数据（1秒的静音）
            sample_rate = 16000
            audio_data = np.zeros(sample_rate, dtype=np.float32)
            audio_data = (audio_data * 32768).astype(np.int16).tobytes()
            
            # 测试识别
            print("测试识别...")
            recognizer.AcceptWaveform(audio_data)
            result = recognizer.Result()
            print(f"识别结果: {result}")
            
            # 测试最终结果
            final_result = recognizer.FinalResult()
            print(f"最终结果: {final_result}")
        else:
            print("创建 Sherpa-ONNX int8 识别器失败")
    else:
        print("加载 Sherpa-ONNX int8 模型失败")
    
    # 测试标准模型
    print("\n=== 测试 Sherpa-ONNX 标准模型 ===")
    if model_manager.load_model("sherpa_std"):
        print("成功加载 Sherpa-ONNX 标准模型")
        
        # 创建识别器
        recognizer = model_manager.create_recognizer()
        if recognizer:
            print("成功创建 Sherpa-ONNX 标准识别器")
            
            # 生成测试音频数据（1秒的静音）
            sample_rate = 16000
            audio_data = np.zeros(sample_rate, dtype=np.float32)
            audio_data = (audio_data * 32768).astype(np.int16).tobytes()
            
            # 测试识别
            print("测试识别...")
            recognizer.AcceptWaveform(audio_data)
            result = recognizer.Result()
            print(f"识别结果: {result}")
            
            # 测试最终结果
            final_result = recognizer.FinalResult()
            print(f"最终结果: {final_result}")
        else:
            print("创建 Sherpa-ONNX 标准识别器失败")
    else:
        print("加载 Sherpa-ONNX 标准模型失败")

if __name__ == "__main__":
    test_sherpa_onnx_model()
