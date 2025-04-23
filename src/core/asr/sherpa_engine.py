import os
import numpy as np
from typing import Optional
import sherpa_onnx


class SherpaOnnxASR:
    """Sherpa-ONNX ASR 引擎实现"""
    def __init__(self, model_dir: str = "models/sherpa-onnx"):
        """
        初始化 Sherpa-ONNX ASR 引擎
        
        Args:
            model_dir: 模型目录路径
        """
        self.model_dir = model_dir
        self.recognizer = None
        self.stream = None
        self.config = None
        self.setup()
    
    def setup(self) -> bool:
        """
        初始化 Sherpa-ONNX ASR
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            # 检查模型文件
            required_files = {
                "encoder": "encoder-epoch-99-avg-1.int8.onnx",
                "decoder": "decoder-epoch-99-avg-1.int8.onnx",
                "joiner": "joiner-epoch-99-avg-1.int8.onnx",
                "tokens": "tokens.txt"
            }
            
            # 验证所有必需文件
            for file_type, file_name in required_files.items():
                file_path = os.path.join(self.model_dir, file_name)
                if not os.path.exists(file_path):
                    print(f"错误: 缺少{file_type}模型文件: {file_path}")
                    return False
            
            # 配置识别器
            self.config = {
                "encoder": os.path.join(self.model_dir, "encoder-epoch-99-avg-1.int8.onnx"),
                "decoder": os.path.join(self.model_dir, "decoder-epoch-99-avg-1.int8.onnx"),
                "joiner": os.path.join(self.model_dir, "joiner-epoch-99-avg-1.int8.onnx"),
                "tokens": os.path.join(self.model_dir, "tokens.txt"),
                "num_threads": 1,
                "sample_rate": 16000,
                "feature_dim": 80,
                "decoding_method": "greedy_search",
                "debug": False
            }
            
            # 创建识别器和流
            self.recognizer = sherpa_onnx.OnlineRecognizer(self.config)
            self.stream = self.recognizer.create_stream()
            
            print("Sherpa-ONNX ASR 初始化成功")
            return True
            
        except Exception as e:
            print(f"Sherpa-ONNX ASR 初始化失败: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        转录音频数据
        
        Args:
            audio_data: 音频数据，numpy数组格式
            
        Returns:
            str: 识别结果文本
        """
        try:
            if not self.recognizer or not self.stream:
                return ""
            
            # 确保音频数据是单声道
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # 处理音频数据
            self.stream.accept_waveform(16000, audio_data)
            self.recognizer.decode_stream(self.stream)
            
            # 获取结果
            text = self.stream.get_text()
            if text:
                return text
            
            return ""
            
        except Exception as e:
            print(f"Sherpa-ONNX 转录错误: {e}")
            return ""
    
    def reset(self) -> None:
        """重置识别器状态"""
        try:
            if self.recognizer:
                self.stream = self.recognizer.create_stream()
        except Exception as e:
            print(f"重置 Sherpa-ONNX 识别器错误: {e}")
    
    def get_final_result(self) -> str:
        """
        获取最终结果
        
        Returns:
            str: 最终识别结果文本
        """
        try:
            if not self.recognizer or not self.stream:
                return ""
            
            # 强制解码剩余音频
            self.recognizer.decode_stream(self.stream)
            return self.stream.get_text()
            
        except Exception as e:
            print(f"获取 Sherpa-ONNX 最终结果错误: {e}")
            return ""
    
    def __del__(self):
        """清理资源"""
        self.recognizer = None
        self.stream = None
