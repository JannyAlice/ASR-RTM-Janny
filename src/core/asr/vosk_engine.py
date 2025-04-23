import os
import json
import numpy as np
from typing import Optional, Union
from vosk import Model, KaldiRecognizer


class VoskASR:
    """VOSK ASR 引擎封装类"""
    
    def __init__(self, model_path: str):
        """初始化 VOSK ASR 引擎
        
        Args:
            model_path: VOSK 模型路径
        """
        self.model_path = model_path
        self.model = None
        self.recognizer = None
        self.sample_rate = 16000
        
    def setup(self) -> bool:
        """设置 VOSK ASR 引擎
        
        Returns:
            bool: 是否设置成功
        """
        try:
            if not os.path.exists(self.model_path):
                print(f"VOSK model path not found: {self.model_path}")
                return False
                
            self.model = Model(self.model_path)
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)
            return True
            
        except Exception as e:
            print(f"Error setting up VOSK ASR: {str(e)}")
            return False
            
    def transcribe(self, audio_data: Union[bytes, np.ndarray]) -> Optional[str]:
        """转录音频数据
        
        Args:
            audio_data: 音频数据，可以是字节或 numpy 数组
            
        Returns:
            str: 转录文本，如果失败则返回 None
        """
        if not self.recognizer:
            return None
            
        try:
            # 确保音频数据是字节类型
            if isinstance(audio_data, np.ndarray):
                audio_data = (audio_data * 32767).astype(np.int16).tobytes()
                
            if self.recognizer.AcceptWaveform(audio_data):
                result = json.loads(self.recognizer.Result())
                return result.get("text", "")
            return None
            
        except Exception as e:
            print(f"Error in VOSK transcription: {str(e)}")
            return None
            
    def reset(self) -> None:
        """重置识别器状态"""
        if self.recognizer:
            self.recognizer.Reset()
            
    def get_final_result(self) -> Optional[str]:
        """获取最终识别结果
        
        Returns:
            str: 最终识别文本，如果失败则返回 None
        """
        try:
            if self.recognizer:
                result = json.loads(self.recognizer.FinalResult())
                return result.get("text", "")
            return None
        except Exception as e:
            print(f"Error getting VOSK final result: {str(e)}")
            return None
