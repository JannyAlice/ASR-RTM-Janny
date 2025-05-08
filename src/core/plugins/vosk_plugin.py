"""Vosk ASR 插件实现"""
import os
import json  # 添加此行
import logging
import traceback
import numpy as np
from typing import Optional, Dict, Any, Union
import vosk

from .asr_plugin_base import ASRPluginBase

logger = logging.getLogger(__name__)

class VoskPlugin(ASRPluginBase):
    """Vosk ASR 插件实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.engine_type = "vosk_small"
        
    def setup(self) -> bool:
        """初始化 Vosk 模型"""
        try:
            model_path = self.config.get("model_path")
            if not model_path or not os.path.exists(model_path):
                logger.error(f"Vosk 模型路径无效: {model_path}")
                return False
                
            self.model_dir = model_path
            self.model = vosk.Model(model_path)
            logger.info(f"Vosk 模型加载成功: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Vosk 模型加载失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    def create_recognizer(self):
        """创建 Vosk 识别器"""
        if not self.model:
            logger.error("Vosk 模型未初始化")
            return None
            
        try:
            self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
            return self.recognizer
        except Exception as e:
            logger.error(f"创建识别器失败: {str(e)}")
            return None
            
    def transcribe(self, audio_data: Union[bytes, np.ndarray]) -> Optional[str]:
        """转录音频数据"""
        if not self.recognizer:
            logger.error("识别器未初始化")
            return None
            
        try:
            # 如果输入是numpy数组,转换为bytes
            if isinstance(audio_data, np.ndarray):
                audio_data = audio_data.tobytes()
                
            if self.recognizer.AcceptWaveform(audio_data):
                result = self.recognizer.Result()
                # 解析JSON结果获取文本
                text = json.loads(result).get("text", "")
                return text
            return None
            
        except Exception as e:
            logger.error(f"转录失败: {str(e)}")
            return None
            
    def transcribe_file(self, file_path: str) -> Optional[str]:
        """转录音频文件"""
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return None
            
        try:
            import wave
            with wave.open(file_path, "rb") as wf:
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                    logger.error("不支持的音频格式,需要16bit PCM WAV")
                    return None
                    
                recognizer = self.create_recognizer()
                if not recognizer:
                    return None
                    
                results = []
                while True:
                    data = wf.readframes(4000)  # 读取音频块
                    if len(data) == 0:
                        break
                        
                    if recognizer.AcceptWaveform(data):
                        result = recognizer.Result()
                        text = json.loads(result).get("text", "")
                        if text:
                            results.append(text)
                            
                # 获取最后的结果
                final_result = recognizer.FinalResult()
                final_text = json.loads(final_result).get("text", "")
                if final_text:
                    results.append(final_text)
                    
                return " ".join(results)
                
        except Exception as e:
            logger.error(f"转录文件失败: {str(e)}")
            return None
            
    def validate_files(self) -> bool:
        """验证模型文件完整性"""
        if not self.model_dir or not os.path.exists(self.model_dir):
            return False
            
        # 检查必要的文件
        required_files = [
            "final.mdl",
            "conf/mfcc.conf",
            "conf/model.conf",
            "graph/phones.txt",
            "graph/words.txt",
            "am/final.mdl",
            "graph/HCLG.fst"
        ]
        
        for file in required_files:
            file_path = os.path.join(self.model_dir, file)
            if not os.path.exists(file_path):
                logger.error(f"缺少必要文件: {file}")
                return False
                
        return True