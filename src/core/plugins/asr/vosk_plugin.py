import os
import json
import numpy as np
import traceback
from typing import Dict, Any, Optional
from vosk import Model, KaldiRecognizer
from .asr_plugin import ASRPlugin
import logging

logger = logging.getLogger(__name__)

class VoskPluginError(Exception):
    """Vosk插件基础异常类"""
    pass

class ConfigurationError(VoskPluginError):
    """配置错误"""
    pass

class ModelLoadError(VoskPluginError):
    """模型加载错误"""
    pass

class AudioProcessError(VoskPluginError):
    """音频处理错误"""
    pass

class TranscriptionError(VoskPluginError):
    """转录错误"""
    pass

class VoskPlugin(ASRPlugin):
    """Vosk ASR 插件实现"""
    
    plugin_type = "asr"  # 插件类型
    
    def __init__(self):
        super().__init__()
        self.model_type = "standard"  # 改为与配置文件一致
        self._model = None
        self._recognizer = None
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化插件"""
        try:
            logger.info("Initializing Vosk plugin with config: %s", json.dumps(config, indent=2))
            
            if not config:
                raise ConfigurationError("No configuration provided")
                
            from src.utils.config_manager import config_manager
            model_config_name = config.get("model_config")
            if not model_config_name:
                raise ConfigurationError("No model configuration name provided")
                
            model_config = config_manager.get_config("asr", "models", model_config_name)
            if not model_config:
                raise ConfigurationError(f"Model configuration not found: {model_config_name}")
                
            plugin_config = config.get("plugin_config", {})
            model_config["config"].update(plugin_config)
            
            self._config = model_config
            logger.info("Configuration loaded successfully")
            
            return self.setup()
            
        except VoskPluginError as e:
            logger.error("Configuration error: %s", str(e))
            return False
        except Exception as e:
            logger.error("Unexpected error in initialize: %s", str(e))
            logger.error(traceback.format_exc())
            return False
            
    def setup(self) -> bool:
        """设置 Vosk ASR 引擎"""
        try:
            model_path = self._config.get("path")
            if not model_path:
                raise ModelLoadError("Model path not configured")
                
            if not os.path.exists(model_path):
                raise ModelLoadError(f"Model path not found: {model_path}")
                
            config = self._config.get("config", {})
            self._sample_rate = config.get("sample_rate", 16000)
            
            logger.info("Loading Vosk model from: %s", model_path)
            self.model = Model(model_path)
            self.recognizer = self.create_recognizer()
            
            if not self.recognizer:
                raise ModelLoadError("Failed to create recognizer")
                
            logger.info("Successfully initialized Vosk ASR with model: %s", model_path)
            return True
            
        except VoskPluginError as e:
            logger.error("Model load error: %s", str(e))
            return False
        except Exception as e:
            logger.error("Unexpected error in setup: %s", str(e))
            logger.error(traceback.format_exc())
            return False
            
    def create_recognizer(self) -> Optional[KaldiRecognizer]:
        """创建新的识别器实例"""
        try:
            if not self.model:
                raise ModelLoadError("Model not initialized")
            
            config = self._config.get("config", {})
            use_words = config.get("use_words", True)
            
            recognizer = KaldiRecognizer(self.model, self._sample_rate)
            recognizer.SetWords(use_words)
            
            logger.info("Created Vosk recognizer with sample_rate=%d, use_words=%s", 
                       self._sample_rate, use_words)
            return recognizer
            
        except VoskPluginError as e:
            logger.error("Failed to create recognizer: %s", str(e))
            return None
        except Exception as e:
            logger.error("Unexpected error creating recognizer: %s", str(e))
            logger.error(traceback.format_exc())
            return None
            
    def process_audio(self, audio_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """处理音频数据"""
        try:
            if not self.recognizer:
                raise AudioProcessError("Recognizer not initialized")
                
            if len(audio_data) == 0:
                return None
                
            # 转换为16位整数字节
            audio_data = (audio_data * 32768).astype(np.int16).tobytes()
            
            if self.recognizer.AcceptWaveform(audio_data):
                result = json.loads(self.recognizer.Result())
                logger.debug("Got final result: %s", result)
                if "text" in result:
                    return {
                        "text": result["text"],
                        "is_final": True
                    }
            else:
                result = json.loads(self.recognizer.PartialResult())
                logger.debug("Got partial result: %s", result)
                if "partial" in result:
                    return {
                        "text": result["partial"],
                        "is_partial": True
                    }
            return None
            
        except VoskPluginError as e:
            logger.error("Audio processing error: %s", str(e))
            return None
        except Exception as e:
            logger.error("Unexpected error in process_audio: %s", str(e))
            logger.error(traceback.format_exc())
            return None
            
    def transcribe_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """转录音频文件"""
        try:
            if not os.path.exists(file_path):
                raise TranscriptionError(f"File not found: {file_path}")
                
            if not self.recognizer:
                raise TranscriptionError("Recognizer not initialized")
            
            logger.info("Starting transcription of file: %s", file_path)
            
            import wave
            wf = wave.open(file_path, "rb")
            
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                raise TranscriptionError("Audio file must be WAV format mono PCM")
                
            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    if result["text"]:
                        results.append(result["text"])
                        
            final_result = json.loads(self.recognizer.FinalResult())
            if final_result["text"]:
                results.append(final_result["text"])
                
            text = " ".join(results)
            logger.info("Completed transcription, total length: %d characters", len(text))
            
            return {"text": text, "is_final": True}
            
        except VoskPluginError as e:
            logger.error("Transcription error: %s", str(e))
            return None
        except Exception as e:
            logger.error("Unexpected error in transcribe_file: %s", str(e))
            logger.error(traceback.format_exc())
            return None
            
    def get_info(self) -> Dict[str, str]:
        """获取插件信息"""
        return {
            "id": "vosk_small",
            "name": "VOSK Small Model",
            "type": "ASR",
            "description": "VOSK Small Model for Speech Recognition",
            "version": "1.0.0"
        }
        
    def cleanup(self) -> None:
        """清理资源"""
        if self.recognizer:
            self.recognizer = None
        if self.model:
            self.model = None
        logger.info("Cleaned up Vosk plugin resources")
        super().cleanup()