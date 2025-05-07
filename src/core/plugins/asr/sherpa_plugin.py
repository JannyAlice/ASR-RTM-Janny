import os
import json
import numpy as np
from typing import Dict, Any, Optional
import sherpa_onnx

from .asr_plugin import ASRPlugin
import logging

logger = logging.getLogger(__name__)

class SherpaPlugin(ASRPlugin):
    """Sherpa ONNX ASR 插件实现"""
    
    def __init__(self):
        super().__init__()
        self.recognizer = None
        self._stream = None
        
    def setup(self) -> bool:
        """设置 Sherpa ONNX ASR 引擎"""
        try:
            model_path = self._config.get("model_path")
            if not model_path:
                logger.error("Model path not configured")
                return False
                
            # 获取配置参数
            config = self._config.get("config", {})
            
            # 创建识别器配置
            recognizer_config = sherpa_onnx.FeatureConfig(
                sample_rate=config.get("sample_rate", 16000),
                feature_dim=config.get("feature_dim", 80),
            )
            
            # 创建解码器配置
            decoder_config = sherpa_onnx.DecoderConfig(
                decoder_type="greedy_search",
                max_active_paths=config.get("max_active_paths", 4)
            )
            
            # 创建模型配置
            model_config = sherpa_onnx.OnlineModelConfig(
                encoder=os.path.join(model_path, "encoder.onnx"),
                decoder=os.path.join(model_path, "decoder.onnx"),
                joiner=os.path.join(model_path, "joiner.onnx"),
                tokens=os.path.join(model_path, "tokens.txt"),
                num_threads=config.get("num_threads", 1),
                debug=config.get("debug", False)
            )
            
            # 创建识别器
            self.recognizer = sherpa_onnx.OnlineRecognizer(
                model_config=model_config,
                feature_config=recognizer_config,
                decoder_config=decoder_config,
                enable_endpoint_detection=config.get("enable_endpoint_detection", True),
                rule1_min_trailing_silence=config.get("rule1_min_trailing_silence", 2.4),
                rule2_min_trailing_silence=config.get("rule2_min_trailing_silence", 1.2),
                rule3_min_utterance_length=config.get("rule3_min_utterance_length", 20.0)
            )
            
            logger.info(f"Successfully initialized Sherpa ONNX ASR with model: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up Sherpa ONNX ASR: {str(e)}")
            return False
            
    def process_audio(self, audio_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """处理音频数据"""
        if not self.recognizer:
            return None
            
        try:
            # 创建新的音频流
            if self._stream is None:
                self._stream = self.recognizer.create_stream()
                
            # 处理音频数据
            self._stream.accept_waveform(audio_data)
            
            # 获取识别结果
            self.recognizer.decode_stream(self._stream)
            text = self._stream.result.text.strip()
            
            if text:
                return {
                    "text": text,
                    "is_final": self._stream.result.is_final
                }
            return None
            
        except Exception as e:
            logger.error(f"Error in Sherpa ONNX audio processing: {str(e)}")
            return None
            
    def transcribe_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """转录音频文件"""
        if not self.recognizer:
            return None
            
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
                
            # 只处理 WAV 文件
            if not file_path.lower().endswith('.wav'):
                logger.error(f"File is not a WAV file: {file_path}")
                return None
                
            import wave
            with wave.open(file_path, 'rb') as wf:
                # 检查采样率
                if wf.getframerate() != self._config.get("config", {}).get("sample_rate", 16000):
                    logger.error(f"Sample rate mismatch")
                    return None
                    
                # 创建新的音频流
                stream = self.recognizer.create_stream()
                
                # 分块读取和处理音频
                chunk_size = 4000
                results = []
                
                while True:
                    frames = wf.readframes(chunk_size)
                    if not frames:
                        break
                        
                    # 转换为numpy数组
                    audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # 处理音频数据
                    stream.accept_waveform(audio_data)
                    self.recognizer.decode_stream(stream)
                    
                    if stream.result.text.strip():
                        results.append({
                            "text": stream.result.text.strip(),
                            "is_final": stream.result.is_final
                        })
                        
                # 处理最后的音频数据
                stream.input_finished()
                self.recognizer.decode_stream(stream)
                
                final_text = stream.result.text.strip()
                if final_text:
                    results.append({
                        "text": final_text,
                        "is_final": True
                    })
                    
                # 合并所有结果
                combined_text = " ".join(r["text"] for r in results)
                if combined_text:
                    # 格式化文本
                    if len(combined_text) > 0:
                        combined_text = combined_text[0].upper() + combined_text[1:]
                    if combined_text[-1] not in ['.', '?', '!', ',', ';', ':', '-']:
                        combined_text += '.'
                        
                    return {
                        "text": combined_text,
                        "segments": results
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error in Sherpa ONNX file transcription: {str(e)}")
            return None
            
    def get_info(self) -> Dict[str, str]:
        """获取插件信息"""
        return {
            "id": "sherpa_0626_std",
            "name": "Sherpa 0626 STD Model",
            "type": "ASR",
            "description": "Sherpa ONNX Standard Model for Speech Recognition",
            "version": "1.0.0"
        }
        
    def cleanup(self) -> None:
        """清理资源"""
        super().cleanup()
        self._stream = None
        self.recognizer = None