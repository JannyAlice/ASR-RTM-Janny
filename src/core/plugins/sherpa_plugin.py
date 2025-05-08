"""Sherpa-ONNX ASR 插件实现"""
import os
import json
import logging
import traceback
import numpy as np
from typing import Optional, Dict, Any, Union

from .asr_plugin_base import ASRPluginBase

# 初始化日志记录器
logger = logging.getLogger(__name__)

try:
    import sherpa_onnx
    HAS_SHERPA_ONNX = True
except ImportError:
    HAS_SHERPA_ONNX = False
    logger.warning("警告: 未安装 sherpa_onnx 模块，Sherpa-ONNX 功能将不可用")

# 尝试导入soundfile，如果失败则降级到只支持wav格式
try:
    import soundcard as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    logger.warning(
        "未安装 soundfile 模块，只能处理WAV格式音频。\n"
        "要处理其他格式音频，请运行: pip install soundfile"
    )

# 尝试导入soundcard，如果失败则降级到只支持wav格式
try:
    import soundcard as sc
    HAS_SOUNDCARD = True
except ImportError:
    HAS_SOUNDCARD = False
    logger.warning(
        "未安装 soundcard 模块，只能处理WAV格式音频。\n"
        "要处理其他格式音频，请运行: pip install soundcard"
    )

class SherpaOnnxPlugin(ASRPluginBase):
    """Sherpa-ONNX ASR 插件实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.engine_type = "sherpa_onnx"
        self.is_int8 = config.get("use_int8", False)
        self.is_0626 = False
        
    def setup(self) -> bool:
        """初始化 Sherpa-ONNX 模型"""
        if not HAS_SHERPA_ONNX:
            logger.error("未安装 sherpa_onnx 模块")
            return False
            
        try:
            model_path = self.config.get("model_path")
            if not model_path or not os.path.exists(model_path):
                logger.error(f"Sherpa-ONNX 模型路径无效: {model_path}")
                return False
                
            self.model_dir = model_path
            
            # 检查是否是0626模型
            if "0626" in model_path or "2023-06-26" in model_path:
                self.is_0626 = True
                self.engine_type = "sherpa_0626"
                
            # 根据模型类型选择文件名
            if self.is_0626:
                encoder_file = "encoder-epoch-99-avg-1-chunk-16-left-128.onnx"
                decoder_file = "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"
                joiner_file = "joiner-epoch-99-avg-1-chunk-16-left-128.onnx"
            else:
                suffix = ".int8.onnx" if self.is_int8 else ".onnx"
                encoder_file = f"encoder-epoch-99-avg-1{suffix}"
                decoder_file = f"decoder-epoch-99-avg-1{suffix}"
                joiner_file = f"joiner-epoch-99-avg-1{suffix}"
                
            # 构建模型配置
            model_config = {
                "encoder": os.path.join(model_path, encoder_file),
                "decoder": os.path.join(model_path, decoder_file),
                "joiner": os.path.join(model_path, joiner_file),
                "tokens": os.path.join(model_path, "tokens.txt"),
                "num_threads": 1,
                "debug": False,
                "provider": "cpu"
            }
            
            # 创建识别配置
            self.recognizer_config = sherpa_onnx.FeatureConfig()
            self.decoding_config = sherpa_onnx.OnlineTransducerDecodingConfig(
                max_active_paths=4
            )
            
            # 初始化模型
            self.model = sherpa_onnx.OnlineTransducerModel(**model_config)
            logger.info(f"Sherpa-ONNX 模型加载成功: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Sherpa-ONNX 模型加载失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    def create_recognizer(self):
        """创建 Sherpa-ONNX 识别器"""
        if not self.model:
            logger.error("Sherpa-ONNX 模型未初始化")
            return None
            
        try:
            self.recognizer = sherpa_onnx.OnlineRecognizer(
                self.model,
                self.recognizer_config,
                self.decoding_config
            )
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
            # 如果输入是bytes,转换为numpy数组
            if isinstance(audio_data, bytes):
                audio_data = np.frombuffer(audio_data, dtype=np.int16)
                
            # 进行识别
            self.recognizer.accept_waveform(audio_data)
            result = self.recognizer.get_result()
            return result.text if result else None
            
        except Exception as e:
            logger.error(f"转录失败: {str(e)}")
            return None
            
    def transcribe_file(self, file_path: str) -> Optional[str]:
        """转录音频文件"""
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return None
            
        try:
            # 处理 WAV 文件
            if file_path.endswith('.wav'):
                import wave
                with wave.open(file_path, 'rb') as wf:
                    framerate = wf.getframerate()
                    audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            else:
                # 处理其他音频格式
                try:
                    import soundcard as sc
                    with sc.open_audio_file(file_path) as f:
                        audio_data = f.read()
                        if audio_data.dtype != np.int16:
                            audio_data = (audio_data * 32768).astype(np.int16)
                except ImportError:
                    logger.error("需要安装 soundcard 模块来处理非WAV格式音频。运行: pip install soundcard")
                    return None
                    
            # 创建识别器
            recognizer = self.create_recognizer()
            if not recognizer:
                return None
                
            # 进行识别
            recognizer.accept_waveform(audio_data)
            result = recognizer.get_result()
            return result.text if result else None
            
        except Exception as e:
            logger.error(f"转录文件失败: {str(e)}")
            return None
            
    def validate_files(self) -> bool:
        """验证模型文件完整性"""
        if not self.model_dir or not os.path.exists(self.model_dir):
            return False
            
        # 根据模型类型选择文件名
        if self.is_0626:
            required_files = [
                "encoder-epoch-99-avg-1-chunk-16-left-128.onnx",
                "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
                "joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
                "tokens.txt"
            ]
        else:
            suffix = ".int8.onnx" if self.is_int8 else ".onnx"
            required_files = [
                f"encoder-epoch-99-avg-1{suffix}",
                f"decoder-epoch-99-avg-1{suffix}",
                f"joiner-epoch-99-avg-1{suffix}",
                "tokens.txt"
            ]
            
        for file in required_files:
            file_path = os.path.join(self.model_dir, file)
            if not os.path.exists(file_path):
                logger.error(f"缺少必要文件: {file}")
                return False
                
        return True