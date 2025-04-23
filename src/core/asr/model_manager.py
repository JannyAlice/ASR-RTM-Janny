"""
ASR模型管理模块
负责加载和管理ASR模型
"""
import os
import vosk
import numpy as np
from typing import Optional, Dict, Any, Union

from src.utils.config_manager import config_manager
from .vosk_engine import VoskASR
from .sherpa_engine import SherpaOnnxASR

class ASRModelManager:
    """ASR模型管理器类"""

    def __init__(self):
        """初始化ASR模型管理器"""
        self.config = config_manager
        # 直接从 config.json 获取模型配置
        self.models_config = {}
        if 'asr' in self.config.config and 'models' in self.config.config['asr']:
            self.models_config = self.config.config['asr']['models']
        print(f"[DEBUG] ASRModelManager.__init__: models_config = {self.models_config}")
        print(f"[DEBUG] ASRModelManager.__init__: config = {self.config.config}")
        self.current_model = None
        self.model_path = None
        self.model_type = None

        # 用于音频转录的引擎
        self.current_engine = None

    def load_model(self, model_name: str) -> bool:
        """
        加载ASR模型

        Args:
            model_name: 模型名称

        Returns:
            bool: 加载是否成功
        """
        try:
            # 直接从 models_config 获取模型配置
            print(f"[DEBUG] load_model: models_config = {self.models_config}")
            print(f"[DEBUG] load_model: model_name = {model_name}")
            if model_name not in self.models_config:
                print(f"模型 {model_name} 不存在")
                return False

            model_config = self.models_config[model_name]
            model_path = model_config.get('path', '')

            # 检查模型路径
            if not os.path.exists(model_path):
                print(f"模型路径不存在: {model_path}")
                return False

            # 加载模型
            if model_name == 'vosk':
                self.current_model = self._load_vosk_model(model_path)
            elif model_name.startswith('sherpa'):
                self.current_model = self._load_sherpa_model(model_path, model_config)
            else:
                print(f"不支持的模型类型: {model_name}")
                return False

            # 更新当前模型信息
            self.model_path = model_path
            self.model_type = model_name

            print(f"成功加载模型: {model_name}")
            return True

        except Exception as e:
            print(f"加载模型失败: {e}")
            return False

    def _load_vosk_model(self, model_path: str) -> Any:
        """
        加载VOSK模型

        Args:
            model_path: 模型路径

        Returns:
            Any: VOSK模型实例
        """
        return vosk.Model(model_path)

    def _load_sherpa_model(self, model_path: str, model_config: Dict[str, Any]) -> Any:
        """
        加载Sherpa模型

        Args:
            model_path: 模型路径
            model_config: 模型配置

        Returns:
            Any: Sherpa模型实例
        """
        # 这里是Sherpa模型加载的占位代码
        # 实际实现需要根据Sherpa-ONNX的API进行
        print(f"加载Sherpa模型: {model_path}")
        return {"path": model_path, "config": model_config}

    def create_recognizer(self) -> Optional[Any]:
        """
        创建识别器

        Returns:
            Optional[Any]: 识别器实例
        """
        if not self.current_model:
            print("未加载模型，无法创建识别器")
            return None

        try:
            if self.model_type == 'vosk':
                return vosk.KaldiRecognizer(self.current_model, 16000)
            elif self.model_type.startswith('sherpa'):
                # 这里是Sherpa识别器创建的占位代码
                print("创建Sherpa识别器")
                return {"model": self.current_model, "sample_rate": 16000}
            else:
                print(f"不支持的模型类型: {self.model_type}")
                return None

        except Exception as e:
            print(f"创建识别器失败: {e}")
            return None

    def check_model_directory(self) -> Dict[str, bool]:
        """
        检查模型目录

        Returns:
            Dict[str, bool]: 模型可用性字典
        """
        result = {}
        models = self.models_config.get('models', {})

        for name, config in models.items():
            path = config.get('path', '')
            result[name] = os.path.exists(path)

        return result

    # 以下是从 manager.py 合并的方法

    def _get_nested_config(self, path: str, default: Any = None) -> Any:
        """获取嵌套配置

        Args:
            path: 配置路径，使用点表示法，如 "asr.models.vosk"
            default: 如果路径不存在，返回的默认值

        Returns:
            Any: 配置值或默认值
        """
        parts = path.split('.')
        config = self.config.config  # 获取整个配置字典

        for part in parts:
            if isinstance(config, dict) and part in config:
                config = config[part]
            else:
                return default

        return config

    def initialize_engine(self, engine_type: str = "vosk") -> bool:
        """初始化指定的 ASR 引擎

        Args:
            engine_type: 引擎类型，可选 "vosk" 或 "sherpa"

        Returns:
            bool: 是否初始化成功
        """
        try:
            # 直接从 models_config 获取模型配置
            print(f"[DEBUG] initialize_engine: models_config = {self.models_config}")
            print(f"[DEBUG] initialize_engine: engine_type = {engine_type}")
            if engine_type not in self.models_config:
                print(f"Engine {engine_type} is not found in models_config")
                return False

            model_config = self.models_config[engine_type]
            if not model_config or not model_config.get("enabled", False):
                print(f"Engine {engine_type} is not enabled or not configured")
                return False

            # 初始化引擎
            if engine_type == "vosk":
                self.current_engine = VoskASR(model_config["path"])
            elif engine_type == "sherpa":
                self.current_engine = SherpaOnnxASR(model_config["path"])
            else:
                print(f"Unsupported engine type: {engine_type}")
                return False

            # 设置引擎
            return self.current_engine.setup()

        except Exception as e:
            print(f"Error initializing {engine_type} engine: {str(e)}")
            return False

    def transcribe(self, audio_data: Union[bytes, np.ndarray]) -> Optional[str]:
        """转录音频数据

        Args:
            audio_data: 音频数据，可以是字节或 numpy 数组

        Returns:
            str: 转录文本，如果失败则返回 None
        """
        if not self.current_engine:
            print("No ASR engine initialized")
            return None

        try:
            return self.current_engine.transcribe(audio_data)
        except Exception as e:
            print(f"Error in transcription: {str(e)}")
            return None

    def reset(self) -> None:
        """重置当前引擎状态"""
        if self.current_engine:
            self.current_engine.reset()

    def get_final_result(self) -> Optional[str]:
        """获取最终识别结果

        Returns:
            str: 最终识别文本，如果失败则返回 None
        """
        if not self.current_engine:
            return None

        return self.current_engine.get_final_result()

    def get_current_engine_type(self) -> Optional[str]:
        """获取当前引擎类型

        Returns:
            str: 当前引擎类型，如果未初始化则返回 None
        """
        if isinstance(self.current_engine, VoskASR):
            return "vosk"
        elif isinstance(self.current_engine, SherpaOnnxASR):
            return "sherpa"
        return None

    def get_available_engines(self) -> Dict[str, bool]:
        """获取可用的引擎列表

        Returns:
            Dict[str, bool]: 引擎名称到是否启用的映射
        """
        engines = {}
        for engine_type in ["vosk", "sherpa"]:
            if engine_type in self.models_config:
                model_config = self.models_config[engine_type]
                engines[engine_type] = bool(model_config and model_config.get("enabled", False))
            else:
                engines[engine_type] = False
        return engines
