import os
from typing import Dict, Optional, Tuple, Union
from .opus_engine import OpusMTEngine
from .argos_engine import ArgosEngine

class TranslationManager:
    """翻译引擎管理器
    
    负责管理不同的翻译引擎，提供统一的翻译接口。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化翻译引擎管理器
        
        Args:
            config (Dict, optional): 配置字典，包含各个引擎的配置信息
        """
        self.config = config or {}
        self.engines: Dict[str, Union[OpusMTEngine, ArgosEngine]] = {}
        self.current_engine: Optional[str] = None
        
        # 初始化默认引擎
        self._init_default_engines()
    
    def _init_default_engines(self):
        """初始化默认的翻译引擎"""
        # 初始化 OPUS-MT 引擎
        opus_config = self.config.get('opus_mt', {})
        opus_model_dir = opus_config.get('model_dir')
        self.engines['opus_mt'] = OpusMTEngine(model_dir=opus_model_dir)
        
        # 初始化 ArgosTranslate 引擎
        argos_config = self.config.get('argos', {})
        argos_model_dir = argos_config.get('model_dir')
        self.engines['argos'] = ArgosEngine(model_dir=argos_model_dir)
        
        # 设置默认引擎
        self.current_engine = 'opus_mt'
    
    def set_engine(self, engine_name: str) -> bool:
        """
        设置当前使用的翻译引擎
        
        Args:
            engine_name (str): 引擎名称，如 'opus_mt' 或 'argos'
            
        Returns:
            bool: 是否设置成功
        """
        if engine_name in self.engines:
            self.current_engine = engine_name
            return True
        return False
    
    def get_available_engines(self) -> list:
        """
        获取所有可用的翻译引擎
        
        Returns:
            list: 可用引擎名称列表
        """
        return list(self.engines.keys())
    
    def get_current_engine(self) -> Optional[str]:
        """
        获取当前使用的翻译引擎名称
        
        Returns:
            Optional[str]: 当前引擎名称，如果没有设置则返回 None
        """
        return self.current_engine
    
    def translate(self, text: str, engine_name: Optional[str] = None, **kwargs) -> Tuple[Optional[str], float]:
        """
        翻译文本
        
        Args:
            text (str): 要翻译的文本
            engine_name (str, optional): 指定使用的引擎名称。如果为 None，则使用当前引擎
            **kwargs: 传递给具体引擎的额外参数
            
        Returns:
            Tuple[Optional[str], float]: (翻译结果, 延迟时间)
        """
        if not text:
            return None, 0.0
            
        # 确定使用的引擎
        engine_to_use = engine_name if engine_name else self.current_engine
        if not engine_to_use or engine_to_use not in self.engines:
            return None, 0.0
            
        # 调用对应引擎的翻译方法
        engine = self.engines[engine_to_use]
        return engine.translate(text, **kwargs)
    
    def get_engine_info(self, engine_name: Optional[str] = None) -> Dict:
        """
        获取引擎信息
        
        Args:
            engine_name (str, optional): 引擎名称。如果为 None，则返回当前引擎的信息
            
        Returns:
            Dict: 引擎信息字典
        """
        engine_to_use = engine_name if engine_name else self.current_engine
        if not engine_to_use or engine_to_use not in self.engines:
            return {}
            
        engine = self.engines[engine_to_use]
        info = {
            'name': engine_to_use,
            'type': type(engine).__name__,
            'model_dir': getattr(engine, 'model_dir', None)
        }
        
        # 添加引擎特定的信息
        if isinstance(engine, OpusMTEngine):
            info['supports_onnx'] = hasattr(engine, 'onnx_model') and engine.onnx_model is not None
        elif isinstance(engine, ArgosEngine):
            info['supported_languages'] = engine.get_supported_languages()
        
        return info
