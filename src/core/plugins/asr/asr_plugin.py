from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from ..base.plugin_interface import PluginInterface

class ASRPlugin(PluginInterface):
    """ASR 插件基类"""
    
    def __init__(self):
        super().__init__()
        self.recognizer = None
        
    @property
    def is_ready(self) -> bool:
        """检查识别器是否就绪"""
        return self.recognizer is not None and self.is_enabled()
        
    def create_recognizer(self) -> Any:
        """创建识别器实例
        
        Returns:
            Any: 识别器实例
        """
        raise NotImplementedError
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化插件
        
        Args:
            config: 插件配置
        
        Returns:
            bool: 是否初始化成功
        """
        self._config = config
        return self.setup()
        
    @abstractmethod
    def setup(self) -> bool:
        """设置ASR引擎
        
        Returns:
            bool: 是否设置成功
        """
        pass
        
    def cleanup(self) -> None:
        """清理资源"""
        if self.recognizer:
            self.recognizer = None
        super().cleanup()
        
    def reset(self) -> None:
        """重置识别器状态"""
        if self.recognizer:
            self.cleanup()
            self.recognizer = self.create_recognizer()
        
    def is_enabled(self) -> bool:
        """检查插件是否启用"""
        return self._enabled
        
    def enable(self) -> bool:
        """启用插件"""
        if not self._enabled:
            if self.setup():
                self._enabled = True
                return True
        return False
        
    def disable(self) -> None:
        """禁用插件"""
        if self._enabled:
            self.cleanup()
            self._enabled = False
            
    @abstractmethod
    def process_audio(self, audio_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """处理音频数据
        
        Args:
            audio_data: 音频数据，numpy 数组
            
        Returns:
            Optional[Dict[str, Any]]: 识别结果，如果没有结果则返回 None
        """
        raise NotImplementedError
        
    @abstractmethod
    def transcribe_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """转录音频文件
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            Optional[Dict[str, Any]]: 转录结果，如果失败则返回 None
        """
        raise NotImplementedError