"""插件系统基础类"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Optional

class PluginInterface(ABC):
    """插件基础接口"""
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
        
    @abstractmethod
    def setup(self) -> bool:
        """初始化插件"""
        pass
        
    @abstractmethod
    def cleanup(self) -> None:
        """清理插件资源"""
        pass
        
    def configure(self, config: Dict[str, Any]) -> None:
        """配置插件"""
        self._config = config