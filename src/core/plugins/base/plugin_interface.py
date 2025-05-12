"""插件接口基类模块"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
import traceback

logger = logging.getLogger(__name__)

class PluginError(Exception):
    """插件基础异常类"""
    pass

class PluginInitializationError(PluginError):
    """插件初始化错误"""
    pass

class PluginConfigError(PluginError):
    """插件配置错误"""
    pass

class PluginCleanupError(PluginError):
    """插件清理错误"""
    pass

class PluginInterface(ABC):
    """插件接口基类"""
    
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
        """配置插件
        
        Args:
            config: 配置字典
        """
        self._config = config