from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
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
    """插件基类接口"""
    
    def __init__(self):
        self._enabled = False
        self._config = {}
        self._initialized = False
        logger.debug(f"Initializing plugin: {self.__class__.__name__}")
        
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化插件
        
        Args:
            config: 插件配置
            
        Returns:
            bool: 是否初始化成功
            
        Raises:
            PluginInitializationError: 初始化过程中出现错误
            PluginConfigError: 配置无效或缺失
        """
        try:
            if not config:
                raise PluginConfigError("Plugin configuration is empty")
            self._config = config
            self._initialized = True
            logger.info(f"Plugin {self.__class__.__name__} initialized with config: {config}")
            return True
        except PluginError as e:
            logger.error(f"Plugin initialization error: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during plugin initialization: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    def enable(self) -> bool:
        """启用插件"""
        try:
            if not self._initialized:
                raise PluginInitializationError("Cannot enable plugin: not initialized")
            
            if not self._enabled:
                self._enabled = True
                logger.info(f"Plugin {self.__class__.__name__} enabled")
                return True
            return False
        except PluginError as e:
            logger.error(f"Error enabling plugin: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error enabling plugin: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    def disable(self) -> None:
        """禁用插件"""
        try:
            if self._enabled:
                self._enabled = False
                logger.info(f"Plugin {self.__class__.__name__} disabled")
            self.cleanup()
        except Exception as e:
            logger.error(f"Error disabling plugin: {str(e)}")
            logger.error(traceback.format_exc())
            
    def is_enabled(self) -> bool:
        """检查插件是否启用"""
        return self._enabled
        
    def get_config(self) -> Dict[str, Any]:
        """获取插件配置"""
        return self._config.copy()
        
    @abstractmethod
    def get_info(self) -> Dict[str, str]:
        """获取插件信息
        
        Returns:
            Dict[str, str]: 包含插件ID、名称、描述等信息
        """
        pass
        
    @abstractmethod
    def cleanup(self) -> None:
        """清理插件资源
        
        Raises:
            PluginCleanupError: 清理过程中出现错误
        """
        try:
            self._enabled = False
            self._initialized = False
            logger.info(f"Plugin {self.__class__.__name__} cleaned up")
        except Exception as e:
            error_msg = f"Error cleaning up plugin {self.__class__.__name__}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise PluginCleanupError(error_msg)