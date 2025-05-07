from typing import Dict, Any, Optional, Type
import logging
import traceback
from .plugin_interface import PluginInterface, PluginError

logger = logging.getLogger(__name__)

class PluginRegistryError(Exception):
    """插件注册表异常基类"""
    pass

class PluginRegistrationError(PluginRegistryError):
    """插件注册错误"""
    pass

class PluginLoadError(PluginRegistryError):
    """插件加载错误"""
    pass

class PluginUnloadError(PluginRegistryError):
    """插件卸载错误"""
    pass

class PluginRegistry:
    """插件注册表，用于管理已注册的插件"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._plugin_classes = {}
            cls._instance._loaded_plugins = {}
            logger.info("Created new PluginRegistry instance")
        return cls._instance
        
    def register_plugin(self, plugin_id: str, plugin_class: Type[PluginInterface]) -> None:
        """注册插件类
        
        Args:
            plugin_id: 插件ID
            plugin_class: 插件类
            
        Raises:
            PluginRegistrationError: 注册过程中出现错误
        """
        try:
            if not plugin_id:
                raise PluginRegistrationError("Plugin ID cannot be empty")
            if not plugin_class:
                raise PluginRegistrationError("Plugin class cannot be None")
            if not issubclass(plugin_class, PluginInterface):
                raise PluginRegistrationError(
                    f"Plugin class {plugin_class.__name__} must inherit from PluginInterface"
                )
                
            if plugin_id in self._plugin_classes:
                logger.warning(f"Plugin {plugin_id} already registered, will be overwritten")
            self._plugin_classes[plugin_id] = plugin_class
            logger.info(f"Registered plugin: {plugin_id} ({plugin_class.__name__})")
            
        except PluginRegistryError as e:
            logger.error(f"Error registering plugin: {str(e)}")
            raise
        except Exception as e:
            error_msg = f"Unexpected error registering plugin: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise PluginRegistrationError(error_msg)
        
    def load_plugin(self, plugin_id: str, config: Dict[str, Any]) -> bool:
        """加载插件
        
        Args:
            plugin_id: 插件ID
            config: 插件配置
            
        Returns:
            bool: 是否成功加载
            
        Raises:
            PluginLoadError: 加载过程中出现错误
        """
        try:
            if plugin_id not in self._plugin_classes:
                raise PluginLoadError(f"Plugin {plugin_id} is not registered")
                
            if plugin_id in self._loaded_plugins:
                logger.warning(f"Plugin {plugin_id} is already loaded")
                return True
                
            logger.info(f"Loading plugin: {plugin_id}")
            plugin_class = self._plugin_classes[plugin_id]
            plugin = plugin_class()
            
            if not plugin.initialize(config):
                raise PluginLoadError(f"Failed to initialize plugin: {plugin_id}")
                
            self._loaded_plugins[plugin_id] = plugin
            logger.info(f"Successfully loaded plugin: {plugin_id}")
            return True
            
        except PluginError as e:
            logger.error(f"Plugin error while loading {plugin_id}: {str(e)}")
            return False
        except PluginRegistryError as e:
            logger.error(f"Error loading plugin {plugin_id}: {str(e)}")
            return False
        except Exception as e:
            error_msg = f"Unexpected error loading plugin {plugin_id}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False
            
    def get_plugin(self, plugin_id: str) -> Optional[PluginInterface]:
        """获取已加载的插件实例
        
        Args:
            plugin_id: 插件ID
            
        Returns:
            Optional[PluginInterface]: 插件实例，如果未找到则返回None
        """
        try:
            plugin = self._loaded_plugins.get(plugin_id)
            if not plugin:
                logger.debug(f"Plugin {plugin_id} not found")
            return plugin
        except Exception as e:
            logger.error(f"Error retrieving plugin {plugin_id}: {str(e)}")
            return None
            
    def get_loaded_plugins(self) -> Dict[str, PluginInterface]:
        """获取所有已加载的插件
        
        Returns:
            Dict[str, PluginInterface]: 插件ID到插件实例的映射
        """
        return self._loaded_plugins.copy()

    def unload_plugin(self, plugin_id: str) -> bool:
        """卸载插件
        
        Args:
            plugin_id: 插件ID
            
        Returns:
            bool: 是否成功卸载
            
        Raises:
            PluginUnloadError: 卸载过程中出现错误
        """
        try:
            if plugin_id not in self._loaded_plugins:
                logger.warning(f"Plugin {plugin_id} is not loaded")
                return False
                
            plugin = self._loaded_plugins[plugin_id]
            logger.info(f"Unloading plugin: {plugin_id}")
            
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"Error during plugin cleanup: {str(e)}")
                logger.error(traceback.format_exc())
                
            del self._loaded_plugins[plugin_id]
            logger.info(f"Successfully unloaded plugin: {plugin_id}")
            return True
            
        except Exception as e:
            error_msg = f"Unexpected error unloading plugin {plugin_id}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False
            
    def shutdown(self) -> None:
        """关闭所有插件"""
        try:
            logger.info("Shutting down plugin registry")
            plugins = list(self._loaded_plugins.keys())
            
            for plugin_id in plugins:
                try:
                    self.unload_plugin(plugin_id)
                except Exception as e:
                    logger.error(f"Error unloading plugin {plugin_id} during shutdown: {str(e)}")
                    
            self._loaded_plugins.clear()
            logger.info(f"Successfully unloaded all plugins: {plugins}")
            
        except Exception as e:
            logger.error(f"Error during plugin registry shutdown: {str(e)}")
            logger.error(traceback.format_exc())