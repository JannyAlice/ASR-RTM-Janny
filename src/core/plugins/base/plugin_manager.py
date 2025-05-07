from typing import Dict, Optional, Type
from .plugin_interface import PluginInterface
from .plugin_event import PluginEventSystem
import logging

logger = logging.getLogger(__name__)

class PluginManager:
    """插件管理器，负责插件的注册、加载和生命周期管理"""
    
    def __init__(self):
        self._plugins: Dict[str, PluginInterface] = {}
        self._plugin_classes: Dict[str, Type[PluginInterface]] = {}
        self.event_system = PluginEventSystem()
        
    def register_plugin_class(self, plugin_id: str, plugin_class: Type[PluginInterface]) -> None:
        """注册插件类"""
        if plugin_id in self._plugin_classes:
            logger.warning(f"Plugin {plugin_id} is already registered, overwriting...")
        self._plugin_classes[plugin_id] = plugin_class
        logger.info(f"Registered plugin class: {plugin_id}")
        
    def load_plugin(self, plugin_id: str, config: dict) -> bool:
        """加载插件"""
        if plugin_id not in self._plugin_classes:
            logger.error(f"Plugin {plugin_id} is not registered")
            return False
            
        if plugin_id in self._plugins:
            logger.warning(f"Plugin {plugin_id} is already loaded")
            return True
            
        try:
            plugin = self._plugin_classes[plugin_id]()
            if plugin.initialize(config):
                self._plugins[plugin_id] = plugin
                logger.info(f"Successfully loaded plugin: {plugin_id}")
                return True
            else:
                logger.error(f"Failed to initialize plugin: {plugin_id}")
                return False
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_id}: {e}")
            return False
            
    def unload_plugin(self, plugin_id: str) -> bool:
        """卸载插件"""
        if plugin_id not in self._plugins:
            logger.warning(f"Plugin {plugin_id} is not loaded")
            return False
            
        try:
            plugin = self._plugins[plugin_id]
            plugin.cleanup()
            del self._plugins[plugin_id]
            logger.info(f"Successfully unloaded plugin: {plugin_id}")
            return True
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_id}: {e}")
            return False
            
    def get_plugin(self, plugin_id: str) -> Optional[PluginInterface]:
        """获取已加载的插件实例"""
        return self._plugins.get(plugin_id)
        
    def get_all_plugins(self) -> Dict[str, PluginInterface]:
        """获取所有已加载的插件"""
        return self._plugins.copy()
        
    def shutdown(self) -> None:
        """关闭插件管理器，清理所有插件"""
        for plugin_id in list(self._plugins.keys()):
            self.unload_plugin(plugin_id)
        self.event_system.clear()
        logger.info("Plugin manager shutdown complete")