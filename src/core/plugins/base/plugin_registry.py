"""插件注册表模块"""
import logging
from typing import Dict, Type, Optional
from .plugin_base import PluginInterface

logger = logging.getLogger(__name__)

class PluginRegistry:
    """插件注册表类"""
    
    def __init__(self):
        """初始化插件注册表"""
        self._plugins: Dict[str, Type[PluginInterface]] = {}
        
    def register(self, name: str, plugin_class: Type[PluginInterface]) -> None:
        """注册插件"""
        self._plugins[name] = plugin_class
        logger.debug(f"注册插件: {name}")
        
    def get_plugin(self, name: str) -> Optional[Type[PluginInterface]]:
        """获取插件类"""
        return self._plugins.get(name)
        
    def get_plugins_by_type(self, plugin_type: str) -> Dict[str, Type[PluginInterface]]:
        """获取指定类型的所有插件"""
        return {
            plugin_id: plugin_class
            for plugin_id, plugin_class in self._plugins.items()
            if getattr(plugin_class, 'plugin_type', None) == plugin_type
        }