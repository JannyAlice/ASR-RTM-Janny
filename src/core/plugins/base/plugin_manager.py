"""插件管理器模块"""
import logging
from typing import Any, Dict, Optional, Type
from .plugin_base import PluginInterface
from .plugin_registry import PluginRegistry

logger = logging.getLogger(__name__)

class PluginManager:
    """插件管理器类"""
    
    def __init__(self):
        self._plugins: Dict[str, PluginInterface] = {}
        self._registry = PluginRegistry()
        self._config: Dict[str, Any] = {}
        
    def register_plugin(self, plugin_id: str, plugin_class: Type[PluginInterface]) -> None:
        """注册插件"""
        self._registry.register(plugin_id, plugin_class)
        
    def configure(self, config: Dict[str, Any]) -> None:
        """配置插件管理器
        
        Args:
            config: 配置字典
        """
        self._config = config
        
    def get_plugin(self, plugin_id: str) -> Optional[PluginInterface]:
        """获取插件实例"""
        return self._plugins.get(plugin_id)
        
    def load_plugins(self, plugin_type: str) -> Dict[str, PluginInterface]:
        """加载指定类型的所有插件
        
        Args:
            plugin_type: 插件类型(如'asr')
            
        Returns:
            Dict[str, PluginInterface]: 已加载的插件字典
        """
        loaded_plugins = {}
        plugins = self._registry.get_plugins_by_type(plugin_type)
        
        for plugin_id, plugin_class in plugins.items():
            try:
                # 创建插件实例
                plugin = plugin_class()
                
                # 配置插件
                if plugin_type in self._config and "models" in self._config[plugin_type]:
                    plugin_config = self._config[plugin_type]["models"].get(plugin_id)
                    if plugin_config:
                        plugin.configure(plugin_config)
                
                # 初始化插件
                if plugin.setup():
                    self._plugins[plugin_id] = plugin
                    loaded_plugins[plugin_id] = plugin
                    logger.info(f"成功加载插件: {plugin_id}")
                else:
                    logger.error(f"插件初始化失败: {plugin_id}")
                    
            except Exception as e:
                logger.error(f"加载插件失败 {plugin_id}: {str(e)}")
                
        return loaded_plugins
        
    def get_registry(self) -> PluginRegistry:
        """获取插件注册表"""
        return self._registry
        
    def cleanup(self):
        """清理所有插件资源"""
        for plugin_id, plugin in self._plugins.items():
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"清理插件 {plugin_id} 时发生错误: {str(e)}")
        self._plugins.clear()