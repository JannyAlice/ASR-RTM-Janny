"""插件注册器模块"""
import logging
from typing import Dict, Type, Optional
from .asr_plugin_base import ASRPluginBase
from .vosk_plugin import VoskPlugin
from .sherpa_plugin import SherpaOnnxPlugin

logger = logging.getLogger(__name__)

class PluginRegistry:
    """插件注册器，管理所有可用的ASR插件"""
    
    def __init__(self):
        self._plugins: Dict[str, Type[ASRPluginBase]] = {}
        self._register_builtin_plugins()
        
    def _register_builtin_plugins(self):
        """注册内置插件"""
        self.register_plugin("vosk_small", VoskPlugin)
        self.register_plugin("sherpa_onnx", SherpaOnnxPlugin)
        self.register_plugin("sherpa_0626", SherpaOnnxPlugin)
        logger.info("内置插件注册完成")
        
    def register_plugin(self, name: str, plugin_class: Type[ASRPluginBase]):
        """注册新的插件
        
        Args:
            name: 插件名称
            plugin_class: 插件类
        """
        if name in self._plugins:
            logger.warning(f"插件 {name} 已存在，将被覆盖")
        self._plugins[name] = plugin_class
        logger.info(f"注册插件: {name}")
        
    def get_plugin_class(self, name: str) -> Optional[Type[ASRPluginBase]]:
        """获取插件类
        
        Args:
            name: 插件名称
            
        Returns:
            Type[ASRPluginBase]: 插件类，如果不存在则返回None
        """
        return self._plugins.get(name)
        
    def create_plugin(self, name: str, config: dict) -> Optional[ASRPluginBase]:
        """创建插件实例
        
        Args:
            name: 插件名称
            config: 插件配置
            
        Returns:
            ASRPluginBase: 插件实例，如果创建失败则返回None
        """
        plugin_class = self.get_plugin_class(name)
        if not plugin_class:
            logger.error(f"插件 {name} 不存在")
            return None
            
        try:
            plugin = plugin_class(config)
            logger.info(f"创建插件实例: {name}")
            return plugin
        except Exception as e:
            logger.error(f"创建插件 {name} 实例失败: {str(e)}")
            return None
            
    def list_plugins(self) -> Dict[str, Type[ASRPluginBase]]:
        """获取所有已注册的插件
        
        Returns:
            Dict[str, Type[ASRPluginBase]]: 插件名称到插件类的映射
        """
        return self._plugins.copy()

# 创建全局插件注册器实例
plugin_registry = PluginRegistry()