"""插件系统基础组件"""
from .plugin_base import PluginInterface
from .plugin_manager import PluginManager
from .plugin_registry import PluginRegistry

__all__ = ['PluginInterface', 'PluginManager', 'PluginRegistry']