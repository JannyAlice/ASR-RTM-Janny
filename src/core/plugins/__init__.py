"""ASR 插件系统"""
from .asr_plugin_base import ASRPluginBase
from .vosk_plugin import VoskPlugin
from .sherpa_plugin import SherpaOnnxPlugin
from .plugin_registry import PluginRegistry, plugin_registry

__all__ = [
    'ASRPluginBase',
    'VoskPlugin',
    'SherpaOnnxPlugin',
    'PluginRegistry',
    'plugin_registry'
]