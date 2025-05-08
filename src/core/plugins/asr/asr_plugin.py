"""ASR插件基类模块"""
from typing import Optional, Dict, Any
from ..base.plugin_interface import PluginInterface

class ASRPlugin(PluginInterface):
    """ASR插件基类"""
    
    def __init__(self):
        super().__init__()
        self.model_type = None
        self._recognizer = None
        
    def create_recognizer(self):
        """创建识别器实例"""
        raise NotImplementedError
        
    def transcribe_file(self, file_path: str) -> Optional[str]:
        """转录音频文件"""
        raise NotImplementedError
        
    def cleanup(self):
        """清理资源"""
        if self._recognizer:
            self._recognizer = None