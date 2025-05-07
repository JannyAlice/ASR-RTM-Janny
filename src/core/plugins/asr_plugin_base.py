"""ASR 插件基类模块"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import numpy as np

class ASRPluginBase(ABC):
    """ASR插件基类,定义了所有ASR插件必须实现的接口"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化插件
        
        Args:
            config: 插件配置
        """
        self.config = config
        self.model = None
        self.recognizer = None
        self.model_dir = None
        self.engine_type = None
        
    @abstractmethod
    def setup(self) -> bool:
        """设置和初始化插件
        
        Returns:
            bool: 是否成功初始化
        """
        pass
        
    @abstractmethod
    def create_recognizer(self):
        """创建识别器实例"""
        pass
        
    @abstractmethod
    def transcribe(self, audio_data: Union[bytes, np.ndarray]) -> Optional[str]:
        """转录音频数据
        
        Args:
            audio_data: 音频数据,可以是字节或numpy数组
            
        Returns:
            str: 转录文本,失败返回None
        """
        pass
        
    @abstractmethod
    def transcribe_file(self, file_path: str) -> Optional[str]:
        """转录音频文件
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            str: 转录文本,失败返回None
        """
        pass

    @abstractmethod
    def validate_files(self) -> bool:
        """验证模型文件完整性
        
        Returns:
            bool: 文件是否完整有效
        """
        pass
        
    def reset(self) -> None:
        """重置识别器状态"""
        if hasattr(self.recognizer, 'Reset'):
            self.recognizer.Reset()
            
    def get_final_result(self) -> Optional[str]:
        """获取最终识别结果"""
        if hasattr(self.recognizer, 'FinalResult'):
            return self.recognizer.FinalResult()
        return None
        
    def cleanup(self) -> None:
        """清理资源"""
        self.model = None
        self.recognizer = None