"""ASR 插件基类模块"""
# [修改] 增加新的导入
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, List  # 新增 List 类型
from pathlib import Path  # 新增用于路径处理
import logging  # 新增用于日志记录
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
        # [新增] 添加日志记录器
        self.logger = logging.getLogger(self.__class__.__name__)
        # [新增] 添加初始化状态标志
        self._is_initialized = False
        # [新增] 添加支持的模型列表
        self._supported_models = []
        
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

    # [新增] 模型信息相关抽象方法
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        pass

    # [新增] 支持的模型列表属性
    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """支持的模型列表"""
        pass

    # [新增] 模型加载抽象方法
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """加载模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            bool: 是否成功加载
        """
        pass
        
    # [新增] 插件初始化方法
    def initialize(self) -> bool:
        """初始化插件"""
        try:
            if self._is_initialized:
                return True
            
            success = self.setup()
            if success:
                self._is_initialized = True
            return success
        except Exception as e:
            self.logger.error(f"Failed to initialize plugin: {e}")
            return False

    # [新增] 初始化状态检查方法
    def is_initialized(self) -> bool:
        """检查插件是否已初始化"""
        return self._is_initialized

    # [新增] 获取引擎类型方法
    def get_engine_type(self) -> str:
        """获取引擎类型"""
        return self.engine_type if self.engine_type else "unknown"
        
    def reset(self) -> None:
        """重置识别器状态"""
        try:
            if self.recognizer is not None and hasattr(self.recognizer, 'Reset'):
                self.recognizer.Reset()
        except AttributeError as e:
            self.logger.debug(f"Reset operation failed: {e}")
            
    def get_final_result(self) -> Optional[str]:
        """获取最终识别结果"""
        try:
            if self.recognizer and hasattr(self.recognizer, 'FinalResult'):
                result = self.recognizer.FinalResult()
                return str(result) if result is not None else None
            return None
        except AttributeError:
            return None

    # [修改] 增强清理方法
    def cleanup(self) -> None:
        """清理资源"""
        try:
            self.reset()  # 先重置识别器
            self.model = None
            self.recognizer = None
            self._is_initialized = False  # 重置初始化状态
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
