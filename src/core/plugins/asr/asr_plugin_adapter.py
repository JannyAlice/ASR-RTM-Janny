from typing import Optional, Dict, Any
from PySide6.QtCore import QObject, Signal
import traceback
from ...asr.model_manager import ASRModelManager
from ..base.plugin_registry import PluginRegistry
import logging

logger = logging.getLogger(__name__)

class ASRPluginAdapterError(Exception):
    """ASR插件适配器基础异常类"""
    pass

class PluginInitError(ASRPluginAdapterError):
    """插件初始化错误"""
    pass

class PluginNotFoundError(ASRPluginAdapterError):
    """插件未找到错误"""
    pass

class ASRPluginAdapter(QObject):
    """ASR插件适配器，用于将插件系统与现有的ASRModelManager集成"""
    
    # 定义信号
    model_loaded = Signal(bool)  # 模型加载状态信号
    transcription_result = Signal(str)  # 转录结果信号
    error_occurred = Signal(str)  # 错误信号
    
    def __init__(self):
        super().__init__()
        self.registry = PluginRegistry()
        self.current_plugin_id: Optional[str] = None
        logger.info("ASRPluginAdapter initialized")
        
    def initialize_engine(self, plugin_id: str) -> bool:
        """初始化指定的ASR引擎插件"""
        try:
            logger.info(f"Attempting to initialize ASR plugin: {plugin_id}")
            
            # 如果已经加载了相同的插件，直接返回成功
            if self.current_plugin_id == plugin_id and self.registry.get_plugin(plugin_id):
                logger.info(f"Plugin {plugin_id} already loaded")
                return True
                
            # 如果之前加载了其他插件，先卸载它
            if self.current_plugin_id and self.current_plugin_id != plugin_id:
                logger.info(f"Unloading previous plugin: {self.current_plugin_id}")
                self.registry.unload_plugin(self.current_plugin_id)
                
            # 获取插件实例
            plugin = self.registry.get_plugin(plugin_id)
            if not plugin:
                raise PluginNotFoundError(f"Plugin {plugin_id} not found")
                
            # 启用插件
            if not plugin.enable():
                raise PluginInitError(f"Failed to enable plugin {plugin_id}")
                
            self.current_plugin_id = plugin_id
            self.model_loaded.emit(True)
            logger.info(f"Successfully initialized ASR plugin: {plugin_id}")
            return True
            
        except ASRPluginAdapterError as e:
            error_msg = str(e)
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            self.model_loaded.emit(False)
            return False
        except Exception as e:
            error_msg = f"Unexpected error initializing ASR plugin {plugin_id}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.error_occurred.emit(error_msg)
            self.model_loaded.emit(False)
            return False
            
    def process_audio(self, audio_data) -> Optional[str]:
        """处理音频数据"""
        try:
            if not self.current_plugin_id:
                raise PluginNotFoundError("No plugin currently loaded")
                
            plugin = self.registry.get_plugin(self.current_plugin_id)
            if not plugin:
                raise PluginNotFoundError(f"Current plugin {self.current_plugin_id} not found")
                
            result = plugin.process_audio(audio_data)
            if result and "text" in result:
                text = result["text"]
                logger.debug(f"Processed audio result: {text}")
                self.transcription_result.emit(text)
                return text
            return None
            
        except ASRPluginAdapterError as e:
            error_msg = f"Error processing audio: {str(e)}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return None
        except Exception as e:
            error_msg = f"Unexpected error processing audio: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.error_occurred.emit(error_msg)
            return None
            
    def transcribe_file(self, file_path: str) -> Optional[str]:
        """转录文件"""
        try:
            if not self.current_plugin_id:
                raise PluginNotFoundError("No plugin currently loaded")
                
            plugin = self.registry.get_plugin(self.current_plugin_id)
            if not plugin:
                raise PluginNotFoundError(f"Current plugin {self.current_plugin_id} not found")
                
            logger.info(f"Starting file transcription: {file_path}")
            result = plugin.transcribe_file(file_path)
            if result and "text" in result:
                text = result["text"]
                logger.info(f"File transcription completed, length: {len(text)} characters")
                self.transcription_result.emit(text)
                return text
            return None
            
        except ASRPluginAdapterError as e:
            error_msg = f"Error transcribing file: {str(e)}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return None
        except Exception as e:
            error_msg = f"Unexpected error transcribing file: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.error_occurred.emit(error_msg)
            return None
            
    def get_current_engine_type(self) -> Optional[str]:
        """获取当前引擎类型"""
        try:
            if not self.current_plugin_id:
                logger.debug("No plugin currently loaded")
                return None
                
            plugin = self.registry.get_plugin(self.current_plugin_id)
            if plugin:
                engine_type = plugin.get_info().get("id")
                logger.debug(f"Current engine type: {engine_type}")
                return engine_type
            return None
            
        except Exception as e:
            logger.error(f"Error getting engine type: {e}")
            return None
            
    def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.current_plugin_id:
                logger.info(f"Cleaning up plugin: {self.current_plugin_id}")
                self.registry.unload_plugin(self.current_plugin_id)
                self.current_plugin_id = None
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            logger.error(traceback.format_exc())