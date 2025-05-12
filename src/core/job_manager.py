import os
from typing import Optional, Union, Dict, Any
from src.utils.logger import logger
from src.core.asr.sherpa_engine import SherpaOnnxASR

class JobManager:
    """作业管理器
    
    负责管理 ASR 识别任务，并提供详细的日志记录。
    """
    
    def __init__(self, model_dir: str, model_config: Optional[Dict[str, Any]] = None):
        """
        初始化作业管理器
        
        Args:
            model_dir (str): 模型目录路径
            model_config (Dict[str, Any], optional): 模型配置，如果为None则使用默认配置
        """
        self.model_dir = model_dir
        self.model_config = model_config
        self.asr_engine = SherpaOnnxASR(model_dir=model_dir, model_config=model_config)
        self.setup_successful = self.asr_engine.setup()
        
        if self.setup_successful:
            logger.info("ASR 引擎初始化成功")
        else:
            logger.error("ASR 引擎初始化失败")
    
    def transcribe_audio(self, audio_data: Union[bytes, np.ndarray]) -> Optional[str]:
        """
        转录音频数据
        
        Args:
            audio_data (Union[bytes, np.ndarray]): 音频数据，可以是字节或numpy数组
            
        Returns:
            Optional[str]: 转录文本，如果失败则返回None
        """
        if not self.setup_successful:
            logger.error("ASR 引擎未初始化，无法转录音频")
            return None
        
        try:
            logger.info("开始转录音频数据...")
            result = self.asr_engine.transcribe(audio_data)
            if result:
                logger.info(f"转录结果: {result}")
            else:
                logger.warning("没有转录结果")
            return result
        except Exception as e:
            logger.error(f"转录音频数据失败: {e}")
            return None
    
    def transcribe_file(self, file_path: str) -> Optional[str]:
        """
        转录音频文件
        
        Args:
            file_path (str): 音频文件路径
            
        Returns:
            Optional[str]: 转录文本，如果失败则返回None
        """
        if not self.setup_successful:
            logger.error("ASR 引擎未初始化，无法转录文件")
            return None
        
        try:
            logger.info(f"开始转录文件: {file_path}")
            result = self.asr_engine.transcribe_file(file_path)
            if result:
                logger.info(f"转录结果: {result}")
            else:
                logger.warning("没有转录结果")
            return result
        except Exception as e:
            logger.error(f"转录文件失败: {e}")
            return None
    
    def get_final_result(self) -> Optional[str]:
        """
        获取最终结果
        
        Returns:
            Optional[str]: 最终识别结果文本，如果失败则返回None
        """
        if not self.setup_successful:
            logger.error("ASR 引擎未初始化，无法获取最终结果")
            return None
        
        try:
            logger.info("获取最终结果...")
            result = self.asr_engine.get_final_result()
            if result:
                logger.info(f"最终结果: {result}")
            else:
                logger.warning("没有最终结果")
            return result
        except Exception as e:
            logger.error(f"获取最终结果失败: {e}")
            return None