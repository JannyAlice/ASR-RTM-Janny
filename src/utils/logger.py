import os
import logging
from datetime import datetime
from typing import Optional


class Logger:
    """日志管理器类"""
    
    def __init__(self, name: str = "transcript", log_dir: str = "logs"):
        """初始化日志管理器
        
        Args:
            name: 日志名称
            log_dir: 日志目录
        """
        self.name = name
        self.log_dir = log_dir
        self.logger = None
        self.setup()
        
    def setup(self) -> None:
        """设置日志管理器"""
        try:
            # 确保日志目录存在
            os.makedirs(self.log_dir, exist_ok=True)
            
            # 创建日志文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(self.log_dir, f"{self.name}_{timestamp}.log")
            
            # 配置日志记录器
            self.logger = logging.getLogger(self.name)
            self.logger.setLevel(logging.DEBUG)
            
            # 创建文件处理器
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            # 创建控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 创建格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # 添加处理器
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
        except Exception as e:
            print(f"Error setting up logger: {str(e)}")
            
    def debug(self, message: str) -> None:
        """记录调试信息
        
        Args:
            message: 调试信息
        """
        if self.logger:
            self.logger.debug(message)
            
    def info(self, message: str) -> None:
        """记录信息
        
        Args:
            message: 信息
        """
        if self.logger:
            self.logger.info(message)
            
    def warning(self, message: str) -> None:
        """记录警告信息
        
        Args:
            message: 警告信息
        """
        if self.logger:
            self.logger.warning(message)
            
    def error(self, message: str) -> None:
        """记录错误信息
        
        Args:
            message: 错误信息
        """
        if self.logger:
            self.logger.error(message)
            
    def critical(self, message: str) -> None:
        """记录严重错误信息
        
        Args:
            message: 严重错误信息
        """
        if self.logger:
            self.logger.critical(message)
            
    def get_log_file(self) -> Optional[str]:
        """获取当前日志文件路径
        
        Returns:
            str: 当前日志文件路径，如果未设置则返回 None
        """
        if not self.logger or not self.logger.handlers:
            return None
            
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return handler.baseFilename
        return None
