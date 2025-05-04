"""
Sherpa-ONNX 日志工具模块
负责记录 Sherpa-ONNX 相关日志
"""
import os
import sys
import time  # 添加 time 模块导入
import logging
import datetime
from typing import Optional

class SherpaLogger:
    """Sherpa-ONNX 日志工具类"""

    def __init__(self, log_dir: str = "logs", log_level: int = logging.DEBUG):
        """
        初始化日志工具
        
        Args:
            log_dir: 日志目录
            log_level: 日志级别
        """
        # 创建日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 初始化日志记录器
        self.logger = logging.getLogger("sherpa")
        self.logger.setLevel(log_level)
        self.logger.propagate = False

        # 清除现有处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # 创建控制台处理器
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setLevel(logging.INFO)  # 控制台始终显示INFO级别
        console_formatter = logging.Formatter("%(message)s")
        self.console_handler.setFormatter(console_formatter)
        self.logger.addHandler(self.console_handler)
        
        # 创建文件处理器
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"sherpa_debug_{timestamp}.log")
        self.file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        self.file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.file_handler.setFormatter(file_formatter)
        self.logger.addHandler(self.file_handler)
        
        # 记录初始化信息
        self.logger.info(f"Sherpa-ONNX 日志文件: {self.log_file}")
        self.logger.info(f"日志级别: {logging.getLevelName(log_level)}")

    def get_log_file(self) -> Optional[str]:
        """
        获取日志文件路径

        Returns:
            str: 日志文件路径
        """
        return self.log_file

    def debug(self, message: str) -> None:
        """
        记录调试日志

        Args:
            message: 日志消息
        """
        if self.logger:
            self.logger.debug(message)

    def info(self, message: str) -> None:
        """
        记录信息日志

        Args:
            message: 日志消息
        """
        if self.logger:
            self.logger.info(message)

    def warning(self, message: str) -> None:
        """
        记录警告日志

        Args:
            message: 日志消息
        """
        if self.logger:
            self.logger.warning(message)

    def error(self, message: str) -> None:
        """
        记录错误日志

        Args:
            message: 日志消息
        """
        if self.logger:
            self.logger.error(message)

    def critical(self, message: str) -> None:
        """
        记录严重错误日志

        Args:
            message: 日志消息
        """
        if self.logger:
            self.logger.critical(message)

# 创建全局 Sherpa-ONNX 日志工具实例
sherpa_logger = SherpaLogger()
