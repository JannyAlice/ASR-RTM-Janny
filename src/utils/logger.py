"""
日志模块
提供统一的日志配置和访问
"""
import os
import sys
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional, Union, List

# 日志级别映射
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

class LogManager:
    """日志管理器类，单例模式"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.loggers = {}
        self._log_dir = "logs"
        self._default_level = logging.INFO
        self._max_file_size = 10 * 1024 * 1024  # 10MB
        self._backup_count = 5

        # 确保日志目录存在
        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)

        self._initialized = True

    def configure(self, log_dir=None, default_level=None, max_file_size=None, backup_count=None):
        """配置日志系统

        Args:
            log_dir: 日志目录
            default_level: 默认日志级别
            max_file_size: 最大日志文件大小（字节）
            backup_count: 备份文件数量
        """
        if log_dir:
            self._log_dir = log_dir
            if not os.path.exists(self._log_dir):
                os.makedirs(self._log_dir)

        if default_level is not None:
            if isinstance(default_level, str):
                self._default_level = LOG_LEVELS.get(default_level.upper(), logging.INFO)
            else:
                self._default_level = default_level

        if max_file_size is not None:
            self._max_file_size = max_file_size

        if backup_count is not None:
            self._backup_count = backup_count

    def get_logger(self, name: str, level: Optional[Union[str, int]] = None,
                  file_name: Optional[str] = None) -> logging.Logger:
        """获取或创建日志记录器

        Args:
            name: 日志记录器名称
            level: 日志级别
            file_name: 日志文件名，默认为name.log

        Returns:
            logging.Logger: 日志记录器
        """
        if name in self.loggers:
            return self.loggers[name]

        # 创建新的日志记录器
        logger = logging.getLogger(name)

        # 设置日志级别
        logger_level = level if level is not None else self._default_level
        if isinstance(logger_level, str):
            logger_level = LOG_LEVELS.get(logger_level.upper(), self._default_level)

        logger.setLevel(logger_level)

        # 如果已经有处理器，不再添加
        if logger.handlers:
            self.loggers[name] = logger
            return logger

        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logger_level)

        # 创建文件处理器
        if file_name is None:
            file_name = f"{name}.log"
        log_file = os.path.join(self._log_dir, file_name)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self._max_file_size,
            backupCount=self._backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logger_level)

        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # 添加处理器
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        # 缓存日志记录器
        self.loggers[name] = logger

        return logger

    def log_system_info(self, logger_name: str = "system"):
        """记录系统信息

        Args:
            logger_name: 日志记录器名称
        """
        logger = self.get_logger(logger_name)

        # 记录系统信息
        logger.info("=" * 50)
        logger.info(f"系统启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Python版本: {sys.version}")
        logger.info(f"平台: {sys.platform}")

        # 记录Qt信息
        try:
            from src.utils.qt_compat import get_qt_version_info
            qt_info = get_qt_version_info()
            logger.info(f"Qt版本: {qt_info['qt_version']}")
            logger.info(f"Qt绑定: {qt_info['binding']} {qt_info['binding_version']}")
        except ImportError:
            logger.warning("无法导入Qt兼容性模块")

        logger.info("=" * 50)

    def shutdown(self):
        """关闭所有日志处理器"""
        for name, logger in self.loggers.items():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

        self.loggers.clear()
        logging.shutdown()

    def get_log_files(self) -> List[str]:
        """获取所有日志文件路径

        Returns:
            List[str]: 日志文件路径列表
        """
        log_files = []
        for name, logger in self.loggers.items():
            for handler in logger.handlers:
                if isinstance(handler, (logging.FileHandler, RotatingFileHandler)):
                    log_files.append(handler.baseFilename)
        return log_files

# 创建全局单例实例
log_manager = LogManager()

# 便捷函数
def get_logger(name: str, level: Optional[Union[str, int]] = None,
               file_name: Optional[str] = None) -> logging.Logger:
    """获取日志记录器的便捷函数

    Args:
        name: 日志记录器名称
        level: 日志级别
        file_name: 日志文件名，默认为name.log

    Returns:
        logging.Logger: 日志记录器
    """
    return log_manager.get_logger(name, level, file_name)

def configure_logging(log_dir: Optional[str] = None,
                     default_level: Optional[Union[str, int]] = None,
                     max_file_size: Optional[int] = None,
                     backup_count: Optional[int] = None):
    """配置日志系统的便捷函数

    Args:
        log_dir: 日志目录
        default_level: 默认日志级别
        max_file_size: 最大日志文件大小（字节）
        backup_count: 备份文件数量
    """
    log_manager.configure(log_dir, default_level, max_file_size, backup_count)

def log_system_info(logger_name: str = "system"):
    """记录系统信息的便捷函数

    Args:
        logger_name: 日志记录器名称
    """
    log_manager.log_system_info(logger_name)

# 为了兼容旧代码，保留Logger类
class Logger:
    """兼容旧版本的日志管理器类"""

    def __init__(self, name: str = "transcript", log_dir: str = "logs"):
        """初始化日志管理器

        Args:
            name: 日志名称
            log_dir: 日志目录
        """
        self.name = name
        self.log_dir = log_dir
        self.logger = log_manager.get_logger(name)

    def debug(self, message: str) -> None:
        """记录调试信息"""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """记录信息"""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """记录警告信息"""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """记录错误信息"""
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """记录严重错误信息"""
        self.logger.critical(message)

    def get_log_file(self) -> Optional[str]:
        """获取当前日志文件路径"""
        for handler in self.logger.handlers:
            if isinstance(handler, (logging.FileHandler, RotatingFileHandler)):
                return handler.baseFilename
        return None

# 导出
__all__ = [
    'LogManager',
    'log_manager',
    'get_logger',
    'configure_logging',
    'log_system_info',
    'LOG_LEVELS',
    'Logger'
]
