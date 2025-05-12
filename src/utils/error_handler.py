"""
全局异常处理模块
负责捕获和处理整个应用程序的未捕获异常
"""
import sys
import logging
import traceback
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QObject, pyqtSignal

# 配置日志
logger = logging.getLogger(__name__)

class GlobalExceptionHandler(QObject):
    """全局异常处理器"""
    
    # 定义错误信号
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        """初始化异常处理器"""
        super().__init__()
        # 设置全局异常钩子
        sys.excepthook = self.handle_exception
        logger.info("全局异常处理器已初始化")
    
    def handle_exception(self, exc_type, exc_value, exc_traceback):
        """处理未捕获的异常"""
        try:
            # 构建错误消息
            error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            
            # 记录日志
            logger.error("未捕获的异常:\n%s", error_msg)
            
            # 发送错误信号
            self.error_occurred.emit(str(exc_value))
            
            # 显示错误对话框
            QMessageBox.critical(
                None,
                "错误",
                f"发生未处理的错误:\n{str(exc_value)}\n\n请查看日志获取详细信息。"
            )
            
        except Exception as e:
            # 如果在处理异常时发生错误，确保它不会被吞掉
            print(f"错误处理器失败: {e}")
            print(f"原始异常: {exc_value}")

# 创建全局异常处理器实例
exception_handler = GlobalExceptionHandler()