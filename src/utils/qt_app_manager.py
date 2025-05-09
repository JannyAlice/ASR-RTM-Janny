"""
Qt应用程序管理模块
提供Qt应用程序的初始化、管理和清理功能
"""
import os
import sys
import logging
from typing import Optional, List, Dict, Any, Union

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

logger = logging.getLogger(__name__)

class QtAppManager:
    """Qt应用程序管理器类，单例模式"""
    _instance = None
    _app = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QtAppManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._com_initialized = False
        
    def set_environment_variables(self):
        """设置Qt环境变量"""
        # 设置Qt环境变量
        os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"  # 启用自动缩放
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"    # 禁用高DPI缩放
        os.environ["PYTHONCOM_INITIALIZE"] = "0"         # 禁止pythoncom自动初始化
        
        logger.info("已设置Qt环境变量")
        
    def initialize_com(self):
        """初始化COM环境"""
        if not self._com_initialized:
            try:
                import pythoncom
                # 在UI线程中使用单线程模式
                pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)
                self._com_initialized = True
                logger.info("COM环境初始化成功")
            except Exception as e:
                # 检查是否是因为COM已经初始化导致的错误
                error_msg = str(e).lower()
                if "already initialized" in error_msg or "cannot change thread mode" in error_msg:
                    logger.info("COM环境已经初始化，继续执行")
                    self._com_initialized = True
                    return
                else:
                    logger.error(f"COM环境初始化失败: {str(e)}")
                    raise
                    
    def uninitialize_com(self):
        """清理COM环境"""
        if self._com_initialized:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
                self._com_initialized = False
                logger.info("COM环境清理完成")
            except Exception as e:
                logger.error(f"COM环境清理失败: {str(e)}")
                
    def create_application(self, args=None):
        """创建Qt应用程序实例
        
        Args:
            args: 命令行参数，默认为sys.argv
            
        Returns:
            QApplication: Qt应用程序实例
        """
        if QtAppManager._app is not None:
            logger.warning("Qt应用程序实例已存在")
            return QtAppManager._app
            
        # 使用传入的参数或系统参数
        if args is None:
            args = sys.argv
            
        # 创建应用程序实例
        QtAppManager._app = QApplication(args)
        QtAppManager._app.setQuitOnLastWindowClosed(True)
        
        logger.info("已创建Qt应用程序实例")
        return QtAppManager._app
        
    def get_application(self):
        """获取Qt应用程序实例
        
        Returns:
            QApplication: Qt应用程序实例，如果不存在则创建
        """
        if QtAppManager._app is None:
            return self.create_application()
        return QtAppManager._app
        
    def exec_application(self):
        """运行Qt应用程序事件循环
        
        Returns:
            int: 应用程序退出代码
        """
        app = self.get_application()
        
        # PyQt5 使用 exec_()，而 PySide6 使用 exec()
        if hasattr(app, 'exec_'):
            return app.exec_()
        else:
            return app.exec()
            
    def cleanup(self):
        """清理资源"""
        # 清理应用程序
        if QtAppManager._app:
            if QtAppManager._app.thread().isRunning():
                QtAppManager._app.quit()
            QtAppManager._app = None
            
        # 清理COM环境
        self.uninitialize_com()
        
        logger.info("Qt应用程序资源已清理")
        
    def get_screen_info(self) -> List[Dict[str, Any]]:
        """获取屏幕信息
        
        Returns:
            List[Dict[str, Any]]: 屏幕信息列表
        """
        app = self.get_application()
        screens = []
        
        for i, screen in enumerate(app.screens()):
            geometry = screen.geometry()
            screens.append({
                "index": i,
                "name": screen.name(),
                "geometry": {
                    "x": geometry.x(),
                    "y": geometry.y(),
                    "width": geometry.width(),
                    "height": geometry.height()
                },
                "dpi": screen.physicalDotsPerInch(),
                "primary": screen.virtualGeometry() == app.primaryScreen().virtualGeometry()
            })
            
        return screens
        
    def set_application_style(self, style_name: str) -> bool:
        """设置应用程序样式
        
        Args:
            style_name: 样式名称
            
        Returns:
            bool: 设置是否成功
        """
        app = self.get_application()
        
        try:
            app.setStyle(style_name)
            logger.info(f"已设置应用程序样式: {style_name}")
            return True
        except Exception as e:
            logger.error(f"设置应用程序样式失败: {str(e)}")
            return False

# 创建全局单例实例
qt_app_manager = QtAppManager()

# 便捷函数
def get_application():
    """获取Qt应用程序实例的便捷函数"""
    return qt_app_manager.get_application()

def initialize_qt():
    """初始化Qt环境的便捷函数"""
    qt_app_manager.set_environment_variables()
    qt_app_manager.initialize_com()
    return qt_app_manager.create_application()

# 导出
__all__ = [
    'QtAppManager',
    'qt_app_manager',
    'get_application',
    'initialize_qt'
]
