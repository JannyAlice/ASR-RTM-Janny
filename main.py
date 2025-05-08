#!/usr/bin/env python3

import os
# Qt 和 COM 环境变量设置必须在最开始
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"  # 启用自动缩放
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"    # 禁用高DPI缩放
os.environ["PYTHONCOM_INITIALIZE"] = "0"          # 禁止 pythoncom 自动初始化

"""
实时语音转录应用程序
主要功能：
1. 实时语音识别（ASR）
2. 字幕显示
3. 系统音频捕获
4. 配置文件管理
"""

# 然后导入其他库
import sys
from pathlib import Path
import logging
import traceback
import json
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication

# 确保能够导入src目录下的模块
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 配置日志目录
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'main.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# COM初始化标志
_com_initialized = False

def initialize_com():
    """初始化COM环境"""
    global _com_initialized
    if not _com_initialized:
        try:
            import pythoncom
            # 在UI线程中使用单线程模式
            pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)
            _com_initialized = True
            logger.info("COM环境初始化成功")
        except Exception as e:
            logger.error(f"COM环境初始化失败: {str(e)}")
            raise

def uninitialize_com():
    """清理COM环境"""
    global _com_initialized
    if _com_initialized:
        try:
            import pythoncom
            pythoncom.CoUninitialize()
            _com_initialized = False
            logger.info("COM环境清理完成")
        except Exception as e:
            logger.error(f"COM环境清理失败: {str(e)}")

# 推迟导入以避免循环依赖
# 使用正确的导入路径
from src.utils.config_manager import ConfigManager
from src.core.plugins import PluginManager, PluginRegistry
from src.core.asr import ASRModelManager
from src.ui.main_window import MainWindow

def main():
    """主程序入口"""
    app = None
    try:
        # 1. 在创建任何Qt对象之前初始化COM
        initialize_com()
        
        # 2. 创建QApplication实例
        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(True)
        
        # 3. 加载配置
        config_manager = ConfigManager()
        config = config_manager.load_config()
        logger.info("配置加载成功")
        
        # 4. 初始化插件系统
        plugin_manager = PluginManager()
        plugin_manager.configure(config)
        
        # 5. 获取并配置插件注册表
        plugin_registry = plugin_manager.get_registry()
        
        # 6. 注册插件
        from src.core.plugins.asr.vosk_plugin import VoskPlugin
        plugin_registry.register("vosk_small", VoskPlugin)
        
        # 7. 创建并配置ASR管理器
        asr_manager = ASRModelManager(config)
        
        # 8. 加载默认模型
        default_model = config["asr"].get("default_model", "vosk_small")
        if not asr_manager.load_model(default_model):
            logger.error(f"加载默认模型失败: {default_model}")
            return 1
            
        # 9. 创建主窗口
        window = MainWindow(
            model_manager=asr_manager,
            config=config
        )
        window.setAttribute(Qt.WA_DeleteOnClose)  # 确保窗口关闭时被删除
        window.show()

        # 10. 进入事件循环
        return app.exec()
        
    except Exception as e:
        logger.error(f"程序运行错误: {str(e)}")
        traceback.print_exc()
        return 1
    finally:
        # 确保按正确的顺序清理资源
        if app and app.thread().isRunning():
            app.quit()
        uninitialize_com()

if __name__ == "__main__":
    sys.exit(main())
