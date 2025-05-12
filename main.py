#!/usr/bin/env python3
"""
实时语音转录应用程序
主要功能：
1. 实时语音识别（ASR）
2. 字幕显示
3. 系统音频捕获
4. 配置文件管理
"""

import sys
import traceback
from pathlib import Path

# 确保能够导入src目录下的模块
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 初始化日志系统
from src.utils.logger import configure_logging, get_logger, log_system_info
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
configure_logging(
    log_dir=str(log_dir),
    default_level="INFO"
)
logger = get_logger("main")

# 记录系统启动信息
log_system_info()

# 导入Qt应用程序管理器
from src.utils.qt_app_manager import qt_app_manager, initialize_qt
from src.utils.qt_compat import log_qt_info

# 导入配置管理器和插件系统
from src.utils.config_manager import config_manager
from src.core.plugins import PluginManager
from src.core.asr import ASRModelManager
from src.ui.main_window import MainWindow

def main():
    """主程序入口"""
    try:
        # 1. 初始化Qt环境和应用程序
        app = initialize_qt()
        logger.info("Qt环境和应用程序初始化成功")
        log_qt_info(logger)

        # 2. 加载配置
        config = config_manager.load_config()
        logger.info("配置加载成功")

        # 3. 初始化插件系统
        plugin_manager = PluginManager()
        plugin_manager.configure(config)
        logger.info("插件系统初始化成功")

        # 4. 获取并配置插件注册表
        plugin_registry = plugin_manager.get_registry()
        logger.info("获取插件注册表成功")

        # 5. 注册插件
        from src.core.plugins.asr.vosk_plugin import VoskPlugin
        from src.core.plugins.asr.sherpa_onnx_plugin import SherpaOnnxPlugin

        # 注册Vosk插件
        plugin_registry.register("vosk_small", VoskPlugin)
        logger.info("注册Vosk插件成功")

        # 注册Sherpa-ONNX插件（支持所有sherpa-onnx系列模型）
        plugin_registry.register("sherpa_onnx_std", SherpaOnnxPlugin)
        plugin_registry.register("sherpa_onnx_int8", SherpaOnnxPlugin)
        plugin_registry.register("sherpa_0626_std", SherpaOnnxPlugin)
        plugin_registry.register("sherpa_0626_int8", SherpaOnnxPlugin)
        logger.info("注册Sherpa-ONNX插件成功")

        # 6. 创建ASR管理器
        asr_manager = ASRModelManager()
        logger.info("创建ASR管理器成功")

        # 7. 加载默认模型
        default_model = config_manager.get_default_model()
        if not asr_manager.load_model(default_model):
            logger.error(f"加载默认模型失败: {default_model}")
            return 1
        logger.info(f"加载默认模型成功: {default_model}")

        # 8. 创建主窗口
        from PyQt5.QtCore import Qt
        window = MainWindow(
            model_manager=asr_manager,
            config_manager=config_manager
        )
        # 确保窗口关闭时被删除
        try:
            window.setAttribute(Qt.WA_DeleteOnClose)
        except AttributeError:
            logger.warning("无法设置WA_DeleteOnClose属性")

        window.show()
        logger.info("主窗口创建并显示成功")

        # 9. 进入事件循环
        return qt_app_manager.exec_application()

    except Exception as e:
        logger.error(f"程序运行错误: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
    finally:
        # 确保按正确的顺序清理资源
        qt_app_manager.cleanup()
        logger.info("程序退出，资源已清理")

if __name__ == "__main__":
    sys.exit(main())
