"""
ASR模型管理模块
负责加载和管理ASR模型
"""
import os
import logging
import traceback
import numpy as np
import vosk
from typing import Optional, Dict, Any, Union, List
from PyQt5.QtCore import QObject, pyqtSignal

# 信号管理器类
class SignalManager(QObject):
    """信号管理器类，用于管理ASR相关的信号"""

    # 定义信号
    new_text = pyqtSignal(str)  # 新文本信号，参数为文本内容
    status_updated = pyqtSignal(str)  # 状态更新信号，参数为状态信息
    error_occurred = pyqtSignal(str)  # 错误信号，参数为错误信息

    def __init__(self):
        """初始化信号管理器"""
        super().__init__()

# 配置日志
logger = logging.getLogger(__name__)
if not logger.handlers:
    # 创建处理器
    file_handler = logging.FileHandler('logs/asr_model_manager.log', encoding='utf-8')
    console_handler = logging.StreamHandler()

    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 设置日志级别
    logger.setLevel(logging.INFO)

from src.utils.config_manager import config_manager
from .vosk_engine import VoskASR
from .sherpa_engine import SherpaOnnxASR

# 导入 sherpa_onnx 模块
try:
    import sherpa_onnx
    HAS_SHERPA_ONNX = True
except ImportError:
    HAS_SHERPA_ONNX = False
    print("警告: 未安装 sherpa_onnx 模块，Sherpa-ONNX 功能将不可用")

class ASRModelManager(QObject):
    """ASR模型管理器类，作为插件适配器"""
    # 定义信号
    model_loaded = pyqtSignal(bool)  # 模型加载完成信号，参数为是否成功
    model_load_progress = pyqtSignal(int)  # 模型加载进度信号
    model_changed = pyqtSignal(str)  # 模型更改信号

    def __init__(self, config: Dict[str, Any] = None):
        """初始化ASR模型管理器"""
        super().__init__()  # 调用父类构造函数

        self.config = config or config_manager.config
        self.models_config = self.config.get("asr", {}).get("models", {})

        # 创建信号管理器
        self.signals = SignalManager()

        # 添加插件管理器
        from src.core.plugins import PluginManager
        self.plugin_manager = PluginManager()

        # 保持现有属性
        self.current_model_type = None
        self.current_engine = None
        self.model_type = None

        # 初始化插件系统
        self._initialize_plugins()

    def _load_plugin_config(self):
        """加载插件配置"""
        try:
            import json
            with open("config/plugins.json", "r", encoding="utf-8") as f:
                self.plugin_config = json.load(f)
            logger.info("插件配置加载成功")
        except Exception as e:
            logger.error(f"加载插件配置失败: {str(e)}")
            self.plugin_config = {}

    def _check_plugin_availability(self, plugin_name: str) -> bool:
        """检查插件是否可用并启用"""
        if not self.plugin_config:
            return False
            
        plugin_info = self.plugin_config.get("asr", {}).get(plugin_name, {})
        if not plugin_info:
            logger.error(f"插件 {plugin_name} 未在配置中定义")
            return False
            
        if not plugin_info.get("enabled", False):
            logger.warning(f"插件 {plugin_name} 已禁用")
            return False
            
        return True

    def _initialize_plugins(self):
        """初始化插件系统"""
        logger.info("初始化ASR插件系统")
        try:
            # 加载插件配置
            self._load_plugin_config()
            
            # 加载可用的ASR插件
            available_plugins = self.plugin_manager.get_available_plugins("asr")
            logger.info(f"可用的ASR插件: {available_plugins}")
            
            # 检查并加载已启用的插件
            enabled_plugins = []
            for plugin_name in available_plugins:
                if self._check_plugin_availability(plugin_name):
                    try:
                        self.plugin_manager.load_plugin("asr", plugin_name)
                        enabled_plugins.append(plugin_name)
                        logger.info(f"成功加载插件: {plugin_name}")
                    except Exception as e:
                        logger.error(f"加载插件 {plugin_name} 失败: {str(e)}")
                        
            logger.info(f"已启用的ASR插件: {enabled_plugins}")
            
        except Exception as e:
            logger.error(f"初始化插件系统失败: {str(e)}")
            logger.error(traceback.format_exc())

    def load_model(self, model_name: str) -> bool:
        """通过插件加载ASR模型"""
        logger.info(f"尝试加载模型: {model_name}")
        try:
            # 获取对应的插件
            plugin = self.plugin_manager.get_plugin(model_name)
            if not plugin:
                logger.error(f"未找到模型 {model_name} 对应的插件")
                return False

            # 通过插件加载模型
            success = plugin.setup()
            if success:
                self.current_model_type = model_name
                self.current_engine = plugin
                self.model_type = model_name
                logger.info(f"模型 {model_name} 加载成功")
                self.model_loaded.emit(True)
                return True
            else:
                logger.error(f"模型 {model_name} 加载失败")
                self.model_loaded.emit(False)
                return False

        except Exception as e:
            logger.error(f"加载模型 {model_name} 时发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            self.model_loaded.emit(False)
            return False

    def create_recognizer(self):
        """通过当前插件创建识别器"""
        if not self.current_engine:
            logger.error("当前没有加载任何模型")
            return None

        try:
            return self.current_engine.create_recognizer()
        except Exception as e:
            logger.error(f"创建识别器失败: {str(e)}")
            return None

    def get_current_engine_type(self) -> Optional[str]:
        """获取当前引擎类型"""
        if not self.current_engine:
            return None
        return self.current_model_type

    def transcribe(self, audio_data: Union[bytes, np.ndarray]) -> Optional[str]:
        """转录音频数据

        Args:
            audio_data: 音频数据，可以是字节或 numpy 数组

        Returns:
            str: 转录文本，如果失败则返回 None
        """
        if not self.current_engine:
            logger.error("当前没有加载任何模型")
            return None

        try:
            return self.current_engine.transcribe(audio_data)
        except Exception as e:
            logger.error(f"转录失败: {str(e)}")
            return None

    def transcribe_file(self, file_path: str) -> Optional[str]:
        """转录音频文件

        Args:
            file_path: 音频文件路径

        Returns:
            str: 转录文本，如果失败则返回 None
        """
        if not self.current_engine:
            logger.error("当前没有加载任何模型")
            return None

        try:
            return self.current_engine.transcribe_file(file_path)
        except Exception as e:
            logger.error(f"转录文件失败: {str(e)}")
            return None

    def reset(self) -> None:
        """重置当前引擎状态"""
        if self.current_engine:
            self.current_engine.reset()

    def get_final_result(self) -> Optional[str]:
        """获取最终识别结果

        Returns:
            str: 最终识别文本，如果失败则返回 None
        """
        if not self.current_engine:
            return None

        return self.current_engine.get_final_result()

    def get_available_engines(self) -> Dict[str, bool]:
        """获取可用的引擎列表

        Returns:
            Dict[str, bool]: 引擎名称到是否启用的映射
        """
        return self.plugin_manager.get_available_plugins("asr")

    def get_audio_devices(self) -> List[Dict[str, Any]]:
        """获取可用的音频设备列表

        Returns:
            List[Dict[str, Any]]: 音频设备列表
        """
        try:
            import soundcard as sc

            # 获取所有扬声器（输出设备）
            speakers = sc.all_speakers()

            # 获取所有麦克风（输入设备）
            microphones = sc.all_microphones(include_loopback=True)

            # 合并设备列表
            devices = []

            # 添加扬声器
            for i, speaker in enumerate(speakers):
                devices.append({
                    'index': f"speaker_{i}",
                    'id': speaker.id,
                    'name': f"[输出] {speaker.name}",
                    'channels': 2,  # 假设立体声
                    'sample_rate': 44100,  # 假设标准采样率
                    'is_input': False
                })

            # 添加麦克风
            for i, mic in enumerate(microphones):
                devices.append({
                    'index': f"mic_{i}",
                    'id': mic.id,
                    'name': f"[输入] {mic.name}",
                    'channels': 1,  # 假设单声道
                    'sample_rate': 44100,  # 假设标准采样率
                    'is_input': True
                })

            logger.info(f"找到 {len(devices)} 个音频设备")
            return devices

        except Exception as e:
            logger.error(f"获取音频设备列表失败: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def set_audio_device(self, device: Dict[str, Any]) -> bool:
        """设置音频设备

        Args:
            device: 音频设备信息

        Returns:
            bool: 是否设置成功
        """
        try:
            if not device:
                logger.error("音频设备为空")
                self.signals.error_occurred.emit("音频设备为空")
                return False

            logger.info(f"设置音频设备: {device.get('name', '未知设备')}")
            self.current_device = device

            # 更新状态
            self.signals.status_updated.emit(f"已选择设备: {device.get('name', '未知设备')}")

            return True
        except Exception as e:
            error_msg = f"设置音频设备失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.signals.error_occurred.emit(error_msg)
            return False

    def start_recognition(self) -> bool:
        """开始识别

        Returns:
            bool: 是否成功启动识别
        """
        try:
            # 检查引擎是否已初始化
            if not self.current_engine:
                logger.error("ASR引擎未初始化")
                self.signals.error_occurred.emit("ASR引擎未初始化，请先加载模型")
                return False

            # 检查设备是否已设置
            if not self.current_device:
                logger.error("未设置音频设备")
                self.signals.error_occurred.emit("未设置音频设备，请先选择音频设备")
                return False

            # 检查是否已经在识别
            if self.is_recognizing:
                logger.warning("识别已经在进行中")
                return True

            # 开始识别
            logger.info("开始识别")
            self.is_recognizing = True

            # 发射识别开始信号
            self.signals.status_updated.emit("识别已开始")

            return True
        except Exception as e:
            error_msg = f"启动识别失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.signals.error_occurred.emit(error_msg)
            return False

    def stop_recognition(self) -> bool:
        """停止识别

        Returns:
            bool: 是否成功停止识别
        """
        try:
            # 检查是否正在识别
            if not self.is_recognizing:
                logger.warning("识别未在进行中")
                return True

            # 停止识别
            logger.info("停止识别")
            self.is_recognizing = False

            # 发射识别停止信号
            self.signals.status_updated.emit("识别已停止")

            return True
        except Exception as e:
            error_msg = f"停止识别失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.signals.error_occurred.emit(error_msg)
            return False

    def get_available_models(self) -> List[str]:
        """获取所有可用的模型列表"""
        return list(self.plugin_manager.get_loaded_plugins())
        
    def validate_model_files(self, model_type: str) -> bool:
        """通过插件验证模型文件"""
        plugin = self.plugin_manager.get_plugin(model_type)
        if not plugin:
            logger.error(f"未找到模型 {model_type} 对应的插件")
            return False
            
        try:
            return plugin.validate_files()
        except Exception as e:
            logger.error(f"验证模型文件失败: {str(e)}")
            return False
    
    def transcribe_file(self, file_path: str) -> bool:
        """通过插件转录文件"""
        if not self.current_engine:
            logger.error("当前没有加载任何模型")
            return False
            
        try:
            return self.current_engine.transcribe_file(file_path)
        except Exception as e:
            logger.error(f"文件转录失败: {str(e)}")
            return False
    
    def unload_current_model(self):
        """卸载当前模型"""
        if self.current_engine:
            try:
                self.current_engine.cleanup()
            except Exception as e:
                logger.error(f"卸载模型时发生错误: {str(e)}")
            finally:
                self.current_engine = None
                self.current_model_type = None
                self.model_type = None
                
    def __del__(self):
        """确保在对象销毁时清理资源"""
        self.unload_current_model()
