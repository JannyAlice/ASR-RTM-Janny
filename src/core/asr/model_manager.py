"""
ASR模型管理模块
负责加载和管理ASR模型
"""
import os
import json
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
    """ASR模型管理器类"""

    # 定义信号
    model_loaded = pyqtSignal(bool)  # 模型加载完成信号，参数为是否成功
    recognition_started = pyqtSignal()  # 识别开始信号
    recognition_stopped = pyqtSignal()  # 识别停止信号
    recognition_result = pyqtSignal(str)  # 识别结果信号，参数为识别文本
    error_occurred = pyqtSignal(str)  # 错误信号，参数为错误信息

    def __init__(self):
        """初始化ASR模型管理器"""
        super().__init__()  # 调用父类构造函数

        # 创建信号管理器
        self.signals = SignalManager()

        self.config = config_manager
        # 直接从 config.json 获取模型配置
        self.models_config = {}
        if 'asr' in self.config.config and 'models' in self.config.config['asr']:
            self.models_config = self.config.config['asr']['models']
        print(f"[DEBUG] ASRModelManager.__init__: models_config = {self.models_config}")
        print(f"[DEBUG] ASRModelManager.__init__: config = {self.config.config}")
        self.current_model = None
        self.model_path = None

        # 从配置文件获取默认模型类型
        self.model_type = self.config.config.get('transcription', {}).get('default_model', 'sherpa_0626')
        logger.info(f"使用默认模型类型: {self.model_type}")

        # 用于音频转录的引擎
        self.current_engine = None

        # 音频设备相关
        self.current_device = None
        self.is_recognizing = False

    def validate_model_files(self, model_path: str, model_type: str = None) -> bool:
        """
        验证模型文件完整性

        Args:
            model_path: 模型路径
            model_type: 模型类型，如果为None则使用self.model_type

        Returns:
            bool: 验证是否通过
        """
        try:
            # 如果没有传入model_type，则使用self.model_type
            if model_type is None:
                model_type = self.model_type

            # 根据模型路径判断是否为Vosk模型
            is_vosk_model = "vosk" in model_path.lower() or model_type == "vosk_small"

            # 根据模型类型选择不同的验证方式
            if is_vosk_model:
                # Vosk模型只需要检查目录是否存在
                if os.path.exists(model_path) and os.path.isdir(model_path):
                    logger.info(f"Vosk模型目录验证通过: {model_path}")
                    return True
                else:
                    logger.error(f"Vosk模型目录不存在或不是目录: {model_path}")
                    return False
            elif model_type.startswith("sherpa"):
                # 获取模型配置
                model_config = self.models_config.get(model_type, {})

                # 确定模型类型和文件名
                config_model_type = model_config.get("type", "int8").lower()
                model_name = model_config.get("name", "")

                # 确定是否使用int8模型和是否是0626模型
                is_int8 = config_model_type == "int8"
                is_0626 = "0626" in model_name or "2023-06-26" in model_path

                # 根据模型类型选择不同的文件名
                if is_0626:
                    # 针对sherpa_0626模型的具体文件名
                    required_files = [
                        "encoder-epoch-99-avg-1-chunk-16-left-128.onnx",
                        "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
                        "joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
                        "tokens.txt"
                    ]
                else:
                    # 针对其他Sherpa模型的具体文件名
                    required_files = [
                        "encoder-epoch-99-avg-1.int8.onnx" if is_int8 else "encoder-epoch-99-avg-1.onnx",
                        "decoder-epoch-99-avg-1.int8.onnx" if is_int8 else "decoder-epoch-99-avg-1.onnx",
                        "joiner-epoch-99-avg-1.int8.onnx" if is_int8 else "joiner-epoch-99-avg-1.onnx",
                        "tokens.txt"
                    ]

                # 检查每个文件是否存在
                for file in required_files:
                    file_path = os.path.join(model_path, file)
                    if not os.path.exists(file_path):
                        logger.error(f"模型文件不存在: {file_path}")
                        return False

                logger.info(f"Sherpa模型文件验证通过: {model_path}")
                return True
            else:
                # 未知模型类型，尝试检测模型类型
                if "vosk" in model_path.lower():
                    # 可能是Vosk模型
                    if os.path.exists(model_path) and os.path.isdir(model_path):
                        logger.info(f"检测到Vosk模型目录: {model_path}")
                        return True
                elif "sherpa" in model_path.lower() or "onnx" in model_path.lower():
                    # 可能是Sherpa模型，但无法确定具体类型，只检查目录是否存在
                    if os.path.exists(model_path) and os.path.isdir(model_path):
                        logger.info(f"检测到可能的Sherpa模型目录: {model_path}")
                        return True

                # 如果无法确定模型类型，只检查目录是否存在
                if os.path.exists(model_path) and os.path.isdir(model_path):
                    logger.info(f"模型目录存在: {model_path}")
                    return True
                else:
                    logger.error(f"模型目录不存在或不是目录: {model_path}")
                    return False

        except Exception as e:
            logger.error(f"验证模型文件时发生错误: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def load_model(self, model_name: str) -> bool:
        """加载ASR模型"""
        try:
            logger.info(f"开始加载模型: {model_name}")

            # 调试信息
            print(f"[DEBUG] 尝试加载模型: {model_name}")
            print(f"[DEBUG] 当前配置中的模型列表: {list(self.models_config.keys())}")

            # 检查模型是否存在
            if model_name not in self.models_config:
                logger.error(f"错误: 模型 {model_name} 在配置中不存在")
                return False

            # 获取模型配置
            model_config = self.models_config[model_name]
            print(f"[DEBUG] 找到模型配置: {model_name}")

            # 获取模型路径
            model_path = model_config.get('path', '')
            if not model_path:
                logger.error(f"错误: 模型 {model_name} 路径为空")
                return False

            print(f"[DEBUG] 模型路径: {model_path}")

            # 验证模型路径和文件
            if not os.path.exists(model_path):
                logger.error(f"错误: 模型路径不存在: {model_path}")
                return False

            # 记录当前尝试加载的模型类型
            logger.info(f"尝试加载模型类型: {model_name}，当前模型类型: {self.model_type}")

            # 验证模型文件时传入模型名称，确保使用正确的验证逻辑
            if not self.validate_model_files(model_path, model_name):
                logger.error(f"错误: 模型路径验证失败: {model_path}")
                return False

            # 更新当前模型信息
            self.current_model_type = model_name
            self.model_type = model_name  # 确保model_type与current_model_type一致

            # 初始化引擎
            if not self.initialize_engine(model_name):
                logger.error(f"初始化引擎失败: {model_name}")
                return False

            # 设置current_model为True，表示模型已加载
            self.current_model = True

            logger.info(f"模型加载成功: {model_name}")

            # 发射模型加载成功信号
            self.model_loaded.emit(True)

            # 更新状态
            self.signals.status_updated.emit(f"已加载模型: {model_name}")

            return True

        except Exception as e:
            error_msg = f"加载模型失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())

            # 发射模型加载失败信号
            self.model_loaded.emit(False)

            # 发射错误信号
            self.error_occurred.emit(error_msg)

            return False

    def _load_vosk_model(self, model_path: str) -> Any:
        """
        加载VOSK模型

        Args:
            model_path: 模型路径

        Returns:
            Any: VOSK模型实例
        """
        return vosk.Model(model_path)

    def _load_sherpa_model(self, model_path: str, model_config: Dict[str, Any]) -> Any:
        """
        加载Sherpa模型

        Args:
            model_path: 模型路径
            model_config: 模型配置

        Returns:
            Any: Sherpa模型实例
        """
        try:
            if not HAS_SHERPA_ONNX:
                print("未安装 sherpa_onnx 模块，无法加载 Sherpa-ONNX 模型")
                return None

            # 确定模型类型和文件名
            model_type = model_config.get("type", "int8").lower()
            model_name = model_config.get("name", "")

            # 确定是否使用int8模型
            is_int8 = model_type == "int8"
            is_0626 = "0626" in model_name or model_path and "2023-06-26" in model_path

            # 从配置文件中获取模型文件名
            config_section = model_config.get("config", {})

            # 如果配置文件中有指定模型文件名，则使用配置文件中的值
            if "encoder" in config_section and "decoder" in config_section and "joiner" in config_section:
                # 检查是否是完整路径，如果不是则拼接 model_path
                encoder_file = config_section["encoder"]
                decoder_file = config_section["decoder"]
                joiner_file = config_section["joiner"]

                # 如果不是绝对路径，则拼接 model_path
                if not os.path.isabs(encoder_file):
                    encoder_file = os.path.join(model_path, encoder_file)
                if not os.path.isabs(decoder_file):
                    decoder_file = os.path.join(model_path, decoder_file)
                if not os.path.isabs(joiner_file):
                    joiner_file = os.path.join(model_path, joiner_file)

                print(f"使用配置文件中指定的模型文件名: encoder={encoder_file}, decoder={decoder_file}, joiner={joiner_file}")
            else:
                # 否则使用默认值
                if is_0626:
                    # 使用0626模型的文件名
                    encoder_file = "encoder-epoch-99-avg-1-chunk-16-left-128.onnx"
                    decoder_file = "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"
                    joiner_file = "joiner-epoch-99-avg-1-chunk-16-left-128.onnx"
                    print(f"使用0626模型的默认文件名: encoder={encoder_file}, decoder={decoder_file}, joiner={joiner_file}")
                else:
                    # 使用现有模型的文件名
                    encoder_file = "encoder-epoch-99-avg-1.int8.onnx" if is_int8 else "encoder-epoch-99-avg-1.onnx"
                    decoder_file = "decoder-epoch-99-avg-1.int8.onnx" if is_int8 else "decoder-epoch-99-avg-1.onnx"
                    joiner_file = "joiner-epoch-99-avg-1.int8.onnx" if is_int8 else "joiner-epoch-99-avg-1.onnx"
                    print(f"使用现有模型的默认文件名: encoder={encoder_file}, decoder={decoder_file}, joiner={joiner_file}")

            # 检查模型文件是否存在
            # 获取 tokens 文件路径
            tokens_file = config_section.get("tokens", "tokens.txt")
            if not os.path.isabs(tokens_file):
                tokens_file = os.path.join(model_path, tokens_file)

            required_files = [
                encoder_file,  # 已经在上面处理过路径
                decoder_file,  # 已经在上面处理过路径
                joiner_file,   # 已经在上面处理过路径
                tokens_file
            ]

            for file_path in required_files:
                if not os.path.exists(file_path):
                    print(f"模型文件不存在: {file_path}")
                    return None

            # 使用 OnlineRecognizer 类的 from_transducer 静态方法创建实例
            # 这是 sherpa-onnx 1.11.2 版本的正确 API
            try:
                # 从配置中获取参数
                num_threads = config_section.get("num_threads", 4)
                sample_rate = config_section.get("sample_rate", 16000)
                feature_dim = config_section.get("feature_dim", 80)
                decoding_method = config_section.get("decoding_method", "greedy_search")

                print(f"开始创建 OnlineRecognizer 实例...")
                print(f"encoder: {encoder_file}")
                print(f"decoder: {decoder_file}")
                print(f"joiner: {joiner_file}")
                print(f"tokens: {tokens_file}")
                print(f"num_threads: {num_threads}")
                print(f"sample_rate: {sample_rate}")
                print(f"feature_dim: {feature_dim}")
                print(f"decoding_method: {decoding_method}")

                model = sherpa_onnx.OnlineRecognizer.from_transducer(
                    encoder=encoder_file,  # 已经是完整路径
                    decoder=decoder_file,  # 已经是完整路径
                    joiner=joiner_file,    # 已经是完整路径
                    tokens=tokens_file,     # 已经是完整路径
                    num_threads=num_threads,
                    sample_rate=sample_rate,
                    feature_dim=feature_dim,
                    decoding_method=decoding_method
                )
                print("成功创建 OnlineRecognizer 实例")
            except Exception as e:
                print(f"使用 from_transducer 创建实例失败: {e}")
                import traceback
                print(traceback.format_exc())
                return None

            model_type_str = "int8量化" if is_int8 else "标准"
            print(f"成功加载Sherpa-ONNX {model_type_str}模型: {model_path}")

            return model

        except Exception as e:
            print(f"加载Sherpa模型失败: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def _validate_model_path(self, model_path: str, model_type: str = None) -> bool:
        """
        验证模型路径是否有效

        Args:
            model_path: 模型路径
            model_type: 模型类型，如果为None则使用self.model_type

        Returns:
            bool: 验证是否通过
        """
        try:
            if not model_path:
                logger.error("模型路径为空")
                return False

            if not os.path.exists(model_path):
                logger.error(f"模型路径不存在: {model_path}")
                return False

            # 如果没有传入model_type，则使用self.model_type
            if model_type is None:
                model_type = self.model_type

            # 根据模型路径判断是否为Vosk模型
            is_vosk_model = "vosk" in model_path.lower() or model_type == "vosk_small"

            # 根据模型类型选择不同的验证方式
            if is_vosk_model:
                # Vosk模型只需要检查目录是否存在
                if os.path.exists(model_path) and os.path.isdir(model_path):
                    # 检查是否存在 model 文件夹，这是 Vosk 模型的特征
                    model_folder = os.path.join(model_path, "model")
                    if os.path.exists(model_folder) and os.path.isdir(model_folder):
                        logger.info(f"Vosk模型目录验证通过: {model_path}")
                        return True
                    else:
                        # 检查是否存在 final.mdl 文件，这也是 Vosk 模型的特征
                        final_mdl = os.path.join(model_path, "final.mdl")
                        if os.path.exists(final_mdl):
                            logger.info(f"Vosk模型目录验证通过: {model_path}")
                            return True
                        else:
                            # 对于Vosk模型，只要目录存在就认为验证通过
                            logger.info(f"Vosk模型目录存在: {model_path}")
                            return True
                else:
                    logger.error(f"Vosk模型目录不存在或不是目录: {model_path}")
                    return False
            elif model_type and model_type.startswith('sherpa'):
                # 获取模型配置
                model_config = self.models_config.get(model_type, {})

                # 确定模型类型和文件名
                config_model_type = model_config.get("type", "int8").lower()
                model_name = model_config.get("name", "")

                # 确定是否使用int8模型和是否是0626模型
                is_int8 = config_model_type == "int8"
                is_0626 = "0626" in model_name or "2023-06-26" in model_path

                # 根据模型类型选择不同的文件名
                if is_0626:
                    # 针对sherpa_0626模型的具体文件名
                    required_files = [
                        "encoder-epoch-99-avg-1-chunk-16-left-128.onnx" if not is_int8 else "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
                        "decoder-epoch-99-avg-1-chunk-16-left-128.onnx" if not is_int8 else "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
                        "joiner-epoch-99-avg-1-chunk-16-left-128.onnx" if not is_int8 else "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
                        "tokens.txt"
                    ]
                else:
                    # 针对其他Sherpa模型的具体文件名
                    required_files = [
                        "encoder-epoch-99-avg-1.int8.onnx" if is_int8 else "encoder-epoch-99-avg-1.onnx",
                        "decoder-epoch-99-avg-1.int8.onnx" if is_int8 else "decoder-epoch-99-avg-1.onnx",
                        "joiner-epoch-99-avg-1.int8.onnx" if is_int8 else "joiner-epoch-99-avg-1.onnx",
                        "tokens.txt"
                    ]

                # 检查每个文件是否存在
                for file in required_files:
                    file_path = os.path.join(model_path, file)
                    if not os.path.exists(file_path):
                        logger.error(f"模型文件不存在: {file_path}")
                        return False

                logger.info(f"Sherpa模型文件验证通过: {model_path}")
                return True
            else:
                # 未知模型类型，尝试检测模型类型
                if "vosk" in model_path.lower():
                    # 可能是Vosk模型
                    if os.path.exists(model_path) and os.path.isdir(model_path):
                        logger.info(f"检测到Vosk模型目录: {model_path}")
                        return True
                elif "sherpa" in model_path.lower() or "onnx" in model_path.lower():
                    # 可能是Sherpa模型，但无法确定具体类型，只检查目录是否存在
                    if os.path.exists(model_path) and os.path.isdir(model_path):
                        logger.info(f"检测到可能的Sherpa模型目录: {model_path}")
                        return True

                # 如果无法确定模型类型，只检查目录是否存在
                if os.path.exists(model_path) and os.path.isdir(model_path):
                    logger.info(f"模型目录存在: {model_path}")
                    return True
                else:
                    logger.error(f"模型目录不存在或不是目录: {model_path}")
                    return False

        except Exception as e:
            logger.error(f"验证模型路径时发生错误: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def create_recognizer(self) -> Optional[Any]:
        """
        创建识别器

        Returns:
            Optional[Any]: 识别器实例
        """
        try:
            # 导入 Sherpa-ONNX 日志工具
            try:
                from src.utils.sherpa_logger import sherpa_logger
            except ImportError:
                # 如果导入失败，创建一个简单的日志记录器
                class DummyLogger:
                    def debug(self, msg): print(f"DEBUG: {msg}")
                    def info(self, msg): print(f"INFO: {msg}")
                    def warning(self, msg): print(f"WARNING: {msg}")
                    def error(self, msg): print(f"ERROR: {msg}")
                sherpa_logger = DummyLogger()

            # 检查当前模型是否已加载
            if self.current_model is None:
                logger.error("当前模型未加载，请先加载模型")
                sherpa_logger.error("当前模型未加载，请先加载模型")
                return None

            # 检查模型配置
            model_config = self.models_config.get(self.model_type, {})
            if not model_config:
                logger.error(f"模型配置 {self.model_type} 不存在")
                sherpa_logger.error(f"模型配置 {self.model_type} 不存在")
                return None

            # 检查模型路径
            model_path = model_config.get('path', '')
            if not os.path.exists(model_path):
                logger.error(f"模型路径不存在: {model_path}")
                sherpa_logger.error(f"模型路径不存在: {model_path}")
                return None

            # 获取当前引擎类型
            engine_type = self.get_current_engine_type()
            sherpa_logger.info(f"创建识别器，当前引擎类型: {engine_type}")

            # 检查当前引擎是否已初始化
            if not self.current_engine:
                sherpa_logger.error("当前引擎未初始化，尝试初始化...")
                if not self.initialize_engine(self.model_type):
                    sherpa_logger.error(f"初始化引擎 {self.model_type} 失败")
                    return None
                sherpa_logger.info("引擎初始化成功")

            # 根据引擎类型创建识别器
            if engine_type == "vosk" or engine_type == "vosk_small":
                # 对于Vosk模型，返回VoskRecognizer实例
                sherpa_logger.info(f"创建Vosk识别器，引擎类型: {engine_type}")
                model = vosk.Model(model_path)
                recognizer = vosk.KaldiRecognizer(model, 16000)
                recognizer.engine_type = engine_type  # 添加引擎类型标记，保持与模型类型一致
                sherpa_logger.info(f"Vosk识别器创建成功，引擎类型: {engine_type}")
                return recognizer
            elif engine_type and engine_type.startswith("sherpa"):
                # 对于Sherpa-ONNX模型，创建SherpaRecognizer适配器类
                sherpa_logger.info(f"创建SherpaRecognizer适配器，引擎类型: {engine_type}")

                # 创建包装器，使其接口与Vosk兼容
                class SherpaRecognizer:
                    def __init__(self, model):
                        """
                        初始化SherpaRecognizer

                        这是一个适配器类，使Sherpa-ONNX模型的接口与Vosk兼容。
                        它实现了与VoskRecognizer相同的方法（AcceptWaveform, Result, PartialResult, FinalResult, Reset），
                        但内部使用Sherpa-ONNX的API。

                        Args:
                            model: Sherpa-ONNX模型实例
                        """
                        self.model = model
                        self.partial_result = ""  # 存储部分识别结果，用于PartialResult方法返回
                        self.final_result = ""    # 存储最终识别结果，用于Result方法返回
                        self.previous_result = "" # 存储上一次的完整识别结果，用于文本增量检测
                        self.current_text = ""    # 存储当前累积的完整文本，用于连续识别
                        self.logger = sherpa_logger
                        self.engine_type = engine_type  # 存储引擎类型

                        # 添加累积文本功能
                        self.accumulated_text = []  # 存储所有完整的转录文本，用于保存完整历史

                        # 创建流
                        try:
                            self.current_stream = self.model.recognizer.create_stream()
                            self.logger.info("成功创建初始流")
                        except Exception as e:
                            self.logger.error(f"创建初始流失败: {e}")
                            import traceback
                            self.logger.error(traceback.format_exc())
                            self.current_stream = None

                    def AcceptWaveform(self, audio_data):
                        """
                        处理音频数据

                        Args:
                            audio_data: 音频数据，可以是字节或numpy数组

                        Returns:
                            bool: 是否有完整的识别结果
                        """
                        try:
                            # 检查模型和流是否已初始化
                            if not self.model or not self.model.recognizer:
                                self.logger.error("模型或识别器未初始化")
                                print("[ERROR] 模型或识别器未初始化")
                                return False

                            # 如果流未初始化，创建新的流
                            if not hasattr(self, 'current_stream') or self.current_stream is None:
                                try:
                                    self.current_stream = self.model.recognizer.create_stream()
                                    self.logger.info("创建新的流")
                                    print("[INFO] 创建新的流")
                                except Exception as e:
                                    self.logger.error(f"创建流失败: {e}")
                                    print(f"[ERROR] 创建流失败: {e}")
                                    import traceback
                                    error_trace = traceback.format_exc()
                                    self.logger.error(error_trace)
                                    print(error_trace)
                                    return False

                            # 处理音频数据
                            try:
                                # 确保音频数据是numpy数组
                                if isinstance(audio_data, bytes):
                                    # 将字节转换为16位整数数组
                                    import array
                                    audio_array = array.array('h', audio_data)
                                    audio_data = np.array(audio_array, dtype=np.float32) / 32768.0
                                    self.logger.debug(f"将字节转换为numpy数组，形状: {audio_data.shape}")
                                    print(f"[DEBUG] 将字节转换为numpy数组，形状: {audio_data.shape}")

                                # 确保音频数据是单声道
                                if len(audio_data.shape) > 1:
                                    audio_data = np.mean(audio_data, axis=1)
                                    self.logger.debug(f"将多声道转换为单声道，形状: {audio_data.shape}")
                                    print(f"[DEBUG] 将多声道转换为单声道，形状: {audio_data.shape}")

                                # 检查音频数据是否有效
                                max_amplitude = np.max(np.abs(audio_data))
                                if max_amplitude < 0.01:
                                    self.logger.debug(f"音频数据几乎是静音，最大振幅: {max_amplitude}，跳过")
                                    print(f"[DEBUG] 音频数据几乎是静音，最大振幅: {max_amplitude}，跳过")
                                    return False

                                # 接受音频数据
                                self.logger.debug(f"接受音频数据，形状: {audio_data.shape}, 最大值: {max_amplitude}")
                                print(f"[DEBUG] 接受音频数据，形状: {audio_data.shape}, 最大值: {max_amplitude}")
                                print(f"[DEBUG] 音频数据类型: {type(audio_data)}, 数据类型: {audio_data.dtype}")
                                print(f"[DEBUG] 音频数据样本: {audio_data[:10]}")  # 打印前10个样本
                                print(f"[DEBUG] 模型采样率: {self.model.sample_rate}")

                                try:
                                    self.current_stream.accept_waveform(self.model.sample_rate, audio_data)
                                    self.logger.debug("音频数据接受成功")
                                    print("[DEBUG] 音频数据接受成功")
                                except Exception as e:
                                    self.logger.error(f"接受音频数据错误: {e}")
                                    print(f"[ERROR] 接受音频数据错误: {e}")
                                    import traceback
                                    error_trace = traceback.format_exc()
                                    self.logger.error(error_trace)
                                    print(error_trace)
                                    return False

                                # 解码
                                try:
                                    decode_count = 0
                                    while self.model.recognizer.is_ready(self.current_stream):
                                        self.model.recognizer.decode_stream(self.current_stream)
                                        decode_count += 1
                                    self.logger.debug(f"解码完成，执行了 {decode_count} 次解码")
                                    print(f"[DEBUG] 解码完成，执行了 {decode_count} 次解码")
                                except Exception as e:
                                    self.logger.error(f"解码错误: {e}")
                                    print(f"[ERROR] 解码错误: {e}")
                                    import traceback
                                    error_trace = traceback.format_exc()
                                    self.logger.error(error_trace)
                                    print(error_trace)
                                    return False

                                # 获取当前结果
                                try:
                                    result = self.model.recognizer.get_result(self.current_stream)
                                    has_result = bool(result and result.strip())
                                    self.logger.debug(f"当前结果: {result}, 是否有结果: {has_result}")
                                    print(f"[DEBUG] 当前结果: {result}, 是否有结果: {has_result}")
                                except Exception as e:
                                    self.logger.error(f"获取结果错误: {e}")
                                    print(f"[ERROR] 获取结果错误: {e}")
                                    import traceback
                                    error_trace = traceback.format_exc()
                                    self.logger.error(error_trace)
                                    print(error_trace)
                                    return False

                                # 如果有结果，保存为最终结果
                                if has_result:
                                    # 使用正则表达式处理结果，确保每个句子以句号结尾
                                    import re
                                    result = re.sub(r'(?<=[a-zA-Z0-9])(?=[A-Z])', '. ', result)
                                    result = re.sub(r'\s+$', '', result)  # 去除末尾空格
                                    if not result.endswith('.'):
                                        result += '.'  # 确保结果以句号结尾

                                    # 首字母大写
                                    if result and len(result) > 0:
                                        result = result[0].upper() + result[1:]

                                    self.final_result = result

                                    # 检查是否是新的结果
                                    if self.final_result != self.previous_result:
                                        self.previous_result = self.final_result
                                        self.accumulated_text.append(self.final_result)
                                        self.logger.info(f"新的完整结果: {self.final_result}")
                                        print(f"[INFO] 新的完整结果: {self.final_result}")
                                        return True

                                return False

                            except Exception as e:
                                self.logger.error(f"处理音频数据错误: {e}")
                                print(f"[ERROR] 处理音频数据错误: {e}")
                                import traceback
                                error_trace = traceback.format_exc()
                                self.logger.error(error_trace)
                                print(error_trace)
                                return False

                        except Exception as e:
                            self.logger.error(f"AcceptWaveform 错误: {e}")
                            print(f"[ERROR] AcceptWaveform 错误: {e}")
                            import traceback
                            error_trace = traceback.format_exc()
                            self.logger.error(error_trace)
                            print(error_trace)
                            return False

                    def Result(self):
                        """
                        获取当前识别结果（兼容Vosk API）

                        Returns:
                            str: 当前识别结果的JSON字符串
                        """
                        try:
                            # 检查识别器和流是否已初始化
                            if not self.model or not self.model.recognizer or not hasattr(self, 'current_stream') or self.current_stream is None:
                                self.logger.warning("模型或流未初始化，返回空结果")
                                print("[WARNING] 模型或流未初始化，返回空结果")
                                return json.dumps({"text": ""})

                            # 解码
                            try:
                                decode_count = 0
                                while self.model.recognizer.is_ready(self.current_stream):
                                    self.model.recognizer.decode_stream(self.current_stream)
                                    decode_count += 1
                                self.logger.debug(f"解码完成，执行了 {decode_count} 次解码")
                                print(f"[DEBUG] 解码完成，执行了 {decode_count} 次解码")
                            except Exception as e:
                                self.logger.error(f"解码错误: {e}")
                                print(f"[ERROR] 解码错误: {e}")
                                import traceback
                                error_trace = traceback.format_exc()
                                self.logger.error(error_trace)
                                print(error_trace)
                                return json.dumps({"text": ""})

                            # 获取结果
                            try:
                                result = self.model.recognizer.get_result(self.current_stream)
                                self.logger.debug(f"原始结果: {result}")
                                print(f"[DEBUG] 原始结果: {result}")
                            except Exception as e:
                                self.logger.error(f"获取结果错误: {e}")
                                print(f"[ERROR] 获取结果错误: {e}")
                                import traceback
                                error_trace = traceback.format_exc()
                                self.logger.error(error_trace)
                                print(error_trace)
                                return json.dumps({"text": ""})

                            # 重置流
                            try:
                                old_stream = self.current_stream
                                self.current_stream = self.model.recognizer.create_stream()
                                self.logger.debug("流重置成功")
                                print("[DEBUG] 流重置成功")
                            except Exception as e:
                                self.logger.error(f"重置流失败: {e}")
                                print(f"[ERROR] 重置流失败: {e}")
                                import traceback
                                error_trace = traceback.format_exc()
                                self.logger.error(error_trace)
                                print(error_trace)
                                # 继续使用旧的流
                                self.current_stream = old_stream

                            # 处理结果
                            if result:
                                # 使用正则表达式处理结果，确保每个句子以句号结尾
                                import re
                                processed_result = re.sub(r'(?<=[a-zA-Z0-9])(?=[A-Z])', '. ', result)
                                processed_result = re.sub(r'\s+$', '', processed_result)  # 去除末尾空格
                                if not processed_result.endswith('.'):
                                    processed_result += '.'  # 确保结果以句号结尾

                                # 首字母大写
                                if processed_result and len(processed_result) > 0:
                                    processed_result = processed_result[0].upper() + processed_result[1:]

                                self.logger.debug(f"处理后的结果: {processed_result}")
                                print(f"[DEBUG] 处理后的结果: {processed_result}")

                                # 保存为最终结果
                                self.final_result = processed_result

                                # 检查是否是新的结果
                                if self.final_result != self.previous_result:
                                    self.previous_result = self.final_result
                                    self.accumulated_text.append(self.final_result)
                                    self.logger.info(f"新的完整结果: {self.final_result}")
                                    print(f"[INFO] 新的完整结果: {self.final_result}")

                                # 返回JSON格式的结果
                                json_result = json.dumps({"text": processed_result})
                                self.logger.debug(f"返回JSON结果: {json_result}")
                                print(f"[DEBUG] 返回JSON结果: {json_result}")
                                return json_result

                            self.logger.warning("结果为空，返回空JSON")
                            print("[WARNING] 结果为空，返回空JSON")
                            return json.dumps({"text": ""})

                        except Exception as e:
                            self.logger.error(f"Result 错误: {e}")
                            print(f"[ERROR] Result 错误: {e}")
                            import traceback
                            error_trace = traceback.format_exc()
                            self.logger.error(error_trace)
                            print(error_trace)
                            return json.dumps({"text": ""})

                    def PartialResult(self):
                        """
                        获取部分识别结果（兼容Vosk API）

                        Returns:
                            str: 部分识别结果的JSON字符串
                        """
                        try:
                            # 检查识别器和流是否已初始化
                            if not self.model or not self.model.recognizer or not hasattr(self, 'current_stream') or self.current_stream is None:
                                self.logger.warning("模型或流未初始化，返回空部分结果")
                                print("[WARNING] 模型或流未初始化，返回空部分结果")
                                return json.dumps({"partial": ""})

                            # 解码
                            try:
                                decode_count = 0
                                while self.model.recognizer.is_ready(self.current_stream):
                                    self.model.recognizer.decode_stream(self.current_stream)
                                    decode_count += 1
                                self.logger.debug(f"部分结果解码完成，执行了 {decode_count} 次解码")
                                print(f"[DEBUG] 部分结果解码完成，执行了 {decode_count} 次解码")
                            except Exception as e:
                                self.logger.error(f"部分结果解码错误: {e}")
                                print(f"[ERROR] 部分结果解码错误: {e}")
                                import traceback
                                error_trace = traceback.format_exc()
                                self.logger.error(error_trace)
                                print(error_trace)
                                return json.dumps({"partial": ""})

                            # 获取部分结果
                            try:
                                result = self.model.recognizer.get_result(self.current_stream)
                                self.logger.debug(f"原始部分结果: {result}")
                                print(f"[DEBUG] 原始部分结果: {result}")
                            except Exception as e:
                                self.logger.error(f"获取部分结果错误: {e}")
                                print(f"[ERROR] 获取部分结果错误: {e}")
                                import traceback
                                error_trace = traceback.format_exc()
                                self.logger.error(error_trace)
                                print(error_trace)
                                return json.dumps({"partial": ""})

                            # 处理结果
                            if result:
                                # 使用正则表达式处理结果，确保每个句子以句号结尾
                                import re
                                processed_result = re.sub(r'(?<=[a-zA-Z0-9])(?=[A-Z])', '. ', result)
                                processed_result = re.sub(r'\s+$', '', processed_result)  # 去除末尾空格

                                # 首字母大写
                                if processed_result and len(processed_result) > 0:
                                    processed_result = processed_result[0].upper() + processed_result[1:]

                                self.logger.debug(f"处理后的部分结果: {processed_result}")
                                print(f"[DEBUG] 处理后的部分结果: {processed_result}")

                                # 保存为部分结果
                                self.partial_result = processed_result

                                # 返回JSON格式的结果
                                json_result = json.dumps({"partial": processed_result})
                                self.logger.debug(f"返回JSON部分结果: {json_result}")
                                print(f"[DEBUG] 返回JSON部分结果: {json_result}")
                                return json_result

                            self.logger.warning("部分结果为空，返回空JSON")
                            print("[WARNING] 部分结果为空，返回空JSON")
                            return json.dumps({"partial": ""})

                        except Exception as e:
                            self.logger.error(f"PartialResult 错误: {e}")
                            print(f"[ERROR] PartialResult 错误: {e}")
                            import traceback
                            error_trace = traceback.format_exc()
                            self.logger.error(error_trace)
                            print(error_trace)
                            return json.dumps({"partial": ""})

                    def FinalResult(self):
                        """
                        获取最终识别结果（兼容Vosk API）

                        Returns:
                            str: 最终识别结果的JSON字符串
                        """
                        try:
                            # 检查识别器和流是否已初始化
                            if not self.model or not self.model.recognizer:
                                return json.dumps({"text": ""})

                            # 创建一个新的流
                            try:
                                stream = self.model.recognizer.create_stream()
                            except Exception as e:
                                self.logger.error(f"创建流失败: {e}")
                                import traceback
                                self.logger.error(traceback.format_exc())
                                return json.dumps({"text": ""})

                            # 标记输入结束
                            try:
                                stream.input_finished()

                                # 解码
                                while self.model.recognizer.is_ready(stream):
                                    self.model.recognizer.decode_stream(stream)
                            except Exception as e:
                                self.logger.error(f"解码剩余音频错误: {e}")
                                import traceback
                                self.logger.error(traceback.format_exc())
                                return json.dumps({"text": ""})

                            # 获取结果
                            try:
                                result = self.model.recognizer.get_result(stream)

                                # 处理结果
                                if result:
                                    # 使用正则表达式处理结果，确保每个句子以句号结尾
                                    import re
                                    result = re.sub(r'(?<=[a-zA-Z0-9])(?=[A-Z])', '. ', result)
                                    result = re.sub(r'\s+$', '', result)  # 去除末尾空格
                                    if not result.endswith('.'):
                                        result += '.'  # 确保结果以句号结尾

                                    # 首字母大写
                                    if result and len(result) > 0:
                                        result = result[0].upper() + result[1:]

                                    # 返回JSON格式的结果
                                    return json.dumps({"text": result})

                                return json.dumps({"text": ""})

                            except Exception as e:
                                self.logger.error(f"获取结果错误: {e}")
                                import traceback
                                self.logger.error(traceback.format_exc())
                                return json.dumps({"text": ""})

                        except Exception as e:
                            self.logger.error(f"FinalResult 错误: {e}")
                            import traceback
                            self.logger.error(traceback.format_exc())
                            return json.dumps({"text": ""})

                    def Reset(self):
                        """
                        重置识别器状态（兼容Vosk API）
                        """
                        try:
                            # 重置流
                            if hasattr(self, 'current_stream') and self.current_stream is not None:
                                try:
                                    self.current_stream = self.model.recognizer.create_stream()
                                    self.logger.info("重置流成功")
                                except Exception as e:
                                    self.logger.error(f"重置流失败: {e}")
                                    import traceback
                                    self.logger.error(traceback.format_exc())
                                    self.current_stream = None

                            # 重置结果
                            self.partial_result = ""
                            self.final_result = ""
                            self.previous_result = ""

                        except Exception as e:
                            self.logger.error(f"Reset 错误: {e}")
                            import traceback
                            self.logger.error(traceback.format_exc())

                # 返回包装后的识别器
                model_type_str = "int8量化" if hasattr(self.current_engine, 'is_int8') and self.current_engine.is_int8 else "标准"
                sherpa_logger.info(f"创建 Sherpa-ONNX {model_type_str}识别器")

                try:
                    recognizer = SherpaRecognizer(self.current_engine)
                    sherpa_logger.info(f"SherpaRecognizer 实例创建成功: {recognizer}")

                    # 设置引擎类型
                    recognizer.engine_type = engine_type
                    sherpa_logger.info(f"设置 Sherpa-ONNX 识别器引擎类型: {engine_type}")

                    sherpa_logger.info(f"Sherpa-ONNX 识别器创建成功，引擎类型: {recognizer.engine_type}")
                    return recognizer
                except Exception as e:
                    error_msg = f"创建 SherpaRecognizer 实例失败: {e}"
                    sherpa_logger.error(error_msg)
                    print(error_msg)
                    import traceback
                    error_trace = traceback.format_exc()
                    sherpa_logger.error(error_trace)
                    print(error_trace)
                    return None
            else:
                sherpa_logger.error(f"不支持的引擎类型: {engine_type}")
                return None

        except Exception as e:
            logger.error(f"创建识别器时发生错误: {str(e)}")
            logger.error(f"当前工作目录: {os.getcwd()}")
            logger.error(f"模型类型: {self.model_type}")
            logger.error(f"模型配置: {self.models_config.get(self.model_type, {})}")
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"错误堆栈:\n{error_trace}")

            try:
                from src.utils.sherpa_logger import sherpa_logger
                sherpa_logger.error(f"创建识别器时发生错误: {str(e)}")
                sherpa_logger.error(error_trace)
            except:
                pass

            return None

    def check_model_directory(self) -> Dict[str, bool]:
        """
        检查模型目录

        Returns:
            Dict[str, bool]: 模型可用性字典
        """
        result = {}
        models = self.models_config.get('models', {})

        for name, config in models.items():
            path = config.get('path', '')
            result[name] = os.path.exists(path)

        return result

    # 以下是从 manager.py 合并的方法

    def _get_nested_config(self, path: str, default: Any = None) -> Any:
        """获取嵌套配置

        Args:
            path: 配置路径，使用点表示法，如 "asr.models.vosk"
            default: 如果路径不存在，返回的默认值

        Returns:
            Any: 配置值或默认值
        """
        parts = path.split('.')
        config = self.config.config  # 获取整个配置字典

        for part in parts:
            if isinstance(config, dict) and part in config:
                config = config[part]
            else:
                return default

        return config

    def initialize_engine(self, engine_type: str = "vosk") -> bool:
        """初始化指定的 ASR 引擎

        Args:
            engine_type: 引擎类型，可选 "vosk"、"sherpa_int8" 或 "sherpa_std"

        Returns:
            bool: 是否初始化成功
        """
        # 导入 Sherpa-ONNX 日志工具
        try:
            from src.utils.sherpa_logger import sherpa_logger
        except ImportError:
            # 如果导入失败，创建一个简单的日志记录器
            class DummyLogger:
                def debug(self, msg): print(f"DEBUG: {msg}")
                def info(self, msg): print(f"INFO: {msg}")
                def warning(self, msg): print(f"WARNING: {msg}")
                def error(self, msg): print(f"ERROR: {msg}")
            sherpa_logger = DummyLogger()

        try:
            # 直接从 models_config 获取模型配置
            sherpa_logger.info(f"初始化引擎: {engine_type}")
            sherpa_logger.debug(f"models_config = {self.models_config}")
            sherpa_logger.info(f"当前模型类型: {self.model_type}")

            # 检查模型类型和引擎类型是否一致
            if self.model_type and self.model_type != engine_type:
                sherpa_logger.warning(f"模型类型 {self.model_type} 与引擎类型 {engine_type} 不一致")

                # 更新模型类型为引擎类型
                old_model_type = self.model_type
                self.model_type = engine_type
                sherpa_logger.info(f"模型类型已从 {old_model_type} 更新为: {self.model_type}")

            if engine_type not in self.models_config:
                sherpa_logger.error(f"引擎 {engine_type} 在配置中不存在")
                return False

            model_config = self.models_config[engine_type]
            sherpa_logger.debug(f"模型配置: {model_config}")

            if not model_config or not model_config.get("enabled", False):
                sherpa_logger.error(f"引擎 {engine_type} 未启用或未配置")
                return False

            # 记录当前引擎状态
            sherpa_logger.info(f"当前引擎: {type(self.current_engine).__name__ if self.current_engine else None}")

            # 初始化引擎
            if engine_type == "vosk" or engine_type == "vosk_small":
                sherpa_logger.info(f"创建 VoskASR 实例，路径: {model_config['path']}")
                # 检查模型路径是否存在
                if not os.path.exists(model_config["path"]):
                    sherpa_logger.error(f"Vosk 模型路径不存在: {model_config['path']}")
                    return False

                # 使用更新后的验证方法，明确传入模型类型
                if not self._validate_model_path(model_config["path"], engine_type):
                    sherpa_logger.error(f"Vosk 模型路径验证失败: {model_config['path']}")
                    return False

                try:
                    # 创建 VoskASR 实例
                    self.current_engine = VoskASR(model_config["path"])
                    sherpa_logger.info(f"VoskASR 实例创建成功: {self.current_engine}")

                    # 检查引擎是否成功初始化
                    if not self.current_engine.model or not self.current_engine.recognizer:
                        error_msg = "VoskASR 引擎初始化失败，模型或识别器为空"
                        sherpa_logger.error(error_msg)
                        print(error_msg)
                        return False

                    # 为VoskASR实例添加engine_type属性，确保与模型类型一致
                    self.current_engine.engine_type = engine_type
                    sherpa_logger.info(f"设置VoskASR引擎类型为: {engine_type}")

                    sherpa_logger.info("VoskASR 引擎初始化成功")
                except Exception as e:
                    error_msg = f"创建 VoskASR 实例失败: {e}"
                    sherpa_logger.error(error_msg)
                    print(error_msg)
                    import traceback
                    error_trace = traceback.format_exc()
                    sherpa_logger.error(error_trace)
                    print(error_trace)
                    return False

            elif engine_type.startswith("sherpa"):
                sherpa_logger.info(f"创建 SherpaOnnxASR 实例，路径: {model_config.get('path', '')}")

                # 检查模型路径是否存在
                if not os.path.exists(model_config["path"]):
                    sherpa_logger.error(f"Sherpa-ONNX 模型路径不存在: {model_config.get('path', '未知路径')}")
                    return False

                # 使用更新后的验证方法，明确传入模型类型
                if not self._validate_model_path(model_config["path"], engine_type):
                    sherpa_logger.error(f"Sherpa-ONNX 模型路径验证失败: {model_config['path']}")
                    return False

                # 检查模型类型
                if engine_type == "sherpa_0626_int8":
                    model_type = "int8"
                    model_name = "0626"
                elif engine_type == "sherpa_0626_std" or engine_type == "sherpa_0626":
                    model_type = "standard"
                    model_name = "0626"
                else:
                    model_type = "int8" if engine_type == "sherpa_int8" else "standard"
                    model_name = ""
                sherpa_logger.info(f"Sherpa-ONNX 模型类型: {model_type}, 模型名称: {model_name}")

                # 记录模型路径和配置
                sherpa_logger.info(f"模型路径: {model_config['path']}")
                sherpa_logger.info(f"模型配置: {model_config.get('config', {})}")

                # 创建 SherpaOnnxASR 实例
                try:
                    sherpa_logger.info(f"创建 SherpaOnnxASR 实例，路径: {model_config['path']}, 类型: {model_type}, 名称: {model_name}")
                    self.current_engine = SherpaOnnxASR(model_config["path"], {"type": model_type, "name": model_name})
                    sherpa_logger.info(f"SherpaOnnxASR 实例创建成功: {self.current_engine}")

                    # 调用 setup 方法初始化引擎
                    sherpa_logger.info("开始初始化 Sherpa-ONNX 引擎...")
                    if not self.current_engine.setup():
                        error_msg = "初始化 Sherpa-ONNX 引擎失败"
                        sherpa_logger.error(error_msg)
                        print(error_msg)
                        return False

                    sherpa_logger.info("Sherpa-ONNX 引擎初始化成功")

                except Exception as e:
                    error_msg = f"创建 SherpaOnnxASR 实例失败: {e}"
                    sherpa_logger.error(error_msg)
                    print(error_msg)
                    return False
            else:
                sherpa_logger.error(f"不支持的引擎类型: {engine_type}")
                return False

            # 记录最终引擎状态
            sherpa_logger.info(f"初始化后的引擎: {type(self.current_engine).__name__ if self.current_engine else None}")

            # 获取并检查引擎类型
            current_engine_type = self.get_current_engine_type()
            sherpa_logger.info(f"初始化后的引擎类型: {current_engine_type}")

            # 确保模型类型和引擎类型一致
            if current_engine_type != engine_type:
                sherpa_logger.warning(f"初始化后的引擎类型 {current_engine_type} 与请求的引擎类型 {engine_type} 不一致")
                # 更新模型类型为实际的引擎类型
                old_model_type = self.model_type
                self.model_type = current_engine_type
                sherpa_logger.info(f"模型类型已从 {old_model_type} 更新为: {self.model_type}")

            # 发射模型加载完成信号
            sherpa_logger.info("发射模型加载完成信号")
            self.model_loaded.emit(True)

            return True

        except Exception as e:
            sherpa_logger.error(f"初始化 {engine_type} 引擎错误: {str(e)}")
            import traceback
            sherpa_logger.error(traceback.format_exc())

            # 发射模型加载失败信号
            sherpa_logger.info("发射模型加载失败信号")
            self.model_loaded.emit(False)

            return False

    def transcribe(self, audio_data: Union[bytes, np.ndarray]) -> Optional[str]:
        """转录音频数据

        Args:
            audio_data: 音频数据，可以是字节或 numpy 数组

        Returns:
            str: 转录文本，如果失败则返回 None
        """
        if not self.current_engine:
            print("No ASR engine initialized")
            return None

        try:
            return self.current_engine.transcribe(audio_data)
        except Exception as e:
            print(f"Error in transcription: {str(e)}")
            return None

    def transcribe_file(self, file_path: str) -> Optional[str]:
        """转录音频文件

        Args:
            file_path: 音频文件路径

        Returns:
            str: 转录文本，如果失败则返回 None
        """
        # 导入 Sherpa-ONNX 日志工具
        try:
            from src.utils.sherpa_logger import sherpa_logger
        except ImportError:
            # 如果导入失败，创建一个简单的日志记录器
            class DummyLogger:
                def debug(self, msg): print(f"DEBUG: {msg}")
                def info(self, msg): print(f"INFO: {msg}")
                def warning(self, msg): print(f"WARNING: {msg}")
                def error(self, msg): print(f"ERROR: {msg}")
            sherpa_logger = DummyLogger()

        sherpa_logger.info(f"开始转录文件: {file_path}")
        sherpa_logger.info(f"当前模型类型: {self.model_type}")
        sherpa_logger.info(f"当前引擎: {type(self.current_engine).__name__ if self.current_engine else None}")

        # 确保引擎已初始化
        if not self.current_engine:
            sherpa_logger.info("ASR 引擎未初始化，尝试初始化...")
            # 尝试初始化当前选择的引擎
            if self.model_type:
                # 使用完整的模型类型名称
                sherpa_logger.info(f"尝试初始化引擎: {self.model_type}")
                success = self.initialize_engine(self.model_type)
                sherpa_logger.info(f"引擎初始化结果: {success}")
            else:
                # 默认使用 vosk
                sherpa_logger.info("尝试初始化默认引擎: vosk")
                success = self.initialize_engine('vosk')
                sherpa_logger.info(f"默认引擎初始化结果: {success}")

        # 再次检查引擎是否已初始化
        if not self.current_engine:
            sherpa_logger.error("无法初始化 ASR 引擎")
            return None

        # 检查当前引擎类型
        engine_type = self.get_current_engine_type()
        sherpa_logger.info(f"当前引擎类型: {engine_type}")

        # 检查模型类型和引擎类型是否一致
        if self.model_type and engine_type and self.model_type != engine_type:
            sherpa_logger.warning(f"模型类型 {self.model_type} 与引擎类型 {engine_type} 不一致")

            # 尝试重新初始化引擎
            sherpa_logger.info(f"尝试重新初始化引擎: {self.model_type}")
            if not self.initialize_engine(self.model_type):
                sherpa_logger.error(f"重新初始化引擎失败")

                # 如果重新初始化失败，尝试使用当前引擎
                sherpa_logger.warning(f"使用当前引擎 {engine_type} 继续转录")
            else:
                # 更新引擎类型
                engine_type = self.get_current_engine_type()
                sherpa_logger.info(f"引擎重新初始化成功，当前引擎类型: {engine_type}")
                sherpa_logger.info(f"当前引擎: {type(self.current_engine).__name__ if self.current_engine else None}")

        # 检查当前引擎是否支持文件转录
        if not hasattr(self.current_engine, 'transcribe_file'):
            sherpa_logger.error(f"当前引擎 {engine_type} 不支持文件转录")

            # 尝试切换到支持文件转录的引擎
            sherpa_logger.info("尝试切换到支持文件转录的引擎")

            # 优先尝试 sherpa_0626 引擎
            if self.initialize_engine('sherpa_0626'):
                sherpa_logger.info("成功切换到 sherpa_0626 引擎")
            # 然后尝试 sherpa_std 引擎
            elif self.initialize_engine('sherpa_std'):
                sherpa_logger.info("成功切换到 sherpa_std 引擎")
            # 最后尝试 vosk 引擎
            elif self.initialize_engine('vosk'):
                sherpa_logger.info("成功切换到 vosk 引擎")
            else:
                sherpa_logger.error("无法切换到支持文件转录的引擎")
                return None

            # 再次检查引擎是否支持文件转录
            if not hasattr(self.current_engine, 'transcribe_file'):
                sherpa_logger.error("切换后的引擎仍不支持文件转录")
                return None

            # 更新引擎类型
            engine_type = self.get_current_engine_type()
            sherpa_logger.info(f"切换后的引擎类型: {engine_type}")

        try:
            # 记录详细的引擎信息
            sherpa_logger.info(f"调用引擎的 transcribe_file 方法")
            sherpa_logger.info(f"使用引擎: {type(self.current_engine).__name__}")
            sherpa_logger.info(f"引擎类型: {engine_type}")

            # 检查文件是否存在
            import os
            if not os.path.exists(file_path):
                sherpa_logger.error(f"文件不存在: {file_path}")
                return None

            # 检查文件大小
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            sherpa_logger.info(f"文件大小: {file_size:.2f} MB")

            # 调用引擎的 transcribe_file 方法
            result = self.current_engine.transcribe_file(file_path)

            # 记录转录结果
            if result:
                result_preview = result[:100] + "..." if len(result) > 100 else result
                sherpa_logger.info(f"转录结果: {result_preview}")
            else:
                sherpa_logger.warning("转录结果为空")

            return result
        except Exception as e:
            sherpa_logger.error(f"文件转录错误: {str(e)}")
            import traceback
            sherpa_logger.error(traceback.format_exc())
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

    def get_current_engine_type(self) -> Optional[str]:
        """获取当前引擎类型

        Returns:
            str: 当前引擎类型，如果未初始化则返回 None
        """
        # 导入 Sherpa-ONNX 日志工具
        try:
            from src.utils.sherpa_logger import sherpa_logger
        except ImportError:
            # 如果导入失败，创建一个简单的日志记录器
            class DummyLogger:
                def debug(self, msg): print(f"DEBUG: {msg}")
                def info(self, msg): print(f"INFO: {msg}")
                def warning(self, msg): print(f"WARNING: {msg}")
                def error(self, msg): print(f"ERROR: {msg}")
            sherpa_logger = DummyLogger()

        sherpa_logger.debug(f"获取当前引擎类型")
        sherpa_logger.debug(f"self.current_engine = {self.current_engine}")
        sherpa_logger.debug(f"self.model_type = {self.model_type}")

        # 如果已经有明确设置的model_type，优先使用它
        if self.model_type and self.current_engine:
            sherpa_logger.debug(f"使用已设置的模型类型: {self.model_type}")

            # 验证当前引擎是否与model_type匹配
            is_match = False

            # 检查VoskASR
            if self.model_type == "vosk" and isinstance(self.current_engine, VoskASR):
                is_match = True

            # 检查SherpaOnnxASR
            elif isinstance(self.current_engine, SherpaOnnxASR):
                # 检查是否是0626模型
                if "0626" in self.model_type:
                    model_dir = getattr(self.current_engine, 'model_dir', '')
                    if "0626" in model_dir or "2023-06-26" in model_dir:
                        # 检查是否是int8模型
                        if "int8" in self.model_type and hasattr(self.current_engine, 'is_int8') and self.current_engine.is_int8:
                            is_match = True
                        elif "int8" not in self.model_type and (not hasattr(self.current_engine, 'is_int8') or not self.current_engine.is_int8):
                            is_match = True
                # 检查是否是普通Sherpa模型
                else:
                    # 检查是否是int8模型
                    if "int8" in self.model_type and hasattr(self.current_engine, 'is_int8') and self.current_engine.is_int8:
                        is_match = True
                    elif "std" in self.model_type and (not hasattr(self.current_engine, 'is_int8') or not self.current_engine.is_int8):
                        is_match = True

            if is_match:
                sherpa_logger.debug(f"当前引擎与模型类型 {self.model_type} 匹配")
                return self.model_type
            else:
                sherpa_logger.warning(f"当前引擎与模型类型 {self.model_type} 不匹配，需要重新推断")

        # 如果没有model_type或者不匹配，根据current_engine推断
        engine_type = None

        # 首先根据 current_engine 的类型推断
        if isinstance(self.current_engine, VoskASR):
            sherpa_logger.debug("当前引擎是 VoskASR")
            # 使用vosk_small作为引擎类型，与配置文件中的模型类型保持一致
            engine_type = "vosk_small"
        # 检查是否有engine_type属性（适用于SherpaRecognizer适配器类）
        elif hasattr(self.current_engine, 'engine_type') and self.current_engine.engine_type:
            sherpa_logger.debug(f"当前引擎有engine_type属性: {self.current_engine.engine_type}")
            engine_type = self.current_engine.engine_type
        elif isinstance(self.current_engine, SherpaOnnxASR):
            sherpa_logger.debug("当前引擎是 SherpaOnnxASR")

            # 记录引擎的详细信息，帮助调试
            sherpa_logger.debug(f"引擎属性:")
            if hasattr(self.current_engine, 'model_dir'):
                sherpa_logger.debug(f"  model_dir: {self.current_engine.model_dir}")
            if hasattr(self.current_engine, 'model_config'):
                sherpa_logger.debug(f"  model_config: {self.current_engine.model_config}")
            if hasattr(self.current_engine, 'is_int8'):
                sherpa_logger.debug(f"  is_int8: {self.current_engine.is_int8}")

            # 尝试从引擎实例获取更具体的类型
            if hasattr(self.current_engine, 'model_config') and self.current_engine.model_config:
                model_config = self.current_engine.model_config
                sherpa_logger.debug(f"使用model_config确定引擎类型: {model_config}")

                # 检查是否是0626模型
                if model_config.get("name") == "0626":
                    sherpa_logger.debug("检测到0626模型")
                    # 检查模型目录名称是否包含0626或2023-06-26
                    model_dir = getattr(self.current_engine, 'model_dir', '')
                    if "0626" in model_dir or "2023-06-26" in model_dir:
                        sherpa_logger.debug(f"模型目录确认为0626: {model_dir}")

                    # 根据类型确定具体的引擎类型
                    if model_config.get("type") == "int8":
                        engine_type = "sherpa_0626_int8"
                    else:
                        engine_type = "sherpa_0626"  # 使用简化名称，与config.json一致

                    sherpa_logger.debug(f"当前引擎是 SherpaOnnxASR ({engine_type})")
                else:
                    # 非0626模型
                    if model_config.get("type") == "int8":
                        engine_type = "sherpa_onnx_int8"
                    else:
                        engine_type = "sherpa_onnx_std"
                    sherpa_logger.debug(f"当前引擎是 SherpaOnnxASR ({engine_type})")
            else:
                # 如果没有model_config，使用默认逻辑
                sherpa_logger.debug("没有model_config，使用默认逻辑")

                # 检查模型目录名称是否包含0626或2023-06-26
                model_dir = getattr(self.current_engine, 'model_dir', '')
                if "0626" in model_dir or "2023-06-26" in model_dir:
                    sherpa_logger.debug(f"从模型目录检测到0626模型: {model_dir}")
                    # 根据is_int8确定具体的引擎类型
                    if hasattr(self.current_engine, 'is_int8') and self.current_engine.is_int8:
                        engine_type = "sherpa_0626_int8"
                    else:
                        engine_type = "sherpa_0626"
                else:
                    # 非0626模型
                    if hasattr(self.current_engine, 'is_int8') and self.current_engine.is_int8:
                        engine_type = "sherpa_onnx_int8"
                    else:
                        engine_type = "sherpa_onnx_std"

                sherpa_logger.debug(f"当前引擎是 SherpaOnnxASR ({engine_type})")
        else:
            sherpa_logger.debug("未识别的引擎类型")
            engine_type = None

        # 如果推断出了引擎类型，但与当前model_type不一致，记录警告但不自动更新
        if self.model_type and engine_type and self.model_type != engine_type:
            sherpa_logger.warning(f"模型类型 {self.model_type} 与推断的引擎类型 {engine_type} 不一致")
            sherpa_logger.warning(f"保持当前模型类型: {self.model_type}，但可能导致功能异常")
            # 返回推断的引擎类型，而不是model_type
            return engine_type
        elif not self.model_type and engine_type:
            # 如果model_type为空但能推断出引擎类型，则更新model_type
            self.model_type = engine_type
            sherpa_logger.info(f"模型类型已设置为: {self.model_type}")
            return engine_type

        # 返回最终的引擎类型
        if engine_type:
            sherpa_logger.debug(f"返回引擎类型: {engine_type}")
            return engine_type
        elif self.model_type:
            sherpa_logger.debug(f"返回模型类型: {self.model_type}")
            return self.model_type

        sherpa_logger.debug("无法确定引擎类型")
        return None

    def get_available_engines(self) -> Dict[str, bool]:
        """获取可用的引擎列表

        Returns:
            Dict[str, bool]: 引擎名称到是否启用的映射
        """
        engines = {}
        # 检查 vosk_small 引擎
        if "vosk_small" in self.models_config:
            model_config = self.models_config["vosk_small"]
            engines["vosk_small"] = bool(model_config and model_config.get("enabled", False))
        else:
            engines["vosk_small"] = False

        # 检查 sherpa_onnx_int8 引擎
        if "sherpa_onnx_int8" in self.models_config:
            model_config = self.models_config["sherpa_onnx_int8"]
            engines["sherpa_onnx_int8"] = bool(model_config and model_config.get("enabled", False))
        else:
            engines["sherpa_onnx_int8"] = False

        # 检查 sherpa_onnx_std 引擎
        if "sherpa_onnx_std" in self.models_config:
            model_config = self.models_config["sherpa_onnx_std"]
            engines["sherpa_onnx_std"] = bool(model_config and model_config.get("enabled", False))
        else:
            engines["sherpa_onnx_std"] = False

        # 检查 sherpa_0626_int8 引擎
        if "sherpa_0626_int8" in self.models_config:
            model_config = self.models_config["sherpa_0626_int8"]
            engines["sherpa_0626_int8"] = bool(model_config and model_config.get("enabled", False))
        else:
            engines["sherpa_0626_int8"] = False

        # 检查 sherpa_0626_std 引擎
        if "sherpa_0626_std" in self.models_config:
            model_config = self.models_config["sherpa_0626_std"]
            engines["sherpa_0626_std"] = bool(model_config and model_config.get("enabled", False))
        else:
            engines["sherpa_0626_std"] = False

        return engines

    def get_audio_devices(self) -> List[Dict[str, Any]]:
        """获取可用的音频设备列表

        Returns:
            List[Dict[str, Any]]: 音频设备列表
        """
        try:
            # 使用 soundcard 库获取音频设备
            try:
                import soundcard as sc
            except ImportError:
                logger.error("未安装 soundcard 模块，无法获取音频设备列表")
                return []

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
                self.error_occurred.emit("音频设备为空")
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
            self.error_occurred.emit(error_msg)
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
                self.error_occurred.emit("ASR引擎未初始化，请先加载模型")
                return False

            # 检查设备是否已设置
            if not self.current_device:
                logger.error("未设置音频设备")
                self.error_occurred.emit("未设置音频设备，请先选择音频设备")
                return False

            # 检查是否已经在识别
            if self.is_recognizing:
                logger.warning("识别已经在进行中")
                return True

            # 开始识别
            logger.info("开始识别")
            self.is_recognizing = True

            # 发射识别开始信号
            self.recognition_started.emit()

            # 更新状态
            self.signals.status_updated.emit("识别已开始")

            return True
        except Exception as e:
            error_msg = f"启动识别失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.error_occurred.emit(error_msg)
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
            self.recognition_stopped.emit()

            # 更新状态
            self.signals.status_updated.emit("识别已停止")

            return True
        except Exception as e:
            error_msg = f"停止识别失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.error_occurred.emit(error_msg)
            return False
