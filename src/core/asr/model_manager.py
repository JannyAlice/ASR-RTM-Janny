"""
ASR模型管理模块
负责加载和管理ASR模型
"""
import os
import json
import time
import vosk
import logging
import traceback
import numpy as np
from typing import Optional, Dict, Any, Union
from pathlib import Path

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

class ASRModelManager:
    """ASR模型管理器类"""

    def __init__(self):
        """初始化ASR模型管理器"""
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

    def validate_model_files(self, model_path: str) -> bool:
        """验证模型文件完整性"""
        try:
            # 针对sherpa_0626模型的具体文件名
            required_files = [
                "encoder-epoch-99-avg-1-chunk-16-left-128.onnx",  # 标准模型文件
                "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
                "joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
                "tokens.txt"
            ]
            
            # 检查每个文件是否存在
            for file in required_files:
                file_path = os.path.join(model_path, file)
                if not os.path.exists(file_path):
                    logger.error(f"模型文件不存在: {file_path}")
                    return False
                    
            logger.info(f"模型文件验证通过: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"验证模型文件时发生错误: {str(e)}")
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
                
            if not self.validate_model_files(model_path):
                logger.error(f"错误: 模型路径验证失败: {model_path}")
                return False

            # 更新当前模型信息
            self.current_model_type = model_name
            logger.info(f"模型加载成功: {model_name}")
            return True

        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            logger.error(traceback.format_exc())
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

    def _validate_model_path(self, model_path: str) -> bool:
        """
        验证模型路径是否有效
        """
        try:
            if not model_path:
                logger.error("模型路径为空")
                return False
                
            if not os.path.exists(model_path):
                logger.error(f"模型路径不存在: {model_path}")
                return False
                
            # 检查必要的模型文件
            if self.model_type.startswith('sherpa'):
                required_files = ['encoder', 'decoder', 'joiner', 'tokens.txt']
                for file in required_files:
                    file_path = os.path.join(model_path, file)
                    if not os.path.exists(file_path):
                        logger.error(f"模型文件不存在: {file_path}")
                        return False
                        
            return True
            
        except Exception as e:
            logger.error(f"验证模型路径时发生错误: {str(e)}")
            return False

    def create_recognizer(self) -> Optional[Any]:
        """
        创建识别器

        Returns:
            Optional[Any]: 识别器实例
        """
        try:
            # 检查当前模型是否已加载
            if self.current_model is None:
                logger.error("当前模型未加载，请先加载模型")
                return None

            # 检查模型配置
            model_config = self.models_config.get(self.model_type, {})
            if not model_config:
                logger.error(f"模型配置 {self.model_type} 不存在")
                return None

            # 检查模型路径
            model_path = model_config.get('path', '')
            if not os.path.exists(model_path):
                logger.error(f"模型路径不存在: {model_path}")
                return None

        except Exception as e:
            logger.error(f"创建识别器时发生错误: {str(e)}")
            logger.error(f"当前工作目录: {os.getcwd()}")
            logger.error(f"模型类型: {self.model_type}")
            logger.error(f"模型配置: {self.models_config.get(self.model_type, {})}")
            import traceback
            logger.error(f"错误堆栈:\n{traceback.format_exc()}")
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
            if engine_type == "vosk":
                sherpa_logger.info(f"创建 VoskASR 实例，路径: {model_config['path']}")
                # 检查模型路径是否存在
                if not os.path.exists(model_config["path"]):
                    sherpa_logger.error(f"Vosk 模型路径不存在: {model_config['path']}")
                    return False

                try:
                    # 创建 VoskASR 实例
                    self.current_engine = VoskASR(model_config["path"])
                    sherpa_logger.info(f"VoskASR 实例创建成功: {self.current_engine}")
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

                # 检查模型类型
                if engine_type == "sherpa_0626_int8":
                    model_type = "int8"
                    model_name = "0626"
                elif engine_type == "sherpa_0626_std":
                    model_type = "standard"
                    model_name = "0626"
                else:
                    model_type = "int8" if engine_type == "sherpa_int8" else "standard"
                    model_name = ""
                sherpa_logger.info(f"Sherpa-ONNX 模型类型: {model_type}, 模型名称: {model_name}")

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

            return True

        except Exception as e:
            sherpa_logger.error(f"初始化 {engine_type} 引擎错误: {str(e)}")
            import traceback
            sherpa_logger.error(traceback.format_exc())
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
        if self.model_type != engine_type:
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
            return None

        try:
            sherpa_logger.info(f"调用引擎的 transcribe_file 方法")
            sherpa_logger.info(f"使用引擎: {type(self.current_engine).__name__}")
            sherpa_logger.info(f"引擎类型: {engine_type}")
            result = self.current_engine.transcribe_file(file_path)
            sherpa_logger.info(f"转录结果: {result[:100]}..." if result and len(result) > 100 else f"转录结果: {result}")
            return True
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

        # 首先根据 current_engine 的类型推断
        if isinstance(self.current_engine, VoskASR):
            sherpa_logger.debug("当前引擎是 VoskASR")
            engine_type = "vosk"
        elif isinstance(self.current_engine, SherpaOnnxASR):
            sherpa_logger.debug("当前引擎是 SherpaOnnxASR")
            # 尝试从引擎实例获取更具体的类型
            if hasattr(self.current_engine, 'model_config') and self.current_engine.model_config:
                # 检查是否是0626模型
                if hasattr(self.current_engine, 'model_config') and self.current_engine.model_config.get("name") == "0626":
                    if self.current_engine.model_config.get("type") == "int8":
                        engine_type = "sherpa_0626_int8"
                    elif self.current_engine.model_config.get("type") == "standard":
                        engine_type = "sherpa_0626_std"
                    sherpa_logger.debug(f"当前引擎是 SherpaOnnxASR ({engine_type})")
                elif hasattr(self.current_engine, 'is_int8') and self.current_engine.is_int8:
                    engine_type = "sherpa_int8"
                    sherpa_logger.debug("当前引擎是 SherpaOnnxASR (int8)")
                else:
                    engine_type = "sherpa_std"
                    sherpa_logger.debug("当前引擎是 SherpaOnnxASR (std)")
            else:
                # 如果没有model_config，使用默认逻辑
                if hasattr(self.current_engine, 'is_int8') and self.current_engine.is_int8:
                    engine_type = "sherpa_int8"
                    sherpa_logger.debug("当前引擎是 SherpaOnnxASR (int8)")
                else:
                    engine_type = "sherpa_std"
                    sherpa_logger.debug("当前引擎是 SherpaOnnxASR (std)")
        else:
            sherpa_logger.debug("未识别的引擎类型")
            engine_type = None

        # 检查 model_type 和推断的引擎类型是否一致
        if self.model_type and engine_type and self.model_type != engine_type:
            sherpa_logger.warning(f"模型类型 {self.model_type} 与推断的引擎类型 {engine_type} 不一致")
            # 更新 model_type 为推断的引擎类型
            old_model_type = self.model_type
            self.model_type = engine_type
            sherpa_logger.info(f"模型类型已从 {old_model_type} 更新为: {self.model_type}")
        elif not self.model_type and engine_type:
            # 如果 model_type 为空但能推断出引擎类型，则更新 model_type
            self.model_type = engine_type
            sherpa_logger.info(f"模型类型已设置为: {self.model_type}")

        # 返回最终的引擎类型
        if self.model_type:
            sherpa_logger.debug(f"返回引擎类型: {self.model_type}")
            return self.model_type

        sherpa_logger.debug("无法确定引擎类型")
        return None

    def get_available_engines(self) -> Dict[str, bool]:
        """获取可用的引擎列表

        Returns:
            Dict[str, bool]: 引擎名称到是否启用的映射
        """
        engines = {}
        # 检查 vosk 引擎
        if "vosk" in self.models_config:
            model_config = self.models_config["vosk"]
            engines["vosk"] = bool(model_config and model_config.get("enabled", False))
        else:
            engines["vosk"] = False

        # 检查 sherpa_int8 引擎
        if "sherpa_int8" in self.models_config:
            model_config = self.models_config["sherpa_int8"]
            engines["sherpa_int8"] = bool(model_config and model_config.get("enabled", False))
        else:
            engines["sherpa_int8"] = False

        # 检查 sherpa_std 引擎
        if "sherpa_std" in self.models_config:
            model_config = self.models_config["sherpa_std"]
            engines["sherpa_std"] = bool(model_config and model_config.get("enabled", False))
        else:
            engines["sherpa_std"] = False

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
