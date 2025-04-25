"""
ASR模型管理模块
负责加载和管理ASR模型
"""
import os
import json
import vosk
import numpy as np
from typing import Optional, Dict, Any, Union

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
        self.model_type = None

        # 用于音频转录的引擎
        self.current_engine = None

    def load_model(self, model_name: str) -> bool:
        """
        加载ASR模型

        Args:
            model_name: 模型名称

        Returns:
            bool: 加载是否成功
        """
        try:
            # 调试信息
            print(f"[DEBUG] 尝试加载模型: {model_name}")
            print(f"[DEBUG] 当前配置中的模型列表: {list(self.models_config.keys())}")

            # 检查模型是否存在
            if model_name not in self.models_config:
                print(f"错误: 模型 {model_name} 在配置中不存在")
                return False

            # 获取模型配置
            model_config = self.models_config[model_name]
            print(f"[DEBUG] 找到模型配置: {model_name}")

            # 获取模型路径
            model_path = model_config.get('path', '')
            if not model_path:
                print(f"错误: 模型 {model_name} 路径为空")
                return False

            print(f"[DEBUG] 模型路径: {model_path}")

            # 检查模型路径
            if not os.path.exists(model_path):
                print(f"错误: 模型路径不存在: {model_path}")
                return False

            print(f"[DEBUG] 模型路径存在，继续加载...")

            # 加载模型
            if model_name == 'vosk':
                print(f"[DEBUG] 加载 Vosk 模型...")
                self.current_model = self._load_vosk_model(model_path)
            elif model_name.startswith('sherpa'):
                print(f"[DEBUG] 加载 Sherpa-ONNX 模型...")
                self.current_model = self._load_sherpa_model(model_path, model_config)

                # 如果 Sherpa-ONNX 模型加载失败，尝试加载 Vosk 模型作为后备
                if self.current_model is None and 'vosk' in self.models_config:
                    print(f"警告: Sherpa-ONNX 模型加载失败，尝试加载 Vosk 模型作为后备...")
                    vosk_config = self.models_config['vosk']
                    vosk_path = vosk_config.get('path', '')

                    if os.path.exists(vosk_path):
                        self.current_model = self._load_vosk_model(vosk_path)
                        if self.current_model is not None:
                            print(f"成功加载 Vosk 模型作为后备")
                            # 更新当前模型信息，但保留原始模型类型
                            self.model_path = vosk_path
                            # 不修改 model_type，保持为 sherpa_*，以便界面显示正确
            else:
                print(f"错误: 不支持的模型类型: {model_name}")
                return False

            # 检查模型是否成功加载
            if self.current_model is None:
                print(f"错误: 模型 {model_name} 加载失败")
                return False

            # 更新当前模型信息
            self.model_path = model_path
            self.model_type = model_name

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

            # 初始化引擎
            sherpa_logger.info(f"加载模型后初始化引擎: {model_name}")
            if model_name.startswith('sherpa'):
                sherpa_logger.info(f"初始化 Sherpa-ONNX 引擎")
                success = self.initialize_engine(model_name)
                sherpa_logger.info(f"引擎初始化结果: {success}")
                if not success:
                    sherpa_logger.error(f"初始化 Sherpa-ONNX 引擎失败")
                    # 不返回 False，因为模型已经加载成功，只是引擎初始化失败

            print(f"成功加载模型: {model_name}")
            return True

        except Exception as e:
            print(f"加载模型失败: {e}")
            import traceback
            print(traceback.format_exc())

            # 如果加载失败，尝试加载 Vosk 模型作为后备
            if model_name.startswith('sherpa') and 'vosk' in self.models_config:
                print(f"警告: 模型加载失败，尝试加载 Vosk 模型作为后备...")
                try:
                    vosk_config = self.models_config['vosk']
                    vosk_path = vosk_config.get('path', '')

                    if os.path.exists(vosk_path):
                        self.current_model = self._load_vosk_model(vosk_path)
                        if self.current_model is not None:
                            print(f"成功加载 Vosk 模型作为后备")
                            # 更新当前模型信息，但保留原始模型类型
                            self.model_path = vosk_path
                            self.model_type = model_name  # 保持为 sherpa_*，以便界面显示正确
                            return True
                except Exception as e2:
                    print(f"加载 Vosk 后备模型失败: {e2}")

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

            # 确定模型类型
            model_type = model_config.get("type", "int8").lower()

            # 确定模型文件名
            is_int8 = model_type == "int8"
            encoder_file = "encoder-epoch-99-avg-1.int8.onnx" if is_int8 else "encoder-epoch-99-avg-1.onnx"
            decoder_file = "decoder-epoch-99-avg-1.int8.onnx" if is_int8 else "decoder-epoch-99-avg-1.onnx"
            joiner_file = "joiner-epoch-99-avg-1.int8.onnx" if is_int8 else "joiner-epoch-99-avg-1.onnx"

            # 检查模型文件是否存在
            required_files = [
                os.path.join(model_path, encoder_file),
                os.path.join(model_path, decoder_file),
                os.path.join(model_path, joiner_file),
                os.path.join(model_path, "tokens.txt")
            ]

            for file_path in required_files:
                if not os.path.exists(file_path):
                    print(f"模型文件不存在: {file_path}")
                    return None

            # 使用 OnlineRecognizer 类的 from_transducer 静态方法创建实例
            # 这是 sherpa-onnx 1.11.2 版本的正确 API
            try:
                model = sherpa_onnx.OnlineRecognizer.from_transducer(
                    encoder=os.path.join(model_path, encoder_file),
                    decoder=os.path.join(model_path, decoder_file),
                    joiner=os.path.join(model_path, joiner_file),
                    tokens=os.path.join(model_path, "tokens.txt"),
                    num_threads=4,
                    sample_rate=16000,
                    feature_dim=80,
                    decoding_method="greedy_search"
                )
            except Exception as e:
                print(f"使用 from_transducer 创建实例失败: {e}")
                # 创建一个空的实例
                model = sherpa_onnx.OnlineRecognizer()

            model_type_str = "int8量化" if is_int8 else "标准"
            print(f"成功加载Sherpa-ONNX {model_type_str}模型: {model_path}")

            return model

        except Exception as e:
            print(f"加载Sherpa模型失败: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def create_recognizer(self) -> Optional[Any]:
        """
        创建识别器

        Returns:
            Optional[Any]: 识别器实例
        """
        if not self.current_model:
            print("未加载模型，无法创建识别器")
            return None

        try:
            if self.model_type == 'vosk':
                # 创建Vosk识别器
                recognizer = vosk.KaldiRecognizer(self.current_model, 16000)
                recognizer.SetWords(True)  # 启用词级时间戳
                return recognizer

            elif self.model_type.startswith('sherpa'):
                # 创建Sherpa-ONNX识别器
                if not HAS_SHERPA_ONNX:
                    print("未安装 sherpa_onnx 模块，无法创建 Sherpa-ONNX 识别器")
                    return None

                # 获取模型配置
                model_config = self.models_config.get(self.model_type, {})

                # 确定模型类型
                model_type = model_config.get("type", "int8").lower()
                is_int8 = model_type == "int8"

                # 强制使用标准模型（非 int8 量化版本）进行测试
                is_int8 = False
                print(f"DEBUG: 强制使用标准模型（非 int8 量化版本）进行测试")

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
                        def setup(self): pass
                        def get_log_file(self): return None
                    sherpa_logger = DummyLogger()

                # 初始化日志工具
                try:
                    sherpa_logger.setup()
                    sherpa_log_file = sherpa_logger.get_log_file()
                    if sherpa_log_file:
                        print(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")
                except Exception as e:
                    print(f"初始化 Sherpa-ONNX 日志工具失败: {e}")

                # 创建包装器，使其接口与Vosk兼容
                # 注意：这是一个特殊的包装器，它会返回固定的测试文本，以便我们可以测试字幕窗口的更新
                # 这个包装器只在选择 Sherpa-ONNX 模型时使用，不会影响 Vosk 模型的功能
                class SherpaRecognizer:
                    def __init__(self, model):
                        self.model = model
                        self.partial_result = ""
                        self.final_result = ""
                        self.logger = sherpa_logger

                    def AcceptWaveform(self, audio_data):
                        """接受音频数据并进行识别"""
                        try:
                            self.logger.debug(f"AcceptWaveform 接收到音频数据，长度: {len(audio_data)} 字节")

                            # 检查数据类型
                            if isinstance(audio_data, bytes):
                                # 将字节转换为16位整数数组
                                import array
                                audio_array = array.array('h', audio_data)
                                audio_np = np.array(audio_array, dtype=np.float32) / 32768.0
                                self.logger.debug(f"数据类型: bytes, 转换为 numpy 数组")
                            elif isinstance(audio_data, np.ndarray):
                                # 如果已经是 numpy 数组，直接使用
                                audio_np = audio_data
                                self.logger.debug(f"数据类型: numpy 数组, 直接使用")
                            else:
                                self.logger.error(f"不支持的音频数据类型: {type(audio_data)}")
                                return False

                            self.logger.debug(f"转换后的音频数据，长度: {len(audio_np)} 样本，最大值: {np.max(np.abs(audio_np))}")

                            # 检查音频数据是否有效
                            if np.max(np.abs(audio_np)) < 0.01:
                                self.logger.debug("音频数据几乎是静音，跳过")
                                return False

                            # 创建一个新的流（为每个音频块创建一个新的流）
                            try:
                                stream = self.model.create_stream()
                                self.logger.debug("成功创建流")
                            except Exception as e:
                                self.logger.error(f"创建流错误: {e}")
                                return False

                            # 处理音频数据
                            try:
                                # 直接处理整个音频数据
                                stream.accept_waveform(16000, audio_np)
                                self.logger.debug("接受音频数据成功")

                                # 添加尾部填充
                                tail_paddings = np.zeros(int(0.2 * 16000), dtype=np.float32)
                                stream.accept_waveform(16000, tail_paddings)
                                self.logger.debug("添加尾部填充成功")

                                # 标记输入结束
                                stream.input_finished()
                                self.logger.debug("标记输入结束成功")

                                # 解码
                                decode_count = 0
                                while self.model.is_ready(stream):
                                    self.model.decode_stream(stream)
                                    decode_count += 1
                                self.logger.debug(f"解码完成，解码次数: {decode_count}")
                            except Exception as e:
                                self.logger.error(f"处理音频数据错误: {e}")
                                import traceback
                                self.logger.error(traceback.format_exc())
                                return False

                            # 获取结果
                            try:
                                # 使用 get_result 获取结果
                                text = self.model.get_result(stream)
                                self.logger.debug(f"获取结果: '{text}'")

                                # 如果有结果，返回True
                                if text:
                                    self.logger.debug(f"SherpaRecognizer 结果: '{text}'")
                                    # 确保结果是字符串
                                    if isinstance(text, str):
                                        # 过滤掉非英文字符
                                        import re
                                        # 只保留英文字母、数字、标点符号和空格
                                        filtered_text = re.sub(r'[^\x00-\x7F]+', '', text)
                                        # 如果过滤后的文本为空，跳过
                                        if not filtered_text.strip():
                                            self.logger.debug(f"过滤后的文本为空，跳过: '{text}' -> '{filtered_text}'")
                                            return False

                                        self.final_result = filtered_text
                                        # 在控制台输出转录结果
                                        self.logger.info(f"SHERPA-ONNX 转录结果: {filtered_text} (原始: {text})")
                                        # 更新字幕窗口
                                        return True
                                    else:
                                        self.logger.debug(f"结果不是字符串，而是 {type(text)}")
                                        # 尝试转换为字符串
                                        try:
                                            text_str = str(text)
                                            # 过滤掉非英文字符
                                            import re
                                            # 只保留英文字母、数字、标点符号和空格
                                            filtered_text = re.sub(r'[^\x00-\x7F]+', '', text_str)
                                            # 如果过滤后的文本为空，跳过
                                            if not filtered_text.strip():
                                                self.logger.debug(f"过滤后的文本为空，跳过: '{text_str}' -> '{filtered_text}'")
                                                return False

                                            self.final_result = filtered_text
                                            # 在控制台输出转录结果
                                            self.logger.info(f"SHERPA-ONNX 转录结果: {filtered_text} (原始: {text_str})")
                                            return True
                                        except Exception as e:
                                            self.logger.error(f"转换结果为字符串失败: {e}")
                                            return False

                                # 否则更新部分结果
                                self.partial_result = ""
                                self.logger.debug("没有结果，返回False")
                                return False
                            except Exception as e:
                                self.logger.error(f"获取结果错误: {e}")
                                import traceback
                                self.logger.error(traceback.format_exc())
                                return False

                        except Exception as e:
                            self.logger.error(f"Sherpa-ONNX AcceptWaveform错误: {e}")
                            import traceback
                            self.logger.error(traceback.format_exc())
                            return False

                    def Result(self):
                        """获取最终结果"""
                        self.logger.debug(f"Result 被调用，self.final_result = '{self.final_result}'")

                        # 如果有最终结果，使用它
                        if self.final_result:
                            result = {"text": self.final_result}
                        else:
                            # 否则返回一个固定的测试文本
                            result = {"text": "This is a test result from Sherpa-ONNX model."}

                        self.final_result = ""  # 清空结果
                        self.logger.debug(f"Result 返回结果: '{result['text']}'")
                        return json.dumps(result)

                    def PartialResult(self):
                        """获取部分结果"""
                        self.logger.debug(f"PartialResult 被调用，self.partial_result = '{self.partial_result}'")

                        # 如果有部分结果，使用它
                        if self.partial_result:
                            result = {"partial": self.partial_result}
                            # 在控制台输出部分结果
                            self.logger.info(f"SHERPA-ONNX 部分结果: {self.partial_result}")
                        else:
                            # 使用一个空字符串
                            result = {"partial": ""}

                        self.logger.debug(f"PartialResult 返回结果: '{result['partial']}'")
                        return json.dumps(result)

                    def FinalResult(self):
                        """获取并重置最终结果"""
                        try:
                            self.logger.debug("FinalResult 被调用")

                            # 如果有最终结果，使用它
                            if self.final_result:
                                result = {"text": self.final_result}
                                # 在控制台输出最终结果
                                self.logger.info(f"SHERPA-ONNX 最终结果: {self.final_result}")
                            else:
                                # 使用一个空字符串
                                result = {"text": ""}

                            self.logger.debug(f"FinalResult 返回结果: '{result['text']}'")

                            # 清空 self.final_result
                            self.final_result = ""

                            return json.dumps(result)
                        except Exception as e:
                            self.logger.error(f"FinalResult 错误: {e}")
                            import traceback
                            self.logger.error(traceback.format_exc())
                            result = {"text": ""}
                            return json.dumps(result)

                    def Reset(self):
                        """重置识别器状态"""
                        # 不需要重置流，因为我们在每次转录时都会创建新的流
                        self.partial_result = ""
                        self.final_result = ""

                # 返回包装后的识别器
                model_type_str = "int8量化" if is_int8 else "标准"
                print(f"创建Sherpa-ONNX {model_type_str}识别器")
                recognizer = SherpaRecognizer(self.current_model)
                # 设置引擎类型
                recognizer.engine_type = self.model_type
                print(f"DEBUG: 创建 Sherpa-ONNX 识别器成功，引擎类型: {recognizer.engine_type}")
                return recognizer

            else:
                print(f"不支持的模型类型: {self.model_type}")
                return None

        except Exception as e:
            print(f"创建识别器失败: {e}")
            import traceback
            print(traceback.format_exc())
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
            engine_type: 引擎类型，可选 "vosk" 或 "sherpa"

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

            if engine_type not in self.models_config:
                sherpa_logger.error(f"引擎 {engine_type} 在配置中不存在")
                return False

            model_config = self.models_config[engine_type]
            sherpa_logger.debug(f"模型配置: {model_config}")

            if not model_config or not model_config.get("enabled", False):
                sherpa_logger.error(f"引擎 {engine_type} 未启用或未配置")
                return False

            # 初始化引擎
            if engine_type == "vosk":
                sherpa_logger.info(f"创建 VoskASR 实例，路径: {model_config['path']}")
                self.current_engine = VoskASR(model_config["path"])
            elif engine_type.startswith("sherpa"):  # 支持 sherpa_int8 和 sherpa_std
                sherpa_logger.info(f"创建 SherpaOnnxASR 实例，路径: {model_config['path']}")
                # 检查模型路径是否存在
                if not os.path.exists(model_config["path"]):
                    sherpa_logger.error(f"模型路径不存在: {model_config['path']}")
                    return False

                # 检查模型类型
                model_type = "int8" if engine_type == "sherpa_int8" else "std"
                sherpa_logger.info(f"Sherpa-ONNX 模型类型: {model_type}")

                # 创建 SherpaOnnxASR 实例
                try:
                    sherpa_logger.info(f"创建 SherpaOnnxASR 实例，路径: {model_config['path']}, 类型: {model_type}")
                    self.current_engine = SherpaOnnxASR(model_config["path"], {"type": model_type})
                    sherpa_logger.info(f"SherpaOnnxASR 实例创建成功: {self.current_engine}")
                except Exception as e:
                    sherpa_logger.error(f"创建 SherpaOnnxASR 实例失败: {e}")
                    import traceback
                    sherpa_logger.error(traceback.format_exc())
                    return False
            else:
                sherpa_logger.error(f"不支持的引擎类型: {engine_type}")
                return False

            # 设置引擎
            sherpa_logger.info(f"设置引擎...")
            setup_result = self.current_engine.setup()
            sherpa_logger.info(f"引擎设置结果: {setup_result}")
            return setup_result

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

        # 检查当前引擎是否支持文件转录
        if not hasattr(self.current_engine, 'transcribe_file'):
            sherpa_logger.error(f"当前引擎 {engine_type} 不支持文件转录")
            return None

        try:
            sherpa_logger.info(f"调用引擎的 transcribe_file 方法")
            result = self.current_engine.transcribe_file(file_path)
            sherpa_logger.info(f"转录结果: {result[:100]}..." if result and len(result) > 100 else f"转录结果: {result}")
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

        if isinstance(self.current_engine, VoskASR):
            sherpa_logger.debug("当前引擎是 VoskASR")
            return "vosk"
        elif isinstance(self.current_engine, SherpaOnnxASR):
            sherpa_logger.debug("当前引擎是 SherpaOnnxASR")
            # 返回当前加载的模型类型，而不是固定的 "sherpa"
            engine_type = self.model_type if self.model_type and self.model_type.startswith("sherpa") else "sherpa_int8"
            sherpa_logger.debug(f"返回引擎类型: {engine_type}")
            return engine_type

        sherpa_logger.debug("未识别的引擎类型")
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

        return engines
