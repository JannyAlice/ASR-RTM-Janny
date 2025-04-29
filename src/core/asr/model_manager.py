"""
ASR模型管理模块
负责加载和管理ASR模型
"""
import os
import json
import time
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

                # 不再自动降级到Vosk模型
                if self.current_model is None:
                    print(f"错误: Sherpa-ONNX 模型 {model_name} 加载失败，不会自动降级到 Vosk 模型")
                    return False
            else:
                print(f"错误: 不支持的模型类型: {model_name}")
                return False

            # 检查模型是否成功加载
            if self.current_model is None:
                print(f"错误: 模型 {model_name} 加载失败")
                return False

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

            # 记录当前状态
            sherpa_logger.info(f"开始加载模型: {model_name}")
            sherpa_logger.info(f"当前模型类型: {self.model_type}")
            sherpa_logger.info(f"当前引擎类型: {self.get_current_engine_type()}")
            sherpa_logger.info(f"当前引擎: {type(self.current_engine).__name__ if self.current_engine else None}")

            # 更新当前模型信息
            self.model_path = model_path
            self.model_type = model_name

            # 记录更新后的状态
            sherpa_logger.info(f"模型路径已更新: {self.model_path}")
            sherpa_logger.info(f"模型类型已更新: {self.model_type}")

            # 初始化引擎 - 对所有模型类型都初始化引擎
            sherpa_logger.info(f"加载模型后初始化引擎: {model_name}")
            success = self.initialize_engine(model_name)
            sherpa_logger.info(f"引擎初始化结果: {success}")

            # 记录初始化后的状态
            sherpa_logger.info(f"初始化后的引擎类型: {self.get_current_engine_type()}")
            sherpa_logger.info(f"初始化后的引擎: {type(self.current_engine).__name__ if self.current_engine else None}")

            if not success:
                sherpa_logger.error(f"初始化引擎失败: {model_name}")
                # 不返回 False，因为模型已经加载成功，只是引擎初始化失败

            print(f"成功加载模型: {model_name}")
            return True

        except Exception as e:
            print(f"加载模型失败: {e}")
            import traceback
            print(traceback.format_exc())

            # 不再自动降级到Vosk模型
            if model_name.startswith('sherpa'):
                print(f"错误: Sherpa-ONNX 模型 {model_name} 加载失败，不会自动降级到 Vosk 模型")

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
            is_0626 = "0626" in model_name or self.model_path and "2023-06-26" in self.model_path

            # 从配置文件中获取模型文件名
            config_section = model_config.get("config", {})

            # 如果配置文件中有指定模型文件名，则使用配置文件中的值
            if "encoder" in config_section and "decoder" in config_section and "joiner" in config_section:
                encoder_file = config_section["encoder"]
                decoder_file = config_section["decoder"]
                joiner_file = config_section["joiner"]
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
            sherpa_logger.info("创建识别器")
            sherpa_logger.info(f"当前引擎: {type(self.current_engine).__name__ if self.current_engine else None}")
            sherpa_logger.info(f"当前模型类型: {self.model_type}")

            if not self.current_model:
                error_msg = "未加载模型，无法创建识别器"
                sherpa_logger.error(error_msg)
                print(error_msg)
                return None

            # 检查模型类型和实际模型的一致性
            if isinstance(self.current_model, vosk.Model):
                # 如果实际模型是 Vosk 模型，但模型类型不是 vosk
                if self.model_type != 'vosk':
                    # 严格检查：如果用户明确要求使用sherpa模型，但实际加载的是vosk模型，则报错
                    if self.model_type.startswith('sherpa'):
                        error_msg = f"错误：用户请求使用 {self.model_type} 模型，但实际加载的是 vosk 模型。不允许自动降级到 vosk 模型。"
                        sherpa_logger.error(error_msg)
                        print(error_msg)
                        return None
                    else:
                        # 其他情况下，更新模型类型为vosk
                        sherpa_logger.warning(f"模型类型 {self.model_type} 与实际模型类型 vosk 不一致，更新为 vosk")
                        self.model_type = 'vosk'

                # 创建Vosk识别器
                sherpa_logger.info("创建 Vosk 识别器")
                recognizer = vosk.KaldiRecognizer(self.current_model, 16000)
                recognizer.SetWords(True)  # 启用词级时间戳

                # 设置引擎类型
                recognizer.engine_type = self.model_type
                sherpa_logger.info(f"设置 Vosk 识别器引擎类型: {self.model_type}")

                sherpa_logger.info(f"Vosk 识别器创建成功: {type(recognizer).__name__}")
                return recognizer

            elif self.model_type.startswith('sherpa'):
                # 检查是否实际是 Sherpa-ONNX 模型
                if not isinstance(self.current_model, sherpa_onnx.OnlineRecognizer):
                    error_msg = f"错误：模型类型 {self.model_type} 与实际模型类型不一致，无法创建 Sherpa-ONNX 识别器"
                    sherpa_logger.error(error_msg)
                    print(error_msg)

                    # 不再自动降级到Vosk模型
                    return None

                # 创建Sherpa-ONNX识别器
                sherpa_logger.info("创建 Sherpa-ONNX 识别器")

                if not HAS_SHERPA_ONNX:
                    error_msg = "未安装 sherpa_onnx 模块，无法创建 Sherpa-ONNX 识别器"
                    sherpa_logger.error(error_msg)
                    print(error_msg)
                    return None

                # 获取模型配置
                model_config = self.models_config.get(self.model_type, {})
                sherpa_logger.info(f"模型配置: {model_config}")

                # 确定模型类型
                model_type = model_config.get("type", "int8").lower()
                is_int8 = model_type == "int8"
                sherpa_logger.info(f"Sherpa-ONNX 模型类型: {model_type}, is_int8: {is_int8}")

                # 不再强制使用标准模型
                # sherpa_logger.info("使用配置中指定的模型类型")

                # 初始化日志工具
                try:
                    sherpa_logger.setup()
                    sherpa_log_file = sherpa_logger.get_log_file()
                    if sherpa_log_file:
                        sherpa_logger.info(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")
                        print(f"Sherpa-ONNX 日志文件: {sherpa_log_file}")
                except Exception as e:
                    error_msg = f"初始化 Sherpa-ONNX 日志工具失败: {e}"
                    sherpa_logger.error(error_msg)
                    print(error_msg)

                # 创建包装器，使其接口与Vosk兼容
                # 注意：这是一个特殊的包装器，它会返回固定的测试文本，以便我们可以测试字幕窗口的更新
                # 这个包装器只在选择 Sherpa-ONNX 模型时使用，不会影响 Vosk 模型的功能
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

                        # 添加累积文本功能
                        self.accumulated_text = []  # 存储所有完整的转录文本，用于保存完整历史

                        # 添加时间戳功能
                        self.start_time = time.time()  # 记录开始时间
                        self.timestamps = []  # 存储每段文本的时间戳 [(text, timestamp), ...]

                        # 添加句子完成事件回调函数
                        self.sentence_done_callback = None
                        self.text_changed_callback = None

                        # 添加日志文件路径
                        self.log_file = None

                        # 添加句子长度限制（参考TMSpeech）
                        self.max_sentence_length = 80  # 字符数

                        # 添加端点检测参数
                        self.enable_endpoint_detection = True
                        self.rule1_min_trailing_silence = 2.4  # 基本静音阈值(秒)
                        self.rule2_min_trailing_silence = 1.2  # 长句中的静音阈值(秒)
                        self.rule3_min_utterance_length = 20   # 长句判定阈值(帧)

                        # 添加静音检测参数
                        self.silence_threshold = 0.01  # 静音阈值
                        self.silence_duration = 0.0    # 当前静音持续时间
                        self.last_audio_time = time.time()  # 上次接收到有效音频的时间

                        # 在初始化时创建一个持久的流
                        # 这与Sherpa-ONNX官方测试文件的做法一致，与Vosk的处理方式不同
                        # Vosk每次AcceptWaveform都使用同一个识别器实例，而Sherpa-ONNX需要使用流来管理状态
                        try:
                            self.stream = self.model.create_stream()
                            self.logger.info("成功创建持久的流")
                        except Exception as e:
                            self.logger.error(f"创建持久的流错误: {e}")
                            import traceback
                            self.logger.error(traceback.format_exc())
                            self.stream = None

                    def AcceptWaveform(self, audio_data):
                        """
                        接受音频数据并进行识别

                        这个方法模拟了Vosk的AcceptWaveform方法，但内部使用Sherpa-ONNX的API。
                        与Vosk不同，Sherpa-ONNX需要使用流来管理状态，并且需要特殊的处理步骤：
                        1. 接受音频数据
                        2. 添加尾部填充
                        3. 标记输入结束
                        4. 解码
                        5. 获取结果

                        此外，为了解决Sherpa-ONNX模型的文本累积问题，我们添加了文本增量检测逻辑，
                        只返回新增的部分作为最终结果，而不是完整的句子。这样可以模拟Vosk的行为，
                        避免字幕窗口中出现重复文本。

                        Args:
                            audio_data: 音频数据，可以是字节或numpy数组

                        Returns:
                            bool: 如果有新的最终结果返回True，否则返回False
                        """
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

                            # 检查持久的流是否存在，如果不存在则创建
                            if not hasattr(self, 'stream') or self.stream is None:
                                try:
                                    self.stream = self.model.create_stream()
                                    self.logger.info("创建新的持久流")
                                except Exception as e:
                                    self.logger.error(f"创建持久流错误: {e}")
                                    import traceback
                                    self.logger.error(traceback.format_exc())
                                    return False

                            # 处理音频数据
                            try:
                                # 1. 接受音频数据
                                self.stream.accept_waveform(16000, audio_np)
                                self.logger.debug("接受音频数据成功")

                                # 2. 添加尾部填充
                                tail_paddings = np.zeros(int(0.2 * 16000), dtype=np.float32)
                                self.stream.accept_waveform(16000, tail_paddings)
                                self.logger.debug("添加尾部填充成功")

                                # 3. 不再标记输入结束，以保持流的持久性
                                # 只有在最终结果时才标记输入结束
                                # self.stream.input_finished()
                                self.logger.debug("保持流的持久性，不标记输入结束")

                                # 4. 解码
                                decode_count = 0
                                while self.model.is_ready(self.stream):
                                    self.model.decode_stream(self.stream)
                                    decode_count += 1
                                self.logger.debug(f"解码完成，解码次数: {decode_count}")
                            except Exception as e:
                                self.logger.error(f"处理音频数据错误: {e}")
                                import traceback
                                self.logger.error(traceback.format_exc())
                                return False

                            # 获取结果
                            try:
                                # 5. 使用 get_result 获取结果
                                text = self.model.get_result(self.stream)
                                self.logger.debug(f"获取结果: '{text}'")

                                # 如果有结果，返回True
                                if text and text.strip():
                                    self.logger.debug(f"SherpaRecognizer 结果: '{text}'")
                                    # 确保结果是字符串
                                    if isinstance(text, str):
                                        # 不再过滤非英文字符，保留所有字符
                                        # 只进行基本的清理
                                        filtered_text = text.strip()
                                        # 如果文本为空，跳过
                                        if not filtered_text:
                                            self.logger.debug(f"清理后的文本为空，跳过: '{text}' -> '{filtered_text}'")
                                            return False

                                        # 不在这里添加标点符号，让Result方法处理
                                        # 只进行基本的文本清理
                                        filtered_text = filtered_text.strip()
                                        self.logger.debug(f"清理后的文本: '{filtered_text}'")

                                        # 不在这里进行首字母大写处理，让Result方法处理
                                        # 保留原始大小写
                                        self.logger.debug(f"保留原始大小写的文本: '{filtered_text}'")

                                        # 设置部分结果
                                        self.partial_result = filtered_text

                                        # 检查文本是否足够长
                                        words = filtered_text.split()

                                        # 如果文本足够长（超过20个单词或100个字符），则视为最终结果
                                        # 这个阈值可以根据实际情况调整
                                        is_long_enough = len(words) >= 20 or len(filtered_text) >= 100

                                        # 检查是否包含句子结束标志
                                        contains_end_mark = '.' in filtered_text or '?' in filtered_text or '!' in filtered_text

                                        # 如果文本足够长或包含句子结束标志，则视为最终结果
                                        if is_long_enough or contains_end_mark:
                                            # 更新当前累积的文本
                                            self.current_text = filtered_text

                                            # 设置最终结果
                                            self.final_result = filtered_text

                                            # 更新上一次的结果
                                            self.previous_result = filtered_text

                                            # 记录时间戳
                                            current_time = time.time()
                                            elapsed_time = current_time - self.start_time
                                            minutes = int(elapsed_time // 60)
                                            seconds = int(elapsed_time % 60)
                                            timestamp = f"{minutes:02d}:{seconds:02d}"

                                            # 添加到累积文本和时间戳列表
                                            self.accumulated_text.append(filtered_text)
                                            self.timestamps.append((filtered_text, timestamp))

                                            # 记录日志
                                            self.logger.info(f"SHERPA-ONNX 最终结果 [{timestamp}]: {filtered_text}")

                                            # 返回True，表示这是最终结果
                                            return True
                                        else:
                                            # 文本不够长，视为部分结果
                                            self.logger.debug(f"文本不够长，视为部分结果: '{filtered_text}', 单词数: {len(words)}, 字符数: {len(filtered_text)}")
                                            return False

                                        # 以下代码不会被执行，因为上面已经返回
                                        # 保留这些代码是为了参考，以及在将来可能需要时使用
                                        # 检查是否与上一次结果相同或只是增量
                                        if self.previous_result and filtered_text.startswith(self.previous_result):
                                            # 检查是否有新增内容
                                            new_text = filtered_text[len(self.previous_result):].strip()
                                            if new_text:
                                                # 更新当前累积的文本
                                                self.current_text = filtered_text

                                                # 只返回新增的部分作为最终结果
                                                # 这样可以模拟Vosk模型的行为
                                                self.final_result = new_text

                                                # 更新上一次的结果
                                                self.previous_result = filtered_text

                                                # 记录日志
                                                self.logger.info(f"SHERPA-ONNX 增量结果: {new_text} (完整: {filtered_text})")

                                                # 设置部分结果为空，因为我们已经有了最终结果
                                                self.partial_result = ""

                                                return True
                                            else:
                                                self.logger.debug(f"没有新增内容，跳过")

                                                # 设置部分结果为当前文本
                                                self.partial_result = filtered_text

                                                return False
                                        else:
                                            # 完全新的文本或不是增量
                                            # 检查是否是第一次结果
                                            if not self.previous_result:
                                                # 第一次结果，直接使用
                                                self.current_text = filtered_text
                                                self.final_result = filtered_text
                                                self.previous_result = filtered_text
                                                self.logger.info(f"SHERPA-ONNX 首次结果: {filtered_text} (原始: {text})")
                                                self.partial_result = ""
                                                return True
                                            else:
                                                # 不是第一次结果，但也不是增量
                                                # 这可能是一个新的句子，或者是识别错误
                                                # 我们将其视为新的句子，完全替换之前的结果
                                                self.current_text = filtered_text
                                                self.final_result = filtered_text
                                                self.previous_result = filtered_text
                                                self.logger.info(f"SHERPA-ONNX 新句子: {filtered_text} (原始: {text})")
                                                self.partial_result = ""
                                                return True
                                    else:
                                        self.logger.debug(f"结果不是字符串，而是 {type(text)}")
                                        # 尝试转换为字符串
                                        try:
                                            text_str = str(text)
                                            # 不再过滤非英文字符，保留所有字符
                                            # 只进行基本的清理
                                            filtered_text = text_str.strip()
                                            # 如果文本为空，跳过
                                            if not filtered_text:
                                                self.logger.debug(f"清理后的文本为空，跳过: '{text_str}' -> '{filtered_text}'")
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
                        """
                        获取最终结果

                        这个方法模拟了Vosk的Result方法，返回当前的最终识别结果。
                        与Vosk不同，我们添加了文本格式化功能，使结果更加易读：
                        - 首字母大写
                        - 添加句尾标点
                        - 处理问句开头

                        此外，如果没有最终结果，但有当前累积的文本，则返回当前文本。
                        这样可以确保在连续识别过程中不会丢失已识别的文本。

                        注意：此方法不会重置previous_result和current_text变量，
                        因为我们需要它们来跟踪文本增量。这与Vosk的行为不同。

                        Returns:
                            str: JSON格式的结果，包含"text"字段
                        """
                        self.logger.debug(f"Result 被调用，self.final_result = '{self.final_result}'")

                        # 使用最终结果
                        if self.final_result:
                            # 格式化文本：首字母大写，添加标点符号
                            formatted_text = self._format_text(self.final_result)
                            result = {"text": formatted_text}
                            self.logger.debug(f"使用已有的最终结果: '{formatted_text}' (原始: '{self.final_result}')")
                        else:
                            # 如果没有最终结果，但有当前累积的文本，则返回当前文本
                            if self.current_text:
                                # 格式化文本：首字母大写，添加标点符号
                                formatted_text = self._format_text(self.current_text)
                                result = {"text": formatted_text}
                                self.logger.debug(f"使用当前文本: '{formatted_text}' (原始: '{self.current_text}')")
                            else:
                                # 没有结果，返回空字符串
                                result = {"text": ""}
                                self.logger.debug("没有最终结果，返回空字符串")

                        # 记录最终结果
                        final_result = result["text"]
                        self.logger.debug(f"Result 返回结果: '{final_result}'")

                        # 清空结果，准备下一次识别
                        self.final_result = ""
                        # 注意：这里不重置 previous_result 和 current_text，因为我们需要它们来跟踪文本增量

                        return json.dumps(result)

                    def _format_text(self, text):
                        """
                        格式化文本

                        对文本进行格式化处理：
                        1. 确保段首字母大写
                        2. 添加句尾标点符号
                        3. 处理问句开头
                        4. 修复常见的格式问题

                        Args:
                            text (str): 原始文本

                        Returns:
                            str: 格式化后的文本
                        """
                        if not text:
                            return text

                        # 去除多余的空格
                        text = ' '.join(text.split())

                        # 修复常见的格式问题
                        # 1. 移除中间的多余句号（Sherpa模型常见问题）
                        # 但保留句子结尾的句号
                        sentences = []
                        for part in text.split('.'):
                            part = part.strip()
                            if part:
                                sentences.append(part)
                        text = ' '.join(sentences)

                        # 确保段首字母大写
                        if len(text) > 0:
                            text = text[0].upper() + text[1:]

                        # 如果文本末尾没有标点符号，添加句号
                        if text and text[-1] not in ['.', '?', '!', ',', ';', ':', '-']:
                            text += '.'

                        # 处理常见的问句开头
                        question_starters = ['what', 'who', 'where', 'when', 'why', 'how',
                                           'is', 'are', 'do', 'does', 'did', 'can', 'could',
                                           'will', 'would', 'should']
                        words = text.split()
                        if words and words[0].lower() in question_starters:
                            # 将句尾的句号替换为问号
                            if text[-1] == '.':
                                text = text[:-1] + '?'

                        # 修复常见的拼写错误
                        common_fixes = {
                            ' i ': ' I ',
                            ' im ': ' I\'m ',
                            ' dont ': ' don\'t ',
                            ' cant ': ' can\'t ',
                            ' wont ': ' won\'t ',
                            ' ive ': ' I\'ve ',
                            ' id ': ' I\'d ',
                            ' ill ': ' I\'ll '
                        }

                        for wrong, correct in common_fixes.items():
                            text = text.replace(wrong, correct)

                        # 确保句子之间有适当的空格
                        text = text.replace('.','. ').replace('?','? ').replace('!','! ')
                        text = ' '.join(text.split())  # 再次去除多余的空格

                        return text

                    def _format_partial_text(self, text):
                        """
                        格式化部分文本

                        对部分文本进行格式化处理：
                        1. 确保段首字母大写
                        2. 不添加额外的标点符号（与完整文本不同）

                        Args:
                            text (str): 原始文本

                        Returns:
                            str: 格式化后的文本
                        """
                        if not text:
                            return text

                        # 确保段首字母大写
                        if len(text) > 0:
                            text = text[0].upper() + text[1:]

                        # 部分文本不添加句尾标点符号，以区别于完整文本

                        return text

                    def _calculate_similarity(self, text1, text2):
                        """
                        计算两段文本的相似度

                        使用简单的字符匹配算法计算两段文本的相似度

                        Args:
                            text1 (str): 第一段文本
                            text2 (str): 第二段文本

                        Returns:
                            float: 相似度，范围为0-1
                        """
                        if not text1 or not text2:
                            return 0

                        # 如果其中一个文本是另一个的子串，返回较短文本长度与较长文本长度的比值
                        if text1 in text2:
                            return len(text1) / len(text2)
                        if text2 in text1:
                            return len(text2) / len(text1)

                        # 计算最长公共子串
                        m = len(text1)
                        n = len(text2)
                        dp = [[0] * (n + 1) for _ in range(m + 1)]
                        max_len = 0

                        for i in range(1, m + 1):
                            for j in range(1, n + 1):
                                if text1[i - 1] == text2[j - 1]:
                                    dp[i][j] = dp[i - 1][j - 1] + 1
                                    max_len = max(max_len, dp[i][j])

                        # 返回最长公共子串长度与较长文本长度的比值
                        return max_len / max(m, n)

                    def PartialResult(self):
                        """
                        获取部分结果

                        这个方法模拟了Vosk的PartialResult方法，返回当前的部分识别结果。
                        与Vosk不同，Sherpa-ONNX没有原生的部分结果概念，我们通过以下方式模拟：
                        1. 如果有部分结果（在AcceptWaveform中设置），则返回它
                        2. 如果没有部分结果，但有当前累积的文本，则返回当前文本
                        3. 如果都没有，则返回空字符串

                        此外，我们使用_format_partial_text方法格式化部分结果，
                        只进行首字母大写处理，不添加句尾标点。这与Vosk的行为类似。

                        Returns:
                            str: JSON格式的结果，包含"partial"字段
                        """
                        self.logger.debug(f"PartialResult 被调用")

                        # 返回当前的部分结果
                        if self.partial_result:
                            # 格式化部分文本（不添加句尾标点）
                            formatted_text = self._format_partial_text(self.partial_result)
                            result = {"partial": formatted_text}
                            self.logger.debug(f"PartialResult 返回结果: '{formatted_text}' (原始: '{self.partial_result}')")
                        else:
                            # 如果没有部分结果，但有当前累积的文本，则返回当前文本
                            if self.current_text:
                                # 格式化部分文本（不添加句尾标点）
                                formatted_text = self._format_partial_text(self.current_text)
                                result = {"partial": formatted_text}
                                self.logger.debug(f"PartialResult 返回当前文本: '{formatted_text}' (原始: '{self.current_text}')")
                            else:
                                # 没有结果，返回空字符串
                                result = {"partial": ""}
                                self.logger.debug(f"PartialResult 返回空结果")

                        return json.dumps(result)

                    def FinalResult(self):
                        """
                        获取并重置最终结果

                        这个方法模拟了Vosk的FinalResult方法，返回当前的最终识别结果，
                        并重置所有状态变量，准备下一次识别。

                        与Vosk不同，我们添加了文本格式化功能，使结果更加易读：
                        - 首字母大写
                        - 添加句尾标点
                        - 处理问句开头

                        此外，如果没有最终结果，但有当前累积的文本，则返回当前文本。
                        这样可以确保在连续识别过程中不会丢失已识别的文本。

                        注意：此方法会重置所有状态变量，包括previous_result、current_text和partial_result。
                        这与Vosk的行为一致，表示一次完整的识别过程结束。

                        Returns:
                            str: JSON格式的结果，包含"text"字段
                        """
                        try:
                            self.logger.debug("FinalResult 被调用")

                            # 标记输入结束，获取最终结果
                            if hasattr(self, 'stream') and self.stream:
                                try:
                                    self.stream.input_finished()
                                    self.logger.debug("标记输入结束成功")
                                except Exception as e:
                                    self.logger.error(f"标记输入结束错误: {e}")
                                    pass

                            # 使用部分结果作为最终结果
                            # 这是因为我们在AcceptWaveform中始终返回False，所以所有文本都被视为部分结果
                            if self.partial_result:
                                # 格式化文本：首字母大写，添加标点符号
                                formatted_text = self._format_text(self.partial_result)
                                result = {"text": formatted_text}
                                # 在控制台输出最终结果
                                self.logger.info(f"SHERPA-ONNX 最终结果(使用部分结果): {formatted_text} (原始: {self.partial_result})")
                            # 如果没有部分结果，但有当前累积的文本，则返回当前文本
                            elif self.current_text:
                                # 格式化文本：首字母大写，添加标点符号
                                formatted_text = self._format_text(self.current_text)
                                result = {"text": formatted_text}
                                self.logger.info(f"SHERPA-ONNX 最终结果(使用当前文本): {formatted_text} (原始: {self.current_text})")
                            # 如果没有部分结果和当前累积的文本，但有最终结果，则返回最终结果
                            elif self.final_result:
                                # 格式化文本：首字母大写，添加标点符号
                                formatted_text = self._format_text(self.final_result)
                                result = {"text": formatted_text}
                                # 在控制台输出最终结果
                                self.logger.info(f"SHERPA-ONNX 最终结果: {formatted_text} (原始: {self.final_result})")
                            else:
                                # 没有结果，返回空字符串
                                result = {"text": ""}
                                self.logger.debug("没有最终结果，返回空字符串")

                            self.logger.debug(f"FinalResult 返回结果: '{result['text']}'")

                            # 清空结果，准备下一次识别
                            self.final_result = ""
                            self.previous_result = ""  # 重置上一次的结果
                            self.current_text = ""     # 重置当前累积的文本
                            self.partial_result = ""   # 重置部分结果

                            return json.dumps(result)
                        except Exception as e:
                            self.logger.error(f"FinalResult 错误: {e}")
                            import traceback
                            self.logger.error(traceback.format_exc())
                            result = {"text": ""}
                            return json.dumps(result)

                    def Reset(self):
                        """
                        重置识别器状态

                        这个方法模拟了Vosk的Reset方法，重置所有状态变量，
                        并重新创建持久的流，准备下一次识别。

                        与Vosk不同，Sherpa-ONNX需要使用流来管理状态，
                        所以我们需要重新创建流，而不仅仅是重置状态变量。

                        Returns:
                            bool: 如果重置成功返回True，否则返回False
                        """
                        self.logger.debug("Reset 被调用")
                        # 清空结果
                        self.partial_result = ""
                        self.final_result = ""
                        self.previous_result = ""  # 重置上一次的结果
                        self.current_text = ""     # 重置当前累积的文本

                        # 重置累积文本和时间戳
                        self.accumulated_text = []
                        self.start_time = time.time()
                        self.timestamps = []

                        # 重新创建持久的流
                        try:
                            self.stream = self.model.create_stream()
                            self.logger.info("重新创建持久的流")
                        except Exception as e:
                            self.logger.error(f"重新创建持久的流错误: {e}")
                            import traceback
                            self.logger.error(traceback.format_exc())
                            self.stream = None
                            return False

                        return True

                    def get_accumulated_text_with_timestamps(self):
                        """
                        获取带时间戳的累积文本

                        Returns:
                            list: 包含(文本, 时间戳)元组的列表
                        """
                        return self.timestamps

                    def get_accumulated_text(self):
                        """
                        获取累积文本

                        Returns:
                            list: 文本列表
                        """
                        return self.accumulated_text

                    def get_formatted_transcript(self):
                        """
                        获取格式化的转录文本，包含时间戳

                        Returns:
                            str: 格式化的转录文本
                        """
                        self.logger.debug("get_formatted_transcript 被调用")

                        # 如果没有时间戳，但有最终结果，使用最终结果
                        if not self.timestamps and self.final_result:
                            self.logger.debug(f"没有时间戳，但有最终结果: {self.final_result}")
                            # 格式化文本：首字母大写，添加标点符号
                            formatted_text = self._format_text(self.final_result)
                            # 添加当前时间戳
                            current_time = time.time()
                            elapsed_time = current_time - self.start_time
                            minutes = int(elapsed_time // 60)
                            seconds = int(elapsed_time % 60)
                            timestamp = f"{minutes:02d}:{seconds:02d}"
                            return f"[{timestamp}] {formatted_text}"

                        # 如果没有时间戳，但有当前文本，使用当前文本
                        elif not self.timestamps and self.current_text:
                            self.logger.debug(f"没有时间戳，但有当前文本: {self.current_text}")
                            # 格式化文本：首字母大写，添加标点符号
                            formatted_text = self._format_text(self.current_text)
                            # 添加当前时间戳
                            current_time = time.time()
                            elapsed_time = current_time - self.start_time
                            minutes = int(elapsed_time // 60)
                            seconds = int(elapsed_time % 60)
                            timestamp = f"{minutes:02d}:{seconds:02d}"
                            return f"[{timestamp}] {formatted_text}"

                        # 如果没有时间戳，但有部分结果，使用部分结果
                        elif not self.timestamps and self.partial_result:
                            self.logger.debug(f"没有时间戳，但有部分结果: {self.partial_result}")
                            # 格式化文本：首字母大写，添加标点符号
                            formatted_text = self._format_text(self.partial_result)
                            # 添加当前时间戳
                            current_time = time.time()
                            elapsed_time = current_time - self.start_time
                            minutes = int(elapsed_time // 60)
                            seconds = int(elapsed_time % 60)
                            timestamp = f"{minutes:02d}:{seconds:02d}"
                            return f"[{timestamp}] {formatted_text}"

                        # 如果有时间戳，使用时间戳
                        elif self.timestamps:
                            self.logger.debug(f"有时间戳，数量: {len(self.timestamps)}")
                            formatted_text = []
                            for text, timestamp in self.timestamps:
                                formatted_text.append(f"[{timestamp}] {text}")

                            return "\n".join(formatted_text)

                        # 如果什么都没有，返回空字符串
                        else:
                            self.logger.debug("没有任何文本可用")
                            return ""

                # 返回包装后的识别器
                model_type_str = "int8量化" if is_int8 else "标准"
                sherpa_logger.info(f"创建 Sherpa-ONNX {model_type_str}识别器")

                try:
                    recognizer = SherpaRecognizer(self.current_model)
                    sherpa_logger.info(f"SherpaRecognizer 实例创建成功: {recognizer}")

                    # 设置引擎类型
                    recognizer.engine_type = self.model_type
                    sherpa_logger.info(f"设置 Sherpa-ONNX 识别器引擎类型: {self.model_type}")

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
                error_msg = f"不支持的模型类型: {self.model_type}"
                sherpa_logger.error(error_msg)
                print(error_msg)
                return None

        except Exception as e:
            error_msg = f"创建识别器失败: {e}"
            sherpa_logger.error(error_msg)
            print(error_msg)
            import traceback
            error_trace = traceback.format_exc()
            sherpa_logger.error(error_trace)
            print(error_trace)
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
                    sherpa_logger.error(f"创建 VoskASR 实例失败: {e}")
                    import traceback
                    sherpa_logger.error(traceback.format_exc())
                    return False

            elif engine_type.startswith("sherpa"):  # 支持 sherpa_int8、sherpa_std 和 sherpa_0626
                sherpa_logger.info(f"创建 SherpaOnnxASR 实例，路径: {model_config['path']}")
                # 检查模型路径是否存在
                if not os.path.exists(model_config["path"]):
                    sherpa_logger.error(f"Sherpa-ONNX 模型路径不存在: {model_config['path']}")
                    return False

                # 检查模型类型
                if engine_type == "sherpa_0626":
                    # 使用0626模型
                    model_type = "standard"
                    model_name = "0626"
                    sherpa_logger.info(f"Sherpa-ONNX 2023-06-26 模型配置: {model_config}")
                else:
                    # 使用现有模型
                    model_type = "int8" if engine_type == "sherpa_int8" else "std"
                    model_name = ""

                sherpa_logger.info(f"Sherpa-ONNX 模型类型: {model_type}, 模型名称: {model_name}")

                # 创建 SherpaOnnxASR 实例
                try:
                    sherpa_logger.info(f"创建 SherpaOnnxASR 实例，路径: {model_config['path']}, 类型: {model_type}, 名称: {model_name}")
                    self.current_engine = SherpaOnnxASR(model_config["path"], {"type": model_type, "name": model_name})
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
                    engine_type = "sherpa_0626"
                    sherpa_logger.debug("当前引擎是 SherpaOnnxASR (0626)")
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

        # 检查 sherpa_0626 引擎
        if "sherpa_0626" in self.models_config:
            model_config = self.models_config["sherpa_0626"]
            engines["sherpa_0626"] = bool(model_config and model_config.get("enabled", False))
        else:
            engines["sherpa_0626"] = False

        return engines
