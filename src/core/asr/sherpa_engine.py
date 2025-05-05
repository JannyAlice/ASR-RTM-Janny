import os
import numpy as np
from typing import Optional, Union, Dict, Any
import sherpa_onnx

class SherpaOnnxASR:
    """Sherpa-ONNX ASR 引擎实现"""

    def _get_logger(self):
        """
        获取日志记录器

        Returns:
            Logger: 日志记录器实例
        """
        try:
            from src.utils.sherpa_logger import SherpaLogger
            return SherpaLogger().logger
        except ImportError:
            # 如果导入失败，创建一个标准的日志记录器
            import logging
            logger = logging.getLogger("sherpa_engine")
            if not logger.handlers:  # 避免重复添加处理器
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
                logger.addHandler(handler)
                logger.setLevel(logging.DEBUG)
            return logger

    def __init__(self, model_dir: str, model_config: Dict[str, Any] = None):
        """
        初始化 Sherpa-ONNX ASR 引擎

        Args:
            model_dir: 模型目录路径
            model_config: 模型配置，如果为None则使用默认配置

        Raises:
            ValueError: 模型配置无效时
        """
        self.model_dir = model_dir
        self.model_config = model_config
        self.recognizer = None
        self.stream = None
        self.config = None
        self.sample_rate = 16000
        self.is_int8 = True  # 默认使用int8量化模型

        # 如果提供了配置，检查是否使用int8模型
        if model_config and "type" in model_config:
            self.is_int8 = model_config["type"].lower() == "int8"

        # 初始化日志记录器
        self._logger = self._get_logger()

    def _validate_model_config(self) -> None:
        """
        验证模型配置的有效性

        Raises:
            ValueError: 配置无效时
        """
        if not self.model_config:
            self.model_config = {}

        # 设置默认配置
        default_config = {
            "type": "int8",
            "name": "0626",
            "sample_rate": 16000,
            "num_threads": 4,
            "enable_endpoint": 1,
            "rule1_min_trailing_silence": 3.0,
            "rule2_min_trailing_silence": 1.5,
            "rule3_min_utterance_length": 25
        }

        for key, value in default_config.items():
            if key not in self.model_config:
                self.model_config[key] = value

    def _detect_model_files(self) -> Dict[str, str]:
        """
        智能检测模型文件（改进版）

        Returns:
            Dict[str, str]: 检测到的模型文件映射

        Raises:
            RuntimeError: 未找到有效模型文件
        """
        sherpa_logger = self._logger

        # 结构化文件匹配规则（Sherpa-ONNX标准命名规范）
        file_patterns = {
            "encoder": {
                "required": ["encoder", "chunk-16-left-128"],
                "exclude": ["decoder", "joiner"],
                "extension": ".onnx"
            },
            "decoder": {
                "required": ["decoder", "chunk-16-left-128"],
                "exclude": ["encoder", "joiner"],
                "extension": ".onnx"
            },
            "joiner": {
                "required": ["joiner", "chunk-16-left-128"],
                "exclude": ["encoder", "decoder"],
                "extension": ".onnx"
            },
            "tokens": {
                "required": ["tokens"],
                "exclude": ["encoder", "decoder", "joiner"],
                "extension": ".txt"
            }
        }

        detected_files = {}

        # 遍历每个文件类型进行检测
        for file_type, patterns in file_patterns.items():
            matching_files = []

            # 遍历模型目录中的文件
            for file in os.listdir(self.model_dir):
                file_lower = file.lower()

                # 1. 检查文件扩展名（不区分大小写）
                if not file_lower.endswith(patterns["extension"].lower()):
                    continue

                # 2. 检查必须包含的关键词
                if not all(p in file_lower for p in patterns["required"]):
                    continue

                # 3. 排除冲突关键词
                if any(e in file_lower for e in patterns["exclude"]):
                    continue

                # 4. 只对ONNX文件检查int8配置一致性
                if patterns["extension"].lower() == ".onnx":
                    is_int8_file = ".int8." in file_lower
                    if self.is_int8 != is_int8_file:
                        continue

                matching_files.append(file)
                sherpa_logger.debug(f"找到候选文件: {file}")

            # 在循环外检查匹配结果
            if not matching_files:
                error_details = (
                    f"未找到{file_type}模型文件\n"
                    f"搜索路径: {self.model_dir}\n"
                    f"必须包含: {patterns['required']}\n"
                    f"排除包含: {patterns['exclude']}\n"
                    f"文件类型: {'int8' if self.is_int8 else '标准'}"
                )
                sherpa_logger.error(error_details)
                raise RuntimeError(error_details)

            # 选择第一个匹配的文件
            detected_files[file_type] = os.path.join(self.model_dir, matching_files[0])
            sherpa_logger.info(f"已选择{file_type}模型文件: {detected_files[file_type]}")

        return detected_files

    def setup(self) -> bool:
        """初始化 Sherpa-ONNX ASR 引擎

        Returns:
            bool: 初始化是否成功
        Raises:
            RuntimeError: 模型初始化失败时
        """
        """
        初始化 Sherpa-ONNX ASR

        Returns:
            bool: 初始化是否成功
        """
        # 导入 Sherpa-ONNX 日志工具
        try:
            from src.utils.sherpa_logger import SherpaLogger
            sherpa_logger = SherpaLogger().logger
        except ImportError:
            # 如果导入失败，创建一个简单的日志记录器
            import logging
            sherpa_logger = logging.getLogger("sherpa_engine")
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            sherpa_logger.addHandler(handler)
            sherpa_logger.setLevel(logging.DEBUG)

        try:
            # 验证模型配置
            self._validate_model_config()

            # 检测模型文件
            model_files = self._detect_model_files()

            sherpa_logger.info(f"初始化 Sherpa-ONNX ASR，模型目录: {self.model_dir}")
            sherpa_logger.info(f"是否使用 int8 量化模型: {self.is_int8}")

            # 配置识别器
            self.config = {
                "encoder": model_files["encoder"],
                "decoder": model_files["decoder"],
                "joiner": model_files["joiner"],
                "tokens": model_files["tokens"],
                "num_threads": self.model_config.get("num_threads", 4),
                "sample_rate": self.model_config.get("sample_rate", 16000),
                "feature_dim": 80,
                "decoding_method": "greedy_search",
                "debug": False,
                # 添加端点检测参数，参考TMSpeech项目
                "enable_endpoint": self.model_config.get("enable_endpoint", 1),
                "rule1_min_trailing_silence": self.model_config.get("rule1_min_trailing_silence", 3.0),
                "rule2_min_trailing_silence": self.model_config.get("rule2_min_trailing_silence", 1.5),
                "rule3_min_utterance_length": self.model_config.get("rule3_min_utterance_length", 25)
            }

            sherpa_logger.info(f"识别器配置: {self.config}")

            # 如果提供了配置，使用配置中的值覆盖默认值
            if self.model_config and "config" in self.model_config:
                user_config = self.model_config["config"]
                for key, value in user_config.items():
                    if key in self.config and key not in ["encoder", "decoder", "joiner", "tokens"]:
                        self.config[key] = value
                sherpa_logger.info(f"使用用户配置覆盖默认值: {user_config}")

            # 从 config_manager 获取配置
            try:
                from src.utils.config_manager import config_manager
                asr_config = config_manager.get_config("asr")
                if asr_config:
                    for key, value in asr_config.items():
                        if key in self.config:
                            self.config[key] = value
                    sherpa_logger.info(f"从 config_manager 获取 ASR 配置: {asr_config}")
            except ImportError:
                sherpa_logger.warning("无法导入 config_manager，使用默认配置")

            # 使用 OnlineRecognizer 类的 from_transducer 静态方法创建实例
            # 这是 sherpa-onnx 版本的 API
            try:
                sherpa_logger.info("创建 OnlineRecognizer 实例...")

                # 确保所有路径都是绝对路径
                for file_type in ["encoder", "decoder", "joiner", "tokens"]:
                    if not os.path.isabs(self.config[file_type]):
                        self.config[file_type] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", self.config[file_type])

                    file_path = self.config[file_type]
                    sherpa_logger.info(f"检查模型文件: {file_type} = {file_path}")

                    if not os.path.exists(file_path):
                        error_msg = f"模型文件不存在: {file_path}"
                        sherpa_logger.error(error_msg)
                        raise FileNotFoundError(error_msg)

                    if not os.access(file_path, os.R_OK):
                        error_msg = f"无法读取模型文件: {file_path}"
                        sherpa_logger.error(error_msg)
                        raise PermissionError(error_msg)

                    # 记录文件大小和修改时间
                    file_stat = os.stat(file_path)
                    sherpa_logger.info(f"{file_type}文件大小: {file_stat.st_size} 字节, 修改时间: {file_stat.st_mtime}")

                # 尝试获取模型文件的详细信息
                def log_model_file_details(file_path):
                    try:
                        import onnx
                        model = onnx.load(file_path)
                        sherpa_logger.info(f"模型文件 {file_path} 详细信息:")
                        sherpa_logger.info(f"图名称: {model.graph.name}")
                        sherpa_logger.info(f"节点数: {len(model.graph.node)}")
                        sherpa_logger.info(f"输入数: {len(model.graph.input)}")
                        sherpa_logger.info(f"输出数: {len(model.graph.output)}")
                    except Exception as e:
                        sherpa_logger.warning(f"无法获取 {file_path} 的模型详细信息: {e}")

                # 记录模型文件详细信息
                log_model_file_details(self.config["encoder"])
                log_model_file_details(self.config["decoder"])
                log_model_file_details(self.config["joiner"])

                # 尝试创建 OnlineRecognizer
                sherpa_logger.info("配置参数:")
                for key, value in self.config.items():
                    sherpa_logger.info(f"{key}: {value}")

                try:
                    # 记录详细的参数信息
                    sherpa_logger.info("OnlineRecognizer.from_transducer 参数:")
                    sherpa_logger.info(f"  encoder: {self.config['encoder']}")
                    sherpa_logger.info(f"  decoder: {self.config['decoder']}")
                    sherpa_logger.info(f"  joiner: {self.config['joiner']}")
                    sherpa_logger.info(f"  tokens: {self.config['tokens']}")
                    sherpa_logger.info(f"  num_threads: {self.config.get('num_threads', 4)}")
                    sherpa_logger.info(f"  sample_rate: {self.config.get('sample_rate', 16000)}")
                    sherpa_logger.info(f"  feature_dim: {self.config.get('feature_dim', 80)}")
                    sherpa_logger.info(f"  decoding_method: {self.config.get('decoding_method', 'greedy_search')}")
                    sherpa_logger.info(f"  enable_endpoint_detection: {bool(self.config.get('enable_endpoint', 1))}")
                    sherpa_logger.info(f"  rule1_min_trailing_silence: {float(self.config.get('rule1_min_trailing_silence', 3.0))}")
                    sherpa_logger.info(f"  rule2_min_trailing_silence: {float(self.config.get('rule2_min_trailing_silence', 1.5))}")
                    sherpa_logger.info(f"  rule3_min_utterance_length: {float(self.config.get('rule3_min_utterance_length', 25))}")

                    # 检查sherpa_onnx版本
                    sherpa_logger.info(f"sherpa_onnx版本: {sherpa_onnx.__version__ if hasattr(sherpa_onnx, '__version__') else '未知'}")

                    # 创建OnlineRecognizer实例
                    self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                        encoder=self.config["encoder"],
                        decoder=self.config["decoder"],
                        joiner=self.config["joiner"],
                        tokens=self.config["tokens"],
                        num_threads=self.config.get("num_threads", 4),
                        sample_rate=self.config.get("sample_rate", 16000),
                        feature_dim=self.config.get("feature_dim", 80),
                        decoding_method=self.config.get("decoding_method", "greedy_search"),
                        # 添加端点检测参数
                        enable_endpoint_detection=bool(self.config.get("enable_endpoint", 1)),
                        rule1_min_trailing_silence=float(self.config.get("rule1_min_trailing_silence", 3.0)),
                        rule2_min_trailing_silence=float(self.config.get("rule2_min_trailing_silence", 1.5)),
                        rule3_min_utterance_length=float(self.config.get("rule3_min_utterance_length", 25))
                    )
                    sherpa_logger.info("OnlineRecognizer 实例创建成功")

                    # 测试创建流
                    try:
                        test_stream = self.recognizer.create_stream()
                        sherpa_logger.info("测试创建流成功")
                        # 不需要保存测试流
                        del test_stream
                    except Exception as stream_error:
                        sherpa_logger.warning(f"测试创建流失败: {stream_error}")
                        # 这不是致命错误，继续执行

                except Exception as e:
                    error_msg = f"OnlineRecognizer 创建失败: {str(e)}"
                    sherpa_logger.error(error_msg)

                    # 记录更详细的错误信息
                    import traceback
                    error_trace = traceback.format_exc()
                    sherpa_logger.error(f"详细错误信息:\n{error_trace}")

                    # 检查是否是已知的常见错误
                    error_str = str(e).lower()
                    if "no such file or directory" in error_str:
                        sherpa_logger.error("错误原因: 模型文件不存在")
                    elif "permission denied" in error_str:
                        sherpa_logger.error("错误原因: 无权限访问模型文件")
                    elif "invalid argument" in error_str:
                        sherpa_logger.error("错误原因: 参数无效，可能是模型文件格式不正确")
                    elif "out of memory" in error_str:
                        sherpa_logger.error("错误原因: 内存不足")

                    raise RuntimeError(error_msg) from e
            except Exception as e:
                error_msg = f"创建 OnlineRecognizer 实例失败: {e}"
                sherpa_logger.error(error_msg)
                # 打印详细的异常堆栈信息
                import traceback
                sherpa_logger.error(traceback.format_exc())
                raise RuntimeError(error_msg)

            model_type = "int8量化" if self.is_int8 else "标准"
            success_msg = f"Sherpa-ONNX ASR ({model_type}模型) 初始化成功"
            sherpa_logger.info(success_msg)
            print(success_msg)
            return True

        except Exception as e:
            error_msg = f"Sherpa-ONNX ASR 初始化失败: {e}"
            sherpa_logger.error(error_msg)
            print(error_msg)
            import traceback
            error_trace = traceback.format_exc()
            sherpa_logger.error(error_trace)
            print(error_trace)
            return False

    def transcribe(self, audio_data: Union[bytes, np.ndarray]) -> Optional[str]:
        """
        转录音频数据

        Args:
            audio_data: 音频数据，可以是字节或numpy数组

        Returns:
            str: 转录文本，如果失败则返回None
        """
        try:
            if not self.recognizer:
                return None

            # 每次转录都创建一个新的流，避免状态累积导致的问题
            try:
                stream = self.recognizer.create_stream()
            except Exception as e:
                print(f"创建流错误: {e}")
                return None

            # 确保音频数据是numpy数组
            if isinstance(audio_data, bytes):
                # 将字节转换为16位整数数组
                import array
                audio_array = array.array('h', audio_data)
                audio_data = np.array(audio_array, dtype=np.float32) / 32768.0

            # 确保音频数据是单声道
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # 处理音频数据
            try:
                # 直接处理整个音频数据（完全按照官方测试文件的方法）
                stream.accept_waveform(self.sample_rate, audio_data)

                # 添加尾部填充（这是关键步骤，来自官方测试文件）
                tail_paddings = np.zeros(int(0.2 * self.sample_rate), dtype=np.float32)
                stream.accept_waveform(self.sample_rate, tail_paddings)

                # 标记输入结束
                stream.input_finished()

                # 解码
                while self.recognizer.is_ready(stream):
                    self.recognizer.decode_stream(stream)
            except Exception as e:
                print(f"处理音频数据错误: {e}")
                import traceback
                print(traceback.format_exc())
                return None

            # 获取结果
            try:
                # 使用 get_result 获取结果
                result = self.recognizer.get_result(stream)
                if result:
                    # 使用正则表达式处理结果，确保每个句子以句号结尾
                    import re
                    result = re.sub(r'(?<=[a-zA-Z0-9])(?=[A-Z])', '. ', result)
                    result = re.sub(r'\s+$', '', result)  # 去除末尾空格
                    if not result.endswith('.'):
                        result += '.'  # 确保结果以句号结尾
                    print(f"DEBUG: 转录结果: {result}")
                return result if result else None
            except Exception as e:
                print(f"获取结果错误: {e}")
                return None

        except Exception as e:
            print(f"Sherpa-ONNX 转录错误: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def reset(self) -> None:
        """重置识别器状态"""
        # 不需要做任何事情，因为我们在每次转录时都会创建新的流
        pass

    def get_final_result(self) -> Optional[str]:
        """
        获取最终结果

        Returns:
            str: 最终识别结果文本，如果失败则返回None
        """
        try:
            if not self.recognizer:
                return None

            # 创建一个新的流
            try:
                stream = self.recognizer.create_stream()
            except Exception as e:
                print(f"创建流错误: {e}")
                return None

            # 标记输入结束
            try:
                stream.input_finished()

                # 解码
                while self.recognizer.is_ready(stream):
                    self.recognizer.decode_stream(stream)
            except Exception as e:
                print(f"解码剩余音频错误: {e}")
                return None

            # 获取最终结果
            try:
                # 尝试获取完整结果
                try:
                    # 使用 get_result_all 获取更详细的结果
                    result_all = self.recognizer.get_result_all(stream)
                    if hasattr(result_all, 'text') and result_all.text:
                        print(f"DEBUG: 最终完整结果: {result_all.text}")
                        return result_all.text
                except Exception as e:
                    print(f"获取完整最终结果错误: {e}")
                    # 继续尝试使用 get_result

                # 使用 get_result 获取结果
                final_result = self.recognizer.get_result(stream)
                if final_result:
                    print(f"DEBUG: 最终基本结果: {final_result}")
                return final_result if final_result else None
            except Exception as e:
                print(f"获取最终结果错误: {e}")
                return None

        except Exception as e:
            print(f"获取 Sherpa-ONNX 最终结果错误: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def transcribe_file(self, file_path: str) -> Optional[str]:
        """
        转录音频文件

        Args:
            file_path: 音频文件路径

        Returns:
            str: 转录文本，如果失败则返回None
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
            sherpa_logger.info(f"开始转录文件: {file_path}")

            if not self.recognizer:
                error_msg = "未初始化识别器，无法转录文件"
                sherpa_logger.error(error_msg)
                print(error_msg)
                return None

            # 使用 ffmpeg 将文件转换为 16kHz 单声道 PCM 格式
            import subprocess
            import tempfile
            import wave

            # 创建临时 WAV 文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name

            sherpa_logger.info(f"创建临时 WAV 文件: {temp_wav_path}")

            # 使用 ffmpeg 转换
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', file_path,
                '-ar', '16000',  # 采样率 16kHz
                '-ac', '1',      # 单声道
                '-f', 'wav',     # WAV 格式
                '-y',            # 覆盖已有文件
                temp_wav_path
            ]

            # 确保 self.recognizer 已经正确初始化
            if not self.recognizer:
                error_msg = "未初始化识别器，无法转录文件"
                sherpa_logger.error(error_msg)
                print(error_msg)
                return None

            cmd_str = ' '.join(ffmpeg_cmd)
            sherpa_logger.info(f"执行命令: {cmd_str}")
            print(f"执行命令: {cmd_str}")

            try:
                subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                sherpa_logger.info("ffmpeg 命令执行成功")
            except subprocess.CalledProcessError as e:
                error_msg = f"ffmpeg 命令执行失败: {e}"
                sherpa_logger.error(error_msg)
                print(error_msg)
                return None

            # 读取 WAV 文件
            try:
                with wave.open(temp_wav_path, 'rb') as wav_file:
                    # 检查格式
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    sample_rate = wav_file.getframerate()

                    sherpa_logger.info(f"WAV 文件格式: 通道数={channels}, 采样宽度={sample_width}, 采样率={sample_rate}")

                    if channels != 1 or sample_width != 2 or sample_rate != 16000:
                        error_msg = (f"WAV 文件格式不正确: 通道数={channels}, "
                                     f"采样宽度={sample_width}, 采样率={sample_rate}")
                        sherpa_logger.error(error_msg)
                        print(error_msg)
                        return None

                    # 读取所有帧
                    frames = wav_file.readframes(wav_file.getnframes())
                    sherpa_logger.info(f"读取 WAV 文件帧数: {wav_file.getnframes()}")

                    # 转换为 numpy 数组
                    audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    sherpa_logger.info(f"音频数据长度: {len(audio_data)} 样本，最大值: {np.max(np.abs(audio_data))}")
            except Exception as e:
                error_msg = f"读取 WAV 文件失败: {e}"
                sherpa_logger.error(error_msg)
                print(error_msg)
                import traceback
                sherpa_logger.error(traceback.format_exc())
                return None

            # 删除临时文件
            try:
                os.unlink(temp_wav_path)
                sherpa_logger.info(f"删除临时 WAV 文件: {temp_wav_path}")
            except Exception as e:
                sherpa_logger.warning(f"删除临时 WAV 文件失败: {e}")
                pass

            # 创建流
            try:
                sherpa_logger.info("创建流...")
                stream = self.recognizer.create_stream()
                sherpa_logger.info("流创建成功")
            except Exception as e:
                error_msg = f"创建流失败: {e}"
                sherpa_logger.error(error_msg)
                print(error_msg)
                import traceback
                sherpa_logger.error(traceback.format_exc())
                return None

            # 处理音频数据
            # 分块处理，每次处理 10 秒的数据
            chunk_size = 16000 * 10  # 10 秒的数据
            sherpa_logger.info(f"分块处理音频数据，每块 {chunk_size} 样本 (10 秒)")

            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size]
                sherpa_logger.debug(f"处理块 {i//chunk_size + 1}/{(len(audio_data) + chunk_size - 1)//chunk_size}，长度: {len(chunk)} 样本")

                # 处理这个块
                try:
                    stream.accept_waveform(16000, chunk)
                    sherpa_logger.debug("接受音频数据成功")

                    # 检查端点
                    if self.config["enable_endpoint"]:
                        # 移除重复的端点检测逻辑
                        pass
                except Exception as e:
                    error_msg = f"接受音频数据失败: {e}"
                    sherpa_logger.error(error_msg)
                    print(error_msg)
                    continue

                # 解码
                try:
                    decode_count = 0
                    while self.recognizer.is_ready(stream):
                        self.recognizer.decode_stream(stream)
                        decode_count += 1
                    sherpa_logger.debug(f"解码完成，解码次数: {decode_count}")
                except Exception as e:
                    error_msg = f"解码失败: {e}"
                    sherpa_logger.error(error_msg)
                    print(error_msg)
                    continue

            # 添加尾部填充
            try:
                sherpa_logger.info("添加尾部填充...")
                tail_paddings = np.zeros(int(0.2 * 16000), dtype=np.float32)
                stream.accept_waveform(16000, tail_paddings)
                sherpa_logger.info("尾部填充添加成功")
            except Exception as e:
                error_msg = f"添加尾部填充失败: {e}"
                sherpa_logger.error(error_msg)
                print(error_msg)

            # 标记输入结束
            try:
                sherpa_logger.info("标记输入结束...")
                stream.input_finished()
                sherpa_logger.info("输入结束标记成功")
            except Exception as e:
                error_msg = f"标记输入结束失败: {e}"
                sherpa_logger.error(error_msg)
                print(error_msg)

            # 解码
            try:
                sherpa_logger.info("最终解码...")

                # 检查端点
                if self.config["enable_endpoint"]:
                    # 移除重复的端点检测逻辑
                    pass

                decode_count = 0
                while self.recognizer.is_ready(stream):
                    self.recognizer.decode_stream(stream)
                    decode_count += 1
                sherpa_logger.info(f"最终解码完成，解码次数: {decode_count}")
            except Exception as e:
                error_msg = f"最终解码失败: {e}"
                sherpa_logger.error(error_msg)
                print(error_msg)

            # 获取最终结果
            try:
                sherpa_logger.info("获取最终结果...")
                result = self.recognizer.get_result(stream)
                sherpa_logger.info(f"原始最终结果: {result}")
            except Exception as e:
                error_msg = f"获取最终结果失败: {e}"
                sherpa_logger.error(error_msg)
                print(error_msg)
                return None

            # 过滤掉非英文字符
            if result:
                try:
                    import re
                    # 只保留英文字母、数字、标点符号和空格
                    filtered_result = re.sub(r'[^\x00-\x7F]+', '', result)
                    sherpa_logger.info(f"过滤后的最终结果: {filtered_result}")
                    return filtered_result
                except Exception as e:
                    error_msg = f"过滤结果失败: {e}"
                    sherpa_logger.error(error_msg)
                    print(error_msg)
                    return result  # 返回未过滤的结果
            else:
                sherpa_logger.warning("没有最终结果")
                return None

        except Exception as e:
            error_msg = f"Sherpa-ONNX 转录文件错误: {e}"
            try:
                sherpa_logger.error(error_msg)
            except:
                pass
            print(error_msg)
            import traceback
            error_trace = traceback.format_exc()
            try:
                sherpa_logger.error(error_trace)
            except:
                pass
            print(error_trace)
            return None

    def AcceptWaveform(self, audio_data: np.ndarray) -> bool:
        """
        接受音频数据并进行处理（兼容Vosk API）

        Args:
            audio_data: 音频数据，numpy数组

        Returns:
            bool: 是否有完整的识别结果
        """
        try:
            # 获取日志记录器
            try:
                from src.utils.sherpa_logger import sherpa_logger
            except ImportError:
                class DummyLogger:
                    def debug(self, msg): print(f"DEBUG: {msg}")
                    def info(self, msg): print(f"INFO: {msg}")
                    def warning(self, msg): print(f"WARNING: {msg}")
                    def error(self, msg): print(f"ERROR: {msg}")
                sherpa_logger = DummyLogger()

            # 检查识别器是否已初始化
            if not self.recognizer:
                sherpa_logger.error("识别器未初始化")
                return False

            # 创建新的流
            if not hasattr(self, 'current_stream') or self.current_stream is None:
                try:
                    self.current_stream = self.recognizer.create_stream()
                    sherpa_logger.debug("创建新的流")
                except Exception as e:
                    sherpa_logger.error(f"创建流错误: {e}")
                    return False

            # 确保音频数据是numpy数组
            if isinstance(audio_data, bytes):
                # 将字节转换为16位整数数组
                import array
                audio_array = array.array('h', audio_data)
                audio_data = np.array(audio_array, dtype=np.float32) / 32768.0
                sherpa_logger.debug(f"将字节数据转换为numpy数组，长度: {len(audio_data)}")

            # 确保音频数据是单声道
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
                sherpa_logger.debug(f"将多声道数据转换为单声道，形状: {audio_data.shape}")

            # 处理音频数据
            try:
                self.current_stream.accept_waveform(self.sample_rate, audio_data)
                sherpa_logger.debug(f"接受音频数据，长度: {len(audio_data)}")
            except Exception as e:
                sherpa_logger.error(f"接受音频数据错误: {e}")
                import traceback
                sherpa_logger.error(traceback.format_exc())
                return False

            # 检查是否有完整的识别结果
            # 这里我们使用一个简单的启发式方法：如果当前流中有文本，则认为有完整的结果
            try:
                # 解码
                while self.recognizer.is_ready(self.current_stream):
                    self.recognizer.decode_stream(self.current_stream)

                # 获取当前结果
                result = self.recognizer.get_result(self.current_stream)
                has_result = bool(result and result.strip())
                sherpa_logger.debug(f"当前结果: {result}, 是否有结果: {has_result}")

                return has_result
            except Exception as e:
                sherpa_logger.error(f"检查结果错误: {e}")
                import traceback
                sherpa_logger.error(traceback.format_exc())
                return False

        except Exception as e:
            print(f"AcceptWaveform 错误: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    def Result(self) -> str:
        """
        获取当前识别结果（兼容Vosk API）

        Returns:
            str: 当前识别结果
        """
        try:
            # 检查识别器和流是否已初始化
            if not self.recognizer or not hasattr(self, 'current_stream') or self.current_stream is None:
                return ""

            # 解码
            while self.recognizer.is_ready(self.current_stream):
                self.recognizer.decode_stream(self.current_stream)

            # 获取结果
            result = self.recognizer.get_result(self.current_stream)

            # 重置流
            self.current_stream = self.recognizer.create_stream()

            # 处理结果
            if result:
                # 使用正则表达式处理结果，确保每个句子以句号结尾
                import re
                result = re.sub(r'(?<=[a-zA-Z0-9])(?=[A-Z])', '. ', result)
                result = re.sub(r'\s+$', '', result)  # 去除末尾空格
                if not result.endswith('.'):
                    result += '.'  # 确保结果以句号结尾

            return result if result else ""
        except Exception as e:
            print(f"Result 错误: {e}")
            import traceback
            print(traceback.format_exc())
            return ""

    def PartialResult(self) -> str:
        """
        获取部分识别结果（兼容Vosk API）

        Returns:
            str: 部分识别结果
        """
        try:
            # 检查识别器和流是否已初始化
            if not self.recognizer or not hasattr(self, 'current_stream') or self.current_stream is None:
                return ""

            # 解码
            while self.recognizer.is_ready(self.current_stream):
                self.recognizer.decode_stream(self.current_stream)

            # 获取部分结果
            result = self.recognizer.get_result(self.current_stream)

            return result if result else ""
        except Exception as e:
            print(f"PartialResult 错误: {e}")
            import traceback
            print(traceback.format_exc())
            return ""

    def __del__(self):
        """清理资源"""
        self.recognizer = None
        self.stream = None
        if hasattr(self, 'current_stream'):
            self.current_stream = None

    def on_sentence_done(self, text: str) -> None:
        """
        处理句子结束事件

        Args:
            text: 识别到的句子文本
        """
        try:
            from src.utils.sherpa_logger import sherpa_logger
        except ImportError:
            class DummyLogger:
                def debug(self, msg): print(f"DEBUG: {msg}")
                def info(self, msg): print(f"INFO: {msg}")
                def warning(self, msg): print(f"WARNING: {msg}")
                def error(self, msg): print(f"ERROR: {msg}")
            sherpa_logger = DummyLogger()

        sherpa_logger.info(f"句子结束: {text}")
        # 这里可以添加更多的处理逻辑，例如将结果发送到UI或其他模块
        print(f"句子结束: {text}")
