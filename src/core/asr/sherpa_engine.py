import os
import numpy as np
from typing import Optional, Union, Dict, Any
import sherpa_onnx


class SherpaOnnxASR:
    """Sherpa-ONNX ASR 引擎实现"""
    def __init__(self, model_dir: str, model_config: Dict[str, Any] = None):
        """
        初始化 Sherpa-ONNX ASR 引擎

        Args:
            model_dir: 模型目录路径
            model_config: 模型配置，如果为None则使用默认配置
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

    def setup(self) -> bool:
        """
        初始化 Sherpa-ONNX ASR

        Returns:
            bool: 初始化是否成功
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
            sherpa_logger.info(f"初始化 Sherpa-ONNX ASR，模型目录: {self.model_dir}")
            sherpa_logger.info(f"是否使用 int8 量化模型: {self.is_int8}")

            # 确定模型文件名
            # 检查是否是0626模型
            is_0626 = self.model_config and self.model_config.get("name") == "0626" or "2023-06-26" in self.model_dir

            # 首先尝试检测目录中的实际文件
            encoder_files = []
            decoder_files = []
            joiner_files = []

            # 列出目录中的所有文件
            try:
                for file in os.listdir(self.model_dir):
                    if file.startswith("encoder") and file.endswith(".onnx"):
                        encoder_files.append(file)
                    elif file.startswith("decoder") and file.endswith(".onnx"):
                        decoder_files.append(file)
                    elif file.startswith("joiner") and file.endswith(".onnx"):
                        joiner_files.append(file)

                sherpa_logger.info(f"找到的encoder文件: {encoder_files}")
                sherpa_logger.info(f"找到的decoder文件: {decoder_files}")
                sherpa_logger.info(f"找到的joiner文件: {joiner_files}")
            except Exception as e:
                sherpa_logger.error(f"列出目录文件失败: {e}")

            if is_0626:
                # 使用0626模型的文件名
                # 优先使用非int8版本
                if encoder_files:
                    # 优先选择非int8版本
                    encoder_file = next((f for f in encoder_files if "chunk-16-left-128.onnx" in f and ".int8." not in f),
                                       next((f for f in encoder_files if "chunk-16-left-128" in f),
                                            "encoder-epoch-99-avg-1-chunk-16-left-128.onnx"))
                else:
                    encoder_file = "encoder-epoch-99-avg-1-chunk-16-left-128.onnx"

                if decoder_files:
                    decoder_file = next((f for f in decoder_files if "chunk-16-left-128.onnx" in f and ".int8." not in f),
                                       next((f for f in decoder_files if "chunk-16-left-128" in f),
                                            "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"))
                else:
                    decoder_file = "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"

                if joiner_files:
                    joiner_file = next((f for f in joiner_files if "chunk-16-left-128.onnx" in f and ".int8." not in f),
                                      next((f for f in joiner_files if "chunk-16-left-128" in f),
                                           "joiner-epoch-99-avg-1-chunk-16-left-128.onnx"))
                else:
                    joiner_file = "joiner-epoch-99-avg-1-chunk-16-left-128.onnx"

                sherpa_logger.info(f"使用0626模型文件名: encoder={encoder_file}, decoder={decoder_file}, joiner={joiner_file}")
            else:
                # 使用现有模型的文件名
                if self.is_int8:
                    # 优先使用int8版本
                    if encoder_files:
                        encoder_file = next((f for f in encoder_files if ".int8." in f), "encoder-epoch-99-avg-1.int8.onnx")
                    else:
                        encoder_file = "encoder-epoch-99-avg-1.int8.onnx"

                    if decoder_files:
                        decoder_file = next((f for f in decoder_files if ".int8." in f), "decoder-epoch-99-avg-1.int8.onnx")
                    else:
                        decoder_file = "decoder-epoch-99-avg-1.int8.onnx"

                    if joiner_files:
                        joiner_file = next((f for f in joiner_files if ".int8." in f), "joiner-epoch-99-avg-1.int8.onnx")
                    else:
                        joiner_file = "joiner-epoch-99-avg-1.int8.onnx"
                else:
                    # 优先使用非int8版本
                    if encoder_files:
                        encoder_file = next((f for f in encoder_files if ".int8." not in f), "encoder-epoch-99-avg-1.onnx")
                    else:
                        encoder_file = "encoder-epoch-99-avg-1.onnx"

                    if decoder_files:
                        decoder_file = next((f for f in decoder_files if ".int8." not in f), "decoder-epoch-99-avg-1.onnx")
                    else:
                        decoder_file = "decoder-epoch-99-avg-1.onnx"

                    if joiner_files:
                        joiner_file = next((f for f in joiner_files if ".int8." not in f), "joiner-epoch-99-avg-1.onnx")
                    else:
                        joiner_file = "joiner-epoch-99-avg-1.onnx"

                sherpa_logger.info(f"使用现有模型文件名: encoder={encoder_file}, decoder={decoder_file}, joiner={joiner_file}")

            sherpa_logger.info(f"使用模型文件: encoder={encoder_file}, decoder={decoder_file}, joiner={joiner_file}")

            # 检查模型文件
            required_files = {
                "encoder": encoder_file,
                "decoder": decoder_file,
                "joiner": joiner_file,
                "tokens": "tokens.txt"
            }

            # 验证所有必需文件
            for file_type, file_name in required_files.items():
                file_path = os.path.join(self.model_dir, file_name)
                if not os.path.exists(file_path):
                    error_msg = f"错误: 缺少{file_type}模型文件: {file_path}"
                    sherpa_logger.error(error_msg)
                    print(error_msg)
                    return False
                else:
                    sherpa_logger.info(f"找到{file_type}模型文件: {file_path}")

            # 配置识别器
            self.config = {
                "encoder": os.path.join(self.model_dir, encoder_file),
                "decoder": os.path.join(self.model_dir, decoder_file),
                "joiner": os.path.join(self.model_dir, joiner_file),
                "tokens": os.path.join(self.model_dir, "tokens.txt"),
                "num_threads": 4,  # 使用多线程加速
                "sample_rate": self.sample_rate,
                "feature_dim": 80,
                "decoding_method": "greedy_search",
                "debug": False,
                # 添加端点检测参数，参考TMSpeech项目
                "enable_endpoint": 1,  # 启用端点检测
                "rule1_min_trailing_silence": 2.4,  # 基本静音阈值(秒)
                "rule2_min_trailing_silence": 1.2,  # 长句中的静音阈值(秒)
                "rule3_min_utterance_length": 20    # 长句判定阈值(帧)
            }

            sherpa_logger.info(f"识别器配置: {self.config}")

            # 如果提供了配置，使用配置中的值覆盖默认值
            if self.model_config and "config" in self.model_config:
                user_config = self.model_config["config"]
                for key, value in user_config.items():
                    if key in self.config and key not in ["encoder", "decoder", "joiner", "tokens"]:
                        self.config[key] = value
                sherpa_logger.info(f"使用用户配置覆盖默认值: {user_config}")

            # 使用 OnlineRecognizer 类的 from_transducer 静态方法创建实例
            # 这是 sherpa-onnx 1.11.2 版本的正确 API
            try:
                sherpa_logger.info("创建 OnlineRecognizer 实例...")
                self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                    encoder=self.config["encoder"],
                    decoder=self.config["decoder"],
                    joiner=self.config["joiner"],
                    tokens=self.config["tokens"],
                    num_threads=self.config["num_threads"],
                    sample_rate=self.config["sample_rate"],
                    feature_dim=self.config["feature_dim"],
                    decoding_method=self.config["decoding_method"]
                )
                sherpa_logger.info("OnlineRecognizer 实例创建成功")
            except Exception as e:
                error_msg = f"使用 from_transducer 创建实例失败: {e}"
                sherpa_logger.error(error_msg)
                print(error_msg)
                import traceback
                sherpa_logger.error(traceback.format_exc())
                # 创建一个空的实例
                try:
                    self.recognizer = sherpa_onnx.OnlineRecognizer()
                    sherpa_logger.warning("创建了一个空的 OnlineRecognizer 实例")
                except Exception as e2:
                    error_msg = f"创建空的 OnlineRecognizer 实例失败: {e2}"
                    sherpa_logger.error(error_msg)
                    print(error_msg)
                    return False

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
                        error_msg = f"WAV 文件格式不正确: 通道数={channels}, 采样宽度={sample_width}, 采样率={sample_rate}"
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
                    sherpa_logger.debug(f"接受音频数据成功")
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

                # 获取部分结果
                try:
                    partial_result = self.recognizer.get_result(stream)
                    if partial_result:
                        sherpa_logger.info(f"部分结果: {partial_result}")
                        print(f"部分结果: {partial_result}")
                except Exception as e:
                    error_msg = f"获取部分结果失败: {e}"
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

    def __del__(self):
        """清理资源"""
        self.recognizer = None
        self.stream = None
