"""
Sherpa-ONNX 插件模块
提供基于sherpa-onnx系列模型的语音识别功能

本模块整合了所有sherpa-onnx系列模型的功能，包括：
- sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20（中英双语模型）
- sherpa-onnx-streaming-zipformer-en-2023-06-26（英语专用模型）

每种模型都支持标准版和int8量化版本，共四种变体：
1. sherpa_onnx_std: 2023-02-20标准版（中英双语）
2. sherpa_onnx_int8: 2023-02-20量化版（中英双语）
3. sherpa_0626_std: 2023-06-26标准版（英语专用）
4. sherpa_0626_int8: 2023-06-26量化版（英语专用）

本插件支持以下功能：
- 在线实时语音识别
- 音频/视频文件转录
- 自动格式化转录文本（首字母大写，添加标点符号）
- 支持WAV和非WAV格式音频（需要安装soundcard模块）

作者: RealtimeTrans Team
版本: 1.0.0
日期: 2023-07-15
"""
import os
import logging
import traceback
import tempfile
import numpy as np
import subprocess
from typing import Dict, Any, Optional, Union, List
import wave

# 导入基础插件类
from src.core.plugins.base.plugin_base import PluginBase

# 设置日志记录器
logger = logging.getLogger(__name__)

# 尝试导入sherpa_onnx
try:
    import sherpa_onnx
    HAS_SHERPA_ONNX = True
except ImportError:
    sherpa_onnx = None  # 确保变量存在，即使导入失败
    HAS_SHERPA_ONNX = False
    logger.warning(
        "未安装 sherpa_onnx 模块，Sherpa-ONNX 功能将不可用。\n"
        "要使用 Sherpa-ONNX 功能，请运行: pip install sherpa_onnx"
    )

# 尝试导入soundcard，如果失败则降级到只支持wav格式
# 注意：虽然这里导入了soundcard，但实际上并没有直接使用它
# 我们只是检查它是否可用，以决定是否支持非WAV格式的音频
try:
    import soundcard  # noqa: F401 - 导入但不直接使用
    HAS_SOUNDCARD = True
except ImportError:
    soundcard = None  # 确保变量存在，即使导入失败
    HAS_SOUNDCARD = False
    logger.warning(
        "未安装 soundcard 模块，只能处理WAV格式音频。\n"
        "要处理其他格式音频，请运行: pip install soundcard"
    )

class SherpaOnnxPlugin(PluginBase):
    """Sherpa-ONNX 插件类，支持所有sherpa-onnx系列模型

    这个类实现了PluginBase接口，提供了对所有sherpa-onnx系列模型的支持。
    它可以处理以下模型：
    - sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20（标准版和int8量化版）
    - sherpa-onnx-streaming-zipformer-en-2023-06-26（标准版和int8量化版）

    主要功能：
    1. 在线实时语音识别（process_audio方法）
    2. 音频/视频文件转录（transcribe_file方法）
    3. 自动格式化转录文本（get_formatted_transcript方法）

    使用方法：
    1. 创建插件实例：plugin = SherpaOnnxPlugin(config)
    2. 初始化插件：plugin.initialize()
    3. 处理音频数据：result = plugin.process_audio(audio_data)
    4. 转录文件：text = plugin.transcribe_file(file_path)
    5. 清理资源：plugin.cleanup()

    注意：
    - 使用前需要安装sherpa_onnx模块
    - 处理非WAV格式音频需要安装soundcard模块
    - 模型文件需要放在正确的位置
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化Sherpa-ONNX插件

        创建一个新的Sherpa-ONNX插件实例，并设置初始状态。
        注意：初始化不会加载模型，需要调用initialize()方法加载模型。

        Args:
            config: 插件配置字典，包含以下键：
                - path: 模型路径，必须指向有效的sherpa-onnx模型目录
                - type: 模型类型，'standard'或'int8'
                - num_threads: 使用的线程数，默认为4
                - sample_rate: 音频采样率，默认为16000
                - feature_dim: 特征维度，默认为80
                - decoding_method: 解码方法，默认为'greedy_search'
                - enable_endpoint_detection: 是否启用端点检测，默认为True
                - rule1_min_trailing_silence: 端点检测规则1的最小尾部静音，默认为2.4
                - rule2_min_trailing_silence: 端点检测规则2的最小尾部静音，默认为1.2
                - rule3_min_utterance_length: 端点检测规则3的最小语音长度，默认为20.0

        示例:
            ```python
            config = {
                'path': 'models/asr/sherpa-onnx-streaming-zipformer-en-2023-06-26',
                'type': 'standard',
                'num_threads': 4
            }
            plugin = SherpaOnnxPlugin(config)
            ```
        """
        PluginBase.__init__(self)
        # 保存配置
        self._config = config
        self.config = config  # 为了兼容ASRPluginBase
        self.model = None
        self.recognizer = None
        self.stream = None
        self.model_dir = None
        self.is_int8 = False
        self.is_0626 = False  # 是否是2023-06-26模型
        self.engine_type = "sherpa_onnx_std"  # 默认使用标准模型
        self.temp_files = []  # 用于存储临时文件路径
        self._model_version = "2023-02-20"  # 默认使用2023-02-20模型
        self.logger = logging.getLogger(self.__class__.__name__)  # 为了兼容ASRPluginBase
        self._is_initialized = False  # 为了兼容ASRPluginBase
        self._supported_models = [
            "sherpa_onnx_std",
            "sherpa_onnx_int8",
            "sherpa_0626_std",
            "sherpa_0626_int8"
        ]  # 为了兼容ASRPluginBase

    def get_id(self) -> str:
        """获取插件ID"""
        if self.is_0626:
            return "sherpa_0626_std" if not self.is_int8 else "sherpa_0626_int8"
        else:
            return "sherpa_onnx_std" if not self.is_int8 else "sherpa_onnx_int8"

    def get_name(self) -> str:
        """获取插件名称"""
        model_type = "Int8" if self.is_int8 else "Standard"
        if self.is_0626:
            return f"Sherpa-ONNX 2023-06-26 {model_type} Model"
        else:
            return f"Sherpa-ONNX 2023-02-20 {model_type} Model"

    def get_version(self) -> str:
        """获取插件版本"""
        return "1.0.0"

    def get_description(self) -> str:
        """获取插件描述"""
        if self.is_0626:
            return f"Sherpa-ONNX 2023-06-26 英语语音识别模型（{'量化' if self.is_int8 else '标准'}版）"
        else:
            return f"Sherpa-ONNX 2023-02-20 中英双语语音识别模型（{'量化' if self.is_int8 else '标准'}版）"

    def get_author(self) -> str:
        """获取插件作者"""
        return "RealtimeTrans Team"

    def setup(self) -> bool:
        """设置插件并加载模型

        此方法执行以下操作：
        1. 检查sherpa_onnx模块是否已安装
        2. 从配置中获取模型路径和类型
        3. 检查模型路径是否存在
        4. 确定模型版本（2023-02-20或2023-06-26）
        5. 确定模型文件名（根据模型版本和类型）
        6. 检查模型文件是否存在
        7. 创建OnlineRecognizer实例

        Returns:
            bool: 设置是否成功。如果成功，返回True；否则返回False。

        注意：
            - 如果sherpa_onnx模块未安装，将返回False
            - 如果模型路径不存在，将返回False
            - 如果模型文件不存在，将返回False
            - 如果创建OnlineRecognizer实例失败，将返回False
        """
        try:
            # 检查sherpa_onnx是否已安装
            if not HAS_SHERPA_ONNX:
                logger.error("未安装 sherpa_onnx 模块，无法使用 Sherpa-ONNX 功能")
                return False

            # 获取配置
            model_path = self.config.get('path', '')
            model_type = self.config.get('type', 'standard').lower()
            self.is_int8 = model_type == 'int8'

            # 检查模型路径
            if not model_path or not os.path.exists(model_path):
                logger.error(f"模型路径不存在: {model_path}")
                return False

            self.model_dir = model_path

            # 检查是否是2023-06-26模型
            if "0626" in model_path or "2023-06-26" in model_path:
                self.is_0626 = True
                self._model_version = "2023-06-26"
                self.engine_type = "sherpa_0626_std" if not self.is_int8 else "sherpa_0626_int8"
            else:
                self.is_0626 = False
                self._model_version = "2023-02-20"
                self.engine_type = "sherpa_onnx_std" if not self.is_int8 else "sherpa_onnx_int8"

            # 确定模型文件名
            if self.is_0626:
                # 2023-06-26模型文件名
                if self.is_int8:
                    encoder_file = "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
                    decoder_file = "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
                    joiner_file = "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
                else:
                    encoder_file = "encoder-epoch-99-avg-1-chunk-16-left-128.onnx"
                    decoder_file = "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"
                    joiner_file = "joiner-epoch-99-avg-1-chunk-16-left-128.onnx"
            else:
                # 2023-02-20模型文件名
                if self.is_int8:
                    encoder_file = "encoder-epoch-99-avg-1.int8.onnx"
                    decoder_file = "decoder-epoch-99-avg-1.int8.onnx"
                    joiner_file = "joiner-epoch-99-avg-1.int8.onnx"
                else:
                    encoder_file = "encoder-epoch-99-avg-1.onnx"
                    decoder_file = "decoder-epoch-99-avg-1.onnx"
                    joiner_file = "joiner-epoch-99-avg-1.onnx"

            tokens_file = "tokens.txt"

            # 检查模型文件是否存在
            for file_name in [encoder_file, decoder_file, joiner_file, tokens_file]:
                file_path = os.path.join(model_path, file_name)
                if not os.path.exists(file_path):
                    logger.error(f"模型文件不存在: {file_path}")
                    return False

            # 获取配置参数
            num_threads = self.config.get('num_threads', 4)
            sample_rate = self.config.get('sample_rate', 16000)
            feature_dim = self.config.get('feature_dim', 80)
            decoding_method = self.config.get('decoding_method', 'greedy_search')

            # 创建模型实例
            logger.info(f"创建 Sherpa-ONNX {self._model_version} 模型实例 (is_int8={self.is_int8})...")

            try:
                # 使用 from_transducer 方法创建 OnlineRecognizer 实例
                if not HAS_SHERPA_ONNX:
                    logger.error("未安装 sherpa_onnx 模块，无法创建识别器")
                    return False

                # 确保sherpa_onnx已导入
                if not HAS_SHERPA_ONNX or sherpa_onnx is None:
                    logger.error("sherpa_onnx模块未正确导入")
                    return False

                # 确保OnlineRecognizer类存在
                if not hasattr(sherpa_onnx, 'OnlineRecognizer'):
                    logger.error("sherpa_onnx.OnlineRecognizer类不存在")
                    return False

                # 确保from_transducer方法存在
                if not hasattr(sherpa_onnx.OnlineRecognizer, 'from_transducer'):
                    logger.error("sherpa_onnx.OnlineRecognizer.from_transducer方法不存在")
                    return False

                self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                    encoder=os.path.join(model_path, encoder_file),
                    decoder=os.path.join(model_path, decoder_file),
                    joiner=os.path.join(model_path, joiner_file),
                    tokens=os.path.join(model_path, tokens_file),
                    num_threads=num_threads,
                    sample_rate=sample_rate,
                    feature_dim=feature_dim,
                    decoding_method=decoding_method,
                    enable_endpoint_detection=self.config.get('enable_endpoint_detection', True),
                    rule1_min_trailing_silence=self.config.get('rule1_min_trailing_silence', 2.4),
                    rule2_min_trailing_silence=self.config.get('rule2_min_trailing_silence', 1.2),
                    rule3_min_utterance_length=self.config.get('rule3_min_utterance_length', 20.0)
                )

                # 保存引擎类型（不直接设置到recognizer对象，因为sherpa_onnx.OnlineRecognizer没有engine_type属性）
                # self.recognizer.engine_type = self.engine_type

                logger.info(f"Sherpa-ONNX {self._model_version} 插件设置成功 (engine_type={self.engine_type})")
                return True

            except Exception as e:
                logger.error(f"创建 Sherpa-ONNX {self._model_version} 模型实例失败: {str(e)}")
                logger.error(traceback.format_exc())
                return False

        except Exception as e:
            logger.error(f"Sherpa-ONNX {self._model_version} 插件设置失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def teardown(self) -> bool:
        """清理插件资源"""
        try:
            # 释放模型和识别器资源
            self.model = None
            self.recognizer = None
            self.stream = None

            # 清理临时文件
            for temp_file in self.temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                        logger.debug(f"已删除临时文件: {temp_file}")
                    except Exception as e:
                        logger.warning(f"删除临时文件失败: {temp_file}, 错误: {str(e)}")

            self.temp_files = []

            logger.info(f"Sherpa-ONNX {self._model_version} 插件资源已清理")
            return True

        except Exception as e:
            logger.error(f"清理 Sherpa-ONNX {self._model_version} 插件资源失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    # 实现PluginBase的initialize方法
    def initialize(self) -> bool:
        """初始化插件（PluginBase接口）"""
        try:
            if self._initialized:
                logger.warning(f"插件已初始化: {self.get_id()}")
                return True

            # 调用setup方法
            if not self.setup():
                logger.error(f"插件初始化失败: {self.get_id()}")
                return False

            self._initialized = True
            logger.info(f"插件初始化成功: {self.get_id()}")
            return True

        except Exception as e:
            logger.error(f"插件初始化时出错: {self.get_id()}, 错误: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    # 实现PluginBase的cleanup方法
    def cleanup(self) -> bool:
        """清理资源（PluginBase接口）"""
        try:
            if not self._initialized:
                logger.warning(f"插件未初始化: {self.get_id()}")
                return True

            # 调用teardown方法
            if not self.teardown():
                logger.error(f"插件清理失败: {self.get_id()}")
                return False

            # 清理ASRPluginBase相关资源
            try:
                # 手动实现ASRPluginBase的cleanup逻辑
                # 重置流
                self.stream = None
                self.model = None
                self.recognizer = None
                self._is_initialized = False  # 重置初始化状态
            except Exception as e:
                logger.warning(f"ASRPluginBase资源清理失败: {str(e)}")

            self._initialized = False
            logger.info(f"插件清理成功: {self.get_id()}")
            return True

        except Exception as e:
            logger.error(f"清理资源失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def process_audio(self, audio_data) -> Dict[str, Any]:
        """处理音频数据进行实时语音识别

        此方法用于在线实时语音识别，处理流式音频数据。
        它会创建一个持久化的识别流（如果不存在），然后将音频数据送入流中进行处理。

        处理流程：
        1. 检查识别器是否已初始化
        2. 将输入的音频数据转换为numpy数组（如果是bytes）
        3. 创建流（如果不存在）
        4. 将音频数据送入流中
        5. 解码流
        6. 获取识别结果
        7. 格式化文本并返回结果

        Args:
            audio_data: 音频数据，可以是bytes或numpy数组。
                        如果是bytes，将被转换为numpy数组。
                        音频数据应该是16kHz采样率、16位PCM、单声道的。

        Returns:
            Dict[str, Any]: 处理结果，包含以下键：
                - text: 识别的文本
                - is_final: 是否是最终结果
                - error: 如果发生错误，包含错误信息

        示例:
            ```python
            # 处理音频数据
            result = plugin.process_audio(audio_data)
            if "error" not in result:
                text = result["text"]
                is_final = result["is_final"]
                print(f"识别结果: {text}, 是否最终: {is_final}")
            else:
                print(f"处理失败: {result['error']}")
            ```
        """
        try:
            if not self.recognizer:
                logger.error(f"Sherpa-ONNX {self._model_version} 识别器未初始化")
                return {"error": f"Sherpa-ONNX {self._model_version} 识别器未初始化"}

            # 如果输入是bytes，转换为numpy数组
            if isinstance(audio_data, bytes):
                audio_data = np.frombuffer(audio_data, dtype=np.int16)

            # 创建流（如果不存在）
            if self.stream is None:
                self.stream = self.recognizer.create_stream()
                logger.debug(f"已创建新的 Sherpa-ONNX {self._model_version} 流")

            # 处理音频数据
            self.stream.accept_waveform(audio_data)

            # 解码流
            self.recognizer.decode_stream(self.stream)

            # 获取结果
            result = self.stream.result

            if result.text:
                # 格式化文本
                text = result.text.strip()

                # 返回结果
                return {
                    "text": text,
                    "is_final": result.is_final
                }
            else:
                return {"text": "", "is_final": False}

        except Exception as e:
            logger.error(f"处理音频数据失败: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def transcribe_file(self, file_path: str) -> Optional[str]:
        """转录音频文件

        此方法用于转录整个音频文件，支持WAV和非WAV格式。
        对于非WAV格式，需要安装soundcard模块，并使用ffmpeg将其转换为WAV格式。

        处理流程：
        1. 检查识别器是否已初始化
        2. 检查文件是否存在
        3. 读取音频数据（对于WAV格式直接读取，对于非WAV格式先转换为WAV）
        4. 创建新的识别流
        5. 分块处理音频数据
        6. 处理最后的音频数据
        7. 合并所有结果并格式化文本

        Args:
            file_path: 音频文件路径，支持WAV和非WAV格式（如MP3、MP4等）

        Returns:
            Optional[str]: 转录文本，如果成功则返回格式化后的文本，失败返回None

        注意：
            - 如果识别器未初始化，将返回None
            - 如果文件不存在，将返回None
            - 如果处理非WAV格式但未安装soundcard模块，将返回None
            - 如果ffmpeg转换失败，将返回None

        示例:
            ```python
            # 转录音频文件
            text = plugin.transcribe_file("audio.mp3")
            if text:
                print(f"转录结果: {text}")
            else:
                print("转录失败")
            ```
        """
        try:
            if not self.recognizer:
                logger.error(f"Sherpa-ONNX {self._model_version} 识别器未初始化")
                return None

            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                return None

            # 处理不同格式的音频文件
            if file_path.lower().endswith('.wav'):
                # 直接处理WAV文件
                with wave.open(file_path, 'rb') as wf:
                    sample_rate = wf.getframerate()
                    if sample_rate != 16000:
                        logger.warning(f"WAV文件采样率 {sample_rate}Hz 与模型要求的 16000Hz 不匹配，可能影响识别效果")

                    # 读取音频数据
                    audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            else:
                # 对于非WAV格式，先转换为WAV
                if not HAS_SOUNDCARD:
                    logger.error("未安装 soundcard 模块，无法处理非WAV格式音频")
                    return None

                # 创建临时WAV文件
                fd, temp_wav = tempfile.mkstemp(suffix='.wav')
                os.close(fd)
                self.temp_files.append(temp_wav)

                # 使用ffmpeg转换为WAV格式
                logger.info(f"使用ffmpeg将 {file_path} 转换为WAV格式")
                try:
                    subprocess.run([
                        'ffmpeg',
                        '-i', file_path,
                        '-ar', '16000',  # 采样率16kHz
                        '-ac', '1',      # 单声道
                        '-y',            # 覆盖已有文件
                        temp_wav
                    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    # 读取转换后的WAV文件
                    with wave.open(temp_wav, 'rb') as wf:
                        audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

                except subprocess.CalledProcessError as e:
                    logger.error(f"ffmpeg转换失败: {e}")
                    return None

            # 创建新的流
            stream = self.recognizer.create_stream()

            # 处理音频数据
            chunk_size = 8000  # 每次处理的样本数
            results = []

            # 分块处理音频数据
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size]
                stream.accept_waveform(chunk)
                self.recognizer.decode_stream(stream)

                # 如果有文本结果，添加到结果列表
                if stream.result.text:
                    results.append({
                        "text": stream.result.text.strip(),
                        "is_final": stream.result.is_final
                    })

            # 处理最后的音频数据
            stream.input_finished()
            self.recognizer.decode_stream(stream)

            final_text = stream.result.text.strip()
            if final_text:
                results.append({
                    "text": final_text,
                    "is_final": True
                })

            # 合并所有结果
            combined_text = " ".join(r["text"] for r in results)
            if combined_text:
                # 格式化文本
                if len(combined_text) > 0:
                    combined_text = combined_text[0].upper() + combined_text[1:]
                if combined_text[-1] not in ['.', '?', '!', ',', ';', ':', '-']:
                    combined_text += '.'

                return combined_text
            return None

        except Exception as e:
            logger.error(f"转录文件失败: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def get_formatted_transcript(self, text: str) -> str:
        """格式化转录文本

        此方法用于格式化转录文本，使其更易于阅读。
        它会执行以下操作：
        1. 如果文本为空，返回空字符串
        2. 将文本的首字母大写
        3. 如果文本末尾没有标点符号，添加句号

        Args:
            text: 原始转录文本，可能是从识别器直接获取的未格式化文本

        Returns:
            str: 格式化后的文本，首字母大写，末尾有标点符号

        示例:
            ```python
            # 格式化文本
            original_text = "hello world"
            formatted_text = plugin.get_formatted_transcript(original_text)
            print(formatted_text)  # 输出: "Hello world."
            ```
        """
        if not text:
            return ""

        # 首字母大写
        formatted_text = text[0].upper() + text[1:] if text else ""

        # 确保句子以标点符号结尾
        if formatted_text and formatted_text[-1] not in ['.', '?', '!']:
            formatted_text += '.'

        return formatted_text

    def reset_stream(self) -> None:
        """重置识别流

        此方法用于重置识别流，清除所有已处理的音频数据和识别结果。
        在需要重新开始识别时调用此方法，例如：
        - 当用户手动停止识别并重新开始时
        - 当识别出现错误需要重新开始时
        - 当切换识别模式（如从在线识别切换到文件识别）时

        注意：
        - 此方法不会重置识别器，只会重置流
        - 下次调用process_audio时会自动创建新的流

        示例:
            ```python
            # 重置识别流
            plugin.reset_stream()
            # 继续处理新的音频数据
            result = plugin.process_audio(new_audio_data)
            ```
        """
        self.stream = None
        logger.debug(f"已重置 Sherpa-ONNX {self._model_version} 识别流")

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息

        此方法返回当前加载的模型的详细信息，包括名称、版本、类型、引擎类型、路径和语言。
        这些信息可用于显示在UI中，或者用于调试和日志记录。

        Returns:
            Dict[str, Any]: 模型信息，包含以下键：
                - name: 模型名称
                - version: 模型版本（2023-02-20或2023-06-26）
                - type: 模型类型（standard或int8）
                - engine: 引擎类型（sherpa_onnx_std、sherpa_onnx_int8、sherpa_0626_std或sherpa_0626_int8）
                - path: 模型路径
                - language: 模型支持的语言（en-zh或en）

        示例:
            ```python
            # 获取模型信息
            model_info = plugin.get_model_info()
            print(f"模型名称: {model_info['name']}")
            print(f"模型版本: {model_info['version']}")
            print(f"模型类型: {model_info['type']}")
            print(f"引擎类型: {model_info['engine']}")
            print(f"模型路径: {model_info['path']}")
            print(f"支持语言: {model_info['language']}")
            ```
        """
        return {
            "name": self.get_name(),
            "version": self._model_version,
            "type": "int8" if self.is_int8 else "standard",
            "engine": self.engine_type,
            "path": self.model_dir,
            "language": "en-zh" if not self.is_0626 else "en"
        }

    def validate_model_files(self, model_path: str) -> bool:
        """验证模型文件是否完整

        此方法检查指定路径下是否包含所有必要的模型文件。
        根据模型版本（2023-02-20或2023-06-26）和类型（标准或int8），
        检查不同的文件名。

        检查的文件包括：
        - encoder文件（.onnx或.int8.onnx）
        - decoder文件（.onnx或.int8.onnx）
        - joiner文件（.onnx或.int8.onnx）
        - tokens.txt文件

        Args:
            model_path: 模型路径，应该指向包含所有必要模型文件的目录

        Returns:
            bool: 文件是否完整。如果所有必要的文件都存在，返回True；否则返回False。

        注意：
            - 此方法会根据路径名自动检测模型版本（是否包含"0626"或"2023-06-26"）
            - 此方法会根据配置自动检测模型类型（是否为int8）
            - 如果任何必要的文件不存在，将返回False并记录错误日志

        示例:
            ```python
            # 验证模型文件
            is_valid = plugin.validate_model_files("/path/to/model")
            if is_valid:
                print("模型文件完整")
            else:
                print("模型文件不完整")
            ```
        """
        try:
            # 检查是否是2023-06-26模型
            is_0626 = "0626" in model_path or "2023-06-26" in model_path
            is_int8 = self.config.get('type', 'standard').lower() == 'int8'

            # 确定模型文件名
            if is_0626:
                # 2023-06-26模型文件名
                if is_int8:
                    encoder_file = "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
                    decoder_file = "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
                    joiner_file = "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
                else:
                    encoder_file = "encoder-epoch-99-avg-1-chunk-16-left-128.onnx"
                    decoder_file = "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"
                    joiner_file = "joiner-epoch-99-avg-1-chunk-16-left-128.onnx"
            else:
                # 2023-02-20模型文件名
                if is_int8:
                    encoder_file = "encoder-epoch-99-avg-1.int8.onnx"
                    decoder_file = "decoder-epoch-99-avg-1.int8.onnx"
                    joiner_file = "joiner-epoch-99-avg-1.int8.onnx"
                else:
                    encoder_file = "encoder-epoch-99-avg-1.onnx"
                    decoder_file = "decoder-epoch-99-avg-1.onnx"
                    joiner_file = "joiner-epoch-99-avg-1.onnx"

            tokens_file = "tokens.txt"

            # 检查模型文件是否存在
            for file_name in [encoder_file, decoder_file, joiner_file, tokens_file]:
                file_path = os.path.join(model_path, file_name)
                if not os.path.exists(file_path):
                    logger.error(f"模型文件不存在: {file_path}")
                    return False

            return True

        except Exception as e:
            logger.error(f"验证模型文件失败: {str(e)}")
            return False

    def transcribe(self, audio_data: Union[bytes, np.ndarray]) -> Optional[str]:
        """转录音频数据

        Args:
            audio_data: 音频数据

        Returns:
            Optional[str]: 转录文本
        """
        result = self.process_audio(audio_data)
        if "error" in result:
            return None
        return result.get("text", "")

    def validate_files(self) -> bool:
        """验证当前加载的模型文件完整性

        此方法是对validate_model_files的封装，用于验证当前加载的模型文件是否完整。
        它会检查self.model_dir是否已设置，然后调用validate_model_files方法验证文件。

        Returns:
            bool: 文件是否完整有效。如果模型目录未设置或文件不完整，返回False；否则返回True。

        注意：
            - 此方法必须在setup方法之后调用，否则self.model_dir可能未设置
            - 如果模型目录未设置，将返回False并记录错误日志

        示例:
            ```python
            # 验证当前模型文件
            is_valid = plugin.validate_files()
            if is_valid:
                print("当前模型文件完整")
            else:
                print("当前模型文件不完整或模型未加载")
            ```
        """
        if self.model_dir is None:
            logger.error("模型目录未设置")
            return False
        return self.validate_model_files(self.model_dir)

    @property
    def supported_models(self) -> List[str]:
        """支持的模型列表

        Returns:
            List[str]: 支持的模型列表
        """
        return [
            "sherpa_onnx_std",
            "sherpa_onnx_int8",
            "sherpa_0626_std",
            "sherpa_0626_int8"
        ]

    def load_model(self, model_path: str) -> bool:
        """加载指定路径的模型

        此方法用于加载指定路径的模型，替换当前已加载的模型。
        它会执行以下操作：
        1. 保存当前配置
        2. 更新模型路径
        3. 清理当前资源
        4. 重新设置（加载新模型）
        5. 如果失败，恢复原配置

        Args:
            model_path: 模型路径，应该指向包含所有必要模型文件的目录

        Returns:
            bool: 是否成功加载。如果成功加载新模型，返回True；否则返回False。

        注意：
            - 此方法会清理当前资源，包括模型、识别器和流
            - 如果加载失败，会恢复原配置，但不会重新加载原模型
            - 加载成功后，需要重新创建流才能进行识别

        示例:
            ```python
            # 加载新模型
            success = plugin.load_model("/path/to/new/model")
            if success:
                print("新模型加载成功")
                # 重置流
                plugin.reset_stream()
                # 继续处理音频数据
                result = plugin.process_audio(audio_data)
            else:
                print("新模型加载失败")
            ```
        """
        # 保存当前配置
        current_config = self._config.copy()

        # 更新模型路径
        self._config["path"] = model_path

        # 清理当前资源
        self.teardown()

        # 重新设置
        success = self.setup()

        # 如果失败，恢复原配置
        if not success:
            self._config = current_config

        return success
