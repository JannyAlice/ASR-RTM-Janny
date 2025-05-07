"""
音频处理模块
负责音频捕获和处理
"""
import time
import json
import numpy as np
import soundcard as sc
from typing import List, Any
from PyQt5.QtCore import QObject, pyqtSignal, QThread

from src.core.signals import TranscriptionSignals

class AudioDevice:
    """音频设备类"""

    def __init__(self, id, name, is_input=True):
        """
        初始化音频设备

        Args:
            id: 设备ID
            name: 设备名称
            is_input: 是否为输入设备
        """
        self.id = id
        self.name = name
        self.is_input = is_input

    def __str__(self):
        return f"{self.name} ({self.id})"

class AudioWorker(QObject):
    """音频处理工作线程"""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    new_text = pyqtSignal(str)
    status = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, device, sample_rate, buffer_size, recognizer):
        super().__init__()
        self.device = device
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.recognizer = recognizer
        self.running = True
        self._last_partial_result = ""  # 保存最后一个部分结果

        # 静音检测相关参数
        self.silence_threshold = 0.01  # 静音阈值
        self.silence_frames = 0  # 连续静音帧计数
        self.silence_frames_threshold = 15  # 静音帧阈值（约1.5秒，取决于buffer_size和采样率）
        self.last_sentence_end_time = time.time()  # 上次句子结束时间
        self.current_partial_text = ""  # 当前累积的部分文本
        self.sentence_in_progress = False  # 是否有句子正在进行中

        # 尝试初始化COM（在主线程中）
        try:
            from src.utils.com_handler import com_handler
            if not hasattr(com_handler, "_initialized") or not com_handler._initialized:
                print("AudioWorker初始化时初始化COM...")
                com_handler.initialize_com()
                print("AudioWorker初始化时COM初始化成功")
            else:
                print("COM已经初始化，AudioWorker初始化跳过COM初始化")
        except Exception as e:
            print(f"AudioWorker初始化时COM初始化错误: {e}")
            # 即使COM初始化失败，也继续执行

    def process(self):
        """处理音频数据"""
        start_time = time.time()
        last_progress_update = time.time()

        # 导入日志工具
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
            # 确保COM已初始化
            try:
                from src.utils.com_handler import com_handler
                if not hasattr(com_handler, "_initialized") or not com_handler._initialized:
                    sherpa_logger.info("AudioWorker线程中初始化COM...")
                    com_handler.initialize_com()
                    sherpa_logger.info("AudioWorker线程中COM初始化成功")
                else:
                    sherpa_logger.info("COM已经初始化，AudioWorker线程跳过初始化")
            except Exception as e:
                sherpa_logger.error(f"COM初始化错误: {e}")
                import traceback
                sherpa_logger.error(traceback.format_exc())
                # 即使COM初始化失败，也尝试继续执行

            # 记录引擎类型
            engine_type = getattr(self.recognizer, 'engine_type', None)
            sherpa_logger.info(f"开始音频处理，引擎类型: {engine_type}")

            with sc.get_microphone(id=str(self.device.id), include_loopback=True).recorder(
                samplerate=self.sample_rate
            ) as mic:
                self.status.emit(f"正在从 {self.device.name} 捕获音频...")
                sherpa_logger.info(f"正在从 {self.device.name} 捕获音频...")

                while self.running:
                    # 捕获音频数据
                    data = mic.record(numframes=self.buffer_size)

                    # 记录音频数据信息
                    sherpa_logger.debug(f"捕获音频数据，形状: {data.shape}")

                    # 转换为单声道
                    if data.shape[1] > 1:
                        data = np.mean(data, axis=1)
                        sherpa_logger.debug(f"转换为单声道，形状: {data.shape}")

                    # 检查音频数据是否有效
                    max_amplitude = np.max(np.abs(data))

                    # 静音检测
                    if max_amplitude < self.silence_threshold:
                        sherpa_logger.debug(f"检测到静音，最大振幅: {max_amplitude}，静音帧计数: {self.silence_frames}")
                        self.silence_frames += 1

                        # 如果有句子正在进行中，且静音持续足够长时间，认为句子结束
                        if self.sentence_in_progress and self.silence_frames >= self.silence_frames_threshold:
                            sherpa_logger.info(f"检测到静音持续{self.silence_frames}帧，判定当前句子结束")

                            # 如果有当前部分文本，将其作为完整句子提交
                            if self._last_partial_result:
                                # 格式化文本
                                text = self._last_partial_result
                                if len(text) > 0:
                                    text = text[0].upper() + text[1:]
                                if text[-1] not in ['.', '?', '!']:
                                    text += '.'

                                sherpa_logger.info(f"静音检测触发句子结束，发送完整文本: {text}")
                                self.new_text.emit(text)

                                # 重置状态
                                self._last_partial_result = ""
                                self.sentence_in_progress = False
                                self.last_sentence_end_time = time.time()

                        # 如果静音时间太长，跳过此帧
                        if self.silence_frames > 2:  # 允许短暂静音
                            continue
                    else:
                        # 如果检测到声音，重置静音计数
                        if self.silence_frames > 0:
                            sherpa_logger.debug(f"检测到声音，重置静音帧计数，之前为: {self.silence_frames}")
                            self.silence_frames = 0

                        # 标记有句子正在进行中
                        self.sentence_in_progress = True

                    try:
                        # 处理音频数据
                        engine_type = getattr(self.recognizer, 'engine_type', None)
                        sherpa_logger.debug(f"处理音频数据，引擎类型: {engine_type}")

                        if engine_type and engine_type.startswith('sherpa'):
                            # 对于 Sherpa-ONNX 模型，直接传递 numpy 数组
                            sherpa_logger.debug(f"使用 Sherpa-ONNX 模型，直接传递 numpy 数组")
                            accept_result = self.recognizer.AcceptWaveform(data)
                        else:
                            # 对于 Vosk 模型，转换为 16 位整数字节
                            data_bytes = (data * 32767).astype(np.int16).tobytes()
                            sherpa_logger.debug(f"使用 Vosk 模型，转换为 16 位整数字节，长度: {len(data_bytes)}")
                            accept_result = self.recognizer.AcceptWaveform(data_bytes)

                        sherpa_logger.debug(f"AcceptWaveform 结果: {accept_result}")

                        if accept_result:
                            # 获取完整结果
                            result = self.recognizer.Result()
                            sherpa_logger.info(f"完整结果: {result}, 类型: {type(result)}")

                            text = self._parse_result(result)
                            sherpa_logger.info(f"解析后的完整结果: {text}")

                            if text:
                                sherpa_logger.info(f"发送完整文本: {text}")
                                self.new_text.emit(text)
                            else:
                                sherpa_logger.warning(f"完整文本为空，不发送")
                        else:
                            # 获取部分结果
                            partial = self.recognizer.PartialResult()
                            sherpa_logger.debug(f"部分结果: {partial}, 类型: {type(partial)}")

                            text = self._parse_partial_result(partial)
                            sherpa_logger.debug(f"解析后的部分结果: {text}")

                            # 保存最新的部分结果，无论是否发送
                            if text:
                                self._last_partial_result = text
                                print(f"在process中保存最新部分结果: {text}")

                                # 检查是否需要因为静音而结束句子
                                current_time = time.time()
                                time_since_last_sentence = current_time - self.last_sentence_end_time

                                # 如果静音持续足够长，且距离上次句子结束已经过了足够时间，认为是新句子
                                if self.silence_frames >= self.silence_frames_threshold and time_since_last_sentence > 2.0:
                                    # 格式化文本
                                    complete_text = text
                                    if len(complete_text) > 0:
                                        complete_text = complete_text[0].upper() + complete_text[1:]
                                    if complete_text[-1] not in ['.', '?', '!']:
                                        complete_text += '.'

                                    sherpa_logger.info(f"静音检测触发句子结束，发送完整文本: {complete_text}")
                                    self.new_text.emit(complete_text)

                                    # 重置状态
                                    self._last_partial_result = ""
                                    self.sentence_in_progress = False
                                    self.last_sentence_end_time = current_time
                                else:
                                    # 正常发送部分文本
                                    sherpa_logger.debug(f"发送部分文本: {text}")
                                    self.new_text.emit("PARTIAL:" + text)
                            else:
                                sherpa_logger.debug(f"部分文本为空，不发送")

                        # 更新进度
                        current_time = time.time()
                        if current_time - last_progress_update >= 0.5:
                            elapsed_seconds = current_time - start_time
                            minutes = int(elapsed_seconds // 60)
                            seconds = int(elapsed_seconds % 60)
                            time_str = f"转录时长: {minutes:02d}:{seconds:02d}"
                            self.progress.emit(50, time_str)
                            last_progress_update = current_time

                    except Exception as e:
                        error_msg = f"音频处理错误: {str(e)}"
                        self.error.emit(error_msg)
                        sherpa_logger.error(error_msg)
                        import traceback
                        error_trace = traceback.format_exc()
                        sherpa_logger.error(error_trace)
                        print(error_trace)

        except Exception as e:
            error_msg = f"音频捕获错误: {str(e)}"
            self.error.emit(error_msg)
            sherpa_logger.error(error_msg)
            import traceback
            error_trace = traceback.format_exc()
            sherpa_logger.error(error_trace)
            print(error_trace)
        finally:
            # 在结束前获取最终结果
            try:
                if hasattr(self, 'recognizer') and self.recognizer:
                    sherpa_logger.info("获取最终识别结果")

                    # 检查引擎类型
                    engine_type = getattr(self.recognizer, 'engine_type', None)
                    sherpa_logger.info(f"引擎类型: {engine_type}")

                    # 获取最终结果
                    if engine_type == "vosk_small" or engine_type == "vosk":
                        final_result = self.recognizer.FinalResult()
                        sherpa_logger.info(f"Vosk最终结果: {final_result}")

                        # 解析最终结果
                        if isinstance(final_result, str):
                            try:
                                result_json = json.loads(final_result)
                                text = result_json.get('text', '').strip()
                                sherpa_logger.info(f"解析后的最终结果: {text}")

                                # 如果有文本，发送到UI
                                if text:
                                    # 格式化文本
                                    if len(text) > 0:
                                        text = text[0].upper() + text[1:]
                                    if text[-1] not in ['.', '?', '!']:
                                        text += '.'
                                    sherpa_logger.info(f"发送最终文本: {text}")
                                    self.new_text.emit(text)
                                elif hasattr(self, '_last_partial_result') and self._last_partial_result:
                                    # 如果最终结果为空但有最后一个部分结果，使用部分结果作为最终结果
                                    text = self._last_partial_result
                                    # 记录使用的部分结果
                                    sherpa_logger.info(f"使用的最后一个部分结果原始值: {text}")

                                    # 检查是否是完整的句子（通过检查是否包含"and"等连接词判断）
                                    # 如果不是完整句子，可能是因为部分结果被截断了
                                    if text and (" and " in text or text.endswith(" and")):
                                        # 尝试从完整结果中找到匹配的句子
                                        # 这里假设完整结果已经在UI中显示
                                        try:
                                            from src.ui.main_window import MainWindow
                                            if hasattr(MainWindow, 'instance') and MainWindow.instance:
                                                if hasattr(MainWindow.instance, 'subtitle_widget'):
                                                    subtitle_widget = MainWindow.instance.subtitle_widget
                                                    if hasattr(subtitle_widget, 'transcript_text') and subtitle_widget.transcript_text:
                                                        # 查找最近的完整结果中是否包含当前部分结果
                                                        for complete_text in reversed(subtitle_widget.transcript_text):
                                                            if text in complete_text:
                                                                text = complete_text
                                                                sherpa_logger.info(f"找到匹配的完整结果: {text}")
                                                                break
                                        except Exception as e:
                                            sherpa_logger.error(f"尝试查找完整结果时出错: {e}")

                                    # 格式化文本
                                    if len(text) > 0:
                                        text = text[0].upper() + text[1:]
                                    if text[-1] not in ['.', '?', '!']:
                                        text += '.'
                                    sherpa_logger.info(f"使用最后一个部分结果作为最终文本: {text}")
                                    self.new_text.emit(text)
                            except json.JSONDecodeError:
                                sherpa_logger.error(f"解析最终结果JSON失败: {final_result}")

                                # 如果JSON解析失败但有最后一个部分结果，使用部分结果作为最终结果
                                if hasattr(self, '_last_partial_result') and self._last_partial_result:
                                    text = self._last_partial_result
                                    # 格式化文本
                                    if len(text) > 0:
                                        text = text[0].upper() + text[1:]
                                    if text[-1] not in ['.', '?', '!']:
                                        text += '.'
                                    sherpa_logger.info(f"JSON解析失败，使用最后一个部分结果作为最终文本: {text}")
                                    self.new_text.emit(text)
                    else:
                        # 对于其他模型，尝试调用FinalResult方法
                        if hasattr(self.recognizer, 'FinalResult'):
                            final_result = self.recognizer.FinalResult()
                            sherpa_logger.info(f"其他模型最终结果: {final_result}")

                            # 如果是字符串，尝试解析
                            if isinstance(final_result, str):
                                text = final_result.strip()
                                if text:
                                    sherpa_logger.info(f"发送最终文本: {text}")
                                    self.new_text.emit(text)
                                elif hasattr(self, '_last_partial_result') and self._last_partial_result:
                                    # 如果最终结果为空但有最后一个部分结果，使用部分结果作为最终结果
                                    text = self._last_partial_result
                                    # 格式化文本
                                    if len(text) > 0:
                                        text = text[0].upper() + text[1:]
                                    if text[-1] not in ['.', '?', '!']:
                                        text += '.'
                                    sherpa_logger.info(f"使用最后一个部分结果作为最终文本: {text}")
                                    self.new_text.emit(text)
            except Exception as e:
                sherpa_logger.error(f"获取最终结果错误: {e}")
                import traceback
                sherpa_logger.error(traceback.format_exc())

                # 如果获取最终结果失败但有最后一个部分结果，使用部分结果作为最终结果
                if hasattr(self, '_last_partial_result') and self._last_partial_result:
                    text = self._last_partial_result
                    # 记录使用的部分结果
                    sherpa_logger.info(f"获取最终结果失败，使用的最后一个部分结果原始值: {text}")

                    # 检查是否是完整的句子（通过检查是否包含"and"等连接词判断）
                    # 如果不是完整句子，可能是因为部分结果被截断了
                    if text and (" and " in text or text.endswith(" and")):
                        # 尝试从完整结果中找到匹配的句子
                        # 这里假设完整结果已经在UI中显示
                        try:
                            from src.ui.main_window import MainWindow
                            if hasattr(MainWindow, 'instance') and MainWindow.instance:
                                if hasattr(MainWindow.instance, 'subtitle_widget'):
                                    subtitle_widget = MainWindow.instance.subtitle_widget
                                    if hasattr(subtitle_widget, 'transcript_text') and subtitle_widget.transcript_text:
                                        # 查找最近的完整结果中是否包含当前部分结果
                                        for complete_text in reversed(subtitle_widget.transcript_text):
                                            if text in complete_text:
                                                text = complete_text
                                                sherpa_logger.info(f"找到匹配的完整结果: {text}")
                                                break
                        except Exception as e:
                            sherpa_logger.error(f"尝试查找完整结果时出错: {e}")

                    # 格式化文本
                    if len(text) > 0:
                        text = text[0].upper() + text[1:]
                    if text[-1] not in ['.', '?', '!']:
                        text += '.'
                    sherpa_logger.info(f"获取最终结果失败，使用最后一个部分结果作为最终文本: {text}")
                    self.new_text.emit(text)

            sherpa_logger.info("音频处理结束")
            self.finished.emit()

    def _parse_result(self, result):
        """解析完整识别结果"""
        try:
            # 导入日志工具
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

            # 记录原始结果
            sherpa_logger.debug(f"解析完整结果: {result}, 类型: {type(result)}")

            # 检查引擎类型
            engine_type = getattr(self.recognizer, 'engine_type', None)
            sherpa_logger.debug(f"引擎类型: {engine_type}")

            # 如果是Vosk引擎，特殊处理
            if engine_type == "vosk_small" or engine_type == "vosk":
                sherpa_logger.debug("使用Vosk特殊处理逻辑")
                # Vosk引擎返回的是JSON字符串
                if isinstance(result, str):
                    try:
                        result_json = json.loads(result)
                        text = result_json.get('text', '').strip()
                        sherpa_logger.debug(f"Vosk JSON解析结果: {text}")
                    except json.JSONDecodeError:
                        # 如果不是有效的JSON，直接使用文本
                        text = result.strip()
                        sherpa_logger.debug(f"Vosk非JSON结果: {text}")
                else:
                    # 如果不是字符串，尝试转换为字符串
                    text = str(result).strip()
                    sherpa_logger.debug(f"Vosk非字符串结果: {text}")

                # 格式化文本
                if text:
                    if len(text) > 0:
                        text = text[0].upper() + text[1:]
                    if text[-1] not in ['.', '?', '!']:
                        text += '.'
                    sherpa_logger.debug(f"Vosk格式化后结果: {text}")
                    return text
                return None

            # 如果是纯文本字符串（不是 JSON），直接使用
            if isinstance(result, str) and not result.startswith('{'):
                text = result.strip()
                if text:
                    # 格式化文本
                    if len(text) > 0:
                        text = text[0].upper() + text[1:]
                    if text[-1] not in ['.', '?', '!']:
                        text += '.'
                return text

            # 尝试解析 JSON 或其他格式
            try:
                if isinstance(result, str):
                    result_json = json.loads(result)
                elif hasattr(result, 'text'):
                    result_json = {"text": result.text}
                elif hasattr(result, '__str__'):
                    result_json = {"text": str(result)}
                else:
                    return None

                text = result_json.get('text', '').strip()
                if text:
                    # 格式化文本
                    if len(text) > 0:
                        text = text[0].upper() + text[1:]
                    if text[-1] not in ['.', '?', '!']:
                        text += '.'
                    return text
            except Exception as e:
                sherpa_logger.error(f"解析结果错误: {e}")
                # 如果解析失败，尝试直接使用结果
                text = str(result).strip()
                if text:
                    # 格式化文本
                    if len(text) > 0:
                        text = text[0].upper() + text[1:]
                    if text[-1] not in ['.', '?', '!']:
                        text += '.'
                    return text
        except Exception as e:
            print(f"_parse_result 错误: {e}")
            import traceback
            print(traceback.format_exc())

        return None

    def _parse_partial_result(self, partial):
        """解析部分识别结果"""
        try:
            # 导入日志工具
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

            # 记录原始结果
            sherpa_logger.debug(f"解析部分结果: {partial}, 类型: {type(partial)}")

            # 检查引擎类型
            engine_type = getattr(self.recognizer, 'engine_type', None)
            sherpa_logger.debug(f"引擎类型: {engine_type}")

            # 如果是Vosk引擎，特殊处理
            if engine_type == "vosk_small" or engine_type == "vosk":
                sherpa_logger.debug("使用Vosk特殊处理逻辑")
                # Vosk引擎返回的是JSON字符串
                if isinstance(partial, str):
                    try:
                        partial_json = json.loads(partial)
                        partial_text = partial_json.get('partial', '').strip()
                        sherpa_logger.debug(f"Vosk JSON解析部分结果: {partial_text}")

                        # 格式化部分文本 - 首字母大写，但不添加句尾标点
                        if partial_text:
                            if len(partial_text) > 0:
                                partial_text = partial_text[0].upper() + partial_text[1:]
                            sherpa_logger.debug(f"Vosk格式化后的部分结果: {partial_text}")

                        # 保存最新的部分结果，用于后续处理
                        # 这对于在停止转录时获取最后一个单词特别有用
                        self._last_partial_result = partial_text
                        print(f"保存最新部分结果: {partial_text}")  # 添加打印，便于调试

                        return partial_text
                    except json.JSONDecodeError:
                        # 如果不是有效的JSON，直接使用文本
                        partial_text = partial.strip()
                        sherpa_logger.debug(f"Vosk非JSON部分结果: {partial_text}")

                        # 格式化部分文本
                        if partial_text:
                            if len(partial_text) > 0:
                                partial_text = partial_text[0].upper() + partial_text[1:]

                        # 保存最新的部分结果
                        self._last_partial_result = partial_text
                        print(f"保存最新部分结果(非JSON): {partial_text}")  # 添加打印，便于调试

                        return partial_text
                else:
                    # 如果不是字符串，尝试转换为字符串
                    partial_text = str(partial).strip()
                    sherpa_logger.debug(f"Vosk非字符串部分结果: {partial_text}")

                    # 格式化部分文本
                    if partial_text:
                        if len(partial_text) > 0:
                            partial_text = partial_text[0].upper() + partial_text[1:]

                    # 保存最新的部分结果
                    self._last_partial_result = partial_text

                    return partial_text

            # 如果是纯文本字符串（不是 JSON），直接使用
            if isinstance(partial, str) and not partial.startswith('{'):
                partial_text = partial.strip()

                # 格式化部分文本
                if partial_text:
                    if len(partial_text) > 0:
                        partial_text = partial_text[0].upper() + partial_text[1:]

                # 保存最新的部分结果
                self._last_partial_result = partial_text

                return partial_text

            # 尝试解析 JSON 或其他格式
            try:
                if isinstance(partial, str):
                    try:
                        partial_json = json.loads(partial)
                    except json.JSONDecodeError:
                        partial_text = partial.strip()

                        # 格式化部分文本
                        if partial_text:
                            if len(partial_text) > 0:
                                partial_text = partial_text[0].upper() + partial_text[1:]

                        # 保存最新的部分结果
                        self._last_partial_result = partial_text

                        return partial_text
                elif isinstance(partial, dict):
                    partial_json = partial
                elif hasattr(partial, 'partial'):
                    partial_json = {'partial': str(partial.partial)}
                else:
                    partial_json = {'partial': str(partial)}

                partial_text = partial_json.get('partial', '').strip()

                # 格式化部分文本
                if partial_text:
                    if len(partial_text) > 0:
                        partial_text = partial_text[0].upper() + partial_text[1:]

                # 保存最新的部分结果
                self._last_partial_result = partial_text

                return partial_text
            except Exception as e:
                sherpa_logger.error(f"解析部分结果错误: {e}")
                # 如果解析失败，尝试直接使用结果
                partial_text = str(partial).strip()

                # 格式化部分文本
                if partial_text:
                    if len(partial_text) > 0:
                        partial_text = partial_text[0].upper() + partial_text[1:]

                # 保存最新的部分结果
                self._last_partial_result = partial_text

                return partial_text
        except Exception as e:
            print(f"_parse_partial_result 错误: {e}")
            import traceback
            print(traceback.format_exc())
            return ""

class AudioProcessor(QObject):
    """音频处理器类"""
    # 定义 Qt 信号
    new_text_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)

    def __init__(self, signals: TranscriptionSignals):
        super().__init__()
        self.signals = signals
        self.current_device = None
        self.is_capturing = False
        self.capture_thread = None
        self.sample_rate = 16000
        self.buffer_size = 4000
        self.worker_thread = None

    def get_audio_devices(self) -> List[AudioDevice]:
        """
        获取音频设备列表

        Returns:
            List[AudioDevice]: 音频设备列表
        """
        devices = []

        try:
            # 获取所有输入设备
            speakers = sc.all_speakers()
            for speaker in speakers:
                devices.append(AudioDevice(speaker.id, speaker.name, False))

            # 获取所有输出设备
            mics = sc.all_microphones(include_loopback=True)
            for mic in mics:
                devices.append(AudioDevice(mic.id, mic.name, True))

            return devices

        except Exception as e:
            print(f"获取音频设备失败: {e}")
            return []

    def set_current_device(self, device: AudioDevice) -> bool:
        """
        设置当前设备

        Args:
            device: 音频设备

        Returns:
            bool: 设置是否成功
        """
        if not device:
            return False

        self.current_device = device
        return True

    def start_capture(self, recognizer: Any) -> bool:
        """开始捕获音频"""
        if self.is_capturing:
            return False

        if not self.current_device:
            self.error_signal.emit("未选择音频设备")
            return False

        # 创建工作线程
        self.worker_thread = QThread()
        self.worker = AudioWorker(
            self.current_device,
            self.sample_rate,
            self.buffer_size,
            recognizer
        )
        self.worker.moveToThread(self.worker_thread)

        # 连接信号
        self.worker_thread.started.connect(self.worker.process)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        # 转发信号到TranscriptionSignals实例
        self.worker.new_text.connect(lambda x: self.signals.new_text.emit(x))
        self.worker.error.connect(lambda x: self.signals.error_occurred.emit(x))
        self.worker.status.connect(lambda x: self.signals.status_updated.emit(x))
        self.worker.progress.connect(lambda x, y: self.signals.progress_updated.emit(x, y))

        # 启动线程
        self.is_capturing = True
        self.worker_thread.start()

        return True

    def stop_capture(self) -> bool:
        """停止捕获音频"""
        if not self.is_capturing:
            return False

        # 导入日志工具
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

        # 标记停止状态，防止在停止后继续处理部分结果
        if hasattr(self, 'worker') and self.worker:
            self.worker.running = False
            sherpa_logger.info("已标记工作线程为停止状态")

        # 在停止捕获前，获取最终结果
        try:
            if hasattr(self, 'worker') and self.worker and hasattr(self.worker, 'recognizer'):
                recognizer = self.worker.recognizer
                sherpa_logger.info("获取最终识别结果")

                # 检查引擎类型
                engine_type = getattr(recognizer, 'engine_type', None)
                sherpa_logger.info(f"引擎类型: {engine_type}")

                # 获取最终结果
                try:
                    # 对于Vosk模型，调用FinalResult方法
                    if engine_type == "vosk_small" or engine_type == "vosk":
                        final_result = recognizer.FinalResult()
                        sherpa_logger.info(f"Vosk最终结果: {final_result}")

                        # 解析最终结果
                        if isinstance(final_result, str):
                            try:
                                result_json = json.loads(final_result)
                                text = result_json.get('text', '').strip()
                                sherpa_logger.info(f"解析后的最终结果: {text}")

                                # 如果有文本，发送到UI
                                if text:
                                    # 格式化文本
                                    if len(text) > 0:
                                        text = text[0].upper() + text[1:]
                                    if text[-1] not in ['.', '?', '!']:
                                        text += '.'
                                    sherpa_logger.info(f"发送最终文本: {text}")
                                    self.signals.new_text.emit(text)
                                elif hasattr(self.worker, '_last_partial_result') and self.worker._last_partial_result:
                                    # 如果最终结果为空但有最后一个部分结果，使用部分结果作为最终结果
                                    # 注意：此时self.worker._last_partial_result可能已经是匹配过的完整结果
                                    # 因为在SubtitleWidget的update_text方法中，我们已经尝试匹配完整结果
                                    text = self.worker._last_partial_result
                                    # 记录使用的部分结果
                                    sherpa_logger.info(f"使用的最后一个部分结果原始值: {text}")

                                    # 尝试查找匹配的完整句子
                                    try:
                                        from src.ui.main_window import MainWindow
                                        if hasattr(MainWindow, 'instance') and MainWindow.instance:
                                            if hasattr(MainWindow.instance, 'subtitle_widget'):
                                                subtitle_widget = MainWindow.instance.subtitle_widget
                                                if hasattr(subtitle_widget, '_find_matching_complete_text'):
                                                    matched_text = subtitle_widget._find_matching_complete_text(text)
                                                    if matched_text:
                                                        sherpa_logger.info(f"找到匹配的完整句子: {matched_text}")
                                                        text = matched_text
                                    except Exception as e:
                                        sherpa_logger.error(f"尝试查找匹配的完整句子时出错: {e}")

                                    # 检查是否是完整的句子（通过检查是否包含"and"等连接词判断）
                                    # 如果不是完整句子，可能是因为部分结果被截断了
                                    if text:
                                        # 尝试从完整结果中找到匹配的句子
                                        # 这里假设完整结果已经在UI中显示
                                        try:
                                            from src.ui.main_window import MainWindow
                                            if hasattr(MainWindow, 'instance') and MainWindow.instance:
                                                if hasattr(MainWindow.instance, 'subtitle_widget'):
                                                    subtitle_widget = MainWindow.instance.subtitle_widget
                                                    if hasattr(subtitle_widget, 'transcript_text') and subtitle_widget.transcript_text:
                                                        # 打印完整的transcript_text列表，便于调试
                                                        sherpa_logger.info(f"当前完整文本列表: {subtitle_widget.transcript_text}")

                                                        # 首先检查是否有以"and"结尾的部分结果
                                                        if " and " in text or text.endswith(" and"):
                                                            sherpa_logger.info(f"检测到以'and'结尾的部分结果: {text}")
                                                            # 特殊处理以"and"结尾的情况
                                                            for complete_text in reversed(subtitle_widget.transcript_text):
                                                                # 检查是否有包含相同前缀但更完整的句子
                                                                if complete_text.startswith(text.rstrip(" and")) and "and " in complete_text:
                                                                    text = complete_text
                                                                    sherpa_logger.info(f"找到匹配的完整结果(and特殊处理): {text}")
                                                                    break

                                                        # 如果没有找到匹配，继续使用常规匹配逻辑
                                                        if text.endswith(" and"):
                                                            # 查找最近的完整结果中是否包含当前部分结果
                                                            for complete_text in reversed(subtitle_widget.transcript_text):
                                                                # 检查部分结果是否是完整结果的前缀（去掉末尾的"and"）
                                                                prefix = text.rstrip(" and")
                                                                if prefix and complete_text.startswith(prefix):
                                                                    text = complete_text
                                                                    sherpa_logger.info(f"找到匹配的完整结果(前缀匹配-去除and): {text}")
                                                                    break

                                                        # 如果仍然没有找到匹配，使用常规匹配逻辑
                                                        if text.endswith(" and"):
                                                            for complete_text in reversed(subtitle_widget.transcript_text):
                                                                # 检查部分结果（去掉末尾的"and"）是否包含在完整结果中
                                                                prefix = text.rstrip(" and")
                                                                if prefix and prefix in complete_text:
                                                                    text = complete_text
                                                                    sherpa_logger.info(f"找到匹配的完整结果(子串匹配-去除and): {text}")
                                                                    break

                                                        # 如果仍然没有找到匹配，使用常规匹配逻辑
                                                        for complete_text in reversed(subtitle_widget.transcript_text):
                                                            # 检查部分结果是否是完整结果的前缀
                                                            if complete_text.startswith(text):
                                                                text = complete_text
                                                                sherpa_logger.info(f"找到匹配的完整结果(前缀匹配): {text}")
                                                                break
                                                            # 检查部分结果是否包含在完整结果中
                                                            elif text in complete_text:
                                                                text = complete_text
                                                                sherpa_logger.info(f"找到匹配的完整结果(子串匹配): {text}")
                                                                break
                                                            # 检查完整结果是否包含部分结果的大部分内容
                                                            elif len(text) > 10:  # 只对较长的部分结果进行相似度检查
                                                                # 计算部分结果的单词
                                                                partial_words = text.split()
                                                                # 计算完整结果的单词
                                                                complete_words = complete_text.split()
                                                                # 计算共同单词的数量
                                                                common_words = set(partial_words) & set(complete_words)
                                                                # 如果共同单词的数量超过部分结果单词数量的80%，认为匹配
                                                                if len(common_words) >= 0.8 * len(partial_words):
                                                                    text = complete_text
                                                                    sherpa_logger.info(f"找到匹配的完整结果(相似度匹配): {text}")
                                                                    break

                                                                # 如果部分结果以"and"结尾，特殊处理
                                                                if text.endswith(" and"):
                                                                    # 计算去掉"and"后的相似度
                                                                    partial_words_no_and = text.rstrip(" and").split()
                                                                    if partial_words_no_and:
                                                                        common_words_no_and = set(partial_words_no_and) & set(complete_words)
                                                                        if len(common_words_no_and) >= 0.8 * len(partial_words_no_and):
                                                                            text = complete_text
                                                                            sherpa_logger.info(f"找到匹配的完整结果(相似度匹配-去除and): {text}")
                                                                            break
                                        except Exception as e:
                                            sherpa_logger.error(f"尝试查找完整结果时出错: {e}")

                                    # 格式化文本
                                    if len(text) > 0:
                                        text = text[0].upper() + text[1:]
                                    if text[-1] not in ['.', '?', '!']:
                                        text += '.'
                                    sherpa_logger.info(f"使用最后一个部分结果作为最终文本: {text}")
                                    self.signals.new_text.emit(text)
                            except json.JSONDecodeError:
                                sherpa_logger.error(f"解析最终结果JSON失败: {final_result}")

                                # 如果JSON解析失败但有最后一个部分结果，使用部分结果作为最终结果
                                if hasattr(self.worker, '_last_partial_result') and self.worker._last_partial_result:
                                    text = self.worker._last_partial_result
                                    # 记录使用的部分结果
                                    sherpa_logger.info(f"JSON解析失败，使用的最后一个部分结果原始值: {text}")

                                    # 检查是否是完整的句子（通过检查是否包含"and"等连接词判断）
                                    # 如果不是完整句子，可能是因为部分结果被截断了
                                    if text:
                                        # 尝试从完整结果中找到匹配的句子
                                        # 这里假设完整结果已经在UI中显示
                                        try:
                                            from src.ui.main_window import MainWindow
                                            if hasattr(MainWindow, 'instance') and MainWindow.instance:
                                                if hasattr(MainWindow.instance, 'subtitle_widget'):
                                                    subtitle_widget = MainWindow.instance.subtitle_widget
                                                    if hasattr(subtitle_widget, 'transcript_text') and subtitle_widget.transcript_text:
                                                        # 打印完整的transcript_text列表，便于调试
                                                        sherpa_logger.info(f"当前完整文本列表: {subtitle_widget.transcript_text}")

                                                        # 首先检查是否有以"and"结尾的部分结果
                                                        if " and " in text or text.endswith(" and"):
                                                            sherpa_logger.info(f"检测到以'and'结尾的部分结果: {text}")
                                                            # 特殊处理以"and"结尾的情况
                                                            for complete_text in reversed(subtitle_widget.transcript_text):
                                                                # 检查是否有包含相同前缀但更完整的句子
                                                                if complete_text.startswith(text.rstrip(" and")) and "and " in complete_text:
                                                                    text = complete_text
                                                                    sherpa_logger.info(f"找到匹配的完整结果(and特殊处理): {text}")
                                                                    break

                                                        # 如果没有找到匹配，继续使用常规匹配逻辑
                                                        if text.endswith(" and"):
                                                            # 查找最近的完整结果中是否包含当前部分结果
                                                            for complete_text in reversed(subtitle_widget.transcript_text):
                                                                # 检查部分结果是否是完整结果的前缀（去掉末尾的"and"）
                                                                prefix = text.rstrip(" and")
                                                                if prefix and complete_text.startswith(prefix):
                                                                    text = complete_text
                                                                    sherpa_logger.info(f"找到匹配的完整结果(前缀匹配-去除and): {text}")
                                                                    break

                                                        # 如果仍然没有找到匹配，使用常规匹配逻辑
                                                        if text.endswith(" and"):
                                                            for complete_text in reversed(subtitle_widget.transcript_text):
                                                                # 检查部分结果（去掉末尾的"and"）是否包含在完整结果中
                                                                prefix = text.rstrip(" and")
                                                                if prefix and prefix in complete_text:
                                                                    text = complete_text
                                                                    sherpa_logger.info(f"找到匹配的完整结果(子串匹配-去除and): {text}")
                                                                    break

                                                        # 如果仍然没有找到匹配，使用常规匹配逻辑
                                                        for complete_text in reversed(subtitle_widget.transcript_text):
                                                            # 检查部分结果是否是完整结果的前缀
                                                            if complete_text.startswith(text):
                                                                text = complete_text
                                                                sherpa_logger.info(f"找到匹配的完整结果(前缀匹配): {text}")
                                                                break
                                                            # 检查部分结果是否包含在完整结果中
                                                            elif text in complete_text:
                                                                text = complete_text
                                                                sherpa_logger.info(f"找到匹配的完整结果(子串匹配): {text}")
                                                                break
                                                            # 检查完整结果是否包含部分结果的大部分内容
                                                            elif len(text) > 10:  # 只对较长的部分结果进行相似度检查
                                                                # 计算部分结果的单词
                                                                partial_words = text.split()
                                                                # 计算完整结果的单词
                                                                complete_words = complete_text.split()
                                                                # 计算共同单词的数量
                                                                common_words = set(partial_words) & set(complete_words)
                                                                # 如果共同单词的数量超过部分结果单词数量的80%，认为匹配
                                                                if len(common_words) >= 0.8 * len(partial_words):
                                                                    text = complete_text
                                                                    sherpa_logger.info(f"找到匹配的完整结果(相似度匹配): {text}")
                                                                    break

                                                                # 如果部分结果以"and"结尾，特殊处理
                                                                if text.endswith(" and"):
                                                                    # 计算去掉"and"后的相似度
                                                                    partial_words_no_and = text.rstrip(" and").split()
                                                                    if partial_words_no_and:
                                                                        common_words_no_and = set(partial_words_no_and) & set(complete_words)
                                                                        if len(common_words_no_and) >= 0.8 * len(partial_words_no_and):
                                                                            text = complete_text
                                                                            sherpa_logger.info(f"找到匹配的完整结果(相似度匹配-去除and): {text}")
                                                                            break
                                        except Exception as e:
                                            sherpa_logger.error(f"尝试查找完整结果时出错: {e}")

                                    # 格式化文本
                                    if len(text) > 0:
                                        text = text[0].upper() + text[1:]
                                    if text[-1] not in ['.', '?', '!']:
                                        text += '.'
                                    sherpa_logger.info(f"JSON解析失败，使用最后一个部分结果作为最终文本: {text}")
                                    self.signals.new_text.emit(text)
                    else:
                        # 对于其他模型，尝试调用FinalResult方法
                        if hasattr(recognizer, 'FinalResult'):
                            final_result = recognizer.FinalResult()
                            sherpa_logger.info(f"其他模型最终结果: {final_result}")

                            # 如果是字符串，尝试解析
                            if isinstance(final_result, str):
                                text = final_result.strip()
                                if text:
                                    sherpa_logger.info(f"发送最终文本: {text}")
                                    self.signals.new_text.emit(text)
                                elif hasattr(self.worker, '_last_partial_result') and self.worker._last_partial_result:
                                    # 如果最终结果为空但有最后一个部分结果，使用部分结果作为最终结果
                                    text = self.worker._last_partial_result
                                    # 记录使用的部分结果
                                    sherpa_logger.info(f"其他模型使用的最后一个部分结果原始值: {text}")

                                    # 检查是否是完整的句子（通过检查是否包含"and"等连接词判断）
                                    # 如果不是完整句子，可能是因为部分结果被截断了
                                    if text:
                                        # 尝试从完整结果中找到匹配的句子
                                        # 这里假设完整结果已经在UI中显示
                                        try:
                                            from src.ui.main_window import MainWindow
                                            if hasattr(MainWindow, 'instance') and MainWindow.instance:
                                                if hasattr(MainWindow.instance, 'subtitle_widget'):
                                                    subtitle_widget = MainWindow.instance.subtitle_widget
                                                    if hasattr(subtitle_widget, 'transcript_text') and subtitle_widget.transcript_text:
                                                        # 打印完整的transcript_text列表，便于调试
                                                        sherpa_logger.info(f"当前完整文本列表: {subtitle_widget.transcript_text}")

                                                        # 首先检查是否有以"and"结尾的部分结果
                                                        if " and " in text or text.endswith(" and"):
                                                            sherpa_logger.info(f"检测到以'and'结尾的部分结果: {text}")
                                                            # 特殊处理以"and"结尾的情况
                                                            for complete_text in reversed(subtitle_widget.transcript_text):
                                                                # 检查是否有包含相同前缀但更完整的句子
                                                                if complete_text.startswith(text.rstrip(" and")) and "and " in complete_text:
                                                                    text = complete_text
                                                                    sherpa_logger.info(f"找到匹配的完整结果(and特殊处理): {text}")
                                                                    break

                                                        # 如果没有找到匹配，继续使用常规匹配逻辑
                                                        if text.endswith(" and"):
                                                            # 查找最近的完整结果中是否包含当前部分结果
                                                            for complete_text in reversed(subtitle_widget.transcript_text):
                                                                # 检查部分结果是否是完整结果的前缀（去掉末尾的"and"）
                                                                prefix = text.rstrip(" and")
                                                                if prefix and complete_text.startswith(prefix):
                                                                    text = complete_text
                                                                    sherpa_logger.info(f"找到匹配的完整结果(前缀匹配-去除and): {text}")
                                                                    break

                                                        # 如果仍然没有找到匹配，使用常规匹配逻辑
                                                        if text.endswith(" and"):
                                                            for complete_text in reversed(subtitle_widget.transcript_text):
                                                                # 检查部分结果（去掉末尾的"and"）是否包含在完整结果中
                                                                prefix = text.rstrip(" and")
                                                                if prefix and prefix in complete_text:
                                                                    text = complete_text
                                                                    sherpa_logger.info(f"找到匹配的完整结果(子串匹配-去除and): {text}")
                                                                    break

                                                        # 如果仍然没有找到匹配，使用常规匹配逻辑
                                                        for complete_text in reversed(subtitle_widget.transcript_text):
                                                            # 检查部分结果是否是完整结果的前缀
                                                            if complete_text.startswith(text):
                                                                text = complete_text
                                                                sherpa_logger.info(f"找到匹配的完整结果(前缀匹配): {text}")
                                                                break
                                                            # 检查部分结果是否包含在完整结果中
                                                            elif text in complete_text:
                                                                text = complete_text
                                                                sherpa_logger.info(f"找到匹配的完整结果(子串匹配): {text}")
                                                                break
                                                            # 检查完整结果是否包含部分结果的大部分内容
                                                            elif len(text) > 10:  # 只对较长的部分结果进行相似度检查
                                                                # 计算部分结果的单词
                                                                partial_words = text.split()
                                                                # 计算完整结果的单词
                                                                complete_words = complete_text.split()
                                                                # 计算共同单词的数量
                                                                common_words = set(partial_words) & set(complete_words)
                                                                # 如果共同单词的数量超过部分结果单词数量的80%，认为匹配
                                                                if len(common_words) >= 0.8 * len(partial_words):
                                                                    text = complete_text
                                                                    sherpa_logger.info(f"找到匹配的完整结果(相似度匹配): {text}")
                                                                    break

                                                                # 如果部分结果以"and"结尾，特殊处理
                                                                if text.endswith(" and"):
                                                                    # 计算去掉"and"后的相似度
                                                                    partial_words_no_and = text.rstrip(" and").split()
                                                                    if partial_words_no_and:
                                                                        common_words_no_and = set(partial_words_no_and) & set(complete_words)
                                                                        if len(common_words_no_and) >= 0.8 * len(partial_words_no_and):
                                                                            text = complete_text
                                                                            sherpa_logger.info(f"找到匹配的完整结果(相似度匹配-去除and): {text}")
                                                                            break
                                        except Exception as e:
                                            sherpa_logger.error(f"尝试查找完整结果时出错: {e}")

                                    # 格式化文本
                                    if len(text) > 0:
                                        text = text[0].upper() + text[1:]
                                    if text[-1] not in ['.', '?', '!']:
                                        text += '.'
                                    sherpa_logger.info(f"使用最后一个部分结果作为最终文本: {text}")
                                    self.signals.new_text.emit(text)
                except Exception as e:
                    sherpa_logger.error(f"获取最终结果错误: {e}")
                    import traceback
                    sherpa_logger.error(traceback.format_exc())
        except Exception as e:
            sherpa_logger.error(f"处理最终结果时发生错误: {e}")
            import traceback
            sherpa_logger.error(traceback.format_exc())

        # 添加安全检查，防止访问已删除的对象
        try:
            # 首先设置worker的running标志为False
            if hasattr(self, 'worker') and self.worker:
                try:
                    self.worker.running = False
                except RuntimeError as e:
                    # 如果worker对象已被删除，记录错误但继续执行
                    sherpa_logger.warning(f"警告: 设置worker.running=False时出错: {e}")

            # 然后尝试停止线程
            if hasattr(self, 'worker_thread') and self.worker_thread:
                try:
                    # 检查线程是否仍在运行
                    if self.worker_thread.isRunning():
                        self.worker_thread.quit()
                        # 设置较短的超时时间，避免长时间等待
                        if not self.worker_thread.wait(1000):  # 等待最多1秒
                            sherpa_logger.warning("警告: 线程未能在1秒内停止")
                except RuntimeError as e:
                    # 如果线程对象已被删除，记录错误但继续执行
                    sherpa_logger.warning(f"警告: 停止线程时出错: {e}")
        except Exception as e:
            # 捕获所有异常，确保is_capturing标志被重置
            sherpa_logger.error(f"停止捕获时发生错误: {e}")
            import traceback
            sherpa_logger.error(traceback.format_exc())

        # 无论如何，确保捕获标志被重置
        self.is_capturing = False
        return True

    def _capture_audio_thread(self, recognizer: Any) -> None:
        """
        音频捕获线程（已废弃）

        此方法已废弃，请使用AudioWorker类和QThread代替。
        保留此方法仅为兼容性目的。

        Args:
            recognizer: 识别器实例
        """
        print("警告: _capture_audio_thread方法已废弃，请使用AudioWorker类和QThread")
        # 使用start_capture方法代替
        self.start_capture(recognizer)
        return

        # 以下代码保留但不会执行
        start_time = time.time()
        last_progress_update = time.time()

        try:
            # 获取音频设备
            with sc.get_microphone(id=str(self.current_device.id), include_loopback=True).recorder(samplerate=self.sample_rate) as mic:
                # 发送状态信号
                self.signals.status_updated.emit(f"正在从 {self.current_device.name} 捕获音频...")

                # 循环捕获音频
                while self.is_capturing:
                    # 更新进度条显示转录时长
                    current_time = time.time()
                    if current_time - last_progress_update >= 0.5:  # 每0.5秒更新一次
                        elapsed_seconds = current_time - start_time
                        minutes = int(elapsed_seconds // 60)
                        seconds = int(elapsed_seconds % 60)
                        time_str = f"转录时长: {minutes:02d}:{seconds:02d}"
                        self.signals.progress_updated.emit(50, time_str)  # 使用固定的50%进度
                        last_progress_update = current_time

                    # 捕获音频数据
                    data = mic.record(numframes=self.buffer_size)

                    # 转换为单声道
                    if data.shape[1] > 1:
                        data = np.mean(data, axis=1)

                    # 处理音频数据
                    try:
                        # 检查数据是否有效
                        if np.max(np.abs(data)) < 0.01:
                            print("DEBUG: 音频数据几乎是静音，跳过")
                            continue

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

                        # 检查当前引擎类型
                        engine_type = getattr(recognizer, 'engine_type', None)
                        if engine_type and engine_type.startswith('sherpa'):
                            # 对于 Sherpa-ONNX 模型，直接传递 numpy 数组
                            sherpa_logger.debug(f"使用 Sherpa-ONNX 模型，直接传递 numpy 数组")
                            sherpa_logger.debug(f"音频数据类型: {type(data)}, 形状: {data.shape}, 最大值: {np.max(np.abs(data))}")
                            accept_result = recognizer.AcceptWaveform(data)
                        else:
                            # 对于 Vosk 模型，转换为 16 位整数字节
                            data_bytes = (data * 32767).astype(np.int16).tobytes()
                            print(f"DEBUG: 处理音频数据，类型: {type(data_bytes)}, 长度: {len(data_bytes)}")
                            accept_result = recognizer.AcceptWaveform(data_bytes)
                        sherpa_logger.debug(f"AcceptWaveform 结果: {accept_result}")

                        if accept_result:
                            # 获取完整结果
                            result = recognizer.Result()
                            print(f"完整结果: {result}, 类型: {type(result)}, 引擎类型: {engine_type}")
                            sherpa_logger.info(f"完整结果: {result}, 类型: {type(result)}, 引擎类型: {engine_type}")

                            # 直接使用结果文本（针对 Sherpa-ONNX 模型）
                            if isinstance(result, str) and not result.startswith('{'):
                                # 如果是纯文本字符串（不是 JSON），直接使用
                                text = result.strip()
                                print(f"直接使用文本结果: {text}")
                                sherpa_logger.info(f"直接使用文本结果: {text}")
                            else:
                                # 尝试解析 JSON 或其他格式
                                try:
                                    if isinstance(result, str):
                                        result_json = json.loads(result)
                                    elif hasattr(result, 'text'):
                                        result_json = {"text": result.text}
                                    elif hasattr(result, '__str__'):
                                        result_json = {"text": str(result)}
                                    else:
                                        result_json = {"text": "无法解析的结果"}

                                    print(f"解析后的结果: {result_json}")
                                    sherpa_logger.info(f"解析后的结果: {result_json}")

                                    # 提取文本
                                    if 'text' in result_json and result_json['text'].strip():
                                        text = result_json['text'].strip()
                                    else:
                                        print(f"结果中没有文本或文本为空")
                                        sherpa_logger.warning(f"结果中没有文本或文本为空")
                                        continue
                                except Exception as e:
                                    print(f"解析结果错误: {e}")
                                    sherpa_logger.error(f"解析结果错误: {e}")
                                    import traceback
                                    error_trace = traceback.format_exc()
                                    sherpa_logger.error(error_trace)
                                    print(error_trace)
                                    # 如果解析失败，尝试直接使用结果
                                    text = str(result).strip()
                                    if not text:
                                        continue

                            # 格式化文本
                            if text:
                                if len(text) > 0:
                                    text = text[0].upper() + text[1:]
                                if text[-1] not in ['.', '?', '!']:
                                    text += '.'
                                print(f"DEBUG: 发送文本: {text}")
                                self.signals.new_text.emit(text)
                            else:
                                print(f"DEBUG: 文本为空，不发送")
                        else:
                            # 获取部分结果
                            partial_result = recognizer.PartialResult()
                            print(f"部分结果: {partial_result}, 类型: {type(partial_result)}, 引擎类型: {engine_type}")
                            sherpa_logger.info(f"部分结果: {partial_result}, 类型: {type(partial_result)}, 引擎类型: {engine_type}")

                            # 直接使用结果文本（针对 Sherpa-ONNX 模型）
                            if isinstance(partial_result, str) and not partial_result.startswith('{'):
                                # 如果是纯文本字符串（不是 JSON），直接使用
                                partial_text = partial_result.strip()
                                print(f"直接使用部分文本结果: {partial_text}")
                                sherpa_logger.info(f"直接使用部分文本结果: {partial_text}")
                                if partial_text:
                                    print(f"发送部分文本: {partial_text}")
                                    sherpa_logger.info(f"发送部分文本: {partial_text}")
                                    self.signals.new_text.emit("PARTIAL:" + partial_text)
                                else:
                                    print(f"部分文本为空，不发送")
                                    sherpa_logger.debug(f"部分文本为空，不发送")
                            else:
                                # 尝试解析 JSON 或其他格式
                                try:
                                    if isinstance(partial_result, str):
                                        try:
                                            partial = json.loads(partial_result)
                                        except json.JSONDecodeError:
                                            # 如果不是有效的 JSON，直接使用文本
                                            partial = {"partial": partial_result}
                                    elif isinstance(partial_result, dict):
                                        partial = partial_result
                                    elif hasattr(partial_result, 'partial'):
                                        partial = {'partial': str(partial_result.partial)}
                                    else:
                                        partial = {'partial': str(partial_result)}

                                    print(f"解析后的部分结果: {partial}")
                                    sherpa_logger.info(f"解析后的部分结果: {partial}")

                                    # 确保partial字段存在且有效
                                    partial_text = partial.get('partial', '').strip()
                                    if partial_text:
                                        print(f"发送部分文本: {partial_text}")
                                        sherpa_logger.info(f"发送部分文本: {partial_text}")
                                        self.signals.new_text.emit("PARTIAL:" + partial_text)
                                    else:
                                        print(f"部分结果中没有文本或文本为空")
                                        sherpa_logger.warning(f"部分结果中没有文本或文本为空")
                                except Exception as e:
                                    print(f"解析部分结果错误: {e}")
                                    sherpa_logger.error(f"解析部分结果错误: {e}")
                                    import traceback
                                    error_trace = traceback.format_exc()
                                    sherpa_logger.error(error_trace)
                                    print(error_trace)
                                    # 如果解析失败，尝试直接使用结果
                                    partial_text = str(partial_result).strip()
                                    if partial_text:
                                        print(f"发送部分文本: {partial_text}")
                                        sherpa_logger.info(f"发送部分文本: {partial_text}")
                                        self.signals.new_text.emit("PARTIAL:" + partial_text)
                                    else:
                                        print(f"部分文本为空，不发送")
                                        sherpa_logger.debug(f"部分文本为空，不发送")
                    except Exception as e:
                        error_msg = f"音频处理错误: {str(e)}"
                        self.signals.error_occurred.emit(error_msg)
                        print(f"错误: {error_msg}")
                        sherpa_logger.error(error_msg)
                        import traceback
                        error_trace = traceback.format_exc()
                        sherpa_logger.error(error_trace)
                        print(error_trace)

        except Exception as e:
            error_msg = f"音频捕获错误: {e}"
            self.signals.error_occurred.emit(error_msg)
            print(f"错误: {error_msg}")
            try:
                from src.utils.sherpa_logger import sherpa_logger
                sherpa_logger.error(error_msg)
                import traceback
                error_trace = traceback.format_exc()
                sherpa_logger.error(error_trace)
                print(error_trace)
            except ImportError:
                import traceback
                traceback.print_exc()

        finally:
            # 发送状态信号
            elapsed_time = time.time() - start_time
            self.signals.status_updated.emit(f"音频捕获已停止，持续时间: {elapsed_time:.2f}秒")

            # 确保捕获标志被清除
            self.is_capturing = False
