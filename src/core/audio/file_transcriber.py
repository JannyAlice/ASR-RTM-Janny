"""
文件转录模块
负责音频/视频文件的转录处理
"""
import os
import json
import time
import threading
import subprocess
import tempfile
import numpy as np
from typing import Any, Callable, Optional

from src.core.signals import TranscriptionSignals

class FileTranscriber:
    """文件转录器类"""

    def __init__(self, signals: TranscriptionSignals):
        """
        初始化文件转录器

        Args:
            signals: 信号实例
        """
        self.signals = signals
        self.is_transcribing = False
        self.transcription_thread = None
        self.temp_files = []  # 临时文件列表，用于清理
        self.ffmpeg_process = None

    def start_transcription(self, file_path: str, recognizer: Any) -> bool:
        """
        开始文件转录

        Args:
            file_path: 文件路径
            recognizer: 识别器实例

        Returns:
            bool: 开始转录是否成功
        """
        if self.is_transcribing:
            return False

        if not os.path.exists(file_path):
            self.signals.error_occurred.emit(f"文件不存在: {file_path}")
            return False

        # 设置转录标志
        self.is_transcribing = True

        # 获取文件信息
        try:
            # 获取文件大小
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

            # 获取文件时长
            probe = subprocess.run([
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                file_path
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            probe_data = json.loads(probe.stdout)
            duration = float(probe_data['format']['duration'])

            self.signals.status_updated.emit(
                f"文件信息: {os.path.basename(file_path)} ({file_size_mb:.2f}MB, {duration:.2f}秒)"
            )

            # 创建转录线程
            self.transcription_thread = threading.Thread(
                target=self._transcribe_file_thread,
                args=(file_path, recognizer, duration),
                daemon=True
            )

            # 启动线程
            self.transcription_thread.start()

            return True

        except Exception as e:
            self.signals.error_occurred.emit(f"获取文件信息错误: {e}")
            self.is_transcribing = False
            return False

    def stop_transcription(self) -> bool:
        """
        停止文件转录

        Returns:
            bool: 停止转录是否成功
        """
        if not self.is_transcribing:
            return False

        # 清除转录标志
        self.is_transcribing = False

        # 终止ffmpeg进程
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=1.0)
            except:
                try:
                    self.ffmpeg_process.kill()
                except:
                    pass
            self.ffmpeg_process = None

        # 等待线程结束
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=1.0)

        self.transcription_thread = None

        # 清理临时文件
        self._cleanup_temp_files()

        return True

    def _transcribe_file_thread(self, file_path: str, recognizer: Any, duration: float) -> None:
        """
        文件转录线程

        Args:
            file_path: 文件路径
            recognizer: 识别器实例
            duration: 文件时长(秒)
        """
        try:
            # 检查是否是 ASRModelManager 实例
            if hasattr(recognizer, 'transcribe_file'):
                # 使用 ASRModelManager 的 transcribe_file 方法
                self._transcribe_file_with_manager(file_path, recognizer, duration)
            else:
                # 使用传统的 Vosk 方法
                self._transcribe_file_with_vosk(file_path, recognizer, duration)

        except Exception as e:
            self.signals.error_occurred.emit(f"转录过程错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 清理临时文件
            self._cleanup_temp_files()

            # 清除转录标志
            self.is_transcribing = False

            # 发送完成信号
            self.signals.transcription_finished.emit()

    def _transcribe_file_with_manager(self, file_path: str, model_manager: Any, duration: float) -> None:
        """
        使用 ASRModelManager 转录文件

        Args:
            file_path: 文件路径
            model_manager: ASRModelManager 实例
            duration: 文件时长(秒)
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

        sherpa_logger.info(f"开始使用 ASRModelManager 转录文件: {file_path}")
        sherpa_logger.info(f"文件时长: {duration} 秒")

        # 检查 model_manager 是否有 transcribe_file 方法
        if not hasattr(model_manager, 'transcribe_file'):
            error_msg = "model_manager 没有 transcribe_file 方法"
            sherpa_logger.error(error_msg)
            self.signals.error_occurred.emit(error_msg)
            return

        # 检查 model_manager 的类型
        sherpa_logger.info(f"model_manager 类型: {type(model_manager)}")

        # 第一阶段：转换为WAV格式（0-20%）
        sherpa_logger.info("第一阶段：转换音频格式...")
        self.signals.status_updated.emit("第一阶段：转换音频格式...")
        self.signals.progress_updated.emit(10, "转换格式: 10%")

        # 第二阶段：转录文件（20-90%）
        sherpa_logger.info("第二阶段：转录文件...")
        self.signals.status_updated.emit("第二阶段：转录文件...")
        self.signals.progress_updated.emit(20, "转录中: 20%")

        # 使用 model_manager 的 transcribe_file 方法
        sherpa_logger.info(f"调用 model_manager.transcribe_file({file_path})")
        result = model_manager.transcribe_file(file_path)
        sherpa_logger.info(f"转录结果: {result[:100]}..." if result and len(result) > 100 else f"转录结果: {result}")

        # 第三阶段：处理结果（90-100%）
        sherpa_logger.info("第三阶段：处理结果...")
        self.signals.status_updated.emit("第三阶段：处理结果...")
        self.signals.progress_updated.emit(90, "处理结果: 90%")

        # 如果有结果，发送到字幕窗口
        if result:
            text = self._format_text(result)
            sherpa_logger.info(f"格式化后的文本: {text[:100]}..." if len(text) > 100 else f"格式化后的文本: {text}")
            self.signals.new_text.emit(text)
        else:
            sherpa_logger.warning("没有转录结果")

        # 转录完成，设置进度为 100%
        self.signals.progress_updated.emit(100, "转录完成 (100%)")
        self.signals.status_updated.emit("文件转录完成")
        sherpa_logger.info("文件转录完成")

    def _transcribe_file_with_vosk(self, file_path: str, recognizer: Any, duration: float) -> None:
        """
        使用 Vosk 转录文件

        Args:
            file_path: 文件路径
            recognizer: Vosk 识别器实例
            duration: 文件时长(秒)
        """
        # 第一阶段：转换为WAV格式（0-20%）
        self.signals.status_updated.emit("第一阶段：转换音频格式...")
        wav_file = self._convert_to_wav(file_path)
        if not wav_file or not self.is_transcribing:
            self.signals.transcription_finished.emit()
            return

        # 第二阶段：读取音频数据（20-50%）
        self.signals.status_updated.emit("第二阶段：读取音频数据...")
        all_chunks = []
        total_bytes = 0
        last_update_time = time.time()

        # 使用 ffmpeg 提取音频
        self.ffmpeg_process = subprocess.Popen([
            'ffmpeg',
            '-i', wav_file,
            '-ar', '16000',
            '-ac', '1',
            '-f', 's16le',
            '-'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 读取所有音频数据
        while self.is_transcribing:
            chunk = self.ffmpeg_process.stdout.read(4000)
            if not chunk:
                break

            all_chunks.append(chunk)
            total_bytes += len(chunk)

            # 更新读取进度（20-50%）
            current_time = time.time()
            if current_time - last_update_time >= 0.2:  # 每0.2秒更新一次
                current_position = total_bytes / (16000 * 2)  # 16kHz, 16-bit
                progress = 20 + min(30, int((current_position / duration) * 30))

                time_str = f"{int(current_position//60):02d}:{int(current_position%60):02d}"
                total_str = f"{int(duration//60):02d}:{int(duration%60):02d}"
                format_text = f"读取中: {time_str} / {total_str} ({progress}%)"

                self.signals.progress_updated.emit(progress, format_text)
                last_update_time = current_time

        # 确保 ffmpeg 进程终止
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            try:
                self.ffmpeg_process.wait(timeout=5)
            except:
                self.ffmpeg_process.kill()
            self.ffmpeg_process = None

        if not self.is_transcribing:
            self.signals.transcription_finished.emit()
            return

        # 第三阶段：处理音频数据（50-99%）
        self.signals.status_updated.emit(f"第三阶段：处理 {len(all_chunks)} 个音频块...")
        total_chunks = len(all_chunks)

        for i, chunk in enumerate(all_chunks):
            if not self.is_transcribing:
                break

            # 处理音频数据
            if recognizer.AcceptWaveform(chunk):
                result = json.loads(recognizer.Result())
                if result.get('text', '').strip():
                    text = self._format_text(result['text'])
                    self.signals.new_text.emit(text)

            # 更新处理进度（50-99%）
            current_time = time.time()
            if current_time - last_update_time >= 0.2:
                progress = 50 + min(49, int((i / total_chunks) * 49))
                format_text = f"处理中: {progress}%"
                self.signals.progress_updated.emit(progress, format_text)
                last_update_time = current_time

        # 处理最终结果
        final_result = json.loads(recognizer.FinalResult())
        if final_result.get('text', '').strip():
            text = self._format_text(final_result['text'])
            self.signals.new_text.emit(text)

            # 转录完成，设置进度为 100%
            self.signals.progress_updated.emit(100, "转录完成 (100%)")
            self.signals.status_updated.emit("文件转录完成")

    def _convert_to_wav(self, file_path: str) -> Optional[str]:
        """
        将文件转换为WAV格式

        Args:
            file_path: 原始文件路径

        Returns:
            str: WAV文件路径，如果转换失败则返回None
        """
        try:
            # 创建临时文件
            fd, temp_wav = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            self.temp_files.append(temp_wav)

            # 使用ffmpeg转换为WAV格式
            self.signals.status_updated.emit(f"正在转换文件格式...")
            self.signals.progress_updated.emit(5, "转换格式: 5%")

            # 使用ffmpeg转换
            self.ffmpeg_process = subprocess.Popen([
                'ffmpeg',
                '-i', file_path,
                '-ar', '16000',  # 采样率16kHz
                '-ac', '1',      # 单声道
                '-y',            # 覆盖已有文件
                temp_wav
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # 等待转换完成
            while self.ffmpeg_process.poll() is None:
                if not self.is_transcribing:
                    self.ffmpeg_process.terminate()
                    try:
                        self.ffmpeg_process.wait(timeout=1.0)
                    except:
                        self.ffmpeg_process.kill()
                    return None
                time.sleep(0.1)

            # 检查转换结果
            if self.ffmpeg_process.returncode != 0:
                stderr = self.ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                self.signals.error_occurred.emit(f"转换格式失败: {stderr}")
                return None

            self.signals.progress_updated.emit(20, "转换格式完成: 20%")
            return temp_wav

        except Exception as e:
            self.signals.error_occurred.emit(f"转换格式错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _cleanup_temp_files(self) -> None:
        """清理临时文件"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"清理临时文件错误: {e}")
        self.temp_files = []

    def _format_text(self, text: str) -> str:
        """
        格式化文本：添加标点、首字母大写等

        Args:
            text: 原始文本

        Returns:
            str: 格式化后的文本
        """
        if not text:
            return text

        # 首字母大写
        text = text[0].upper() + text[1:]

        # 如果文本末尾没有标点符号，添加句号
        if text[-1] not in ['.', '?', '!', ',', ';', ':', '-']:
            text += '.'

        # 处理常见的问句开头
        question_starters = ['what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'do', 'does', 'did', 'can', 'could', 'will', 'would']
        words = text.split()
        if words and words[0].lower() in question_starters:
            # 将句尾的句号替换为问号
            if text[-1] == '.':
                text = text[:-1] + '?'

        return text
