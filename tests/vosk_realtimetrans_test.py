#!/usr/bin/env python3
"""
Vosk 实时转录测试
使用 vosk small model 实现实时音频转录并在窗口中显示
"""
import os
import sys
import json
import time
import wave
import threading
import subprocess
import numpy as np
from queue import Queue, Empty
import soundcard as sc
import vosk
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                            QWidget, QPushButton, QProgressBar, QComboBox, QFileDialog,
                            QScrollArea, QFrame, QHBoxLayout, QMenu, QAction, QActionGroup)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject

# 设置 vosk 模型路径
MODEL_PATH = "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\models\\asr\\vosk\\vosk-model-small-en-us-0.15"

class TranscriptionSignals(QObject):
    """转录信号类"""
    new_text = pyqtSignal(str)
    progress_updated = pyqtSignal(int, str)
    transcription_finished = pyqtSignal()

class TranscriptionThread(QThread):
    """转录线程，负责音频捕获和处理"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = TranscriptionSignals()
        self.is_running = False
        self.transcript_text = []
        self.current_device = None
        self.file_path = None
        self.model = None

    def set_device(self, device):
        """设置音频设备"""
        self.current_device = device

    def set_file_path(self, file_path):
        """设置文件路径"""
        self.file_path = file_path

    def create_recognizer(self):
        """创建语音识别器"""
        try:
            if not self.model:
                # 加载模型
                if not os.path.exists(MODEL_PATH):
                    self.signals.new_text.emit(f"模型路径不存在: {MODEL_PATH}")
                    return None

                self.signals.new_text.emit(f"正在加载模型: {MODEL_PATH}")
                self.model = vosk.Model(MODEL_PATH)
                self.signals.new_text.emit("模型加载完成")

            # 创建识别器
            rec = vosk.KaldiRecognizer(self.model, 16000)
            rec.SetWords(True)  # 启用词级时间戳

            print("成功创建识别器")
            return rec

        except Exception as e:
            print(f"创建识别器错误: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def run(self):
        """线程主函数"""
        try:
            # 加载模型
            if not os.path.exists(MODEL_PATH):
                self.signals.new_text.emit(f"模型路径不存在: {MODEL_PATH}")
                return

            self.signals.new_text.emit(f"正在加载模型: {MODEL_PATH}")
            model = vosk.Model(MODEL_PATH)
            self.signals.new_text.emit("模型加载完成")

            # 创建识别器
            rec = vosk.KaldiRecognizer(model, 16000)

            # 获取音频设备
            if not self.current_device:
                # 如果没有指定设备，使用默认设备
                speakers = sc.all_speakers()
                if not speakers:
                    self.signals.new_text.emit("未找到音频设备")
                    return
                self.current_device = speakers[0]

            self.signals.new_text.emit(f"使用音频设备: {self.current_device.name}")

            # 设置采样率
            sample_rate = 16000

            # 开始捕获音频
            self.signals.new_text.emit("开始捕获音频...")
            self.is_running = True
            start_time = time.time()

            # 使用 soundcard 捕获音频
            with sc.get_microphone(id=str(self.current_device.id), include_loopback=True).recorder(samplerate=sample_rate) as mic:
                # 错误计数
                error_count = 0
                max_errors = 5

                while self.is_running:
                    try:
                        # 读取音频数据
                        data = mic.record(4000)

                        # 将数据转换为单声道
                        if data.shape[1] > 1:
                            data = data.mean(axis=1)

                        # 检查数据是否有效
                        if np.isnan(data).any() or np.isinf(data).any():
                            continue

                        # 检查是否有声音
                        if np.max(np.abs(data)) < 0.001:
                            continue

                        # 重置错误计数
                        error_count = 0

                        # 更新运行时间
                        elapsed_time = int(time.time() - start_time)
                        self.signals.progress_updated.emit(
                            elapsed_time % 100,
                            f"运行时间: {elapsed_time//60:02d}:{elapsed_time%60:02d}"
                        )

                        # 将浮点数据转换为16位整数
                        data_int16 = (data * 32767).astype(np.int16).tobytes()

                        # 处理音频数据
                        if rec.AcceptWaveform(data_int16):
                            result = json.loads(rec.Result())
                            if result.get('text', '').strip():
                                text = result['text'].strip()
                                self.transcript_text.append(text)
                                self.signals.new_text.emit(text)
                        else:
                            # 显示部分结果
                            partial = json.loads(rec.PartialResult())
                            if partial.get('partial', '').strip():
                                self.signals.new_text.emit("PARTIAL: " + partial['partial'].strip())

                    except Exception as e:
                        print(f"捕获音频错误: {e}")
                        error_count += 1
                        if error_count >= max_errors:
                            self.signals.new_text.emit(f"连续错误超过 {max_errors} 次，停止转录")
                            break
                        time.sleep(0.1)  # 等待一下再重试

            # 处理最后的结果
            final_result = json.loads(rec.FinalResult())
            if final_result.get('text', '').strip():
                text = final_result['text'].strip()
                if text not in self.transcript_text:
                    self.transcript_text.append(text)
                    self.signals.new_text.emit(text)

            # 发送完成信号
            self.signals.transcription_finished.emit()

        except Exception as e:
            self.signals.new_text.emit(f"转录错误: {e}")
            import traceback
            print(traceback.format_exc())
            self.signals.transcription_finished.emit()

    def process_audio_data(self, rec, data):
        """处理音频数据"""
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if 'text' in result and result['text'].strip():
                text = result['text'].strip()

                # 添加标点符号和首字母大写
                text = self.add_punctuation(text)

                # 只有当结果是完整句子时才添加到转录文本列表和显示
                if text not in self.transcript_text:
                    self.transcript_text.append(text)
                    # 更新显示
                    self.signals.new_text.emit(text)
        else:
            partial = json.loads(rec.PartialResult())
            if 'partial' in partial and partial['partial'].strip():
                # 只显示部分结果，不保存
                partial_text = partial['partial'].strip()
                # 对部分结果也应用标点符号处理
                partial_text = self.format_partial_text(partial_text)
                # 使用特殊标记表示这是部分结果
                self.signals.new_text.emit("PARTIAL:" + partial_text)

    def add_punctuation(self, text):
        """添加标点符号和首字母大写"""
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

    def format_partial_text(self, text):
        """格式化部分文本，不添加句尾标点"""
        if not text:
            return text

        # 首字母大写
        text = text[0].upper() + text[1:]

        return text

    def process_audio_data(self, rec, data):
        """处理音频数据"""
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if 'text' in result and result['text'].strip():
                text = result['text'].strip()

                # 添加标点符号和首字母大写
                text = self.add_punctuation(text)

                # 只有当结果是完整句子时才添加到转录文本列表和显示
                if text not in self.transcript_text:
                    self.transcript_text.append(text)
                    # 更新显示
                    self.signals.new_text.emit(text)
        else:
            partial = json.loads(rec.PartialResult())
            if 'partial' in partial and partial['partial'].strip():
                # 只显示部分结果，不保存
                partial_text = partial['partial'].strip()
                # 对部分结果也应用标点符号处理
                partial_text = self.format_partial_text(partial_text)
                # 使用特殊标记表示这是部分结果
                self.signals.new_text.emit("PARTIAL:" + partial_text)

    def add_punctuation(self, text):
        """添加标点符号和首字母大写"""
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

    def format_partial_text(self, text):
        """格式化部分文本，不添加句尾标点"""
        if not text:
            return text

        # 首字母大写
        text = text[0].upper() + text[1:]

        return text

    def process_audio_data(self, rec, data):
        """处理音频数据"""
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if 'text' in result and result['text'].strip():
                text = result['text'].strip()

                # 添加标点符号和首字母大写
                text = self.add_punctuation(text)

                # 只有当结果是完整句子时才添加到转录文本列表和显示
                if text not in self.transcript_text:
                    self.transcript_text.append(text)
                    # 更新显示
                    self.signals.new_text.emit(text)
        else:
            partial = json.loads(rec.PartialResult())
            if 'partial' in partial and partial['partial'].strip():
                # 只显示部分结果，不保存
                partial_text = partial['partial'].strip()
                # 对部分结果也应用标点符号处理
                partial_text = self.format_partial_text(partial_text)
                # 使用特殊标记表示这是部分结果
                self.signals.new_text.emit("PARTIAL:" + partial_text)

    def add_punctuation(self, text):
        """添加标点符号和首字母大写"""
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

    def format_partial_text(self, text):
        """格式化部分文本，不添加句尾标点"""
        if not text:
            return text

        # 首字母大写
        text = text[0].upper() + text[1:]

        return text

    def process_audio_data(self, rec, data):
        """处理音频数据"""
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if 'text' in result and result['text'].strip():
                text = result['text'].strip()

                # 添加标点符号和首字母大写
                text = self.add_punctuation(text)

                # 只有当结果是完整句子时才添加到转录文本列表和显示
                if text not in self.transcript_text:
                    self.transcript_text.append(text)
                    # 更新显示
                    self.signals.new_text.emit(text)
        else:
            partial = json.loads(rec.PartialResult())
            if 'partial' in partial and partial['partial'].strip():
                # 只显示部分结果，不保存
                partial_text = partial['partial'].strip()
                # 对部分结果也应用标点符号处理
                partial_text = self.format_partial_text(partial_text)
                # 使用特殊标记表示这是部分结果
                self.signals.new_text.emit("PARTIAL:" + partial_text)

    def add_punctuation(self, text):
        """添加标点符号和首字母大写"""
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

    def format_partial_text(self, text):
        """格式化部分文本，不添加句尾标点"""
        if not text:
            return text

        # 首字母大写
        text = text[0].upper() + text[1:]

        return text

    def stop(self):
        """停止转录"""
        self.is_running = False

class TranscriptionWindow(QMainWindow):
    """转录窗口，显示实时转录结果"""

    def __init__(self):
        super().__init__()

        # 初始化变量
        self.file_path = ""
        self.is_system_audio = True  # 默认使用系统音频

        # 设置窗口属性
        self.setWindowTitle("Vosk 实时转录测试")
        self.setGeometry(100, 100, 800, 600)

        # 创建中央部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 创建主布局
        self.main_layout = QVBoxLayout(self.central_widget)

        # 创建菜单栏
        self.create_menu_bar()

        # 创建控制区域
        self.control_layout = QHBoxLayout()

        # 创建开始/停止按钮
        self.start_button = QPushButton("开始转录")
        self.start_button.clicked.connect(self.toggle_transcription)
        self.control_layout.addWidget(self.start_button)

        # 创建进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("就绪")
        self.control_layout.addWidget(self.progress_bar)

        # 添加控制布局到主布局
        self.main_layout.addLayout(self.control_layout)

        # 创建滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)

        # 创建字幕容器
        self.subtitle_container = QWidget()
        self.subtitle_layout = QVBoxLayout(self.subtitle_container)

        # 创建字幕标签
        self.subtitle_label = QLabel("准备就绪，点击'开始转录'按钮开始...")
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.subtitle_label.setStyleSheet("""
            font-size: 16px;
            color: black;
            background-color: rgba(255, 255, 255, 180);
            padding: 10px;
            border-radius: 10px;
        """)

        # 添加字幕标签到字幕布局
        self.subtitle_layout.addWidget(self.subtitle_label)

        # 设置滚动区域的部件
        self.scroll_area.setWidget(self.subtitle_container)

        # 添加滚动区域到主布局
        self.main_layout.addWidget(self.scroll_area)

        # 初始化转录线程
        self.transcription_thread = None
        self.is_transcribing = False

        # 设置窗口样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QProgressBar {
                border: 1px solid #bbb;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
            }
        """)

    def toggle_transcription(self):
        """切换转录状态"""
        if self.is_transcribing:
            self.stop_transcription()
        else:
            self.start_transcription()

    def start_transcription(self):
        """开始转录"""
        if self.is_transcribing:
            return

        # 更新UI状态
        self.start_button.setText("停止转录")
        self.start_button.setStyleSheet("background-color: #f44336; color: white;")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("准备中...")
        self.subtitle_label.setText("正在初始化转录...")

        # 创建并启动转录线程
        self.transcription_thread = TranscriptionThread(self)

        # 根据模式处理
        if not self.is_system_audio:
            # 文件转录模式
            if not self.file_path or not os.path.exists(self.file_path):
                self.subtitle_label.setText("错误：请选择有效的音频/视频文件")
                self.stop_transcription()
                return

            # 显示文件信息
            self.subtitle_label.setText(f"已选择文件: {os.path.basename(self.file_path)}")
            # 文件处理功能尚未实现
            # 显示一个简单的消息框
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "文件选择",
                f"已选择文件: {os.path.basename(self.file_path)}\n\n文件处理功能尚未实现。"
            )
            self.stop_transcription()
            return
        else:
            # 系统音频模式
            # 获取音频设备
            speakers = sc.all_speakers()
            if speakers:
                self.transcription_thread.set_device(speakers[0])

        # 连接信号
        self.transcription_thread.signals.new_text.connect(self.update_text)
        self.transcription_thread.signals.progress_updated.connect(self.update_progress)
        self.transcription_thread.signals.transcription_finished.connect(self.on_transcription_finished)

        # 启动线程
        self.transcription_thread.start()

        self.is_transcribing = True

    def stop_transcription(self):
        """停止转录"""
        if not self.is_transcribing:
            return

        # 停止转录线程
        if self.transcription_thread and self.transcription_thread.isRunning():
            self.transcription_thread.stop()
            self.transcription_thread.wait()

        # 更新UI状态
        self.start_button.setText("开始转录")
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("已停止")

        self.is_transcribing = False

    def update_text(self, text):
        """更新转录文本"""
        if text.startswith("PARTIAL:"):
            # 显示部分结果，但不添加到历史记录
            partial_text = text[9:]  # 去掉 "PARTIAL: " 前缀
            current_text = self.subtitle_label.text()

            # 如果当前文本以 "PARTIAL: " 开头，则替换它
            if "PARTIAL: " in current_text:
                new_text = current_text.split("PARTIAL: ")[0] + "PARTIAL: " + partial_text
            else:
                new_text = current_text + "\nPARTIAL: " + partial_text

            self.subtitle_label.setText(new_text)
        else:
            # 显示完整结果，添加到历史记录
            current_text = self.subtitle_label.text()

            # 如果当前文本包含部分结果，则替换部分结果
            if "PARTIAL: " in current_text:
                new_text = current_text.split("PARTIAL: ")[0] + text
            else:
                new_text = current_text + "\n" + text

            self.subtitle_label.setText(new_text)

        # 滚动到底部
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )

    def update_progress(self, progress, format_text):
        """更新进度条"""
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(format_text)

    def on_transcription_finished(self):
        """转录完成处理"""
        self.stop_transcription()
        self.subtitle_label.setText(self.subtitle_label.text() + "\n\n转录已完成")

    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        menu = menubar.addMenu('转录模式(&T)')

        # 转录模式子菜单
        mode_menu = QMenu('转录模式', self)
        self.system_audio_action = QAction('系统音频', self, checkable=True)
        self.file_audio_action = QAction('音频/视频文件', self, checkable=True)
        self.system_audio_action.setChecked(True)
        mode_menu.addAction(self.system_audio_action)
        mode_menu.addAction(self.file_audio_action)

        # 将动作添加到动作组
        mode_group = QActionGroup(self)
        mode_group.addAction(self.system_audio_action)
        mode_group.addAction(self.file_audio_action)
        mode_group.setExclusive(True)

        # 添加子菜单到主菜单
        menu.addMenu(mode_menu)
        menu.addSeparator()

        # 添加文件选择动作
        select_file_action = QAction('选择文件...', self)
        select_file_action.triggered.connect(self.select_file)
        menu.addAction(select_file_action)

        # 连接信号
        self.system_audio_action.triggered.connect(lambda: self.switch_mode(True))
        self.file_audio_action.triggered.connect(lambda: self.switch_mode(False))

    def switch_mode(self, is_system_audio):
        """切换转录模式"""
        try:
            self.is_system_audio = is_system_audio
            if is_system_audio:
                self.subtitle_label.setText("已切换到系统音频模式，点击开始按钮开始转录")
            else:
                self.subtitle_label.setText("已切换到文件模式，请选择音频/视频文件")
                # 使用 QTimer 延迟调用文件选择对话框，避免菜单动作引起的问题
                QTimer.singleShot(100, self.select_file)
        except Exception as e:
            print(f"切换模式错误: {e}")
            self.subtitle_label.setText(f"切换模式错误: {e}")

    def select_file(self):
        """打开文件选择对话框"""
        try:
            # 使用 Windows 风格的文件对话框
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择音频/视频文件",
                "",  # 默认目录
                "媒体文件 (*.mp3 *.wav *.mp4 *.avi *.mkv *.mov);;所有文件 (*)",
                options=QFileDialog.DontUseNativeDialog
            )

            if file_path:
                self.file_path = file_path
                self.subtitle_label.setText(f"已选择文件: {os.path.basename(file_path)}")
                # 自动切换到文件转录模式
                self.is_system_audio = False
                self.file_audio_action.setChecked(True)
            else:
                # 如果用户取消选择，切换回系统音频模式
                self.is_system_audio = True
                self.system_audio_action.setChecked(True)

        except Exception as e:
            print(f"选择文件错误: {e}")
            # 发生错误时切换回系统音频模式
            self.is_system_audio = True
            self.system_audio_action.setChecked(True)

    def is_valid_media_file(self, file_path):
        """验证文件类型"""
        valid_extensions = ['.mp3', '.wav', '.mp4', '.avi', '.mkv', '.mov']
        return any(file_path.lower().endswith(ext) for ext in valid_extensions)

    def closeEvent(self, event):
        """窗口关闭事件处理"""
        # 确保在关闭窗口时停止转录
        self.stop_transcription()
        event.accept()

def main():
    """主函数"""
    # 检查模型路径
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型路径不存在: {MODEL_PATH}")
        print(f"请确保 vosk small model 已下载并放置在正确的位置")
        return

    # 创建应用
    app = QApplication(sys.argv)

    # 创建主窗口
    window = TranscriptionWindow()
    window.show()

    # 运行应用
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
