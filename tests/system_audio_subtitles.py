#!/usr/bin/env python3 
# 修正了临时文件的删除问题，添加了视频文件的WAV格式转换，>20M的视频文件会转换为WAV格式,解决进度条不准确的问题
# 修正了<20M的视频文件也转换为WAV格式，以及小文件转录生成重复文本的问题
import sys
import os
import json
import subprocess
import threading
from threading import Thread  # 添加 Thread 的显式导入
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, 
                            QPushButton, QSlider, QHBoxLayout, QComboBox, QFileDialog, 
                            QTabWidget, QRadioButton, QButtonGroup, QShortcut, QLineEdit, QProgressBar, QScrollArea, QMenuBar, QMenu, QAction, QActionGroup, QSizePolicy, QGraphicsOpacityEffect)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QFont, QColor, QKeySequence
import vosk
import soundcard as sc  # 只保留 Windows 的音频捕获库
import warnings
import ctypes
from ctypes import windll, c_int, byref
import psutil  # 导入 psutil 库用于检查系统资源
import gc  # 导入垃圾回收模块
import select
from queue import Queue, Empty
import wave

# 忽略警告
warnings.filterwarnings("ignore", message="data discontinuity in recording")

# 添加 Windows API 常量
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
LWA_ALPHA = 0x00000002

class TranscriptionSignals(QObject):
    new_text = pyqtSignal(str)
    progress_updated = pyqtSignal(int, str)  # 添加进度信号
    transcription_finished = pyqtSignal()    # 添加完成信号

class SubtitleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 设置窗口属性 - 使用 Qt.Window 而不是 Qt.Tool 以显示最小化和最大化按钮
        self.setWindowTitle("实时字幕")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Window)  # 使用 Qt.Window 代替 Qt.Tool
        
        # 初始化变量
        self.current_device = None  # 添加当前设备变量
        self.is_system_audio = True  # 默认使用系统音频
        self.file_path = ""
        self.audio_thread = None
        self.ffmpeg_process = None
        self.is_running = False
        self.transcript_text = []  # 存储转录文本
        self.output_file = None  # 输出文件路径
        
        # 模型相关变量
        self.model_language = 'en'  # 默认英文
        self.model_size = 'small'   # 默认小型模型
        self.model = None
        self.model_path_base = "model"  # 模型基础路径
        
        # 创建信号对象
        self.signals = TranscriptionSignals()
        self.signals.new_text.connect(self.update_subtitle)
        self.signals.progress_updated.connect(self.update_progress)
        self.signals.transcription_finished.connect(self.on_transcription_finished)
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建主窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 设置窗口背景
        self.central_widget.setStyleSheet("background-color: rgba(30, 30, 30, 255);")
        
        # 创建布局
        self.layout = QVBoxLayout(self.central_widget)
        
        # 创建滚动区域用于显示转录内容
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: rgba(50, 50, 50, 150);
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: rgba(100, 100, 100, 200);
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # 创建内容容器
        self.transcript_container = QWidget()
        self.transcript_layout = QVBoxLayout(self.transcript_container)
        
        # 创建字幕标签
        self.subtitle_label = QLabel("正在检查系统资源，请稍候...")
        self.subtitle_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.subtitle_label.setStyleSheet("color: red; background-color: rgba(0, 0, 0, 150); padding: 10px; border-radius: 10px;")
        self.subtitle_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setTextInteractionFlags(Qt.TextSelectableByMouse)  # 允许选择文本
        self.subtitle_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 允许标签扩展
        
        # 添加字幕标签到容器
        self.transcript_layout.addWidget(self.subtitle_label, 1)  # 使用拉伸因子1
        
        # 设置滚动区域的内容
        self.scroll_area.setWidget(self.transcript_container)
        
        # 创建底部控制面板
        self.bottom_panel = QWidget()
        self.bottom_layout = QHBoxLayout(self.bottom_panel)
        
        # 创建开始/停止按钮
        self.start_button = QPushButton("开始转录")
        self.start_button.clicked.connect(self.toggle_transcription)
        self.start_button.setStyleSheet("background-color: rgba(50, 150, 50, 200); color: white;")
        
        # 创建退出按钮
        self.exit_button = QPushButton("退出程序")
        self.exit_button.clicked.connect(self.force_quit)
        self.exit_button.setStyleSheet("background-color: rgba(150, 50, 50, 200); color: white;")
        
        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                background-color: rgba(50, 50, 50, 200);
                color: white;
            }
            QProgressBar::chunk {
                background-color: rgba(60, 100, 150, 200);
                width: 10px;
                margin: 0.5px;
            }
        """)
        self.progress_bar.hide()
        
        # 添加控件到底部面板
        self.bottom_layout.addWidget(self.start_button)
        self.bottom_layout.addWidget(self.exit_button)
        self.bottom_layout.addWidget(self.progress_bar)
        
        # 添加部件到主布局
        self.layout.addWidget(self.scroll_area, 1)
        self.layout.addWidget(self.bottom_panel, 0)
        
        # 设置窗口大小和位置
        self.resize(800, 400)  # 增加默认高度
        self.move(100, 300)
        
        # 设置键盘快捷键
        self.setup_shortcuts()
        
        # 设置默认文件路径
        default_file = r"C:\Users\crige\RealtimeTrans\RealT-whisper\debug_audio\mytest.mp4"
        if os.path.exists(default_file):
            self.file_path = default_file
            print(f"已加载默认文件: {default_file}")
        
        # 设置菜单栏样式 - 始终保持不透明
        self.menuBar().setStyleSheet("background-color: rgba(60, 60, 60, 255); color: white;")
        
        # 显示初始消息
        self.subtitle_label.setText("正在检查系统资源，请稍候...")
        
        # 使用更长的延迟来确保窗口已经显示
        QTimer.singleShot(1000, self.check_system_resources)
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        menu = menubar.addMenu('Menu')
        
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
        
        # 音频设备子菜单
        device_menu = QMenu('音频设备', self)
        self.populate_audio_devices_menu(device_menu)
        
        # 模型选择子菜单 - 新结构
        model_menu = QMenu('模型选择', self)
        
        # ASR 转录模型子菜单
        asr_menu = QMenu('ASR 转录模型', self)
        self.vosk_small_action = QAction('VOSK Small 模型', self, checkable=True)
        self.sherpa_int8_action = QAction('Sherpa-ONNX int8量化模型', self, checkable=True)
        self.sherpa_std_action = QAction('Sherpa-ONNX 标准模型', self, checkable=True)
        
        # RTM 翻译模型子菜单
        rtm_menu = QMenu('RTM 翻译模型', self)
        self.argos_action = QAction('Argostranslate 模型', self, checkable=True)
        self.opus_mt_action = QAction('Opus-Mt-ONNX 模型', self, checkable=True)
        
        # 默认选中 VOSK Small 模型
        self.vosk_small_action.setChecked(True)
        
        # 将所有模型动作添加到动作组
        model_group = QActionGroup(self)
        model_group.addAction(self.vosk_small_action)
        model_group.addAction(self.sherpa_int8_action)
        model_group.addAction(self.sherpa_std_action)
        model_group.addAction(self.argos_action)
        model_group.addAction(self.opus_mt_action)
        model_group.setExclusive(True)
        
        # 添加动作到对应子菜单
        asr_menu.addAction(self.vosk_small_action)
        asr_menu.addAction(self.sherpa_int8_action)
        asr_menu.addAction(self.sherpa_std_action)
        
        rtm_menu.addAction(self.argos_action)
        rtm_menu.addAction(self.opus_mt_action)
        
        # 添加子菜单到模型菜单
        model_menu.addMenu(asr_menu)
        model_menu.addMenu(rtm_menu)
        
        # 背景模式子菜单
        bg_menu = QMenu('背景模式', self)
        self.opaque_action = QAction('不透明', self, checkable=True)
        self.translucent_action = QAction('半透明', self, checkable=True)
        self.transparent_action = QAction('全透明', self, checkable=True)
        self.opaque_action.setChecked(True)
        
        bg_group = QActionGroup(self)
        bg_group.addAction(self.opaque_action)
        bg_group.addAction(self.translucent_action)
        bg_group.addAction(self.transparent_action)
        bg_group.setExclusive(True)
        
        bg_menu.addAction(self.opaque_action)
        bg_menu.addAction(self.translucent_action)
        bg_menu.addAction(self.transparent_action)
        
        # 连接信号
        self.system_audio_action.triggered.connect(lambda: self.switch_mode(True))
        self.file_audio_action.triggered.connect(lambda: self.switch_mode(False))
        self.opaque_action.triggered.connect(lambda: self.change_background_mode('opaque'))
        self.translucent_action.triggered.connect(lambda: self.change_background_mode('translucent'))
        self.transparent_action.triggered.connect(lambda: self.change_background_mode('transparent'))
        
        # 连接模型选择信号 - 保持原有的 change_model 方法
        self.vosk_small_action.triggered.connect(lambda: self.change_model('en', 'small'))
        # 其他模型暂时连接到相同的处理方法
        self.sherpa_int8_action.triggered.connect(lambda: self.change_model('en', 'small'))
        self.sherpa_std_action.triggered.connect(lambda: self.change_model('en', 'small'))
        self.argos_action.triggered.connect(lambda: self.change_model('en', 'small'))
        self.opus_mt_action.triggered.connect(lambda: self.change_model('en', 'small'))
        
        # 添加所有子菜单到主菜单
        menu.addMenu(mode_menu)
        menu.addMenu(device_menu)
        menu.addMenu(model_menu)
        menu.addMenu(bg_menu)
        menu.addSeparator()
        
        # 添加文件选择动作
        select_file_action = QAction('选择文件...', self)
        select_file_action.triggered.connect(self.select_file)
        menu.addAction(select_file_action)
        
        # 添加调试菜单
        debug_menu = QMenu('调试', self)
        search_docs_action = QAction('搜索模型文档', self)
        search_docs_action.triggered.connect(lambda: self.subtitle_label.setText(self.search_model_documentation()))
        debug_menu.addAction(search_docs_action)
        menu.addMenu(debug_menu)
    
    def populate_audio_devices_menu(self, menu):
        """填充音频设备菜单"""
        speakers = sc.all_speakers()
        device_group = QActionGroup(self)
        
        for speaker in speakers:
            action = QAction(speaker.name, self, checkable=True)
            action.setData(speaker)
            device_group.addAction(action)
            menu.addAction(action)
            
            if not self.current_device:
                action.setChecked(True)
                self.current_device = speaker
            
        device_group.setExclusive(True)
        device_group.triggered.connect(self.select_audio_device)
    
    def select_audio_device(self, action):
        """选择音频设备"""
        self.current_device = action.data()
        self.subtitle_label.setText(f"已选择音频设备: {self.current_device.name}")
    
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
    
    def setup_shortcuts(self):
        # 设置 ESC 键退出
        self.esc_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.esc_shortcut.activated.connect(self.force_quit)
        
        # 设置 Ctrl+Q 退出
        self.quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        self.quit_shortcut.activated.connect(self.force_quit)
        
        # 设置空格键开始/停止转录
        self.space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.space_shortcut.activated.connect(self.toggle_transcription)
    
    def load_model(self, keep_info=False):
        """加载语音识别模型"""
        # 在加载大型模型前强制垃圾回收
        if self.model_size == 'large':
            gc.collect()  # 强制垃圾回收
        
        # 根据语言和大小确定模型路径
        model_paths = {
            'en': {
                'small': "vosk-model-small-en-us-0.15",  # 40M
                'medium': "vosk-model-en-us-0.22-lgraph",  # 128M
                'large': "vosk-model-en-us-0.22"  # 1.8G
            },
            'cn': {
                'small': "vosk-model-small-cn-0.22",
                'medium': "vosk-model-cn-0.22",
                # 移除不存在的大型中文模型
                'large': "vosk-model-cn-0.22"  # 使用中型模型作为替代
            }
        }
        
        # 获取当前选择的模型路径
        model_name = model_paths.get(self.model_language, {}).get(self.model_size, "vosk-model-small-en-us-0.15")
        full_model_path = os.path.join(self.model_path_base, model_name)
        
        # 打印详细的模型信息
        print(f"尝试加载模型: {model_name}")
        print(f"完整路径: {full_model_path}")
        
        # 检查模型目录结构
        if os.path.exists(full_model_path):
            print(f"模型目录存在，检查内容:")
            for root, dirs, files in os.walk(full_model_path):
                level = root.replace(full_model_path, '').count(os.sep)
                indent = ' ' * 4 * level
                print(f"{indent}{os.path.basename(root)}/")
                sub_indent = ' ' * 4 * (level + 1)
                for f in files[:5]:  # 只显示前5个文件，避免输出过多
                    print(f"{sub_indent}{f}")
                if len(files) > 5:
                    print(f"{sub_indent}... 还有 {len(files)-5} 个文件")
        
        # 检查模型是否存在
        if not os.path.exists(full_model_path):
            print(f"警告: 模型 '{model_name}' 不存在，尝试使用默认模型")
            # 尝试使用默认模型目录
            full_model_path = self.model_path_base
            if not os.path.exists(full_model_path):
                error_msg = f"错误: 默认模型目录 '{full_model_path}' 也不存在"
                print(error_msg)
                
                if keep_info:
                    self.subtitle_label.setText(self.system_info + error_msg)
                else:
                    self.subtitle_label.setText(error_msg)
                    
                return False  # 返回加载失败标志
        
        # 检查系统资源
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 打印系统资源信息
        print(f"系统内存: 总计 {memory.total / (1024**3):.2f} GB, 可用 {memory.available / (1024**3):.2f} GB")
        print(f"磁盘空间: 总计 {disk.total / (1024**3):.2f} GB, 可用 {disk.free / (1024**3):.2f} GB")
        
        # 检查模型大小
        model_size_bytes = 0
        for root, dirs, files in os.walk(full_model_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    model_size_bytes += os.path.getsize(file_path)
        
        print(f"模型大小: {model_size_bytes / (1024**3):.2f} GB")
        
        # 检查是否有足够的内存
        if model_size_bytes > memory.available:
            error_msg = f"内存不足: 模型需要 {model_size_bytes / (1024**3):.2f} GB, 但可用内存只有 {memory.available / (1024**3):.2f} GB"
            print(error_msg)
            
            if keep_info:
                self.subtitle_label.setText(self.system_info + error_msg)
            else:
                self.subtitle_label.setText(error_msg)
            return False
        
        try:
            # 如果已经有模型加载，先释放资源
            if self.model:
                del self.model
                gc.collect()  # 强制垃圾回收
            
            print(f"开始加载模型: {model_name}")
            # 加载新模型
            self.model = vosk.Model(full_model_path)
            success_msg = f"成功加载模型: {model_name}"
            print(success_msg)
            
            # 检查模型是否正确初始化
            try:
                # 创建一个临时识别器来测试模型
                rec = vosk.KaldiRecognizer(self.model, 16000)
                print("模型测试成功: 能够创建识别器")
                del rec  # 释放临时识别器
            except Exception as e:
                print(f"模型测试失败: {e}")
            
            # 更新界面显示
            direction = 'CN' if self.model_language == 'en' else 'EN'
            model_info = f"已加载模型: {self.model_language.upper()}->{direction} ({self.model_size})\n模型文件: {model_name}"
            
            if keep_info:
                # 保留系统资源信息，并添加模型加载信息
                self.subtitle_label.setText(self.system_info + "\n" + model_info)
            else:
                self.subtitle_label.setText(model_info)
            
            return True  # 返回加载成功标志
        except Exception as e:
            error_msg = f"模型加载失败: {e}"
            print(error_msg)
            
            # 获取更详细的错误信息
            import traceback
            traceback_str = traceback.format_exc()
            print(f"详细错误信息:\n{traceback_str}")
            
            if keep_info:
                self.subtitle_label.setText(self.system_info + "\n" + error_msg)
            else:
                self.subtitle_label.setText(error_msg)
            
            self.model = None  # 确保模型为 None
            return False  # 返回加载失败标志
    
    def toggle_transcription(self):
        """切换转录状态"""
        if self.is_running:
            self.stop_transcription()
        else:
            # 清除之前的输出文件路径
            self.output_file = None
            self.start_transcription()
    
    def start_transcription(self):
        """开始转录"""
        # 检查是否已经在运行
        if self.is_running:
            return
        
        # 检查模型是否已加载
        if self.model is None:
            self.subtitle_label.setText("错误: 模型未加载或加载失败，请选择其他模型")
            return
        
        # 设置运行状态
        self.is_running = True
        
        # 更新按钮文本
        self.start_button.setText("停止转录")
        self.start_button.setStyleSheet("background-color: rgba(150, 50, 50, 200); color: white;")
        
        # 显示进度条
        self.progress_bar.show()
        
        # 创建输出文件
        if not self.is_system_audio:
            # 使用输入文件名作为基础
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        else:
            # 使用时间戳
            base_name = time.strftime("转录_%Y%m%d_%H%M%S")
        
        self.output_file = f"{base_name}_转录.txt"
        
        # 清空转录文本列表
        self.transcript_text = []
        
        # 根据模式启动相应的转录
        if not self.is_system_audio:
            # 文件转录模式
            if not self.file_path or not os.path.exists(self.file_path):
                self.signals.new_text.emit("错误：请选择有效的音频/视频文件")
                self.stop_transcription()
                return
            
            try:
                # 获取文件大小
                file_size_mb = os.path.getsize(self.file_path) / (1024 * 1024)  # MB
                
                # 获取文件时长
                probe = subprocess.run([
                    'ffprobe',
                    '-v', 'quiet',
                    '-print_format', 'json',
                    '-show_format',
                    self.file_path
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                probe_data = json.loads(probe.stdout)
                duration = float(probe_data['format']['duration'])
                print(f"文件总时长: {duration:.2f} 秒, 文件大小: {file_size_mb:.2f} MB")
                
                # 根据文件大小选择转录方法
                # 如果文件超过20MB，使用大文件处理方法
                large_file_threshold = 20  # MB
                
                if file_size_mb > large_file_threshold:
                    print(f"检测到大型文件: {file_size_mb:.2f} MB，使用增强模式处理")
                    self.signals.new_text.emit(f"检测到大型文件: {file_size_mb:.2f} MB，使用增强模式处理")
                    
                    # 启动大文件转录线程
                    self.audio_thread = threading.Thread(
                        target=self.transcribe_large_file,
                        args=(self.file_path, duration)
                    )
                else:
                    print(f"检测到普通文件: {file_size_mb:.2f} MB，使用标准模式处理")
                    # 启动标准文件转录线程
                    self.audio_thread = threading.Thread(
                        target=self.transcribe_file,
                        args=(self.file_path, duration)
                    )
                
                # 设置线程为守护线程并启动
                self.audio_thread.daemon = True
                self.audio_thread.start()
                
            except Exception as e:
                print(f"获取文件信息错误: {e}")
                self.signals.new_text.emit(f"获取文件信息错误: {e}")
                self.stop_transcription()
                return
        else:
            # 系统音频模式
            # 创建识别器
            rec = vosk.KaldiRecognizer(self.model, 16000)
            rec.SetWords(True)
            
            # 启动音频捕获线程
            self.audio_thread = threading.Thread(target=self.capture_audio, args=(rec,))
            self.audio_thread.daemon = True
            self.audio_thread.start()

    def stop_transcription(self):
        """停止转录并清理资源"""
        self.is_running = False
        
        if hasattr(self, 'audio_thread') and self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        
        if hasattr(self, 'ffmpeg_process') and self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                time.sleep(0.1)  # 给进程一点时间来终止
                self.ffmpeg_process.wait(timeout=1.0)
            except:
                try:
                    self.ffmpeg_process.kill()
                except:
                    pass
            finally:
                self.ffmpeg_process = None
        
        # 发送转录完成信号，这会触发 on_transcription_finished
        if hasattr(self, 'signals'):
            self.signals.transcription_finished.emit()
        
        # 使用 QTimer 延迟更新 UI
        QTimer.singleShot(100, self._update_ui_after_stop)
    
    def _update_ui_after_stop(self):
        """在停止转录后更新 UI（通过 QTimer 调用）"""
        try:
            # 更新按钮状态
            self.start_button.setText("开始转录")
            self.start_button.setStyleSheet("background-color: rgba(50, 150, 50, 200); color: white;")
        
            # 隐藏进度条
            self.progress_bar.hide()
            
            # 更新字幕标签
            current_text = self.subtitle_label.text()
            if not current_text.endswith("转录已完成"):
                self.subtitle_label.setText(current_text + "\n\n转录已完成")
        except Exception as e:
            print(f"更新 UI 错误: {e}")
    
    def capture_audio(self, rec):
        """捕获系统音频"""
        sample_rate = 16000  # 采样率
        self.capture_audio_windows(rec, sample_rate)
    
    def create_recognizer(self):
        """创建语音识别器"""
        try:
            if not self.model:
                print("错误: 模型未加载")
                return None
        
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

    def transcribe_file(self, file_path, duration):
        """转录文件 - 使用两阶段进度处理"""
        try:
            # 创建识别器
            rec = self.create_recognizer()
            if not rec:
                self.signals.new_text.emit("创建识别器失败")
                self.signals.transcription_finished.emit()
                return

            # 使用持久化的 WAV 文件，存放在 transcripts 目录下
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcripts")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            temp_wav = os.path.join(save_dir, f"{base_name}_audio.wav")
            
            # 检查是否已存在转换后的WAV文件，并验证其有效性
            wav_is_valid = False
            if os.path.exists(temp_wav):
                try:
                    with wave.open(temp_wav, 'rb') as wf:
                        if wf.getnchannels() == 1 and wf.getframerate() == 16000:
                            wav_is_valid = True
                            print(f"发现有效的WAV文件: {temp_wav}")
                            self.signals.new_text.emit("发现有效的WAV文件，直接使用...")
                except Exception as e:
                    print(f"WAV文件验证失败: {e}")
                    wav_is_valid = False
                    
                # 如果WAV文件无效，删除它
                if not wav_is_valid and os.path.exists(temp_wav):
                    try:
                        os.remove(temp_wav)
                        print(f"删除无效的WAV文件: {temp_wav}")
                    except Exception as e:
                        print(f"删除无效WAV文件失败: {e}")

            if not wav_is_valid:
                # 转换为WAV格式
                convert_cmd = [
                    'ffmpeg',
                    '-y',  # 强制覆盖
                    '-i', file_path,
                    '-ar', '16000',  # 采样率
                    '-ac', '1',      # 单声道
                    '-vn',           # 不处理视频
                    '-f', 'wav',
                    temp_wav
                ]
                
                # 显示转换开始信息
                self.signals.new_text.emit("正在转换音频格式，请稍候...")
                
                # 使用 subprocess 运行命令
                try:
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = subprocess.SW_HIDE
                    
                    process = subprocess.Popen(
                        convert_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        startupinfo=startupinfo,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                        text=True,
                        bufsize=4096
                    )
                    
                    # 读取进程输出
                    stdout, stderr = process.communicate()
                    
                    if process.returncode != 0:
                        raise RuntimeError(f"转换音频格式错误: {stderr}")
                    
                    # 检查输出文件
                    if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) == 0:
                        raise RuntimeError("转换后的WAV文件无效")
                    
                except Exception as e:
                    print(f"转换音频格式异常: {e}")
                    self.signals.new_text.emit(f"转换音频格式异常: {e}")
                    self.signals.transcription_finished.emit()
                    return
            
            if not self.is_running:
                print("转录被用户终止")
                self.signals.transcription_finished.emit()
                return
            
            # 处理WAV文件
            try:
                # 检查WAV文件是否存在
                if not os.path.exists(temp_wav):
                    print("WAV文件不存在")
                    self.signals.new_text.emit("WAV文件不存在")
                    self.signals.transcription_finished.emit()
                    return
                
                # 获取WAV文件信息
                with wave.open(temp_wav, 'rb') as wf:
                    n_channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                    framerate = wf.getframerate()
                    n_frames = wf.getnframes()
                    
                    print(f"WAV文件信息: 通道数={n_channels}, 采样宽度={sample_width}, "
                          f"采样率={framerate}, 总帧数={n_frames}")
                    
                    # 计算每块的帧数
                    chunk_frames = 8000
                    bytes_per_frame = n_channels * sample_width
                    total_frames_processed = 0
                    
                    # 处理音频数据
                    while self.is_running:
                        frames = wf.readframes(chunk_frames)
                        if not frames:
                            break
                        
                        frames_read = len(frames) // bytes_per_frame
                        total_frames_processed += frames_read
                        
                        # 更新进度（0-90%）
                        progress = int(90 * (total_frames_processed / n_frames))
                        time_position = total_frames_processed / framerate
                        time_str = f"{int(time_position//60):02d}:{int(time_position%60):02d}"
                        total_str = f"{int(duration//60):02d}:{int(duration%60):02d}"
                        format_text = f"处理中: {time_str} / {total_str} ({progress}%)"
                        
                        self.signals.progress_updated.emit(progress, format_text)
                        
                        # 处理音频数据
                        if rec.AcceptWaveform(frames):
                            result = json.loads(rec.Result())
                            if result.get('text', '').strip():
                                text = result['text'].strip()
                                text = self.add_punctuation(text)
                                if text not in self.transcript_text:
                                    self.transcript_text.append(text)
                                    self.signals.new_text.emit(text)
                        else:
                            partial = json.loads(rec.PartialResult())
                            if partial.get('partial', '').strip():
                                self.signals.new_text.emit("PARTIAL:" + partial['partial'].strip())
                
                # 处理最后的结果
                final_result = json.loads(rec.FinalResult())
                if final_result.get('text', '').strip():
                    text = final_result['text'].strip()
                    text = self.add_punctuation(text)
                    if text not in self.transcript_text:
                        self.transcript_text.append(text)
                        self.signals.new_text.emit(text)
                
                # 更新进度到100%
                self.signals.progress_updated.emit(100, "转录完成 (100%)")
                
                # 保存转录结果 - 直接调用 on_transcription_finished
                self.signals.transcription_finished.emit()
                
                # 提示WAV文件位置
                print(f"WAV文件已保存在: {temp_wav}")
                self.signals.new_text.emit(f"\nWAV文件已保存在: {temp_wav}")
                
            except Exception as e:
                print(f"处理WAV文件错误: {e}")
                self.signals.new_text.emit(f"处理WAV文件错误: {e}")
                import traceback
                print(traceback.format_exc())
            
        except Exception as e:
            print(f"转录过程错误: {e}")
            self.signals.new_text.emit(f"转录错误: {e}")
            import traceback
            print(traceback.format_exc())
        
        finally:
            # 发送完成信号
            self.signals.transcription_finished.emit()
    
    def capture_audio_windows(self, rec, sample_rate):
        """捕获 Windows 系统音频"""
        start_time = time.time()
        
        if not self.current_device:
            self.signals.new_text.emit("错误：未选择音频设备")
            return
        
        try:
            with sc.get_microphone(id=str(self.current_device.id), include_loopback=True).recorder(samplerate=sample_rate) as mic:
                # 错误计数
                error_count = 0
                max_errors = 5
                
                while self.is_running:
                    try:
                        data = mic.record(numframes=1024)
                        # 转换为单声道
                        data = data.mean(axis=1)
                        
                        # 检查数据是否有效
                        if np.isnan(data).any() or np.isinf(data).any():
                            continue
                        
                        # 检查是否有声音
                        if np.max(np.abs(data)) < 0.001:
                            continue
                        
                        # 转换为 16 位整数
                        data = (data * 32767).astype(np.int16).tobytes()
                        
                        self.process_audio_data(rec, data)
                        error_count = 0  # 重置错误计数
                        
                        # 更新运行时间
                        elapsed_time = int(time.time() - start_time)
                        self.signals.progress_updated.emit(
                            elapsed_time % 100, 
                            f"运行时间: {elapsed_time//60:02d}:{elapsed_time%60:02d}"
                        )
                        
                    except Exception as e:
                        error_count += 1
                        print(f"音频处理错误: {e}")
                        if error_count >= max_errors:
                            self.signals.new_text.emit(f"音频捕获多次失败，请尝试重新启动程序")
                            break
                        time.sleep(0.1)  # 短暂暂停
                
        except Exception as e:
            self.signals.new_text.emit(f"音频捕获错误: {e}")
        finally:
            # 发送完成信号
            self.signals.transcription_finished.emit()
    
    def process_audio_data(self, rec, data):
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

    def update_subtitle(self, text):
        """更新字幕显示（在主线程中调用）"""
        try:
            # 处理部分结果和完整结果
            if text.startswith("PARTIAL:"):
                # 部分结果只显示，不添加到转录文本列表
                partial_text = text[8:]  # 去掉 "PARTIAL:" 前缀
                
                # 显示最近的完整结果加上当前的部分结果
                display_text = self.transcript_text[-9:] if self.transcript_text else []
                display_text.append(partial_text)
                
                # 更新字幕标签
                self.subtitle_label.setText('\n'.join(display_text))
            else:
                # 完整结果 - 检查是否与最后一个结果相同或相似
                if self.transcript_text and (text == self.transcript_text[-1] or self._is_similar(text, self.transcript_text[-1])):
                    # 如果是重复或非常相似的文本，不添加到列表
                    print(f"跳过重复文本: {text}")
                else:
                    # 添加新的完整结果到转录文本列表
                    self.transcript_text.append(text)
                
                # 只显示最近的几条完整结果
                self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
            
            # 滚动到底部
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            )
            
        except Exception as e:
            print(f"更新字幕错误: {e}")
    
    def _is_similar(self, text1, text2):
        """检查两段文本是否非常相似（用于去除近似重复）"""
        # 如果两段文本长度相差太大，认为不相似
        if abs(len(text1) - len(text2)) > 10:
            return False
        
        # 计算相似度 - 使用简单的字符匹配
        # 这里使用一个简单的算法，可以根据需要改进
        common_chars = sum(1 for c1, c2 in zip(text1, text2) if c1 == c2)
        similarity = common_chars / max(len(text1), len(text2))
        
        # 如果相似度超过 80%，认为是相似文本
        return similarity > 0.8
    
    def change_background_mode(self, mode):
        """切换背景模式"""
        try:
            # 保存当前位置
            current_pos = self.pos()
            
            # 显示切换提示
            mode_names = {
                'opaque': '不透明',
                'translucent': '半透明',
                'transparent': '全透明'
            }
            self.subtitle_label.setText(f"已切换到{mode_names.get(mode, '')}背景模式")
            
            # 先隐藏窗口
            self.hide()
            
            # 获取窗口句柄
            hwnd = int(self.winId())
            
            # 标题栏始终保持不透明 - 设置窗口标志
            self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Window)  # 使用 Qt.Window 代替 Qt.Tool
            
            # 设置菜单栏样式 - 始终保持不透明
            self.menuBar().setStyleSheet("background-color: rgba(60, 60, 60, 255); color: white;")
            
            if mode == 'opaque':
                # 不透明模式
                self.setAttribute(Qt.WA_TranslucentBackground, False)
                self.central_widget.setStyleSheet("background-color: rgba(30, 30, 30, 255);")
                self.subtitle_label.setStyleSheet("color: red; background-color: rgba(0, 0, 0, 150); padding: 10px; border-radius: 10px;")
                
                # 移除 WS_EX_LAYERED 样式
                current_style = windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, current_style & ~WS_EX_LAYERED)
                
            elif mode == 'translucent':
                # 半透明模式 - 使用白色背景
                self.setAttribute(Qt.WA_TranslucentBackground, False)
                
                # 设置白色背景
                self.central_widget.setStyleSheet("background-color: rgba(255, 255, 255, 200);")
                self.subtitle_label.setStyleSheet("color: black; background-color: rgba(255, 255, 255, 180); padding: 10px; border-radius: 10px;")
                
                # 设置 WS_EX_LAYERED 样式，但只应用于内容区域
                current_style = windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, current_style | WS_EX_LAYERED)
                
                # 设置透明度为 200 (0-255)，但菜单栏和标题栏保持不透明
                windll.user32.SetLayeredWindowAttributes(hwnd, 0, 200, LWA_ALPHA)
                
            else:  # transparent
                # 全透明模式 - 使用白色背景
                self.setAttribute(Qt.WA_TranslucentBackground, True)
                
                # 设置白色背景，但透明度更高
                self.central_widget.setStyleSheet("background-color: rgba(255, 255, 255, 100);")
                self.subtitle_label.setStyleSheet("color: black; background-color: rgba(255, 255, 255, 150); padding: 10px; border-radius: 10px;")
                
                # 设置 WS_EX_LAYERED 样式
                current_style = windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, current_style | WS_EX_LAYERED)
                
                # 设置透明度为 150 (半透明)
                windll.user32.SetLayeredWindowAttributes(hwnd, 0, 150, LWA_ALPHA)
            
            # 重新显示窗口并恢复位置
            self.show()
            self.move(current_pos)
            
            # 强制重绘
            self.repaint()
            
        except Exception as e:
            print(f"切换背景模式错误: {e}")
            self.subtitle_label.setText(f"切换背景模式错误: {e}")
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
            event.accept()
    
    def closeEvent(self, event):
        # 确保在关闭窗口时停止转录
        self.stop_transcription()
        print("程序已退出")
        event.accept()
        
        # 使用计时器延迟退出，确保事件循环有时间处理关闭事件
        QTimer.singleShot(200, lambda: os._exit(0))
    
    def force_quit(self):
        print("正在强制退出程序...")
        self.stop_transcription()
        QApplication.quit()
        import os
        os._exit(0)

    def update_file_path(self, text):
        if os.path.isfile(text):
            if self.is_valid_media_file(text):
                self.file_path = text
                self.subtitle_label.setText(f"已选择文件: {os.path.basename(text)}")
            else:
                self.subtitle_label.setText("不支持的文件类型")

    # 添加拖放事件处理方法
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            file_path = str(urls[0].toLocalFile())
            print(f"拖放文件: {file_path}")
            
            # 检查文件类型
            if self.is_valid_media_file(file_path):
                # 自动切换到文件模式
                self.file_audio_radio.setChecked(True)
                
                # 设置文件路径
                self.file_path = file_path
                self.file_input.setText(file_path)
                self.subtitle_label.setText(f"已选择文件: {os.path.basename(file_path)}")
            else:
                self.subtitle_label.setText("不支持的文件类型")

    # 添加文件类型验证方法
    def is_valid_media_file(self, file_path):
        valid_extensions = ['.mp3', '.wav', '.mp4', '.avi', '.mkv', '.mov']
        return any(file_path.lower().endswith(ext) for ext in valid_extensions)

    def save_transcript(self):
        """保存转录文本到文件"""
        try:
            if not self.transcript_text:
                self.subtitle_label.setText("没有可保存的转录内容")
                return
                
            # 创建保存目录
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcripts")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 获取当前时间作为文件名的一部分
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # 添加模型信息到文件名
            model_info = f"{self.model_language.upper()}-{self.model_size}"
            
            # 确定文件名 - 不再生成临时文件
            if self.is_system_audio:
                # 系统音频模式
                filename = f"online_transcript_{model_info}_{timestamp}.txt"
            else:
                # 文件模式
                base_name = os.path.splitext(os.path.basename(self.file_path))[0]
                filename = f"transcript_{base_name}_{model_info}_{timestamp}.txt"
            
            # 完整的保存路径
            save_path = os.path.join(save_dir, filename)
            
            # 保存文件
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.transcript_text))
            
            # 更新界面显示
            self.subtitle_label.setText(f"转录文本已保存到: {save_path}")
            
            # 保存路径用于后续操作
            self.output_file = save_path
            
            # 打印保存信息
            print(f"转录文本已保存到: {save_path}")
            
        except Exception as e:
            print(f"保存转录文本错误: {e}")
            self.subtitle_label.setText(f"保存转录文本错误: {e}")

    def update_progress(self, progress, format_text):
        """更新进度条（在主线程中调用）"""
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(format_text)

    def on_transcription_finished(self):
        """转录完成时的处理"""
        try:
            if not self.transcript_text:
                self.subtitle_label.setText("没有可保存的转录内容")
                return

            # 创建保存目录
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcripts")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 生成最终文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_info = f"{self.model_language.upper()}-{self.model_size}"
            
            if self.is_system_audio:
                # 在线转录模式 - 保存最终文件
                final_filename = f"transcript_{model_info}_{timestamp}.txt"
            else:
                # 文件转录模式 - 保存为新文件
                base_name = os.path.splitext(os.path.basename(self.file_path))[0]
                final_filename = f"transcript_{base_name}_{model_info}_{timestamp}.txt"
            
            final_output_file = os.path.join(save_dir, final_filename)

            # 检查是否已经有相同内容的文件
            content_to_save = '\n'.join(self.transcript_text)
            should_save = True
            
            # 检查最近的文件内容
            for file in sorted(os.listdir(save_dir), reverse=True):
                if file.startswith("transcript_") and file.endswith(".txt"):
                    try:
                        with open(os.path.join(save_dir, file), 'r', encoding='utf-8') as f:
                            existing_content = f.read()
                            if existing_content == content_to_save:
                                print(f"发现相同内容的文件: {file}，跳过保存")
                                should_save = False
                                final_output_file = os.path.join(save_dir, file)
                                break
                    except Exception as e:
                        print(f"检查文件内容错误 {file}: {e}")

            # 只有在内容不同时才保存新文件
            if should_save:
                with open(final_output_file, 'w', encoding='utf-8') as f:
                    f.write(content_to_save)

            # 只删除临时文件，保留所有旧文件
            temp_files_deleted = []
            for file in os.listdir(save_dir):
                # 只删除临时文件（包含 _temp 或 online_transcript_ 的文件）
                if "_temp" in file or file.startswith("online_transcript_"):
                    try:
                        file_path = os.path.join(save_dir, file)
                        os.remove(file_path)
                        temp_files_deleted.append(file)
                    except Exception as e:
                        print(f"删除临时文件失败 {file}: {e}")

            # 更新界面显示
            status_message = f"转录完成，文件已保存到: {final_output_file}"
            if temp_files_deleted:
                status_message += f"\n已删除 {len(temp_files_deleted)} 个临时文件"
                print(f"已删除的临时文件: {', '.join(temp_files_deleted)}")
            
            self.subtitle_label.setText(status_message)
            print(f"转录完成，文件已保存到: {final_output_file}")

        except Exception as e:
            print(f"保存最终转录文本错误: {e}")
            self.subtitle_label.setText(f"保存最终转录文本错误: {e}")
            
        finally:
            # 重置状态
            self.is_running = False
            self.update_ui_state()

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

    def change_model(self, language, size):
        """切换语音识别模型"""
        # 如果正在转录，先停止
        if self.is_running:
            self.stop_transcription()
        
        # 更新模型设置
        self.model_language = language
        self.model_size = size
        
        # 显示正在切换模型的提示
        self.subtitle_label.setText(f"正在切换到 {language.upper()}->{('CN' if language == 'en' else 'EN')} ({size}) 模型...")
        
        # 使用 QTimer 延迟加载模型，让界面先更新
        QTimer.singleShot(100, self._load_model_with_check)

    def _load_model_with_check(self):
        """加载模型并检查结果"""
        success = self.load_model()
        if not success:
            # 如果加载失败，尝试回退到小型模型
            if self.model_size != 'small':
                self.subtitle_label.setText(f"模型加载失败，尝试回退到小型模型...")
                self.model_size = 'small'
                QTimer.singleShot(100, self.load_model)

    def check_system_resources(self):
        """检查系统资源并给出建议"""
        try:
            # 先显示正在检查的消息
            self.subtitle_label.setText("正在检查系统资源，请稍候...")
            QApplication.processEvents()  # 强制处理事件，更新界面
            
            # 获取系统资源信息
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # 计算模型大小
            model_sizes = {
                'small': 0.07,  # GB
                'medium': 0.5,  # GB
                'large': 2.67   # GB
            }
            
            # 打印系统资源信息
            print(f"系统内存: 总计 {memory.total / (1024**3):.2f} GB, 可用 {memory.available / (1024**3):.2f} GB")
            print(f"磁盘空间: 总计 {disk.total / (1024**3):.2f} GB, 可用 {disk.free / (1024**3):.2f} GB")
            
            # 确定可以使用的最大模型
            available_models = []
            for size, model_size_gb in model_sizes.items():
                if model_size_gb < memory.available / (1024**3):
                    available_models.append(size)
            
            # 生成建议信息
            info_text = f"系统资源信息:\n\n"
            info_text += f"内存: 总计 {memory.total / (1024**3):.2f} GB, 可用 {memory.available / (1024**3):.2f} GB\n"
            info_text += f"磁盘空间: 总计 {disk.total / (1024**3):.2f} GB, 可用 {disk.free / (1024**3):.2f} GB\n\n"
            
            info_text += "模型大小参考:\n"
            info_text += "- 小型模型: 约 0.07 GB 内存\n"
            info_text += "- 中型模型: 约 0.5 GB 内存\n"
            info_text += "- 大型模型: 约 2.67 GB 内存\n\n"
            
            if 'large' in available_models:
                info_text += "✅ 您的系统内存充足，可以使用任何大小的模型。\n"
                info_text += "建议: 可以尝试大型模型以获得最佳识别效果。\n\n"
            elif 'medium' in available_models:
                info_text += "⚠️ 您的系统内存不足以加载大型模型。\n"
                info_text += "建议: 使用中型模型以获得较好的识别效果。\n\n"
            else:
                info_text += "❌ 您的系统内存较低，只能使用小型模型。\n"
                info_text += "建议: 关闭其他应用程序以释放更多内存，或增加系统内存。\n\n"
            
            # 保存系统资源信息
            self.system_info = info_text
            
            # 显示系统资源信息和建议
            self.subtitle_label.setText(info_text + "正在加载默认模型，请稍候...")
            QApplication.processEvents()  # 强制处理事件，更新界面
            
            # 延迟加载模型，让用户有时间阅读系统资源信息
            QTimer.singleShot(3000, self.load_model_and_keep_info)
            
        except Exception as e:
            print(f"检查系统资源错误: {e}")
            self.subtitle_label.setText(f"检查系统资源错误: {e}")
            
            # 出错时仍然加载模型，但延迟一下
            QTimer.singleShot(1000, self.load_model)

    def load_model_and_keep_info(self):
        """加载模型并保留系统资源信息"""
        # 加载模型
        success = self.load_model(keep_info=True)
        
        if not success:
            # 如果加载失败，尝试回退到小型模型
            if self.model_size != 'small':
                self.subtitle_label.setText(self.system_info + "模型加载失败，尝试回退到小型模型...")
                self.model_size = 'small'
                QTimer.singleShot(1000, lambda: self.load_model(keep_info=True))

    def search_model_documentation(self):
        """搜索项目中关于模型使用的说明文件并保存结果到文本文件"""
        try:
            # 创建一个字符串缓冲区来收集所有输出
            output_buffer = []
            
            # 添加时间戳
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            output_buffer.append(f"模型检查报告 - {timestamp}\n")
            output_buffer.append("="*50 + "\n")
            
            # 打印当前工作目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_buffer.append(f"当前工作目录: {current_dir}\n")
            
            # 搜索的目录列表
            search_dirs = [
                current_dir,  # 当前目录
                os.path.join(current_dir, "model"),  # 模型目录
                os.path.join(current_dir, "docs"),  # 可能的文档目录
                os.path.join(current_dir, "..")  # 上级目录
            ]
            
            # 可能的文档文件名
            doc_filenames = [
                "README.md", "README.txt", "readme.md", "readme.txt",
                "MODELS.md", "MODELS.txt", "models.md", "models.txt",
                "USAGE.md", "USAGE.txt", "usage.md", "usage.txt",
                "DOCUMENTATION.md", "DOCUMENTATION.txt", "documentation.md", "documentation.txt"
            ]
            
            # 搜索模型目录中的 README 文件
            model_dir = os.path.join(current_dir, "model")
            if os.path.exists(model_dir):
                output_buffer.append(f"\n模型目录存在: {model_dir}\n")
                output_buffer.append("模型目录内容:\n")
                
                # 列出模型目录中的所有文件夹
                model_folders = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]
                output_buffer.append(f"模型文件夹: {model_folders}\n")
                
                # 检查每个模型文件夹中的 README 文件
                for model_folder in model_folders:
                    model_path = os.path.join(model_dir, model_folder)
                    output_buffer.append(f"\n检查模型: {model_folder}\n")
                    
                    # 检查模型文件夹中的 README 文件
                    for doc_filename in doc_filenames:
                        doc_path = os.path.join(model_path, doc_filename)
                        if os.path.exists(doc_path):
                            output_buffer.append(f"找到模型文档: {doc_path}\n")
                            try:
                                with open(doc_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    output_buffer.append(f"文档内容:\n{content[:1000]}...\n")
                            except Exception as e:
                                output_buffer.append(f"读取文档错误: {e}\n")
            
            # 检查小型模型的具体结构
            output_buffer.append("\n小型模型检查:\n")
            small_model_path = os.path.join(model_dir, "vosk-model-small-en-us-0.15")
            if os.path.exists(small_model_path):
                output_buffer.append(f"小型模型路径存在: {small_model_path}\n")
                output_buffer.append("小型模型目录结构:\n")
                for root, dirs, files in os.walk(small_model_path):
                    level = root.replace(small_model_path, '').count(os.sep)
                    indent = ' ' * 4 * level
                    output_buffer.append(f"{indent}{os.path.basename(root)}/\n")
                    sub_indent = ' ' * 4 * (level + 1)
                    for f in files:
                        file_path = os.path.join(root, f)
                        file_size = os.path.getsize(file_path) / (1024*1024)
                        output_buffer.append(f"{sub_indent}{f} ({file_size:.2f} MB)\n")
            else:
                output_buffer.append(f"小型模型路径不存在: {small_model_path}\n")
            
            # 检查系统信息
            output_buffer.append("\n系统信息:\n")
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            output_buffer.append(f"系统内存: 总计 {memory.total / (1024**3):.2f} GB, 可用 {memory.available / (1024**3):.2f} GB\n")
            output_buffer.append(f"磁盘空间: 总计 {disk.total / (1024**3):.2f} GB, 可用 {disk.free / (1024**3):.2f} GB\n")
            
            # 检查 Python 和库版本
            output_buffer.append("\nPython和库版本:\n")
            output_buffer.append(f"Python版本: {sys.version}\n")
            try:
                output_buffer.append(f"Vosk版本: {vosk.__version__}\n")
            except:
                output_buffer.append("无法获取Vosk版本\n")
            
            # 将结果保存到文件
            report_dir = os.path.join(current_dir, "reports")
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            
            report_filename = f"model_check_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            report_path = os.path.join(report_dir, report_filename)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(''.join(output_buffer))
            
            print(f"模型检查报告已保存到: {report_path}")
            
            # 同时在控制台输出
            for line in output_buffer:
                print(line, end='')
            
            return f"模型检查报告已保存到: {report_path}"
        except Exception as e:
            error_msg = f"搜索模型文档错误: {e}"
            print(error_msg)
            import traceback
            traceback_str = traceback.format_exc()
            print(f"详细错误信息:\n{traceback_str}")
            return error_msg

    def check_media_file(self, file_path):
        """检查媒体文件是否可用"""
        try:
            # 使用 ffprobe 检查文件信息
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration,size,bit_rate",
                "-show_streams",
                "-of", "json",
                file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"文件检查错误: {result.stderr}")
                return False, f"文件检查错误: {result.stderr}"
            
            info = json.loads(result.stdout)
            print(f"文件信息: {info}")
            
            # 检查是否包含音频流
            has_audio = False
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    has_audio = True
                    break
                
            if not has_audio:
                return False, "文件不包含音频流"
            
            return True, info
        
        except Exception as e:
            print(f"检查媒体文件错误: {e}")
            import traceback
            print(traceback.format_exc())
            return False, f"检查媒体文件错误: {e}"

    def save_online_transcript(self):
        """专门用于在线转录的文本保存"""
        try:
            if not self.transcript_text:
                self.subtitle_label.setText("没有可保存的转录内容")
                return
            
            # 创建保存目录
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcripts")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # 删除之前的临时文件（如果存在）
            if hasattr(self, 'online_output_file') and os.path.exists(self.online_output_file):
                try:
                    os.remove(self.online_output_file)
                    print(f"已删除旧文件: {self.online_output_file}")
                except Exception as e:
                    print(f"删除旧文件失败: {e}")
            
            # 生成新的文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_info = f"{self.model_language.upper()}-{self.model_size}"
            filename = f"online_transcript_{model_info}_{timestamp}.txt"
            
            # 设置输出文件路径
            self.online_output_file = os.path.join(save_dir, filename)
            
            # 保存文件
            with open(self.online_output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.transcript_text))
            
            # 更新界面显示
            self.subtitle_label.setText(f"转录文本已保存到: {self.online_output_file}")
            print(f"转录文本已保存到: {self.online_output_file}")
            
        except Exception as e:
            print(f"保存转录文本错误: {e}")
            self.subtitle_label.setText(f"保存转录文本错误: {e}")

    def update_ui_state(self):
        """更新UI状态"""
        try:
            # 更新进度条到 100%
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("转录完成 (100%)")
            
            # 更新按钮状态
            self.start_button.setText("开始转录")
            self.start_button.setStyleSheet("background-color: rgba(50, 150, 50, 200); color: white;")
            
            # 滚动到底部
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            )
            
            # 延迟隐藏进度条
            QTimer.singleShot(5000, self.progress_bar.hide)
            
        except Exception as e:
            print(f"更新UI状态错误: {e}")

    def transcribe_large_file(self, file_path, duration):
        """转录大型文件 - 专门处理大型媒体文件，避免卡死问题"""
        try:
            print(f"使用增强模式处理大型文件: {file_path}")
            # 创建识别器
            rec = self.create_recognizer()
            if not rec:
                self.signals.new_text.emit("创建识别器失败")
                self.signals.transcription_finished.emit()
                return
            
            # 显示文件信息
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"文件大小: {file_size:.2f} MB, 时长: {duration:.2f} 秒")
            self.signals.new_text.emit(f"文件大小: {file_size:.2f} MB, 时长: {duration:.2f} 秒")
            
            # 第一阶段：使用更加健壮的方式读取音频数据
            print("第一阶段：分块读取音频数据...")
            self.signals.progress_updated.emit(0, f"准备中: 0%")
            
            # 使用持久化的 WAV 文件，存放在 transcripts 目录下
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcripts")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            temp_wav = os.path.join(save_dir, f"{base_name}_audio.wav")
            
            # 检查是否已存在转换后的WAV文件，并验证其有效性
            wav_is_valid = False
            if os.path.exists(temp_wav):
                try:
                    with wave.open(temp_wav, 'rb') as wf:
                        if wf.getnchannels() == 1 and wf.getframerate() == 16000:
                            wav_is_valid = True
                            print(f"发现有效的WAV文件: {temp_wav}")
                            self.signals.new_text.emit("发现有效的WAV文件，直接使用...")
                except Exception as e:
                    print(f"WAV文件验证失败: {e}")
                    wav_is_valid = False
                    
                # 如果WAV文件无效，删除它
                if not wav_is_valid and os.path.exists(temp_wav):
                    try:
                        os.remove(temp_wav)
                        print(f"删除无效的WAV文件: {temp_wav}")
                    except Exception as e:
                        print(f"删除无效WAV文件失败: {e}")

            if not wav_is_valid:
                # 1. 转换视频文件为 WAV 格式
                convert_cmd = [
                    'ffmpeg',
                    '-y',  # 强制覆盖
                    '-i', file_path,
                    '-ar', '16000',  # 采样率
                    '-ac', '1',      # 单声道
                    '-vn',           # 不处理视频
                    '-f', 'wav',
                    temp_wav
                ]
                
                # 显示转换开始信息
                self.signals.new_text.emit("正在转换音频格式，请稍候...")
                
                # 使用 subprocess 运行命令，并设置超时
                try:
                    # 创建进程，并设置管道
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = subprocess.SW_HIDE
                    
                    process = subprocess.Popen(
                        convert_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        startupinfo=startupinfo,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                        text=True,
                        bufsize=4096
                    )
                    
                    # 设置超时时间（根据文件大小动态计算）
                    timeout = max(30, int(file_size * 0.5))  # 至少30秒，每MB增加0.5秒
                    start_time = time.time()
                    
                    # 创建线程来读取进程输出
                    stderr_queue = Queue()
                    
                    def read_stderr():
                        while True:
                            line = process.stderr.readline()
                            if not line and process.poll() is not None:
                                break
                            if line:
                                stderr_queue.put(line.strip())
                    
                    stderr_thread = Thread(target=read_stderr, daemon=True)
                    stderr_thread.start()
                    
                    # 监控进程状态和输出
                    while process.poll() is None and self.is_running:
                        # 检查是否超时
                        current_time = time.time()
                        if current_time - start_time > timeout:
                            process.terminate()
                            time.sleep(0.5)
                            if process.poll() is None:
                                process.kill()
                            raise TimeoutError(f"转换超时（{timeout}秒）")
                        
                        # 非阻塞方式读取stderr
                        try:
                            while True:  # 读取所有当前可用的输出
                                line = stderr_queue.get_nowait()
                                print(f"FFmpeg输出: {line.strip()}")
                                # 如果发现错误关键字，提前终止
                                if "error" in line.lower():
                                    process.kill()
                                    raise RuntimeError(f"FFmpeg错误: {line.strip()}")
                        except Empty:
                            pass  # 队列为空，继续处理
                        
                        # 更新进度（基于时间）
                        progress = min(10, int((time.time() - start_time) / timeout * 10))
                        self.signals.progress_updated.emit(progress, f"转换中: {progress}%")
                        
                        # 短暂休眠以减少CPU使用
                        time.sleep(0.1)
                    
                    # 检查进程返回值
                    if process.returncode != 0 and self.is_running:
                        error = process.stderr.read()
                        raise RuntimeError(f"转换音频格式错误: {error}")
                    
                    # 检查输出文件
                    if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) == 0:
                        raise RuntimeError("转换后的WAV文件无效")
                    
                except Exception as e:
                    print(f"转换音频格式异常: {e}")
                    self.signals.new_text.emit(f"转换音频格式异常: {e}")
                    self.signals.transcription_finished.emit()
                    return
                
                finally:
                    # 确保进程被终止
                    if 'process' in locals():
                        try:
                            process.kill()
                        except:
                            pass
            
            if not self.is_running:
                print("转录被用户终止")
                self.signals.transcription_finished.emit()
                return
            
            # 2. 分块读取和处理 WAV 文件
            try:
                # 检查临时 WAV 文件是否创建成功
                if not os.path.exists(temp_wav):
                    print("WAV 文件不存在")
                    self.signals.new_text.emit("WAV 文件不存在")
                    self.signals.transcription_finished.emit()
                    return
                
                # 获取文件大小，用于计算进度
                wav_size = os.path.getsize(temp_wav)
                
                # 打开 WAV 文件
                import wave
                with wave.open(temp_wav, 'rb') as wf:
                    # 获取音频文件信息
                    n_channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                    framerate = wf.getframerate()
                    n_frames = wf.getnframes()
                    
                    print(f"WAV文件信息: 通道数={n_channels}, 采样宽度={sample_width}, "
                          f"采样率={framerate}, 总帧数={n_frames}")
                    
                    # 计算每块的帧数 - 动态调整块大小
                    chunk_frames = 8000  # 较大的块能减少处理次数
                    bytes_per_frame = n_channels * sample_width
                    total_frames_processed = 0
                    
                    # 处理音频数据
                    while self.is_running:
                        # 读取一块数据
                        frames = wf.readframes(chunk_frames)
                        if not frames:
                            break  # 文件读取完毕
                        
                        # 更新已处理的帧数
                        frames_read = len(frames) // bytes_per_frame
                        total_frames_processed += frames_read
                        
                        # 更新进度（10%-90%）
                        progress = 10 + int(80 * (total_frames_processed / n_frames))
                        time_position = total_frames_processed / framerate
                        time_str = f"{int(time_position//60):02d}:{int(time_position%60):02d}"
                        total_str = f"{int(duration//60):02d}:{int(duration%60):02d}"
                        format_text = f"处理中: {time_str} / {total_str} ({progress}%)"
                        
                        self.signals.progress_updated.emit(progress, format_text)
                        
                        # 处理音频数据
                        if rec.AcceptWaveform(frames):
                            result = json.loads(rec.Result())
                            if result.get('text', '').strip():
                                text = result['text'].strip()
                                text = self.add_punctuation(text)
                                if text not in self.transcript_text:
                                    self.transcript_text.append(text)
                                    self.signals.new_text.emit(text)
                        else:
                            partial = json.loads(rec.PartialResult())
                            if partial.get('partial', '').strip():
                                self.signals.new_text.emit("PARTIAL:" + partial['partial'].strip())
                
                # 处理最后的结果
                final_result = json.loads(rec.FinalResult())
                if final_result.get('text', '').strip():
                    text = final_result['text'].strip()
                    text = self.add_punctuation(text)
                    if text not in self.transcript_text:
                        self.transcript_text.append(text)
                        self.signals.new_text.emit(text)
                
                # 更新进度到100%
                self.signals.progress_updated.emit(100, "转录完成 (100%)")
                
                # 保存转录结果 - 直接调用 on_transcription_finished
                self.signals.transcription_finished.emit()
                
                # 提示WAV文件位置
                print(f"WAV文件已保存在: {temp_wav}")
                self.signals.new_text.emit(f"\nWAV文件已保存在: {temp_wav}")
                
            except Exception as e:
                print(f"处理WAV文件错误: {e}")
                self.signals.new_text.emit(f"处理WAV文件错误: {e}")
                import traceback
                print(traceback.format_exc())
            
        except Exception as e:
            print(f"转录大型文件错误: {e}")
            self.signals.new_text.emit(f"转录错误: {e}")
            import traceback
            print(traceback.format_exc())
        
        finally:
            # 发送完成信号
            self.signals.transcription_finished.emit()

if __name__ == "__main__":
        app = QApplication(sys.argv)
        window = SubtitleWindow()
        window.show()
        sys.exit(app.exec_())