#!/usr/bin/env python3

import sys
import os
import json
import subprocess
import threading
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, 
                            QPushButton, QSlider, QHBoxLayout, QComboBox, QFileDialog, 
                            QTabWidget, QRadioButton, QButtonGroup, QShortcut, QLineEdit, QProgressBar, QScrollArea, QMenuBar, QMenu, QAction, QActionGroup, QSizePolicy, QGraphicsOpacityEffect, QDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QFont, QColor, QKeySequence
import vosk
import soundcard as sc  # 只保留 Windows 的音频捕获库
import warnings
import ctypes
from ctypes import windll, c_int, byref
import psutil  # 导入 psutil 库用于检查系统资源
import gc  # 导入垃圾回收模块
from vosk import Model, KaldiRecognizer, SetLogLevel  # 移除 EndpointerMode
from vosk import SpkModel  # 导入说话人识别模型

# 忽略警告
warnings.filterwarnings("ignore", message="data discontinuity in recording")

# 添加 Windows API 常量
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
LWA_ALPHA = 0x00000002

os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"  # 添加这行来抑制 COM 错误警告

class TranscriptionSignals(QObject):
    new_text = pyqtSignal(str)
    progress_updated = pyqtSignal(int, str)  # 添加进度信号
    transcription_finished = pyqtSignal()    # 添加完成信号

class SubtitleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 设置窗口属性 - 使用 Qt.Window 而不是 Qt.Tool 以显示最小化和最大化按钮
        self.setWindowTitle("实时字幕")
        self.setWindowFlags(Qt.WindowFlags.WindowStaysOnTopHint | Qt.WindowFlags.Window)  # 使用 Qt.Window 代替 Qt.Tool
        
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
        
        # 添加识别模式属性
        self.recognition_mode = 'en'  # 默认为英文识别
        
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
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.subtitle_label.setStyleSheet("color: red; background-color: rgba(0, 0, 0, 150); padding: 10px; border-radius: 10px;")
        self.subtitle_label.setFont(QFont("Arial", 20, QFont.Bold))  # 增加字体大小
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)  # 允许选择文本
        self.subtitle_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 允许标签扩展
        
        # 添加字幕标签到容器
        self.transcript_layout.addWidget(self.subtitle_label, 1)  # 使用拉伸因子1
        
        # 设置滚动区域的内容
        self.scroll_area.setWidget(self.transcript_container)
        
        # 确保滚动条可见
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        
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
        
        # 在 __init__ 方法中添加
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 设置默认值
        self.is_system_audio = True  # 默认为系统音频模式
        self.model_language = 'en'   # 默认为英文
        self.model_size = 'small'    # 默认为小型模型
        self.recognition_mode = 'en' # 默认为英文识别
        
        # 在资源检查和模型加载完成后自动开始转录
        QTimer.singleShot(5000, self.auto_start_transcription)
        
        # 添加说话人识别器
        self.speaker_identifier = None
        self.spk_model_path = "vosk-model-spk-0.4"
        self.enable_speaker_id = False  # 默认不启用说话人识别
        self.speaker_audio_buffer = b''  # 用于累积足够的音频数据进行说话人识别
        self.min_speaker_audio_size = 64000  # 增加到4秒的音频数据
        self.current_speaker_id = None  # 当前说话人ID
        
        # 在 __init__ 方法中添加
        self.config = self.load_config()
        self.model_paths = self.config["models"]
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        menu = menubar.addMenu('Menu')
        
        # 转录模式子菜单
        mode_menu = QMenu('转录模式', self)
        
        # 系统音频子菜单（添加语言选择）
        system_audio_menu = QMenu('系统音频', self)
        
        # 创建系统音频下的语言选择动作
        self.sys_en_action = QAction('英文识别', self, checkable=True)
        self.sys_cn_action = QAction('中文识别', self, checkable=True)
        self.sys_auto_action = QAction('自动识别', self, checkable=True)
        
        # 默认选中英文识别
        self.sys_en_action.setChecked(True)
        
        # 将语言选择动作添加到系统音频菜单
        system_audio_menu.addAction(self.sys_en_action)
        system_audio_menu.addAction(self.sys_cn_action)
        system_audio_menu.addAction(self.sys_auto_action)
        
        # 创建语言选择动作组
        sys_lang_group = QActionGroup(self)
        sys_lang_group.addAction(self.sys_en_action)
        sys_lang_group.addAction(self.sys_cn_action)
        sys_lang_group.addAction(self.sys_auto_action)
        sys_lang_group.setExclusive(True)
        
        # 文件转录动作
        self.file_audio_action = QAction('音频/视频文件', self, checkable=True)
        
        # 将系统音频菜单和文件转录动作添加到转录模式菜单
        mode_menu.addMenu(system_audio_menu)
        mode_menu.addAction(self.file_audio_action)
        
        # 将动作添加到模式动作组
        mode_group = QActionGroup(self)
        mode_group.addAction(system_audio_menu.menuAction())
        mode_group.addAction(self.file_audio_action)
        mode_group.setExclusive(True)
        
        # 音频设备子菜单
        device_menu = QMenu('音频设备', self)
        self.populate_audio_devices_menu(device_menu)
        
        # 模型选择子菜单
        model_menu = QMenu('模型选择', self)
        
        # 英文模型子菜单 - 保存为类属性
        self.en_model_menu = QMenu('英文 (En->Cn)', self)
        self.en_small_action = QAction('小型模型', self, checkable=True)
        self.en_medium_action = QAction('中型模型', self, checkable=True)
        self.en_large_action = QAction('大型模型', self, checkable=True)
        
        # 中文模型子菜单 - 保存为类属性
        self.cn_model_menu = QMenu('中文 (Cn->En)', self)
        self.cn_small_action = QAction('小型模型', self, checkable=True)
        self.cn_medium_action = QAction('中型模型', self, checkable=True)
        self.cn_large_action = QAction('大型模型', self, checkable=True)
        
        # 将模型动作添加到动作组
        model_group = QActionGroup(self)
        model_group.addAction(self.en_small_action)
        model_group.addAction(self.en_medium_action)
        model_group.addAction(self.en_large_action)
        model_group.addAction(self.cn_small_action)
        model_group.addAction(self.cn_medium_action)
        model_group.addAction(self.cn_large_action)
        model_group.setExclusive(True)
        
        # 添加模型动作到子菜单
        self.en_model_menu.addAction(self.en_small_action)
        self.en_model_menu.addAction(self.en_medium_action)
        self.en_model_menu.addAction(self.en_large_action)
        
        self.cn_model_menu.addAction(self.cn_small_action)
        self.cn_model_menu.addAction(self.cn_medium_action)
        self.cn_model_menu.addAction(self.cn_large_action)
        
        # 添加子菜单到模型菜单
        model_menu.addMenu(self.en_model_menu)
        model_menu.addMenu(self.cn_model_menu)
        
        # 默认启用英文模型菜单，禁用中文模型菜单
        self.en_model_menu.setEnabled(True)
        self.cn_model_menu.setEnabled(False)
        
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
        self.sys_en_action.triggered.connect(lambda: self.switch_recognition_mode('en'))
        self.sys_cn_action.triggered.connect(lambda: self.switch_recognition_mode('cn'))
        self.sys_auto_action.triggered.connect(lambda: self.switch_recognition_mode('auto'))
        self.file_audio_action.triggered.connect(lambda: self.switch_mode(False))
        
        # 连接模型选择信号
        self.en_small_action.triggered.connect(lambda: self.change_model('en', 'small'))
        self.en_medium_action.triggered.connect(lambda: self.change_model('en', 'medium'))
        self.en_large_action.triggered.connect(lambda: self.change_model('en', 'large'))
        self.cn_small_action.triggered.connect(lambda: self.change_model('cn', 'small'))
        self.cn_medium_action.triggered.connect(lambda: self.change_model('cn', 'medium'))
        self.cn_large_action.triggered.connect(lambda: self.change_model('cn', 'large'))
        
        # 连接背景模式信号 - 添加这些连接
        self.opaque_action.triggered.connect(lambda: self.change_background_mode('opaque'))
        self.translucent_action.triggered.connect(lambda: self.change_background_mode('translucent'))
        self.transparent_action.triggered.connect(lambda: self.change_background_mode('transparent'))
        
        # 添加子菜单到主菜单
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
        
        # 连接菜单信号
        self.connect_menu_signals()
        
        # 添加说话人识别菜单
        speaker_menu = QMenu('说话人识别', self)
        self.speaker_enable_action = QAction('启用说话人识别', self, checkable=True)
        self.speaker_enable_action.setChecked(False)
        speaker_menu.addAction(self.speaker_enable_action)
        
        # 添加到主菜单
        menubar.addMenu(speaker_menu)
        
        # 连接信号
        self.speaker_enable_action.triggered.connect(self.toggle_speaker_id)
    
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
            # 如果正在运行，先停止转录
            if self.is_running:
                self.stop_transcription()
                
            self.is_system_audio = is_system_audio
            if is_system_audio:
                self.subtitle_label.setText("已切换到系统音频模式，点击开始按钮开始转录")
            else:
                self.subtitle_label.setText("已切换到文件模式，请选择音频/视频文件")
                # 使用延迟调用文件选择对话框
                QTimer.singleShot(200, self.select_file)
        except Exception as e:
            print(f"切换模式错误: {e}")
            import traceback
            print(traceback.format_exc())
            self.subtitle_label.setText(f"切换模式错误: {e}")

    def safe_select_file(self):
        """安全地调用文件选择对话框"""
        try:
            # 确保当前不在运行状态
            if self.is_running:
                print("警告: 尝试在运行状态下选择文件，先停止转录")
                self.stop_transcription()
                # 等待一段时间确保转录完全停止
                time.sleep(0.5)
                
            # 调用文件选择对话框
            self.select_file()
        except Exception as e:
            print(f"安全调用文件选择对话框错误: {e}")
            import traceback
            print(traceback.format_exc())
            self.subtitle_label.setText(f"选择文件错误: {e}")
    
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
        try:
            # 设置日志级别
            try:
                SetLogLevel(0)  # 0 表示正常日志级别
            except Exception as e:
                print(f"设置日志级别失败: {e}")
            
            # 确定模型路径
            model_name = self.get_model_name()
            model_path = os.path.join(self.model_path_base, model_name)
            
            # 检查模型目录是否存在
            if not os.path.exists(model_path):
                error_msg = f"错误: 模型目录不存在: {model_path}"
                print(error_msg)
                
                # 添加错误信息到转录文本列表
                self.transcript_text.append(error_msg)
                self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
                
                # 滚动到底部
                self.scroll_area.verticalScrollBar().setValue(
                    self.scroll_area.verticalScrollBar().maximum()
                )
                
                return False
            
            # 显示加载信息
            loading_msg = f"开始加载模型: {model_name}"
            print(loading_msg)
            
            # 添加加载信息到转录文本列表
            self.transcript_text.append(loading_msg)
            self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
            
            # 滚动到底部
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            )
            
            # 强制更新界面
            QApplication.processEvents()
            
            # 加载模型
            self.model = Model(model_path)
            
            # 测试模型是否正确初始化
            try:
                # 创建一个临时识别器来测试模型
                test_rec = KaldiRecognizer(self.model, 16000)
                # 设置识别参数
                test_rec.SetWords(True)  # 启用词级别时间戳
                test_rec.SetPartialWords(True)  # 启用部分结果的词级别信息
                test_rec.SetMaxAlternatives(3)  # 设置最大候选结果数
                
                success_msg = "模型测试成功: 能够创建识别器"
                print(success_msg)
                
                # 添加成功信息到转录文本列表
                self.transcript_text.append(success_msg)
                
                del test_rec  # 释放临时识别器
            except Exception as e:
                error_msg = f"模型测试失败: {e}"
                print(error_msg)
                
                # 添加错误信息到转录文本列表
                self.transcript_text.append(error_msg)
                
                raise
            
            # 显示成功信息
            success_msg = f"成功加载模型: {model_name}"
            print(success_msg)
            
            # 添加成功信息到转录文本列表
            self.transcript_text.append(success_msg)
            
            # 如果需要保留系统信息，则在前面添加系统信息
            if keep_info and hasattr(self, 'system_info'):
                display_text = self.transcript_text[-10:]
                self.subtitle_label.setText('\n'.join(display_text))
            else:
                # 否则只显示最近的几条消息
                self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
            
            # 滚动到底部
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            )
            
            return True
            
        except Exception as e:
            error_msg = f"模型加载失败: {e}"
            print(error_msg)
            
            # 获取详细的错误信息
            import traceback
            traceback_str = traceback.format_exc()
            print(f"详细错误信息:\n{traceback_str}")
            
            # 添加错误信息到转录文本列表
            self.transcript_text.append(error_msg)
            self.transcript_text.append(f"详细错误信息: {str(e)}")
            
            if keep_info:
                display_text = self.transcript_text[-10:]
                self.subtitle_label.setText('\n'.join(display_text))
            else:
                self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
            
            # 滚动到底部
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            )
            
            self.model = None
            return False

    def create_recognizer(self):
        """创建新的识别器"""
        try:
            if not self.model:
                print("错误: 模型未加载")
                return None
            
            # 创建识别器
            rec = KaldiRecognizer(self.model, 16000)
            
            # 基本设置
            rec.SetWords(True)  # 启用词级别时间戳
            rec.SetPartialWords(True)  # 启用部分结果的词级别信息
            
            # 设置最大候选结果数
            rec.SetMaxAlternatives(3)
            
            # 如果启用了说话人识别，设置说话人识别模型
            if self.enable_speaker_id:
                if self.speaker_identifier is None:
                    # 如果模型未加载，先加载模型
                    success = self.load_spk_model()
                    if not success:
                        print("说话人识别模型加载失败")
                
                if self.speaker_identifier:
                    print("正在设置说话人识别模型到识别器...")
                    rec.SetSpkModel(self.speaker_identifier.spk_model)
                    print("说话人识别模型设置成功")
            
            return rec
            
        except Exception as e:
            print(f"创建识别器失败: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def process_audio(self, audio_data):
        """处理音频数据"""
        try:
            if not self.rec:
                self.rec = self.create_recognizer()
                if not self.rec:
                    return
            
            # 添加一个标志来标识是否是第一次识别
            self.is_first_recognition = True
            # 添加缓冲区
            self.audio_buffer = b''
            
            if self.is_first_recognition:
                # 累积一定量的音频数据再开始识别
                self.audio_buffer += audio_data
                if len(self.audio_buffer) < 32000:  # 等待约2秒的音频数据
                    return  # 这里缺少了缩进
            else:
                    # 使用累积的音频数据进行第一次识别
                    audio_data = self.audio_buffer
                    self.is_first_recognition = False
            
            if self.rec.AcceptWaveform(audio_data):
                # 获取完整的识别结果
                result = json.loads(self.rec.Result())
                
                # 转换数字
                text = result.get("text", "")
                if text:
                    converted_text = self._convert_words_to_numbers(text)
                    return converted_text
            else:
                # 处理部分结果
                partial = json.loads(self.rec.PartialResult())
                partial_text = partial.get("partial", "")
                if partial_text:
                    converted_text = self._convert_words_to_numbers(partial_text)
                    return "PARTIAL:" + converted_text
                
        except Exception as e:
            print(f"处理音频数据错误: {e}")
            return ""
    
    def toggle_transcription(self):
        """切换转录状态"""
        if self.is_running:
            self.stop_transcription()
        else:
            if self.is_system_audio:
                self.start_system_audio_transcription()
            else:
                if self.file_path:
                    self.start_file_transcription(self.file_path)
                else:
                    self.select_file()
    
    def start_transcription(self):
        """开始转录"""
        try:
            if self.is_running:
                return
            
            self.is_running = True
            self.is_first_recognition = True  # 重置第一次识别标志
            self.audio_buffer = b''  # 重置音频缓冲区
            
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
                
                # 启动文件转录线程
                self.audio_thread = threading.Thread(target=self.transcribe_file, args=(self.file_path,))
                self.audio_thread.daemon = True
                self.audio_thread.start()
            else:
                # 系统音频模式
                # 创建识别器
                rec = vosk.KaldiRecognizer(self.model, 16000)
                rec.SetWords(True)
                
                # 启动音频捕获线程
                self.audio_thread = threading.Thread(target=self.capture_audio, args=(rec,))
                self.audio_thread.daemon = True
                self.audio_thread.start()
        
        except Exception as e:
            print(f"开始转录错误: {e}")
            self.signals.new_text.emit(f"转录错误: {e}")
    
    def stop_transcription(self):
        """停止转录并清理资源"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 保存转录文本 - 每次使用新的文件名
        if self.transcript_text:
            saved_path = self.save_transcript(use_timestamp=True)  # 添加参数，强制使用时间戳
            if saved_path:
                print(f"成功保存转录文件到: {saved_path}")
                # 在内容窗口显示保存成功的提示
                save_info = f"转录已完成！文件已保存到: {saved_path}"
                self.transcript_text.append(save_info)
                self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
                
                # 滚动到底部
                self.scroll_area.verticalScrollBar().setValue(
                    self.scroll_area.verticalScrollBar().maximum()
                )
        
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        
        # 重置状态
        self.audio_thread = None
        self.start_button.setText("开始转录")
        self.start_button.setStyleSheet("background-color: rgba(50, 50, 50, 200); color: white;")
        
        # 隐藏进度条
        self.progress_bar.hide()
    
    def capture_audio(self, rec):
        """捕获系统音频"""
        sample_rate = 16000  # 采样率
        self.capture_audio_windows(rec, sample_rate)
    
    def transcribe_file(self, file_path):
        sample_rate = 16000
        
        # 获取文件总时长
        try:
            probe = subprocess.run([
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                file_path
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            probe_data = json.loads(probe.stdout)
            total_duration = float(probe_data['format']['duration'])
            print(f"文件总时长: {total_duration:.2f} 秒")
        except Exception as e:
            print(f"获取文件时长错误: {e}")
            total_duration = 0
        
        # 创建识别器
        rec = vosk.KaldiRecognizer(self.model, sample_rate)
        rec.SetWords(True)
        
        try:
            # 使用 ffmpeg 转换音频
            self.ffmpeg_process = subprocess.Popen([
                'ffmpeg',
                '-i', file_path,
                '-ar', str(sample_rate),
                '-ac', '1',
                '-f', 's16le',
                '-'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
            
            # 处理音频数据
            total_bytes = 0
            last_progress_time = time.time()
            
            # 直接处理音频流，不预先读取到内存
            print("开始处理音频数据...")
            
            # 收集所有音频块
            all_chunks = []
            
            # 第一阶段：读取所有数据
            print("第一阶段：读取所有音频数据...")
            while self.is_running:
                chunk = self.ffmpeg_process.stdout.read(4000)
                if not chunk:
                    print("文件读取完成")
                    break
                all_chunks.append(chunk)
                
                # 更新读取进度
                total_bytes += len(chunk)
                current_time = time.time()
                
                if current_time - last_progress_time >= 0.2:
                    if total_duration > 0:
                        # 计算读取进度 (最大50%)
                        current_position = (total_bytes / (sample_rate * 2))
                        progress = min(50, int((current_position / total_duration) * 50))
                        
                        time_str = f"{int(current_position//60):02d}:{int(current_position%60):02d}"
                        total_str = f"{int(total_duration//60):02d}:{int(total_duration%60):02d}"
                        format_text = f"读取中: {time_str} / {total_str} ({progress}%)"
                        
                        # 发送进度更新信号
                        self.signals.progress_updated.emit(progress, format_text)
                        print(f"读取进度: {progress}% ({time_str} / {total_str})")
                    
                    last_progress_time = current_time
            
            # 确保 FFmpeg 进程终止
            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
                try:
                    self.ffmpeg_process.wait(timeout=1)
                except:
                    try:
                        self.ffmpeg_process.kill()
                    except:
                        pass
                self.ffmpeg_process = None
            
            # 第二阶段：处理所有数据
            print(f"第二阶段：处理 {len(all_chunks)} 个音频块...")
            total_chunks = len(all_chunks)
            
            for i, chunk in enumerate(all_chunks):
                if not self.is_running:
                    break
                
                # 处理音频数据
                if rec.AcceptWaveform(chunk):
                    result = json.loads(rec.Result())
                    if 'text' in result and result['text'].strip():
                        self.signals.new_text.emit(result['text'])
                        self.transcript_text.append(result['text'])
                
                # 更新处理进度
                current_time = time.time()
                if current_time - last_progress_time >= 0.2:
                    # 计算处理进度 (50%-99%)
                    progress = 50 + min(49, int((i / total_chunks) * 49))
                    
                    format_text = f"处理中: {progress}%"
                    
                    # 发送进度更新信号
                    self.signals.progress_updated.emit(progress, format_text)
                    print(f"处理进度: {progress}%")
                    
                    last_progress_time = current_time
            
            # 处理最后的结果
            final_result = json.loads(rec.FinalResult())
            if 'text' in final_result and final_result['text'].strip():
                self.signals.new_text.emit(final_result['text'])
                self.transcript_text.append(final_result['text'])
            
            # 转录完成，更新进度为 100%
            self.signals.progress_updated.emit(100, "转录完成 (100%)")
            print("转录处理完成，设置进度为 100%")
            
            # 发送完成信号
            self.signals.transcription_finished.emit()
        
        except Exception as e:
            print(f"转录过程错误: {e}")
            self.signals.new_text.emit(f"转录错误: {e}")
            
            # 确保 FFmpeg 进程终止
            if self.ffmpeg_process:
                try:
                    self.ffmpeg_process.terminate()
                    self.ffmpeg_process.wait(timeout=1)
                except:
                    try:
                        self.ffmpeg_process.kill()
                    except:
                        pass
                self.ffmpeg_process = None
            
            self.signals.transcription_finished.emit()
    
    def capture_audio_windows(self, rec, sample_rate):
        """捕获 Windows 系统音频"""
        start_time = time.time()
        
        try:
            # 在设备识别部分添加CABLE关键字
            loopback_found = False
            for mic in sc.all_microphones(include_loopback=True):
                if "立体声混音" in mic.name or "Stereo Mix" in mic.name or "混音" in mic.name or "CABLE" in mic.name:
                    loopback_mic = mic
                    loopback_found = True
                    print(f"找到音频捕获设备: {mic.name}")
                    break
            
            # 如果没找到立体声混音，尝试使用默认扬声器的环回
            if not loopback_found:
                if self.current_device:
                    print(f"使用选定的设备: {self.current_device.name}")
                    loopback_mic = sc.get_microphone(id=str(self.current_device.id), include_loopback=True)
                else:
                    default_speaker = sc.default_speaker()
                    print(f"未找到立体声混音，使用默认扬声器环回: {default_speaker.name}")
                    loopback_mic = sc.get_microphone(id=str(default_speaker.id), include_loopback=True)
            
            # 创建录制器并继续原有逻辑
            with loopback_mic.recorder(samplerate=sample_rate) as mic:
                print("成功创建音频捕获")
                # 错误计数
                error_count = 0
                max_errors = 5
                
                while self.is_running:
                    try:
                        # 降低音频电平阈值
                        data = mic.record(numframes=2048)
                        # 转换为单声道
                        data = data.mean(axis=1)
                        
                        # 检查数据是否有效
                        if np.isnan(data).any() or np.isinf(data).any():
                            continue
                        
                        # 检查是否有声音
                        audio_level = np.max(np.abs(data))
                        if audio_level > 0.01:  # 只有当有声音时才打印
                            print(f"音频电平: {audio_level:.4f}")
                        
                        if audio_level < 0.0005:  # 从0.001降低到0.0005
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
            print(f"音频捕获错误: {e}")
            import traceback
            print(traceback.format_exc())
            self.signals.new_text.emit(f"音频捕获错误: {e}")
        finally:
            # 发送完成信号
            self.signals.transcription_finished.emit()
    
    def process_audio_data(self, rec, data):
        try:
            # 如果启用了说话人识别，累积音频数据
            if self.enable_speaker_id and self.speaker_identifier:
                try:
                    self.speaker_audio_buffer += data
                    
                    # 当累积足够的音频数据时，进行说话人识别
                    if len(self.speaker_audio_buffer) >= self.min_speaker_audio_size:
                        print(f"进行说话人识别，音频数据大小: {len(self.speaker_audio_buffer)} 字节")
                        
                        # 创建一个线程来处理说话人识别，避免阻塞主线程
                        def identify_speaker_thread():
                            try:
                                speaker_id = self.speaker_identifier.identify_speaker(self.speaker_audio_buffer)
                                if speaker_id is not None:
                                    self.current_speaker_id = speaker_id
                                    print(f"当前说话人ID: {self.current_speaker_id}")
                                else:
                                    print("说话人识别失败，可能需要更多音频数据")
                            except Exception as e:
                                print(f"说话人识别线程错误: {e}")
                        
                        # 启动说话人识别线程
                        spk_thread = threading.Thread(target=identify_speaker_thread)
                        spk_thread.daemon = True
                        spk_thread.start()
                        
                        # 保留一部分数据以保持连续性，丢弃旧数据
                        self.speaker_audio_buffer = self.speaker_audio_buffer[-16000:]  # 保留最后1秒
                except Exception as e:
                    print(f"处理说话人识别缓冲区错误: {e}")
                    import traceback
                    print(traceback.format_exc())
            
            # 处理语音识别
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                print(f"识别结果: {result}")
                
                if 'text' in result and result['text'].strip():
                    text = result['text'].strip()
                    
                    # 添加标点符号和首字母大写
                    text = self.add_punctuation(text)
                    print(f"处理后的文本: {text}")
                    
                    # 添加说话人标识
                    if self.enable_speaker_id and self.current_speaker_id is not None:
                        # 获取说话人名称
                        speaker_name = f"说话人{self.current_speaker_id}"
                        if hasattr(self.speaker_identifier, 'speaker_names') and \
                           len(self.speaker_identifier.speaker_names) >= self.current_speaker_id:
                            speaker_name = self.speaker_identifier.speaker_names[self.current_speaker_id-1]
                        
                        # 使用冒号格式
                        text = f"{speaker_name}：{text}"
                        print(f"添加说话人标识: {text}")
                    
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
                    
                    # 添加说话人标识
                    if self.enable_speaker_id and self.current_speaker_id is not None:
                        # 获取说话人名称
                        speaker_name = f"说话人{self.current_speaker_id}"
                        if hasattr(self.speaker_identifier, 'speaker_names') and \
                           len(self.speaker_identifier.speaker_names) >= self.current_speaker_id:
                            speaker_name = self.speaker_identifier.speaker_names[self.current_speaker_id-1]
                        
                        partial_text = f"{speaker_name}：{partial_text}"
                    
                    # 使用特殊标记表示这是部分结果，并立即发送到UI更新
                    self.signals.new_text.emit("PARTIAL:" + partial_text)
                    
                    # 确保每次有新的部分结果时都更新UI
                    QApplication.processEvents()
        except Exception as e:
            print(f"处理音频数据错误: {e}")
            import traceback
            print(traceback.format_exc())
    
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
    
    def _convert_words_to_numbers(self, text):
        """将文本中的数字单词转换为阿拉伯数字"""
        # 数字映射字典
        number_mapping = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
            'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
            'eighteen': '18', 'nineteen': '19', 'twenty': '20',
            'thirty': '30', 'forty': '40', 'fifty': '50',
            'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90',
            'hundred': '100', 'thousand': '1000', 'million': '1000000',
            'billion': '1000000000'
        }
        
        # 特殊年份组合
        year_patterns = {
            r'nineteen\s+hundred': '1900',
            r'twenty\s+hundred': '2000',
            r'nineteen\s+(\w+)': lambda m: f'19{number_mapping.get(m.group(1), "00")}',
            r'twenty\s+(\w+)': lambda m: f'20{number_mapping.get(m.group(1), "00")}'
        }
        
        # 首先处理年份
        import re
        for pattern, replacement in year_patterns.items():
            if callable(replacement):
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            else:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # 处理普通数字
        words = text.split()
        result = []
        i = 0
        while i < len(words):
            current_word = words[i].lower()
            
            # 检查是否是复合数字（如 twenty one）
            if i + 1 < len(words):
                compound = f"{current_word} {words[i+1].lower()}"
                if compound in number_mapping:
                    result.append(number_mapping[compound])
                    i += 2
                    continue
            
            # 检查单个数字词
            if current_word in number_mapping:
                # 特殊处理 hundred/thousand/million/billion
                if current_word in ['hundred', 'thousand', 'million', 'billion']:
                    if result and result[-1].isdigit():
                        last_num = int(result.pop())
                        result.append(str(last_num * int(number_mapping[current_word])))
                    else:
                        result.append(number_mapping[current_word])
                else:
                    result.append(number_mapping[current_word])
            else:
                result.append(words[i])
            i += 1
        
        return ' '.join(result)

    def update_subtitle(self, text):
        """更新字幕显示"""
        try:
            if text.startswith("PARTIAL:"):
                # 处理部分结果
                partial_text = text[8:]  # 去掉 "PARTIAL:" 前缀
                
                # 保留当前显示的完整结果，只更新部分结果
                display_text = []
                for line in self.transcript_text[-8:]:  # 减少显示行数，确保有足够空间
                    display_text.append(line)
                
                # 添加当前的部分结果
                display_text.append(partial_text)
                
                # 更新显示 - 使用单个换行符而不是双换行符
                self.subtitle_label.setText('\n'.join(display_text))
                
                # 强制滚动到底部，确保最新内容可见
                QTimer.singleShot(10, lambda: self.scroll_area.verticalScrollBar().setValue(
                    self.scroll_area.verticalScrollBar().maximum()))
            else:
                # 处理完整结果 - 检查是否与最后一个结果相同或相似
                if self.transcript_text and (text == self.transcript_text[-1] or self._is_similar(text, self.transcript_text[-1])):
                    # 如果是重复或非常相似的文本，不添加到列表
                    print(f"跳过重复文本: {text}")
                else:
                    # 添加新的完整结果到转录文本列表
                    # 检查是否是状态信息，如果是则不添加到转录文本列表
                    if not text.startswith("已启用") and not text.startswith("已禁用") and \
                       not text.startswith("开始") and not text.startswith("转录") and \
                       not text.startswith("加载") and not text.startswith("成功"):
                        self.transcript_text.append(text)
                    
                    # 限制历史记录最多保留500行
                    if len(self.transcript_text) > 500:
                        self.transcript_text = self.transcript_text[-500:]
                    
                    # 计算当前字幕标签可以显示的行数 - 调整计算方法
                    font_metrics = self.subtitle_label.fontMetrics()
                    label_height = self.subtitle_label.height()
                    line_height = font_metrics.lineSpacing() + 4  # 增加行间距，确保完整显示
                    visible_lines = max(3, int((label_height - 20) / line_height))  # 减去边距，确保不会溢出
                    
                    # 只显示最近的几条完整结果，数量取决于可见行数
                    display_text = []
                    for line in self.transcript_text[-visible_lines:]:
                        display_text.append(line)
                    
                    # 使用单个换行符分隔文本，不插入空行
                    self.subtitle_label.setText('\n'.join(display_text))
                    
                    # 使用延时滚动，确保UI更新后再滚动
                    QTimer.singleShot(10, lambda: self.scroll_area.verticalScrollBar().setValue(
                        self.scroll_area.verticalScrollBar().maximum()))
        except Exception as e:
            print(f"更新字幕错误: {e}")
            import traceback
            print(traceback.format_exc())
    
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
    
    def change_model(self, language, size):
        """切换模型大小"""
        try:
            # 如果语言和大小都没有改变，不需要任何操作
            if self.model_language == language and self.model_size == size:
                return
            
            # 如果正在运行，先停止转录
            if self.is_running:
                self.stop_transcription()
        
            # 更新模型语言和大小
            self.model_language = language
            self.model_size = size
            
            # 更新界面提示
            size_text = {'small': '小型', 'medium': '中型', 'large': '大型'}
            lang_text = {'en': '英文', 'cn': '中文'}
            status_text = f"切换到{lang_text[language]}{size_text[size]}模型"
            self.subtitle_label.setText(status_text)
            
            # 模型大小改变时一定要重新加载模型
            print(f"切换到{language}语言的{size}型模型...")
            self.load_model(keep_info=True)
            
        except Exception as e:
            error_msg = f"切换模型错误: {e}"
            print(error_msg)
            self.subtitle_label.setText(error_msg)

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
                self.file_audio_action.setChecked(True)
                
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

    def save_transcript(self, use_timestamp=False):
        """保存转录文本到文件"""
        try:
            if not self.transcript_text:
                return
                
            # 创建转录目录
            transcript_dir = os.path.join(self.base_dir, "transcripts")
            os.makedirs(transcript_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_info = f"{self.model_language}-{self.model_size}"
            
            if use_timestamp or not hasattr(self, 'last_transcript_file'):
                # 使用时间戳创建新文件
                filename = f"transcript_{model_info}_{timestamp}.txt"
                self.last_transcript_file = filename
            else:
                # 使用上次的文件名
                filename = self.last_transcript_file
            
            filepath = os.path.join(transcript_dir, filename)
            
            # 保存文本
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(self.transcript_text))
            
            # 只在控制台打印保存信息，不发送到字幕窗口
            print(f"转录完成，文件已保存到: {filepath}")
            
            return filepath
            
        except Exception as e:
            print(f"保存转录文本错误: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def update_progress(self, progress, format_text):
        """更新进度条（在主线程中调用）"""
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(format_text)

    def on_transcription_finished(self):
        """转录完成后的处理（在主线程中调用）"""
        try:
            # 使用 QTimer 延迟更新其他 UI 元素
            QTimer.singleShot(100, self._update_ui_after_finish)
            
            # 设置运行状态为 False
            self.is_running = False
            
            # 如果是文件转录模式，自动保存文件
            if not self.is_system_audio and self.transcript_text:
                saved_path = self.save_transcript()
                if saved_path:
                    print(f"文件转录完成，自动保存到: {saved_path}")
                    # 在内容窗口显示保存成功的提示
                    save_info = f"转录已完成！文件已保存到: {saved_path}"
                    self.transcript_text.append(save_info)
                    self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
                    
                    # 滚动到底部
                    self.scroll_area.verticalScrollBar().setValue(
                        self.scroll_area.verticalScrollBar().maximum()
                    )
        
        except Exception as e:
            print(f"转录完成处理错误: {e}")

    def _update_ui_after_finish(self):
        """在转录完成后更新 UI（通过 QTimer 调用）"""
        try:
            # 更新进度条到 100%
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("转录完成 (100%)")
            
            # 更新按钮状态
            self.start_button.setText("开始转录")
            self.start_button.setStyleSheet("background-color: rgba(50, 150, 50, 200); color: white;")
            
            # 根据转录模式显示不同的提示
            if self.is_system_audio:
                # 系统音频模式：提示用户点击停止按钮保存
                self.transcript_text.append("转录已完成！点击停止按钮保存文件。")
                self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
            # 文件转录模式的提示会在 on_transcription_finished 中添加
            
            # 滚动到底部
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            )
            
            # 延迟隐藏进度条
            QTimer.singleShot(5000, self.progress_bar.hide)
            
        except Exception as e:
            print(f"更新完成 UI 错误: {e}")

    def select_file(self):
        """打开文件选择对话框 - 使用非模态对话框"""
        try:
            # 创建非模态文件对话框
            self.file_dialog = QFileDialog()
            self.file_dialog.setWindowTitle("选择音频/视频文件")
            self.file_dialog.setFileMode(QFileDialog.ExistingFile)
            self.file_dialog.setNameFilter("媒体文件 (*.mp3 *.wav *.mp4 *.avi *.mkv *.mov);;所有文件 (*)")
            
            # 连接文件选择信号
            self.file_dialog.fileSelected.connect(self.on_file_selected)
            
            # 显示非模态对话框
            self.file_dialog.show()
            
        except Exception as e:
            print(f"选择文件错误: {e}")
            import traceback
            print(traceback.format_exc())
            # 发生错误时切换回系统音频模式
            self.is_system_audio = True
            self.sys_en_action.setChecked(True)
            self.subtitle_label.setText(f"文件选择出错: {e}")

    def on_file_selected(self, file_path):
        """文件选择完成后的回调"""
        try:
            if file_path:
                print(f"已选择文件: {file_path}")
                self.file_path = file_path
                self.subtitle_label.setText(f"已选择文件: {os.path.basename(file_path)}")
                
                # 自动切换到文件转录模式
                self.is_system_audio = False
                self.file_audio_action.setChecked(True)
                
                # 使用延迟，确保UI完全更新
                QTimer.singleShot(300, lambda: self.start_file_transcription(file_path))
            else:
                print("未选择文件")
                # 如果用户取消选择，切换回系统音频模式
                self.is_system_audio = True
                self.sys_en_action.setChecked(True)
                self.subtitle_label.setText("未选择文件，保持当前模式")
            
        except Exception as e:
            print(f"处理选择的文件错误: {e}")
            import traceback
            print(traceback.format_exc())
            # 发生错误时切换回系统音频模式
            self.is_system_audio = True
            self.sys_en_action.setChecked(True)
            self.subtitle_label.setText(f"处理选择的文件错误: {e}")

    def delayed_start_file_transcription(self, file_path):
        """延迟启动文件转录，避免UI冻结"""
        try:
            print(f"准备开始转录文件: {file_path}")
            # 再次检查文件是否存在
            if not os.path.exists(file_path):
                self.subtitle_label.setText(f"错误: 文件不存在: {file_path}")
                return
                
            # 开始文件转录
            self.start_file_transcription(file_path)
        except Exception as e:
            print(f"延迟启动文件转录错误: {e}")
            import traceback
            print(traceback.format_exc())
            self.subtitle_label.setText(f"启动文件转录错误: {e}")

    def switch_recognition_mode(self, mode):
        """切换识别模式（英文/中文/自动）"""
        try:
            # 如果模式没有改变，不需要任何操作
            if hasattr(self, 'recognition_mode') and self.recognition_mode == mode:
                return
            
            # 如果正在运行，先停止转录
            if self.is_running:
                self.stop_transcription()
            
            # 设置识别模式
            self.recognition_mode = mode
            
            # 根据模式切换语言和模型
            if mode == 'en':
                self.model_language = 'en'
                # 默认使用英文小型模型
                self.model_size = 'small'
                self.en_small_action.setChecked(True)
                # 确保英文模型菜单可见
                self.en_model_menu.setEnabled(True)
                self.cn_model_menu.setEnabled(False)
                status_text = "已切换到英文识别模式（使用小型模型）"
            elif mode == 'cn':
                self.model_language = 'cn'
                # 默认使用中文小型模型
                self.model_size = 'small'
                self.cn_small_action.setChecked(True)
                # 确保中文模型菜单可见
                self.en_model_menu.setEnabled(False)
                self.cn_model_menu.setEnabled(True)
                status_text = "已切换到中文识别模式（使用小型模型）"
            else:  # auto 模式
                # 自动模式暂时保持当前模型
                self.en_model_menu.setEnabled(True)
                self.cn_model_menu.setEnabled(True)
                status_text = "已切换到自动识别模式（实验性功能）"
            
            # 更新界面提示 - 添加到转录文本列表以支持滚动
            self.transcript_text.append(status_text)
            
            # 限制历史记录最多保留500行
            if len(self.transcript_text) > 500:
                self.transcript_text = self.transcript_text[-500:]
            
            # 显示最近的几条完整结果
            self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
            
            # 滚动到底部
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            )
            
            # 切换语言时一定要重新加载对应的模型
            print(f"切换到{mode}模式，加载{self.model_language}语言的{self.model_size}型模型...")
            self.load_model(keep_info=True)
            
        except Exception as e:
            error_msg = f"切换识别模式错误: {e}"
            print(error_msg)
            
            # 添加错误信息到转录文本列表
            self.transcript_text.append(error_msg)
            self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
            
            # 滚动到底部
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            )

    def force_quit(self):
        """强制退出程序"""
        try:
            # 如果正在运行，先停止转录
            if self.is_running:
                self.stop_transcription()
            
            # 保存转录文本（如果有）
            if self.transcript_text:
                self.save_transcript()
            
            # 清理资源
            if self.model:
                del self.model
                self.model = None
            
            # 强制进行垃圾回收
            gc.collect()
            
            # 退出程序
            QApplication.quit()
            
        except Exception as e:
            print(f"退出程序错误: {e}")
            # 如果出错，仍然尝试退出
            QApplication.quit()

    def check_system_resources(self):
        """检查系统资源并显示相关信息"""
        try:
            # 先显示正在检查的消息
            checking_msg = "正在检查系统资源，请稍候..."
            self.transcript_text.append(checking_msg)
            self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
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
            
            # 添加系统资源信息到转录文本列表
            self.transcript_text.append(f"系统内存: 总计 {memory.total / (1024**3):.2f} GB, 可用 {memory.available / (1024**3):.2f} GB")
            self.transcript_text.append(f"磁盘空间: 总计 {disk.total / (1024**3):.2f} GB, 可用 {disk.free / (1024**3):.2f} GB")
            
            # 确定可以使用的最大模型
            available_models = []
            for size, model_size_gb in model_sizes.items():
                if model_size_gb < memory.available / (1024**3):
                    available_models.append(size)
            
            # 生成建议信息
            info_text = "模型大小参考:"
            self.transcript_text.append(info_text)
            self.transcript_text.append("- 小型模型: 约 0.07 GB 内存")
            self.transcript_text.append("- 中型模型: 约 0.5 GB 内存")
            self.transcript_text.append("- 大型模型: 约 2.67 GB 内存")
            
            if 'large' in available_models:
                recommendation = "✅ 您的系统内存充足，可以使用任何大小的模型。"
                self.transcript_text.append(recommendation)
                self.transcript_text.append("建议: 可以尝试大型模型以获得最佳识别效果。")
            elif 'medium' in available_models:
                recommendation = "⚠️ 您的系统内存不足以加载大型模型。"
                self.transcript_text.append(recommendation)
                self.transcript_text.append("建议: 使用中型模型以获得较好的识别效果。")
            else:
                recommendation = "❌ 您的系统内存较低，只能使用小型模型。"
                self.transcript_text.append(recommendation)
                self.transcript_text.append("建议: 关闭其他应用程序以释放更多内存，或增加系统内存。")
            
            # 保存系统资源信息（用于其他方法）
            self.system_info = '\n'.join(self.transcript_text[-10:])
            
            # 显示系统资源信息和建议
            self.transcript_text.append("正在加载默认模型，请稍候...")
            self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
            
            # 滚动到底部
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            )
            
            QApplication.processEvents()  # 强制处理事件，更新界面
            
            # 延迟加载模型，让用户有时间阅读系统资源信息
            QTimer.singleShot(3000, self.load_model_and_keep_info)
            
        except Exception as e:
            error_msg = f"检查系统资源错误: {e}"
            print(error_msg)
            
            # 添加错误信息到转录文本列表
            self.transcript_text.append(error_msg)
            self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
            
            # 滚动到底部
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            )
            
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

    def get_model_name(self):
        """根据当前语言和大小获取模型名称"""
        # 在加载大型模型前强制垃圾回收
        if self.model_size == 'large':
            gc.collect()
        
        # 根据语言和大小确定模型路径
        model_paths = {
            'en': {
                'small': "vosk-model-small-en-us-0.15",
                'medium': "vosk-model-en-us-0.22",  # 移除 -lgraph 后缀
                'large': "vosk-model-en-us-0.22"
            },
            'cn': {
                'small': "vosk-model-small-cn-0.22",
                'medium': "vosk-model-cn-0.22",
                'large': "vosk-model-cn-0.22"
            }
        }
        
        # 获取当前选择的模型路径
        model_name = model_paths.get(self.model_language, {}).get(self.model_size, "vosk-model-small-en-us-0.15")
        
        return model_name

    def change_background_mode(self, mode):
        """切换背景模式"""
        try:
            if mode == 'opaque':
                # 不透明模式
                self.central_widget.setStyleSheet("background-color: rgba(30, 30, 30, 255);")
                self.subtitle_label.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 150); padding: 10px; border-radius: 10px;")
                # 记录当前模式
                self.transcript_text.append("已切换到不透明背景模式")
            elif mode == 'translucent':
                # 半透明模式
                self.central_widget.setStyleSheet("background-color: rgba(30, 30, 30, 150);")
                self.subtitle_label.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 100); padding: 10px; border-radius: 10px;")
                # 记录当前模式
                self.transcript_text.append("已切换到半透明背景模式")
            elif mode == 'transparent':
                # 全透明模式
                self.central_widget.setStyleSheet("background-color: rgba(30, 30, 30, 0);")
                self.subtitle_label.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 50); padding: 10px; border-radius: 10px;")
                # 记录当前模式
                self.transcript_text.append("已切换到全透明背景模式")
            
            # 更新显示
            self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
            
            # 滚动到底部
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            )
            
            # 对于 Windows，使用 WinAPI 设置窗口透明度
            if mode == 'transparent' or mode == 'translucent':
                # 获取窗口句柄
                hwnd = int(self.winId())  # 转换为整数
                
                # 设置窗口扩展样式
                style = windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                style = style | WS_EX_LAYERED
                windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
                
                # 设置透明度
                alpha = 200 if mode == 'translucent' else 150  # 半透明或全透明的透明度值
                windll.user32.SetLayeredWindowAttributes(hwnd, 0, alpha, LWA_ALPHA)
            else:
                # 不透明模式，恢复正常窗口样式
                hwnd = int(self.winId())  # 转换为整数
                style = windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                style = style & ~WS_EX_LAYERED
                windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        
        except Exception as e:
            error_msg = f"切换背景模式错误: {e}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())  # 打印详细错误信息
            self.transcript_text.append(error_msg)
            self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))

    def connect_menu_signals(self):
        """连接菜单信号"""
        # 系统音频语言选择
        self.sys_en_action.triggered.connect(lambda: self.switch_to_system_audio('en'))
        self.sys_cn_action.triggered.connect(lambda: self.switch_to_system_audio('cn'))
        self.sys_auto_action.triggered.connect(lambda: self.switch_to_system_audio('auto'))
        
        # 文件转录
        self.file_audio_action.triggered.connect(self.switch_to_file_audio)
        
        # 模型选择
        self.en_small_action.triggered.connect(lambda: self.switch_model('en', 'small'))
        self.en_medium_action.triggered.connect(lambda: self.switch_model('en', 'medium'))
        self.en_large_action.triggered.connect(lambda: self.switch_model('en', 'large'))
        
        self.cn_small_action.triggered.connect(lambda: self.switch_model('cn', 'small'))
        self.cn_medium_action.triggered.connect(lambda: self.switch_model('cn', 'medium'))
        self.cn_large_action.triggered.connect(lambda: self.switch_model('cn', 'large'))
        
        # 背景模式
        self.opaque_action.triggered.connect(lambda: self.change_background_mode('opaque'))
        self.translucent_action.triggered.connect(lambda: self.change_background_mode('translucent'))
        self.transparent_action.triggered.connect(lambda: self.change_background_mode('transparent'))
    
    def switch_to_system_audio(self, mode):
        """切换到系统音频模式"""
        # 如果正在运行，先停止转录
        if self.is_running:
            self.stop_transcription()
        
        # 设置为系统音频模式
        self.is_system_audio = True
        
        # 切换识别模式
        self.switch_recognition_mode(mode)
        
        # 更新界面提示
        status_text = f"已切换到系统音频模式 ({mode} 识别)"
        self.transcript_text.append(status_text)
        self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
        
        # 滚动到底部
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )

    def switch_to_file_audio(self):
        """切换到文件转录模式"""
        # 如果正在运行，先停止转录
        if self.is_running:
            self.stop_transcription()
        
        # 设置为文件转录模式
        self.is_system_audio = False
        
        # 打开文件选择对话框
        self.select_file()

    def start_system_audio_transcription(self):
        """开始系统音频转录"""
        try:
            if self.is_running:
                    return
                
            self.is_running = True
            
            # 更新按钮文本
            self.start_button.setText("停止转录")
            self.start_button.setStyleSheet("background-color: rgba(150, 50, 50, 200); color: white;")
            
            # 显示进度条
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("准备中...")
            self.progress_bar.show()
            
            # 清空转录文本列表
            self.transcript_text = []
            
            # 添加单次状态消息 - 但不添加到转录文本列表
            lang_text = "英文" if self.recognition_mode == "en" else "中文"
            status_msg = f"开始系统音频转录 ({lang_text} 识别)..."
            
            # 直接设置到标签，但不添加到历史记录
            self.subtitle_label.setText(status_msg)
            
            # 创建识别器
            rec = vosk.KaldiRecognizer(self.model, 16000)
            rec.SetWords(True)
            
            # 如果启用了说话人识别，设置说话人识别模型
            if self.enable_speaker_id and self.speaker_identifier and self.speaker_identifier.spk_model:
                print("设置说话人识别模型到识别器")
                rec.SetSpkModel(self.speaker_identifier.spk_model)
            
            # 启动音频捕获线程
            self.audio_thread = threading.Thread(target=self.capture_audio, args=(rec,))
            self.audio_thread.daemon = True
            self.audio_thread.start()
                        
        except Exception as e:
            print(f"开始系统音频转录错误: {e}")
            import traceback
            print(traceback.format_exc())
            self.signals.new_text.emit(f"转录错误: {e}")
            self.is_running = False

    def start_file_transcription(self, file_path):
        """开始文件转录"""
        try:
            if self.is_running:
                return
            
            if not os.path.exists(file_path):
                self.subtitle_label.setText(f"错误: 文件不存在: {file_path}")
                return
            
            self.is_running = True
            
            # 更新按钮文本
            self.start_button.setText("停止转录")
            self.start_button.setStyleSheet("background-color: rgba(150, 50, 50, 200); color: white;")
            
            # 显示进度条
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("准备中...")
            self.progress_bar.show()
            
            # 清空转录文本列表
            self.transcript_text = []
            
            # 添加状态消息
            status_msg = f"开始转录文件: {os.path.basename(file_path)}"
            self.subtitle_label.setText(status_msg)
            print(status_msg)
            
            # 获取文件总时长
            try:
                duration = self.get_file_duration(file_path)
                print(f"文件总时长: {duration} 秒")
            except Exception as e:
                print(f"获取文件时长失败: {e}")
                duration = 0
            
            # 启动文件转录线程 - 使用非阻塞方式
            self.file_thread = threading.Thread(
                target=self.process_file_non_blocking,
                args=(file_path, duration)
            )
            self.file_thread.daemon = True
            self.file_thread.start()
            
        except Exception as e:
            print(f"开始文件转录错误: {e}")
            import traceback
            print(traceback.format_exc())
            self.signals.new_text.emit(f"转录错误: {e}")
            self.is_running = False
            
            # 更新按钮状态
            self.start_button.setText("开始转录")
            self.start_button.setStyleSheet("background-color: rgba(50, 150, 50, 200); color: white;")

    def process_file_non_blocking(self, file_path, duration):
        """非阻塞方式处理文件 - 使用子进程和管道"""
        try:
            # 创建临时文件用于存储转录结果
            temp_dir = os.path.dirname(file_path)
            temp_file = os.path.join(temp_dir, "temp_transcript.txt")
            
            # 创建识别器
            rec = self.create_recognizer()
            if not rec:
                self.signals.new_text.emit("创建识别器失败")
                self.signals.transcription_finished.emit()
                return
                
            # 使用 ffmpeg 提取音频并实时处理
            cmd = [
                "ffmpeg",
                "-i", file_path,
                "-ar", "16000",
                "-ac", "1",
                "-f", "s16le",
                "-"
            ]
            
            # 使用 subprocess.Popen 创建子进程
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=4096
            )
            
            # 初始化变量
            total_bytes = 0
            last_update_time = time.time()
            update_interval = 0.5  # 更新间隔（秒）
            
            # 创建实际转录内容列表
            self.actual_transcript = []
            
            # 读取并处理音频数据
            while self.is_running:
                # 读取一块音频数据
                audio_data = process.stdout.read(4000)
                
                # 如果没有更多数据，退出循环
                if not audio_data:
                    break
                    
                # 更新总字节数
                total_bytes += len(audio_data)
                
                # 处理音频数据
                if rec.AcceptWaveform(audio_data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "")
                    
                    if text:
                        # 添加标点符号和首字母大写
                        text = self.add_punctuation(text)
                        
                        # 添加到实际转录内容列表
                        self.actual_transcript.append(text)
                        
                        # 发送结果到UI
                        self.signals.new_text.emit(text)
                
                # 定期更新进度
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    # 计算进度
                    if duration > 0:
                        # 估算当前时间点
                        bytes_per_second = 32000  # 16kHz, 16-bit, mono = 32000 bytes/sec
                        current_second = total_bytes / bytes_per_second
                        progress = min(int(current_second / duration * 100), 99)
                        
                        # 格式化时间
                        current_time_str = self.format_time(current_second)
                        total_time_str = self.format_time(duration)
                        
                        # 更新进度条
                        progress_text = f"转录中: {current_time_str} / {total_time_str} ({progress}%)"
                        self.signals.progress_updated.emit(progress, progress_text)
                    
                    last_update_time = current_time
            
            # 获取最终结果
            final_result = json.loads(rec.FinalResult())
            final_text = final_result.get("text", "")
            
            if final_text:
                # 添加标点符号和首字母大写
                final_text = self.add_punctuation(final_text)
                
                # 添加到实际转录内容列表
                self.actual_transcript.append(final_text)
                
                # 发送结果到UI
                self.signals.new_text.emit(final_text)
            
            # 保存转录结果到临时文件
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write("\n".join(self.actual_transcript))
            
            # 发送完成信号
            self.signals.transcription_finished.emit()
            
            # 终止 ffmpeg 进程
            process.terminate()
            
        except Exception as e:
            print(f"处理文件错误: {e}")
            import traceback
            print(traceback.format_exc())
            self.signals.new_text.emit(f"处理文件错误: {e}")
            self.signals.transcription_finished.emit()
            
            # 尝试终止 ffmpeg 进程
            try:
                if 'process' in locals():
                    process.terminate()
            except:
                pass

    def add_punctuation(self, text):
        """添加标点符号和首字母大写"""
        if not text:
            return text
            
        # 首字母大写
        text = text[0].upper() + text[1:]
        
        # 如果文本不以标点符号结尾，添加句号
        if text[-1] not in ['.', '!', '?', ',', ';', ':', '-']:
            text += '.'
            
        return text

    def get_file_duration(self, file_path):
        """获取文件时长"""
        try:
            cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return float(result.stdout.strip())
        except Exception as e:
            print(f"获取文件时长失败: {e}")
            return 0

    def auto_start_transcription(self):
        """自动开始转录（程序启动后调用）"""
        try:
            # 检查模型是否已加载
            if self.model is None:
                print("模型尚未加载完成，延迟自动启动")
                # 如果模型尚未加载完成，再等待2秒
                QTimer.singleShot(2000, self.auto_start_transcription)
                return
                
            # 如果已经在运行，不需要再次启动
            if self.is_running:
                return
                
            # 添加自动启动提示
            self.transcript_text.append("系统已自动启动英文转录功能...")
            self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
            
            # 滚动到底部
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            )
            
            # 启动系统音频转录
            self.start_system_audio_transcription()
            
            print("自动启动转录功能完成")
            
        except Exception as e:
            print(f"自动启动转录错误: {e}")
            import traceback
            print(traceback.format_exc())

    def load_spk_model(self):
        """加载说话人识别模型"""
        try:
            # 检查模型目录是否存在
            if not os.path.exists(self.spk_model_path):
                error_msg = f"错误: 说话人识别模型目录不存在: {self.spk_model_path}"
                print(error_msg)
                self.transcript_text.append(error_msg)
                self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
                return False
            
            # 显示加载信息
            loading_msg = f"开始加载说话人识别模型..."
            print(loading_msg)
            self.transcript_text.append(loading_msg)
            self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
            
            # 强制更新界面
            QApplication.processEvents()
            
            # 加载模型
            self.speaker_identifier = SpeakerIdentifier(self.spk_model_path)
            
            if self.speaker_identifier.spk_model is None:
                error_msg = "说话人识别模型加载失败"
                print(error_msg)
                self.transcript_text.append(error_msg)
                self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
                self.speaker_enable_action.setChecked(False)
                return False
            
            # 显示成功信息
            success_msg = "成功加载说话人识别模型"
            print(success_msg)
            self.transcript_text.append(success_msg)
            self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
            
            return True
            
        except Exception as e:
            error_msg = f"加载说话人识别模型失败: {e}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            
            self.transcript_text.append(error_msg)
            self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
            
            self.speaker_identifier = None
            return False
    
    def toggle_speaker_id(self, checked):
        """切换说话人识别功能"""
        try:
            if checked:
                # 启用说话人识别
                if self.speaker_identifier is None:
                    # 显示加载信息
                    loading_msg = f"开始加载说话人识别模型..."
                    print(loading_msg)
                    self.transcript_text.append(loading_msg)
                    self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
                    
                    # 强制更新界面
                    QApplication.processEvents()
                    
                    # 在后台线程中加载模型
                    def load_model_thread():
                        try:
                            # 创建说话人识别器
                            self.speaker_identifier = SpeakerIdentifier(self.spk_model_path)
                            
                            # 检查模型是否加载成功
                            if self.speaker_identifier.spk_model is None:
                                error_msg = "说话人识别模型加载失败"
                                print(error_msg)
                                self.transcript_text.append(error_msg)
                                # 使用信号更新UI
                                self.signals.new_text.emit("PARTIAL:" + '\n'.join(self.transcript_text[-10:]))
                                # 重置复选框状态
                                self.speaker_enable_action.setChecked(False)
                                self.enable_speaker_id = False
                                return
                            
                            # 显示成功信息
                            success_msg = "成功加载说话人识别模型"
                            print(success_msg)
                            self.transcript_text.append(success_msg)
                            self.transcript_text.append("已启用说话人识别功能")
                            # 使用信号更新UI
                            self.signals.new_text.emit('\n'.join(self.transcript_text[-10:]))
                            
                            # 设置标志
                            self.enable_speaker_id = True
                            self.speaker_audio_buffer = b''  # 重置缓冲区
                            self.current_speaker_id = None  # 重置当前说话人
                            
                            print("已启用说话人识别功能")
                            
                            # 如果正在运行，需要重新启动转录以应用更改
                            if self.is_running:
                                print("重新启动转录以应用说话人识别设置...")
                                # 使用信号在主线程中安全地重启转录
                                QTimer.singleShot(0, lambda: self.restart_transcription())
                        except Exception as e:
                            error_msg = f"加载说话人识别模型错误: {e}"
                            print(error_msg)
                            import traceback
                            print(traceback.format_exc())
                            self.transcript_text.append(error_msg)
                            # 使用信号更新UI
                            self.signals.new_text.emit('\n'.join(self.transcript_text[-10:]))
                            # 重置复选框状态
                            self.speaker_enable_action.setChecked(False)
                            self.enable_speaker_id = False
                    
                    # 启动加载线程
                    load_thread = threading.Thread(target=load_model_thread)
                    load_thread.daemon = True
                    load_thread.start()
                    
                    # 暂时返回，等待线程完成加载
                    return
                
                # 如果模型已加载，直接启用
                self.enable_speaker_id = True
                self.speaker_audio_buffer = b''  # 重置缓冲区
                self.current_speaker_id = None  # 重置当前说话人
                self.transcript_text.append("已启用说话人识别功能")
                print("已启用说话人识别功能")
                
                # 更新显示
                self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
            else:
                # 禁用说话人识别
                self.enable_speaker_id = False
                self.transcript_text.append("已禁用说话人识别功能")
                print("已禁用说话人识别功能")
                
                # 更新显示
                self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
            
            # 如果正在运行，需要重新启动转录以应用更改
            if self.is_running:
                print("重新启动转录以应用说话人识别设置...")
                self.restart_transcription()
                
        except Exception as e:
            error_msg = f"切换说话人识别功能错误: {e}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            self.transcript_text.append(error_msg)
            self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
            # 重置复选框状态
            self.speaker_enable_action.setChecked(False)
            self.enable_speaker_id = False

    def restart_transcription(self):
        """重新启动转录以应用设置更改"""
        try:
            if self.is_running:
                self.stop_transcription()
                # 使用更长的延迟确保完全停止
                QTimer.singleShot(1000, self.start_system_audio_transcription)
        except Exception as e:
            print(f"重启转录错误: {e}")

    def get_speaker_id(self, spk_data):
        """根据说话人识别结果获取说话人ID"""
        try:
            # 说话人数据是一个浮点数列表，表示说话人嵌入向量
            # 我们需要将其与已知说话人进行比较
            
            # 如果没有已知说话人，创建第一个说话人
            if not hasattr(self, 'speaker_embeddings'):
                self.speaker_embeddings = []
                self.speaker_embeddings.append(spk_data)
                return 1  # 第一个说话人
            
            # 计算与已知说话人的相似度
            max_similarity = -1
            best_speaker_id = -1
            
            for i, known_embedding in enumerate(self.speaker_embeddings):
                similarity = self.compute_similarity(spk_data, known_embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_speaker_id = i + 1  # 说话人ID从1开始
            
            # 如果相似度低于阈值，认为是新说话人
            if max_similarity < 0.7:  # 阈值可以调整
                self.speaker_embeddings.append(spk_data)
                return len(self.speaker_embeddings)
            
            return best_speaker_id
            
        except Exception as e:
            print(f"获取说话人ID错误: {e}")
            return "?"
    
    def compute_similarity(self, embedding1, embedding2):
        """计算两个说话人嵌入向量的余弦相似度"""
        try:
            # 转换为numpy数组
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # 计算余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0
                
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            print(f"计算相似度错误: {e}")
            return 0

    def load_config(self):
        """加载配置文件"""
        try:
            # 首先尝试加载用户配置文件
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    print(f"已加载用户配置文件: {config_path}")
                    return config
            
            # 如果用户配置文件不存在，加载默认配置文件
            default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "default_config.json")
            if os.path.exists(default_config_path):
                with open(default_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    print(f"已加载默认配置文件: {default_config_path}")
                    return config
            
            # 如果两个配置文件都不存在，使用硬编码的默认配置
            print("警告: 未找到配置文件，使用硬编码的默认配置")
            return {
                "models": {
                    "vosk": "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\model\\vosk-model-small-en-us-0.15",
                    "sherpa": "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\models\\sherpa-onnx",
                    "sherpa_std": "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\models\\sherpa-onnx",
                    "speaker": "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\vosk-model-spk-0.4",
                    "argos": "C:\\Users\\crige\\.local\\share\\argos-translate\\packages\\translate-en_zh-1_9\\model",
                    "opus_mt": "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\models\\opus-mt\\en-zh"
                }
            }
        except Exception as e:
            print(f"加载配置文件错误: {e}")
            import traceback
            print(traceback.format_exc())
            
            # 返回硬编码的默认配置
            return {
                "models": {
                    "vosk": "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\model\\vosk-model-small-en-us-0.15",
                    "sherpa": "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\models\\sherpa-onnx",
                    "sherpa_std": "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\models\\sherpa-onnx",
                    "speaker": "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\vosk-model-spk-0.4",
                    "argos": "C:\\Users\\crige\\.local\\share\\argos-translate\\packages\\translate-en_zh-1_9\\model",
                    "opus_mt": "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\models\\opus-mt\\en-zh"
                }
            }

# 修改 SpeakerIdentifier 类
class SpeakerIdentifier:
    def __init__(self, model_path):
        """初始化说话人识别器"""
        self.model_path = model_path
        self.spk_model = None
        self.speaker_embeddings = []  # 存储已知说话人的嵌入向量
        self.speaker_names = []  # 存储说话人名称
        self.distance_threshold = 0.35  # 调整阈值
        self.min_frames = 40  # 最小帧数要求
        self.max_speakers = 2  # 限制最大说话人数量为2
        self.load_model()
    
    def load_model(self):
        """加载说话人识别模型"""
        try:
            if not os.path.exists(self.model_path):
                print(f"错误: 说话人识别模型目录不存在: {self.model_path}")
                return False
            
            print("正在加载说话人识别模型...")
            self.spk_model = SpkModel(self.model_path)
            print("说话人识别模型加载成功")
            return True
        except Exception as e:
            print(f"加载说话人识别模型失败: {e}")
            import traceback
            print(traceback.format_exc())
            self.spk_model = None
            return False
    
    def identify_speaker(self, audio_data):
        """识别说话人"""
        try:
            if self.spk_model is None:
                return None
            
            # 创建一个临时识别器，专门用于提取说话人特征
            # 使用小型英文模型，减少资源占用
            try:
                # 尝试使用已加载的小型模型
                model_path = "vosk-model-small-en-us-0.15"
                if os.path.exists(model_path):
                    model = Model(model_path)
                else:
                    # 如果找不到小型模型，使用默认英文模型
                    model = Model(lang="en-us")
                    
                print("成功创建临时识别器模型")
                rec = KaldiRecognizer(model, 16000)
                rec.SetSpkModel(self.spk_model)
                print("成功设置说话人模型到临时识别器")
            except Exception as e:
                print(f"创建临时识别器失败: {e}")
                import traceback
                print(traceback.format_exc())
                return None
            
            # 处理音频数据
            try:
                print(f"处理音频数据进行说话人识别，大小: {len(audio_data)} 字节")
                if rec.AcceptWaveform(audio_data):
                    result = json.loads(rec.Result())
                    print(f"临时识别器结果: {result}")
                    if 'spk' in result and 'spk_frames' in result:
                        spk_data = result['spk']
                        spk_frames = result['spk_frames']
                        
                        print(f"说话人特征提取成功，基于 {spk_frames} 帧")
                        
                        # 如果帧数太少，结果不可靠
                        if spk_frames < self.min_frames:
                            print(f"警告: 帧数 ({spk_frames}) 太少，识别结果可能不可靠")
                            return None
                        
                        # 获取说话人ID
                        return self.get_speaker_id(spk_data)
                
                # 尝试获取最终结果
                final_result = json.loads(rec.FinalResult())
                print(f"临时识别器最终结果: {final_result}")
                if 'spk' in final_result and 'spk_frames' in final_result:
                    spk_data = final_result['spk']
                    spk_frames = final_result['spk_frames']
                    
                    print(f"最终结果中提取到说话人特征，基于 {spk_frames} 帧")
                    
                    # 如果帧数太少，结果不可靠
                    if spk_frames < self.min_frames:
                        print(f"警告: 帧数 ({spk_frames}) 太少，识别结果可能不可靠")
                        return None
                    
                    # 获取说话人ID
                    return self.get_speaker_id(spk_data)
            except Exception as e:
                print(f"处理音频数据失败: {e}")
                import traceback
                print(traceback.format_exc())
                
            return None
        except Exception as e:
            print(f"识别说话人错误: {e}")
            import traceback
            print(traceback.format_exc())
            return None
    
    def get_speaker_id(self, spk_data):
        """根据说话人嵌入向量获取说话人ID"""
        try:
            # 如果没有已知说话人，创建第一个说话人
            if not self.speaker_embeddings:
                self.speaker_embeddings.append(spk_data)
                self.speaker_names.append("男声")  # 默认第一个说话人为男声
                print("添加第一个说话人特征 (男声)")
                return 1  # 第一个说话人
            
            # 计算与已知说话人的距离
            distances = []
            for i, known_embedding in enumerate(self.speaker_embeddings):
                distance = self.cosine_distance(spk_data, known_embedding)
                print(f"与{self.speaker_names[i]}的距离: {distance:.4f}")
                distances.append((i+1, distance))
            
            # 按距离排序
            distances.sort(key=lambda x: x[1])
            best_speaker_id, min_distance = distances[0]
            
            # 如果距离大于阈值且当前说话人数量小于最大限制，认为是新说话人
            if min_distance > self.distance_threshold and len(self.speaker_embeddings) < self.max_speakers:
                self.speaker_embeddings.append(spk_data)
                new_id = len(self.speaker_embeddings)
                
                # 如果已经有一个说话人，第二个默认为女声
                if new_id == 2:
                    self.speaker_names.append("女声")
                    print(f"添加第二个说话人特征 (女声)")
                else:
                    self.speaker_names.append(f"说话人{new_id}")
                    print(f"添加新说话人 {new_id}")
                
                print(f"新说话人距离: {min_distance:.4f}")
                return new_id
            
            # 如果已经有两个说话人，强制选择最接近的一个
            print(f"识别为已知说话人 {best_speaker_id} ({self.speaker_names[best_speaker_id-1]}, 距离: {min_distance:.4f})")
            
            # 更新说话人特征向量（使用滑动平均）
            # 这有助于适应说话人声音的微小变化
            alpha = 0.8  # 权重因子
            old_embedding = self.speaker_embeddings[best_speaker_id-1]
            new_embedding = [alpha * old + (1-alpha) * new for old, new in zip(old_embedding, spk_data)]
            self.speaker_embeddings[best_speaker_id-1] = new_embedding
            
            return best_speaker_id
        except Exception as e:
            print(f"获取说话人ID错误: {e}")
            import traceback
            print(traceback.format_exc())
            return 1  # 出错时返回第一个说话人
    
    def cosine_distance(self, embedding1, embedding2):
        """计算两个说话人嵌入向量的余弦距离"""
        try:
            # 转换为numpy数组
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # 计算余弦距离 (1 - 余弦相似度)
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 1.0  # 最大距离
                
            similarity = dot_product / (norm1 * norm2)
            return 1.0 - similarity
        except Exception as e:
            print(f"计算距离错误: {e}")
            return 1.0  # 出错时返回最大距离

if __name__ == "__main__":
        app = QApplication(sys.argv)
        window = SubtitleWindow()
        window.show()
        sys.exit(app.exec_())