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
                            QTabWidget, QRadioButton, QButtonGroup, QShortcut, QLineEdit, QProgressBar, QScrollArea, QMenuBar, QMenu, QAction, QActionGroup, QSizePolicy, QGraphicsOpacityEffect, QMessageBox)
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
import sherpa_onnx
import pythoncom  # 用于 COM 初始化
import random  # 添加随机模块用于滚动控制

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
        # 在 __init__ 开头添加 COM 初始化
        pythoncom.CoInitialize()  # 初始化 COM
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
        self.subtitle_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.subtitle_label.setStyleSheet("color: red; background-color: rgba(0, 0, 0, 150); padding: 10px; border-radius: 10px;")
        self.subtitle_label.setFont(QFont("Arial", 20, QFont.Bold))  # 增加字体大小
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setTextInteractionFlags(Qt.TextSelectableByMouse)  # 允许选择文本
        self.subtitle_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 允许标签扩展
        
        # 添加字幕标签到容器
        self.transcript_layout.addWidget(self.subtitle_label, 1)  # 使用拉伸因子1
        
        # 设置滚动区域的内容
        self.scroll_area.setWidget(self.transcript_container)
        
        # 确保滚动条可见
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
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
        
        # 设置按钮初始状态
        self.start_button.setText("开始转录")
        self.start_button.setStyleSheet("background-color: rgba(50, 150, 50, 200); color: white;")
        
        # 添加转录内容专用列表
        self.actual_transcript = []  # 仅存储实际转录内容，不包含系统信息
        self.transcript_text = []    # 确保这个列表也被初始化
        
        # 在资源检查和模型加载完成后自动开始转录 - 修改为不自动启动
        # QTimer.singleShot(5000, self.auto_start_transcription)
        # 改为只检查资源，不自动启动
        QTimer.singleShot(5000, self.check_system_resources)
        
        # 添加说话人识别器
        self.speaker_identifier = None
        self.spk_model_path = "vosk-model-spk-0.4"
        self.enable_speaker_id = False  # 默认不启用说话人识别
        self.speaker_audio_buffer = b''  # 用于累积足够的音频数据进行说话人识别
        self.min_speaker_audio_size = 64000  # 增加到4秒的音频数据
        self.current_speaker_id = None  # 当前说话人ID
        
        # 添加模型路径配置
        self.model_paths = {
            # ASR 模型
            "vosk": "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\model\\vosk-model-small-en-us-0.15",
            "sherpa": "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\models\\sherpa-onnx",  # 两种模型共用同一目录
            
            # RTM 模型
            "opus_mt": "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\models\\opus-mt\\en-zh",
            "argos": "C:\\Users\\crige\\.local\\share\\argos-translate\\packages\\translate-en_zh-1_9\\model\\model.bin"
        }
        
        # 当前使用的模型 - 确保默认为 vosk
        self.current_asr_model = "vosk"  # 默认使用 vosk
        self.current_rtm_model = "argos"  # 默认使用 argostranslate
        
        # 初始化 Sherpa-ONNX 相关属性
        self.recognizer = None
        self.stream = None
        
        # 初始化 VOSK 模型
        self.model = None
        self.rec = None
    
    def create_menu_bar(self):
        """创建菜单栏"""
        # 创建菜单栏
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)
        
        # 1. 转录模式菜单
        self.mode_menu = self.menu_bar.addMenu("转录模式")
        
        # 系统音频子菜单
        self.system_audio_menu = QMenu("系统音频", self)
        self.mode_menu.addMenu(self.system_audio_menu)
        
        # 添加识别选项（修改变量名以匹配）
        self.sys_en_action = QAction("英文识别", self)
        self.sys_cn_action = QAction("中文识别", self)
        self.sys_auto_action = QAction("自动识别", self)
        
        self.system_audio_menu.addAction(self.sys_en_action)
        self.system_audio_menu.addAction(self.sys_cn_action)
        self.system_audio_menu.addAction(self.sys_auto_action)
        
        # 添加音频/视频文件选项
        self.file_audio_action = QAction("音频/视频文件", self)  # 修改变量名
        self.mode_menu.addAction(self.file_audio_action)
        
        # 2. 音频设备菜单
        self.device_menu = self.menu_bar.addMenu("音频设备")
        self.update_audio_devices()  # 动态生成设备列表
        
        # 3. 模型选择菜单
        self.model_menu = self.menu_bar.addMenu("模型选择")
        
        # ASR 模型子菜单
        self.asr_menu = QMenu("ASR 模型", self)
        self.model_menu.addMenu(self.asr_menu)
        
        # 添加 ASR 模型选项
        self.vosk_action = QAction("VOSK Small 模型", self)
        self.sherpa_action = QAction("Sherpa-ONNX量化模型", self)
        self.sherpa_std_action = QAction("Sherpa-ONNX标准模型", self)  # 添加新菜单项
        
        self.asr_menu.addAction(self.vosk_action)
        self.asr_menu.addAction(self.sherpa_action)
        self.asr_menu.addAction(self.sherpa_std_action)  # 添加到菜单
        
        # RTM 模型子菜单
        self.rtm_menu = QMenu("RTM 模型", self)
        self.model_menu.addMenu(self.rtm_menu)
        
        # 添加 RTM 模型选项
        self.argos_action = QAction("Argostranslate 模型", self)
        self.opus_action = QAction("Opus-Mt-ONNX 模型", self)
        self.rtm_menu.addAction(self.argos_action)
        self.rtm_menu.addAction(self.opus_action)
        
        # 4. 背景模式菜单
        self.bg_menu = self.menu_bar.addMenu("背景模式")
        
        # 添加背景模式选项
        self.opaque_action = QAction("不透明", self)
        self.semi_action = QAction("半透明", self)
        self.trans_action = QAction("全透明", self)
        
        self.bg_menu.addAction(self.opaque_action)
        self.bg_menu.addAction(self.semi_action)
        self.bg_menu.addAction(self.trans_action)
        
        # 5. 字体大小菜单
        self.font_menu = self.menu_bar.addMenu("字体大小")
        
        # 添加字体大小选项
        self.small_action = QAction("小", self)
        self.medium_action = QAction("中", self)
        self.large_action = QAction("大", self)
        
        self.font_menu.addAction(self.small_action)
        self.font_menu.addAction(self.medium_action)
        self.font_menu.addAction(self.large_action)
        
        # 6. 调试菜单
        self.debug_menu = self.menu_bar.addMenu("调试")
        
        # 添加调试选项
        self.sys_info_action = QAction("显示系统信息", self)
        self.check_model_action = QAction("检查模型目录", self)
        self.test_audio_action = QAction("测试音频设备", self)
        self.search_doc_action = QAction("搜索模型文档", self)
        
        self.debug_menu.addAction(self.sys_info_action)
        self.debug_menu.addAction(self.check_model_action)
        self.debug_menu.addAction(self.test_audio_action)
        self.debug_menu.addAction(self.search_doc_action)
        
        # 7. 说话人识别菜单
        self.speaker_menu = self.menu_bar.addMenu("说话人识别")
        
        # 添加说话人识别选项
        self.enable_speaker_action = QAction("启用说话人识别", self)
        self.enable_speaker_action.setCheckable(True)
        self.speaker_menu.addAction(self.enable_speaker_action)
        
        # 连接信号
        self.connect_menu_signals()
    
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
        """加载模型 - 使用预定义的路径"""
        try:
            # 使用预定义的路径，不要动态生成
            model_path = self.model_paths["vosk"]
            model_name = os.path.basename(model_path)
            
            # 显示加载信息
            if not keep_info:
                self.subtitle_label.setText(f"开始加载模型: {model_name}")
            else:
                self.subtitle_label.setText(self.system_info + f"\n开始加载模型: {model_name}")
            
            print(f"开始加载模型: {model_path}")
            
            # 检查路径是否存在
            if not os.path.exists(model_path):
                error_msg = f"错误: VOSK 模型路径不存在: {model_path}"
                print(error_msg)
                
                if not keep_info:
                    self.subtitle_label.setText(error_msg)
                else:
                    self.subtitle_label.setText(self.system_info + "\n" + error_msg)
                
                return False
            
            # 加载模型
            self.model = vosk.Model(model_path)
            
            # 测试模型 - 创建识别器
            test_rec = vosk.KaldiRecognizer(self.model, 16000)
            if test_rec:
                print("模型测试成功: 能够创建识别器")
                
            # 更新显示
            if not keep_info:
                self.subtitle_label.setText(f"成功加载模型: {model_name}")
            else:
                self.subtitle_label.setText(self.system_info + f"\n成功加载模型: {model_name}")
            
            print(f"成功加载模型: {model_path}")
            
            # 确保当前 ASR 模型设置为 VOSK
            self.current_asr_model = "vosk"
            
            return True
            
        except Exception as e:
            error_msg = f"加载模型失败: {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            
            if not keep_info:
                self.subtitle_label.setText(error_msg)
            else:
                self.subtitle_label.setText(self.system_info + "\n" + error_msg)
            
            return False

    def create_recognizer(self):
        """创建 VOSK 识别器 - 与 voskIS_Normal.py 保持一致"""
        try:
            if not self.model:
                print("错误: 模型未加载，无法创建识别器")
                return None
                
            # 创建识别器
            rec = vosk.KaldiRecognizer(self.model, 16000)
            rec.SetWords(True)  # 始终启用词级别时间戳，不考虑语言
            
            return rec
        except Exception as e:
            print(f"创建识别器失败: {e}")
            return None

    def process_audio(self, audio_data):
        """处理音频数据"""
        try:
            # 1. 音频预处理
            if isinstance(audio_data, np.ndarray):
                # 转换为单声道
                if audio_data.ndim > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # 音频电平检测
                audio_level = np.max(np.abs(audio_data))
                if audio_level < 0.0005:  # 优化后的阈值
                    return
            
            # 2. 根据当前ASR模型类型选择处理方法
            if self.current_asr_model == "vosk":
                result = self.process_vosk_audio(audio_data)
            elif self.current_asr_model == "sherpa":
                result = self.process_sherpa_audio(audio_data)
            else:
                return
            
            # 3. 处理识别结果
            if result:
                # 应用文本处理
                text = self.process_recognition_result(result)
                
                # 处理说话人识别
                if self.enable_speaker_id:
                    text = self.add_speaker_identification(text, audio_data)
                
                # 应用翻译（如果启用）
                if self.enable_translation:
                    text = self.apply_translation(text)
                
                # 发送结果
                self.signals.new_text.emit(text)
                
        except Exception as e:
            print(f"音频处理错误: {e}")
            import traceback
            print(traceback.format_exc())
    
    def toggle_transcription(self):
        """切换转录状态"""
        try:
            if self.is_running:
                print("停止转录...")
                self.stop_transcription()
            else:
                print("开始转录...")
                # 确保当前 ASR 模型设置正确
                if not hasattr(self, 'current_asr_model') or not self.current_asr_model:
                    self.current_asr_model = "vosk"  # 默认使用 vosk
                    print(f"未设置 ASR 模型，使用默认模型: {self.current_asr_model}")
                
                # 根据当前模型类型选择不同的启动方法
                if self.current_asr_model == "vosk":
                    self.start_vosk_transcription()
                elif self.current_asr_model in ["sherpa", "sherpa_std"]:
                    self.start_sherpa_transcription()
                else:
                    print(f"未知的 ASR 模型类型: {self.current_asr_model}")
                    QMessageBox.warning(self, "错误", f"未知的 ASR 模型类型: {self.current_asr_model}")
        except Exception as e:
            print(f"切换转录状态失败: {e}")
            import traceback
            print(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"切换转录状态失败: {str(e)}")

    def start_transcription(self):
        """开始转录 - 根据当前模型类型选择不同的启动方式"""
        try:
            print(f"开始转录，当前 ASR 模型: {self.current_asr_model}")
            
            # 检查音频设备
            if not self.current_device:
                print("错误: 未选择音频设备，尝试使用默认设备")
                try:
                    speakers = sc.all_speakers()
                    if speakers:
                        self.current_device = speakers[0]
                        print(f"已自动选择设备: {self.current_device.name}")
                    else:
                        print("错误: 未找到可用的音频设备")
                        QMessageBox.warning(self, "错误", "未找到可用的音频设备")
                        return
                except Exception as e:
                    print(f"获取音频设备失败: {e}")
                    QMessageBox.warning(self, "错误", f"获取音频设备失败: {str(e)}")
                    return
            
            # 创建音频捕获
            try:
                self.mic = sc.get_microphone(
                    id=str(self.current_device.id),
                    include_loopback=True
                ).recorder(samplerate=16000)
                print(f"成功创建音频捕获: {self.current_device.name}")
            except Exception as e:
                print(f"创建音频捕获失败: {e}")
                QMessageBox.warning(self, "错误", f"创建音频捕获失败: {str(e)}")
                return
            
            # 根据当前模型类型加载模型
            if self.current_asr_model in ["sherpa", "sherpa_std"]:
                if self.recognizer is None or self.stream is None:
                    print("加载 Sherpa-ONNX 模型...")
                    success = self.load_sherpa_model()
                    if not success:
                        print("加载 Sherpa-ONNX 模型失败")
                        QMessageBox.warning(self, "错误", "加载 Sherpa-ONNX 模型失败")
                        return
            elif self.current_asr_model == "vosk":
                if self.model is None:
                    print("加载 VOSK 模型...")
                    success = self.load_vosk_model()
                    if not success:
                        print("加载 VOSK 模型失败")
                        QMessageBox.warning(self, "错误", "加载 VOSK 模型失败")
                        return
                
                    # 创建 VOSK 识别器
                    self.rec = self.create_recognizer()
                    if not self.rec:
                        print("创建 VOSK 识别器失败")
                        QMessageBox.warning(self, "错误", "创建 VOSK 识别器失败")
                        return
            
            # 启动音频处理线程 - 使用统一的处理方法
            self.is_running = True
            self.audio_thread = threading.Thread(target=self.process_audio_stream)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            # 更新按钮状态
            self.start_button.setText("停止转录")
            self.start_button.setStyleSheet("background-color: rgba(150, 50, 50, 200); color: white;")
            
        except Exception as e:
            print(f"启动转录失败: {e}")
            import traceback
            print(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"启动转录失败: {str(e)}")

    def stop_transcription(self):
        """停止转录 - 根据当前模型类型选择不同的清理方法"""
        # 如果不在运行状态，直接返回
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 清理音频资源
        if hasattr(self, 'mic') and self.mic:
            self.mic = None
        
        # 根据当前模型类型选择不同的清理方法
        if self.current_asr_model in ["sherpa", "sherpa_std"]:
            # Sherpa 模型不需要特殊清理
            pass
        elif self.current_asr_model == "vosk":
            # VOSK 模型可能需要特殊清理
            pass
        
        # 等待线程结束
        if self.audio_thread:
            self.audio_thread.join(timeout=1)
            self.audio_thread = None
        
        # 保存转录文本
        if self.transcript_text and len(self.transcript_text) > 1:
            saved_path = self.save_transcript(use_timestamp=True)
            if saved_path:
                print(f"成功保存转录文件到: {saved_path}")
                self.transcript_text.append(f"转录已完成！文件已保存到: {saved_path}")
                self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
                
                # 滚动到底部
                self.scroll_area.verticalScrollBar().setValue(
                    self.scroll_area.verticalScrollBar().maximum()
                )
        
        # 重置状态
        self.start_button.setText("开始转录")
        self.start_button.setStyleSheet("background-color: rgba(50, 150, 50, 200); color: white;")
        
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
        """添加标点符号和首字母大写 - 从 voskIS_Normal.py 复制"""
        if not text:
            return text
            
        # 首字母大写
        text = text[0].upper() + text[1:]
        
        # 如果文本不以标点符号结尾，添加句号
        if text[-1] not in ['.', '!', '?', ',', ';', ':', '-']:
            text += '.'
        
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
                if not partial_text.strip():
                    return  # 忽略空的部分结果
                
                # 保留当前显示的完整结果，只更新部分结果
                display_text = []
                for line in self.transcript_text[-8:]:  # 减少显示行数，确保有足够空间
                    if not line.startswith("PARTIAL:"):
                        display_text.append(line)
                
                # 添加当前的部分结果
                display_text.append(f"正在识别: {partial_text}")
                
                # 更新显示 - 使用单个换行符而不是双换行符
                self.subtitle_label.setText('\n'.join(display_text))
                
                # 强制滚动到底部，确保最新内容可见
                self.scroll_area.verticalScrollBar().setValue(
                    self.scroll_area.verticalScrollBar().maximum())
            else:
                # 处理完整结果 - 检查是否与最后一个结果相同或相似
                if self.transcript_text and (text == self.transcript_text[-1] or self._is_similar(text, self.transcript_text[-1])):
                    # 如果是重复或非常相似的文本，不添加到列表
                    print(f"跳过重复文本: {text}")
                    return
                
                # 添加新的完整结果到转录文本列表
                # 检查是否是状态信息，如果是则不添加到转录文本列表
                if not text.startswith("已启用") and not text.startswith("已禁用") and \
                   not text.startswith("开始") and not text.startswith("转录") and \
                   not text.startswith("加载") and not text.startswith("成功"):
                    self.transcript_text.append(text)
                    
                    # 添加到实际转录内容列表 - 只保存真正的转录内容
                    if not text.startswith("系统") and not text.startswith("警告") and \
                       not text.startswith("错误") and not text.startswith("模型") and \
                       "已保存到" not in text and text.strip():
                        # 确保不是空文本
                        if text not in self.actual_transcript:
                            self.actual_transcript.append(text)
                    
                # 限制历史记录最多保留500行
                if len(self.transcript_text) > 500:
                    self.transcript_text = self.transcript_text[-500:]
                
                # 计算当前字幕标签可以显示的行数
                font_metrics = self.subtitle_label.fontMetrics()
                label_height = self.subtitle_label.height()
                line_height = font_metrics.lineSpacing()
                visible_lines = max(3, int(label_height / line_height) - 1)
                
                # 只显示最近的几条完整结果
                display_text = self.transcript_text[-visible_lines:]
                
                # 更新显示
                self.subtitle_label.setText('\n'.join(display_text))
                
                # 强制滚动到底部
                self.scroll_area.verticalScrollBar().setValue(
                    self.scroll_area.verticalScrollBar().maximum())
                
                # 打印调试信息
                print(f"更新字幕: {text}")
                print(f"当前转录文本数量: {len(self.transcript_text)}")
                print(f"当前实际转录内容数量: {len(self.actual_transcript)}")
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
            # 创建转录目录
            transcript_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcripts")
            os.makedirs(transcript_dir, exist_ok=True)
            
            # 生成文件名
            model_name = "en-small" if self.model_language == 'en' else "cn-small"
            if use_timestamp:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"transcript_{model_name}_{timestamp}.txt"
            else:
                filename = f"transcript_{model_name}.txt"
            
            # 完整文件路径
            file_path = os.path.join(transcript_dir, filename)
            
            # 保存文件 - 使用实际转录内容而非所有文本
            with open(file_path, 'w', encoding='utf-8') as f:
                if hasattr(self, 'actual_transcript') and self.actual_transcript:
                    f.write('\n'.join(self.actual_transcript))
                else:
                    # 兼容旧代码，如果没有实际转录内容列表，则过滤系统信息
                    filtered_text = [line for line in self.transcript_text 
                                   if not line.startswith("系统") and not line.startswith("警告") and 
                                      not line.startswith("错误") and not line.startswith("模型") and
                                      "已保存到" not in line]
                    f.write('\n'.join(filtered_text))
            
            return file_path
        except Exception as e:
            print(f"保存转录文件错误: {e}")
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
                # 使用 sys_en_action 而不是不存在的 system_audio_action
                self.sys_en_action.setChecked(True)
                self.subtitle_label.setText("已取消文件选择，保持当前模式")
            
        except Exception as e:
            print(f"选择文件错误: {e}")
            # 发生错误时切换回系统音频模式
            self.is_system_audio = True
            # 使用 sys_en_action 而不是不存在的 system_audio_action
            self.sys_en_action.setChecked(True)
            self.subtitle_label.setText(f"文件选择出错: {e}")

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
        try:
            # 系统音频语言选择
            self.sys_en_action.triggered.connect(lambda: self.switch_to_system_audio('en'))
            self.sys_cn_action.triggered.connect(lambda: self.switch_to_system_audio('cn'))
            self.sys_auto_action.triggered.connect(lambda: self.switch_to_system_audio('auto'))
            
            # 文件转录
            self.file_audio_action.triggered.connect(self.switch_to_file_audio)
            
            # 模型选择
            self.vosk_action.triggered.connect(lambda: self.switch_asr_model("vosk"))
            self.sherpa_action.triggered.connect(lambda: self.switch_asr_model("sherpa"))
            self.sherpa_std_action.triggered.connect(lambda: self.switch_asr_model("sherpa_std"))  # 连接新的信号
            
            # RTM 模型切换
            self.argos_action.triggered.connect(lambda: self.switch_rtm_model("argos"))
            self.opus_action.triggered.connect(lambda: self.switch_rtm_model("opus"))
            
            # 背景模式
            self.opaque_action.triggered.connect(lambda: self.change_background_mode('opaque'))
            self.semi_action.triggered.connect(lambda: self.change_background_mode('translucent'))
            self.trans_action.triggered.connect(lambda: self.change_background_mode('transparent'))
            
            # 字体大小
            self.small_action.triggered.connect(lambda: self.change_font_size('small'))
            self.medium_action.triggered.connect(lambda: self.change_font_size('medium'))
            self.large_action.triggered.connect(lambda: self.change_font_size('large'))
            
            # 调试选项
            self.sys_info_action.triggered.connect(self.show_system_info)
            self.check_model_action.triggered.connect(self.check_model_directories)
            self.test_audio_action.triggered.connect(self.test_audio_device)
            self.search_doc_action.triggered.connect(self.search_model_docs)
            
            # 说话人识别
            self.enable_speaker_action.triggered.connect(self.toggle_speaker_recognition)
            
        except Exception as e:
            print(f"连接菜单信号错误: {e}")
            import traceback
            print(traceback.format_exc())
    
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
        """开始系统音频转录 - 通用方法，适用于 VOSK 和 Sherpa-ONNX"""
        try:
            # 检查设备
            if not self.current_device:
                print("错误: 未选择音频设备，尝试使用默认设备")
                # 尝试获取默认设备
                speakers = sc.all_speakers()
                if speakers:
                    self.current_device = speakers[0]
                    print(f"已自动选择设备: {self.current_device.name}")
                else:
                    print("错误: 未找到可用的音频设备")
                    return
                
            # 创建音频捕获
            self.mic = sc.get_microphone(
                id=str(self.current_device.id),
                include_loopback=True
            ).recorder(samplerate=16000)
            
            print(f"找到音频捕获设备: {self.current_device.name}")
            print("成功创建音频捕获")
            
            # 如果是 VOSK 模型，创建识别器
            if self.current_asr_model == "vosk":
                self.rec = self.create_recognizer()
                if not self.rec:
                    print("创建 VOSK 识别器失败")
                    return
                print("成功创建 VOSK 识别器")
            
            # 启动音频处理线程
            self.is_running = True
            
            # 根据当前模型类型选择处理方法
            if self.current_asr_model in ["sherpa", "sherpa_std"]:
                self.audio_thread = threading.Thread(target=self.process_sherpa_stream)
            else:  # 默认使用 VOSK 处理
                self.audio_thread = threading.Thread(target=self.process_vosk_stream)
                
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            # 更新按钮状态
            self.start_button.setText("停止转录")
            self.start_button.setStyleSheet("background-color: rgba(150, 50, 50, 200); color: white;")
            
        except Exception as e:
            print(f"开始系统音频转录错误: {e}")
            import traceback
            print(traceback.format_exc())
            self.is_running = False

    def process_sherpa_stream(self):
        """处理 Sherpa-ONNX 音频流"""
        try:
            with self.mic as mic:
                while self.is_running:
                    data = mic.record(numframes=8000)
                    
                    # 检查音频电平
                    audio_level = np.max(np.abs(data))
                    if audio_level < 0.0005:
                        continue
                        
                    print(f"音频电平: {audio_level:.4f}")
                    
                    # 处理音频数据
                    result = self.process_sherpa_audio(data)
                    if result:
                        # 确保结果是字符串
                        if isinstance(result, str):
                            text = result
                        else:
                            text = result.text if hasattr(result, 'text') else str(result)
                        
                        print(f"识别结果: {text}")  # 添加调试输出
                        
                        # 发送结果到UI
                        self.signals.new_text.emit(text)
                        
                        # 强制更新UI
                        QApplication.processEvents()
                        
                        # 强制滚动到底部
                        QTimer.singleShot(10, lambda: self.scroll_area.verticalScrollBar().setValue(
                            self.scroll_area.verticalScrollBar().maximum()))
                    else:
                        print("未获取到识别结果")  # 添加调试输出
                    
                    # 即使没有结果，也定期更新UI以保持响应性
                    if audio_level > 0.01:
                        QApplication.processEvents()
                        
                        # 定期强制滚动到底部
                        if random.random() < 0.2:  # 20%的概率执行滚动，避免过于频繁
                            QTimer.singleShot(10, lambda: self.scroll_area.verticalScrollBar().setValue(
                                self.scroll_area.verticalScrollBar().maximum()))
                    
        except Exception as e:
            print(f"Sherpa-ONNX 音频流处理错误: {e}")
            import traceback
            print(traceback.format_exc())
            self.is_running = False

    def start_file_transcription(self, file_path):
        """开始文件转录"""
        try:
            if self.is_running:
                return
            
            if not file_path or not os.path.exists(file_path):
                self.signals.new_text.emit("错误：请选择有效的音频/视频文件")
                return
            
            self.is_running = True
            
            # 检查模型是否已加载
            if self.model is None:
                self.subtitle_label.setText("错误: 模型未加载或加载失败，请选择其他模型")
                self.is_running = False
                return
            
            # 更新按钮文本
            self.start_button.setText("停止转录")
            self.start_button.setStyleSheet("background-color: rgba(150, 50, 50, 200); color: white;")
            
            # 显示进度条
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("准备中...")
            self.progress_bar.show()
            
            # 清空转录文本列表
            self.transcript_text = []
            self.transcript_text.append(f"开始转录文件: {os.path.basename(file_path)}")
            self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
            
            # 启动文件转录线程
            self.audio_thread = threading.Thread(target=self.transcribe_file, args=(file_path,))
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
        except Exception as e:
            print(f"开始文件转录错误: {e}")
            import traceback
            print(traceback.format_exc())
            self.signals.new_text.emit(f"转录错误: {e}")
            self.is_running = False

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
            
            # 更新按钮状态
            self.start_button.setText("停止转录")
            self.start_button.setStyleSheet("background-color: rgba(150, 50, 50, 200); color: white;")
            
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

    def switch_asr_model(self, model_type):
        """切换 ASR 模型 - 确保完全清理旧模型资源"""
        try:
            print(f"\n开始切换到 {model_type} 模型...")
            
            # 停止当前转录
            if self.is_running:
                print("停止当前转录...")
                self.stop_transcription()
            
            # 完全清理旧模型资源
            if self.current_asr_model in ["sherpa", "sherpa_std"]:
                # 清理 Sherpa 资源
                self.recognizer = None
                self.stream = None
            elif self.current_asr_model == "vosk":
                # 清理 VOSK 资源
                self.model = None
                self.rec = None
            
            # 设置新的模型类型
            self.current_asr_model = model_type
            
            # 加载新模型
            if model_type in ["sherpa", "sherpa_std"]:
                print(f"正在加载 Sherpa-ONNX {model_type} 模型...")
                success = self.load_sherpa_model()
                if not success:
                    print(f"Sherpa-ONNX {model_type} 模型加载失败")
                    QMessageBox.warning(self, "错误", f"Sherpa-ONNX {model_type} 模型加载失败")
                    return
                print(f"Sherpa-ONNX {model_type} 模型加载成功")
            elif model_type == "vosk":
                print("正在加载 VOSK 模型...")
                success = self.load_vosk_model()
                if not success:
                    print("VOSK 模型加载失败")
                    QMessageBox.warning(self, "错误", "VOSK 模型加载失败")
                    return
                print("VOSK 模型加载成功")
            
            # 如果之前在运行，重新启动转录
            if self.is_running:
                print("重新启动转录...")
                self.start_transcription()
            
            print(f"模型切换完成: {self.current_asr_model}")
            
        except Exception as e:
            print(f"切换模型失败: {e}")
            import traceback
            print(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"切换模型失败: {str(e)}")

    def switch_rtm_model(self, model_type):
        """切换 RTM 模型"""
        try:
            if model_type == "argos":
                self.current_rtm_model = "argos"
                self.init_argos_translator()
            elif model_type == "opus":
                if os.path.exists(self.model_paths["opus_mt"]):
                    self.current_rtm_model = "opus"
                    self.init_opus_translator()
                else:
                    QMessageBox.warning(self, "错误", "Opus-MT 模型目录不存在")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"切换翻译模型失败: {str(e)}")

    def load_vosk_model(self):
        """专门加载 VOSK 模型 - 使用预定义的路径"""
        try:
            # 使用预定义的路径，不要动态生成
            model_path = self.model_paths["vosk"]
            print(f"加载 VOSK 模型: {model_path}")
            
            # 检查路径是否存在
            if not os.path.exists(model_path):
                print(f"错误: VOSK 模型路径不存在: {model_path}")
                return False
                
            # 加载模型
            self.model = vosk.Model(model_path)
            print(f"VOSK 模型加载成功: {model_path}")
            
            return True
        except Exception as e:
            print(f"VOSK 模型加载失败: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    def load_sherpa_model(self):
        """加载 Sherpa-ONNX 模型 - 使用预定义的路径"""
        try:
            # 根据当前模型类型选择配置
            if self.current_asr_model == "sherpa":
                # 使用量化模型配置
                config = {
                    "encoder": f"{self.model_paths['sherpa']}/encoder-epoch-99-avg-1.int8.onnx",
                    "decoder": f"{self.model_paths['sherpa']}/decoder-epoch-99-avg-1.int8.onnx",
                    "joiner": f"{self.model_paths['sherpa']}/joiner-epoch-99-avg-1.int8.onnx",
                    "tokens": f"{self.model_paths['sherpa']}/tokens.txt",
                    "num_threads": 1,
                    "sample_rate": 16000,
                    "feature_dim": 80,
                    "decoding_method": "greedy_search",
                    "debug": False,
                }
                print("使用 Sherpa-ONNX 量化模型")
            else:
                # 使用标准模型配置
                config = {
                    "encoder": f"{self.model_paths['sherpa']}/encoder-epoch-99-avg-1.onnx",
                    "decoder": f"{self.model_paths['sherpa']}/decoder-epoch-99-avg-1.onnx",
                    "joiner": f"{self.model_paths['sherpa']}/joiner-epoch-99-avg-1.onnx",
                    "tokens": f"{self.model_paths['sherpa']}/tokens.txt",
                    "num_threads": 1,
                    "sample_rate": 16000,
                    "feature_dim": 80,
                    "decoding_method": "greedy_search",
                    "debug": False,
                }
                print("使用 Sherpa-ONNX 标准模型")
            
            # 检查模型文件是否存在
            for key in ["encoder", "decoder", "joiner", "tokens"]:
                if not os.path.exists(config[key]):
                    print(f"错误: Sherpa-ONNX 模型文件不存在: {config[key]}")
                    return False
            
            # 创建识别器
            self.recognizer = sherpa_onnx.OnlineRecognizer(
                **config
            )
            
            # 创建流
            self.stream = self.recognizer.create_stream()
            
            print("Sherpa-ONNX 模型加载成功")
            return True
        except Exception as e:
            print(f"Sherpa-ONNX 模型加载失败: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    def process_vosk_audio(self, audio_data):
        """处理 VOSK 音频 - 恢复原始逻辑"""
        try:
            # 确保音频数据是字节格式
            if isinstance(audio_data, np.ndarray):
                # 确保数据在 [-1, 1] 范围内
                if np.max(np.abs(audio_data)) > 1.0:
                    audio_data = audio_data / 32768.0
                
                # 转换为 16 位整数
                audio_data = (audio_data * 32767).astype(np.int16).tobytes()
            
            # 确保识别器存在
            if not self.rec:
                self.rec = self.create_recognizer()
                if not self.rec:
                    return None
            
            # 处理音频数据
            if self.rec.AcceptWaveform(audio_data):
                result = json.loads(self.rec.Result())
                text = result.get("text", "")
                
                # 将结果添加到实际转录内容列表
                if text and text.strip():
                    self.actual_transcript.append(text)
                
                return text
            else:
                partial = json.loads(self.rec.PartialResult())
                return "PARTIAL:" + partial.get("partial", "")
                
        except Exception as e:
            print(f"VOSK 音频处理错误: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def process_sherpa_audio(self, audio_data):
        """处理 Sherpa-ONNX 音频 - 确保与 VOSK 完全独立"""
        try:
            # 检查 stream 是否为 None，如果是则尝试重新初始化
            if self.stream is None or self.recognizer is None:
                print("警告: Sherpa-ONNX 流或识别器为空，尝试重新初始化...")
                success = self.load_sherpa_model()
                if not success:
                    print("重新初始化 Sherpa-ONNX 模型失败")
                    return None
                print("成功重新初始化 Sherpa-ONNX 模型")
            
            # 确保音频数据是 numpy 数组并归一化 - Sherpa 专用处理
            if isinstance(audio_data, np.ndarray):
                if len(audio_data.shape) > 1:
                    audio_data = audio_data[:, 0]  # 取第一个声道
                audio_data = audio_data.astype(np.float32) / 32768  # 归一化到 [-1, 1]
            
            # 处理音频数据
            self.stream.accept_waveform(16000, audio_data)
            
            # 检查是否准备好解码
            if self.recognizer.is_ready(self.stream):
                self.recognizer.decode_stream(self.stream)
                result = self.recognizer.get_result(self.stream)
                if result and hasattr(result, 'text') and result.text.strip():
                    # 将结果添加到实际转录内容列表 - 使用 Sherpa 专用的处理逻辑
                    if result.text not in self.actual_transcript:
                        self.actual_transcript.append(result.text)
                    return result.text
            
            return None
            
        except Exception as e:
            print(f"Sherpa-ONNX 音频处理错误: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def process_recognition_result(self, result):
        """处理识别结果"""
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            return result.get("text", "")
        return ""

    def apply_translation(self, text):
        """应用翻译 - 根据当前翻译模型选择不同的处理方法"""
        if not text or not text.strip():
            return text
            
        try:
            # 根据当前翻译模型选择不同的处理方法
            if self.current_rtm_model == "argos":
                return self.translate_with_argos(text)
            elif self.current_rtm_model == "opus":
                return self.translate_with_opus(text)
            else:
                return text
        except Exception as e:
            print(f"翻译错误: {e}")
            return text

    def validate_model_paths(self):
        """验证模型路径"""
        try:
            # 检查 VOSK 模型
            if not os.path.exists(self.model_paths["vosk"]):
                print(f"警告: VOSK 模型路径不存在: {self.model_paths['vosk']}")
                return False
                
            # 检查 Sherpa-ONNX 模型
            sherpa_path = self.model_paths["sherpa"]
            print(f"正在检查 Sherpa-ONNX 模型路径: {sherpa_path}")
            
            if not os.path.exists(sherpa_path):
                print(f"警告: Sherpa-ONNX 模型路径不存在: {sherpa_path}")
                return False
            
            # 检查所有必需的文件
            required_files = [
                # int8 量化模型文件
                "encoder-epoch-99-avg-1.int8.onnx",
                "decoder-epoch-99-avg-1.int8.onnx",
                "joiner-epoch-99-avg-1.int8.onnx",
                # 标准模型文件
                "encoder-epoch-99-avg-1.onnx",
                "decoder-epoch-99-avg-1.onnx",
                "joiner-epoch-99-avg-1.onnx",
                # 共用文件
                "tokens.txt"
            ]
            
            for file in required_files:
                full_path = os.path.join(sherpa_path, file)
                if not os.path.exists(full_path):
                    print(f"警告: Sherpa-ONNX 模型文件不存在: {full_path}")
                    # 不要立即返回 False，继续检查其他文件
                    continue
                else:
                    print(f"找到文件: {full_path}")
            
            # 检查 Opus-MT 模型
            if not os.path.exists(self.model_paths["opus_mt"]):
                print(f"警告: Opus-MT 模型路径不存在: {self.model_paths['opus_mt']}")
                return False
                
            # 检查 Argos Translate 模型
            if not os.path.exists(self.model_paths["argos"]):
                print(f"警告: Argos Translate 模型文件不存在: {self.model_paths['argos']}")
                return False
                
            # 不再单独检查标准模型路径，因为标准模型和量化模型共用同一目录
            
            print("所有模型路径验证通过")
            return True
            
        except Exception as e:
            print(f"验证模型路径时出错: {e}")
            return False

    def update_audio_devices(self):
        """更新音频设备列表"""
        try:
            # 清空现有设备菜单
            self.device_menu.clear()
            
            # 获取所有音频设备
            speakers = sc.all_speakers()
            device_group = QActionGroup(self)
            
            # 添加设备到菜单
            for speaker in speakers:
                action = QAction(speaker.name, self, checkable=True)
                action.setData(speaker)
                device_group.addAction(action)
                self.device_menu.addAction(action)
                
                # 如果没有选择当前设备，选择第一个
                if not hasattr(self, 'current_device'):
                    action.setChecked(True)
                    self.current_device = speaker
                
            # 设置互斥选择
            device_group.setExclusive(True)
            device_group.triggered.connect(self.select_audio_device)
            
        except Exception as e:
            print(f"更新音频设备列表失败: {e}")
            import traceback
            print(traceback.format_exc())
    
    def closeEvent(self, event):
        """窗口关闭时释放 COM"""
        try:
            pythoncom.CoUninitialize()  # 释放 COM
        except:
            pass
        super().closeEvent(event)

    def show_system_info(self):
        """显示系统信息"""
        try:
            # 获取系统信息
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # 格式化信息
            info = (
                f"系统信息:\n"
                f"CPU 使用率: {psutil.cpu_percent()}%\n"
                f"内存使用: {memory.percent}% (总计: {memory.total/1024/1024/1024:.2f}GB)\n"
                f"磁盘使用: {disk.percent}% (可用: {disk.free/1024/1024/1024:.2f}GB)\n"
                f"当前 ASR 模型: {self.current_asr_model}\n"
                f"当前 RTM 模型: {self.current_rtm_model}\n"
                f"说话人识别: {'启用' if self.enable_speaker_id else '禁用'}"
            )
            
            # 显示信息对话框
            QMessageBox.information(self, "系统信息", info)
            
        except Exception as e:
            print(f"获取系统信息错误: {e}")
            QMessageBox.warning(self, "错误", f"获取系统信息失败: {e}")

    def check_model_directories(self):
        """检查模型目录"""
        try:
            result = self.validate_model_paths()
            if result:
                QMessageBox.information(self, "模型检查", "所有模型路径验证通过")
            else:
                QMessageBox.warning(self, "模型检查", "部分模型路径验证失败，请查看控制台输出")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"检查模型目录失败: {e}")

    def test_audio_device(self):
        """测试音频设备"""
        try:
            if not self.current_device:
                QMessageBox.warning(self, "警告", "未选择音频设备")
                return
                
            # 创建测试录音
            with sc.get_microphone(id=str(self.current_device.id), include_loopback=True).recorder(samplerate=16000) as mic:
                data = mic.record(numframes=16000)  # 录制1秒
                
                # 检查音频电平
                level = np.max(np.abs(data))
                if level > 0.01:
                    QMessageBox.information(self, "测试结果", "音频设备工作正常，检测到音频信号")
                else:
                    QMessageBox.warning(self, "测试结果", "未检测到有效音频信号，请检查音频设置")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"测试音频设备失败: {e}")

    def search_model_docs(self):
        """搜索模型文档"""
        try:
            # 打开模型文档网页
            if self.current_asr_model == "vosk":
                url = "https://alphacephei.com/vosk/models"
            elif self.current_asr_model == "sherpa":
                url = "https://github.com/k2-fsa/sherpa-onnx"
            else:
                url = "https://github.com/alphacep/vosk-api/wiki"
                
            # 使用系统默认浏览器打开
            import webbrowser
            webbrowser.open(url)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开文档失败: {e}")

    def change_font_size(self, size):
        """更改字体大小"""
        try:
            sizes = {
                'small': 14,
                'medium': 20,
                'large': 26
            }
            
            if size in sizes:
                self.subtitle_label.setFont(QFont("Arial", sizes[size], QFont.Bold))
                self.subtitle_label.adjustSize()
                
        except Exception as e:
            print(f"更改字体大小错误: {e}")

    def toggle_speaker_recognition(self):
        """切换说话人识别功能"""
        try:
            # 获取当前复选框状态
            is_enabled = self.enable_speaker_action.isChecked()
            
            if is_enabled:
                # 尝试加载说话人识别模型
                if not hasattr(self, 'speaker_identifier') or self.speaker_identifier is None:
                    self.speaker_identifier = SpeakerIdentifier(self.spk_model_path)
                    if not self.speaker_identifier.spk_model:
                        QMessageBox.warning(self, "警告", "说话人识别模型加载失败")
                        self.enable_speaker_action.setChecked(False)
                        return
                
                # 启用说话人识别
                self.enable_speaker_id = True
                print("说话人识别功能已启用")
                
                # 重新创建识别器以应用说话人识别设置
                if hasattr(self, 'rec'):
                    self.rec = self.create_recognizer()
                
                # 更新状态提示
                self.subtitle_label.setText("说话人识别功能已启用")
            else:
                # 禁用说话人识别
                self.enable_speaker_id = False
                print("说话人识别功能已禁用")
                
                # 清除说话人相关数据
                if hasattr(self, 'speaker_identifier'):
                    self.speaker_identifier.speaker_embeddings = []
                    self.speaker_identifier.speaker_names = []
                
                # 更新状态提示
                self.subtitle_label.setText("说话人识别功能已禁用")
            
            # 如果正在转录，重新启动以应用新设置
            if self.is_running:
                self.restart_transcription()
                
        except Exception as e:
            print(f"切换说话人识别功能错误: {e}")
            import traceback
            print(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"切换说话人识别功能失败: {e}")
            # 恢复复选框状态
            self.enable_speaker_action.setChecked(not is_enabled)

    def process_audio_stream(self):
        """处理音频流"""
        try:
            with self.mic as mic:
                while self.is_running:
                    # 录制音频
                    data = mic.record(numframes=8000)  # 0.5秒的音频
                    
                    # 检查音频电平
                    audio_level = np.max(np.abs(data))
                    if audio_level < 0.0005:  # 静音检测
                        continue
                        
                    print(f"音频电平: {audio_level:.4f}")
                    
                    # 根据当前模型类型处理音频
                    if self.current_asr_model == "vosk":
                        result = self.process_vosk_audio(data)
                    elif self.current_asr_model in ["sherpa", "sherpa_std"]:
                        result = self.process_sherpa_audio(data)
                    
                    # 处理识别结果
                    if result:
                        self.signals.new_text.emit(result)
                        
        except Exception as e:
            print(f"音频流处理错误: {e}")
            import traceback
            print(traceback.format_exc())
            self.is_running = False

    def start_vosk_transcription(self):
        """专门用于 VOSK 模型的转录启动 - 与 voskIS_Normal.py 保持一致"""
        try:
            # 检查设备
            if not self.current_device:
                print("错误: 未选择音频设备，尝试使用默认设备")
                # 尝试获取默认设备
                speakers = sc.all_speakers()
                if speakers:
                    self.current_device = speakers[0]
                    print(f"已自动选择设备: {self.current_device.name}")
                else:
                    print("错误: 未找到可用的音频设备")
                    QMessageBox.warning(self, "错误", "未找到可用的音频设备")
                    return
            
            # 创建音频捕获 - 添加错误处理
            try:
                self.mic = sc.get_microphone(
                    id=str(self.current_device.id),
                    include_loopback=True
                ).recorder(samplerate=16000)
                print(f"成功创建音频捕获: {self.current_device.name}")
            except Exception as e:
                print(f"创建音频捕获失败: {e}")
                QMessageBox.warning(self, "错误", f"创建音频捕获失败: {str(e)}")
                return
            
            # 确保模型已加载
            if not self.model:
                success = self.load_vosk_model()
                if not success:
                    print("加载 VOSK 模型失败")
                    QMessageBox.warning(self, "错误", "加载 VOSK 模型失败")
                    return
            
            # 创建新的识别器 - 不重用之前的识别器
            rec = vosk.KaldiRecognizer(self.model, 16000)
            rec.SetWords(True)  # 始终启用词级别时间戳
            
            # 启动音频处理线程 - 传递新创建的识别器
            self.is_running = True
            self.audio_thread = threading.Thread(target=lambda: self.capture_vosk_audio(rec))
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            # 更新按钮状态
            self.start_button.setText("停止转录")
            self.start_button.setStyleSheet("background-color: rgba(150, 50, 50, 200); color: white;")
            
        except Exception as e:
            print(f"启动 VOSK 转录失败: {e}")
            import traceback
            print(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"启动 VOSK 转录失败: {str(e)}")
            self.is_running = False

    def capture_vosk_audio(self, rec):
        """捕获系统音频并进行转录 - 与 voskIS_Normal.py 保持一致"""
        try:
            # 确保转录文本列表已初始化
            if not hasattr(self, 'transcript_text'):
                self.transcript_text = []
            
            with self.mic as mic:
                print("开始 VOSK 音频捕获")
                while self.is_running:
                    data = mic.record(numframes=8000)
                    
                    # 检查音频电平
                    audio_level = np.max(np.abs(data))
                    if audio_level < 0.0005:
                        continue
                    
                    # 转换为字节
                    audio_data = (data * 32767).astype(np.int16).tobytes()
                    
                    # 处理音频数据
                    self.process_vosk_audio_data(rec, audio_data)
        
            print("VOSK 音频捕获结束")
        except Exception as e:
            print(f"VOSK 音频捕获错误: {e}")
            import traceback
            print(traceback.format_exc())
            self.is_running = False

    def process_vosk_audio_data(self, rec, audio_data):
        """处理 VOSK 音频数据 - 与 voskIS_Normal.py 保持一致"""
        try:
            # 处理语音识别
            if rec.AcceptWaveform(audio_data):
                result = json.loads(rec.Result())
                print(f"VOSK 识别结果: {result}")
                
                if 'text' in result and result['text'].strip():
                    text = result['text'].strip()
                    
                    # 添加标点符号和首字母大写
                    text = self.add_punctuation(text)
                    print(f"处理后的文本: {text}")
                    
                    # 添加到转录文本列表
                    self.transcript_text.append(text)
                    
                    # 更新显示 - 只显示最近的几行
                    self.subtitle_label.setText('\n'.join(self.transcript_text[-10:]))
                    
                    # 滚动到底部
                    QTimer.singleShot(10, lambda: self.scroll_area.verticalScrollBar().setValue(
                        self.scroll_area.verticalScrollBar().maximum()
                    ))
                    
                    # 强制更新UI
                    QApplication.processEvents()
        except Exception as e:
            print(f"处理 VOSK 音频数据错误: {e}")
            import traceback
            print(traceback.format_exc())

    def start_sherpa_transcription(self):
        """专门用于 Sherpa-ONNX 模型的转录启动"""
        try:
            # 检查设备
            if not self.current_device:
                print("错误: 未选择音频设备")
                return
            
            # 创建音频捕获
            self.mic = sc.get_microphone(
                id=str(self.current_device.id),
                include_loopback=True
            ).recorder(samplerate=16000)
            
            # 启动音频处理线程
            self.is_running = True
            self.audio_thread = threading.Thread(target=self.process_sherpa_stream)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            # 更新按钮状态
            self.start_button.setText("停止转录")
            self.start_button.setStyleSheet("background-color: rgba(150, 50, 50, 200); color: white;")
            
        except Exception as e:
            print(f"启动 Sherpa-ONNX 转录失败: {e}")
            self.is_running = False

    def translate_with_argos(self, text):
        """使用 Argostranslate 翻译文本"""
        try:
            if not hasattr(self, 'translator') or self.translator is None:
                print("Argostranslate 翻译器未初始化")
                return text
                
            translation = self.translator.translate(text)
            return f"{text}\n译文: {translation}"
        except Exception as e:
            print(f"Argostranslate 翻译错误: {e}")
            return text

    def translate_with_opus(self, text):
        """使用 Opus-MT-ONNX 翻译文本"""
        try:
            if not hasattr(self, 'opus_translator') or self.opus_translator is None:
                print("Opus-MT-ONNX 翻译器未初始化")
                return text
                
            translation = self.opus_translator.translate(text)
            return f"{text}\n译文: {translation}"
        except Exception as e:
            print(f"Opus-MT-ONNX 翻译错误: {e}")
            return text

    def check_vosk_model(self):
        """检查 VOSK 模型文件"""
        model_path = self.model_paths["vosk"]
        print(f"检查 VOSK 模型: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"错误: VOSK 模型路径不存在: {model_path}")
            return False
        
        # 检查模型目录中的关键文件
        required_files = ["am/final.mdl", "conf/mfcc.conf", "ivector/final.dubm"]
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if not os.path.exists(file_path):
                print(f"错误: VOSK 模型文件不存在: {file_path}")
                return False
        
        print("VOSK 模型文件检查通过")
        return True

    def test_vosk_basic(self):
        """测试 VOSK 基本功能"""
        try:
            model_path = self.model_paths["vosk"]
            print(f"测试 VOSK 基本功能，使用模型: {model_path}")
            
            # 加载模型
            model = vosk.Model(model_path)
            
            # 创建识别器
            rec = vosk.KaldiRecognizer(model, 16000)
            
            # 测试简单的音频数据
            test_audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
            test_audio = (test_audio * 32767).astype(np.int16).tobytes()
            
            rec.AcceptWaveform(test_audio)
            result = json.loads(rec.Result())
            
            print(f"VOSK 基本测试结果: {result}")
            return True
        except Exception as e:
            print(f"VOSK 基本测试失败: {e}")
            import traceback
            print(traceback.format_exc())
            return False

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
    try:
        # 禁用 COM 自动初始化
        os.environ["PYTHONCOM_INITIALIZE"] = "0"
        
        # 创建应用
        app = QApplication(sys.argv)
        app.setStyle('Fusion')  # 使用 Fusion 风格
        
        # 创建窗口
        window = SubtitleWindow()
        window.show()
        
        # 运行应用
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"程序启动错误: {e}")
        import traceback
        print(traceback.format_exc()) 