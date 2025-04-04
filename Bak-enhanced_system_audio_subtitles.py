#!/usr/bin/env python3

# 在导入任何库之前设置环境变量
import os
os.environ["PYTHONCOM_INITIALIZE"] = "0"  # 禁止 pythoncom 自动初始化
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"

import sys
import os
import json
import subprocess
import threading
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, 
                            QPushButton, QSlider, QHBoxLayout, QComboBox, QFileDialog, 
                            QTabWidget, QRadioButton, QButtonGroup, QShortcut, QLineEdit, 
                            QProgressBar, QScrollArea, QMenuBar, QMenu, QAction, QActionGroup, 
                            QSizePolicy, QGraphicsOpacityEffect, QSplitter, QInputDialog, QMessageBox,
                            QStatusBar, QLabel)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QFont, QColor, QKeySequence
import vosk
import soundcard as sc
import warnings
import ctypes
from ctypes import windll, c_int, byref
import psutil
import gc
from vosk import Model, KaldiRecognizer, SetLogLevel
from vosk import SpkModel
import datetime
import logging
import traceback  # 添加在文件开头的导入部分
from transformers import MarianTokenizer, MarianMTModel
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# 忽略警告
warnings.filterwarnings("ignore", message="data discontinuity in recording")

# Windows API 常量
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
LWA_ALPHA = 0x00000002

# 检查可选依赖
try:
    import sherpa_onnx
    import sounddevice as sd
    SHERPA_AVAILABLE = True
except ImportError:
    print("警告: Sherpa-ONNX 库未安装，将禁用 Sherpa-ONNX ASR 功能")
    SHERPA_AVAILABLE = False

try:
    import argostranslate.package
    import argostranslate.translate
    ARGOS_AVAILABLE = True
except ImportError:
    print("警告: Argos Translate 库未安装，翻译功能将不可用")
    ARGOS_AVAILABLE = False

class ArgosTranslator:
    def __init__(self):
        self.translator = None
        self.translation_cache = {}
        self.setup()
    
    def setup(self):
        """初始化 Argos Translate 翻译器"""
        if not ARGOS_AVAILABLE:
            print("Argos Translate 库未安装，无法设置翻译器")
            return False
            
        try:
            # 检查是否已安装语言包
            from argostranslate import package
            available_packages = package.get_available_packages()
            en_zh_package = None
            
            # 查找英文到中文的包
            for pkg in available_packages:
                if pkg.from_code == 'en' and pkg.to_code == 'zh':
                    en_zh_package = pkg
                    break
            
            # 如果找不到包，则下载
            if not en_zh_package:
                print("正在下载英文到中文的翻译模型...")
                package.update_package_index()
                available_packages = package.get_available_packages()
                
                # 再次查找并安装包
                for pkg in available_packages:
                    if pkg.from_code == 'en' and pkg.to_code == 'zh':
                        print(f"安装包: {pkg.name}")
                        package.install_from_path(pkg.download())
                        break
            
            # 初始化翻译器
            from argostranslate import translate
            self.translator = translate.get_translation_from_codes('en', 'zh')
            
            print("Argos Translate 翻译器初始化成功")
            return True
            
        except Exception as e:
            print(f"Argos Translate 初始化错误: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def translate(self, text):
        """翻译文本"""
        if text in self.translation_cache:
            return self.translation_cache[text]
        
        if not self.translator:
            return text + " (翻译器未初始化)"
        
        try:
            translated = self.translator.translate(text)
            self.translation_cache[text] = translated
            return translated
        except Exception as e:
            print(f"Argos Translate 翻译错误: {e}")
            return text + " (翻译失败)"
    
    def clear_cache(self):
        """清除翻译缓存"""
        self.translation_cache.clear()

class SherpaOnnxASR:
    """Sherpa-ONNX ASR 引擎实现"""
    def __init__(self, model_dir="sherpa_models"):
        self.model_dir = model_dir
        self.recognizer = None
        self.stream = None
        self.setup()
    
    def setup(self):
        """初始化 Sherpa-ONNX ASR"""
        try:
            if not SHERPA_AVAILABLE:
                print("Sherpa-ONNX 库未安装")
                return False
            
            # 检查模型文件
            model_files = {
                "encoder.onnx": "编码器模型",
                "decoder.onnx": "解码器模型",
                "joiner.onnx": "连接器模型",
                "tokens.txt": "词元文件"
            }
            
            for file, desc in model_files.items():
                path = os.path.join(self.model_dir, file)
                if not os.path.exists(path):
                    print(f"缺少{desc}: {path}")
                    return False
            
            # 配置识别器
            config = {
                "encoder": os.path.join(self.model_dir, "encoder.onnx"),
                "decoder": os.path.join(self.model_dir, "decoder.onnx"),
                "joiner": os.path.join(self.model_dir, "joiner.onnx"),
                "tokens": os.path.join(self.model_dir, "tokens.txt"),
                "sample_rate": 16000,
                "feature_config": {
                    "feature_type": "fbank",
                    "num_mel_bins": 80,
                    "frame_shift_ms": 10,
                    "frame_length_ms": 25,
                    "dither": 0.0,
                }
            }
            
            # 创建识别器和流
            self.recognizer = sherpa_onnx.OnlineRecognizer(config)
            self.stream = self.recognizer.create_stream()
            print("Sherpa-ONNX ASR 初始化成功")
            return True
            
        except Exception as e:
            print(f"Sherpa-ONNX ASR 初始化失败: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def transcribe(self, audio_data):
        """转录音频数据"""
        try:
            if not self.recognizer or not self.stream:
                return ""
            
            # 处理音频数据
            self.stream.accept_waveform(16000, audio_data)
            self.recognizer.decode_stream(self.stream)
            
            # 获取结果
            text = self.stream.get_text()
            if text:
                return text
            
            return ""
            
        except Exception as e:
            print(f"Sherpa-ONNX 转录错误: {e}")
            return ""
    
    def reset(self):
        """重置识别器状态"""
        try:
            if self.recognizer:
                self.stream = self.recognizer.create_stream()
        except Exception as e:
            print(f"重置 Sherpa-ONNX 识别器错误: {e}")
    
    def get_final_result(self):
        """获取最终结果"""
        try:
            if not self.recognizer or not self.stream:
                return ""
            
            # 强制解码剩余音频
            self.recognizer.decode_stream(self.stream)
            return self.stream.get_text()
            
        except Exception as e:
            print(f"获取 Sherpa-ONNX 最终结果错误: {e}")
            return ""
    
    def __del__(self):
        """清理资源"""
        self.recognizer = None
        self.stream = None

class VoskASR:
    """Vosk ASR 引擎"""
    def __init__(self, model_dir="model"):
        self.model_dir = model_dir
        self.model = None
        self.recognizer = None
        self.sample_rate = 16000
        self.partial_result = ""
        self.setup()
    
    def setup(self):
        """初始化 Vosk ASR 引擎"""
        try:
            # 获取模型目录的绝对路径
            self.model_dir = os.path.abspath(self.model_dir)
            print(f"Vosk 模型目录: {self.model_dir}")
            
            # 检查模型目录是否存在
            if not os.path.exists(self.model_dir):
                print(f"错误: Vosk 模型目录不存在: {self.model_dir}")
                return False
            
            # 尝试加载模型
            try:
                self.model = vosk.Model(self.model_dir)
                print("Vosk 模型加载成功")
            except Exception as e:
                print(f"Vosk 模型加载失败: {e}")
                
                # 尝试查找子目录中的模型
                found = False
                if os.path.isdir(self.model_dir):
                    for subdir in os.listdir(self.model_dir):
                        subdir_path = os.path.join(self.model_dir, subdir)
                        if os.path.isdir(subdir_path):
                            try:
                                self.model = vosk.Model(subdir_path)
                                self.model_dir = subdir_path
                                found = True
                                print(f"在子目录中找到 Vosk 模型: {subdir_path}")
                                break
                            except Exception:
                                continue
                
                # 如果仍然找不到模型，尝试使用系统音频字幕文件中的模型路径
                if not found:
                    alt_paths = [
                        "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\model\\vosk-model-small-en-us-0.15",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "vosk-model-small-en-us-0.15"),
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "vosk-model-small-cn-0.22")
                    ]
                    
                    for path in alt_paths:
                        if os.path.exists(path):
                            try:
                                self.model = vosk.Model(path)
                                self.model_dir = path
                                found = True
                                print(f"使用替代路径加载 Vosk 模型: {path}")
                                break
                            except Exception:
                                continue
                
                if not found:
                    print("无法找到可用的 Vosk 模型")
                    return False
            
            # 创建识别器
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)
            
            print("Vosk ASR 引擎初始化成功")
            return True
            
        except Exception as e:
            print(f"Vosk ASR 引擎初始化失败: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def transcribe(self, audio_data):
        """转录音频数据"""
        if not self.recognizer:
            return ""
        
        try:
            # 确保音频数据是字节类型
            if isinstance(audio_data, np.ndarray):
                audio_data = (audio_data * 32767).astype(np.int16).tobytes()
            
            # 处理音频数据
            if self.recognizer.AcceptWaveform(audio_data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "")
                self.partial_result = ""
                return text
            else:
                # 返回部分结果
                partial = json.loads(self.recognizer.PartialResult())
                partial_text = partial.get("partial", "")
                
                # 只有当部分结果发生变化时才返回
                if partial_text and partial_text != self.partial_result:
                    self.partial_result = partial_text
                    return "PARTIAL:" + partial_text
                
                return ""
                
        except Exception as e:
            print(f"Vosk 转录错误: {e}")
            return ""
    
    def reset(self):
        """重置识别器"""
        if self.model:
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)
            self.partial_result = ""
    
    def get_final_result(self):
        """获取最终结果"""
        if self.recognizer:
            result = json.loads(self.recognizer.FinalResult())
            return result.get("text", "")
        return ""

class SpeakerIdentifier:
    def __init__(self, model_dir="speaker_model"):
        self.model_dir = model_dir
        self.model = None
        self.recognizer = None
        self.speaker_embeddings = []
        self.min_frames = 50
        self.distance_threshold = 0.7
        self.max_speakers = 4
        self.setup()
    
    def setup(self):
        """初始化说话人识别模型"""
        try:
            if not os.path.exists(self.model_dir):
                print(f"错误: 说话人识别模型目录 {self.model_dir} 不存在")
                return False
            
            # 加载说话人识别模型
            self.model = SpkModel(self.model_dir)
            
            # 创建临时识别器用于特征提取
            self.recognizer = KaldiRecognizer(self.model, 16000)
            self.recognizer.SetWords(True)
            
            print("说话人识别模型初始化成功")
            return True
            
        except Exception as e:
            print(f"说话人识别模型初始化失败: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def identify_speaker(self, audio_data):
        """识别说话人"""
        try:
            if not self.model or not self.recognizer:
                return -1
            
            # 创建临时识别器
            try:
                rec = KaldiRecognizer(self.model, 16000)
                rec.SetWords(True)
            except Exception as e:
                print(f"创建临时识别器失败: {e}")
                import traceback
                print(traceback.format_exc())
                return None
            
            # 处理音频数据
            try:
                if rec.AcceptWaveform(audio_data):
                    result = json.loads(rec.Result())
                    if 'spk' in result and 'spk_frames' in result:
                        spk_data = result['spk']
                        spk_frames = result['spk_frames']
                        
                        if spk_frames < self.min_frames:
                            return None
                        
                        return self.get_speaker_id(spk_data)
                
                final_result = json.loads(rec.FinalResult())
                if 'spk' in final_result and 'spk_frames' in final_result:
                    spk_data = final_result['spk']
                    spk_frames = final_result['spk_frames']
                    
                    if spk_frames < self.min_frames:
                        return None
                    
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
            # 首个说话人直接添加
            if not self.speaker_embeddings:
                self.speaker_embeddings.append(spk_data)
                return 0
            
            # 计算与已知说话人的距离
            distances = []
            for i, embedding in enumerate(self.speaker_embeddings):
                distance = self.cosine_distance(spk_data, embedding)
                distances.append((i, distance))
            
            # 按距离排序
            distances.sort(key=lambda x: x[1])
            best_match_id, min_distance = distances[0]
            
            # 如果距离大于阈值且说话人数量未达到上限，添加新说话人
            if min_distance > self.distance_threshold and len(self.speaker_embeddings) < self.max_speakers:
                self.speaker_embeddings.append(spk_data)
                return len(self.speaker_embeddings) - 1
            
            return best_match_id
            
        except Exception as e:
            print(f"获取说话人ID错误: {e}")
            import traceback
            print(traceback.format_exc())
            return 0
    
    def cosine_distance(self, embedding1, embedding2):
        """计算两个说话人嵌入向量的余弦距离"""
        try:
            # 转换为numpy数组
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # 计算余弦距离
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 1.0
                
            similarity = dot_product / (norm1 * norm2)
            return 1.0 - similarity
            
        except Exception as e:
            print(f"计算距离错误: {e}")
            return 1.0
    
    def reset(self):
        """重置说话人识别状态"""
        self.speaker_embeddings = []
        if self.model:
            self.recognizer = KaldiRecognizer(self.model, 16000)
            self.recognizer.SetWords(True)
    
    def __del__(self):
        """清理资源"""
        self.model = None
        self.recognizer = None
        self.speaker_embeddings = []

class TranscriptionSignals(QObject):
    new_text = pyqtSignal(str)
    progress_updated = pyqtSignal(int, str)  # 进度信号
    transcription_finished = pyqtSignal()    # 完成信号
    error_occurred = pyqtSignal(str, str)    # 错误信号

    def __init__(self):
        super().__init__()
        self.last_update = time.time()
        self.update_interval = 0.1  # 100ms

    def emit_new_text(self, text):
        """发送新文本信号，带节流"""
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.new_text.emit(text)
            self.last_update = current_time

    def emit_progress(self, value, text):
        """发送进度信号"""
        self.progress_updated.emit(value, text)

    def emit_finished(self):
        """发送完成信号"""
        self.transcription_finished.emit()

    def emit_error(self, context, message):
        """发送错误信号"""
        self.error_occurred.emit(context, message)

class SubtitleWindow(QMainWindow):
    """主窗口类"""
    def __init__(self):
        super().__init__()
        
        # 设置应用程序基本信息
        self.setWindowTitle("实时字幕")
        self.resize(800, 400)
        
        # 初始化变量
        self.is_running = False
        self.transcript_text = []
        self.translation_text = []
        self.file_path = None
        self.transcription_thread = None
        self.audio_thread = None
        self.translation_thread = None
        
        # 创建信号对象
        self.signals = TranscriptionSignals()
        
        # 初始化配置管理器
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
        os.makedirs(config_dir, exist_ok=True)
        config_file = os.path.join(config_dir, "config.json")
        default_config_file = os.path.join(config_dir, "default_config.json")
        self.config_manager = ConfigManager(config_file, default_config_file)
        
        # 检查模型目录
        self.check_model_directories()
        
        # 更新模型路径
        self.update_model_paths()
        
        # 初始化其他管理器
        self.init_managers()
        
        # 创建用户界面
        self.create_ui()
        
        # 设置信号连接
        self.setup_signals()
        
        # 设置窗口样式
        self.apply_theme()
        
        # 设置窗口透明度
        self.set_opacity(self.config_manager.get("window.opacity", 0.8))
        
        # 如果配置了自动启动，则启动转录
        if self.config_manager.get("startup.auto_start", False):
            QTimer.singleShot(500, self.start_transcription)
    
    def init_managers(self):
        # 初始化模型管理器
        self.model_manager = ModelManager(os.path.dirname(os.path.abspath(__file__)))
        
        # 初始化音频设备管理器
        self.audio_manager = AudioDeviceManager()
        
        # 初始化说话人识别管理器
        self.speaker_identifier = SpeakerIdentifier()
        
        # 初始化翻译管理器
        self.translation_manager = TranslationManager()
        
        # 初始化 ASR 管理器
        self.asr_manager = ASRManager()
    
    def setup_signals(self):
        """设置信号连接"""
        # ASR 引擎切换信号
        if hasattr(self, 'vosk_action'):
            self.vosk_action.triggered.connect(lambda: self.switch_asr_engine("vosk"))
        if hasattr(self, 'sherpa_action') and self.sherpa_action.isEnabled():
            self.sherpa_action.triggered.connect(lambda: self.switch_asr_engine("sherpa"))
        
        # 翻译引擎切换信号
        if hasattr(self, 'opusmt_action'):
            self.opusmt_action.triggered.connect(lambda: self.switch_translation_engine("opusmt"))
        if hasattr(self, 'argos_action') and self.argos_action.isEnabled():
            self.argos_action.triggered.connect(lambda: self.switch_translation_engine("argos"))
        
        # 性能监控信号
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self.update_performance_metrics)
        self.metrics_timer.start(1000)  # 每秒更新一次
    
    def switch_asr_engine(self, engine_type):
        """切换 ASR 引擎"""
        if engine_type == self.config_manager.get("asr.engine", "vosk"):
            return  # 已经是当前引擎
        
        # 显示加载状态
        self.status_label.setText(f"正在切换到 {engine_type.upper()} 引擎...")
        QApplication.processEvents()  # 确保 UI 更新
        
        # 停止当前转录
        was_running = self.is_running
        if was_running:
            self.stop_transcription()
        
        try:
            # 更新配置
            self.config_manager.set("asr.engine", engine_type)
            
            # 重新初始化 ASR 引擎
            success = self.asr_manager.initialize_engine(engine_type)
            
            if not success:
                raise Exception(f"初始化 {engine_type} 引擎失败")
            
            # 更新菜单状态
            if hasattr(self, 'vosk_action'):
                self.vosk_action.setChecked(engine_type == "vosk")
            if hasattr(self, 'sherpa_action'):
                self.sherpa_action.setChecked(engine_type == "sherpa")
            
            # 显示成功消息
            self.status_label.setText(f"已切换到 {engine_type.upper()} 引擎")
            
            # 如果之前在运行，则重新启动
            if was_running:
                self.start_transcription()
                
            return True
            
        except Exception as e:
            # 处理错误
            error_msg = f"切换 ASR 引擎失败: {str(e)}"
            self.status_label.setText(error_msg)
            self.error_handler.handle_error("ASR引擎切换", error_msg)
            
            # 尝试回退到之前的引擎
            prev_engine = self.config_manager.get("asr.engine", "vosk")
            self.config_manager.set("asr.engine", prev_engine)
            
            # 更新菜单状态
            if hasattr(self, 'vosk_action'):
                self.vosk_action.setChecked(prev_engine == "vosk")
            if hasattr(self, 'sherpa_action'):
                self.sherpa_action.setChecked(prev_engine == "sherpa")
            
            return False
    
    def switch_translator(self, translator_type):
        """切换翻译器"""
        try:
            success = self.translation_manager.switch_translator(translator_type)
            if not success:
                raise Exception(f"切换到 {translator_type} 翻译器失败")
            
            # 更新菜单状态
            self.argos_action.setChecked(translator_type == "argos")
            self.bergamot_action.setChecked(translator_type == "bergamot")
            
            # 保存配置
            self.config_manager.set("translator_type", translator_type)
            
            # 更新状态栏
            self.statusBar().showMessage(f"已切换到 {translator_type} 翻译器")
            
        except Exception as e:
            self.error_handler.handle_error("翻译器切换", str(e))
    
    def update_performance_metrics(self):
        """更新性能指标显示"""
        if self.is_running:
            try:
                metrics = self.performance_monitor.get_average_metrics()
                self.cpu_label.setText(f"CPU: {metrics['avg_cpu']:.1f}%")
                self.memory_label.setText(f"内存: {metrics['avg_memory']:.1f}MB")
                self.latency_label.setText(f"延迟: {metrics['avg_latency']:.2f}s")
            except Exception as e:
                self.error_handler.handle_error("性能监控", str(e))
    
    def setup_window(self):
        """设置窗口属性"""
        self.setWindowTitle("增强版实时字幕")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Window)
        
        # 加载主题
        theme = self.config_manager.get("theme", "dark")
        self.setStyleSheet(self.theme_manager.get_stylesheet(theme))
        
        # 设置窗口大小和位置
        geometry = self.config_manager.get("window_geometry")
        if geometry:
            self.setGeometry(*geometry)
        else:
            self.resize(800, 400)
            self.move(100, 300)
    
    def create_ui(self):
        """创建用户界面"""
        # 创建中心部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 创建主布局
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 创建分割器
        self.splitter = QSplitter(Qt.Vertical)
        
        # 创建转录区域
        self.create_transcript_area()
        
        # 创建翻译区域
        self.create_translation_area()
        
        # 创建控制面板
        self.create_control_panel()
        
        # 创建菜单栏 - 确保使用新方法
        self.create_menu()
        
        # 创建状态栏
        self.create_status_bar()
        
        # 添加到主布局
        self.main_layout.addWidget(self.splitter)
        self.main_layout.addWidget(self.control_panel)
    
    def create_transcript_area(self):
        """创建转录区域"""
        # 创建转录容器
        self.transcript_container = QWidget()
        self.transcript_layout = QVBoxLayout(self.transcript_container)
        
        # 创建转录标签
        self.transcript_label = QLabel("等待转录...")
        self.transcript_label.setWordWrap(True)
        self.transcript_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        # 创建滚动区域
        self.transcript_scroll = QScrollArea()
        self.transcript_scroll.setWidget(self.transcript_label)
        self.transcript_scroll.setWidgetResizable(True)
        
        # 添加到布局
        self.transcript_layout.addWidget(self.transcript_scroll)
        self.splitter.addWidget(self.transcript_container)
    
    def create_translation_area(self):
        """创建翻译区域"""
        # 创建翻译容器
        self.translation_container = QWidget()
        self.translation_layout = QVBoxLayout(self.translation_container)
        
        # 创建翻译标签
        self.translation_label = QLabel("等待翻译...")
        self.translation_label.setWordWrap(True)
        self.translation_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        # 创建滚动区域
        self.translation_scroll = QScrollArea()
        self.translation_scroll.setWidget(self.translation_label)
        self.translation_scroll.setWidgetResizable(True)
        
        # 添加到布局
        self.translation_layout.addWidget(self.translation_scroll)
        self.splitter.addWidget(self.translation_container)
        
        # 默认隐藏翻译区域
        self.translation_container.setVisible(False)

    def create_control_panel(self):
        """创建控制面板"""
        try:
            control_layout = QHBoxLayout()
            
            # 开始/停止按钮
            self.start_button = QPushButton("开始转录")
            self.start_button.setStyleSheet("""
                QPushButton {
                    background-color: rgba(50, 150, 50, 200);
                    color: white;
                    border-radius: 3px;
                    padding: 5px 15px;
                }
                QPushButton:hover {
                    background-color: rgba(60, 170, 60, 200);
                }
            """)
            self.start_button.clicked.connect(self.toggle_transcription)
            control_layout.addWidget(self.start_button)
            
            # 文件选择按钮
            self.file_button = QPushButton("选择文件")
            self.file_button.setStyleSheet("""
                QPushButton {
                    background-color: rgba(100, 100, 100, 200);
                    color: white;
                    border-radius: 3px;
                    padding: 5px 15px;
                }
                QPushButton:hover {
                    background-color: rgba(120, 120, 120, 200);
                }
            """)
            self.file_button.clicked.connect(self.select_file)
            self.file_button.hide()  # 默认隐藏
            control_layout.addWidget(self.file_button)
            
            # 透明度滑块
            opacity_label = QLabel("透明度:")
            control_layout.addWidget(opacity_label)
            
            self.opacity_slider = QSlider(Qt.Horizontal)
            self.opacity_slider.setMinimum(20)
            self.opacity_slider.setMaximum(100)
            self.opacity_slider.setValue(80)
            self.opacity_slider.valueChanged.connect(lambda v: self.set_opacity(v/100))
            control_layout.addWidget(self.opacity_slider)
            
            control_layout.addStretch()
            return control_layout
            
        except Exception as e:
            self.handle_error("控制面板", f"创建控制面板失败: {e}")
            return QHBoxLayout()

    def create_text_display(self):
        """创建文本显示区域"""
        text_layout = QHBoxLayout()
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        
        # 创建原文显示区域
        original_widget = QWidget()
        original_layout = QVBoxLayout(original_widget)
        
        original_label = QLabel("原文:")
        original_label.setStyleSheet("color: rgba(0, 0, 0, 200);")
        original_layout.addWidget(original_label)
        
        self.original_scroll_area = QScrollArea()
        self.original_scroll_area.setWidgetResizable(True)
        self.original_scroll_area.setMinimumHeight(200)
        self.original_scroll_area.setStyleSheet("""
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
        
        original_container = QWidget()
        self.original_label = QLabel()
        self.original_label.setWordWrap(True)
        self.original_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.original_label.setStyleSheet("""
            background-color: rgba(255, 255, 255, 180);
            padding: 10px;
            border-radius: 5px;
        """)
        
        container_layout = QVBoxLayout(original_container)
        container_layout.addWidget(self.original_label)
        self.original_scroll_area.setWidget(original_container)
        original_layout.addWidget(self.original_scroll_area)
        
        splitter.addWidget(original_widget)
        
        # 创建译文显示区域
        translation_widget = QWidget()
        translation_layout = QVBoxLayout(translation_widget)
        
        translation_label = QLabel("译文:")
        translation_label.setStyleSheet("color: rgba(0, 0, 0, 200);")
        translation_layout.addWidget(translation_label)
        
        self.translation_scroll_area = QScrollArea()
        self.translation_scroll_area.setWidgetResizable(True)
        self.translation_scroll_area.setMinimumHeight(200)
        self.translation_scroll_area.setStyleSheet("""
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
        
        translation_container = QWidget()
        self.translation_label = QLabel()
        self.translation_label.setWordWrap(True)
        self.translation_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.translation_label.setStyleSheet("""
            background-color: rgba(255, 255, 255, 180);
            padding: 10px;
            border-radius: 5px;
        """)
        
        translation_container_layout = QVBoxLayout(translation_container)
        translation_container_layout.addWidget(self.translation_label)
        self.translation_scroll_area.setWidget(translation_container)
        translation_layout.addWidget(self.translation_scroll_area)
        
        splitter.addWidget(translation_widget)
        
        # 设置分割器比例
        splitter.setSizes([int(splitter.width() * 0.5)] * 2)
        
        text_layout.addWidget(splitter)
        return text_layout

    def create_shortcuts(self):
        """创建快捷键"""
        # 开始/停止转录 - Space
        self.shortcut_toggle = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.shortcut_toggle.activated.connect(self.toggle_transcription)
        
        # 切换窗口置顶 - Ctrl+T
        self.shortcut_top = QShortcut(QKeySequence("Ctrl+T"), self)
        self.shortcut_top.activated.connect(self.toggle_always_on_top)
        
        # 切换窗口模式 - Ctrl+M
        self.shortcut_mode = QShortcut(QKeySequence("Ctrl+M"), self)
        self.shortcut_mode.activated.connect(self.toggle_window_mode)
        
        # 退出程序 - Ctrl+Q
        self.shortcut_quit = QShortcut(QKeySequence("Ctrl+Q"), self)
        self.shortcut_quit.activated.connect(self.close)

    def load_config(self):
        """加载配置"""
        try:
            config_path = os.path.join(self.base_dir, "config.json")
            if not os.path.exists(config_path):
                return
            
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            # 应用配置
            self.auto_start = config.get("auto_start", False)
            self.enable_translation = config.get("enable_translation", False)
            self.sync_scroll = config.get("sync_scroll", True)
            self.enable_speaker_id = config.get("enable_speaker_id", False)
            self.is_system_audio = config.get("is_system_audio", True)
            
            # 恢复窗口位置和大小
            geometry = config.get("window_geometry")
            if geometry:
                self.setGeometry(*geometry)
            
            # 设置透明度
            opacity = config.get("opacity", 0.8)
            self.set_opacity(opacity)
            
            # 更新说话人名称
            speaker_names = config.get("speaker_names")
            if speaker_names:
                self.speaker_names = speaker_names
            
            # 更新菜单状态
            self.update_menu_state()
            
            print("配置加载成功")
            
        except Exception as e:
            self.handle_error("配置", f"加载配置失败: {e}")

    def save_config(self):
        """保存配置"""
        try:
            config = {
                "auto_start": self.auto_start,
                "enable_translation": self.enable_translation,
                "sync_scroll": self.sync_scroll,
                "enable_speaker_id": self.enable_speaker_id,
                "is_system_audio": self.is_system_audio,
                "window_geometry": [self.x(), self.y(), self.width(), self.height()],
                "opacity": self.windowOpacity(),
                "speaker_names": self.speaker_names
            }
            
            config_path = os.path.join(self.base_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
                
            print("配置已保存")
            
        except Exception as e:
            self.handle_error("配置", f"保存配置失败: {e}")

    def start_system_audio_transcription(self):
        """开始系统音频转录"""
        try:
            # 获取默认扬声器
            default_speaker = sc.default_speaker()
            print(f"使用默认扬声器: {default_speaker.name}")
            
            # 创建录音器
            mic = sc.get_microphone(id=str(default_speaker.id), include_loopback=True)
            
            # 重置 ASR 引擎
            if isinstance(self.asr_engine, VoskASR):
                self.asr_engine.reset()
            
            # 开始录音
            with mic.recorder(samplerate=16000) as recorder:
                while self.is_running:
                    # 读取音频数据
                    data = recorder.record(numframes=4000)
                    
                    # 确保数据是单声道
                    if len(data.shape) > 1:
                        data = data[:, 0]
                    
                    # 更新说话人识别
                    if self.enable_speaker_id:
                        self.update_speaker_buffer(data)
                    
                    # 转录音频
                    result = self.asr_engine.transcribe(data)
                    if result:
                        if result.startswith("PARTIAL:"):
                            self.signals.new_text.emit(result)
                        else:
                            self.process_transcription_result(result)
                    
        except Exception as e:
            self.handle_error("系统音频", f"系统音频转录错误: {e}")
            import traceback
            print(traceback.format_exc())
            
        finally:
            self.is_running = False
            self.signals.transcription_finished.emit()

    def start_microphone_transcription(self):
        """开始麦克风转录"""
        try:
            # 获取默认麦克风
            mic = sc.default_microphone()
            print(f"使用默认麦克风: {mic.name}")
            
            # 重置 ASR 引擎
            if isinstance(self.asr_engine, VoskASR):
                self.asr_engine.reset()
            
            # 开始录音
            with mic.recorder(samplerate=16000) as recorder:
                while self.is_running:
                    # 读取音频数据
                    data = recorder.record(numframes=4000)
                    
                    # 确保数据是单声道
                    if len(data.shape) > 1:
                        data = data[:, 0]
                    
                    # 更新说话人识别
                    if self.enable_speaker_id:
                        self.update_speaker_buffer(data)
                    
                    # 转录音频
                    result = self.asr_engine.transcribe(data)
                    if result:
                        if result.startswith("PARTIAL:"):
                            self.signals.new_text.emit(result)
                        else:
                            self.process_transcription_result(result)
                    
        except Exception as e:
            self.handle_error("麦克风", f"麦克风转录错误: {e}")
            import traceback
            print(traceback.format_exc())
            
        finally:
            self.is_running = False
            self.signals.transcription_finished.emit()

    def toggle_transcription(self):
        """切换转录状态"""
        if self.is_running:
            self.stop_transcription()
        else:
            self.start_transcription()

    def start_transcription(self):
        """开始转录"""
        if self.is_running:
            return
        
        try:
            self.is_running = True
            
            # 重置状态
            if self.enable_speaker_id and self.speaker_identifier:
                self.speaker_identifier.reset()
            
            # 根据音频源选择转录方式
            if self.file_path:
                self.audio_thread = threading.Thread(target=self.transcribe_file, args=(self.file_path,))
            elif self.is_system_audio:
                self.audio_thread = threading.Thread(target=self.start_system_audio_transcription)
            else:
                self.audio_thread = threading.Thread(target=self.start_microphone_transcription)
            
            # 启动线程
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            # 更新按钮状态
            self.start_button.setText("停止转录")
            self.start_button.setStyleSheet("""
                QPushButton {
                    background-color: rgba(150, 50, 50, 200);
                    color: white;
                    border-radius: 3px;
                    padding: 5px 15px;
                }
                QPushButton:hover {
                    background-color: rgba(170, 60, 60, 200);
                }
            """)
            
            # 显示进度条
            self.progress_bar.setValue(0)
            self.progress_bar.show()
            
        except Exception as e:
            self.handle_error("启动转录", f"启动转录失败: {e}")
            self.is_running = False

    def process_audio_data(self, data):
        """处理音频数据"""
        try:
            # 确保数据是单声道
            if len(data.shape) > 1:
                data = data[:, 0]
            
            # 更新说话人识别
            if self.enable_speaker_id:
                self.update_speaker_buffer(data)
            
            # 转录音频
            result = self.asr_engine.transcribe(data)
            if result:
                if result.startswith("PARTIAL:"):
                    self.signals.new_text.emit(result)
                else:
                    self.process_transcription_result(result)
                    
        except Exception as e:
            self.handle_error("音频处理", f"处理音频数据失败: {e}")

    def process_file_audio(self, file_path):
        """处理音频文件"""
        try:
            # 获取文件时长
            total_duration = self.get_file_duration(file_path)
            if total_duration <= 0:
                self.handle_error("文件错误", "无法获取文件时长")
                return
            
            # 设置采样率
            sample_rate = 16000
            
            # 启动 FFmpeg 进程
            self.ffmpeg_process = subprocess.Popen([
                'ffmpeg', '-loglevel', 'quiet',
                '-i', file_path,
                '-ar', str(sample_rate),
                '-ac', '1',
                '-f', 's16le',
                '-'
            ], stdout=subprocess.PIPE)
            
            # 重置 ASR 引擎
            if isinstance(self.asr_engine, VoskASR):
                self.asr_engine.reset()
            
            # 读取和处理音频数据
            chunk_size = 4000
            total_bytes = 0
            last_progress_time = time.time()
            
            while self.is_running:
                chunk = self.ffmpeg_process.stdout.read(chunk_size)
                if not chunk:
                    break
                
                total_bytes += len(chunk)
                
                # 更新进度
                current_time = time.time()
                if current_time - last_progress_time >= 0.2:
                    current_position = (total_bytes / (sample_rate * 2))
                    progress = min(100, int((current_position / total_duration) * 100))
                    
                    time_str = f"{int(current_position//60):02d}:{int(current_position%60):02d}"
                    total_str = f"{int(total_duration//60):02d}:{int(total_duration%60):02d}"
                    format_text = f"转录中: {time_str} / {total_str}"
                    
                    self.signals.progress_updated.emit(progress, format_text)
                    last_progress_time = current_time
                
                # 处理音频数据
                self.process_audio_data(np.frombuffer(chunk, dtype=np.int16))
            
            # 处理最终结果
            final_result = self.asr_engine.get_final_result()
            if final_result:
                self.process_transcription_result(final_result)
            
        except Exception as e:
            self.handle_error("文件处理", f"处理音频文件失败: {e}")
            import traceback
            print(traceback.format_exc())
            
        finally:
            # 清理资源
            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
                try:
                    self.ffmpeg_process.wait(timeout=1)
                except:
                    self.ffmpeg_process.kill()
                self.ffmpeg_process = None

    def process_transcription_result(self, text):
        """处理转录结果"""
        try:
            # 如果启用了说话人识别，添加说话人标识
            if self.enable_speaker_id and self.speaker_identifier:
                speaker_id = self.speaker_identifier.identify_speaker(self.current_audio_data)
                if speaker_id is not None and 0 <= speaker_id < len(self.speaker_names):
                    text = f"{self.speaker_names[speaker_id]}: {text}"
            
            # 发送文本更新信号
            self.signals.new_text.emit(text)
            
        except Exception as e:
            self.handle_error("处理结果", f"处理转录结果错误: {e}")

    def update_speaker_buffer(self, audio_data):
        """更新说话人音频缓冲区"""
        try:
            if not hasattr(self, 'current_audio_data'):
                self.current_audio_data = audio_data
            else:
                # 将新数据添加到缓冲区
                self.current_audio_data = np.concatenate([self.current_audio_data, audio_data])
                
                # 如果缓冲区太大，保留最后的部分
                max_samples = 16000 * 2  # 保留2秒的数据
                if len(self.current_audio_data) > max_samples:
                    self.current_audio_data = self.current_audio_data[-max_samples:]
                    
        except Exception as e:
            self.handle_error("说话人缓冲", f"更新说话人缓冲区错误: {e}")

    def get_file_duration(self, file_path):
        """获取音频文件时长"""
        try:
            # 使用 FFmpeg 获取文件信息
            result = subprocess.run([
                'ffprobe', 
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                file_path
            ], capture_output=True, text=True)
            
            duration = float(result.stdout.strip())
            return duration
            
        except Exception as e:
            self.handle_error("文件时长", f"获取文件时长失败: {e}")
            return 0

    def update_subtitle(self, text):
        """更新字幕显示"""
        try:
            if text.startswith("PARTIAL:"):
                # 更新临时字幕
                partial_text = text[8:]
                self.original_label.setText(partial_text)
                
                # 更新译文
                if self.enable_translation:
                    self.update_translation(partial_text)
            else:
                # 添加到转录文本
                self.transcript_text.append(text)
                
                # 更新显示
                display_text = '\n'.join(self.transcript_text[-10:])
                self.original_label.setText(display_text)
                
                # 同步滚动
                if self.sync_scroll:
                    QTimer.singleShot(10, lambda: self.original_scroll_area.verticalScrollBar().setValue(
                        self.original_scroll_area.verticalScrollBar().maximum()
                    ))
                
                # 更新译文
                if self.enable_translation:
                    self.update_translation(text)
                    
        except Exception as e:
            self.handle_error("字幕", f"更新字幕显示失败: {e}")

    def update_progress(self, value, text):
        """更新进度条"""
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(text)

    def on_transcription_finished(self):
        """转录完成处理"""
        self.is_running = False
        self.start_button.setText("开始转录")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(50, 150, 50, 200);
                color: white;
                border-radius: 3px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: rgba(60, 170, 60, 200);
            }
        """)
        self.progress_bar.hide()

    def handle_error(self, context, message):
        """错误处理"""
        print(f"错误 [{context}]: {message}")
        self.signals.error_occurred.emit(context, message)
        
        # 显示错误消息
        QMessageBox.warning(self, f"错误 - {context}", message)

    def check_system_resources(self):
        """检查系统资源"""
        try:
            # 检查内存使用
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                self.handle_error("系统资源", "内存使用率过高，可能影响转录性能")
            
            # 检查 CPU 使用
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                self.handle_error("系统资源", "CPU 使用率过高，可能影响转录性能")
            
            # 检查磁盘空间
            disk = psutil.disk_usage('/')
            if disk.percent > 95:
                self.handle_error("系统资源", "磁盘空间不足，可能无法保存转录结果")
            
            print("系统资源检查完成")
            
        except Exception as e:
            self.handle_error("系统检查", f"检查系统资源失败: {e}")

    def center_window(self):
        """将窗口居中显示"""
        try:
            # 获取屏幕几何信息
            screen = QApplication.primaryScreen().geometry()
            
            # 计算窗口位置
            x = (screen.width() - self.width()) // 2
            y = (screen.height() - self.height()) // 2
            
            # 移动窗口
            self.move(x, y)
            
        except Exception as e:
            self.handle_error("窗口位置", f"居中窗口失败: {e}")

    def force_quit(self):
        """强制退出程序"""
        try:
            # 停止转录
            if self.is_running:
                self.stop_transcription()
            
            # 保存配置
            self.save_config()
            
            # 清理资源
            if hasattr(self, 'asr_engine'):
                del self.asr_engine
            if hasattr(self, 'translator'):
                del self.translator
            if hasattr(self, 'speaker_identifier'):
                del self.speaker_identifier
            
            # 强制垃圾回收
            gc.collect()
            
            # 退出程序
            QApplication.quit()
            
        except Exception as e:
            print(f"强制退出错误: {e}")
            sys.exit(1)

    def cleanup_resources(self):
        """清理资源"""
        try:
            # 清理 ASR 引擎
            if hasattr(self, 'asr_engine'):
                self.asr_engine = None
            
            # 清理说话人识别
            if hasattr(self, 'speaker_identifier'):
                self.speaker_identifier = None
            
            # 清理翻译器
            if hasattr(self, 'translator'):
                self.translator = None
            
            # 强制垃圾回收
            gc.collect()
            
        except Exception as e:
            print(f"清理资源错误: {e}")

    def connect_menu_signals(self):
        """连接菜单信号"""
        # 音频源菜单
        self.system_audio_action.triggered.connect(lambda: self.switch_audio_source(True))
        self.microphone_action.triggered.connect(lambda: self.switch_audio_source(False))
        
        # 功能开关菜单
        self.translation_enable_action.triggered.connect(self.toggle_translation)
        self.sync_scroll_action.triggered.connect(self.toggle_sync_scroll)
        self.speaker_enable_action.triggered.connect(self.toggle_speaker_id)
        self.auto_start_action.triggered.connect(self.toggle_auto_start)
        
        # 语言菜单
        self.sys_en_action.triggered.connect(lambda: self.on_language_action_triggered('en'))
        self.sys_cn_action.triggered.connect(lambda: self.on_language_action_triggered('zh'))
        self.sys_auto_action.triggered.connect(lambda: self.on_language_action_triggered('auto'))

    def switch_audio_source(self, is_system):
        """切换音频源"""
        if self.is_running:
            self.stop_transcription()
        
        self.is_system_audio = is_system
        self.file_path = None
        
        # 更新菜单状态
        self.system_audio_action.setChecked(is_system)
        self.microphone_action.setChecked(not is_system)
        
        # 如果设置了自动启动，则重新开始转录
        if self.auto_start:
            self.start_transcription()

    def toggle_translation(self, enable):
        """切换翻译功能"""
        try:
            self.enable_translation = enable
            
            # 更新菜单状态
            if hasattr(self, 'translation_enable_action'):
                self.translation_enable_action.setChecked(enable)
            
            # 更新显示
            self.translation_scroll_area.setVisible(enable)
            
            print(f"翻译功能已{'启用' if enable else '禁用'}")
            
        except Exception as e:
            self.handle_error("翻译", f"切换翻译功能失败: {e}")

    def toggle_sync_scroll(self, enabled):
        """切换同步滚动"""
        self.sync_scroll = enabled

    def toggle_speaker_id(self, enable):
        """切换说话人识别功能"""
        try:
            self.enable_speaker_id = enable
            
            # 更新菜单状态
            if hasattr(self, 'speaker_enable_action'):
                self.speaker_enable_action.setChecked(enable)
            
            # 重置说话人识别
            if enable and self.speaker_identifier:
                self.speaker_identifier.reset()
            
            print(f"说话人识别功能已{'启用' if enable else '禁用'}")
            
        except Exception as e:
            self.handle_error("说话人", f"切换说话人识别功能失败: {e}")

    def toggle_auto_start(self, enabled):
        """切换自动启动"""
        self.auto_start = enabled
        if enabled and not self.is_running:
            self.start_transcription()

    def select_file(self):
        """选择音频文件"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择音频文件",
                "",
                "音频文件 (*.wav *.mp3 *.m4a *.aac *.ogg *.flac);;所有文件 (*)"
            )
            
            if file_path:
                self.file_path = file_path
                if self.is_running:
                    self.stop_transcription()
                self.start_transcription()
                
        except Exception as e:
            self.handle_error("文件选择", f"选择文件失败: {e}")

    def save_transcript(self):
        """保存转录文本"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存转录文本",
                f"transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "文本文件 (*.txt);;所有文件 (*)"
            )
            
            if file_path:
                with open(file_path, "w", encoding="utf-8") as f:
                    # 写入原文
                    f.write("=== 原文 ===\n\n")
                    f.write("\n".join(self.transcript_text))
                    
                    # 写入译文
                    if self.enable_translation:
                        f.write("\n\n=== 译文 ===\n\n")
                        f.write("\n".join(self.translation_text))
                
                print(f"转录已保存到: {file_path}")
                self.signals.new_text.emit(f"PARTIAL:转录已保存到: {file_path}")
                
        except Exception as e:
            self.handle_error("保存转录", f"保存转录文本失败: {e}")

    def stop_transcription(self):
        """停止转录"""
        self.is_running = False
        
        # 等待音频线程结束
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1)
        
        # 终止 FFmpeg 进程
        if hasattr(self, 'ffmpeg_process') and self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            try:
                self.ffmpeg_process.wait(timeout=1)
            except:
                self.ffmpeg_process.kill()
            self.ffmpeg_process = None
        
        # 重置状态
        self.start_button.setText("开始转录")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(50, 150, 50, 200);
                color: white;
                border-radius: 3px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: rgba(60, 170, 60, 200);
            }
        """)
        
        # 隐藏进度条
        self.progress_bar.hide()

    def keyPressEvent(self, event):
        """键盘事件处理"""
        # Ctrl+Q: 退出
        if event.key() == Qt.Key_Q and event.modifiers() == Qt.ControlModifier:
            self.close()
        
        # Ctrl+S: 保存
        elif event.key() == Qt.Key_S and event.modifiers() == Qt.ControlModifier:
            self.save_transcript()
        
        # Ctrl+O: 打开文件
        elif event.key() == Qt.Key_O and event.modifiers() == Qt.ControlModifier:
            self.select_file()
        
        # Space: 开始/停止转录
        elif event.key() == Qt.Key_Space:
            self.toggle_transcription()
        
        # +/-: 调整字体大小
        elif event.key() in [Qt.Key_Plus, Qt.Key_Equal] and event.modifiers() == Qt.ControlModifier:
            self.change_font_size(1)
        elif event.key() == Qt.Key_Minus and event.modifiers() == Qt.ControlModifier:
            self.change_font_size(-1)
        
        event.accept()

    def mousePressEvent(self, event):
        """鼠标按下事件处理"""
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        """鼠标移动事件处理"""
        if event.buttons() == Qt.LeftButton and self.drag_position:
            self.move(event.globalPos() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        """鼠标释放事件处理"""
        if event.button() == Qt.LeftButton:
            self.drag_position = None
            event.accept()

    def closeEvent(self, event):
        """关闭窗口事件"""
        try:
            # 停止转录
            if self.is_running:
                self.stop_transcription()
            
            # 保存配置
            self.save_config()
            
            # 接受关闭事件
            event.accept()
            
        except Exception as e:
            print(f"关闭窗口错误: {e}")
            event.accept()

    def update_window_style(self):
        """更新窗口样式"""
        try:
            # 设置窗口样式
            self.setStyleSheet("""
                QMainWindow {
                    background-color: rgba(255, 255, 255, 200);
                }
                QLabel {
                    color: rgba(0, 0, 0, 200);
                    font-size: 14px;
                }
                QProgressBar {
                    border: 1px solid rgba(200, 200, 200, 200);
                    border-radius: 3px;
                    text-align: center;
                    background-color: rgba(240, 240, 240, 180);
                }
                QProgressBar::chunk {
                    background-color: rgba(0, 120, 215, 180);
                }
                QScrollArea {
                    border: none;
                    background-color: transparent;
                }
                QWidget#central {
                    background-color: rgba(255, 255, 255, 200);
                    border: 1px solid rgba(200, 200, 200, 200);
                    border-radius: 5px;
                }
            """)
            
            # 设置菜单栏样式
            self.menuBar().setStyleSheet("""
                QMenuBar {
                    background-color: rgba(240, 240, 240, 200);
                    border-bottom: 1px solid rgba(200, 200, 200, 200);
                }
                QMenuBar::item {
                    padding: 5px 10px;
                    background-color: transparent;
                }
                QMenuBar::item:selected {
                    background-color: rgba(0, 120, 215, 180);
                    color: white;
                }
                QMenu {
                    background-color: rgba(255, 255, 255, 220);
                    border: 1px solid rgba(200, 200, 200, 200);
                }
                QMenu::item {
                    padding: 5px 20px;
                }
                QMenu::item:selected {
                    background-color: rgba(0, 120, 215, 180);
                    color: white;
                }
            """)
            
        except Exception as e:
            print(f"更新窗口样式错误: {e}")

    def change_font_size(self, delta):
        """改变字体大小"""
        try:
            # 获取当前字体
            font = self.original_label.font()
            size = font.pointSize()
            
            # 调整字体大小
            new_size = max(8, min(72, size + delta))
            font.setPointSize(new_size)
            
            # 应用新字体
            self.original_label.setFont(font)
            self.translation_label.setFont(font)
            
            print(f"字体大小已更改为: {new_size}")
            
        except Exception as e:
            self.handle_error("字体", f"更改字体大小失败: {e}")

    def set_opacity(self, value):
        """设置窗口透明度"""
        try:
            # 限制透明度范围
            value = max(0.2, min(1.0, value))
            
            # 设置窗口透明度
            self.setWindowOpacity(value)
            
            # 更新滑块位置
            if hasattr(self, 'opacity_slider'):
                self.opacity_slider.setValue(int(value * 100))
                
        except Exception as e:
            self.handle_error("透明度", f"设置透明度失败: {e}")

    def on_source_changed(self, source):
        """音频源改变处理"""
        try:
            # 更新文件选择按钮可见性
            self.file_button.setVisible(source == "文件")
            
            # 如果正在运行，停止当前转录
            if self.is_running:
                self.stop_transcription()
            
            # 更新音频源
            self.is_system_audio = (source == "系统音频")
            self.file_path = None
            
            # 如果设置了自动启动，则重新开始转录
            if self.auto_start:
                self.start_transcription()
                
        except Exception as e:
            self.handle_error("音频源", f"切换音频源失败: {e}")

    def update_translation(self, text):
        """更新翻译显示"""
        try:
            if not self.enable_translation or not self.translator:
                return
            
            # 如果是临时文本，直接翻译
            if text.startswith("PARTIAL:"):
                partial_text = text[8:]
                translated = self.translator.translate(partial_text)
                self.translation_label.setText(translated)
                return
            
            # 添加到翻译文本列表
            translated = self.translator.translate(text)
            self.translation_text.append(translated)
            
            # 更新显示
            display_text = '\n'.join(self.translation_text[-10:])
            self.translation_label.setText(display_text)
            
            # 同步滚动
            if self.sync_scroll:
                QTimer.singleShot(10, lambda: self.translation_scroll_area.verticalScrollBar().setValue(
                    self.translation_scroll_area.verticalScrollBar().maximum()
                ))
                
        except Exception as e:
            self.handle_error("翻译", f"更新翻译显示失败: {e}")

    def update_speaker_buffer(self, audio_data):
        """更新说话人识别缓冲区"""
        try:
            if not self.enable_speaker_id or not self.speaker_identifier:
                return
            
            # 识别说话人
            speaker_id = self.speaker_identifier.identify_speaker(audio_data)
            if speaker_id is not None and 0 <= speaker_id < len(self.speaker_names):
                speaker_name = self.speaker_names[speaker_id]
                print(f"当前说话人: {speaker_name}")
                self.signals.new_text.emit(f"PARTIAL:【{speaker_name}】")
                
        except Exception as e:
            self.handle_error("说话人识别", f"更新说话人识别失败: {e}")

    def switch_audio_source(self, is_system_audio):
        """切换音频源"""
        try:
            # 停止当前转录
            if self.is_running:
                self.stop_transcription()
            
            # 切换音频源
            self.is_system_audio = is_system_audio
            self.file_path = None
            
            # 更新菜单状态
            self.update_menu_state()
            
            # 重新启动转录
            self.start_transcription()
            
        except Exception as e:
            self.handle_error("音频源", f"切换音频源失败: {e}")

    def change_font_size(self, delta):
        """改变字体大小"""
        try:
            # 获取当前字体
            font = self.original_label.font()
            size = font.pointSize()
            
            # 调整字体大小
            new_size = max(8, min(72, size + delta))
            font.setPointSize(new_size)
            
            # 应用新字体
            self.original_label.setFont(font)
            self.translation_label.setFont(font)
            
            print(f"字体大小已更改为: {new_size}")
            
        except Exception as e:
            self.handle_error("字体", f"更改字体大小失败: {e}")

    def on_language_action_triggered(self, lang):
        """语言选择处理"""
        try:
            # 停止当前转录
            if self.is_running:
                self.stop_transcription()
            
            # 更新语言设置
            if lang == "auto":
                print("切换到自动语言检测模式")
            elif lang == "en":
                print("切换到英语识别模式")
            elif lang == "cn":
                print("切换到中文识别模式")
            
            # 重新初始化 ASR 引擎
            self.init_asr_engine()
            
            # 重新启动转录
            if self.is_running:
                self.start_transcription()
                
        except Exception as e:
            self.handle_error("语言设置", f"切换语言失败: {e}")

    def process_audio_chunk(self, chunk):
        """处理音频数据块"""
        try:
            # 确保数据是单声道
            if len(chunk.shape) > 1:
                chunk = chunk[:, 0]
            
            # 更新说话人识别
            if self.enable_speaker_id:
                self.update_speaker_buffer(chunk)
            
            # 转录音频
            result = self.asr_engine.transcribe(chunk)
            if result:
                self.signals.new_text.emit(result)
                
        except Exception as e:
            self.handle_error("音频处理", f"处理音频数据块失败: {e}")

    def stop_transcription(self):
        """停止转录"""
        try:
            self.is_running = False
            
            # 等待音频线程结束
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=1)
            
            # 获取最终结果
            if isinstance(self.asr_engine, VoskASR):
                final_result = self.asr_engine.get_final_result()
                if final_result:
                    self.process_transcription_result(final_result)
            
            # 更新界面状态
            self.start_button.setText("开始转录")
            self.start_button.setStyleSheet("""
                QPushButton {
                    background-color: rgba(50, 150, 50, 200);
                    color: white;
                    border-radius: 3px;
                    padding: 5px 15px;
                }
                QPushButton:hover {
                    background-color: rgba(60, 170, 60, 200);
                }
            """)
            
            # 隐藏进度条
            self.progress_bar.hide()
            
            print("转录已停止")
            
        except Exception as e:
            self.handle_error("停止转录", f"停止转录失败: {e}")

    def process_transcription_result(self, text):
        """处理转录结果"""
        try:
            # 添加说话人标记
            if self.enable_speaker_id and hasattr(self, 'current_speaker') and self.current_speaker:
                text = f"【{self.current_speaker}】{text}"
            
            # 添加到转录文本列表
            self.transcript_text.append(text)
            
            # 更新显示
            display_text = '\n'.join(self.transcript_text[-10:])
            self.original_label.setText(display_text)
            
            # 同步滚动
            if self.sync_scroll:
                QTimer.singleShot(10, lambda: self.original_scroll_area.verticalScrollBar().setValue(
                    self.original_scroll_area.verticalScrollBar().maximum()
                ))
            
            # 更新翻译
            if self.enable_translation:
                self.update_translation(text)
                
        except Exception as e:
            self.handle_error("转录结果", f"处理转录结果失败: {e}")

    def dragEnterEvent(self, event):
        """拖拽进入事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """拖拽放下事件"""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('.wav', '.mp3', '.m4a', '.aac', '.ogg', '.flac')):
                self.file_path = file_path
                if self.is_running:
                    self.stop_transcription()
                self.start_transcription()
            else:
                self.handle_error("文件", "不支持的文件格式")

    def contextMenuEvent(self, event):
        """右键菜单事件"""
        menu = QMenu(self)
        
        # 添加基本功能
        start_stop_action = menu.addAction("停止转录" if self.is_running else "开始转录")
        menu.addSeparator()
        
        # 添加音频源选择
        source_menu = menu.addMenu("音频源")
        system_action = source_menu.addAction("系统音频")
        system_action.setCheckable(True)
        system_action.setChecked(self.is_system_audio and not self.file_path)
        
        mic_action = source_menu.addAction("麦克风")
        mic_action.setCheckable(True)
        mic_action.setChecked(not self.is_system_audio and not self.file_path)
        
        file_action = source_menu.addAction("文件")
        file_action.setCheckable(True)
        file_action.setChecked(bool(self.file_path))
        
        # 添加功能开关
        menu.addSeparator()
        translation_action = menu.addAction("启用翻译")
        translation_action.setCheckable(True)
        translation_action.setChecked(self.enable_translation)
        
        speaker_action = menu.addAction("说话人识别")
        speaker_action.setCheckable(True)
        speaker_action.setChecked(self.enable_speaker_id)
        
        # 添加其他功能
        menu.addSeparator()
        save_action = menu.addAction("保存转录")
        clear_action = menu.addAction("清空显示")
        
        # 显示菜单并处理选择
        action = menu.exec_(event.globalPos())
        
        if action == start_stop_action:
            self.toggle_transcription()
        elif action == system_action:
            self.switch_audio_source(True)
        elif action == mic_action:
            self.switch_audio_source(False)
        elif action == file_action:
            self.select_file()
        elif action == translation_action:
            self.toggle_translation(not self.enable_translation)
        elif action == speaker_action:
            self.toggle_speaker_id(not self.enable_speaker_id)
        elif action == save_action:
            self.save_transcript()
        elif action == clear_action:
            self.clear_display()

    def clear_display(self):
        """清空显示"""
        self.transcript_text.clear()
        self.translation_text.clear()
        self.original_label.clear()
        self.translation_label.clear()
        
        if self.enable_speaker_id and self.speaker_identifier:
            self.speaker_identifier.reset()
        
        print("显示已清空")
        self.signals.new_text.emit("PARTIAL:显示已清空")

    def setup_shortcuts(self):
        """设置快捷键"""
        # Ctrl+Q: 退出
        quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        quit_shortcut.activated.connect(self.close)
        
        # Ctrl+S: 保存
        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.save_transcript)
        
        # Ctrl+O: 打开文件
        open_shortcut = QShortcut(QKeySequence("Ctrl+O"), self)
        open_shortcut.activated.connect(self.select_file)
        
        # Space: 开始/停止转录
        toggle_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        toggle_shortcut.activated.connect(self.toggle_transcription)
        
        # Ctrl++/-: 调整字体大小
        font_increase = QShortcut(QKeySequence("Ctrl++"), self)
        font_increase.activated.connect(lambda: self.change_font_size(1))
        
        font_decrease = QShortcut(QKeySequence("Ctrl+-"), self)
        font_decrease.activated.connect(lambda: self.change_font_size(-1))

    def check_updates(self):
        """检查更新"""
        try:
            # 获取当前版本
            current_version = "1.0.0"
            
            # TODO: 实现在线检查更新
            print(f"当前版本: {current_version}")
            self.signals.new_text.emit(f"PARTIAL:当前版本: {current_version}")
            
        except Exception as e:
            self.handle_error("更新检查", f"检查更新失败: {e}")

    def export_settings(self):
        """导出设置"""
        try:
            settings = {
                "auto_start": self.auto_start,
                "enable_translation": self.enable_translation,
                "sync_scroll": self.sync_scroll,
                "enable_speaker_id": self.enable_speaker_id,
                "is_system_audio": self.is_system_audio,
                "window_geometry": [self.x(), self.y(), self.width(), self.height()],
                "opacity": self.windowOpacity(),
                "speaker_names": self.speaker_names
            }
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "导出设置",
                os.path.join(self.base_dir, "settings.json"),
                "JSON文件 (*.json);;所有文件 (*)"
            )
            
            if file_path:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(settings, f, indent=4, ensure_ascii=False)
                print(f"设置已导出到: {file_path}")
                self.signals.new_text.emit(f"PARTIAL:设置已导出到: {file_path}")
                
        except Exception as e:
            self.handle_error("导出设置", f"导出设置失败: {e}")

    def import_settings(self):
        """导入设置"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "导入设置",
                self.base_dir,
                "JSON文件 (*.json);;所有文件 (*)"
            )
            
            if file_path:
                with open(file_path, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                
                # 应用设置
                self.auto_start = settings.get("auto_start", False)
                self.enable_translation = settings.get("enable_translation", False)
                self.sync_scroll = settings.get("sync_scroll", True)
                self.enable_speaker_id = settings.get("enable_speaker_id", False)
                self.is_system_audio = settings.get("is_system_audio", True)
                
                # 恢复窗口位置和大小
                geometry = settings.get("window_geometry")
                if geometry:
                    self.setGeometry(*geometry)
                
                # 设置透明度
                opacity = settings.get("opacity", 1.0)
                self.set_opacity(opacity)
                
                # 更新说话人名称
                speaker_names = settings.get("speaker_names")
                if speaker_names:
                    self.speaker_names = speaker_names
                
                # 更新菜单状态
                self.update_menu_state()
                
                print(f"设置已从 {file_path} 导入")
                self.signals.new_text.emit(f"PARTIAL:设置已导入")
                
        except Exception as e:
            self.handle_error("导入设置", f"导入设置失败: {e}")

    def update_menu_state(self):
        """更新菜单状态"""
        # 更新音频源菜单
        if hasattr(self, 'system_audio_action'):
            self.system_audio_action.setChecked(self.is_system_audio and not self.file_path)
        if hasattr(self, 'microphone_action'):
            self.microphone_action.setChecked(not self.is_system_audio and not self.file_path)
        if hasattr(self, 'file_audio_action'):
            self.file_audio_action.setChecked(bool(self.file_path))
        
        # 更新功能开关菜单
        if hasattr(self, 'translation_enable_action'):
            self.translation_enable_action.setChecked(self.enable_translation)
        if hasattr(self, 'sync_scroll_action'):
            self.sync_scroll_action.setChecked(self.sync_scroll)
        if hasattr(self, 'speaker_enable_action'):
            self.speaker_enable_action.setChecked(self.enable_speaker_id)
        if hasattr(self, 'auto_start_action'):
            self.auto_start_action.setChecked(self.auto_start)

    def auto_start_transcription(self):
        """自动启动转录"""
        if self.auto_start and not self.is_running:
            QTimer.singleShot(1000, self.start_transcription)

    def populate_audio_devices_menu(self):
        """填充音频设备菜单"""
        try:
            # 获取所有音频设备
            speakers = sc.all_speakers()
            microphones = sc.all_microphones()
            
            # 创建设备菜单
            devices_menu = QMenu("音频设备", self)
            
            # 添加扬声器
            speakers_menu = devices_menu.addMenu("扬声器")
            for speaker in speakers:
                action = speakers_menu.addAction(speaker.name)
                action.setCheckable(True)
                action.setChecked(speaker.id == sc.default_speaker().id)
                action.triggered.connect(lambda checked, s=speaker: self.on_speaker_selected(s))
            
            # 添加麦克风
            mics_menu = devices_menu.addMenu("麦克风")
            for mic in microphones:
                action = mics_menu.addAction(mic.name)
                action.setCheckable(True)
                action.setChecked(mic.id == sc.default_microphone().id)
                action.triggered.connect(lambda checked, m=mic: self.on_microphone_selected(m))
            
            return devices_menu
            
        except Exception as e:
            self.handle_error("音频设备", f"获取音频设备列表失败: {e}")
            return None

    def on_speaker_selected(self, speaker):
        """扬声器选择处理"""
        try:
            print(f"选择扬声器: {speaker.name}")
            # TODO: 实现扬声器切换逻辑
            if self.is_running:
                self.restart_transcription()
                
        except Exception as e:
            self.handle_error("扬声器", f"切换扬声器失败: {e}")

    def on_microphone_selected(self, microphone):
        """麦克风选择处理"""
        try:
            print(f"选择麦克风: {microphone.name}")
            # TODO: 实现麦克风切换逻辑
            if self.is_running:
                self.restart_transcription()
                
        except Exception as e:
            self.handle_error("麦克风", f"切换麦克风失败: {e}")

    def restart_transcription(self):
        """重新启动转录"""
        if self.is_running:
            self.stop_transcription()
        QTimer.singleShot(500, self.start_transcription)

    def update_speaker_names(self, names):
        """更新说话人名称"""
        try:
            if len(names) > self.max_speakers:
                names = names[:self.max_speakers]
            self.speaker_names = names
            print(f"说话人名称已更新: {names}")
            
        except Exception as e:
            self.handle_error("说话人", f"更新说话人名称失败: {e}")

    def create_speaker_menu(self):
        """创建说话人菜单"""
        try:
            speaker_menu = QMenu("说话人设置", self)
            
            # 添加说话人名称编辑
            for i, name in enumerate(self.speaker_names):
                action = speaker_menu.addAction(f"说话人 {i+1}: {name}")
                action.triggered.connect(lambda checked, idx=i: self.edit_speaker_name(idx))
            
            speaker_menu.addSeparator()
            
            # 添加最大说话人数量设置
            max_speakers_action = speaker_menu.addAction(f"最大说话人数: {self.max_speakers}")
            max_speakers_action.triggered.connect(self.set_max_speakers)
            
            return speaker_menu
            
        except Exception as e:
            self.handle_error("菜单", f"创建说话人菜单失败: {e}")
            return None

    def edit_speaker_name(self, index):
        """编辑说话人名称"""
        try:
            current_name = self.speaker_names[index]
            new_name, ok = QInputDialog.getText(
                self,
                "编辑说话人名称",
                f"说话人 {index + 1}:",
                QLineEdit.Normal,
                current_name
            )
            
            if ok and new_name:
                self.speaker_names[index] = new_name
                print(f"说话人 {index + 1} 名称已更改为: {new_name}")
                
        except Exception as e:
            self.handle_error("说话人", f"编辑说话人名称失败: {e}")

    def set_max_speakers(self):
        """设置最大说话人数量"""
        try:
            new_max, ok = QInputDialog.getInt(
                self,
                "设置最大说话人数",
                "最大说话人数 (2-10):",
                value=self.max_speakers,
                min=2,
                max=10
            )
            
            if ok:
                self.max_speakers = new_max
                # 调整说话人名称列表
                while len(self.speaker_names) < new_max:
                    self.speaker_names.append(f"说话人{len(self.speaker_names)+1}")
                if len(self.speaker_names) > new_max:
                    self.speaker_names = self.speaker_names[:new_max]
                
                print(f"最大说话人数已更改为: {new_max}")
                
        except Exception as e:
            self.handle_error("说话人", f"设置最大说话人数失败: {e}")

    def create_help_menu(self):
        """创建帮助菜单"""
        try:
            help_menu = QMenu("帮助", self)
            
            # 添加快捷键说明
            shortcuts_action = help_menu.addAction("快捷键说明")
            shortcuts_action.triggered.connect(self.show_shortcuts_help)
            
            # 添加关于信息
            about_action = help_menu.addAction("关于")
            about_action.triggered.connect(self.show_about)
            
            # 添加检查更新
            update_action = help_menu.addAction("检查更新")
            update_action.triggered.connect(self.check_updates)
            
            return help_menu
            
        except Exception as e:
            self.handle_error("菜单", f"创建帮助菜单失败: {e}")
            return None

    def show_shortcuts_help(self):
        """显示快捷键说明"""
        shortcuts_text = """
快捷键说明：

Ctrl+Q: 退出程序
Ctrl+S: 保存转录文本
Ctrl+O: 打开音频文件
Space: 开始/停止转录
Ctrl++: 增大字体
Ctrl+-: 减小字体
        """
        
        QMessageBox.information(self, "快捷键说明", shortcuts_text)

    def show_about(self):
        """显示关于信息"""
        about_text = """
实时字幕转录工具 v1.0.0

支持功能：
- 系统音频转录
- 麦克风音频转录
- 音频文件转录
- 说话人识别
- 实时翻译

作者：Your Name
许可证：MIT License
        """
        
        QMessageBox.about(self, "关于", about_text)

    def switch_translation_engine(self, engine_type):
        """切换翻译引擎"""
        if engine_type == self.config_manager.get("translation.engine"):
            return  # 已经是当前引擎
        
        # 更新配置
        self.config_manager.set("translation.engine", engine_type)
        
        # 重新初始化翻译引擎
        self.translation_manager.initialize_engine(engine_type)
        
        # 显示状态消息
        self.status_label.setText(f"已切换到 {engine_type} 翻译引擎")

    def create_menu(self):
        """创建菜单栏"""
        print("正在创建菜单...")  # 调试输出
        
        # 清除现有菜单
        if self.menuBar():
            self.menuBar().clear()
        
        menubar = self.menuBar() or QMenuBar(self)
        self.setMenuBar(menubar)
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        # 添加打开文件选项
        open_action = QAction("打开音频/视频文件", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        # 添加保存选项
        save_action = QAction("保存转录文本", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_transcript)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # 添加退出选项
        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 模型选择菜单
        model_menu = menubar.addMenu("模型选择")
        
        # ASR 模型子菜单
        asr_menu = model_menu.addMenu("语音识别模型")
        
        # 创建 ASR 引擎选择组
        self.asr_group = QActionGroup(self)
        self.asr_group.setExclusive(True)  # 确保只能选择一个
        
        # Vosk ASR 选项
        self.vosk_action = QAction("Vosk ASR", self)
        self.vosk_action.setCheckable(True)
        self.vosk_action.setChecked(self.config_manager.get("asr.engine", "vosk") == "vosk")
        self.vosk_action.triggered.connect(lambda: self.switch_asr_engine("vosk"))
        self.asr_group.addAction(self.vosk_action)
        asr_menu.addAction(self.vosk_action)
        
        # Sherpa ONNX ASR 选项
        if SHERPA_AVAILABLE:
            self.sherpa_action = QAction("Sherpa ONNX ASR", self)
            self.sherpa_action.setCheckable(True)
            self.sherpa_action.setChecked(self.config_manager.get("asr.engine", "vosk") == "sherpa")
            self.sherpa_action.triggered.connect(lambda: self.switch_asr_engine("sherpa"))
            self.asr_group.addAction(self.sherpa_action)
            asr_menu.addAction(self.sherpa_action)
        else:
            self.sherpa_action = QAction("Sherpa ONNX ASR (未安装)", self)
            self.sherpa_action.setEnabled(False)
            asr_menu.addAction(self.sherpa_action)
        
        # 翻译模型子菜单
        translation_menu = model_menu.addMenu("翻译模型")
        
        # 翻译方向选择
        direction_menu = translation_menu.addMenu("翻译方向")
        
        # 创建翻译方向选择组
        direction_group = QActionGroup(self)
        direction_group.setExclusive(True)
        
        # 英译中选项
        en_zh_action = QAction("英文 → 中文", self)
        en_zh_action.setCheckable(True)
        en_zh_action.setChecked(True)  # 默认选中
        en_zh_action.triggered.connect(lambda: self.set_translation_direction("en-zh"))
        direction_group.addAction(en_zh_action)
        direction_menu.addAction(en_zh_action)
        
        # 中译英选项 (暂时禁用)
        zh_en_action = QAction("中文 → 英文 (暂不可用)", self)
        zh_en_action.setCheckable(True)
        zh_en_action.setEnabled(False)
        direction_group.addAction(zh_en_action)
        direction_menu.addAction(zh_en_action)
        
        translation_menu.addSeparator()
        
        # 翻译引擎选择
        # 创建翻译引擎选择组
        self.translation_group = QActionGroup(self)
        self.translation_group.setExclusive(True)
        
        # Opus-MT 选项
        self.opusmt_action = QAction("Opus-MT", self)
        self.opusmt_action.setCheckable(True)
        self.opusmt_action.setChecked(self.config_manager.get("translation.engine", "opusmt") == "opusmt")
        self.opusmt_action.triggered.connect(lambda: self.switch_translation_engine("opusmt"))
        self.translation_group.addAction(self.opusmt_action)
        translation_menu.addAction(self.opusmt_action)
        
        # Argos 选项
        if ARGOS_AVAILABLE:
            self.argos_action = QAction("Argos Translate", self)
            self.argos_action.setCheckable(True)
            self.argos_action.setChecked(self.config_manager.get("translation.engine", "opusmt") == "argos")
            self.argos_action.triggered.connect(lambda: self.switch_translation_engine("argos"))
            self.translation_group.addAction(self.argos_action)
            translation_menu.addAction(self.argos_action)
        else:
            self.argos_action = QAction("Argos Translate (未安装)", self)
            self.argos_action.setEnabled(False)
            translation_menu.addAction(self.argos_action)
        
        # 启用/禁用翻译
        translation_menu.addSeparator()
        self.enable_translation_action = QAction("启用翻译", self)
        self.enable_translation_action.setCheckable(True)
        self.enable_translation_action.setChecked(self.config_manager.get("translation.enable", False))
        self.enable_translation_action.triggered.connect(self.toggle_translation)
        translation_menu.addAction(self.enable_translation_action)
        
        # 功能菜单
        feature_menu = menubar.addMenu("功能")
        
        # 添加说话人识别选项
        self.speaker_id_action = QAction("启用说话人识别", self)
        self.speaker_id_action.setCheckable(True)
        self.speaker_id_action.setChecked(self.config_manager.get("asr.enable_speaker_id", False))
        self.speaker_id_action.triggered.connect(self.toggle_speaker_id)
        feature_menu.addAction(self.speaker_id_action)
        
        # 设置菜单
        settings_menu = menubar.addMenu("设置")
        
        # 添加透明度设置
        opacity_menu = settings_menu.addMenu("窗口透明度")
        
        # 创建透明度选择组
        opacity_group = QActionGroup(self)
        opacity_group.setExclusive(True)
        
        # 不透明选项
        opaque_action = QAction("不透明", self)
        opaque_action.setCheckable(True)
        opaque_action.setChecked(self.config_manager.get("window.opacity", 0.8) >= 1.0)
        opaque_action.triggered.connect(lambda: self.set_opacity(1.0))
        opacity_group.addAction(opaque_action)
        opacity_menu.addAction(opaque_action)
        
        # 半透明选项
        translucent_action = QAction("半透明", self)
        translucent_action.setCheckable(True)
        translucent_action.setChecked(0.3 < self.config_manager.get("window.opacity", 0.8) < 1.0)
        translucent_action.triggered.connect(lambda: self.set_opacity(0.8))
        opacity_group.addAction(translucent_action)
        opacity_menu.addAction(translucent_action)
        
        # 全透明选项
        transparent_action = QAction("全透明", self)
        transparent_action.setCheckable(True)
        transparent_action.setChecked(self.config_manager.get("window.opacity", 0.8) <= 0.3)
        transparent_action.triggered.connect(lambda: self.set_opacity(0.3))
        opacity_group.addAction(transparent_action)
        opacity_menu.addAction(transparent_action)
        
        # 添加字体大小设置
        font_menu = settings_menu.addMenu("字体大小")
        
        # 创建字体大小选择组
        font_group = QActionGroup(self)
        font_group.setExclusive(True)
        
        # 小字体选项
        small_font_action = QAction("小", self)
        small_font_action.setCheckable(True)
        small_font_action.setChecked(self.config_manager.get("window.font_size", 14) <= 12)
        small_font_action.triggered.connect(lambda: self.set_font_size(12))
        font_group.addAction(small_font_action)
        font_menu.addAction(small_font_action)
        
        # 中字体选项
        medium_font_action = QAction("中", self)
        medium_font_action.setCheckable(True)
        medium_font_action.setChecked(12 < self.config_manager.get("window.font_size", 14) <= 16)
        medium_font_action.triggered.connect(lambda: self.set_font_size(14))
        font_group.addAction(medium_font_action)
        font_menu.addAction(medium_font_action)
        
        # 大字体选项
        large_font_action = QAction("大", self)
        large_font_action.setCheckable(True)
        large_font_action.setChecked(self.config_manager.get("window.font_size", 14) > 16)
        large_font_action.triggered.connect(lambda: self.set_font_size(18))
        font_group.addAction(large_font_action)
        font_menu.addAction(large_font_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        
        # 添加快捷键说明选项
        shortcuts_action = QAction("快捷键说明", self)
        shortcuts_action.triggered.connect(self.show_shortcuts_help)
        help_menu.addAction(shortcuts_action)
        
        # 添加关于选项
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        # 确保菜单可见
        menubar.setVisible(True)
        
        # 强制更新
        menubar.update()
        QApplication.processEvents()
        
        print(f"菜单创建完成，菜单项数量: {len(menubar.actions())}")  # 调试输出
        
        # 更新 setup_signals 中使用的菜单项引用
        self.update_menu_references()

    def set_translation_direction(self, direction):
        """设置翻译方向"""
        if direction == self.config_manager.get("translation.direction", "en-zh"):
            return  # 已经是当前方向
        
        # 更新配置
        self.config_manager.set("translation.direction", direction)
        
        # 显示状态消息
        self.status_label.setText(f"已设置翻译方向: {direction}")
        
        # 如果需要，重新初始化翻译引擎
        engine_type = self.config_manager.get("translation.engine", "opusmt")
        self.translation_manager.initialize_engine(engine_type, direction)

    def update_menu_references(self):
        """更新菜单引用，确保 setup_signals 使用正确的菜单项"""
        # 这个方法会在 create_menu 之后调用
        # 确保 setup_signals 中使用的菜单项引用是最新的
        
        # 如果 setup_signals 已经运行，可能需要断开旧的连接
        try:
            if hasattr(self, 'metrics_timer'):
                self.metrics_timer.timeout.disconnect()
        except:
            pass
        
        # 重新设置信号连接
        self.setup_signals()

    # 1. 添加缺失的 open_file 方法
    def open_file(self):
        """打开音频/视频文件"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "选择音频/视频文件", 
                "", 
                "媒体文件 (*.mp3 *.wav *.mp4 *.avi *.mkv);;所有文件 (*)"
            )
            
            if file_path:
                # 停止当前转录
                if self.is_running:
                    self.stop_transcription()
                
                # 设置文件路径
                self.file_path = file_path
                self.status_label.setText(f"已选择文件: {os.path.basename(file_path)}")
                
                # 启动文件转录
                self.start_file_transcription(file_path)
        except Exception as e:
            self.handle_error("文件选择", f"打开文件失败: {e}")

    # 2. 添加文件转录方法
    def start_file_transcription(self, file_path):
        """开始文件转录"""
        try:
            # 设置状态
            self.is_running = True
            self.status_label.setText(f"正在转录: {os.path.basename(file_path)}")
            self.update_control_states()
            
            # 创建转录线程
            self.transcription_thread = threading.Thread(
                target=self.transcribe_file,
                args=(file_path,)
            )
            self.transcription_thread.daemon = True
            self.transcription_thread.start()
            
        except Exception as e:
            self.handle_error("文件转录", f"启动文件转录失败: {e}")
            self.is_running = False
            self.update_control_states()

    # 3. 添加文件转录处理方法
    def transcribe_file(self, file_path):
        """处理文件转录"""
        try:
            # 使用 FFmpeg 提取音频
            import subprocess
            import tempfile
            
            # 创建临时文件
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav.close()
            
            # 使用 FFmpeg 转换为 WAV
            cmd = [
                'ffmpeg', '-y', '-i', file_path,
                '-ar', '16000', '-ac', '1', '-f', 'wav', temp_wav.name
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 重置 ASR 引擎
            self.asr_manager.reset()
            
            # 读取 WAV 文件并转录
            import wave
            with wave.open(temp_wav.name, 'rb') as wf:
                # 获取文件信息
                total_frames = wf.getnframes()
                frames_processed = 0
                
                # 分块读取并处理
                chunk_size = 4000  # 每次处理的帧数
                while self.is_running:
                    frames = wf.readframes(chunk_size)
                    if not frames:
                        break
                    
                    # 更新进度
                    frames_processed += len(frames) // 2  # 16位音频，每帧2字节
                    progress = min(100, int(frames_processed / total_frames * 100))
                    self.signals.progress_update.emit(progress)
                    
                    # 转录音频
                    result = self.asr_manager.transcribe(frames)
                    if result:
                        if result.startswith("PARTIAL:"):
                            self.signals.new_text.emit(result[9:])  # 去掉 "PARTIAL:" 前缀
                        else:
                            self.signals.new_text.emit(result)
                            self.process_transcription_result(result)
                
                # 获取最终结果
                final_result = self.asr_manager.get_final_result()
                if final_result:
                    self.signals.new_text.emit(final_result)
                    self.process_transcription_result(final_result)
            
            # 清理临时文件
            try:
                os.unlink(temp_wav.name)
            except:
                pass
            
            # 完成转录
            self.signals.transcription_finished.emit()
            
        except Exception as e:
            self.handle_error("文件转录", f"文件转录错误: {e}")
            import traceback
            print(traceback.format_exc())
        
        finally:
            # 确保状态正确
            self.is_running = False
            self.signals.transcription_finished.emit()

    # 4. 添加保存转录文本方法
    def save_transcript(self):
        """保存转录文本"""
        try:
            # 如果没有转录内容，则不保存
            if not self.transcript_text:
                self.status_label.setText("没有可保存的转录内容")
                return
            
            # 打开保存对话框
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "保存转录文本", 
                "", 
                "文本文件 (*.txt);;所有文件 (*)"
            )
            
            if file_path:
                # 确保文件扩展名
                if not file_path.endswith('.txt'):
                    file_path += '.txt'
                
                # 保存文本
                with open(file_path, 'w', encoding='utf-8') as f:
                    for line in self.transcript_text:
                        f.write(line + '\n')
                
                self.status_label.setText(f"转录已保存到: {os.path.basename(file_path)}")
        except Exception as e:
            self.handle_error("保存转录", f"保存转录文本失败: {e}")

    # 5. 添加快捷键帮助方法
    def show_shortcuts_help(self):
        """显示快捷键帮助"""
        help_text = """
        <h3>快捷键说明</h3>
        <table>
            <tr><td><b>Ctrl+O</b></td><td>打开音频/视频文件</td></tr>
            <tr><td><b>Ctrl+S</b></td><td>保存转录文本</td></tr>
            <tr><td><b>Ctrl+Q</b></td><td>退出程序</td></tr>
            <tr><td><b>Space</b></td><td>开始/停止转录</td></tr>
            <tr><td><b>Ctrl+C</b></td><td>复制选中文本</td></tr>
            <tr><td><b>Ctrl+A</b></td><td>全选文本</td></tr>
            <tr><td><b>Ctrl+L</b></td><td>清空转录内容</td></tr>
        </table>
        """
        QMessageBox.information(self, "快捷键说明", help_text)

    # 6. 添加翻译方向设置方法
    def set_translation_direction(self, direction):
        """设置翻译方向"""
        if direction == self.config_manager.get("translation.direction", "en-zh"):
            return  # 已经是当前方向
        
        # 更新配置
        self.config_manager.set("translation.direction", direction)
        
        # 显示状态消息
        self.status_label.setText(f"已设置翻译方向: {direction}")
        
        # 如果需要，重新初始化翻译引擎
        engine_type = self.config_manager.get("translation.engine", "opusmt")
        self.translation_manager.initialize_engine(engine_type, direction)

    # 添加缺失的 create_status_bar 方法
    def create_status_bar(self):
        """创建状态栏"""
        # 创建状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # 添加状态标签
        self.status_label = QLabel("就绪")
        self.statusBar.addWidget(self.status_label, 1)
        
        # 添加性能指标标签
        self.cpu_label = QLabel("CPU: 0%")
        self.statusBar.addPermanentWidget(self.cpu_label)
        
        self.memory_label = QLabel("内存: 0MB")
        self.statusBar.addPermanentWidget(self.memory_label)
        
        # 添加引擎标签
        self.engine_label = QLabel("引擎: Vosk")
        self.statusBar.addPermanentWidget(self.engine_label)

    def check_model_directories(self):
        """检查模型目录"""
        # 获取模型目录
        vosk_model_dir = self.config_manager.get("models.vosk", "model")
        sherpa_model_dir = self.config_manager.get("models.sherpa", "sherpa_models")
        speaker_model_dir = self.config_manager.get("models.speaker", "speaker_model")
        bergamot_model_dir = self.config_manager.get("models.bergamot", "bergamot_models")
        
        # 打印模型目录
        print(f"模型目录 vosk: {os.path.abspath(vosk_model_dir)}")
        print(f"模型目录 sherpa: {os.path.abspath(sherpa_model_dir)}")
        print(f"模型目录 speaker: {os.path.abspath(speaker_model_dir)}")
        print(f"模型目录 bergamot: {os.path.abspath(bergamot_model_dir)}")
        
        # 检查 Vosk 模型目录
        if not os.path.exists(vosk_model_dir):
            print(f"警告: Vosk 模型目录不存在: {vosk_model_dir}")
            os.makedirs(vosk_model_dir, exist_ok=True)
        
        # 检查 Sherpa 模型目录
        if not os.path.exists(sherpa_model_dir):
            print(f"警告: Sherpa 模型目录不存在: {sherpa_model_dir}")
            os.makedirs(sherpa_model_dir, exist_ok=True)
        
        # 检查说话人模型目录
        if not os.path.exists(speaker_model_dir):
            print(f"警告: 说话人模型目录不存在: {speaker_model_dir}")
            os.makedirs(speaker_model_dir, exist_ok=True)
        
        # 检查翻译模型目录
        if not os.path.exists(bergamot_model_dir):
            print(f"警告: 翻译模型目录不存在: {bergamot_model_dir}")
            os.makedirs(bergamot_model_dir, exist_ok=True)

    # 添加一个方法来更新模型路径
    def update_model_paths(self):
        """更新模型路径"""
        try:
            # 获取当前目录
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 检查 Vosk 模型
            vosk_model_dir = self.config_manager.get("models.vosk", "model")
            vosk_model_path = os.path.join(base_dir, vosk_model_dir)
            
            # 如果模型目录不存在或为空，尝试查找子目录
            if not os.path.exists(os.path.join(vosk_model_path, "final.mdl")):
                # 查找可能的子目录
                for subdir in os.listdir(vosk_model_path):
                    subdir_path = os.path.join(vosk_model_path, subdir)
                    if os.path.isdir(subdir_path) and os.path.exists(os.path.join(subdir_path, "final.mdl")):
                        # 更新配置
                        new_path = os.path.join(vosk_model_dir, subdir)
                        self.config_manager.set("models.vosk", new_path)
                        print(f"已更新 Vosk 模型路径: {new_path}")
                        break
        
            # 检查 Sherpa 模型
            sherpa_model_dir = self.config_manager.get("models.sherpa", "sherpa_models")
            sherpa_model_path = os.path.join(base_dir, sherpa_model_dir)
            
            # 如果模型目录不存在或为空，尝试查找子目录
            if not os.path.exists(os.path.join(sherpa_model_path, "encoder.onnx")):
                # 查找可能的子目录
                for subdir in os.listdir(sherpa_model_path):
                    subdir_path = os.path.join(sherpa_model_path, subdir)
                    if os.path.isdir(subdir_path) and os.path.exists(os.path.join(subdir_path, "encoder.onnx")):
                        # 更新配置
                        new_path = os.path.join(sherpa_model_dir, subdir)
                        self.config_manager.set("models.sherpa", new_path)
                        print(f"已更新 Sherpa 模型路径: {new_path}")
                        break
    
        except Exception as e:
            print(f"更新模型路径失败: {e}")

class TranslationManager:
    """翻译管理类"""
    def __init__(self):
        self.engine = None
        self.engine_type = None
        self.config_manager = None
        self.setup()
    
    def setup(self):
        """初始化翻译管理器"""
        try:
            # 获取配置管理器
            self.config_manager = ConfigManager(
                os.path.join("config", "config.json"),
                os.path.join("config", "default_config.json")
            )
            
            # 获取引擎类型
            engine_type = self.config_manager.get("translation.engine", "argos")
            
            # 获取翻译方向
            direction = self.config_manager.get("translation.direction", "en-zh")
            
            # 初始化引擎
            self.initialize_engine(engine_type, direction)
            
            return True
            
        except Exception as e:
            print(f"翻译管理器初始化失败: {e}")
            return False
    
    def initialize_engine(self, engine_type="argos", direction="en-zh"):
        """初始化翻译引擎"""
        try:
            # 获取模型目录
            bergamot_model_dir = self.config_manager.get("models.bergamot", "bergamot_models")
            
            # 检查 OpusMT 模型路径
            if engine_type == "opusmt":
                # 尝试查找 OpusMT 模型
                base_dir = os.path.dirname(os.path.abspath(__file__))
                opus_model_path = os.path.join(base_dir, "models", "opus-mt", "en-zh")
                
                # 检查模型文件是否存在
                if not os.path.exists(opus_model_path):
                    # 尝试查找其他可能的位置
                    alt_paths = [
                        "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\models\\opus-mt\\en-zh"
                    ]
                    
                    for path in alt_paths:
                        if os.path.exists(path):
                            opus_model_path = path
                            break
                
                # 初始化 OpusMT 引擎
                self.engine = OpusMTTranslator(model_dir=opus_model_path)
                self.engine_type = "opusmt"
            
            # 初始化 Argos 引擎
            elif engine_type == "argos" and ARGOS_AVAILABLE:
                self.engine = ArgosTranslator()
                self.engine_type = "argos"
            
            # 如果都不可用，使用空翻译器
            else:
                self.engine = DummyTranslator()
                self.engine_type = "dummy"
            
            # 更新配置
            self.config_manager.set("translation.engine", self.engine_type)
            self.config_manager.set("translation.direction", direction)
            
            print(f"翻译管理器初始化成功，当前引擎: {self.engine_type}")
            return True
            
        except Exception as e:
            print(f"翻译管理器初始化失败: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def translate(self, text):
        """翻译文本"""
        if not self.engine:
            return text
        
        try:
            return self.engine.translate(text)
        except Exception as e:
            print(f"翻译错误: {e}")
            return text

class ASRManager:
    """ASR 管理类"""
    def __init__(self):
        self.engine = None
        self.engine_type = None
        self.config_manager = None
        self.setup()
    
    def setup(self):
        """初始化 ASR 管理器"""
        try:
            # 获取配置管理器
            self.config_manager = ConfigManager(
                os.path.join("config", "config.json"),
                os.path.join("config", "default_config.json")
            )
            
            # 获取引擎类型
            engine_type = self.config_manager.get("asr.engine", "vosk")
            
            # 初始化引擎
            self.initialize_engine(engine_type)
            
            return True
            
        except Exception as e:
            print(f"ASR 管理器初始化失败: {e}")
            return False
    
    def initialize_engine(self, engine_type="vosk"):
        """初始化 ASR 引擎"""
        try:
            # 获取模型目录
            vosk_model_dir = self.config_manager.get("models.vosk", "model")
            sherpa_model_dir = self.config_manager.get("models.sherpa", "sherpa_models")
            
            # 初始化引擎
            if engine_type == "vosk":
                self.engine = VoskASR(model_dir=vosk_model_dir)
                self.engine_type = "vosk"
            elif engine_type == "sherpa" and SHERPA_AVAILABLE:
                self.engine = SherpaOnnxASR(model_dir=sherpa_model_dir)
                self.engine_type = "sherpa"
            else:
                print(f"不支持的 ASR 引擎类型: {engine_type}")
                return False
            
            # 更新配置
            self.config_manager.set("asr.engine", engine_type)
            
            print(f"ASR 管理器初始化成功，当前引擎: {engine_type}")
            return True
            
        except Exception as e:
            print(f"ASR 管理器初始化失败: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def transcribe(self, audio_data):
        """转录音频数据"""
        if not self.engine:
            return ""
        
        try:
            return self.engine.transcribe(audio_data)
        except Exception as e:
            print(f"转录错误: {e}")
            return ""
    
    def reset(self):
        """重置当前引擎"""
        if self.engine:
            self.engine.reset()
    
    def get_final_result(self):
        """获取最终结果"""
        if self.engine:
            return self.engine.get_final_result()
        return ""
    
    def get_current_engine_type(self):
        """获取当前引擎类型"""
        return self.engine_type
    
    def get_available_engines(self):
        """获取可用的引擎列表"""
        return [name for name, engine in self.engines.items() if engine is not None]

class ASRFactory:
    """ASR 引擎工厂类"""
    @staticmethod
    def create_asr(engine_type="vosk"):
        """创建 ASR 引擎实例"""
        if engine_type == "vosk":
            return VoskASR()
        elif engine_type == "sherpa":
            return SherpaOnnxASR()
        else:
            raise ValueError(f"不支持的 ASR 引擎类型: {engine_type}")

class TranslatorFactory:
    """翻译器工厂类"""
    @staticmethod
    def create_translator(translator_type="argos"):
        """创建翻译器实例"""
        if translator_type == "argos":
            return ArgosTranslator()
        else:
            raise ValueError(f"不支持的翻译器类型: {translator_type}")

class AudioDeviceManager:
    """音频设备管理类"""
    def __init__(self):
        self.system_speakers = []
        self.microphones = []
        self.current_device = None
        self.setup()
    
    def setup(self):
        """初始化音频设备"""
        try:
            # 获取系统扬声器
            self.system_speakers = sc.all_speakers()
            
            # 获取麦克风
            self.microphones = sc.all_microphones()
            
            # 设置默认设备
            self.current_device = self.get_default_device()
            
            print("音频设备管理器初始化成功")
            print(f"可用扬声器: {len(self.system_speakers)}")
            print(f"可用麦克风: {len(self.microphones)}")
            return True
            
        except Exception as e:
            print(f"音频设备管理器初始化失败: {e}")
            return False
    
    def get_default_device(self):
        """获取默认设备"""
        try:
            # 优先使用系统默认扬声器
            default_speaker = sc.default_speaker()
            if default_speaker:
                return default_speaker
            
            # 如果没有默认扬声器，使用第一个可用扬声器
            if self.system_speakers:
                return self.system_speakers[0]
            
            return None
            
        except Exception as e:
            print(f"获取默认设备错误: {e}")
            return None
    
    def get_device_by_name(self, name, device_type="speaker"):
        """根据名称获取设备"""
        try:
            devices = self.system_speakers if device_type == "speaker" else self.microphones
            for device in devices:
                if name.lower() in device.name.lower():
                    return device
            return None
            
        except Exception as e:
            print(f"获取设备错误: {e}")
            return None
    
    def switch_device(self, device):
        """切换音频设备"""
        try:
            if isinstance(device, str):
                device = self.get_device_by_name(device)
            
            if device:
                self.current_device = device
                print(f"已切换到设备: {device.name}")
                return True
            
            print("切换设备失败: 设备不存在")
            return False
            
        except Exception as e:
            print(f"切换设备错误: {e}")
            return False
    
    def get_device_list(self, device_type="speaker"):
        """获取设备列表"""
        try:
            if device_type == "speaker":
                return [(device.name, device) for device in self.system_speakers]
            else:
                return [(device.name, device) for device in self.microphones]
                
        except Exception as e:
            print(f"获取设备列表错误: {e}")
            return []
    
    def refresh_devices(self):
        """刷新设备列表"""
        try:
            # 保存当前设备名称
            current_name = self.current_device.name if self.current_device else None
            
            # 重新获取设备列表
            self.system_speakers = sc.all_speakers()
            self.microphones = sc.all_microphones()
            
            # 尝试恢复之前的设备
            if current_name:
                device = self.get_device_by_name(current_name)
                if device:
                    self.current_device = device
                else:
                    self.current_device = self.get_default_device()
            
            return True
            
        except Exception as e:
            print(f"刷新设备列表错误: {e}")
            return False

class ConfigManager:
    """配置管理类"""
    def __init__(self, config_file, default_config_file):
        self.config_file = config_file
        self.default_config_file = default_config_file
        self.config = {}
        self.load_config()
    
    def load_config(self):
        """加载配置"""
        try:
            # 确保配置目录存在
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            # 加载默认配置
            default_config = {}
            if os.path.exists(self.default_config_file):
                with open(self.default_config_file, 'r', encoding='utf-8') as f:
                    default_config = json.load(f)
            else:
                # 创建默认配置文件
                default_config = {
                    "window": {
                        "geometry": [100, 100, 800, 400],
                        "opacity": 0.8,
                        "font_size": 14,
                        "theme": "dark"
                    },
                    "audio": {
                        "is_system_audio": True,
                        "sample_rate": 16000,
                        "chunk_size": 4000,
                        "channels": 1
                    },
                    "asr": {
                        "engine": "vosk",
                        "enable_speaker_id": False
                    },
                    "translation": {
                        "engine": "argos",
                        "enable": False,
                        "direction": "en-zh"
                    },
                    "models": {
                        "vosk": "model",
                        "sherpa": "sherpa_models",
                        "speaker": "speaker_model",
                        "bergamot": "bergamot_models"
                    }
                }
                # 保存默认配置
                os.makedirs(os.path.dirname(self.default_config_file), exist_ok=True)
                with open(self.default_config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=4, ensure_ascii=False)
            
            # 加载用户配置
            user_config = {}
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
            
            # 合并配置
            self.config = default_config.copy()
            self.deep_update(self.config, user_config)
            
            print("配置系统初始化成功")
            return True
            
        except Exception as e:
            print(f"加载配置错误: {e}")
            # 确保至少有一个空配置
            self.config = {
                "window": {
                    "geometry": [100, 100, 800, 400],
                    "opacity": 0.8,
                    "font_size": 14,
                    "theme": "dark"
                },
                "audio": {
                    "is_system_audio": True,
                    "sample_rate": 16000,
                    "chunk_size": 4000,
                    "channels": 1
                },
                "asr": {
                    "engine": "vosk",
                    "enable_speaker_id": False
                },
                "translation": {
                    "engine": "argos",
                    "enable": False,
                    "direction": "en-zh"
                },
                "models": {
                    "vosk": "model",
                    "sherpa": "sherpa_models",
                    "speaker": "speaker_model",
                    "bergamot": "bergamot_models"
                }
            }
            print(f"配置系统初始化失败: {e}")
            return False
    
    def deep_update(self, d, u):
        """深度更新字典"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self.deep_update(d[k], v)
            else:
                d[k] = v
    
    def save_config(self):
        """保存配置"""
        try:
            # 确保配置目录存在
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            # 保存配置
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"保存配置错误: {e}")
            return False
    
    def get(self, key, default=None):
        """获取配置项"""
        try:
            # 支持点号分隔的键
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key, value):
        """设置配置项"""
        try:
            # 支持点号分隔的键
            keys = key.split('.')
            config = self.config
            
            # 遍历到最后一个键
            for i, k in enumerate(keys[:-1]):
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # 设置值
            config[keys[-1]] = value
            
            # 保存配置
            self.save_config()
            
            return True
            
        except Exception as e:
            print(f"设置配置错误: {e}")
            return False

class ThemeManager:
    """主题管理类"""
    def __init__(self):
        self.themes = {
            "dark": {
                "window_bg": "#2B2B2B",
                "text_color": "#E0E0E0",
                "button_bg": "#3C3F41",
                "button_text": "#BBBBBB",
                "button_hover": "#4C5052",
                "border_color": "#323232",
                "highlight_color": "#2979FF",
                "error_color": "#FF5252",
                "success_color": "#4CAF50",
                "menu_bg": "#3C3F41",
                "menu_text": "#BBBBBB",
                "menu_hover": "#4C5052",
                "status_bar_bg": "#3C3F41",
                "status_bar_text": "#BBBBBB"
            },
            "light": {
                "window_bg": "#FFFFFF",
                "text_color": "#000000",
                "button_bg": "#F5F5F5",
                "button_text": "#000000",
                "button_hover": "#E0E0E0",
                "border_color": "#CCCCCC",
                "highlight_color": "#2979FF",
                "error_color": "#F44336",
                "success_color": "#4CAF50",
                "menu_bg": "#F5F5F5",
                "menu_text": "#000000",
                "menu_hover": "#E0E0E0",
                "status_bar_bg": "#F5F5F5",
                "status_bar_text": "#000000"
            }
        }
        self.current_theme = "dark"
    
    def get_stylesheet(self, theme_name=None):
        """获取主题样式表"""
        if not theme_name:
            theme_name = self.current_theme
        
        if theme_name not in self.themes:
            theme_name = "dark"
        
        theme = self.themes[theme_name]
        
        return f"""
            QMainWindow {{
                background-color: {theme['window_bg']};
                color: {theme['text_color']};
            }}
            
            QWidget {{
                background-color: {theme['window_bg']};
                color: {theme['text_color']};
            }}
            
            QPushButton {{
                background-color: {theme['button_bg']};
                color: {theme['button_text']};
                border: 1px solid {theme['border_color']};
                padding: 5px;
                border-radius: 3px;
            }}
            
            QPushButton:hover {{
                background-color: {theme['button_hover']};
            }}
            
            QLabel {{
                color: {theme['text_color']};
            }}
            
            QMenuBar {{
                background-color: {theme['menu_bg']};
                color: {theme['menu_text']};
            }}
            
            QMenuBar::item:selected {{
                background-color: {theme['menu_hover']};
            }}
            
            QMenu {{
                background-color: {theme['menu_bg']};
                color: {theme['menu_text']};
                border: 1px solid {theme['border_color']};
            }}
            
            QMenu::item:selected {{
                background-color: {theme['menu_hover']};
            }}
            
            QStatusBar {{
                background-color: {theme['status_bar_bg']};
                color: {theme['status_bar_text']};
            }}
            
            QScrollArea {{
                border: 1px solid {theme['border_color']};
            }}
            
            QProgressBar {{
                border: 1px solid {theme['border_color']};
                border-radius: 3px;
                text-align: center;
            }}
            
            QProgressBar::chunk {{
                background-color: {theme['highlight_color']};
            }}
        """
    
    def switch_theme(self, theme_name):
        """切换主题"""
        if theme_name in self.themes:
            self.current_theme = theme_name
            return self.get_stylesheet()
        return None
    
    def get_color(self, color_name):
        """获取主题颜色"""
        return self.themes[self.current_theme].get(color_name)
    
    def add_theme(self, name, theme_data):
        """添加自定义主题"""
        required_colors = set(self.themes["dark"].keys())
        if set(theme_data.keys()) == required_colors:
            self.themes[name] = theme_data
            return True
        return False

class AudioProcessor:
    """音频处理类"""
    def __init__(self, sample_rate=16000, channels=1, dtype=np.float32):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.resampler = None
        self.setup()
    
    def setup(self):
        """初始化音频处理器"""
        try:
            # 初始化重采样器
            import samplerate
            self.resampler = samplerate.Resampler('sinc_best')
            print("音频处理器初始化成功")
            return True
        except ImportError:
            print("警告: samplerate 库未安装，将使用简单重采样")
            return False
        except Exception as e:
            print(f"音频处理器初始化错误: {e}")
            return False
    
    def process_audio(self, data, original_rate=None):
        """处理音频数据"""
        try:
            # 转换数据类型
            if data.dtype != self.dtype:
                data = data.astype(self.dtype)
            
            # 重采样
            if original_rate and original_rate != self.sample_rate:
                if self.resampler:
                    ratio = self.sample_rate / original_rate
                    data = self.resampler.process(data, ratio)
                else:
                    data = self.simple_resample(data, original_rate)
            
            # 转换声道
            if len(data.shape) > 1 and data.shape[1] > self.channels:
                data = np.mean(data, axis=1)
            
            # 标准化音量
            data = self.normalize_volume(data)
            
            return data
            
        except Exception as e:
            print(f"音频处理错误: {e}")
            return np.zeros(0, dtype=self.dtype)
    
    def normalize_volume(self, data):
        """标准化音量"""
        try:
            max_val = np.max(np.abs(data))
            if max_val > 0:
                return data / max_val * 0.9
            return data
        except Exception as e:
            print(f"音量标准化错误: {e}")
            return data
    
    def simple_resample(self, data, original_rate):
        """简单重采样"""
        try:
            duration = len(data) / original_rate
            new_length = int(duration * self.sample_rate)
            indices = np.linspace(0, len(data)-1, new_length)
            return np.interp(indices, np.arange(len(data)), data)
        except Exception as e:
            print(f"重采样错误: {e}")
            return data

class LogManager:
    """日志管理类"""
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.log_dir = os.path.join(base_dir, "logs")
        self.setup()
    
    def setup(self):
        """初始化日志系统"""
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            
            # 设置日志文件
            log_file = os.path.join(
                self.log_dir, 
                f"transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            
            # 配置日志记录器
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s [%(levelname)s] %(message)s',
                handlers=[
                    logging.FileHandler(log_file, encoding='utf-8'),
                    logging.StreamHandler()
                ]
            )
            
            print(f"日志系统初始化成功，日志文件: {log_file}")
            
        except Exception as e:
            print(f"日志系统初始化错误: {e}")
    
    def log_transcription(self, text, speaker=None, timestamp=None):
        """记录转录文本"""
        try:
            if not timestamp:
                timestamp = datetime.datetime.now()
            
            if speaker:
                logging.info(f"[{timestamp}] [{speaker}] {text}")
            else:
                logging.info(f"[{timestamp}] {text}")
                
        except Exception as e:
            print(f"记录转录文本错误: {e}")
    
    def log_translation(self, original, translated, timestamp=None):
        """记录翻译文本"""
        try:
            if not timestamp:
                timestamp = datetime.datetime.now()
            
            logging.info(f"[{timestamp}] 原文: {original}")
            logging.info(f"[{timestamp}] 译文: {translated}")
            
        except Exception as e:
            print(f"记录翻译文本错误: {e}")
    
    def log_error(self, context, error):
        """记录错误信息"""
        try:
            logging.error(f"[{context}] {error}")
            if isinstance(error, Exception):
                logging.error(f"详细错误: {traceback.format_exc()}")
        except Exception as e:
            print(f"记录错误信息失败: {e}")

class ModelManager:
    """模型管理类"""
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.model_dirs = {
            "vosk": os.path.join(base_dir, "model"),
            "sherpa": os.path.join(base_dir, "sherpa_models"),
            "speaker": os.path.join(base_dir, "speaker_model"),
            "bergamot": os.path.join(base_dir, "bergamot_models")
        }
        self.setup()
    
    def setup(self):
        """初始化模型目录"""
        try:
            for name, path in self.model_dirs.items():
                os.makedirs(path, exist_ok=True)
                print(f"模型目录 {name}: {path}")
        except Exception as e:
            print(f"创建模型目录错误: {e}")
    
    def check_model(self, model_type):
        """检查模型文件"""
        try:
            model_dir = self.model_dirs.get(model_type)
            if not model_dir:
                return False
            
            if model_type == "vosk":
                return os.path.exists(os.path.join(model_dir, "conf", "mfcc.conf"))
            elif model_type == "sherpa":
                required_files = ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"]
                return all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)
            elif model_type == "speaker":
                return os.path.exists(os.path.join(model_dir, "final.raw"))
            elif model_type == "bergamot":
                return os.path.exists(os.path.join(model_dir, "en-zh", "model.bin"))
            
            return False
            
        except Exception as e:
            print(f"检查模型错误: {e}")
            return False
    
    def download_model(self, model_type):
        """下载模型文件"""
        try:
            # 这里应该实现具体的模型下载逻辑
            print(f"开始下载 {model_type} 模型...")
            # TODO: 实现模型下载
            print(f"模型下载完成: {model_type}")
            
        except Exception as e:
            print(f"下载模型错误: {e}")
            return False

class PerformanceMonitor:
    """性能监控类"""
    def __init__(self):
        self.start_time = None
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'latency': [],
            'transcription_speed': []
        }
        self.setup()
    
    def setup(self):
        """初始化监控"""
        try:
            import psutil
            self.process = psutil.Process()
            print("性能监控初始化成功")
        except Exception as e:
            print(f"性能监控初始化错误: {e}")
    
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        self.metrics = {k: [] for k in self.metrics}
    
    def update_metrics(self, audio_length=0):
        """更新性能指标"""
        try:
            # CPU 使用率
            cpu_percent = self.process.cpu_percent()
            self.metrics['cpu_usage'].append(cpu_percent)
            
            # 内存使用
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            self.metrics['memory_usage'].append(memory_mb)
            
            # 延迟
            if self.start_time:
                latency = time.time() - self.start_time
                self.metrics['latency'].append(latency)
            
            # 转录速度
            if audio_length > 0:
                speed = audio_length / latency if latency > 0 else 0
                self.metrics['transcription_speed'].append(speed)
            
        except Exception as e:
            print(f"更新性能指标错误: {e}")
    
    def get_average_metrics(self):
        """获取平均指标"""
        try:
            return {
                'avg_cpu': sum(self.metrics['cpu_usage']) / len(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
                'avg_memory': sum(self.metrics['memory_usage']) / len(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
                'avg_latency': sum(self.metrics['latency']) / len(self.metrics['latency']) if self.metrics['latency'] else 0,
                'avg_speed': sum(self.metrics['transcription_speed']) / len(self.metrics['transcription_speed']) if self.metrics['transcription_speed'] else 0
            }
        except Exception as e:
            print(f"获取平均指标错误: {e}")
            return {}

class ErrorHandler:
    """错误处理类"""
    def __init__(self, log_manager=None):
        self.log_manager = log_manager
        self.error_count = {}
        self.max_retries = 3
        self.error_callbacks = {}
    
    def handle_error(self, context, error, retry_func=None):
        """处理错误"""
        try:
            # 记录错误
            if self.log_manager:
                self.log_manager.log_error(context, error)
            
            # 更新错误计数
            self.error_count[context] = self.error_count.get(context, 0) + 1
            
            # 检查是否需要重试
            if retry_func and self.error_count[context] <= self.max_retries:
                print(f"尝试重试 {context} ({self.error_count[context]}/{self.max_retries})")
                return retry_func()
            
            # 调用错误回调
            if context in self.error_callbacks:
                self.error_callbacks[context](error)
            
            return None
            
        except Exception as e:
            print(f"错误处理失败: {e}")
            return None
    
    def register_callback(self, context, callback):
        """注册错误回调"""
        self.error_callbacks[context] = callback
    
    def reset_error_count(self, context=None):
        """重置错误计数"""
        if context:
            self.error_count[context] = 0
        else:
            self.error_count.clear()

class UpdateManager:
    """更新管理类"""
    def __init__(self, current_version="1.0.0"):
        self.current_version = current_version
        self.update_url = "https://api.github.com/repos/user/repo/releases/latest"
        self.download_url = None
    
    def check_updates(self):
        """检查更新"""
        try:
            import requests
            response = requests.get(self.update_url)
            if response.status_code == 200:
                latest = response.json()
                latest_version = latest['tag_name'].strip('v')
                if self.compare_versions(latest_version, self.current_version) > 0:
                    self.download_url = latest['assets'][0]['browser_download_url']
                    return True, latest_version, latest['body']
            return False, self.current_version, "已是最新版本"
        except Exception as e:
            print(f"检查更新错误: {e}")
            return False, self.current_version, str(e)
    
    def download_update(self, progress_callback=None):
        """下载更新"""
        if not self.download_url:
            return False
        
        try:
            import requests
            response = requests.get(self.download_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            downloaded = 0
            
            with open("update.zip", "wb") as f:
                for data in response.iter_content(block_size):
                    downloaded += len(data)
                    f.write(data)
                    if progress_callback:
                        progress = (downloaded / total_size) * 100
                        progress_callback(progress)
            
            return True
            
        except Exception as e:
            print(f"下载更新错误: {e}")
            return False
    
    def compare_versions(self, version1, version2):
        """比较版本号"""
        v1_parts = [int(x) for x in version1.split('.')]
        v2_parts = [int(x) for x in version2.split('.')]
        
        for i in range(max(len(v1_parts), len(v2_parts))):
            v1 = v1_parts[i] if i < len(v1_parts) else 0
            v2 = v2_parts[i] if i < len(v2_parts) else 0
            if v1 > v2:
                return 1
            elif v1 < v2:
                return -1
        return 0

class OpusMTTranslator:
    """OpusMT 翻译器"""
    def __init__(self, model_dir=None):
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self.setup()
    
    def setup(self):
        """初始化 OpusMT 翻译器"""
        try:
            if not self.model_dir:
                print("错误: 未指定 OpusMT 模型目录")
                return False
            
            # 检查模型目录
            if not os.path.exists(self.model_dir):
                print(f"错误: OpusMT 模型目录不存在: {self.model_dir}")
                return False
            
            # 导入必要的库
            try:
                from transformers import MarianTokenizer
                from optimum.onnxruntime import ORTModelForSeq2SeqLM
            except ImportError:
                print("错误: 未安装 transformers 或 optimum 库")
                return False
            
            # 加载分词器和模型
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_dir)
            self.model = ORTModelForSeq2SeqLM.from_pretrained(self.model_dir)
            
            print("OpusMT 翻译器初始化成功")
            return True
            
        except Exception as e:
            print(f"OpusMT 翻译器初始化失败: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def translate(self, text):
        """翻译文本"""
        try:
            if not text or not text.strip():
                return ""
            
            if not self.tokenizer or not self.model:
                return text
            
            # 准备输入
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            
            # 生成翻译
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=512,
                num_beams=4
            )
            
            # 解码翻译结果
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return translation
            
        except Exception as e:
            print(f"OpusMT 翻译错误: {e}")
            return text

class DummyTranslator:
    """空翻译器，用于在没有可用翻译引擎时使用"""
    def __init__(self):
        pass
    
    def setup(self):
        return True
    
    def translate(self, text):
        return text

def check_com_mode():
    """检查当前线程的 COM 模式"""
    try:
        import ctypes
        
        co_get_apartment_type = ctypes.windll.ole32.CoGetApartmentType
        co_get_apartment_type.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        co_get_apartment_type.restype = ctypes.HRESULT
        
        atype = ctypes.c_int()
        qtype = ctypes.c_int()
        hr = co_get_apartment_type(ctypes.byref(atype), ctypes.byref(qtype))
        
        if hr == 0:  # S_OK
            if atype.value == 1:  # APTTYPE_STA
                print("当前线程处于 STA 模式")
                return "STA"
            elif atype.value == 2:  # APTTYPE_MTA
                print("当前线程处于 MTA 模式，OleInitialize() 将失败")
                return "MTA"
            else:
                print(f"未知的公寓类型: {atype.value}")
                return "UNKNOWN"
        else:
            print(f"CoGetApartmentType() 失败，HRESULT: {hr}")
            return "ERROR"
            
    except Exception as e:
        print(f"检查 COM 模式失败: {e}")
        return "ERROR"

def init_com():
    """在新线程中初始化 COM"""
    try:
        import pythoncom
        pythoncom.CoInitialize()
        print("COM 在新线程中初始化成功")
        return True
    except Exception as e:
        print(f"COM 初始化失败: {e}")
        return False

def reset_com():
    """释放并重新初始化 COM"""
    try:
        import pythoncom
        
        # 1. 先尝试释放当前的 COM 初始化
        try:
            pythoncom.CoUninitialize()
            print("COM 已释放")
        except pythoncom.com_error:
            print("COM 未初始化或已释放")
        
        # 2. 等待一小段时间
        time.sleep(0.1)
        
        # 3. 重新初始化 COM
        pythoncom.CoInitialize()
        print("COM 重新初始化成功")
        return True
        
    except Exception as e:
        print(f"COM 重置失败: {e}")
        return False

class AudioCaptureThread(threading.Thread):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.running = False
        
    def run(self):
        try:
            # 在新线程中初始化 COM
            import pythoncom
            pythoncom.CoInitialize()
            
            # 音频捕获逻辑
            with sc.get_microphone(id=str(sc.default_speaker().id), include_loopback=True).recorder(samplerate=16000) as mic:
                while self.running:
                    data = mic.record(numframes=1600)
                    self.callback(data)
                    
        finally:
            # 清理 COM
            pythoncom.CoUninitialize()

class TranslationThread(threading.Thread):
    def __init__(self, translator, text, callback):
        super().__init__()
        self.translator = translator
        self.text = text
        self.callback = callback
        
    def run(self):
        try:
            # 在新线程中初始化 COM
            import pythoncom
            pythoncom.CoInitialize()
            
            # 执行翻译
            result = self.translator.translate(self.text)
            self.callback(result)
            
        finally:
            # 清理 COM
            pythoncom.CoUninitialize()

class ASRThread(threading.Thread):
    def __init__(self, recognizer, audio_queue, result_callback):
        super().__init__()
        self.recognizer = recognizer
        self.audio_queue = audio_queue
        self.callback = result_callback
        self.running = False
        
    def run(self):
        try:
            # 在新线程中初始化 COM
            import pythoncom
            pythoncom.CoInitialize()
            
            # ASR 处理逻辑
            while self.running:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    result = self.recognizer.transcribe(audio_data)
                    self.callback(result)
                    
        finally:
            # 清理 COM
            pythoncom.CoUninitialize()

class SafeSoundCard:
    """安全的 soundcard 包装类，避免 COM 初始化冲突"""
    
    @staticmethod
    def get_speaker_recorder(sample_rate=16000, channels=1, blocksize=1600):
        """获取系统音频录制器，避免 COM 冲突"""
        try:
            # 尝试使用替代方法获取音频
            import sounddevice as sd
            
            def callback(indata, frames, time, status):
                # 处理音频数据
                pass
                
            # 使用 sounddevice 替代 soundcard
            stream = sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                blocksize=blocksize,
                callback=callback
            )
            return stream
            
        except ImportError:
            print("警告: sounddevice 未安装，尝试使用 soundcard")
            
            # 如果必须使用 soundcard，则在子进程中运行
            import multiprocessing as mp
            
            def recorder_process(queue):
                try:
                    # 在新进程中初始化 COM
                    import soundcard as sc
                    with sc.get_microphone(id=str(sc.default_speaker().id), include_loopback=True).recorder(samplerate=sample_rate) as mic:
                        while True:
                            data = mic.record(numframes=blocksize)
                            queue.put(data)
                except Exception as e:
                    print(f"录音进程错误: {e}")
                    
            # 创建队列和进程
            queue = mp.Queue()
            process = mp.Process(target=recorder_process, args=(queue,))
            process.start()
            
            # 返回队列供主程序使用
            return queue

if __name__ == "__main__":
    # 禁用 COM 自动初始化
    os.environ["PYTHONCOM_INITIALIZE"] = "0"
    
    # 使用多进程模式
    import multiprocessing as mp
    mp.freeze_support()  # Windows 支持
    
    # 启动应用
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = SubtitleWindow()
    window.show()
    sys.exit(app.exec_())