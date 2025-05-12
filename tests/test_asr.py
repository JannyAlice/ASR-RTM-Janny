import os
import sys
import json
import unittest
import logging
import numpy as np
from pathlib import Path
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication

# 忽略警告
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/test_asr.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config_manager import config_manager
from src.core.asr.model_manager import ASRModelManager
import soundcard as sc

class TestASRManager(unittest.TestCase):
    """测试 ASR 模型管理器"""
    
    def setUp(self):
        """测试前的准备工作"""
        logger.info("-" * 50)
        logger.info("开始 ASR 测试")
        self.model_manager = ASRModelManager()
        self.default_model = config_manager.get_default_model()
        logger.info(f"使用默认模型: {self.default_model}")
    
    def test_initialization(self):
        """测试初始化"""
        logger.info("测试 ASR 初始化")
        self.assertIsNotNone(self.model_manager)
        self.assertEqual(self.model_manager.model_type, self.default_model)
    
    def test_load_model(self):
        """测试模型加载"""
        logger.info("测试模型加载")
        # 加载默认模型
        success = self.model_manager.load_model(self.default_model)
        self.assertTrue(success)
        
        # 检查模型是否正确加载
        self.assertIsNotNone(self.model_manager.current_model)
        self.assertEqual(self.model_manager.current_model_type, self.default_model)
        
        # 验证模型路径
        model_path = self.model_manager.model_path
        logger.info(f"模型路径: {model_path}")
        self.assertTrue(os.path.exists(model_path))
        
        # 检查必需的模型文件
        required_files = [
            'encoder-epoch-99-avg-1-chunk-16-left-128.onnx',
            'decoder-epoch-99-avg-1-chunk-16-left-128.onnx',
            'joiner-epoch-99-avg-1-chunk-16-left-128.onnx',
            'tokens.txt'
        ]
        
        for file in required_files:
            file_path = os.path.join(model_path, file)
            self.assertTrue(os.path.exists(file_path), f"缺少模型文件: {file}")
    
    def test_create_recognizer(self):
        """测试创建识别器"""
        logger.info("测试创建识别器")
        # 先加载模型
        success = self.model_manager.load_model(self.default_model)
        self.assertTrue(success)
        
        # 创建识别器
        recognizer = self.model_manager.create_recognizer()
        self.assertIsNotNone(recognizer)
        logger.info(f"识别器类型: {type(recognizer)}")
    
    def test_audio_recognition(self):
        """测试音频识别"""
        logger.info("测试音频识别")
        try:
            # 先加载模型和创建识别器
            self.model_manager.load_model(self.default_model)
            recognizer = self.model_manager.create_recognizer()
            
            # 创建测试音频数据
            sample_rate = 16000
            duration = 1  # 1秒
            samples = np.zeros(sample_rate * duration, dtype=np.float32)
            
            # 创建音频流
            stream = recognizer.create_stream()
            logger.info("创建音频流成功")
            
            # 处理音频数据
            stream.accept_waveform(sample_rate, samples)
            logger.info("音频数据处理完成")
            
            # 正确获取识别结果
            is_endpoint = recognizer.is_endpoint(stream)
            text = recognizer.get_result(stream)
            self.assertIsNotNone(text)
            logger.info(f"检测到终点: {is_endpoint}")
            logger.info(f"识别结果: {text}")
            
        except Exception as e:
            logger.error(f"音频识别失败: {str(e)}")
            self.fail(f"音频识别测试失败: {str(e)}")
    
    def test_system_audio_recognition(self):
        """测试系统音频识别"""
        logger.info("测试系统音频识别")
        try:
            # 导入 pythoncom 并初始化
            import pythoncom
            try:
                pythoncom.CoInitialize()
            except Exception as e:
                logger.warning(f"CoInitialize 失败: {e}")
    
            # 确保有 Qt 应用实例
            app = QApplication.instance()
            if not app:
                app = QApplication(sys.argv)
    
            # 加载模型和创建识别器
            self.model_manager.load_model(self.default_model)
            recognizer = self.model_manager.create_recognizer()
            
            # 音频参数
            sample_rate = 16000
            channels = 1
            buffer_size = 4000
            results = []
    
            # 创建持久流
            stream = recognizer.create_stream()
            logger.info("创建持久流成功")
            
            # 音频处理线程类
            class AudioProcessor(QObject):
                finished = pyqtSignal()
                error = pyqtSignal(str)
                
                def __init__(self):
                    super().__init__()
                    self.running = True
                    self.last_text = ""  # 用于去重
                
                def process(self):
                    try:
                        # 获取系统音频设备
                        speakers = sc.all_speakers()
                        target_speaker = next((s for s in speakers if "CABLE" in s.name), speakers[0])
                        logger.info(f"使用音频设备: {target_speaker.name}")
                        
                        with sc.get_microphone(id=target_speaker.id, include_loopback=True).recorder(
                            samplerate=sample_rate,
                            channels=channels,
                            blocksize=buffer_size
                        ) as recorder:
                            logger.info("开始捕获系统音频...")
                            logger.info("请播放音频文件...")
                            
                            while self.running:
                                # 录制并处理音频
                                data = recorder.record(numframes=buffer_size)
                                if len(data.shape) > 1:
                                    data = data.mean(axis=1)
                                
                                # 发送到识别器
                                audio_data = data.astype(np.float32)
                                stream.accept_waveform(sample_rate, audio_data)
                                
                                # 添加尾部填充
                                tail_padding = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
                                stream.accept_waveform(sample_rate, tail_padding)
                                
                                # 检查识别结果
                                while recognizer.is_ready(stream):
                                    recognizer.decode_stream(stream)
                                    
                                if recognizer.is_endpoint(stream):
                                    text = recognizer.get_result(stream)
                                    if text and text.strip() and text != self.last_text:  # 去重
                                        results.append(text)
                                        logger.info(f"实时识别结果: {text}")
                                        self.last_text = text
                                        
                    except Exception as e:
                        self.error.emit(str(e))
                    finally:
                        self.finished.emit()
    
            # 创建音频处理线程
            self.processor_thread = QThread()
            self.processor = AudioProcessor()
            self.processor.moveToThread(self.processor_thread)
            
            # 连接信号
            self.processor_thread.started.connect(self.processor.process)
            self.processor.finished.connect(self.processor_thread.quit)
            self.processor.finished.connect(self.processor.deleteLater)
            self.processor_thread.finished.connect(self.processor_thread.deleteLater)
            
            # 启动处理
            self.processor_thread.start()
            
            # 等待处理
            import time
            time.sleep(10)  # 录制10秒
            
            # 停止处理
            self.processor.running = False
            self.processor_thread.quit()
            self.processor_thread.wait()
            
            # 获取最终结果
            final_text = recognizer.get_result(stream)
            if final_text and final_text.strip() and final_text != self.processor.last_text:  # 去重
                results.append(final_text)
                
            # 输出结果统计
            logger.info("\n----- 识别结果统计 -----")
            logger.info(f"结果数量: {len(results)}")
            if results:
                logger.info("\n识别结果:")
                for i, text in enumerate(results, 1):
                    logger.info(f"{i}. {text}")
            else:
                logger.warning("未检测到有效音频输入")
            logger.info("-----------------------")
            
            # 验证结果
            self.assertTrue(len(results) > 0, "没有获取到任何识别结果")
                
        except Exception as e:
            logger.error(f"系统音频识别失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.fail(f"系统音频识别测试失败: {e}")
    
    def tearDown(self):
        """测试结束清理"""
        logger.info("ASR 测试结束")
        logger.info("-" * 50)

if __name__ == '__main__':
    unittest.main(verbosity=2)