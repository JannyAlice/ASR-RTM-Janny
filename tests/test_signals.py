#!/usr/bin/env python3
"""
信号测试模块
测试所有信号的连接和发射
"""
import sys
import os
import time
import unittest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.signals import TranscriptionSignals

class SignalReceiver:
    """信号接收器类，用于测试信号"""
    
    def __init__(self):
        """初始化信号接收器"""
        self.received_signals = {
            'new_text': [],
            'partial_result': [],
            'status_updated': [],
            'progress_updated': [],
            'error_occurred': [],
            'transcription_started': False,
            'transcription_finished': False,
            'transcription_paused': False,
            'transcription_resumed': False
        }
    
    def on_new_text(self, text):
        """接收新文本信号"""
        print(f"接收到新文本: {text}")
        self.received_signals['new_text'].append(text)
    
    def on_partial_result(self, text):
        """接收部分结果信号"""
        print(f"接收到部分结果: {text}")
        self.received_signals['partial_result'].append(text)
    
    def on_status_updated(self, status):
        """接收状态更新信号"""
        print(f"接收到状态更新: {status}")
        self.received_signals['status_updated'].append(status)
    
    def on_progress_updated(self, progress, text):
        """接收进度更新信号"""
        print(f"接收到进度更新: {progress}%, {text}")
        self.received_signals['progress_updated'].append((progress, text))
    
    def on_error_occurred(self, error):
        """接收错误信号"""
        print(f"接收到错误: {error}")
        self.received_signals['error_occurred'].append(error)
    
    def on_transcription_started(self):
        """接收转录开始信号"""
        print("接收到转录开始信号")
        self.received_signals['transcription_started'] = True
    
    def on_transcription_finished(self):
        """接收转录完成信号"""
        print("接收到转录完成信号")
        self.received_signals['transcription_finished'] = True
    
    def on_transcription_paused(self):
        """接收转录暂停信号"""
        print("接收到转录暂停信号")
        self.received_signals['transcription_paused'] = True
    
    def on_transcription_resumed(self):
        """接收转录恢复信号"""
        print("接收到转录恢复信号")
        self.received_signals['transcription_resumed'] = True

class TestSignals(unittest.TestCase):
    """信号测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.signals = TranscriptionSignals()
        self.receiver = SignalReceiver()
        
        # 连接信号
        self.signals.new_text.connect(self.receiver.on_new_text)
        self.signals.partial_result.connect(self.receiver.on_partial_result)
        self.signals.status_updated.connect(self.receiver.on_status_updated)
        self.signals.progress_updated.connect(self.receiver.on_progress_updated)
        self.signals.error_occurred.connect(self.receiver.on_error_occurred)
        self.signals.transcription_started.connect(self.receiver.on_transcription_started)
        self.signals.transcription_finished.connect(self.receiver.on_transcription_finished)
        self.signals.transcription_paused.connect(self.receiver.on_transcription_paused)
        self.signals.transcription_resumed.connect(self.receiver.on_transcription_resumed)
    
    def test_text_signals(self):
        """测试文本相关信号"""
        # 发送新文本信号
        self.signals.new_text.emit("测试文本1")
        self.signals.new_text.emit("测试文本2")
        
        # 发送部分结果信号
        self.signals.partial_result.emit("部分结果1")
        self.signals.partial_result.emit("部分结果2")
        
        # 处理事件
        QTimer.singleShot(100, self.app.quit)
        self.app.exec_()
        
        # 验证结果
        self.assertEqual(len(self.receiver.received_signals['new_text']), 2)
        self.assertEqual(self.receiver.received_signals['new_text'][0], "测试文本1")
        self.assertEqual(self.receiver.received_signals['new_text'][1], "测试文本2")
        
        self.assertEqual(len(self.receiver.received_signals['partial_result']), 2)
        self.assertEqual(self.receiver.received_signals['partial_result'][0], "部分结果1")
        self.assertEqual(self.receiver.received_signals['partial_result'][1], "部分结果2")
    
    def test_status_signals(self):
        """测试状态相关信号"""
        # 发送状态更新信号
        self.signals.status_updated.emit("状态1")
        self.signals.status_updated.emit("状态2")
        
        # 发送进度更新信号
        self.signals.progress_updated.emit(50, "进度50%")
        self.signals.progress_updated.emit(100, "进度100%")
        
        # 处理事件
        QTimer.singleShot(100, self.app.quit)
        self.app.exec_()
        
        # 验证结果
        self.assertEqual(len(self.receiver.received_signals['status_updated']), 2)
        self.assertEqual(self.receiver.received_signals['status_updated'][0], "状态1")
        self.assertEqual(self.receiver.received_signals['status_updated'][1], "状态2")
        
        self.assertEqual(len(self.receiver.received_signals['progress_updated']), 2)
        self.assertEqual(self.receiver.received_signals['progress_updated'][0], (50, "进度50%"))
        self.assertEqual(self.receiver.received_signals['progress_updated'][1], (100, "进度100%"))
    
    def test_error_signals(self):
        """测试错误信号"""
        # 发送错误信号
        self.signals.error_occurred.emit("错误1")
        self.signals.error_occurred.emit("错误2")
        
        # 处理事件
        QTimer.singleShot(100, self.app.quit)
        self.app.exec_()
        
        # 验证结果
        self.assertEqual(len(self.receiver.received_signals['error_occurred']), 2)
        self.assertEqual(self.receiver.received_signals['error_occurred'][0], "错误1")
        self.assertEqual(self.receiver.received_signals['error_occurred'][1], "错误2")
    
    def test_lifecycle_signals(self):
        """测试生命周期信号"""
        # 发送生命周期信号
        self.signals.transcription_started.emit()
        self.signals.transcription_paused.emit()
        self.signals.transcription_resumed.emit()
        self.signals.transcription_finished.emit()
        
        # 处理事件
        QTimer.singleShot(100, self.app.quit)
        self.app.exec_()
        
        # 验证结果
        self.assertTrue(self.receiver.received_signals['transcription_started'])
        self.assertTrue(self.receiver.received_signals['transcription_paused'])
        self.assertTrue(self.receiver.received_signals['transcription_resumed'])
        self.assertTrue(self.receiver.received_signals['transcription_finished'])

if __name__ == "__main__":
    unittest.main()
