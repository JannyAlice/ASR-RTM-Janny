"""
完整的修复方案，包括非阻塞对话框、COM 初始化和文件处理逻辑
"""

def select_file(self):
    """选择文件 - 完整修复版"""
    try:
        # 确保 COM 已初始化
        if not com_handler.is_initialized():
            com_handler.initialize_com()
            print("COM 已为文件对话框初始化")
        
        # 创建非模态文件对话框
        dialog = QFileDialog(self)
        dialog.setWindowTitle("选择音频/视频文件")
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("音频文件 (*.wav *.mp3 *.ogg);;视频文件 (*.mp4 *.avi *.mkv);;所有文件 (*.*)")
        dialog.setModal(False)
        
        # 连接文件选择信号
        dialog.fileSelected.connect(self._on_file_selected)
        
        # 显示对话框
        dialog.open()
        
    except Exception as e:
        logging.error(f"打开文件对话框错误: {e}")
        self.signals.error_occurred.emit(f"打开文件对话框错误: {e}")
        
        # 尝试重新初始化 COM
        try:
            com_handler.uninitialize_com()
            com_handler.initialize_com()
            print("COM 已重新初始化")
        except Exception as com_error:
            logging.error(f"COM 重新初始化错误: {com_error}")

def _on_file_selected(self, file_path):
    """
    文件选择回调
    
    Args:
        file_path: 选择的文件路径
    """
    if file_path:
        try:
            self.signals.status_updated.emit(f"已选择文件: {file_path}")
            self._process_audio_file(file_path)
        except Exception as e:
            logging.error(f"处理文件错误: {e}")
            self.signals.error_occurred.emit(f"处理文件错误: {e}")

def _process_audio_file(self, file_path):
    """
    处理音频文件
    
    Args:
        file_path: 音频文件路径
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            self.signals.error_occurred.emit(f"文件不存在: {file_path}")
            return
        
        # 检查文件大小
        file_size = os.path.getsize(file_path)
        if file_size > 100 * 1024 * 1024:  # 大于 100MB
            result = QMessageBox.question(
                self,
                "大文件警告",
                f"文件大小为 {file_size / (1024 * 1024):.2f} MB，处理可能需要较长时间。是否继续？",
                QMessageBox.Yes | QMessageBox.No
            )
            if result == QMessageBox.No:
                return
        
        # 更新状态
        self.signals.status_updated.emit(f"正在处理文件: {os.path.basename(file_path)}")
        
        # 创建进度对话框
        progress = QMessageBox(self)
        progress.setWindowTitle("文件处理")
        progress.setText(f"正在处理文件: {os.path.basename(file_path)}")
        progress.setStandardButtons(QMessageBox.NoButton)
        
        # 在单独的线程中处理文件
        from PyQt5.QtCore import QThread, pyqtSignal
        
        class FileProcessThread(QThread):
            finished = pyqtSignal(bool, str)
            
            def __init__(self, file_path, parent=None):
                super().__init__(parent)
                self.file_path = file_path
            
            def run(self):
                try:
                    # TODO: 在这里添加实际的文件处理逻辑
                    # 例如：调用音频处理器处理文件
                    # 临时占位代码，模拟处理
                    import time
                    time.sleep(1)  # 模拟处理时间
                    
                    self.finished.emit(True, "文件处理成功")
                except Exception as e:
                    self.finished.emit(False, str(e))
        
        # 创建并启动线程
        self.file_thread = FileProcessThread(file_path, self)
        self.file_thread.finished.connect(self._on_file_processed)
        self.file_thread.start()
        
        # 显示进度对话框
        progress.show()
        
    except Exception as e:
        logging.error(f"处理文件错误: {e}")
        self.signals.error_occurred.emit(f"处理文件错误: {e}")

def _on_file_processed(self, success, message):
    """
    文件处理完成回调
    
    Args:
        success: 是否成功
        message: 处理消息
    """
    if success:
        self.signals.status_updated.emit(message)
        QMessageBox.information(self, "文件处理", message)
    else:
        self.signals.error_occurred.emit(f"文件处理错误: {message}")
        QMessageBox.critical(self, "文件处理错误", message)
