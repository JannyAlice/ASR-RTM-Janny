"""
修复后的 select_file 方法
"""

def select_file(self):
    """选择文件 - 非阻塞式"""
    try:
        # 确保 COM 已初始化
        if hasattr(com_handler, 'is_initialized') and not com_handler.is_initialized():
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

def _on_file_selected(self, file_path):
    """
    文件选择回调
    
    Args:
        file_path: 选择的文件路径
    """
    if not file_path:
        return
        
    try:
        self.signals.status_updated.emit(f"已选择文件: {file_path}")
        
        # 显示一个简单的消息框，表示功能尚未实现
        QMessageBox.information(
            self,
            "文件选择",
            f"已选择文件: {os.path.basename(file_path)}\n\n文件处理功能尚未实现。"
        )
        
    except Exception as e:
        logging.error(f"处理文件错误: {e}")
        self.signals.error_occurred.emit(f"处理文件错误: {e}")
