"""
修复后的 MainWindow 类中的 select_file 方法
"""

def select_file(self):
    """选择文件 - 非阻塞方式"""
    try:
        # 使用 QFileDialog 的静态方法，但不阻塞主线程
        dialog = QFileDialog(self)
        dialog.setWindowTitle("选择音频/视频文件")
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("音频文件 (*.wav *.mp3 *.ogg);;视频文件 (*.mp4 *.avi *.mkv);;所有文件 (*.*)")
        
        # 设置为非模态对话框
        dialog.setModal(False)
        
        # 连接文件选择信号
        dialog.fileSelected.connect(self._on_file_selected)
        
        # 显示对话框
        dialog.show()
        
    except Exception as e:
        logging.error(f"打开文件对话框错误: {e}")
        self.signals.error_occurred.emit(f"打开文件对话框错误: {e}")

def _on_file_selected(self, file_path):
    """
    文件选择回调
    
    Args:
        file_path: 选择的文件路径
    """
    if file_path:
        try:
            self.signals.status_updated.emit(f"已选择文件: {file_path}")
            
            # TODO: 在这里添加文件处理逻辑
            # 例如：self._process_audio_file(file_path)
            
        except Exception as e:
            logging.error(f"处理文件错误: {e}")
            self.signals.error_occurred.emit(f"处理文件错误: {e}")
