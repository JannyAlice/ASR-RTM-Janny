"""
修复 COM 初始化问题的 select_file 方法
"""

def select_file(self):
    """选择文件 - 确保 COM 初始化正确"""
    try:
        # 确保 COM 已初始化
        if not com_handler.is_initialized():
            com_handler.initialize_com()
            print("COM 已为文件对话框初始化")
        
        # 使用标准文件对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频/视频文件",
            "",
            "音频文件 (*.wav *.mp3 *.ogg);;视频文件 (*.mp4 *.avi *.mkv);;所有文件 (*.*)")
        
        if file_path:
            self.signals.status_updated.emit(f"已选择文件: {file_path}")
            
            # TODO: 在这里添加文件处理逻辑
            # 例如：self._process_audio_file(file_path)
            
    except Exception as e:
        logging.error(f"文件选择错误: {e}")
        self.signals.error_occurred.emit(f"文件选择错误: {e}")
        
        # 尝试重新初始化 COM
        try:
            com_handler.uninitialize_com()
            com_handler.initialize_com()
            print("COM 已重新初始化")
        except Exception as com_error:
            logging.error(f"COM 重新初始化错误: {com_error}")
            self.signals.error_occurred.emit(f"COM 重新初始化错误: {com_error}")
