"""
主窗口方法模块
包含主窗口类的附加方法，用于支持新的菜单结构
"""
import os
import traceback
from PyQt5.QtCore import QTimer

from src.utils.logger import get_logger

logger = get_logger(__name__)

def set_language_mode(self, mode):
    """
    设置语言模式
    
    Args:
        mode: 语言模式，可选值为"en"、"zh"、"auto"
    """
    try:
        logger.info(f"设置语言模式: {mode}")
        
        # 更新配置
        self.config_manager.set_config(mode, "recognition", "language_mode")
        self.config_manager.save_config("main")
        
        # 更新状态栏
        self.signals.status_updated.emit(f"已设置语言模式: {self._get_language_mode_display(mode)}")
        
        # 在字幕窗口显示语言设置信息
        language_info = f"已设置识别语言: {self._get_language_mode_display(mode)}"
        
        # 只有在没有进行转录时才更新字幕窗口
        if hasattr(self.control_panel, 'is_transcribing') and not self.control_panel.is_transcribing:
            self.subtitle_widget.transcript_text = []
            info_text = f"{language_info}\n准备就绪，点击'开始转录'按钮开始捕获系统音频"
            self.subtitle_widget.subtitle_label.setText(info_text)
            # 滚动到顶部
            QTimer.singleShot(100, lambda: self.subtitle_widget.verticalScrollBar().setValue(0))
    except Exception as e:
        logger.error(f"设置语言模式时出错: {str(e)}")
        logger.error(traceback.format_exc())
        self.signals.status_updated.emit(f"设置语言模式失败: {str(e)}")

def _get_language_mode_display(self, mode):
    """获取语言模式显示名称"""
    if mode == "en":
        return "英文识别"
    elif mode == "zh":
        return "中文识别"
    elif mode == "auto":
        return "自动识别"
    return mode

def set_audio_mode(self, mode):
    """
    设置音频模式
    
    Args:
        mode: 音频模式，可选值为"system"、"file"
    """
    try:
        logger.info(f"设置音频模式: {mode}")
        
        # 更新配置
        self.config_manager.set_config(mode, "recognition", "audio_mode")
        self.config_manager.save_config("main")
        
        # 更新状态栏
        self.signals.status_updated.emit(f"已设置音频模式: {self._get_audio_mode_display(mode)}")
        
        # 如果是文件模式，打开文件选择对话框
        if mode == "file":
            self.select_file()
        else:
            # 系统音频模式
            self.is_file_mode = False
            
            # 更新控制面板
            if hasattr(self.control_panel, 'set_transcription_mode'):
                self.control_panel.set_transcription_mode("system")
            
            # 在字幕窗口显示模式设置信息
            mode_info = f"已设置音频模式: {self._get_audio_mode_display(mode)}"
            
            # 只有在没有进行转录时才更新字幕窗口
            if hasattr(self.control_panel, 'is_transcribing') and not self.control_panel.is_transcribing:
                # 保留现有文本，如果有的话
                current_text = self.subtitle_widget.subtitle_label.text()
                
                # 如果当前文本为空或只包含准备就绪信息，则设置新文本
                if not current_text or "准备就绪" in current_text:
                    self.subtitle_widget.transcript_text = []
                    info_text = f"{mode_info}\n准备就绪，点击'开始转录'按钮开始捕获系统音频"
                    self.subtitle_widget.subtitle_label.setText(info_text)
                else:
                    # 否则，将模式信息添加到当前文本的最下方
                    self.subtitle_widget.subtitle_label.setText(current_text + "\n\n" + mode_info)
                
                # 滚动到底部，确保最新信息可见
                QTimer.singleShot(100, self.subtitle_widget._scroll_to_bottom)
    except Exception as e:
        logger.error(f"设置音频模式时出错: {str(e)}")
        logger.error(traceback.format_exc())
        self.signals.status_updated.emit(f"设置音频模式失败: {str(e)}")

def _get_audio_mode_display(self, mode):
    """获取音频模式显示名称"""
    if mode == "system":
        return "系统音频模式"
    elif mode == "file":
        return "文件音频模式"
    return mode

def toggle_speaker_identification(self, enabled):
    """
    切换说话人识别功能
    
    Args:
        enabled: 是否启用
    """
    try:
        logger.info(f"切换说话人识别功能: {enabled}")
        
        # 更新配置
        self.config_manager.set_config(enabled, "recognition", "speaker_identification")
        self.config_manager.save_config("main")
        
        # 更新状态栏
        status = "启用" if enabled else "禁用"
        self.signals.status_updated.emit(f"已{status}说话人识别功能")
        
        # 在字幕窗口显示功能设置信息
        feature_info = f"已{status}说话人识别功能"
        
        # 只有在没有进行转录时才更新字幕窗口
        if hasattr(self.control_panel, 'is_transcribing') and not self.control_panel.is_transcribing:
            # 保留现有文本，如果有的话
            current_text = self.subtitle_widget.subtitle_label.text()
            
            # 如果当前文本为空或只包含准备就绪信息，则设置新文本
            if not current_text or "准备就绪" in current_text:
                self.subtitle_widget.transcript_text = []
                info_text = f"{feature_info}\n准备就绪，点击'开始转录'按钮开始捕获系统音频"
                self.subtitle_widget.subtitle_label.setText(info_text)
            else:
                # 否则，将功能信息添加到当前文本的最下方
                self.subtitle_widget.subtitle_label.setText(current_text + "\n\n" + feature_info)
            
            # 滚动到底部，确保最新信息可见
            QTimer.singleShot(100, self.subtitle_widget._scroll_to_bottom)
    except Exception as e:
        logger.error(f"切换说话人识别功能时出错: {str(e)}")
        logger.error(traceback.format_exc())
        self.signals.status_updated.emit(f"切换说话人识别功能失败: {str(e)}")

def search_model_documentation(self):
    """搜索模型文档"""
    try:
        logger.info("搜索模型文档")
        
        # 打开浏览器搜索模型文档
        import webbrowser
        webbrowser.open("https://github.com/alphacep/vosk-api/wiki")
        
        # 更新状态栏
        self.signals.status_updated.emit("已打开模型文档搜索页面")
    except Exception as e:
        logger.error(f"搜索模型文档时出错: {str(e)}")
        logger.error(traceback.format_exc())
        self.signals.status_updated.emit(f"搜索模型文档失败: {str(e)}")

def refresh_models(self):
    """刷新模型列表"""
    try:
        logger.info("刷新模型列表")
        
        # 重新加载模型列表
        self.model_manager.refresh_models()
        
        # 更新状态栏
        self.signals.status_updated.emit("模型列表已刷新")
    except Exception as e:
        logger.error(f"刷新模型列表时出错: {str(e)}")
        logger.error(traceback.format_exc())
        self.signals.status_updated.emit(f"刷新模型列表失败: {str(e)}")

def refresh_plugins(self):
    """刷新插件"""
    try:
        logger.info("刷新插件")
        
        # 刷新插件
        from src.core.plugins import PluginManager
        plugin_manager = PluginManager()
        plugin_manager.refresh_plugins()
        
        # 更新状态栏
        self.signals.status_updated.emit("插件已刷新")
    except Exception as e:
        logger.error(f"刷新插件时出错: {str(e)}")
        logger.error(traceback.format_exc())
        self.signals.status_updated.emit(f"刷新插件失败: {str(e)}")

def _show_model_manager(self):
    """显示模型管理器对话框"""
    try:
        logger.info("显示模型管理器对话框")
        
        # 创建并显示模型管理器对话框
        from src.ui.dialogs.model_manager_dialog import ModelManagerDialog
        dialog = ModelManagerDialog(self)
        dialog.exec_()
        
        # 对话框关闭后，刷新模型列表
        self.refresh_models()
    except Exception as e:
        logger.error(f"显示模型管理器对话框时出错: {str(e)}")
        logger.error(traceback.format_exc())
        self.signals.error_occurred.emit(f"显示模型管理器对话框失败: {str(e)}")

def _show_plugin_manager(self):
    """显示插件管理器对话框"""
    try:
        logger.info("显示插件管理器对话框")
        
        # 创建并显示插件管理器对话框
        from src.ui.dialogs.plugin_manager_dialog import PluginManagerDialog
        dialog = PluginManagerDialog(self)
        dialog.exec_()
        
        # 对话框关闭后，刷新插件
        self.refresh_plugins()
    except Exception as e:
        logger.error(f"显示插件管理器对话框时出错: {str(e)}")
        logger.error(traceback.format_exc())
        self.signals.error_occurred.emit(f"显示插件管理器对话框失败: {str(e)}")
