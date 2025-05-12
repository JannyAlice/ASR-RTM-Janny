"""
扩展管理菜单模块
负责创建和管理扩展管理相关的菜单项
"""
import traceback
from PyQt5.QtWidgets import QMenu, QAction
from PyQt5.QtCore import pyqtSignal

from src.utils.logger import get_logger
from src.core.plugins import PluginManager

logger = get_logger(__name__)

class ExtensionMenu(QMenu):
    """扩展管理菜单类"""
    
    model_manager_requested = pyqtSignal()  # 请求打开模型管理器
    plugin_manager_requested = pyqtSignal()  # 请求打开插件管理器
    
    def __init__(self, parent=None):
        """
        初始化扩展管理菜单
        
        Args:
            parent: 父窗口
        """
        super().__init__(parent)
        self.setTitle("扩展管理(&E)")  # 设置菜单标题，带有快捷键
        
        # 初始化插件管理器
        self.plugin_manager = PluginManager()
        
        # 创建子菜单
        self._create_model_management_submenu()
        self._create_plugin_management_submenu()
        self._create_extra_functions_submenu()
        
    def _create_model_management_submenu(self):
        """创建模型管理子菜单"""
        self.model_management_menu = QMenu("模型管理(&M)", self)
        self.addMenu(self.model_management_menu)
        
        # 添加模型管理菜单项
        self.actions = {}
        self.actions['model_manager'] = QAction("模型设定管理(&S)...", self)
        self.actions['model_manager'].triggered.connect(lambda: self.model_manager_requested.emit())
        
        self.actions['refresh_models'] = QAction("刷新模型列表(&R)", self)
        self.actions['refresh_models'].triggered.connect(self._refresh_models)
        
        self.actions['show_sys_info'] = QAction("显示系统信息(&I)", self)
        self.actions['check_model_dir'] = QAction("检查模型目录(&C)", self)
        self.actions['search_model_doc'] = QAction("搜索模型文档(&D)", self)
        
        # 添加到菜单
        self.model_management_menu.addAction(self.actions['model_manager'])
        self.model_management_menu.addAction(self.actions['refresh_models'])
        self.model_management_menu.addSeparator()
        self.model_management_menu.addAction(self.actions['show_sys_info'])
        self.model_management_menu.addAction(self.actions['check_model_dir'])
        self.model_management_menu.addAction(self.actions['search_model_doc'])
        
    def _create_plugin_management_submenu(self):
        """创建插件管理子菜单"""
        self.plugin_management_menu = QMenu("插件管理(&P)", self)
        self.addMenu(self.plugin_management_menu)
        
        # 添加插件管理菜单项
        self.actions['plugin_manager'] = QAction("插件管理(&M)...", self)
        self.actions['plugin_manager'].triggered.connect(lambda: self.plugin_manager_requested.emit())
        
        self.actions['refresh_plugins'] = QAction("刷新插件(&R)", self)
        self.actions['refresh_plugins'].triggered.connect(self._refresh_plugins)
        
        # 添加到菜单
        self.plugin_management_menu.addAction(self.actions['plugin_manager'])
        self.plugin_management_menu.addAction(self.actions['refresh_plugins'])
        
    def _create_extra_functions_submenu(self):
        """创建附加功能子菜单"""
        self.extra_functions_menu = QMenu("附加功能(&F)", self)
        self.addMenu(self.extra_functions_menu)
        
        # 添加附加功能菜单项
        self.actions['speaker_id'] = QAction("启用说话人识别(&S)", self, checkable=True)
        
        # 添加到菜单
        self.extra_functions_menu.addAction(self.actions['speaker_id'])
        
    def _refresh_models(self):
        """刷新模型列表"""
        try:
            # 重新加载插件
            self.plugin_manager.reload_plugins()
            logger.info("已刷新模型列表")
        except Exception as e:
            logger.error(f"刷新模型列表时出错: {str(e)}")
            logger.error(traceback.format_exc())
            
    def _refresh_plugins(self):
        """刷新插件"""
        try:
            # 重新加载插件
            self.plugin_manager.reload_plugins()
            logger.info("已刷新插件")
        except Exception as e:
            logger.error(f"刷新插件时出错: {str(e)}")
            logger.error(traceback.format_exc())
        
    def connect_signals(self, main_window):
        """
        连接信号到主窗口槽函数
        
        Args:
            main_window: 主窗口实例
        """
        try:
            # 模型管理信号
            self.model_manager_requested.connect(main_window._show_model_manager)
            self.actions['show_sys_info'].triggered.connect(main_window.show_system_info)
            self.actions['check_model_dir'].triggered.connect(main_window.check_model_directory)
            self.actions['search_model_doc'].triggered.connect(main_window.search_model_documentation)
            
            # 插件管理信号
            self.plugin_manager_requested.connect(main_window._show_plugin_manager)
            
            # 附加功能信号
            self.actions['speaker_id'].triggered.connect(
                lambda checked: main_window.toggle_speaker_identification(checked)
            )
            
            logger.info("扩展管理菜单信号连接完成")
        except Exception as e:
            logger.error(f"连接扩展管理菜单信号时出错: {str(e)}")
            logger.error(traceback.format_exc())
        
    def update_menu_state(self, is_recording=False):
        """
        更新菜单状态
        
        Args:
            is_recording: 是否正在录音
        """
        # 在录音时禁用某些功能
        self.actions['refresh_models'].setEnabled(not is_recording)
        self.actions['refresh_plugins'].setEnabled(not is_recording)
        self.actions['speaker_id'].setEnabled(not is_recording)
