"""
UI设置菜单模块
负责创建和管理UI设置相关的菜单项
"""
from PyQt5.QtWidgets import QMenu, QAction, QActionGroup

from src.utils.logger import get_logger

logger = get_logger(__name__)

class UISettingsMenu(QMenu):
    """UI设置菜单类"""
    
    def __init__(self, parent=None):
        """
        初始化UI设置菜单
        
        Args:
            parent: 父窗口
        """
        super().__init__(parent)
        self.setTitle("UI设置(&U)")  # 设置菜单标题，带有快捷键
        
        # 创建子菜单
        self._create_background_mode_submenu()
        self._create_font_size_submenu()
        
    def _create_background_mode_submenu(self):
        """创建背景模式子菜单"""
        self.background_menu = QMenu("背景模式(&B)", self)
        self.addMenu(self.background_menu)
        
        # 创建背景模式动作组
        self.bg_group = QActionGroup(self)
        self.bg_group.setExclusive(True)
        
        # 创建背景模式动作
        self.actions = {}
        self.actions['opaque'] = QAction("不透明(&O)", self, checkable=True)
        self.actions['translucent'] = QAction("半透明(&T)", self, checkable=True)
        self.actions['transparent'] = QAction("透明(&R)", self, checkable=True)
        
        # 将动作添加到组
        self.bg_group.addAction(self.actions['opaque'])
        self.bg_group.addAction(self.actions['translucent'])
        self.bg_group.addAction(self.actions['transparent'])
        
        # 将动作添加到菜单
        self.background_menu.addAction(self.actions['opaque'])
        self.background_menu.addAction(self.actions['translucent'])
        self.background_menu.addAction(self.actions['transparent'])
        
        # 设置默认选中项
        self.actions['translucent'].setChecked(True)
        
    def _create_font_size_submenu(self):
        """创建字体大小子菜单"""
        self.font_menu = QMenu("字体大小(&F)", self)
        self.addMenu(self.font_menu)
        
        # 创建字体大小动作组
        self.font_group = QActionGroup(self)
        self.font_group.setExclusive(True)
        
        # 创建字体大小动作
        self.actions['small'] = QAction("小(&S)", self, checkable=True)
        self.actions['medium'] = QAction("中(&M)", self, checkable=True)
        self.actions['large'] = QAction("大(&L)", self, checkable=True)
        
        # 将动作添加到组
        self.font_group.addAction(self.actions['small'])
        self.font_group.addAction(self.actions['medium'])
        self.font_group.addAction(self.actions['large'])
        
        # 将动作添加到菜单
        self.font_menu.addAction(self.actions['small'])
        self.font_menu.addAction(self.actions['medium'])
        self.font_menu.addAction(self.actions['large'])
        
        # 设置默认选中项
        self.actions['medium'].setChecked(True)
        
    def connect_signals(self, main_window):
        """
        连接信号到主窗口槽函数
        
        Args:
            main_window: 主窗口实例
        """
        try:
            # 背景模式选择信号
            self.actions['opaque'].triggered.connect(
                lambda: main_window.set_background_mode("opaque")
            )
            self.actions['translucent'].triggered.connect(
                lambda: main_window.set_background_mode("translucent")
            )
            self.actions['transparent'].triggered.connect(
                lambda: main_window.set_background_mode("transparent")
            )
            
            # 字体大小选择信号
            self.actions['small'].triggered.connect(
                lambda: main_window.set_font_size("small")
            )
            self.actions['medium'].triggered.connect(
                lambda: main_window.set_font_size("medium")
            )
            self.actions['large'].triggered.connect(
                lambda: main_window.set_font_size("large")
            )
            
            logger.info("UI设置菜单信号连接完成")
        except Exception as e:
            logger.error(f"连接UI设置菜单信号时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
    def update_menu_state(self, _=False):
        """
        更新菜单状态
        
        Args:
            _: 忽略参数，保持接口一致性
        """
        # UI设置菜单始终可用，不受录音状态影响
        pass
