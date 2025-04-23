"""
字幕控件模块
负责字幕的显示和样式管理
"""
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QGraphicsOpacityEffect
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, pyqtSlot

from src.utils.config_manager import config_manager

class SubtitleLabel(QLabel):
    """字幕标签类"""

    def __init__(self, parent=None):
        """
        初始化字幕标签

        Args:
            parent: 父控件
        """
        super().__init__(parent)

        # 加载配置
        self.config = config_manager
        self.font_config = self.config.get_ui_config('fonts.subtitle', {})
        self.colors_config = self.config.get_ui_config('colors', {})
        self.styles_config = self.config.get_ui_config('styles', {})

        # 设置样式
        self._apply_styles()

        # 设置初始文本
        self.setText("准备就绪...")

    def _apply_styles(self):
        """应用样式"""
        # 获取字体配置
        font_family = self.font_config.get('family', 'Arial')
        font_size = self.font_config.get('size', {}).get('medium', 24)
        font_weight = self.font_config.get('weight', 'bold')
        font_color = self.font_config.get('color', '#FFFFFF')

        # 获取样式配置
        padding = self.styles_config.get('subtitle_padding', 15)
        border_radius = self.styles_config.get('subtitle_border_radius', 10)
        bg_color = self.colors_config.get('subtitle_background', 'rgba(0, 0, 0, 150)')

        # 设置字体
        font = QFont(font_family, font_size)
        font.setBold(font_weight == 'bold')
        self.setFont(font)

        # 设置样式表
        self.setStyleSheet(f"""
            QLabel {{
                color: {font_color};
                background-color: {bg_color};
                padding: {padding}px;
                border-radius: {border_radius}px;
                qproperty-alignment: AlignCenter;
            }}
        """)

        # 设置自动换行
        self.setWordWrap(True)

    def set_font_size(self, size_key):
        """
        设置字体大小

        Args:
            size_key: 字体大小键（small, medium, large）
        """
        # 获取字体大小配置
        sizes = self.font_config.get('size', {})
        size = sizes.get(size_key, 24)  # 默认中等大小

        # 更新字体
        font = self.font()
        font.setPointSize(size)
        self.setFont(font)

    def set_opacity(self, opacity):
        """
        设置不透明度

        Args:
            opacity: 不透明度值（0.0-1.0）
        """
        effect = QGraphicsOpacityEffect(self)
        effect.setOpacity(opacity)
        self.setGraphicsEffect(effect)

class SubtitleWidget(QWidget):
    """字幕控件类"""

    def __init__(self, parent=None):
        """
        初始化字幕控件

        Args:
            parent: 父控件
        """
        super().__init__(parent)

        # 创建布局
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # 创建字幕标签
        self.subtitle_label = SubtitleLabel(self)
        self.layout.addWidget(self.subtitle_label)

        # 设置布局
        self.setLayout(self.layout)

        # 设置背景透明
        self.setAttribute(Qt.WA_TranslucentBackground)

    @pyqtSlot(str)
    def update_text(self, text):
        """
        更新字幕文本

        Args:
            text: 新的字幕文本
        """
        self.subtitle_label.setText(text)

    def set_font_size(self, size_key):
        """
        设置字体大小

        Args:
            size_key: 字体大小键（small, medium, large）
        """
        self.subtitle_label.set_font_size(size_key)

    def set_background_mode(self, mode):
        """
        设置背景模式

        Args:
            mode: 背景模式（opaque, translucent, transparent）
        """
        if mode == 'opaque':
            self.subtitle_label.set_opacity(1.0)
        elif mode == 'translucent':
            self.subtitle_label.set_opacity(0.8)
        elif mode == 'transparent':
            self.subtitle_label.set_opacity(0.5)
