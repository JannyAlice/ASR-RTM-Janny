"""
字幕控件模块
负责字幕的显示和样式管理
"""
import difflib
from PyQt5.QtWidgets import (QLabel, QVBoxLayout, QWidget, QGraphicsOpacityEffect,
                             QScrollArea, QSizePolicy)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, pyqtSlot, QTimer

from src.utils.config_manager import config_manager

class SubtitleLabel(QLabel):
    """字幕标签类。"""

    def __init__(self, parent=None):
        """初始化字幕标签。

        Args:
            parent (QWidget): 父控件
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
        """应用样式。"""
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
                qproperty-alignment: AlignLeft;
            }}
        """)

        # 设置自动换行
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_font_size(self, size_key):
        """设置字体大小。

        Args:
            size_key (str): 字体大小键('small', 'medium', 'large')
        """
        # 获取字体大小配置
        sizes = self.font_config.get('size', {})
        size = sizes.get(size_key, 24)  # 默认中等大小

        # 更新字体
        font = self.font()
        font.setPointSize(size)
        self.setFont(font)

    def set_opacity(self, opacity):
        """设置不透明度。

        Args:
            opacity (float): 不透明度值(0.0-1.0)
        """
        effect = QGraphicsOpacityEffect(self)
        effect.setOpacity(opacity)
        self.setGraphicsEffect(effect)

class SubtitleWidget(QScrollArea):
    """字幕控件类。"""

    def __init__(self, parent=None):
        """初始化字幕控件。

        Args:
            parent (QWidget): 父控件
        """
        super().__init__(parent)

        # 初始化转录文本列表和输出文件路径（如果尚未初始化）
        if not hasattr(self, 'transcript_text'):
            self.transcript_text = []
        if not hasattr(self, 'output_file'):
            self.output_file = None

        # 初始化完整转录历史记录（包括所有上屏内容）
        self.full_transcript_history = []

        # 初始化部分结果历史记录
        self.partial_results_history = []

        # 初始化带时间戳的转录历史记录
        self.timestamped_transcript_history = []

        # 初始化引擎类型（用于区分不同的ASR引擎）
        # 这个属性由MainWindow类在set_asr_model和_load_default_model方法中设置
        # 可能的值：'vosk', 'sherpa_int8', 'sherpa_std'
        # 用于在update_text方法中根据引擎类型采用不同的处理逻辑
        # 这样可以确保不同引擎的结果都能正确显示，而不会互相影响
        self.current_engine_type = None

        # 设置滚动区域属性
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # 强制垂直滚动条始终显示
        self.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: rgba(50, 50, 50, 150);
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: rgba(100, 100, 100, 200);
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        # 创建内容容器
        self.container = QWidget()
        self.container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)

        # 创建字幕标签
        self.subtitle_label = SubtitleLabel(self.container)
        self.container_layout.addWidget(self.subtitle_label)

        # 设置内容部件
        self.setWidget(self.container)

        # 设置背景透明
        self.setAttribute(Qt.WA_TranslucentBackground)

    def _format_text(self, text):
        """格式化文本：添加标点、首字母大写等

        Args:
            text (str): 原始文本

        Returns:
            str: 格式化后的文本
        """
        if not text:
            return text

        # 不再需要区分引擎类型，对所有模型统一处理

        # 对所有模型统一处理
        # 首字母大写
        text = text[0].upper() + text[1:]

        # 如果文本末尾没有标点符号，添加句号
        if text and text[-1] not in ['.', '?', '!', ',', ';', ':', '-']:
            text += '.'

        # 处理常见的问句开头
        question_starters = ['what', 'who', 'where', 'when', 'why', 'how',
                           'is', 'are', 'do', 'does', 'did', 'can', 'could',
                           'will', 'would', 'should']
        words = text.split()
        if words and words[0].lower() in question_starters:
            # 将句尾的句号替换为问号
            if text[-1] == '.':
                text = text[:-1] + '?'

        return text

    @pyqtSlot(str)
    def update_text(self, text):
        """更新字幕文本。

        Args:
            text (str): 新的字幕文本
        """
        try:
            # 确保转录文本列表存在
            if not hasattr(self, 'transcript_text'):
                self.transcript_text = []

            # 处理部分结果标记
            is_partial = text.startswith("PARTIAL:")
            if is_partial:
                # 部分结果只显示，不添加到转录文本列表
                partial_text = text[8:]  # 移除PARTIAL:标记
                partial_text = self._format_text(partial_text) if partial_text else partial_text

                # 不再需要区分引擎类型，对所有模型使用统一的处理逻辑

                # 对所有模型使用统一的处理逻辑
                # 部分结果只显示，不添加到转录文本列表

                # 初始化当前部分段落变量（如果不存在）
                if not hasattr(self, 'current_partial_paragraph'):
                    self.current_partial_paragraph = ""

                # 如果部分文本不是当前部分段落的一部分，更新当前部分段落
                if partial_text not in self.current_partial_paragraph:
                    # 如果当前部分段落为空，直接设置
                    if not self.current_partial_paragraph:
                        self.current_partial_paragraph = partial_text
                    else:
                        # 检查部分文本是否是当前部分段落的一部分
                        # 如果是，保留当前部分段落
                        # 如果不是，使用新的部分文本
                        if self.current_partial_paragraph in partial_text:
                            # 新的部分文本包含当前部分段落，使用新的部分文本
                            self.current_partial_paragraph = partial_text
                        else:
                            # 新的部分文本与当前部分段落无关，使用新的部分文本
                            self.current_partial_paragraph = partial_text

                # 显示最近的完整结果加上当前的部分段落
                # 但不将部分结果添加到transcript_text列表中
                display_text = self.transcript_text[-5:] if self.transcript_text else []

                # 只有当部分结果不为空时才添加到显示列表中
                if self.current_partial_paragraph:
                    # 将部分结果添加到显示列表中，但不添加到transcript_text列表中
                    display_text.append(self.current_partial_paragraph)

                # 更新字幕标签
                print(f"更新部分结果: {self.current_partial_paragraph}")
                try:
                    # 导入 Sherpa-ONNX 日志工具
                    from src.utils.sherpa_logger import sherpa_logger
                    sherpa_logger.info(f"更新部分结果: {self.current_partial_paragraph}")
                except ImportError:
                    pass

                # 设置字幕文本
                try:
                    self.subtitle_label.setText('\n'.join(display_text))
                except Exception as e:
                    print(f"设置字幕文本错误: {e}")
                    try:
                        from src.utils.sherpa_logger import sherpa_logger
                        sherpa_logger.error(f"设置字幕文本错误: {e}")
                    except ImportError:
                        pass

                # 记录部分结果到历史记录
                self.partial_results_history.append(self.current_partial_paragraph)
            else:
                # 不再需要区分引擎类型，对所有模型使用统一的处理逻辑

                # 对所有模型统一处理，格式化文本
                # 注意：SherpaRecognizer类的Result方法已经进行了格式化，这里不需要再次格式化
                # 但为了保持一致性，我们仍然调用_format_text方法
                text = self._format_text(text)

                # 导入 Sherpa-ONNX 日志工具
                try:
                    from src.utils.sherpa_logger import sherpa_logger
                except ImportError:
                    # 如果导入失败，创建一个简单的日志记录器
                    class DummyLogger:
                        def debug(self, msg): print(f"DEBUG: {msg}")
                        def info(self, msg): print(f"INFO: {msg}")
                        def warning(self, msg): print(f"WARNING: {msg}")
                        def error(self, msg): print(f"ERROR: {msg}")
                    sherpa_logger = DummyLogger()

                # 检查是否与最后一个结果相同或相似
                if self.transcript_text and (text == self.transcript_text[-1] or self._is_similar(text, self.transcript_text[-1])):
                    # 如果是重复或非常相似的文本，不添加到列表
                    print(f"跳过重复文本: {text}")
                    sherpa_logger.info(f"跳过重复文本: {text}")
                else:
                    # 添加新的完整结果到转录文本列表
                    print(f"添加新文本: {text}")
                    sherpa_logger.info(f"添加新文本: {text}")

                    # 直接添加到转录文本列表
                    self.transcript_text.append(text)

                    # 添加到完整转录历史记录
                    self.full_transcript_history.append(text)

                    # 添加带时间戳的转录历史记录
                    import time
                    timestamp = time.strftime("%H:%M:%S")
                    self.timestamped_transcript_history.append((text, timestamp))
                    sherpa_logger.info(f"[{timestamp}] {text}")

                    # 如果列表太长，删除旧的段落
                    if len(self.transcript_text) > 5:
                        self.transcript_text = self.transcript_text[-5:]

                # 显示所有完整结果，但限制最大数量以避免性能问题
                try:
                    self.subtitle_label.setText('\n'.join(self.transcript_text[-500:]))
                    sherpa_logger.debug(f"更新字幕窗口，显示 {len(self.transcript_text[-500:])} 行文本")
                except Exception as e:
                    error_msg = f"设置完整结果文本错误: {e}"
                    print(error_msg)
                    sherpa_logger.error(error_msg)
                    import traceback
                    error_trace = traceback.format_exc()
                    sherpa_logger.error(error_trace)
                    print(error_trace)

            # 滚动到底部
            QTimer.singleShot(100, self._scroll_to_bottom)

        except Exception as e:
            error_msg = f"更新字幕错误: {e}"
            print(error_msg)
            try:
                from src.utils.sherpa_logger import sherpa_logger
                sherpa_logger.error(error_msg)
                import traceback
                error_trace = traceback.format_exc()
                sherpa_logger.error(error_trace)
                print(error_trace)
            except ImportError:
                import traceback
                traceback.print_exc()

    def _scroll_to_bottom(self):
        """滚动到底部。"""
        scroll_bar = self.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
        # 确保滚动到底部后保持位置
        QTimer.singleShot(50, lambda: scroll_bar.setValue(scroll_bar.maximum()))

    def set_font_size(self, size_key):
        """设置字体大小。

        Args:
            size_key (str): 字体大小键('small', 'medium', 'large')
        """
        self.subtitle_label.set_font_size(size_key)

    def set_background_mode(self, mode):
        """设置背景模式。

        Args:
            mode (str): 背景模式('opaque', 'translucent', 'transparent')
        """
        if mode == 'opaque':
            self.subtitle_label.set_opacity(1.0)
        elif mode == 'translucent':
            self.subtitle_label.set_opacity(0.8)
        elif mode == 'transparent':
            self.subtitle_label.set_opacity(0.5)

    def save_transcript(self):
        """
        保存转录文本到文件

        注意：此方法已废弃，保留仅为兼容性目的。
        实际的保存操作已移至 MainWindow 类中处理。
        """
        # 返回空值，表示没有保存任何文件
        # 这样可以防止在 MainWindow 中重复保存文件
        return None

    def get_display_text(self):
        """
        获取当前显示的所有文本

        Returns:
            str: 当前显示的所有文本
        """
        return self.subtitle_label.text()

    def get_full_transcript_history(self):
        """
        获取完整的转录历史记录

        Returns:
            str: 完整的转录历史记录，包括所有完整结果
        """
        return '\n'.join(self.full_transcript_history)

    def get_all_transcript_data(self):
        """
        获取所有转录数据，包括完整结果和部分结果

        Returns:
            dict: 包含完整结果和部分结果的字典
        """
        return {
            'full_transcript': self.full_transcript_history,
            'partial_results': self.partial_results_history,
            'timestamped_transcript': self.timestamped_transcript_history,
            'current_display': self.subtitle_label.text()
        }

    def get_timestamped_transcript(self):
        """
        获取带时间戳的转录文本

        Returns:
            str: 带时间戳的转录文本
        """
        if not self.timestamped_transcript_history:
            return ""

        formatted_text = []
        for text, timestamp in self.timestamped_transcript_history:
            formatted_text.append(f"[{timestamp}] {text}")

        return "\n".join(formatted_text)

    def _is_similar(self, text1, text2):
        """检查两段文本是否相似（相似度阈值60%）

        Args:
            text1 (str): 第一段文本
            text2 (str): 第二段文本

        Returns:
            bool: 如果相似度超过阈值返回True
        """
        if not text1 or not text2:
            return False

        # 长度差异检查
        if abs(len(text1) - len(text2)) > 10:
            return False

        # 使用difflib计算相似度
        seq = difflib.SequenceMatcher(None, text1.lower(), text2.lower())
        ratio = seq.ratio()
        print(f"文本相似度检测: '{text1}' vs '{text2}' = {ratio:.2f}")
        return ratio > 0.8  # 相似度超过80%视为重复
