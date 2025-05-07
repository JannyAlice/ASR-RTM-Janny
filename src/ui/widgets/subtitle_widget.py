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
        # 可能的值：'vosk_small', 'sherpa_onnx_int8', 'sherpa_onnx_std', 'sherpa_0626_int8', 'sherpa_0626_std'
        # 用于在update_text方法中根据引擎类型采用不同的处理逻辑
        # 这样可以确保不同引擎的结果都能正确显示，而不会互相影响
        self.current_engine_type = None

        # 初始化audio_worker属性，用于保存AudioWorker实例
        # 这样可以在停止转录时获取最后一个单词
        self.audio_worker = None

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
        try:
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

            if not text:
                return text

            # 记录原始文本
            sherpa_logger.debug(f"格式化文本前: '{text}'")

            # 检查文本是否已经格式化
            # 如果文本已经首字母大写且末尾有标点符号，则认为已经格式化
            is_already_formatted = (
                len(text) > 0 and
                text[0].isupper() and
                (text[-1] in ['.', '?', '!', ',', ';', ':', '-'])
            )

            if is_already_formatted:
                sherpa_logger.debug(f"文本已格式化，跳过: '{text}'")
                return text

            # 对所有模型统一处理
            # 首字母大写
            if len(text) > 0:
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

            # 记录格式化后的文本
            sherpa_logger.debug(f"格式化文本后: '{text}'")
            return text

        except Exception as e:
            print(f"格式化文本错误: {e}")
            import traceback
            print(traceback.format_exc())
            # 如果发生错误，返回原始文本
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

                # 检查是否与最后一个完整结果相似
                if self.transcript_text and self._is_similar(partial_text, self.transcript_text[-1]):
                    sherpa_logger.debug(f"部分结果与最后一个完整结果相似，不更新: {partial_text}")
                    return

                # 如果部分文本不是当前部分段落的一部分，更新当前部分段落
                if not self.current_partial_paragraph or partial_text not in self.current_partial_paragraph:
                    # 如果当前部分段落为空，直接设置
                    if not self.current_partial_paragraph:
                        sherpa_logger.debug(f"设置首次部分结果: {partial_text}")
                        self.current_partial_paragraph = partial_text
                    else:
                        # 检查部分文本是否是当前部分段落的一部分
                        # 如果是，保留当前部分段落
                        # 如果不是，使用新的部分文本
                        if self.current_partial_paragraph in partial_text:
                            # 新的部分文本包含当前部分段落，使用新的部分文本
                            sherpa_logger.debug(f"部分结果更新 (包含旧结果): 旧={self.current_partial_paragraph}, 新={partial_text}")
                            self.current_partial_paragraph = partial_text
                        elif partial_text in self.current_partial_paragraph:
                            # 当前部分段落包含新的部分文本，保留当前部分段落
                            sherpa_logger.debug(f"保留当前部分结果 (包含新结果): 当前={self.current_partial_paragraph}, 新={partial_text}")
                            # 不更新，保留当前部分段落
                            pass
                        else:
                            # 检查相似度
                            if self._is_similar(partial_text, self.current_partial_paragraph):
                                # 如果相似度高，使用较长的文本
                                if len(partial_text) > len(self.current_partial_paragraph):
                                    sherpa_logger.debug(f"部分结果更新 (相似且更长): 旧={self.current_partial_paragraph}, 新={partial_text}")
                                    self.current_partial_paragraph = partial_text
                                else:
                                    sherpa_logger.debug(f"保留当前部分结果 (相似但更短): 当前={self.current_partial_paragraph}, 新={partial_text}")
                                    # 不更新，保留当前部分段落
                                    pass
                            else:
                                # 新的部分文本与当前部分段落无关，使用新的部分文本
                                sherpa_logger.debug(f"部分结果更新 (全新结果): 旧={self.current_partial_paragraph}, 新={partial_text}")
                                self.current_partial_paragraph = partial_text

                # 保存最新的部分结果，用于后续处理
                # 这对于在停止转录时获取最后一个单词特别有用
                try:
                    # 尝试将最新的部分结果保存到AudioWorker实例中
                    if hasattr(self, 'audio_worker') and self.audio_worker:
                        if hasattr(self.audio_worker, '_last_partial_result'):
                            self.audio_worker._last_partial_result = self.current_partial_paragraph
                            sherpa_logger.debug(f"保存最新部分结果到AudioWorker: {self.current_partial_paragraph}")
                except Exception as e:
                    sherpa_logger.error(f"保存最新部分结果错误: {e}")
                    import traceback
                    sherpa_logger.error(traceback.format_exc())

                # 显示最近的完整结果加上当前的部分段落
                # 但不将部分结果添加到transcript_text列表中
                display_text = self.transcript_text[-9:] if self.transcript_text else []  # 显示更多的历史记录

                # 只有当部分结果不为空时才添加到显示列表中
                if self.current_partial_paragraph:
                    # 将部分结果添加到显示列表中，但不添加到transcript_text列表中
                    display_text.append(self.current_partial_paragraph)
                    print(f"显示部分结果: {self.current_partial_paragraph}")

                # 更新字幕标签
                print(f"[DEBUG] 更新部分结果: {self.current_partial_paragraph}")
                print(f"[DEBUG] 显示文本列表: {display_text}")
                try:
                    # 导入 Sherpa-ONNX 日志工具
                    from src.utils.sherpa_logger import sherpa_logger
                    sherpa_logger.info(f"更新部分结果: {self.current_partial_paragraph}")
                    sherpa_logger.debug(f"显示文本列表: {display_text}")
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

                    # 检查是否是最终结果（通常比部分结果更完整）
                    # 如果当前文本比最后一个文本长，可能是更完整的最终结果
                    if len(text) > len(self.transcript_text[-1]):
                        sherpa_logger.info(f"检测到可能的最终结果，替换最后一个文本")
                        sherpa_logger.info(f"原文本: {self.transcript_text[-1]}")
                        sherpa_logger.info(f"新文本: {text}")

                        # 替换最后一个文本
                        self.transcript_text[-1] = text

                        # 更新完整转录历史记录
                        if self.full_transcript_history:
                            self.full_transcript_history[-1] = text
                        else:
                            self.full_transcript_history.append(text)

                        # 更新带时间戳的转录历史记录
                        import time
                        timestamp = time.strftime("%H:%M:%S")
                        if self.timestamped_transcript_history:
                            self.timestamped_transcript_history[-1] = (text, timestamp)
                        else:
                            self.timestamped_transcript_history.append((text, timestamp))

                        sherpa_logger.info(f"[{timestamp}] 更新最终结果: {text}")
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
                    print(f"[DEBUG] 更新完整结果，显示 {len(self.transcript_text[-500:])} 行文本")
                    print(f"[DEBUG] 完整文本列表: {self.transcript_text}")
                    sherpa_logger.debug(f"更新字幕窗口，显示 {len(self.transcript_text[-500:])} 行文本")
                    sherpa_logger.debug(f"完整文本列表: {self.transcript_text}")
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
        try:
            # 直接使用自身的垂直滚动条
            scroll_bar = self.verticalScrollBar()
            if scroll_bar:
                scroll_bar.setValue(scroll_bar.maximum())
                # 确保滚动到底部后保持位置
                QTimer.singleShot(50, lambda: scroll_bar.setValue(scroll_bar.maximum()))
        except Exception as e:
            print(f"滚动到底部错误: {e}")
            import traceback
            print(traceback.format_exc())

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

    def get_all_transcript_data(self):
        """
        获取所有转录数据，包括带时间戳的转录历史、完整转录历史、部分结果历史和当前显示内容

        Returns:
            dict: 包含所有转录数据的字典
        """
        # 获取当前显示内容
        current_display = self.subtitle_label.text()

        # 返回所有数据
        return {
            'timestamped_transcript': self.timestamped_transcript_history,
            'full_transcript': self.full_transcript_history,
            'partial_results': self.partial_results_history,
            'current_display': current_display
        }

    def _find_matching_complete_text(self, text):
        """查找与给定文本匹配的完整句子

        Args:
            text (str): 要匹配的文本（通常是部分结果）

        Returns:
            str: 匹配的完整句子，如果没有找到则返回None
        """
        try:
            # 导入日志工具
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

            if not text:
                return None

            # 打印完整的transcript_text列表，便于调试
            sherpa_logger.info(f"查找匹配的完整句子，当前完整文本列表: {self.transcript_text}")

            # 首先检查是否有以"and"结尾的部分结果
            if " and " in text or text.endswith(" and"):
                sherpa_logger.info(f"检测到以'and'结尾的部分结果: {text}")
                # 特殊处理以"and"结尾的情况
                for complete_text in reversed(self.transcript_text):
                    # 检查是否有包含相同前缀但更完整的句子
                    if complete_text.startswith(text.rstrip(" and")) and "and " in complete_text:
                        sherpa_logger.info(f"找到匹配的完整结果(and特殊处理): {complete_text}")
                        return complete_text

            # 如果没有找到匹配，继续使用常规匹配逻辑
            if text.endswith(" and"):
                # 查找最近的完整结果中是否包含当前部分结果
                for complete_text in reversed(self.transcript_text):
                    # 检查部分结果是否是完整结果的前缀（去掉末尾的"and"）
                    prefix = text.rstrip(" and")
                    if prefix and complete_text.startswith(prefix):
                        sherpa_logger.info(f"找到匹配的完整结果(前缀匹配-去除and): {complete_text}")
                        return complete_text

            # 如果仍然没有找到匹配，使用常规匹配逻辑
            if text.endswith(" and"):
                for complete_text in reversed(self.transcript_text):
                    # 检查部分结果（去掉末尾的"and"）是否包含在完整结果中
                    prefix = text.rstrip(" and")
                    if prefix and prefix in complete_text:
                        sherpa_logger.info(f"找到匹配的完整结果(子串匹配-去除and): {complete_text}")
                        return complete_text

            # 如果仍然没有找到匹配，使用常规匹配逻辑
            for complete_text in reversed(self.transcript_text):
                # 检查部分结果是否是完整结果的前缀
                if complete_text.startswith(text):
                    sherpa_logger.info(f"找到匹配的完整结果(前缀匹配): {complete_text}")
                    return complete_text
                # 检查部分结果是否包含在完整结果中
                elif text in complete_text:
                    sherpa_logger.info(f"找到匹配的完整结果(子串匹配): {complete_text}")
                    return complete_text
                # 检查完整结果是否包含部分结果的大部分内容
                elif len(text) > 10:  # 只对较长的部分结果进行相似度检查
                    # 计算部分结果的单词
                    partial_words = text.split()
                    # 计算完整结果的单词
                    complete_words = complete_text.split()
                    # 计算共同单词的数量
                    common_words = set(partial_words) & set(complete_words)
                    # 如果共同单词的数量超过部分结果单词数量的80%，认为匹配
                    if len(common_words) >= 0.8 * len(partial_words):
                        sherpa_logger.info(f"找到匹配的完整结果(相似度匹配): {complete_text}")
                        return complete_text

                    # 如果部分结果以"and"结尾，特殊处理
                    if text.endswith(" and"):
                        # 计算去掉"and"后的相似度
                        partial_words_no_and = text.rstrip(" and").split()
                        if partial_words_no_and:
                            common_words_no_and = set(partial_words_no_and) & set(complete_words)
                            if len(common_words_no_and) >= 0.8 * len(partial_words_no_and):
                                sherpa_logger.info(f"找到匹配的完整结果(相似度匹配-去除and): {complete_text}")
                                return complete_text

            # 如果没有找到匹配的完整句子，返回None
            sherpa_logger.info(f"未找到匹配的完整句子: {text}")
            return None
        except Exception as e:
            print(f"查找匹配的完整句子错误: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def _is_similar(self, text1, text2):
        """检查两段文本是否相似（相似度阈值80%）

        Args:
            text1 (str): 第一段文本
            text2 (str): 第二段文本

        Returns:
            bool: 如果相似度超过阈值返回True
        """
        try:
            # 导入日志工具
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

            if not text1 or not text2:
                return False

            # 长度差异检查
            if abs(len(text1) - len(text2)) > 10:
                sherpa_logger.debug(f"文本长度差异过大: {len(text1)} vs {len(text2)}")
                return False

            # 检查一个文本是否是另一个的子串
            # 这对于处理最终结果特别有用，因为最终结果通常包含部分结果
            if text1 in text2:
                sherpa_logger.debug(f"文本1是文本2的子串: '{text1}' in '{text2}'")
                return True
            if text2 in text1:
                sherpa_logger.debug(f"文本2是文本1的子串: '{text2}' in '{text1}'")
                return True

            # 使用difflib计算相似度
            seq = difflib.SequenceMatcher(None, text1.lower(), text2.lower())
            ratio = seq.ratio()
            sherpa_logger.debug(f"文本相似度检测: '{text1}' vs '{text2}' = {ratio:.2f}")

            # 相似度超过80%视为重复
            # 但如果文本长度差异很小且相似度超过70%，也视为重复
            if ratio > 0.8:
                return True
            elif abs(len(text1) - len(text2)) <= 5 and ratio > 0.7:
                sherpa_logger.debug(f"文本长度差异小且相似度较高: {abs(len(text1) - len(text2))} 差异, {ratio:.2f} 相似度")
                return True

            return False
        except Exception as e:
            print(f"相似度检测错误: {e}")
            import traceback
            print(traceback.format_exc())
            # 出错时返回False，避免误判
            return False
