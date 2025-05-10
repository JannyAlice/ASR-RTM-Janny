"""
模型管理对话框模块
用于添加、编辑和删除ASR模型配置
"""
import os
import traceback
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget,
                            QTableWidgetItem, QPushButton, QHeaderView,
                            QMessageBox, QLineEdit, QFileDialog,
                            QFormLayout, QSpinBox, QCheckBox, QWidget,
                            QTabWidget, QMenu, QAction, QApplication)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QCursor

from src.utils.config_manager import config_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelManagerDialog(QDialog):
    """模型管理对话框"""

    models_changed = pyqtSignal()  # 模型配置变更信号

    def __init__(self, parent=None):
        """初始化模型管理对话框"""
        super().__init__(parent)
        self.setWindowTitle("模型管理")
        self.resize(800, 600)

        # 确保配置已加载
        try:
            # 尝试重新加载配置，确保获取最新数据
            config = config_manager.load_config()
            logger.info("已重新加载配置文件")

            # 打印配置内容，用于调试
            if 'asr' in config and 'models' in config['asr']:
                logger.info(f"配置中的ASR模型数量: {len(config['asr']['models'])}")
                logger.info(f"配置中的ASR模型列表: {list(config['asr']['models'].keys())}")
            else:
                logger.warning("配置中没有找到ASR模型配置")

            # 打印整个配置结构
            logger.debug(f"配置结构: {list(config.keys())}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            logger.error(traceback.format_exc())

        # 加载模型配置
        # 直接获取所有配置，用于调试
        all_config = config_manager.get_all_config()
        logger.info(f"配置管理器中的所有配置键: {list(all_config.keys())}")

        if 'asr' in all_config:
            logger.info(f"ASR配置键: {list(all_config['asr'].keys())}")
            if 'models' in all_config['asr']:
                logger.info(f"ASR模型配置键: {list(all_config['asr']['models'].keys())}")
            else:
                logger.warning("ASR配置中没有'models'键")
        else:
            logger.warning("配置中没有'asr'键")

        # 尝试使用get_all_models方法
        all_models = config_manager.get_all_models()
        logger.info(f"通过get_all_models获取的模型: {list(all_models.keys()) if all_models else '无'}")

        # 首先尝试从 asr.models 获取模型配置
        asr_models = config_manager.get_config("asr", "models", {})
        logger.info(f"从asr.models加载的模型配置: {asr_models}")

        # 直接从配置文件读取
        try:
            import json
            config_path = os.path.join('config', 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    logger.info(f"直接从配置文件读取的配置键: {list(config_data.keys())}")
                    if 'asr' in config_data and 'models' in config_data['asr']:
                        file_asr_models = config_data['asr']['models']
                        logger.info(f"直接从配置文件读取的ASR模型: {list(file_asr_models.keys())}")

                        # 使用直接从文件读取的模型配置
                        asr_models = file_asr_models
                        logger.info(f"使用直接从文件读取的ASR模型配置")
            else:
                logger.warning(f"配置文件不存在: {config_path}")
        except Exception as e:
            logger.error(f"直接读取配置文件时出错: {str(e)}")
            logger.error(traceback.format_exc())

        # 初始化模型配置结构
        self.models_config = {
            "vosk": [],
            "sherpa_onnx": []
        }

        # 将asr.models中的模型按类型分类
        if asr_models:
            for model_id, model_config in asr_models.items():
                # 获取模型类型
                model_type = model_config.get("type", "")
                logger.debug(f"处理模型 {model_id}, 类型: {model_type}")

                # 创建模型数据
                model_data = {
                    "id": model_id,
                    "name": model_config.get("name", model_id),
                    "path": model_config.get("path", ""),
                    "sample_rate": model_config.get("config", {}).get("sample_rate", 16000)
                }

                # 根据模型ID分类
                if "vosk" in model_id.lower():
                    model_data["use_words"] = model_config.get("config", {}).get("use_words", True)
                    self.models_config["vosk"].append(model_data)
                    logger.info(f"添加Vosk模型: {model_id}")
                elif "sherpa" in model_id.lower():
                    model_data["use_persistent_stream"] = model_config.get("config", {}).get("use_persistent_stream", True)
                    self.models_config["sherpa_onnx"].append(model_data)
                    logger.info(f"添加Sherpa-ONNX模型: {model_id}")
                # 根据类型分类
                elif model_type == "vosk":
                    model_data["use_words"] = model_config.get("config", {}).get("use_words", True)
                    self.models_config["vosk"].append(model_data)
                    logger.info(f"根据类型添加Vosk模型: {model_id}")
                elif model_type.startswith("sherpa") or model_type == "int8" or model_type == "standard":
                    model_data["use_persistent_stream"] = True  # 默认启用持久流
                    self.models_config["sherpa_onnx"].append(model_data)
                    logger.info(f"根据类型添加Sherpa-ONNX模型: {model_id}")
                # 根据路径分类
                elif "vosk" in model_config.get("path", "").lower():
                    model_data["use_words"] = True
                    self.models_config["vosk"].append(model_data)
                    logger.info(f"根据路径添加Vosk模型: {model_id}")
                elif "sherpa" in model_config.get("path", "").lower():
                    model_data["use_persistent_stream"] = True
                    self.models_config["sherpa_onnx"].append(model_data)
                    logger.info(f"根据路径添加Sherpa-ONNX模型: {model_id}")
                # 根据配置特征分类
                elif "encoder" in model_config.get("config", {}) or "decoder" in model_config.get("config", {}) or "joiner" in model_config.get("config", {}):
                    # 包含encoder/decoder/joiner的可能是Sherpa-ONNX模型
                    model_data["use_persistent_stream"] = True
                    self.models_config["sherpa_onnx"].append(model_data)
                    logger.info(f"根据特征添加为Sherpa-ONNX模型: {model_id}")
                else:
                    # 默认添加为Vosk模型
                    model_data["use_words"] = True
                    self.models_config["vosk"].append(model_data)
                    logger.info(f"默认添加为Vosk模型: {model_id}")

        # 如果没有找到任何模型，尝试从顶级models键获取
        if not any(self.models_config.values()):
            top_level_models = config_manager.get_config("models", {})
            if top_level_models:
                self.models_config = top_level_models
                logger.info(f"从顶级models键加载的模型配置: {top_level_models}")

        logger.info(f"最终加载的模型配置: {self.models_config}")

        self._init_ui()
        self._load_models()

    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)

        # 创建主选项卡（按功能分类）
        self.main_tab_widget = QTabWidget(self)

        # ===== ASR模型选项卡（语音识别） =====
        self.asr_tab = QWidget()
        asr_layout = QVBoxLayout(self.asr_tab)

        # ASR子选项卡
        self.asr_tab_widget = QTabWidget(self.asr_tab)

        # --- Vosk模型选项卡 ---
        self.vosk_tab = QWidget()
        self.vosk_layout = QVBoxLayout(self.vosk_tab)
        self.vosk_table = QTableWidget(0, 5, self.vosk_tab)
        self.vosk_table.setHorizontalHeaderLabels(["ID", "名称", "路径", "采样率", "单词时间戳"])
        self.vosk_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.vosk_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.vosk_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.vosk_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.vosk_table.customContextMenuRequested.connect(lambda pos: self._show_context_menu(pos, "vosk"))
        self.vosk_layout.addWidget(self.vosk_table)

        # Vosk按钮
        vosk_button_layout = QHBoxLayout()
        self.add_vosk_button = QPushButton("添加", self.vosk_tab)
        self.add_vosk_button.clicked.connect(lambda: self._add_model("vosk"))
        self.edit_vosk_button = QPushButton("编辑", self.vosk_tab)
        self.edit_vosk_button.clicked.connect(lambda: self._edit_selected_model("vosk"))
        self.edit_vosk_button.setEnabled(False)  # 初始禁用
        self.delete_vosk_button = QPushButton("删除", self.vosk_tab)
        self.delete_vosk_button.clicked.connect(lambda: self._delete_selected_model("vosk"))
        self.delete_vosk_button.setEnabled(False)  # 初始禁用

        vosk_button_layout.addWidget(self.add_vosk_button)
        vosk_button_layout.addWidget(self.edit_vosk_button)
        vosk_button_layout.addWidget(self.delete_vosk_button)
        vosk_button_layout.addStretch()
        self.vosk_layout.addLayout(vosk_button_layout)

        # 连接选择变化信号
        self.vosk_table.itemSelectionChanged.connect(lambda: self._update_button_states("vosk"))

        # --- Sherpa-ONNX模型选项卡 ---
        self.sherpa_tab = QWidget()
        self.sherpa_layout = QVBoxLayout(self.sherpa_tab)
        self.sherpa_table = QTableWidget(0, 5, self.sherpa_tab)
        self.sherpa_table.setHorizontalHeaderLabels(["ID", "名称", "路径", "采样率", "持久流"])
        self.sherpa_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.sherpa_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.sherpa_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.sherpa_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sherpa_table.customContextMenuRequested.connect(lambda pos: self._show_context_menu(pos, "sherpa_onnx"))
        self.sherpa_layout.addWidget(self.sherpa_table)

        # Sherpa按钮
        sherpa_button_layout = QHBoxLayout()
        self.add_sherpa_button = QPushButton("添加", self.sherpa_tab)
        self.add_sherpa_button.clicked.connect(lambda: self._add_model("sherpa_onnx"))
        self.edit_sherpa_button = QPushButton("编辑", self.sherpa_tab)
        self.edit_sherpa_button.clicked.connect(lambda: self._edit_selected_model("sherpa_onnx"))
        self.edit_sherpa_button.setEnabled(False)  # 初始禁用
        self.delete_sherpa_button = QPushButton("删除", self.sherpa_tab)
        self.delete_sherpa_button.clicked.connect(lambda: self._delete_selected_model("sherpa_onnx"))
        self.delete_sherpa_button.setEnabled(False)  # 初始禁用

        sherpa_button_layout.addWidget(self.add_sherpa_button)
        sherpa_button_layout.addWidget(self.edit_sherpa_button)
        sherpa_button_layout.addWidget(self.delete_sherpa_button)
        sherpa_button_layout.addStretch()
        self.sherpa_layout.addLayout(sherpa_button_layout)

        # 连接选择变化信号
        self.sherpa_table.itemSelectionChanged.connect(lambda: self._update_button_states("sherpa_onnx"))

        # 添加ASR子选项卡
        self.asr_tab_widget.addTab(self.vosk_tab, "Vosk模型")
        self.asr_tab_widget.addTab(self.sherpa_tab, "Sherpa-ONNX模型")
        asr_layout.addWidget(self.asr_tab_widget)

        # ===== RTM模型选项卡（实时翻译） =====
        self.rtm_tab = QWidget()
        rtm_layout = QVBoxLayout(self.rtm_tab)

        # RTM子选项卡
        self.rtm_tab_widget = QTabWidget(self.rtm_tab)

        # --- Opus-MT模型选项卡 ---
        self.opus_tab = QWidget()
        self.opus_layout = QVBoxLayout(self.opus_tab)
        self.opus_table = QTableWidget(0, 5, self.opus_tab)
        self.opus_table.setHorizontalHeaderLabels(["ID", "名称", "路径", "源语言", "目标语言"])
        self.opus_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.opus_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.opus_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.opus_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.opus_table.customContextMenuRequested.connect(lambda pos: self._show_context_menu(pos, "opus"))
        self.opus_layout.addWidget(self.opus_table)

        # Opus按钮
        opus_button_layout = QHBoxLayout()
        self.add_opus_button = QPushButton("添加", self.opus_tab)
        self.add_opus_button.clicked.connect(lambda: self._add_model("opus"))
        opus_button_layout.addWidget(self.add_opus_button)
        opus_button_layout.addStretch()
        self.opus_layout.addLayout(opus_button_layout)

        # --- ArgosTranslate模型选项卡 ---
        self.argos_tab = QWidget()
        self.argos_layout = QVBoxLayout(self.argos_tab)
        self.argos_table = QTableWidget(0, 5, self.argos_tab)
        self.argos_table.setHorizontalHeaderLabels(["ID", "名称", "路径", "源语言", "目标语言"])
        self.argos_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.argos_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.argos_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.argos_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.argos_table.customContextMenuRequested.connect(lambda pos: self._show_context_menu(pos, "argos"))
        self.argos_layout.addWidget(self.argos_table)

        # Argos按钮
        argos_button_layout = QHBoxLayout()
        self.add_argos_button = QPushButton("添加", self.argos_tab)
        self.add_argos_button.clicked.connect(lambda: self._add_model("argos"))
        argos_button_layout.addWidget(self.add_argos_button)
        argos_button_layout.addStretch()
        self.argos_layout.addLayout(argos_button_layout)

        # 添加RTM子选项卡
        self.rtm_tab_widget.addTab(self.opus_tab, "Opus-MT模型")
        self.rtm_tab_widget.addTab(self.argos_tab, "ArgosTranslate模型")
        rtm_layout.addWidget(self.rtm_tab_widget)

        # 添加主选项卡
        self.main_tab_widget.addTab(self.asr_tab, "ASR模型（语音识别）")
        self.main_tab_widget.addTab(self.rtm_tab, "RTM模型（实时翻译）")
        layout.addWidget(self.main_tab_widget)

        # 创建底部按钮
        button_layout = QHBoxLayout()
        self.close_button = QPushButton("关闭", self)
        self.close_button.clicked.connect(self.accept)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)

    def _load_models(self):
        """加载模型配置"""
        try:
            # 加载ASR模型

            # 加载Vosk模型
            self.vosk_table.setRowCount(0)
            vosk_models = self.models_config.get("vosk", [])
            logger.info(f"加载Vosk模型列表: {vosk_models}")

            if not vosk_models:
                logger.warning("没有找到Vosk模型配置")

            for i, model in enumerate(vosk_models):
                logger.debug(f"处理Vosk模型 #{i+1}: {model}")
                self.vosk_table.insertRow(i)
                self.vosk_table.setItem(i, 0, QTableWidgetItem(model.get("id", "")))
                self.vosk_table.setItem(i, 1, QTableWidgetItem(model.get("name", "")))
                self.vosk_table.setItem(i, 2, QTableWidgetItem(model.get("path", "")))
                self.vosk_table.setItem(i, 3, QTableWidgetItem(str(model.get("sample_rate", 16000))))
                self.vosk_table.setItem(i, 4, QTableWidgetItem("是" if model.get("use_words", True) else "否"))
                logger.debug(f"已添加Vosk模型到表格: {model.get('id', '')}")

            # 加载Sherpa-ONNX模型
            self.sherpa_table.setRowCount(0)
            sherpa_models = self.models_config.get("sherpa_onnx", [])
            logger.info(f"加载Sherpa-ONNX模型列表: {sherpa_models}")

            if not sherpa_models:
                logger.warning("没有找到Sherpa-ONNX模型配置")

            for i, model in enumerate(sherpa_models):
                logger.debug(f"处理Sherpa-ONNX模型 #{i+1}: {model}")
                self.sherpa_table.insertRow(i)
                self.sherpa_table.setItem(i, 0, QTableWidgetItem(model.get("id", "")))
                self.sherpa_table.setItem(i, 1, QTableWidgetItem(model.get("name", "")))
                self.sherpa_table.setItem(i, 2, QTableWidgetItem(model.get("path", "")))
                self.sherpa_table.setItem(i, 3, QTableWidgetItem(str(model.get("sample_rate", 16000))))
                self.sherpa_table.setItem(i, 4, QTableWidgetItem("是" if model.get("use_persistent_stream", True) else "否"))
                logger.debug(f"已添加Sherpa-ONNX模型到表格: {model.get('id', '')}")

            # 加载RTM模型

            # 加载Opus-MT模型
            self.opus_table.setRowCount(0)
            opus_models = self.models_config.get("opus", [])
            logger.info(f"加载Opus-MT模型列表: {opus_models}")

            if not opus_models:
                logger.warning("没有找到Opus-MT模型配置")

            for i, model in enumerate(opus_models):
                logger.debug(f"处理Opus-MT模型 #{i+1}: {model}")
                self.opus_table.insertRow(i)
                self.opus_table.setItem(i, 0, QTableWidgetItem(model.get("id", "")))
                self.opus_table.setItem(i, 1, QTableWidgetItem(model.get("name", "")))
                self.opus_table.setItem(i, 2, QTableWidgetItem(model.get("path", "")))
                self.opus_table.setItem(i, 3, QTableWidgetItem(model.get("source_lang", "en")))
                self.opus_table.setItem(i, 4, QTableWidgetItem(model.get("target_lang", "zh")))
                logger.debug(f"已添加Opus-MT模型到表格: {model.get('id', '')}")

            # 加载ArgosTranslate模型
            self.argos_table.setRowCount(0)
            argos_models = self.models_config.get("argos", [])
            logger.info(f"加载ArgosTranslate模型列表: {argos_models}")

            if not argos_models:
                logger.warning("没有找到ArgosTranslate模型配置")

            for i, model in enumerate(argos_models):
                logger.debug(f"处理ArgosTranslate模型 #{i+1}: {model}")
                self.argos_table.insertRow(i)
                self.argos_table.setItem(i, 0, QTableWidgetItem(model.get("id", "")))
                self.argos_table.setItem(i, 1, QTableWidgetItem(model.get("name", "")))
                self.argos_table.setItem(i, 2, QTableWidgetItem(model.get("path", "")))
                self.argos_table.setItem(i, 3, QTableWidgetItem(model.get("source_lang", "en")))
                self.argos_table.setItem(i, 4, QTableWidgetItem(model.get("target_lang", "zh")))
                logger.debug(f"已添加ArgosTranslate模型到表格: {model.get('id', '')}")

            logger.info(f"已加载模型配置，ASR模型: Vosk {len(vosk_models)}个，Sherpa-ONNX {len(sherpa_models)}个；RTM模型: Opus-MT {len(opus_models)}个，ArgosTranslate {len(argos_models)}个")
        except Exception as e:
            logger.error(f"加载模型配置时出错: {str(e)}")
            logger.error(traceback.format_exc())

    def _show_context_menu(self, pos, model_type):
        """显示上下文菜单"""
        # 根据模型类型获取对应的表格
        if model_type == "vosk":
            table = self.vosk_table
        elif model_type == "sherpa_onnx":
            table = self.sherpa_table
        elif model_type == "opus":
            table = self.opus_table
        elif model_type == "argos":
            table = self.argos_table
        else:
            logger.error(f"未知的模型类型: {model_type}")
            return

        selected_rows = table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        model_id = table.item(row, 0).text()
        logger.info(f"显示{model_type}模型的上下文菜单: {model_id}")

        menu = QMenu(self)

        # 编辑操作
        edit_action = QAction("编辑", self)
        edit_action.triggered.connect(lambda: self._edit_model(model_type, row))
        menu.addAction(edit_action)

        # 删除操作
        delete_action = QAction("删除", self)
        delete_action.triggered.connect(lambda: self._delete_model(model_type, row))
        menu.addAction(delete_action)

        menu.exec_(QCursor.pos())

    def _add_model(self, model_type):
        """添加模型"""
        try:
            dialog = ModelConfigDialog(model_type, parent=self)
            if dialog.exec_() == QDialog.Accepted:
                model_data = dialog.get_model_data()
                model_id = model_data.get("id")

                logger.info(f"准备添加{model_type}模型: {model_id}")

                # 检查ID是否已存在
                models = self.models_config.get(model_type, [])
                for model in models:
                    if model.get("id") == model_id:
                        logger.warning(f"模型ID已存在: {model_id}")
                        QMessageBox.warning(self, "添加失败", f"ID '{model_id}' 已存在")
                        return

                # 添加模型到内部配置
                models.append(model_data)
                self.models_config[model_type] = models
                logger.info(f"已添加模型到内部配置: {model_id}")

                # 同时更新asr.models配置
                # 创建ASR模型配置格式
                asr_model_config = {
                    "path": model_data.get("path", ""),
                    "type": "standard",  # 默认类型
                    "enabled": True,
                    "config": {
                        "sample_rate": model_data.get("sample_rate", 16000)
                    }
                }

                # 根据模型类型添加特定配置
                if model_type == "vosk":
                    asr_model_config["config"]["use_words"] = model_data.get("use_words", True)
                elif model_type == "sherpa_onnx":
                    asr_model_config["config"]["use_persistent_stream"] = model_data.get("use_persistent_stream", True)

                # 获取当前ASR模型配置
                asr_models = config_manager.get_config("asr", "models", {})

                # 添加新模型
                asr_models[model_id] = asr_model_config

                # 保存ASR模型配置
                config_manager.set_config(asr_models, "asr", "models")
                config_manager.save_config("main")
                logger.info(f"已更新ASR模型配置: {model_id}")

                # 同时保存models配置（兼容旧版本）
                config_manager.update_and_save("models", self.models_config)
                logger.info(f"已更新models配置")

                # 重新加载模型列表
                self._load_models()

                # 发送信号
                self.models_changed.emit()

                logger.info(f"已完成添加{model_type}模型: {model_id}")
                QMessageBox.information(self, "添加成功", f"已成功添加模型: {model_id}")
        except Exception as e:
            logger.error(f"添加模型时出错: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "添加失败", f"添加模型时出错: {str(e)}")

    def _edit_model(self, model_type, row):
        """编辑模型"""
        try:
            # 获取模型数据
            # 根据模型类型获取对应的表格
            if model_type == "vosk":
                table = self.vosk_table
            elif model_type == "sherpa_onnx":
                table = self.sherpa_table
            elif model_type == "opus":
                table = self.opus_table
            elif model_type == "argos":
                table = self.argos_table
            else:
                logger.error(f"未知的模型类型: {model_type}")
                return

            model_id = table.item(row, 0).text()
            logger.info(f"准备编辑{model_type}模型: {model_id}")

            # 查找模型
            models = self.models_config.get(model_type, [])
            model_data = None
            for model in models:
                if model.get("id") == model_id:
                    model_data = model
                    break

            if not model_data:
                logger.warning(f"未找到模型: {model_id}")
                QMessageBox.warning(self, "编辑失败", f"未找到ID为 '{model_id}' 的模型")
                return

            # 打开编辑对话框
            dialog = ModelConfigDialog(model_type, model_data, parent=self)
            if dialog.exec_() == QDialog.Accepted:
                updated_data = dialog.get_model_data()
                logger.info(f"获取到更新后的模型数据: {updated_data}")

                # 更新内部模型配置
                for i, model in enumerate(models):
                    if model.get("id") == model_id:
                        models[i] = updated_data
                        break

                self.models_config[model_type] = models
                logger.info(f"已更新内部模型配置: {model_id}")

                # 根据模型类型更新配置
                if model_type in ["vosk", "sherpa_onnx"]:
                    # 更新ASR模型配置
                    asr_models = config_manager.get_config("asr", "models", {})

                    if model_id in asr_models:
                        # 更新现有ASR模型配置
                        asr_models[model_id]["path"] = updated_data.get("path", "")
                        asr_models[model_id]["config"]["sample_rate"] = updated_data.get("sample_rate", 16000)

                        # 根据模型类型更新特定配置
                        if model_type == "vosk":
                            asr_models[model_id]["config"]["use_words"] = updated_data.get("use_words", True)
                        elif model_type == "sherpa_onnx":
                            asr_models[model_id]["config"]["use_persistent_stream"] = updated_data.get("use_persistent_stream", True)

                        # 保存ASR模型配置
                        config_manager.set_config(asr_models, "asr", "models")
                        config_manager.save_config("main")
                        logger.info(f"已更新ASR模型配置: {model_id}")
                    else:
                        logger.warning(f"ASR模型配置中未找到模型: {model_id}")
                elif model_type in ["opus", "argos"]:
                    # 更新RTM模型配置
                    rtm_models = config_manager.get_config("rtm", "models", {})

                    if model_id in rtm_models:
                        # 更新现有RTM模型配置
                        rtm_models[model_id]["path"] = updated_data.get("path", "")
                        rtm_models[model_id]["source_lang"] = updated_data.get("source_lang", "en")
                        rtm_models[model_id]["target_lang"] = updated_data.get("target_lang", "zh")

                        # 保存RTM模型配置
                        config_manager.set_config(rtm_models, "rtm", "models")
                        config_manager.save_config("main")
                        logger.info(f"已更新RTM模型配置: {model_id}")
                    else:
                        logger.warning(f"RTM模型配置中未找到模型: {model_id}")

                # 同时保存models配置（兼容旧版本）
                config_manager.update_and_save("models", self.models_config)
                logger.info(f"已更新models配置")

                # 重新加载模型列表
                self._load_models()

                # 发送信号
                self.models_changed.emit()

                logger.info(f"已完成更新{model_type}模型: {model_id}")
                QMessageBox.information(self, "编辑成功", f"已成功更新模型: {model_id}")
        except Exception as e:
            logger.error(f"编辑模型时出错: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "编辑失败", f"编辑模型时出错: {str(e)}")

    def _delete_model(self, model_type, row):
        """删除模型"""
        try:
            # 获取模型数据
            # 根据模型类型获取对应的表格
            if model_type == "vosk":
                table = self.vosk_table
            elif model_type == "sherpa_onnx":
                table = self.sherpa_table
            elif model_type == "opus":
                table = self.opus_table
            elif model_type == "argos":
                table = self.argos_table
            else:
                logger.error(f"未知的模型类型: {model_type}")
                return

            model_id = table.item(row, 0).text()
            logger.info(f"准备删除{model_type}模型: {model_id}")

            # 确认删除
            result = QMessageBox.question(
                self,
                "确认删除",
                f"确定要删除模型 '{model_id}' 吗?",
                QMessageBox.Yes | QMessageBox.No
            )
            if result != QMessageBox.Yes:
                logger.info(f"用户取消删除模型: {model_id}")
                return

            # 从内部配置中删除模型
            models = self.models_config.get(model_type, [])
            for i, model in enumerate(models):
                if model.get("id") == model_id:
                    del models[i]
                    break

            self.models_config[model_type] = models
            logger.info(f"已从内部配置中删除模型: {model_id}")

            # 根据模型类型更新配置
            if model_type in ["vosk", "sherpa_onnx"]:
                # 从ASR模型配置中删除
                asr_models = config_manager.get_config("asr", "models", {})
                if model_id in asr_models:
                    del asr_models[model_id]
                    config_manager.set_config(asr_models, "asr", "models")
                    config_manager.save_config("main")
                    logger.info(f"已从ASR模型配置中删除模型: {model_id}")
                else:
                    logger.warning(f"ASR模型配置中未找到模型: {model_id}")
            elif model_type in ["opus", "argos"]:
                # 从RTM模型配置中删除
                rtm_models = config_manager.get_config("rtm", "models", {})
                if model_id in rtm_models:
                    del rtm_models[model_id]
                    config_manager.set_config(rtm_models, "rtm", "models")
                    config_manager.save_config("main")
                    logger.info(f"已从RTM模型配置中删除模型: {model_id}")
                else:
                    logger.warning(f"RTM模型配置中未找到模型: {model_id}")

            # 同时保存models配置（兼容旧版本）
            config_manager.update_and_save("models", self.models_config)
            logger.info(f"已更新models配置")

            # 重新加载模型列表
            self._load_models()

            # 发送信号
            self.models_changed.emit()

            logger.info(f"已完成删除{model_type}模型: {model_id}")
            QMessageBox.information(self, "删除成功", f"已成功删除模型: {model_id}")
        except Exception as e:
            logger.error(f"删除模型时出错: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "删除失败", f"删除模型时出错: {str(e)}")


class ModelConfigDialog(QDialog):
    """模型配置对话框"""

    def __init__(self, model_type, model_data=None, parent=None):
        """
        初始化模型配置对话框

        Args:
            model_type: 模型类型 (vosk 或 sherpa_onnx)
            model_data: 模型数据 (编辑模式)
            parent: 父窗口
        """
        super().__init__(parent)
        self.model_type = model_type
        self.model_data = model_data or {}
        self.is_edit_mode = bool(model_data)

        self._init_ui()
        self._load_data()

    def _init_ui(self):
        """初始化UI"""
        if self.is_edit_mode:
            self.setWindowTitle(f"编辑{self._get_model_type_display()}模型")
        else:
            self.setWindowTitle(f"添加{self._get_model_type_display()}模型")

        self.resize(500, 400)

        layout = QVBoxLayout(self)

        # 创建表单
        form_layout = QFormLayout()

        # 模型ID
        self.id_edit = QLineEdit(self)
        if self.is_edit_mode:
            self.id_edit.setEnabled(False)  # 编辑模式下不允许修改ID
        form_layout.addRow("模型ID:", self.id_edit)

        # 模型名称
        self.name_edit = QLineEdit(self)
        form_layout.addRow("模型名称:", self.name_edit)

        # 模型路径
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit(self)
        self.browse_button = QPushButton("浏览...", self)
        self.browse_button.clicked.connect(self._browse_path)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(self.browse_button)
        form_layout.addRow("模型路径:", path_layout)

        # 根据模型类型添加特定选项
        if self.model_type in ["vosk", "sherpa_onnx"]:
            # ASR模型特定选项

            # 采样率
            self.sample_rate_spin = QSpinBox(self)
            self.sample_rate_spin.setRange(8000, 48000)
            self.sample_rate_spin.setSingleStep(1000)
            self.sample_rate_spin.setValue(16000)
            form_layout.addRow("采样率:", self.sample_rate_spin)

            if self.model_type == "vosk":
                # 使用单词时间戳
                self.use_words_check = QCheckBox("启用", self)
                form_layout.addRow("使用单词时间戳:", self.use_words_check)
            elif self.model_type == "sherpa_onnx":
                # 使用持久流
                self.use_persistent_stream_check = QCheckBox("启用", self)
                form_layout.addRow("使用持久流:", self.use_persistent_stream_check)

        elif self.model_type in ["opus", "argos"]:
            # RTM模型特定选项

            # 源语言
            self.source_lang_edit = QLineEdit(self)
            self.source_lang_edit.setText("en")
            form_layout.addRow("源语言:", self.source_lang_edit)

            # 目标语言
            self.target_lang_edit = QLineEdit(self)
            self.target_lang_edit.setText("zh")
            form_layout.addRow("目标语言:", self.target_lang_edit)

        layout.addLayout(form_layout)

        # 创建按钮
        button_layout = QHBoxLayout()

        self.cancel_button = QPushButton("取消", self)
        self.cancel_button.clicked.connect(self.reject)

        self.save_button = QPushButton("保存", self)
        self.save_button.setDefault(True)
        self.save_button.clicked.connect(self._save_model)

        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)

        layout.addLayout(button_layout)

    def _load_data(self):
        """加载模型数据"""
        if not self.is_edit_mode:
            return

        self.id_edit.setText(self.model_data.get("id", ""))
        self.name_edit.setText(self.model_data.get("name", ""))
        self.path_edit.setText(self.model_data.get("path", ""))

        if self.model_type in ["vosk", "sherpa_onnx"]:
            # ASR模型特定选项
            self.sample_rate_spin.setValue(self.model_data.get("sample_rate", 16000))

            if self.model_type == "vosk":
                self.use_words_check.setChecked(self.model_data.get("use_words", True))
            elif self.model_type == "sherpa_onnx":
                self.use_persistent_stream_check.setChecked(self.model_data.get("use_persistent_stream", True))

        elif self.model_type in ["opus", "argos"]:
            # RTM模型特定选项
            self.source_lang_edit.setText(self.model_data.get("source_lang", "en"))
            self.target_lang_edit.setText(self.model_data.get("target_lang", "zh"))

    def _browse_path(self):
        """浏览模型路径"""
        logger.info("打开模型路径选择对话框")
        try:
            # 使用更安全的方式打开文件对话框
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog  # 避免使用本地对话框，防止卡死
            # 设置文件对话框不阻塞UI线程
            options |= QFileDialog.ReadOnly  # 减少文件系统操作

            # 处理当前路径，避免无效路径导致卡死
            current_path = self.path_edit.text()
            # 验证路径有效性
            if current_path and not os.path.exists(current_path):
                logger.warning(f"当前路径无效: {current_path}，将使用用户主目录")
                current_path = ""

            start_dir = current_path if current_path else os.path.expanduser("~")
            logger.debug(f"模型路径选择对话框起始目录: {start_dir}")

            # 确保UI响应
            QApplication.processEvents()

            # 创建并显示文件对话框
            path = QFileDialog.getExistingDirectory(
                self,
                "选择模型目录",
                start_dir,
                options=options
            )

            # 确保UI响应
            QApplication.processEvents()

            if path:
                logger.info(f"用户选择的模型路径: {path}")
                self.path_edit.setText(path)
            else:
                logger.warning("用户取消选择模型路径")

        except Exception as e:
            logger.error(f"浏览文件夹时出错: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.warning(self, "错误", f"浏览文件夹时出错: {str(e)}")

    def _save_model(self):
        """保存模型配置"""
        # 验证输入
        model_id = self.id_edit.text().strip()
        if not model_id:
            QMessageBox.warning(self, "输入错误", "模型ID不能为空")
            return

        # 验证ID格式
        if not model_id.isalnum() and "_" not in model_id:
            QMessageBox.warning(self, "输入错误", "模型ID只能包含字母、数字和下划线")
            return

        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "输入错误", "模型名称不能为空")
            return

        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "输入错误", "模型路径不能为空")
            return

        # 检查路径是否存在
        if not os.path.exists(path):
            result = QMessageBox.question(
                self,
                "路径不存在",
                f"路径 '{path}' 不存在，是否继续?",
                QMessageBox.Yes | QMessageBox.No
            )
            if result == QMessageBox.No:
                return

        # 创建模型数据
        model_data = {
            "id": model_id,
            "name": name,
            "path": path
        }

        # 添加特定选项
        if self.model_type in ["vosk", "sherpa_onnx"]:
            # ASR模型特定选项
            model_data["sample_rate"] = self.sample_rate_spin.value()

            if self.model_type == "vosk":
                model_data["use_words"] = self.use_words_check.isChecked()
            elif self.model_type == "sherpa_onnx":
                model_data["use_persistent_stream"] = self.use_persistent_stream_check.isChecked()

        elif self.model_type in ["opus", "argos"]:
            # RTM模型特定选项
            model_data["source_lang"] = self.source_lang_edit.text().strip()
            model_data["target_lang"] = self.target_lang_edit.text().strip()

            # 验证语言代码
            if not model_data["source_lang"]:
                QMessageBox.warning(self, "输入错误", "源语言不能为空")
                return

            if not model_data["target_lang"]:
                QMessageBox.warning(self, "输入错误", "目标语言不能为空")
                return

        # 返回模型数据
        self.model_data = model_data
        self.accept()

    def get_model_data(self):
        """获取模型数据"""
        return self.model_data

    def _get_model_type_display(self):
        """获取模型类型显示名称"""
        if self.model_type == "vosk":
            return "Vosk"
        elif self.model_type == "sherpa_onnx":
            return "Sherpa-ONNX"
        elif self.model_type == "opus":
            return "Opus-MT"
        elif self.model_type == "argos":
            return "ArgosTranslate"
        return self.model_type
