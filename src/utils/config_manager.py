#!/usr/bin/env python3
"""
配置管理模块
负责加载和管理配置信息，提供统一的配置访问接口
"""
import os
import json
import logging
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple

logger = logging.getLogger(__name__)

class ConfigManager:
    """配置管理类，单例模式"""
    _instance = None
    _config = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化配置管理器"""
        if self._initialized:
            return

        # 配置文件路径
        self._config_path = os.path.join('config', 'config.json')
        self._plugins_path = os.path.join('config', 'plugins.json')
        self._ui_config_path = os.path.join('config', 'ui_config.json')
        self._translation_config_path = os.path.join('config', 'translation_config.json')

        # 备份目录
        self._backup_dir = os.path.join('config', 'backups')
        os.makedirs(self._backup_dir, exist_ok=True)

        # 初始化配置
        self._config = {}
        self._initialized = True

    @property
    def config(self) -> Dict[str, Any]:
        """获取配置"""
        return self._config

    def load_config(self) -> Dict[str, Any]:
        """加载所有配置文件"""
        try:
            # 加载主配置
            if os.path.exists(self._config_path):
                logger.debug(f"尝试加载主配置文件: {self._config_path}")
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
                logger.info("主配置文件加载成功")
            else:
                logger.warning(f"主配置文件不存在: {self._config_path}")
                self._config = {}

            # 加载插件配置
            if os.path.exists(self._plugins_path):
                logger.debug(f"尝试加载插件配置文件: {self._plugins_path}")
                with open(self._plugins_path, 'r', encoding='utf-8') as f:
                    self._config['plugins'] = json.load(f)
                logger.info("插件配置文件加载成功")

            # 加载UI配置
            if os.path.exists(self._ui_config_path):
                logger.debug(f"尝试加载UI配置文件: {self._ui_config_path}")
                with open(self._ui_config_path, 'r', encoding='utf-8') as f:
                    self._config['ui'] = json.load(f)
                logger.info("UI配置文件加载成功")

            # 加载翻译配置
            if os.path.exists(self._translation_config_path):
                logger.debug(f"尝试加载翻译配置文件: {self._translation_config_path}")
                with open(self._translation_config_path, 'r', encoding='utf-8') as f:
                    self._config['translation'] = json.load(f)
                logger.info("翻译配置文件加载成功")

            # 验证配置
            if not self.validate_config(self._config):
                logger.warning("配置验证失败，使用默认配置")
                self._init_default_config()

            return self._config

        except json.JSONDecodeError as e:
            logger.error(f"配置文件格式错误: {str(e)}")
            logger.warning("使用默认配置")
            self._init_default_config()
            return self._config
        except Exception as e:
            logger.error(f"加载配置文件时发生错误: {str(e)}")
            logger.warning("使用默认配置")
            self._init_default_config()
            return self._config

    def _init_default_config(self):
        """
        初始化默认配置
        注意：所有模型路径、采样率、use_words 等参数均应通过 config/models.json 配置，禁止硬编码和相对路径。
        """
        self._config = {
            "app": {
                "name": "实时字幕",
                "version": "1.0.0"
            },
            "asr": {
                "default_model": "vosk_small",
                "models": {
                    # 仅保留模型ID、类型、参数说明，实际路径等请在 config/models.json 配置
                    "vosk_small": {
                        "name": "VOSK Small Model",
                        "path": "",  # 路径请在 config/models.json 配置
                        "type": "vosk",
                        "config": {
                            "sample_rate": None,  # 采样率请在 config/models.json 配置
                            "use_words": None     # use_words 请在 config/models.json 配置
                        }
                    }
                }
            },
            "window": {
                "opacity": 0.9,
                "always_on_top": True,
                "font_size": 12
            },
            "transcription": {
                "default_model": "vosk_small",
                "save_transcripts": True,
                "transcripts_dir": "transcripts"
            }
        }
        logger.info("已初始化默认配置（模型路径等请通过 config/models.json 配置）")

    def save_config(self, section: Optional[str] = None) -> bool:
        """保存配置

        Args:
            section: 要保存的配置部分，None表示保存所有配置

        Returns:
            bool: 保存是否成功
        """
        try:
            # 创建备份
            self._create_backup()

            if section is None or section == 'main':
                # 保存主配置
                main_config = {k: v for k, v in self._config.items()
                              if k not in ['plugins', 'ui', 'translation']}
                with open(self._config_path, 'w', encoding='utf-8') as f:
                    json.dump(main_config, f, indent=4, ensure_ascii=False)
                logger.info(f"已保存主配置: {self._config_path}")

            if section is None or section == 'plugins':
                # 保存插件配置
                with open(self._plugins_path, 'w', encoding='utf-8') as f:
                    json.dump(self._config.get('plugins', {}), f, indent=4, ensure_ascii=False)
                logger.info(f"已保存插件配置: {self._plugins_path}")

            if section is None or section == 'ui':
                # 保存UI配置
                with open(self._ui_config_path, 'w', encoding='utf-8') as f:
                    json.dump(self._config.get('ui', {}), f, indent=4, ensure_ascii=False)
                logger.info(f"已保存UI配置: {self._ui_config_path}")

            if section is None or section == 'translation':
                # 保存翻译配置
                with open(self._translation_config_path, 'w', encoding='utf-8') as f:
                    json.dump(self._config.get('translation', {}), f, indent=4, ensure_ascii=False)
                logger.info(f"已保存翻译配置: {self._translation_config_path}")

            return True
        except Exception as e:
            logger.error(f"保存配置失败: {str(e)}")
            return False

    def _create_backup(self):
        """创建配置文件备份"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 备份主配置
            if os.path.exists(self._config_path):
                backup_path = os.path.join(self._backup_dir, f"config_{timestamp}.json")
                shutil.copy2(self._config_path, backup_path)

            # 备份插件配置
            if os.path.exists(self._plugins_path):
                backup_path = os.path.join(self._backup_dir, f"plugins_{timestamp}.json")
                shutil.copy2(self._plugins_path, backup_path)

            # 备份UI配置
            if os.path.exists(self._ui_config_path):
                backup_path = os.path.join(self._backup_dir, f"ui_config_{timestamp}.json")
                shutil.copy2(self._ui_config_path, backup_path)

            # 备份翻译配置
            if os.path.exists(self._translation_config_path):
                backup_path = os.path.join(self._backup_dir, f"translation_config_{timestamp}.json")
                shutil.copy2(self._translation_config_path, backup_path)

            # 清理旧备份
            self._cleanup_old_backups()

            logger.debug("已创建配置文件备份")
        except Exception as e:
            logger.warning(f"创建配置文件备份失败: {str(e)}")

    def _cleanup_old_backups(self, max_backups: int = 10):
        """清理旧备份文件

        Args:
            max_backups: 每种配置文件保留的最大备份数量
        """
        try:
            # 按类型分组备份文件
            backup_files = {}
            for filename in os.listdir(self._backup_dir):
                if not filename.endswith('.json'):
                    continue

                file_type = filename.split('_')[0]
                if file_type not in backup_files:
                    backup_files[file_type] = []

                backup_files[file_type].append(os.path.join(self._backup_dir, filename))

            # 清理每种类型的旧备份
            for file_type, files in backup_files.items():
                if len(files) <= max_backups:
                    continue

                # 按修改时间排序
                files.sort(key=lambda x: os.path.getmtime(x))

                # 删除最旧的文件
                for file_path in files[:-max_backups]:
                    os.remove(file_path)
                    logger.debug(f"已删除旧备份文件: {file_path}")

        except Exception as e:
            logger.warning(f"清理旧备份文件失败: {str(e)}")

    def get_config(self, *keys, default=None) -> Any:
        """
        获取配置值

        Args:
            *keys: 配置键路径，例如 'asr', 'models', 'vosk_small'
            default: 默认值，如果配置不存在则返回此值

        Returns:
            Any: 配置值或默认值
        """
        if not keys:
            return default

        try:
            # 如果只有一个键且包含点号，按点号分割
            if len(keys) == 1 and isinstance(keys[0], str) and '.' in keys[0]:
                keys = keys[0].split('.')

            value = self._config
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            return value
        except Exception:
            return default

    def set_config(self, value: Any, *keys) -> bool:
        """
        设置配置值

        Args:
            value: 要设置的值
            *keys: 配置键路径

        Returns:
            bool: 设置是否成功
        """
        if not keys:
            return False

        try:
            # 如果只有一个键且包含点号，按点号分割
            if len(keys) == 1 and isinstance(keys[0], str) and '.' in keys[0]:
                keys = keys[0].split('.')

            # 递归创建嵌套字典
            config = self._config
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                elif not isinstance(config[key], dict):
                    config[key] = {}
                config = config[key]

            # 设置最终值
            config[keys[-1]] = value
            return True
        except Exception as e:
            logger.error(f"设置配置值失败: {str(e)}")
            return False

    def get_ui_config(self, *keys, default=None) -> Any:
        """
        获取UI配置

        Args:
            *keys: 配置键路径
            default: 默认值，如果配置不存在则返回此值

        Returns:
            Any: 配置值或默认值
        """
        # 构建完整的键路径
        ui_keys = ['ui']
        ui_keys.extend(keys)
        return self.get_config(*ui_keys, default=default)

    def set_ui_config(self, value: Any, *keys) -> bool:
        """
        设置UI配置

        Args:
            value: 要设置的值
            *keys: 配置键路径

        Returns:
            bool: 设置是否成功
        """
        # 构建完整的键路径
        ui_keys = ['ui']
        ui_keys.extend(keys)
        return self.set_config(value, *ui_keys)

    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        获取指定模型的配置

        Args:
            model_name: 模型名称

        Returns:
            Optional[Dict[str, Any]]: 模型配置或None
        """
        return self.get_config('asr', 'models', model_name)

    def get_plugin_config(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """
        获取指定插件的配置

        Args:
            plugin_id: 插件ID

        Returns:
            Optional[Dict[str, Any]]: 插件配置或None
        """
        # 首先尝试从plugins.plugins获取
        plugin_config = self.get_config('plugins', 'plugins', plugin_id)
        if plugin_config:
            return plugin_config

        # 然后尝试从plugins直接获取
        return self.get_config('plugins', plugin_id)

    def get_window_config(self) -> Dict[str, Any]:
        """
        获取窗口配置

        Returns:
            Dict[str, Any]: 窗口配置
        """
        return self.get_config('window', default={})

    def update_window_config(self, config: Dict[str, Any]) -> bool:
        """
        更新窗口配置

        Args:
            config: 新的窗口配置

        Returns:
            bool: 更新是否成功
        """
        if self.set_config(config, 'window'):
            return self.save_config('main')
        return False

    def update_and_save(self, section: str, config: Dict[str, Any]) -> bool:
        """
        更新指定部分的配置并保存

        Args:
            section: 配置部分名称
            config: 新的配置值

        Returns:
            bool: 是否更新成功
        """
        try:
            # 更新配置
            self._config[section] = config

            # 保存配置
            return self.save_config(section)
        except Exception as e:
            logger.error(f"更新并保存配置失败: {str(e)}")
            return False

    def get_default_model(self) -> str:
        """
        获取默认ASR模型名称

        Returns:
            str: 默认模型名称
        """
        # 首先尝试从asr.default_model获取
        default_model = self.get_config('asr', 'default_model')
        if default_model:
            return default_model

        # 然后尝试从transcription.default_model获取
        default_model = self.get_config('transcription', 'default_model')
        if default_model:
            return default_model

        # 最后返回默认值
        return 'vosk_small'

    def validate_model_files(self, model_path: str, model_type: str = "sherpa_onnx") -> bool:
        """
        验证模型文件完整性

        Args:
            model_path: 模型路径
            model_type: 模型类型，支持 "sherpa_onnx" 和 "vosk"

        Returns:
            bool: 验证是否通过
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"模型路径不存在: {model_path}")
                return False

            if model_type.startswith("sherpa"):
                # Sherpa-ONNX 模型文件验证
                required_files = [
                    "encoder.onnx",
                    "decoder.onnx",
                    "joiner.onnx",
                    "tokens.txt"
                ]

                for file in required_files:
                    file_path = os.path.join(model_path, file)
                    if not os.path.exists(file_path):
                        logger.error(f"缺少Sherpa-ONNX模型文件: {file_path}")
                        return False

            elif model_type == "vosk":
                # Vosk 模型文件验证
                # Vosk 模型只需要检查目录是否存在，以及是否包含 am 和 conf 文件夹
                required_dirs = ["am", "conf"]
                for dir_name in required_dirs:
                    dir_path = os.path.join(model_path, dir_name)
                    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
                        logger.error(f"缺少Vosk模型目录: {dir_path}")
                        return False

            else:
                logger.warning(f"未知的模型类型: {model_type}，跳过文件验证")
                return True

            logger.info(f"{model_type}模型文件验证通过: {model_path}")
            return True

        except Exception as e:
            logger.error(f"验证模型文件时发生错误: {str(e)}")
            return False

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        验证配置文件的完整性和正确性

        Args:
            config: 配置字典

        Returns:
            bool: 验证是否通过
        """
        # 最小必要配置
        if not config:
            logger.error("配置为空")
            return False

        # 检查必要的顶级键
        required_top_keys = ['app', 'asr']
        for key in required_top_keys:
            if key not in config:
                logger.error(f"缺少必要的配置键: {key}")
                return False

        # 检查ASR配置
        if 'models' not in config.get('asr', {}):
            logger.error("缺少ASR模型配置")
            return False

        # 检查是否至少有一个模型配置
        if not config.get('asr', {}).get('models', {}):
            logger.error("没有配置任何ASR模型")
            return False

        logger.info("配置验证通过")
        return True

    def get_all_config(self) -> Dict[str, Any]:
        """
        获取所有配置

        Returns:
            Dict[str, Any]: 完整的配置字典
        """
        return self._config

    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有ASR模型配置

        Returns:
            Dict[str, Dict[str, Any]]: 所有模型配置
        """
        return self.get_config('asr', 'models', default={})

    def get_all_plugins(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有插件配置

        Returns:
            Dict[str, Dict[str, Any]]: 所有插件配置
        """
        # 首先尝试从plugins.plugins获取
        plugins = self.get_config('plugins', 'plugins', default={})
        if plugins:
            return plugins

        # 然后尝试从plugins直接获取
        return self.get_config('plugins', default={})

    def register_model(self, model_id: str, model_config: Dict[str, Any]) -> bool:
        """
        注册ASR模型

        Args:
            model_id: 模型ID
            model_config: 模型配置

        Returns:
            bool: 注册是否成功
        """
        if not self.set_config(model_config, 'asr', 'models', model_id):
            return False

        return self.save_config('main')

    def register_plugin(self, plugin_id: str, plugin_config: Dict[str, Any]) -> bool:
        """
        注册插件

        Args:
            plugin_id: 插件ID
            plugin_config: 插件配置

        Returns:
            bool: 注册是否成功
        """
        # 检查plugins.plugins是否存在
        if self.get_config('plugins', 'plugins') is not None:
            if not self.set_config(plugin_config, 'plugins', 'plugins', plugin_id):
                return False
        else:
            if not self.set_config(plugin_config, 'plugins', plugin_id):
                return False

        return self.save_config('plugins')

# 创建全局配置管理器实例
config_manager = ConfigManager()

# 导出
__all__ = [
    'ConfigManager',
    'config_manager'
]
