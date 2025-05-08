"""测试配置文件"""
import os
import sys
import pytest
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入所需的组件
from src.utils.config_manager import config_manager
from src.core.asr.model_manager import ASRModelManager
from src.core.translation import TranslationManager

@pytest.fixture
def config_manager():
    """配置管理器实例"""
    return config_manager

@pytest.fixture
def model_manager():
    """ASR模型管理器实例"""
    return ASRModelManager()

@pytest.fixture
def translation_manager():
    """创建翻译管理器实例"""
    # 使用测试专用的模型目录
    config = {
        'opus_mt': {
            'model_dir': os.path.join('tests', 'models', 'translation', 'opus-mt', 'en-zh')
        },
        'argos': {
            'model_dir': os.path.join('tests', 'models', 'translation', 'argos')
        }
    }
    return TranslationManager(config)

@pytest.fixture
def test_text():
    """测试用的文本"""
    return "Hello, how are you?"

@pytest.fixture
def test_audio_path(tmp_path):
    """创建测试音频文件路径"""
    return tmp_path / "test.wav"

@pytest.fixture
def test_model_path(tmp_path):
    """创建测试模型目录路径"""
    model_dir = tmp_path / "models" / "asr" / "test_model"
    model_dir.mkdir(parents=True)
    return model_dir

@pytest.fixture
def test_data_dir():
    """测试数据目录"""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir