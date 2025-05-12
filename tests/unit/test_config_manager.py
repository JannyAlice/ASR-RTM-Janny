import pytest
from src.utils.config_manager import ConfigManager

def test_validate_config():
    config = {
        'model_path': '/path/to/model',
        'audio_device': 0,
    }
    manager = ConfigManager()
    assert manager.validate_config(config) == True

def test_invalid_config():
    config = {}
    manager = ConfigManager() 
    assert manager.validate_config(config) == False