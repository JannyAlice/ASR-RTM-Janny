import os
import sys
import pytest
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from main import main
from src.utils.config_manager import config_manager

def test_main_startup():
    """测试主程序启动流程"""
    try:
        # 获取默认模型
        default_model = config_manager.get_default_model()
        assert default_model == "sherpa_0626"
        
        # 检查模型路径
        model_config = config_manager.config['asr']['models'][default_model]
        assert model_config['enabled'] == True
        
        # 检查模型文件
        model_path = model_config['path']
        assert os.path.exists(model_path)
        
        required_files = [
            'encoder-epoch-99-avg-1-chunk-16-left-128.onnx',
            'decoder-epoch-99-avg-1-chunk-16-left-128.onnx',
            'joiner-epoch-99-avg-1-chunk-16-left-128.onnx',
            'tokens.txt'
        ]
        
        for file in required_files:
            file_path = os.path.join(model_path, file)
            assert os.path.exists(file_path)
            
    except Exception as e:
        pytest.fail(f"测试失败: {str(e)}")