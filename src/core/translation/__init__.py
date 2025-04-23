"""翻译模块

此模块提供各种翻译引擎的实现，包括 OPUS-MT 和 ArgosTranslate。
"""

from .opus_engine import OpusMTEngine
from .argos_engine import ArgosEngine
from .manager import TranslationManager

__all__ = ['OpusMTEngine', 'ArgosEngine', 'TranslationManager']
