import os
import sys
import platform
from typing import Optional, List, Dict, Any
from datetime import datetime


class CommonUtils:
    """通用工具类"""
    
    @staticmethod
    def get_system_info() -> Dict[str, str]:
        """获取系统信息
        
        Returns:
            Dict[str, str]: 系统信息字典
        """
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "platform_release": platform.release(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version
        }
        
    @staticmethod
    def format_timestamp(timestamp: Optional[float] = None) -> str:
        """格式化时间戳
        
        Args:
            timestamp: 时间戳，如果为 None 则使用当前时间
            
        Returns:
            str: 格式化后的时间字符串
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
    @staticmethod
    def ensure_dir(directory: str) -> bool:
        """确保目录存在
        
        Args:
            directory: 目录路径
            
        Returns:
            bool: 是否成功
        """
        try:
            os.makedirs(directory, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directory {directory}: {str(e)}")
            return False
            
    @staticmethod
    def get_file_size(file_path: str) -> Optional[int]:
        """获取文件大小
        
        Args:
            file_path: 文件路径
            
        Returns:
            int: 文件大小（字节），如果文件不存在则返回 None
        """
        try:
            return os.path.getsize(file_path)
        except Exception:
            return None
            
    @staticmethod
    def format_file_size(size: int) -> str:
        """格式化文件大小
        
        Args:
            size: 文件大小（字节）
            
        Returns:
            str: 格式化后的文件大小
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} PB"
        
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """获取文件扩展名
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 文件扩展名（小写）
        """
        return os.path.splitext(file_path)[1].lower()
        
    @staticmethod
    def is_audio_file(file_path: str) -> bool:
        """判断是否为音频文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否为音频文件
        """
        audio_extensions = {'.wav', '.mp3', '.ogg', '.flac', '.m4a'}
        return CommonUtils.get_file_extension(file_path) in audio_extensions
        
    @staticmethod
    def is_video_file(file_path: str) -> bool:
        """判断是否为视频文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否为视频文件
        """
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv'}
        return CommonUtils.get_file_extension(file_path) in video_extensions
        
    @staticmethod
    def get_file_list(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
        """获取目录中的文件列表
        
        Args:
            directory: 目录路径
            extensions: 文件扩展名列表，如果为 None 则返回所有文件
            
        Returns:
            List[str]: 文件路径列表
        """
        try:
            files = []
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    if extensions is None or CommonUtils.get_file_extension(file) in extensions:
                        files.append(file_path)
            return files
        except Exception as e:
            print(f"Error getting file list from {directory}: {str(e)}")
            return []
