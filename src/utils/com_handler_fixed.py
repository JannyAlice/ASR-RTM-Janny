"""
修复后的 COM 处理模块
"""

def is_initialized(self) -> bool:
    """
    检查当前线程是否已初始化 COM
    
    Returns:
        bool: 当前线程是否已初始化 COM
    """
    thread_id = threading.get_ident()
    
    with self._lock:
        return thread_id in self._initialized_threads
