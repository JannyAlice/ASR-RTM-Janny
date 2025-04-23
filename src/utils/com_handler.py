"""
COM处理模块
负责Windows COM组件的初始化和管理
"""
import os
import pythoncom
import threading
from typing import Optional

# 设置环境变量，防止COM初始化冲突
os.environ["PYTHONCOM_INITIALIZE"] = "0"

class ComHandler:
    """COM处理类"""
    
    _instance = None
    _lock = threading.Lock()
    _initialized_threads = set()
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(ComHandler, cls).__new__(cls)
        return cls._instance
    
    def initialize_com(self, threading_model: Optional[int] = None) -> bool:
        """
        初始化COM
        
        Args:
            threading_model: COM线程模型，默认为None（自动选择）
                可选值：pythoncom.COINIT_APARTMENTTHREADED 或 pythoncom.COINIT_MULTITHREADED
                
        Returns:
            bool: 初始化是否成功
        """
        # 获取当前线程ID
        thread_id = threading.get_ident()
        
        # 检查当前线程是否已初始化
        with self._lock:
            if thread_id in self._initialized_threads:
                print(f"线程 {thread_id} 已初始化COM")
                return True
        
        try:
            # 如果指定了线程模型，使用指定的模型
            if threading_model is not None:
                pythoncom.CoInitializeEx(threading_model)
                model_name = "单线程" if threading_model == pythoncom.COINIT_APARTMENTTHREADED else "多线程"
                print(f"COM初始化成功（{model_name}模式）")
            else:
                # 尝试使用多线程模式
                try:
                    pythoncom.CoInitializeEx(pythoncom.COINIT_MULTITHREADED)
                    print("COM初始化成功（多线程模式）")
                except Exception:
                    # 如果失败，尝试使用单线程模式
                    try:
                        pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)
                        print("COM初始化成功（单线程模式）")
                    except Exception as e:
                        print(f"COM初始化失败: {e}")
                        return False
            
            # 记录已初始化的线程
            with self._lock:
                self._initialized_threads.add(thread_id)
            
            return True
            
        except Exception as e:
            # 如果错误是因为COM已经初始化，这不是真正的错误
            error_msg = str(e).lower()
            if "already initialized" in error_msg or "cannot change thread mode" in error_msg:
                print("COM已经初始化，继续执行")
                
                # 记录已初始化的线程
                with self._lock:
                    self._initialized_threads.add(thread_id)
                
                return True
            else:
                print(f"COM初始化错误: {e}")
                return False
    
    def uninitialize_com(self) -> bool:
        """
        释放COM
        
        Returns:
            bool: 释放是否成功
        """
        # 获取当前线程ID
        thread_id = threading.get_ident()
        
        try:
            # 检查当前线程是否已初始化
            with self._lock:
                if thread_id not in self._initialized_threads:
                    print(f"线程 {thread_id} 未初始化COM，无需释放")
                    return True
            
            # 释放COM
            pythoncom.CoUninitialize()
            print("COM已释放")
            
            # 移除已初始化的线程记录
            with self._lock:
                self._initialized_threads.remove(thread_id)
            
            return True
            
        except Exception as e:
            print(f"COM释放错误: {e}")
            return False

# 创建全局实例
com_handler = ComHandler()
