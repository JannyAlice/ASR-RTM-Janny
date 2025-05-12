#!/usr/bin/env python3
"""
查找PySide6导入的工具
"""
import os
import re
import sys
from pathlib import Path

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# PySide6导入模式
PYSIDE_IMPORT_PATTERNS = [
    r'from\s+PySide6\.',
    r'import\s+PySide6\.',
    r'import\s+PySide6\s+'
]

def scan_file(file_path):
    """
    扫描文件中的PySide6导入
    
    Args:
        file_path: 文件路径
        
    Returns:
        list: 包含PySide6导入的行列表
    """
    result = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # 检查每一行
        for i, line in enumerate(lines):
            for pattern in PYSIDE_IMPORT_PATTERNS:
                if re.search(pattern, line):
                    result.append((i + 1, line.strip()))
                    
        return result
    except Exception as e:
        print(f"扫描文件 {file_path} 时出错: {str(e)}")
        return result

def scan_directory(directory, exclude_dirs=None):
    """
    扫描目录中的所有Python文件
    
    Args:
        directory: 目录路径
        exclude_dirs: 要排除的目录列表
        
    Returns:
        dict: 包含扫描结果的字典
    """
    if exclude_dirs is None:
        exclude_dirs = []
        
    result = {}
    
    for root, dirs, files in os.walk(directory):
        # 排除指定目录
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                # 扫描文件
                file_result = scan_file(file_path)
                
                # 如果有PySide6导入，添加到结果中
                if file_result:
                    rel_path = os.path.relpath(file_path, directory)
                    result[rel_path] = file_result
                    
    return result

def main():
    """主函数"""
    # 扫描项目目录
    exclude_dirs = ['venv', 'env', '__pycache__', 'build', 'dist', 'logs']
    result = scan_directory(str(project_root), exclude_dirs)
    
    # 打印结果
    if result:
        print("\n===== PySide6导入扫描结果 =====")
        for file_path, imports in result.items():
            print(f"\n文件: {file_path}")
            for line_num, line in imports:
                print(f"  行 {line_num}: {line}")
        print("\n===== 扫描结束 =====")
    else:
        print("未找到PySide6导入")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
