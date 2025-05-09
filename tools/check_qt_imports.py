#!/usr/bin/env python3
"""
Qt导入检查工具
用于检查项目中的Qt导入是否一致
"""
import os
import re
import sys
from pathlib import Path

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入日志模块
from src.utils.logger import get_logger
logger = get_logger("qt_import_checker")

# Qt导入模式
QT_IMPORT_PATTERNS = [
    r'from\s+PyQt5\.',
    r'from\s+PySide6\.',
    r'import\s+PyQt5\.',
    r'import\s+PySide6\.',
    r'import\s+PyQt5\s+',
    r'import\s+PySide6\s+'
]

def scan_file(file_path):
    """
    扫描文件中的Qt导入
    
    Args:
        file_path: 文件路径
        
    Returns:
        dict: 包含PyQt5和PySide6导入计数的字典
    """
    result = {
        'PyQt5': 0,
        'PySide6': 0,
        'imports': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 检查每种导入模式
        for pattern in QT_IMPORT_PATTERNS:
            matches = re.findall(pattern, content)
            for match in matches:
                if 'PyQt5' in match:
                    result['PyQt5'] += 1
                    result['imports'].append(match.strip())
                elif 'PySide6' in match:
                    result['PySide6'] += 1
                    result['imports'].append(match.strip())
                    
        return result
    except Exception as e:
        logger.error(f"扫描文件 {file_path} 时出错: {str(e)}")
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
        
    result = {
        'total_files': 0,
        'files_with_qt': 0,
        'pyqt5_count': 0,
        'pyside6_count': 0,
        'mixed_files': 0,
        'details': {}
    }
    
    for root, dirs, files in os.walk(directory):
        # 排除指定目录
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                result['total_files'] += 1
                
                # 扫描文件
                file_result = scan_file(file_path)
                
                # 更新结果
                if file_result['PyQt5'] > 0 or file_result['PySide6'] > 0:
                    result['files_with_qt'] += 1
                    result['pyqt5_count'] += file_result['PyQt5']
                    result['pyside6_count'] += file_result['PySide6']
                    
                    # 检查是否混合使用
                    if file_result['PyQt5'] > 0 and file_result['PySide6'] > 0:
                        result['mixed_files'] += 1
                        
                    # 添加详细信息
                    rel_path = os.path.relpath(file_path, directory)
                    result['details'][rel_path] = file_result
                    
    return result

def print_report(result):
    """
    打印扫描报告
    
    Args:
        result: 扫描结果
    """
    print("\n===== Qt导入扫描报告 =====")
    print(f"总文件数: {result['total_files']}")
    print(f"包含Qt导入的文件数: {result['files_with_qt']}")
    print(f"PyQt5导入计数: {result['pyqt5_count']}")
    print(f"PySide6导入计数: {result['pyside6_count']}")
    print(f"混合使用Qt的文件数: {result['mixed_files']}")
    
    if result['mixed_files'] > 0:
        print("\n混合使用Qt的文件:")
        for file_path, details in result['details'].items():
            if details['PyQt5'] > 0 and details['PySide6'] > 0:
                print(f"  {file_path}:")
                for imp in details['imports']:
                    print(f"    {imp}")
                    
    print("\n===== 报告结束 =====")

def main():
    """主函数"""
    # 扫描项目目录
    exclude_dirs = ['venv', 'env', '__pycache__', 'build', 'dist', 'logs']
    result = scan_directory(str(project_root), exclude_dirs)
    
    # 打印报告
    print_report(result)
    
    # 如果有混合使用Qt的文件，返回非零退出码
    return 1 if result['mixed_files'] > 0 else 0

if __name__ == "__main__":
    sys.exit(main())
