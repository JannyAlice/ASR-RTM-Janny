#!/usr/bin/env python3
"""
检查 vosk 模型路径是否存在
"""
import os
import sys

# 设置 vosk 模型路径
MODEL_PATH = "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\models\\asr\\vosk\\vosk-model-small-en-us-0.15"

def main():
    """主函数"""
    print(f"检查模型路径: {MODEL_PATH}")
    
    if os.path.exists(MODEL_PATH):
        print(f"✓ 模型路径存在")
        
        # 检查目录内容
        files = os.listdir(MODEL_PATH)
        print(f"目录中的文件数量: {len(files)}")
        print("目录内容:")
        for file in files[:10]:  # 只显示前10个文件
            file_path = os.path.join(MODEL_PATH, file)
            size = os.path.getsize(file_path) / (1024 * 1024)  # 转换为MB
            print(f"  - {file} ({size:.2f} MB)")
        
        if len(files) > 10:
            print(f"  ... 以及其他 {len(files) - 10} 个文件")
    else:
        print(f"✗ 模型路径不存在")
        
        # 检查父目录是否存在
        parent_dir = os.path.dirname(MODEL_PATH)
        if os.path.exists(parent_dir):
            print(f"父目录存在: {parent_dir}")
            print("父目录内容:")
            try:
                for item in os.listdir(parent_dir):
                    print(f"  - {item}")
            except Exception as e:
                print(f"无法列出父目录内容: {e}")
        else:
            print(f"父目录不存在: {parent_dir}")
            
            # 继续向上检查
            grand_parent = os.path.dirname(parent_dir)
            if os.path.exists(grand_parent):
                print(f"祖父目录存在: {grand_parent}")
                print("祖父目录内容:")
                try:
                    for item in os.listdir(grand_parent):
                        print(f"  - {item}")
                except Exception as e:
                    print(f"无法列出祖父目录内容: {e}")

if __name__ == "__main__":
    main()
    print("\n按任意键退出...")
    input()
