# 测试终端命令的Python脚本

import os
import sys

def main():
    print("当前工作目录:", os.getcwd())
    print("Python版本:", sys.version)
    print("环境变量 PATH:", os.environ['PATH'])

if __name__ == "__main__":
    main()