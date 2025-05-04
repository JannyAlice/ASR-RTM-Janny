import difflib
import sys

def compare_files(file1, file2):
    with open(file1, 'r', encoding='utf-8') as f1:
        file1_lines = f1.readlines()
    
    with open(file2, 'r', encoding='utf-8') as f2:
        file2_lines = f2.readlines()
    
    diff = difflib.unified_diff(file1_lines, file2_lines, fromfile=file1, tofile=file2)
    
    for line in diff:
        print(line, end='')

if __name__ == "__main__":
    file1 = "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\src\\ui\\main_window.py"
    file2 = "C:\\Users\\crige\\RealtimeTrans\\vosk-api\\src\\ui\\main_window_new.py"
    compare_files(file1, file2)
