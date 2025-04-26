import os
import sys
from pathlib import Path

def count_lines(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return len(f.readlines())
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return 0

def find_python_files(directory='.'):
    file_sizes = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                line_count = count_lines(file_path)
                file_sizes.append((file_path, line_count))
    
    return sorted(file_sizes, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    file_sizes = find_python_files()
    print("\nLargest Python files in the project:")
    print("=" * 80)
    print(f"{'File Path':<70} {'Line Count':>10}")
    print("-" * 80)
    
    for file_path, line_count in file_sizes[:20]:
        print(f"{file_path:<70} {line_count:>10}")
