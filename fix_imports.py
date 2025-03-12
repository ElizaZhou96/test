# fix_imports.py
import os
import re

def ensure_traceback_import(file_path):
    """确保文件导入了traceback模块"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 检查是否已经导入traceback
    if 'import traceback' in content or 'from traceback import' in content:
        print(f"{file_path} - 已导入traceback")
        return False
    
    # 查找适合插入导入语句的位置
    import_pattern = re.compile(r'^import .*$|^from .* import', re.MULTILINE)
    matches = list(import_pattern.finditer(content))
    
    if matches:
        # 找到最后一个导入语句
        last_import = matches[-1]
        insert_pos = last_import.end()
        
        # 插入traceback导入
        new_content = content[:insert_pos] + "\nimport traceback  # 自动添加的导入" + content[insert_pos:]
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(new_content)
        
        print(f"{file_path} - 添加了traceback导入")
        return True
    else:
        print(f"{file_path} - 无法找到合适的插入点")
        return False

def fix_all_model_files():
    """修复所有模型文件的导入"""
    # 修复主文件
    ensure_traceback_import("main.py")
    
    # 修复所有模型文件
    model_files = os.listdir("models")
    for file in model_files:
        if file.endswith(".py"):
            ensure_traceback_import(os.path.join("models", file))

if __name__ == "__main__":
    fix_all_model_files()
    print("导入修复完成！")