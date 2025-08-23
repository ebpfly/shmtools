#!/usr/bin/env python3
"""
Find functions that are actually missing meta sections
"""

import os
import re
from pathlib import Path

def find_missing_meta():
    """Find functions actually missing meta sections."""
    missing_meta = []
    
    shmtools_path = Path('shmtools')
    
    for py_file in shmtools_path.rglob('*.py'):
        if py_file.name in ['__init__.py', 'introspection.py', 'jupyter_extension_installer.py']:
            continue
        if 'checkpoint' in str(py_file):
            continue
            
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all _shm functions with their docstrings
        pattern = r'def ([a-zA-Z_][a-zA-Z0-9_]*_shm)\([^)]*\):\s*"""(.*?)"""'
        matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
        
        for func_name, docstring in matches:
            if '.. meta::' not in docstring:
                missing_meta.append(f"{py_file}:{func_name}")
                
    return sorted(missing_meta)

if __name__ == "__main__":
    missing = find_missing_meta()
    print(f"Functions missing meta sections: {len(missing)}")
    for func in missing:
        print(f"  - {func}")