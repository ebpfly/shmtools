#!/usr/bin/env python3
"""
Analyze all functions in shmtools to identify missing meta sections and verbose_call fields.
"""

import os
import re
from pathlib import Path

def analyze_functions():
    """Find all functions and their meta status."""
    results = {
        'complete': [],      # Has meta with verbose_call
        'missing_verbose': [],  # Has meta but missing verbose_call
        'missing_meta': [],   # Completely missing meta section
        'not_shm_functions': []  # Functions that don't follow _shm naming
    }
    
    shmtools_path = Path('shmtools')
    
    for py_file in shmtools_path.rglob('*.py'):
        # Skip __init__.py, introspection.py, and other utility files
        if py_file.name in ['__init__.py', 'introspection.py', 'jupyter_extension_installer.py']:
            continue
            
        if 'checkpoint' in str(py_file):
            continue
            
        analyze_file(py_file, results)
    
    return results

def analyze_file(py_file, results):
    """Analyze a single Python file for function meta status."""
    try:
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {py_file}: {e}")
        return
    
    # Find all function definitions - more flexible pattern
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith('def ') and '(' in line:
            # Extract function name
            match = re.match(r'\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line)
            if not match:
                i += 1
                continue
            
            func_name = match.group(1)
            
            # Skip private functions and special methods
            if func_name.startswith('_') or func_name in ['__init__', '__call__']:
                i += 1
                continue
            
            # Look for docstring starting within next few lines
            docstring = ""
            j = i + 1
            found_docstring = False
            
            # Skip empty lines and comments after function definition
            while j < len(lines) and (not lines[j].strip() or lines[j].strip().startswith('#')):
                j += 1
            
            # Check if next non-empty line starts a docstring
            if j < len(lines) and '"""' in lines[j]:
                found_docstring = True
                docstring_start = j
                
                # Find end of docstring
                if lines[j].count('"""') >= 2:
                    # Single line docstring
                    docstring = lines[j]
                    docstring_end = j
                else:
                    # Multi-line docstring
                    docstring_lines = [lines[j]]
                    j += 1
                    while j < len(lines):
                        docstring_lines.append(lines[j])
                        if '"""' in lines[j]:
                            docstring_end = j
                            break
                        j += 1
                    if j < len(lines):  # Only join if we found the end
                        docstring = '\n'.join(docstring_lines)
                    else:
                        found_docstring = False  # Incomplete docstring
            
            file_info = f"{py_file}:{func_name}"
            
            if not found_docstring:
                # No docstring at all
                if func_name.endswith('_shm'):
                    results['missing_meta'].append(file_info)
                else:
                    results['not_shm_functions'].append(file_info)
            elif not func_name.endswith('_shm'):
                results['not_shm_functions'].append(file_info)
            elif '.. meta::' not in docstring:
                results['missing_meta'].append(file_info)
            elif 'verbose_call:' not in docstring:
                results['missing_verbose'].append(file_info)
            else:
                results['complete'].append(file_info)
                
        i += 1

def print_results(results):
    """Print analysis results."""
    print("="*70)
    print("FUNCTION META ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\n✅ COMPLETE (has meta with verbose_call): {len(results['complete'])}")
    for func in sorted(results['complete']):
        print(f"   ✓ {func}")
    
    print(f"\n⚠️  MISSING VERBOSE_CALL (has meta, missing verbose_call): {len(results['missing_verbose'])}")
    for func in sorted(results['missing_verbose']):
        print(f"   - {func}")
    
    print(f"\n❌ MISSING META SECTION (no meta at all): {len(results['missing_meta'])}")
    for func in sorted(results['missing_meta']):
        print(f"   ! {func}")
    
    print(f"\n🔍 NON-SHM FUNCTIONS (don't end with _shm): {len(results['not_shm_functions'])}")
    for func in sorted(results['not_shm_functions']):
        print(f"   ? {func}")
    
    total_missing = len(results['missing_verbose']) + len(results['missing_meta'])
    print(f"\n📊 SUMMARY:")
    print(f"   Total functions needing fixes: {total_missing}")
    print(f"   - Missing verbose_call: {len(results['missing_verbose'])}")
    print(f"   - Missing entire meta: {len(results['missing_meta'])}")
    print(f"   - Complete functions: {len(results['complete'])}")

if __name__ == "__main__":
    results = analyze_functions()
    print_results(results)