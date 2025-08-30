#!/usr/bin/env python3
"""
Script to scan all example notebooks and update function docstrings with notebook references.
"""

import os
import re
import json
import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple


def find_notebooks(examples_dir: Path) -> List[Path]:
    """Find all Jupyter notebooks in the examples directory."""
    notebooks = []
    for root, dirs, files in os.walk(examples_dir):
        for file in files:
            if file.endswith('.ipynb') and not file.startswith('Untitled') and 'checkpoint' not in file:
                notebooks.append(Path(root) / file)
    return notebooks


def extract_function_calls_from_notebook(notebook_path: Path) -> Set[str]:
    """Extract all shmtools function calls from a notebook."""
    function_calls = set()
    
    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                
                # Find direct function calls like function_name_shm(
                pattern = r'\b([a-z_]+_shm)\s*\('
                matches = re.findall(pattern, source)
                function_calls.update(matches)
                
                # Also find imports
                import_pattern = r'from\s+shmtools[.\w]*\s+import\s+([^#\n]+)'
                import_matches = re.findall(import_pattern, source)
                for import_match in import_matches:
                    # Handle multiple imports
                    funcs = [f.strip() for f in import_match.split(',')]
                    for func in funcs:
                        if '_shm' in func:
                            function_calls.add(func.strip())
    except Exception as e:
        print(f"Error processing {notebook_path}: {e}")
    
    return function_calls


def find_shmtools_functions(shmtools_dir: Path) -> Dict[str, Path]:
    """Find all _shm functions in the shmtools directory."""
    functions = {}
    
    for root, dirs, files in os.walk(shmtools_dir):
        # Skip __pycache__ and .ipynb_checkpoints directories
        if '__pycache__' in root or '.ipynb_checkpoints' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Find function definitions
                    pattern = r'^def\s+([a-z_]+_shm)\s*\('
                    matches = re.findall(pattern, content, re.MULTILINE)
                    for match in matches:
                        functions[match] = file_path
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return functions


def create_notebook_mapping(notebooks: List[Path], shmtools_dir: Path) -> Dict[str, List[str]]:
    """Create a mapping of functions to notebooks that use them."""
    function_to_notebooks = {}
    
    # Get all functions
    functions = find_shmtools_functions(shmtools_dir)
    
    # Initialize empty lists for all functions
    for func_name in functions:
        function_to_notebooks[func_name] = []
    
    # Scan each notebook
    for notebook_path in notebooks:
        print(f"Scanning {notebook_path.name}...")
        function_calls = extract_function_calls_from_notebook(notebook_path)
        
        # Map function calls to notebooks
        for func_name in function_calls:
            if func_name in function_to_notebooks:
                # Store just the filename, not the full path
                notebook_name = notebook_path.name
                if notebook_name not in function_to_notebooks[func_name]:
                    function_to_notebooks[func_name].append(notebook_name)
    
    # Sort notebook names for consistency
    for func_name in function_to_notebooks:
        function_to_notebooks[func_name].sort()
    
    return function_to_notebooks


def update_function_docstring(file_path: Path, func_name: str, notebook_refs: List[str]) -> bool:
    """Update a function's docstring with notebook references."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find the function definition line
        func_pattern = rf'^def\s+{re.escape(func_name)}\s*\('
        func_line_idx = None
        for i, line in enumerate(lines):
            if re.match(func_pattern, line):
                func_line_idx = i
                break
        
        if func_line_idx is None:
            print(f"Could not find function {func_name} in {file_path}")
            return False
        
        # Find the docstring start (should be right after function definition)
        docstring_start_idx = None
        for i in range(func_line_idx + 1, min(func_line_idx + 10, len(lines))):
            if '"""' in lines[i] or "'''" in lines[i]:
                docstring_start_idx = i
                break
        
        if docstring_start_idx is None:
            print(f"No docstring found for {func_name}")
            return False
        
        # Find the meta section
        meta_line_idx = None
        meta_end_idx = None
        for i in range(docstring_start_idx, len(lines)):
            if '.. meta::' in lines[i]:
                meta_line_idx = i
                # Find end of meta section (next section or end of docstring)
                for j in range(i + 1, len(lines)):
                    # Check for next section or end of docstring
                    if (lines[j].strip().startswith('.. gui::') or 
                        'Parameters' in lines[j] and '---' in lines[j+1] if j+1 < len(lines) else False or
                        'Returns' in lines[j] and '---' in lines[j+1] if j+1 < len(lines) else False or
                        '"""' in lines[j] or "'''" in lines[j]):
                        meta_end_idx = j
                        break
                break
        
        if meta_line_idx is None:
            print(f"No meta section found for {func_name}")
            return False
        
        # Remove existing example_notebooks line if present
        new_lines = lines.copy()
        for i in range(meta_line_idx + 1, meta_end_idx):
            if ':example_notebooks:' in lines[i]:
                new_lines[i] = ''  # Remove the line
        
        # Add new example_notebooks line if we have references
        if notebook_refs:
            # Find the indentation of other meta fields
            indent = '        '  # Default indentation
            for i in range(meta_line_idx + 1, meta_end_idx):
                if lines[i].strip().startswith(':'):
                    # Extract indentation
                    indent = lines[i][:lines[i].index(':')]
                    break
            
            # Insert the new line before the end of meta section
            notebooks_str = ', '.join([f'"{nb}"' for nb in notebook_refs])
            new_line = f'{indent}:example_notebooks: [{notebooks_str}]\n'
            
            # Insert right before meta_end_idx
            new_lines.insert(meta_end_idx, new_line)
        
        # Write back the file
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
        
        return True
    except Exception as e:
        print(f"Error updating {func_name} in {file_path}: {e}")
        return False


def main():
    """Main function to update all notebook references."""
    # Set up paths
    project_root = Path('/Users/eric/repo/shm')
    shmtools_dir = project_root / 'shmtools'
    examples_dir = project_root / 'examples'
    
    print("Finding all notebooks...")
    notebooks = find_notebooks(examples_dir)
    print(f"Found {len(notebooks)} notebooks")
    
    print("\nCreating function to notebook mapping...")
    function_to_notebooks = create_notebook_mapping(notebooks, shmtools_dir)
    
    # Find all function files
    functions = find_shmtools_functions(shmtools_dir)
    
    print(f"\nFound {len(functions)} functions")
    print(f"Functions used in notebooks: {sum(1 for refs in function_to_notebooks.values() if refs)}")
    
    # Update each function's docstring
    print("\nUpdating function docstrings...")
    updated_count = 0
    failed_count = 0
    for func_name, file_path in functions.items():
        notebook_refs = function_to_notebooks.get(func_name, [])
        if notebook_refs:
            print(f"Updating {func_name} with {len(notebook_refs)} notebook reference(s)")
            if update_function_docstring(file_path, func_name, notebook_refs):
                updated_count += 1
            else:
                failed_count += 1
    
    print(f"\nSuccessfully updated {updated_count} functions")
    if failed_count > 0:
        print(f"Failed to update {failed_count} functions")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total notebooks scanned: {len(notebooks)}")
    print(f"Total functions found: {len(functions)}")
    print(f"Functions with notebook references: {sum(1 for refs in function_to_notebooks.values() if refs)}")
    print(f"Functions updated: {updated_count}")
    
    # Print functions with most notebook references
    sorted_funcs = sorted(
        [(func, refs) for func, refs in function_to_notebooks.items() if refs],
        key=lambda x: len(x[1]),
        reverse=True
    )
    
    if sorted_funcs:
        print("\nTop 10 most-used functions:")
        for func, refs in sorted_funcs[:10]:
            print(f"  {func}: {len(refs)} notebook(s)")
            print(f"    Notebooks: {', '.join(refs[:3])}{' ...' if len(refs) > 3 else ''}")


if __name__ == '__main__':
    main()