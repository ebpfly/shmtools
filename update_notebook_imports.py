#!/usr/bin/env python3
"""
Script to update example notebooks to use the installed shmtools package instead of 
searching for the module path.

This script replaces the complex import blocks in Jupyter notebooks with simple 
imports that assume shmtools is installed via pip.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any

def update_notebook_imports(notebook_path: Path) -> bool:
    """
    Update a single notebook to use simple installed package imports.
    
    Returns True if the notebook was modified, False otherwise.
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {notebook_path}: {e}")
        return False
    
    modified = False
    
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            
            # Check if this cell contains the complex import block
            if ('Add shmtools to path' in source or 
                'possible_paths' in source or
                'shmtools_found' in source):
                
                # Extract shmtools imports from the source
                shmtools_imports = extract_shmtools_imports(source)
                
                if shmtools_imports:
                    # Create simplified import block
                    new_source = create_simple_imports(shmtools_imports)
                    
                    # Update the cell
                    cell['source'] = new_source.split('\n')
                    if not cell['source'][-1]:  # Remove empty last line
                        cell['source'] = cell['source'][:-1]
                    
                    modified = True
                    print(f"  Updated imports in {notebook_path.name}")
                    break
    
    if modified:
        try:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)
        except IOError as e:
            print(f"Error writing {notebook_path}: {e}")
            return False
    
    return modified

def extract_shmtools_imports(source: str) -> list:
    """Extract shmtools import statements from source code."""
    imports = []
    
    # Find all shmtools import lines
    for line in source.split('\n'):
        line = line.strip()
        if line.startswith('from shmtools') and 'import' in line:
            imports.append(line)
    
    return imports

def create_simple_imports(shmtools_imports: list) -> str:
    """Create a simplified import block."""
    lines = [
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        ""
    ]
    
    # Add shmtools imports
    if shmtools_imports:
        lines.append("# Import shmtools (installed package)")
        lines.extend(shmtools_imports)
        lines.append("")
    
    # Add standard plotting setup
    lines.extend([
        "# Set up plotting",
        "plt.style.use('default')",
        "plt.rcParams['figure.figsize'] = (12, 8)",
        "plt.rcParams['font.size'] = 10"
    ])
    
    return '\n'.join(lines)

def main():
    """Update all example notebooks to use installed shmtools imports."""
    
    # Find all notebook files
    notebooks_dir = Path('examples/notebooks')
    if not notebooks_dir.exists():
        print("Error: examples/notebooks directory not found!")
        print("Make sure you're running this from the shmtools-python directory.")
        return
    
    notebook_files = list(notebooks_dir.glob('**/*.ipynb'))
    
    if not notebook_files:
        print("No notebook files found!")
        return
    
    print(f"Found {len(notebook_files)} notebook files")
    print("Updating notebooks to use installed shmtools package...")
    print("-" * 60)
    
    updated_count = 0
    
    for notebook_path in notebook_files:
        # Skip certain notebooks
        if any(skip in notebook_path.name.lower() for skip in ['untitled', 'backup']):
            continue
            
        print(f"Checking {notebook_path.relative_to(notebooks_dir)}...")
        
        if update_notebook_imports(notebook_path):
            updated_count += 1
        else:
            print(f"  No changes needed")
    
    print("-" * 60)
    print(f"Updated {updated_count} notebooks")
    print("\nBefore running the notebooks, make sure shmtools is installed:")
    print("  pip install -e .")
    print("\nAll notebooks should now use simple imports like:")
    print("  import numpy as np")
    print("  import matplotlib.pyplot as plt") 
    print("  from shmtools.utils.data_loading import load_3story_data")

if __name__ == '__main__':
    main()