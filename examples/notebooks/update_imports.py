#!/usr/bin/env python
"""
Script to update all notebook imports from shmtools.utils.data_loading to examples.data
"""

import json
import re
from pathlib import Path

def update_notebook_imports(notebook_path):
    """Update imports in a single notebook."""
    
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    modified = False
    
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            
            # Convert to string if it's a list
            if isinstance(source, list):
                source_str = ''.join(source)
            else:
                source_str = source
            
            # Check if this cell has the old import
            if 'from shmtools.utils.data_loading import' in source_str:
                # Replace the import
                new_source = source_str.replace(
                    'from shmtools.utils.data_loading import',
                    'from examples.data import'
                )
                
                # Store back as list (Jupyter format)
                cell['source'] = new_source.splitlines(True)
                modified = True
                print(f"  Updated imports in {notebook_path.name}")
    
    if modified:
        # Write back the notebook
        with open(notebook_path, 'w') as f:
            json.dump(notebook, f, indent=1)
        return True
    return False

def main():
    """Update all notebooks in the examples/notebooks directory."""
    
    notebooks_dir = Path('/Users/eric/repo/shm/examples/notebooks')
    
    # Find all notebooks that need updating
    notebooks_to_update = []
    
    for notebook_path in notebooks_dir.rglob('*.ipynb'):
        # Skip checkpoint files
        if '.ipynb_checkpoints' in str(notebook_path):
            continue
            
        # Check if notebook contains old imports
        try:
            with open(notebook_path, 'r') as f:
                content = f.read()
                if 'shmtools.utils.data_loading' in content:
                    notebooks_to_update.append(notebook_path)
        except Exception as e:
            print(f"Error reading {notebook_path}: {e}")
    
    print(f"Found {len(notebooks_to_update)} notebooks to update:")
    for nb in notebooks_to_update:
        print(f"  - {nb.relative_to(notebooks_dir)}")
    
    print("\nUpdating notebooks...")
    updated_count = 0
    
    for notebook_path in notebooks_to_update:
        try:
            if update_notebook_imports(notebook_path):
                updated_count += 1
        except Exception as e:
            print(f"Error updating {notebook_path}: {e}")
    
    print(f"\nSuccessfully updated {updated_count} notebooks")

if __name__ == '__main__':
    main()