#!/usr/bin/env python3
"""
Fix import statements in all example notebooks to use correct paths and functions.

This script systematically updates notebooks to use the proper import paths:
- examples.data imports for data loading functions
- shmtools.* imports for analysis functions
- Fix syntax errors in import cells
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
import nbformat


def fix_import_cell_syntax(source: str) -> str:
    """Fix syntax errors in import cells by adding missing newlines."""
    
    # Specific patterns found in the broken notebooks
    patterns = [
        # Fix "from pathlib import Path" split across lines
        (r'from pathlib\s+import Path', r'from pathlib import Path'),
        (r'from scipy\s+import', r'from scipy import'),
        (r'from typing\s+import', r'from typing import'),
        (r'from examples\.data\s+import', r'from examples.data import'),
        (r'from shmtools\.[^\s]+\s+import', lambda m: m.group(0).replace('\n', ' ').replace('  ', ' ')),
        (r'from mpl_toolkits\.mplot3d\s+import', r'from mpl_toolkits.mplot3d import'),
        
        # Fix broken function calls
        (r'Path\.cw\s*d\(\)', r'Path.cwd()'),
        (r'project_root\.paren\s*t', r'project_root.parent'),
        (r'current_di\s*r', r'current_dir'),
        (r'notebook_dir\.parent\.parent\.paren\s*t', r'notebook_dir.parent.parent.parent'),
        (r'np\.random\.rand\s*n\(\)', r'np.random.randn()'),
        (r'setup_notebook_environmen\s*t\(\)', r'setup_notebook_environment()'),
        (r'import_3story_structure_shm = nb\[\'load_3story_dat\s*a\'\]', r'import_3story_structure_shm = nb[\'load_3story_data\']'),
        (r'check_data_availability = nb\[\'check_data_availabilit\s*y\'\]', r'check_data_availability = nb[\'check_data_availability\']'),
        
        # Fix broken variable assignments
        (r'np = nb\[\'n\s*p\'\]', r'np = nb[\'np\']'),
        (r'plt = nb\[\'pl\s*t\'\]', r'plt = nb[\'plt\']'),
        
        # Fix broken comments and strings
        (r'# Angular resolutio\s*n', r'# Angular resolution'),
        (r'# Number of revolutions to simulat\s*e', r'# Number of revolutions to simulate'),
        (r'# Single acceleromete\s*r', r'# Single accelerometer'),
        (r'# Healthy vs damaged bearin\s*g', r'# Healthy vs damaged bearing'),
        (r'n_samples = samples_per_rev \* n_revolution\s*s', r'n_samples = samples_per_rev * n_revolutions'),
        (r'theta = np\.linspace\(0, n_revolutions \* 2 \* np\.pi, n_sample\s*s\)', r'theta = np.linspace(0, n_revolutions * 2 * np.pi, n_samples)'),
        (r'# 10th order \(gear mes\s*h\)', r'# 10th order (gear mesh)'),
        (r'bearing_fault_rate = 15\.3  # Ball pass frequency outer rac\s*e', r'bearing_fault_rate = 15.3  # Ball pass frequency outer race'),
        (r'bearing_impulses = np\.zeros_like\(thet\s*a\)', r'bearing_impulses = np.zeros_like(theta)'),
        (r'fault_phases = np\.arange\(0, n_revolutions \* 2 \* np\.pi, 2 \* np\.pi / bearing_fault_rat\s*e\)', r'fault_phases = np.arange(0, n_revolutions * 2 * np.pi, 2 * np.pi / bearing_fault_rate)'),
        (r'actual_phase = phase \+ 0\.1 \* np\.random\.rand\s*n\(\)', r'actual_phase = phase + 0.1 * np.random.randn()'),
        (r'impulse_idx = np\.argmin\(np\.abs\(theta - actual_phas\s*e\)\)', r'impulse_idx = np.argmin(np.abs(theta - actual_phase))'),
        (r'decay = np\.exp\(-0\.5 \* np\.arang\s*e\(10\)\)', r'decay = np.exp(-0.5 * np.arange(10))'),
        (r'amplitude = 1\.0 \+ 0\.3 \* np\.random\.rand\s*n\(\)', r'amplitude = 1.0 + 0.3 * np.random.randn()'),
        (r'noise = 0\.3 \* np\.random\.randn\(n_sample\s*s\)', r'noise = 0.3 * np.random.randn(n_samples)'),
        (r'X_angular = np\.zeros\(\(n_samples, n_channels, n_instance\s*s\)\)', r'X_angular = np.zeros((n_samples, n_channels, n_instances))'),
        
        # Fix broken boolean values
        (r'NIDAQMX_AVAILABLE = Tru\s*e', r'NIDAQMX_AVAILABLE = True'),
        (r'NIDAQMX_AVAILABLE = Fals\s*e', r'NIDAQMX_AVAILABLE = False'),
        
        # Fix concatenated plotting commands
        (r'# Set up plottingplt\.style\.use\(\'default\'\)plt\.rcParams\[\'figure\.figsize\'\] = \(12, 8\)plt\.rcParams\[\'font\.siz\s*e\'\] = 10', 
         r'# Set up plotting\nplt.style.use(\'default\')\nplt.rcParams[\'figure.figsize\'] = (12, 8)\nplt.rcParams[\'font.size\'] = 10'),
        
        # Fix lines that got split incorrectly
        (r'from ([a-zA-Z0-9_.]+)\s+import\s+([^#\n]+)', r'from \1 import \2'),
        
        # Fix broken string literals
        (r'plt\.rcParams\[\'font\.siz\s*e\'\]', r'plt.rcParams[\'font.size\']'),
        
        # Fix unmatched parentheses from broken function calls
        (r'(\d+\.\d+ \* np\.sin\(\d+ \* theta\)) +\s*# [^\n]*\n\s*0\.5 \* np\.sin\(30 \* theta\)\)\s*# 3rd harmonic', 
         r'\1 +      # 2nd harmonic\n               0.5 * np.sin(30 * theta))       # 3rd harmonic'),
    ]
    
    fixed_source = source
    for pattern, replacement in patterns:
        if callable(replacement):
            fixed_source = re.sub(pattern, replacement, fixed_source, flags=re.MULTILINE | re.DOTALL)
        else:
            fixed_source = re.sub(pattern, replacement, fixed_source, flags=re.MULTILINE | re.DOTALL)
    
    return fixed_source


def update_import_statements(source: str) -> str:
    """Update import statements to use correct paths."""
    
    # Map of old imports to new imports
    import_mappings = {
        # Data loading functions - fix imports
        'from examples.data import load_3story_data': 'from examples.data import import_3story_structure_shm',
        'from examples.data import load_sensor_diagnostic_data': 'from examples.data import import_sensor_diagnostic_shm', 
        'from examples.data import load_modal_osp_data': 'from examples.data import import_modal_osp_shm',
        'from examples.data import import_cbm_data_shm': 'from examples.data import import_cbm_data_shm',
        'from examples.data import import_active_sense1_shm': 'from examples.data import import_active_sense1_shm',
        'from examples.data import setup_notebook_environment': '# setup_notebook_environment not needed - import directly',
        
        # Fix specific function names in import statements
        'load_3story_data': 'import_3story_structure_shm',
        'load_sensor_diagnostic_data': 'import_sensor_diagnostic_shm',
        'load_modal_osp_data': 'import_modal_osp_shm',
        
        # Add missing imports that are commonly needed
        'shmtools.import_ModalOSP_shm': 'examples.data.import_modal_osp_shm',
        'shmtools.import_3StoryStructure_shm': 'examples.data.import_3story_structure_shm',
        
        # Fix class-based imports that don't exist yet
        'from shmtools.features import SpectralFeatures': '# SpectralFeatures not implemented yet',
        'from shmtools.features import EnvelopeAnalysis': '# EnvelopeAnalysis not implemented yet', 
        'from shmtools.features import ARModelExtractor': '# ARModelExtractor not implemented yet',
        'from shmtools.cbm import BearingFaultDetector': '# BearingFaultDetector not implemented yet',
        'from shmtools.detection import MahalanobisDetector': '# MahalanobisDetector not implemented yet',
        'from shmtools.daq import simulate_daq_data': '# simulate_daq_data not implemented yet',
        
        # Fix module imports that don't exist
        'from shmtools.core.cbm_processing import': '# cbm_processing module not implemented yet',
        'from shmtools.core.signal_processing import': 'from shmtools.core.signal_processing import',
        'from shmtools.core.spectral import': 'from shmtools.core.spectral import',
        'from shmtools.core.statistics import': 'from shmtools.core.statistics import',
        'from shmtools.sensor_diagnostics import': 'from shmtools.sensor_diagnostics.sensor_diagnostics import',
    }
    
    # Apply mappings
    updated_source = source
    for old_import, new_import in import_mappings.items():
        updated_source = updated_source.replace(old_import, new_import)
    
    return updated_source


def add_missing_imports(source: str, notebook_name: str) -> str:
    """Add missing imports that are commonly needed."""
    
    # If this is a notebook that needs examples.data, add sys.path fix
    if 'from examples.data import' in source and 'sys.path' not in source and 'Add the repository root' not in source:
        sys_path_fix = '''
import sys
from pathlib import Path

# Add the repository root to Python path for examples.data import
repo_root = Path.cwd()
while not (repo_root / 'shmtools').exists() and repo_root.parent != repo_root:
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
'''
        
        # Insert after numpy/matplotlib imports but before examples.data import
        lines = source.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if 'from examples.data import' in line:
                insert_idx = i
                break
            elif 'import matplotlib.pyplot' in line:
                insert_idx = i + 1
            elif 'import numpy' in line:
                insert_idx = i + 1
        
        lines.insert(insert_idx, sys_path_fix)
        source = '\n'.join(lines)
    
    return source


def fix_function_calls(source: str) -> str:
    """Fix function calls to use correct function names."""
    
    # Map of old function calls to new ones
    function_mappings = {
        'load_3story_data()': 'import_3story_structure_shm()',
        'load_sensor_diagnostic_data()': 'import_sensor_diagnostic_shm()',
        'load_modal_osp_data()': 'import_modal_osp_shm()',
        
        # Handle unpacking - the import functions return different structures
        'dataset, damage_states, state_list = load_3story_data()': 'dataset, damage_states, state_list = import_3story_structure_shm()',
        'healthy_data, example_sensor_data = load_sensor_diagnostic_data()': 'healthy_data, example_sensor_data = import_sensor_diagnostic_shm()',
        'node_layout, elements, mode_shapes, resp_dof = load_modal_osp_data()': 'node_layout, elements, mode_shapes, resp_dof = import_modal_osp_shm()',
        
        # Fix shmtools function calls
        'shmtools.import_ModalOSP_shm()': 'import_modal_osp_shm()',
        'shmtools.import_3StoryStructure_shm()': 'import_3story_structure_shm()',
    }
    
    updated_source = source
    for old_call, new_call in function_mappings.items():
        updated_source = updated_source.replace(old_call, new_call)
    
    return updated_source


def fix_notebook_cell(cell: Dict[str, Any]) -> bool:
    """Fix a single notebook cell. Returns True if modified."""
    
    if cell['cell_type'] != 'code':
        return False
    
    original_source = cell['source']
    
    # Apply all fixes
    fixed_source = original_source
    fixed_source = fix_import_cell_syntax(fixed_source)
    fixed_source = update_import_statements(fixed_source)
    fixed_source = add_missing_imports(fixed_source, "notebook")
    fixed_source = fix_function_calls(fixed_source)
    
    # Check if anything changed
    if fixed_source != original_source:
        cell['source'] = fixed_source
        return True
    
    return False


def fix_notebook(notebook_path: Path) -> bool:
    """Fix a single notebook. Returns True if modified."""
    
    print(f"Processing: {notebook_path.name}")
    
    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        modified = False
        
        # Fix each cell
        for i, cell in enumerate(notebook.cells):
            if fix_notebook_cell(cell):
                modified = True
                print(f"  Fixed cell {i}")
        
        # Write back if modified
        if modified:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                nbformat.write(notebook, f)
            print(f"  ✓ Updated {notebook_path.name}")
            return True
        else:
            print(f"  - No changes needed for {notebook_path.name}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error processing {notebook_path.name}: {e}")
        return False


def main():
    """Main entry point."""
    
    # Find all notebooks
    examples_dir = Path("examples/notebooks")
    if not examples_dir.exists():
        print("❌ Error: examples/notebooks directory not found")
        return 1
    
    notebooks = list(examples_dir.rglob("*.ipynb"))
    
    # Filter out checkpoints
    notebooks = [nb for nb in notebooks if '.ipynb_checkpoints' not in str(nb)]
    
    print(f"🔧 Fixing {len(notebooks)} notebooks...")
    print("=" * 50)
    
    modified_count = 0
    
    for notebook_path in sorted(notebooks):
        if fix_notebook(notebook_path):
            modified_count += 1
    
    print("=" * 50)
    print(f"✅ Fixed {modified_count} notebooks")
    print(f"📝 Total processed: {len(notebooks)}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())