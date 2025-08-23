#!/usr/bin/env python3
"""
Fix specific syntax issues in notebooks by rewriting the problematic cells.

This script targets the exact broken patterns and replaces them with correct code.
"""

import json
from pathlib import Path
from typing import Dict, Any
import nbformat


def fix_sensor_diagnostics_notebook():
    """Fix the sensor_diagnostics.ipynb notebook specifically."""
    
    notebook_path = Path("examples/notebooks/active_sensing/sensor_diagnostics.ipynb")
    
    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Fix the import cell (cell 2)
    import_cell = '''# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add the repository root to Python path for examples.data import
repo_root = Path.cwd()
while not (repo_root / 'shmtools').exists() and repo_root.parent != repo_root:
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import shmtools (installed package)
from examples.data import import_sensor_diagnostic_shm
from shmtools.sensor_diagnostics.sensor_diagnostics import sd_feature_shm, sd_autoclassify_shm, sd_plot_shm

# Set up plotting parameters
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12'''

    # Fix the data loading cell (cell 4)
    data_cell = '''# Load the sensor diagnostic data
healthy_data, example_sensor_data = import_sensor_diagnostic_shm()

# Use the example data with known sensor faults
admittance_data = example_sensor_data

print(f"Data shape: {admittance_data.shape}")
print(f"Number of frequency points: {admittance_data.shape[0]}")
print(f"Number of sensors: {admittance_data.shape[1] - 1}")
print(f"Frequency range: {admittance_data[0, 0]:.0f} - {admittance_data[-1, 0]:.0f} Hz")'''

    # Update cells
    for i, cell in enumerate(notebook.cells):
        if cell['cell_type'] == 'code':
            if 'import numpy' in cell['source'] and 'load_sensor_diagnostic_data' in cell['source']:
                cell['source'] = import_cell
                print(f"Fixed import cell {i}")
            elif 'load_sensor_diagnostic_data()' in cell['source']:
                cell['source'] = data_cell
                print(f"Fixed data loading cell {i}")
    
    # Write back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)
    
    print(f"✓ Fixed {notebook_path}")


def fix_time_sync_notebook():
    """Fix the time_synchronous_averaging_demo.ipynb notebook."""
    
    notebook_path = Path("examples/notebooks/integrating_examples/time_synchronous_averaging_demo.ipynb")
    
    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Fix the signal generation cell
    signal_cell = '''# Signal parameters
samples_per_rev = 256  # Angular resolution
n_revolutions = 20     # Number of revolutions to simulate
n_channels = 1         # Single accelerometer
n_instances = 2        # Healthy vs damaged bearing

# Create angular time vector
n_samples = samples_per_rev * n_revolutions
theta = np.linspace(0, n_revolutions * 2 * np.pi, n_samples)

# Gear mesh frequency components (periodic)
# Fundamental gear mesh + harmonics
gear_signal = (2.0 * np.sin(10 * theta) +      # 10th order (gear mesh)
               1.0 * np.sin(20 * theta) +      # 2nd harmonic
               0.5 * np.sin(30 * theta))       # 3rd harmonic

# Random bearing fault impulses (for damaged case)
bearing_fault_rate = 15.3  # Ball pass frequency outer race
bearing_impulses = np.zeros_like(theta)

# Add random impulses at approximately bearing fault frequency
fault_phases = np.arange(0, n_revolutions * 2 * np.pi, 2 * np.pi / bearing_fault_rate)
for phase in fault_phases:
    # Add some randomness to impulse timing and amplitude
    actual_phase = phase + 0.1 * np.random.randn()
    impulse_idx = np.argmin(np.abs(theta - actual_phase))
    if impulse_idx < len(bearing_impulses) - 10:
        # Create decaying impulse
        decay = np.exp(-0.5 * np.arange(10))
        amplitude = 1.0 + 0.3 * np.random.randn()
        bearing_impulses[impulse_idx:impulse_idx + 10] += amplitude * decay

# Background noise
noise = 0.3 * np.random.randn(n_samples)

# Create signal matrix
X_angular = np.zeros((n_samples, n_channels, n_instances))

# Instance 0: Healthy (gear + noise only)
X_angular[:, 0, 0] = gear_signal + 0.2 * np.random.randn(n_samples)

# Instance 1: Damaged (gear + bearing faults + noise)
X_angular[:, 0, 1] = gear_signal + 0.8 * bearing_impulses + 0.3 * np.random.randn(n_samples)

print(f"Generated signal matrix: {X_angular.shape}")
print(f"Signal length: {n_samples} samples ({n_revolutions} revolutions)")
print(f"Angular resolution: {samples_per_rev} samples per revolution")'''

    # Update cells
    for i, cell in enumerate(notebook.cells):
        if cell['cell_type'] == 'code':
            if 'samples_per_rev = 256' in cell['source'] and 'unmatched' in str(cell['source']):
                cell['source'] = signal_cell
                print(f"Fixed signal generation cell {i}")
    
    # Write back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)
    
    print(f"✓ Fixed {notebook_path}")


def create_standard_import_cell(data_function_name: str) -> str:
    """Create a standard import cell for notebooks that use examples.data."""
    
    return f'''import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add the repository root to Python path for examples.data import
repo_root = Path.cwd()
while not (repo_root / 'shmtools').exists() and repo_root.parent != repo_root:
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import shmtools data loading function
from examples.data import {data_function_name}

# Set up plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10'''


def fix_all_notebooks():
    """Fix common patterns in all notebooks."""
    
    examples_dir = Path("examples/notebooks")
    notebooks = list(examples_dir.rglob("*.ipynb"))
    
    # Data function mapping
    data_functions = {
        'load_3story_data': 'import_3story_structure_shm',
        'load_sensor_diagnostic_data': 'import_sensor_diagnostic_shm',
        'load_modal_osp_data': 'import_modal_osp_shm',
        'import_cbm_data_shm': 'import_cbm_data_shm',
        'import_active_sense1_shm': 'import_active_sense1_shm'
    }
    
    for notebook_path in notebooks:
        if '.ipynb_checkpoints' in str(notebook_path):
            continue
            
        try:
            # Read notebook
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            
            modified = False
            
            for i, cell in enumerate(notebook.cells):
                if cell['cell_type'] != 'code':
                    continue
                    
                source = cell['source']
                
                # Fix broken function calls that got split
                for old_func, new_func in data_functions.items():
                    if old_func in source:
                        source = source.replace(old_func, new_func)
                        modified = True
                
                # Fix specific broken patterns
                fixes = [
                    # Fix function calls broken across lines
                    ('Path.cw\\s*d\\(\\)', 'Path.cwd()'),
                    ('np.random.rand\\s*n\\(\\)', 'np.random.randn()'),
                    
                    # Fix broken variable names
                    ('current_di\\s*r', 'current_dir'),
                    ('repo_root.paren\\s*t', 'repo_root.parent'),
                    
                    # Fix broken import statements that got split
                    ('from pathlib\\s+import Path', 'from pathlib import Path'),
                    ('from scipy\\s+import', 'from scipy import'),
                    ('from examples.data\\s+import', 'from examples.data import'),
                    ('from shmtools\\.[^\\s]+\\s+import', lambda m: m.group(0).replace('\\n', ' ').replace('  ', ' ')),
                    
                    # Fix plotting parameters that got concatenated
                    ('plt.rcParams\\[\'font.siz\\s*e\'\\]', 'plt.rcParams[\'font.size\']'),
                ]
                
                import re
                for pattern, replacement in fixes:
                    if callable(replacement):
                        source = re.sub(pattern, replacement, source, flags=re.MULTILINE)
                    else:
                        source = re.sub(pattern, replacement, source, flags=re.MULTILINE)
                
                if source != cell['source']:
                    cell['source'] = source
                    modified = True
            
            if modified:
                # Write back
                with open(notebook_path, 'w', encoding='utf-8') as f:
                    nbformat.write(notebook, f)
                print(f"✓ Fixed {notebook_path.name}")
                
        except Exception as e:
            print(f"✗ Error fixing {notebook_path.name}: {e}")


def main():
    """Main entry point."""
    
    print("🔧 Fixing specific notebook issues...")
    print("=" * 50)
    
    # Fix specific problematic notebooks
    fix_sensor_diagnostics_notebook()
    fix_time_sync_notebook()
    
    # Fix common patterns in all notebooks
    fix_all_notebooks()
    
    print("=" * 50)
    print("✅ Notebook fixes completed!")


if __name__ == "__main__":
    main()