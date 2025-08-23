#!/usr/bin/env python3
"""
Final fix for notebooks - address the escaped quote issue and other syntax problems.
"""

import re
from pathlib import Path
import nbformat


def fix_escaped_quotes(source: str) -> str:
    """Fix escaped quotes that are causing syntax errors."""
    
    # Fix escaped quotes in common patterns
    fixes = [
        # Fix escaped quotes in matplotlib rcParams
        (r"plt\.rcParams\[\\?'([^']+)\\?'\]", r"plt.rcParams['\1']"),
        (r"plt\.style\.use\(\\?'([^']+)\\?'\)", r"plt.style.use('\1')"),
        
        # Fix broken variable assignments
        (r"(\w+)\s*=\s*\w+\[\\'([^']+)\\'\]", r"\1 = nb['\2']"),
        
        # Fix broken string literals
        (r"\\?'([^'\\]+)\\?'", r"'\1'"),
        
        # Fix broken function calls split across lines
        (r'(\w+)\s*\(\s*([^)]*)\s*([a-z])\s*([a-z])\s*\)', r'\1(\2\3\4)'),
        
        # Fix broken variable names
        (r'(\w+)_([a-z])\s*([a-z])', r'\1_\2\3'),
        
        # Fix lines broken in the middle
        (r'(\d+)  # [^\n]*\s*([a-z])\s*([a-z])', r'\1  # \2\3'),
        
        # Fix unmatched parentheses
        (r'\s*([a-z])\s*\)', r'\1)'),
        
        # Fix broken comments
        (r'# ([^#\n]*)\s+([a-z])\s+([a-z])', r'# \1 \2 \3'),
        
        # Fix multiline import statements that got broken
        (r'from\s+([a-zA-Z_.]+)\s+import\s+\(\s*([^)]*)\s*$', r'from \1 import (\2'),
    ]
    
    fixed_source = source
    for pattern, replacement in fixes:
        fixed_source = re.sub(pattern, replacement, fixed_source, flags=re.MULTILINE)
    
    return fixed_source


def create_working_import_cell(notebook_name: str) -> str:
    """Create a standard working import cell for each type of notebook."""
    
    if 'sensor_diagnostic' in notebook_name:
        return '''import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add the repository root to Python path for examples.data import
repo_root = Path.cwd()
while not (repo_root / 'shmtools').exists() and repo_root.parent != repo_root:
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import shmtools functions
from examples.data import import_sensor_diagnostic_shm
from shmtools.sensor_diagnostics.sensor_diagnostics import sd_feature_shm, sd_autoclassify_shm, sd_plot_shm

# Set up plotting parameters
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12'''

    elif 'active_sensing' in notebook_name:
        return '''import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add the repository root to Python path for examples.data import
repo_root = Path.cwd()
while not (repo_root / 'shmtools').exists() and repo_root.parent != repo_root:
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import shmtools functions
from examples.data import import_active_sense1_shm
from shmtools.active_sensing import (
    extract_subsets_shm, flex_logic_filter_shm, sum_mult_dims_shm,
    estimate_group_velocity_shm, propagation_dist_2_points_shm,
    distance_2_index_shm, build_contained_grid_shm,
    sensor_pair_line_of_sight_shm, fill_2d_map_shm,
    incoherent_matched_filter_shm
)

# Set up plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10'''

    elif 'time_sync' in notebook_name:
        return '''import numpy as np
import matplotlib.pyplot as plt
import shmtools

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10'''

    else:
        # Generic import for notebooks that use examples.data
        return '''import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add the repository root to Python path for examples.data import
repo_root = Path.cwd()
while not (repo_root / 'shmtools').exists() and repo_root.parent != repo_root:
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import shmtools functions
from examples.data import import_3story_structure_shm
from shmtools.features.time_series import ar_model_shm
from shmtools.classification.outlier_detection import learn_mahalanobis_shm, score_mahalanobis_shm, roc_shm

# Set up plotting
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10'''


def fix_notebook(notebook_path: Path) -> bool:
    """Fix a single notebook with comprehensive fixes."""
    
    print(f"Fixing: {notebook_path.name}")
    
    try:
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        modified = False
        
        for i, cell in enumerate(notebook.cells):
            if cell['cell_type'] != 'code':
                continue
            
            original_source = cell['source']
            
            # Apply quote fixes
            fixed_source = fix_escaped_quotes(original_source)
            
            # If this is an import cell with problems, replace it entirely
            if ('import numpy' in fixed_source and 
                ('\\' in fixed_source or 'SyntaxError' in str(fixed_source) or 
                 'unexpected character' in str(fixed_source))):
                
                fixed_source = create_working_import_cell(notebook_path.name)
                print(f"  Replaced import cell {i}")
                modified = True
            
            # Fix other specific issues
            elif 'samples_per_rev = 256' in fixed_source and 'unmatched' in str(fixed_source):
                # Time sync averaging demo signal generation
                fixed_source = '''# Signal parameters
samples_per_rev = 256  # Angular resolution
n_revolutions = 20     # Number of revolutions to simulate
n_channels = 1         # Single accelerometer  
n_instances = 2        # Healthy vs damaged bearing

# Create angular time vector
n_samples = samples_per_rev * n_revolutions
theta = np.linspace(0, n_revolutions * 2 * np.pi, n_samples)

# Gear mesh frequency components (periodic)
gear_signal = (2.0 * np.sin(10 * theta) +      # 10th order (gear mesh)
               1.0 * np.sin(20 * theta) +      # 2nd harmonic
               0.5 * np.sin(30 * theta))       # 3rd harmonic

# Create signal matrix
X_angular = np.zeros((n_samples, n_channels, n_instances))

# Instance 0: Healthy (gear + noise only)
X_angular[:, 0, 0] = gear_signal + 0.2 * np.random.randn(n_samples)

# Instance 1: Damaged (gear + slight changes + noise)
X_angular[:, 0, 1] = gear_signal * 0.95 + 0.3 * np.random.randn(n_samples)

print(f"Generated signal matrix: {X_angular.shape}")
print(f"Signal length: {n_samples} samples ({n_revolutions} revolutions)")'''
                print(f"  Fixed signal generation cell {i}")
                modified = True
                
            elif fixed_source != original_source:
                modified = True
                print(f"  Fixed syntax in cell {i}")
            
            cell['source'] = fixed_source
        
        if modified:
            # Write back
            with open(notebook_path, 'w', encoding='utf-8') as f:
                nbformat.write(notebook, f)
            print(f"  ✓ Updated {notebook_path.name}")
            return True
        else:
            print(f"  - No changes needed")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Main entry point."""
    
    examples_dir = Path("examples/notebooks")
    notebooks = list(examples_dir.rglob("*.ipynb"))
    
    # Filter out checkpoints
    notebooks = [nb for nb in notebooks if '.ipynb_checkpoints' not in str(nb)]
    
    print(f"🔧 Final fix for {len(notebooks)} notebooks...")
    print("=" * 50)
    
    fixed_count = 0
    for notebook_path in sorted(notebooks):
        if fix_notebook(notebook_path):
            fixed_count += 1
    
    print("=" * 50)
    print(f"✅ Fixed {fixed_count} notebooks")


if __name__ == "__main__":
    main()