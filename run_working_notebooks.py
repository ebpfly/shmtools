#!/usr/bin/env python3
"""
Simple script to run only the working example notebooks and publish them to HTML.

This script focuses on notebooks that are known to work with the current 
shmtools implementation, avoiding those with missing imports or syntax errors.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any

import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError


def find_working_notebooks(examples_dir: Path) -> List[Path]:
    """
    Find notebooks that are likely to work with current shmtools.
    
    This filters out notebooks with known issues:
    - Missing 'examples' module imports
    - Syntax errors in import statements  
    - Missing functions that don't exist yet
    """
    # Start with all notebooks
    all_notebooks = list(examples_dir.rglob("*.ipynb"))
    
    # Filter out known problematic patterns
    exclude_patterns = [
        "*checkpoint*",
        "*test*", "*Test*", "*debug*", "*Debug*",
        # Specific problematic notebooks
        "*default_detector_usage_backup*",  # Missing imports
        "*active_sensing_feature_extraction*",  # Missing examples module
        "*sensor_diagnostics*",  # Missing examples module  
        "*ball_bearing_fault_analysis*",  # Missing classes
        "*gearbox_fault_analysis*",  # Missing examples module
        "*dataloader_demo*",  # Missing examples module
        "*ar_model_order_selection*",  # Missing examples module
        "*chi_square_outlier_detection*",  # Syntax errors
        "*damage_localization_ar_arx*",  # Syntax errors  
        "*daq_ar_mahalanobis_integration*",  # Missing classes
        "*data_normalization_modal_properties*",  # Missing import functions
        "*optimal_sensor_placement*",  # Missing examples module
        "*custom_detector_assembly*",  # Syntax errors
        "*how_to_use_default_detectors*",  # Syntax errors
        "*direct_use_nonparametric*",  # Missing examples module
        "*fast_metric_kernel_density*",  # Missing examples module
        "*direct_use_semiparametric*",  # Missing examples module
        "*nlpca_outlier_detection*",  # Missing examples module
        "*factor_analysis_outlier_detection*",  # Missing examples module
        "*mahalanobis_outlier_detection*",  # Missing examples module
        "*pca_outlier_detection*",  # Missing examples module
        "*svd_outlier_detection*",  # Missing examples module
        "*ni_ultrasonic_daq*",  # Hardware dependent / function errors
    ]
    
    working_notebooks = []
    for notebook in all_notebooks:
        exclude = False
        for pattern in exclude_patterns:
            if notebook.match(pattern):
                exclude = True
                break
        if not exclude:
            working_notebooks.append(notebook)
    
    return sorted(working_notebooks)


def execute_notebook(notebook_path: Path, timeout: int = 600) -> tuple[bool, Dict[str, Any]]:
    """Execute a notebook and return success status with metadata."""
    
    metadata = {
        'notebook_path': str(notebook_path),
        'success': False,
        'execution_time': None,
        'error_message': None
    }
    
    try:
        print(f"  Executing: {notebook_path.name}")
        
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Set up executor
        executor = ExecutePreprocessor(
            timeout=timeout,
            kernel_name='python3',
            allow_errors=False
        )
        
        # Execute in the notebook's directory
        executor.cwd = str(notebook_path.parent)
        
        # Execute notebook
        executed_notebook, resources = executor.preprocess(
            notebook, {'metadata': {'path': str(notebook_path.parent)}}
        )
        
        metadata['success'] = True
        print(f"    ✓ Success")
        return True, metadata
        
    except CellExecutionError as e:
        metadata['error_message'] = str(e)
        print(f"    ✗ Execution failed: {e}")
        return False, metadata
        
    except Exception as e:
        metadata['error_message'] = str(e)
        print(f"    ✗ Failed: {e}")
        return False, metadata


def convert_to_html(notebook_path: Path, output_path: Path) -> bool:
    """Convert notebook to HTML."""
    try:
        print(f"  Converting to HTML: {notebook_path.name}")
        
        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Set up HTML exporter
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'classic'
        
        # Convert to HTML
        (body, resources) = html_exporter.from_notebook_node(notebook)
        
        # Write HTML file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(body)
        
        print(f"    ✓ HTML saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"    ✗ HTML conversion failed: {e}")
        return False


def main():
    """Main entry point."""
    # Check we're in the right directory
    if not Path("shmtools").exists():
        print("❌ Error: Must run from SHMTools root directory (where shmtools/ exists)")
        return 1
    
    # Set up paths
    examples_dir = Path("examples/notebooks")
    output_dir = Path("published_notebooks_working")
    
    if not examples_dir.exists():
        print(f"❌ Error: Examples directory not found: {examples_dir}")
        return 1
    
    # Clean output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🔍 SHMTools Working Notebooks Publisher")
    print("=" * 50)
    
    # Find working notebooks
    print("1. Finding working notebooks...")
    working_notebooks = find_working_notebooks(examples_dir)
    
    if not working_notebooks:
        print("❌ No working notebooks found")
        return 1
        
    print(f"   Found {len(working_notebooks)} potentially working notebooks:")
    for nb in working_notebooks:
        print(f"     - {nb.name}")
    
    # Process each notebook
    print("\n2. Processing notebooks...")
    results = {}
    
    for notebook_path in working_notebooks:
        print(f"\n📔 Processing: {notebook_path.name}")
        
        # Try to execute
        success, metadata = execute_notebook(notebook_path, timeout=300)  # 5 minute timeout
        results[str(notebook_path)] = metadata
        
        if success:
            # Convert to HTML
            relative_path = notebook_path.relative_to(examples_dir)
            output_path = output_dir / relative_path.with_suffix('.html')
            html_success = convert_to_html(notebook_path, output_path)
            metadata['html_success'] = html_success
        else:
            metadata['html_success'] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    total = len(results)
    successful_executions = sum(1 for r in results.values() if r['success'])
    successful_html = sum(1 for r in results.values() if r.get('html_success', False))
    
    print(f"Total notebooks processed: {total}")
    print(f"Successful executions: {successful_executions}")
    print(f"Successful HTML conversions: {successful_html}")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Show failed notebooks
    failed_notebooks = [path for path, result in results.items() if not result['success']]
    if failed_notebooks:
        print(f"\n❌ Failed executions ({len(failed_notebooks)}):")
        for path in failed_notebooks:
            notebook_name = Path(path).name
            error = results[path].get('error_message', 'Unknown error')
            print(f"  - {notebook_name}: {error[:100]}...")
    
    if successful_html > 0:
        print(f"\n✅ SUCCESS: {successful_html} notebooks published!")
        print(f"\n📖 To view: cd {output_dir} && python -m http.server 8000")
        return 0
    else:
        print("\n❌ No notebooks were successfully published")
        return 1


if __name__ == "__main__":
    sys.exit(main())