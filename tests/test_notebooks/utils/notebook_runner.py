"""
Notebook execution utilities for testing.

This module provides robust notebook execution and validation utilities
for testing Jupyter notebooks in the shmtools package.
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError


class NotebookRunner:
    """
    Robust notebook execution and validation for testing.
    
    Handles various execution contexts and provides detailed error reporting.
    """
    
    def __init__(self, timeout: int = 600, kernel_name: str = "python3"):
        """
        Initialize the notebook runner.
        
        Parameters
        ----------
        timeout : int, default=600
            Maximum time in seconds to wait for notebook execution
        kernel_name : str, default="python3"
            Jupyter kernel to use for execution
        """
        self.timeout = timeout
        self.kernel_name = kernel_name
        self.execution_processor = ExecutePreprocessor(
            timeout=timeout,
            kernel_name=kernel_name,
            allow_errors=False
        )
    
    def execute_notebook(
        self, 
        notebook_path: Path, 
        working_dir: Optional[Path] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute a notebook and return success status with metadata.
        
        Parameters
        ----------
        notebook_path : Path
            Path to the notebook file to execute
        working_dir : Path, optional
            Working directory for execution. If None, uses notebook's directory.
            
        Returns
        -------
        success : bool
            True if notebook executed successfully without errors
        metadata : dict
            Execution metadata including timing, cell counts, errors
        """
        if working_dir is None:
            working_dir = notebook_path.parent
            
        # Initialize metadata
        metadata = {
            'notebook_path': str(notebook_path),
            'working_dir': str(working_dir),
            'start_time': time.time(),
            'end_time': None,
            'execution_time': None,
            'total_cells': 0,
            'executed_cells': 0,
            'error_cells': 0,
            'errors': [],
            'cell_outputs': [],
            'kernel_name': self.kernel_name
        }
        
        try:
            # Read the notebook
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
                
            metadata['total_cells'] = len(notebook.cells)
            
            # Set up execution environment
            self.execution_processor.cwd = str(working_dir)
            
            # Execute the notebook
            executed_notebook, resources = self.execution_processor.preprocess(
                notebook, {'metadata': {'path': str(working_dir)}}
            )
            
            # Collect execution statistics
            for i, cell in enumerate(executed_notebook.cells):
                if cell.cell_type == 'code':
                    metadata['executed_cells'] += 1
                    
                    # Check for execution errors
                    if 'outputs' in cell:
                        for output in cell.outputs:
                            if output.output_type == 'error':
                                metadata['error_cells'] += 1
                                metadata['errors'].append({
                                    'cell_index': i,
                                    'error_name': output.ename,
                                    'error_value': output.evalue,
                                    'traceback': output.traceback
                                })
                            
                            # Store output metadata (without actual data to save memory)
                            metadata['cell_outputs'].append({
                                'cell_index': i,
                                'output_type': output.output_type,
                                'has_data': bool(getattr(output, 'data', None))
                            })
            
            metadata['end_time'] = time.time()
            metadata['execution_time'] = metadata['end_time'] - metadata['start_time']
            
            # Success if no error cells
            success = metadata['error_cells'] == 0
            
            return success, metadata
            
        except CellExecutionError as e:
            metadata['end_time'] = time.time()
            metadata['execution_time'] = metadata['end_time'] - metadata['start_time']
            metadata['errors'].append({
                'type': 'CellExecutionError',
                'cell_index': getattr(e, 'cell_index', -1),
                'error_name': e.__class__.__name__,
                'error_value': str(e),
                'traceback': getattr(e, 'traceback', [])
            })
            return False, metadata
            
        except Exception as e:
            metadata['end_time'] = time.time()
            metadata['execution_time'] = metadata['end_time'] - metadata['start_time']
            metadata['errors'].append({
                'type': 'GeneralError',
                'error_name': e.__class__.__name__,
                'error_value': str(e),
                'traceback': []
            })
            return False, metadata
    
    def validate_notebook_structure(self, notebook_path: Path) -> Tuple[bool, List[str]]:
        """
        Validate notebook structure without execution.
        
        Parameters
        ----------
        notebook_path : Path
            Path to the notebook file
            
        Returns
        -------
        valid : bool
            True if notebook structure is valid
        issues : list of str
            List of validation issues found
        """
        issues = []
        
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
                
            # Check basic structure
            if not notebook.cells:
                issues.append("Notebook has no cells")
                
            # Check for code cells
            code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
            if not code_cells:
                issues.append("Notebook has no code cells")
                
            # Check for markdown cells (educational content)
            markdown_cells = [cell for cell in notebook.cells if cell.cell_type == 'markdown']
            if len(markdown_cells) < 2:
                issues.append("Notebook has minimal markdown content (expected educational notebooks)")
                
            # Check for imports in first code cell
            if code_cells:
                first_code_cell = code_cells[0]
                if 'import' not in first_code_cell.source:
                    issues.append("First code cell should contain imports")
                    
            # Check kernel specification
            if 'kernelspec' not in notebook.metadata:
                issues.append("Notebook missing kernel specification")
                
        except json.JSONDecodeError:
            issues.append("Notebook is not valid JSON")
        except Exception as e:
            issues.append(f"Error reading notebook: {e}")
            
        return len(issues) == 0, issues


def find_notebooks(
    examples_dir: Path, 
    pattern: str = "*.ipynb",
    exclude_patterns: Optional[List[str]] = None
) -> List[Path]:
    """
    Find all notebooks in the examples directory.
    
    Parameters
    ----------
    examples_dir : Path
        Path to examples directory
    pattern : str, default="*.ipynb"
        Glob pattern for notebook files
    exclude_patterns : list of str, optional
        Patterns to exclude (e.g., checkpoints, test files)
        
    Returns
    -------
    notebooks : list of Path
        List of found notebook paths
    """
    if exclude_patterns is None:
        exclude_patterns = [
            "*checkpoint*",
            "*test*",
            "*Test*",
            "*debug*",
            "*Debug*"
        ]
    
    # Find all notebooks
    all_notebooks = list(examples_dir.rglob(pattern))
    
    # Filter out excluded patterns
    filtered_notebooks = []
    for notebook in all_notebooks:
        exclude = False
        for exclude_pattern in exclude_patterns:
            if notebook.match(exclude_pattern):
                exclude = True
                break
        if not exclude:
            filtered_notebooks.append(notebook)
    
    return sorted(filtered_notebooks)


def categorize_notebooks(notebooks: List[Path]) -> Dict[str, List[Path]]:
    """
    Categorize notebooks by difficulty level based on directory structure.
    
    Parameters
    ----------
    notebooks : list of Path
        List of notebook paths
        
    Returns
    -------
    categories : dict
        Dictionary mapping category names to lists of notebook paths
    """
    categories = {
        'basic': [],
        'intermediate': [],
        'advanced': [],
        'specialized': [],
        'other': []
    }
    
    for notebook in notebooks:
        # Get the parent directory name
        parent_dir = notebook.parent.name.lower()
        
        if parent_dir in categories:
            categories[parent_dir].append(notebook)
        else:
            categories['other'].append(notebook)
    
    return categories