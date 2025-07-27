"""
Pytest configuration and fixtures for notebook testing.
"""

import os
import pytest
from pathlib import Path
from typing import Dict, Any, List

from .utils.notebook_runner import NotebookRunner, find_notebooks, categorize_notebooks


@pytest.fixture(scope="session")
def examples_dir():
    """Path to the examples directory."""
    # Get the root directory of the project
    test_dir = Path(__file__).parent
    project_root = test_dir.parent.parent
    examples_dir = project_root / "examples"
    
    if not examples_dir.exists():
        pytest.skip(f"Examples directory not found: {examples_dir}")
    
    return examples_dir


@pytest.fixture(scope="session")
def notebooks_dir(examples_dir):
    """Path to the notebooks directory."""
    notebooks_dir = examples_dir / "notebooks"
    
    if not notebooks_dir.exists():
        pytest.skip(f"Notebooks directory not found: {notebooks_dir}")
    
    return notebooks_dir


@pytest.fixture(scope="session")
def data_dir(examples_dir):
    """Path to the data directory."""
    data_dir = examples_dir / "data"
    
    if not data_dir.exists():
        pytest.skip(f"Data directory not found: {data_dir}")
    
    return data_dir


@pytest.fixture(scope="session")
def all_notebooks(notebooks_dir) -> List[Path]:
    """Find all notebook files in the examples directory."""
    notebooks = find_notebooks(notebooks_dir)
    
    if not notebooks:
        pytest.skip("No notebooks found in examples directory")
    
    return notebooks


@pytest.fixture(scope="session")
def categorized_notebooks(all_notebooks) -> Dict[str, List[Path]]:
    """Categorize notebooks by difficulty level."""
    return categorize_notebooks(all_notebooks)


@pytest.fixture(scope="session")
def notebook_runner():
    """Create a notebook runner instance for testing."""
    return NotebookRunner(timeout=600, kernel_name="python3")


@pytest.fixture
def temp_working_dir(tmp_path):
    """Create a temporary working directory for notebook execution."""
    return tmp_path


@pytest.fixture(scope="session")
def required_data_files():
    """List of required data files for the notebooks."""
    return [
        "data3SS.mat",
        "dataSensorDiagnostic.mat", 
        "data_CBM.mat",
        "data_example_ActiveSense.mat",
        "data_OSPExampleModal.mat"
    ]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "notebook: mark test as a notebook execution test"
    )
    config.addinivalue_line(
        "markers", "basic_notebook: mark test as basic level notebook"
    )
    config.addinivalue_line(
        "markers", "intermediate_notebook: mark test as intermediate level notebook"
    )
    config.addinivalue_line(
        "markers", "advanced_notebook: mark test as advanced level notebook"
    )
    config.addinivalue_line(
        "markers", "specialized_notebook: mark test as specialized level notebook"
    )
    config.addinivalue_line(
        "markers", "requires_data: mark test as requiring example datasets"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark notebook tests based on their category."""
    for item in items:
        # Auto-mark notebook tests
        if "test_notebook" in item.nodeid:
            item.add_marker(pytest.mark.notebook)
            
        # Auto-mark tests requiring data
        if any(marker in item.nodeid.lower() for marker in ["pca", "mahalanobis", "svd", "data"]):
            item.add_marker(pytest.mark.requires_data)
            
        # Auto-mark by difficulty level
        if "basic" in item.nodeid:
            item.add_marker(pytest.mark.basic_notebook)
        elif "intermediate" in item.nodeid:
            item.add_marker(pytest.mark.intermediate_notebook)
        elif "advanced" in item.nodeid:
            item.add_marker(pytest.mark.advanced_notebook)
        elif "specialized" in item.nodeid:
            item.add_marker(pytest.mark.specialized_notebook)


def check_data_availability(data_dir: Path, required_files: List[str]) -> Dict[str, bool]:
    """
    Check which required data files are available.
    
    Parameters
    ----------
    data_dir : Path
        Path to data directory
    required_files : list of str
        List of required data file names
        
    Returns
    -------
    availability : dict
        Dictionary mapping file names to availability status
    """
    availability = {}
    for filename in required_files:
        file_path = data_dir / filename
        availability[filename] = file_path.exists()
    
    return availability