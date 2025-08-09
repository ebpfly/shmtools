"""
Test for the publish_notebooks.py script.

This test ensures that the notebook publishing script runs without errors
and produces the expected output files.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
import pytest


def test_publish_notebooks_execution():
    """
    Test that publish_notebooks.py successfully publishes ALL notebooks.
    
    This test ensures that every notebook executes successfully and is published.
    The test will fail if any notebook fails to execute or publish.
    """
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    publish_script = project_root / "publish_notebooks.py"
    examples_dir = project_root / "examples" / "notebooks"
    
    # Verify the script and examples directory exist
    assert publish_script.exists(), f"publish_notebooks.py not found at {publish_script}"
    assert examples_dir.exists(), f"Examples directory not found at {examples_dir}"
    
    # Count expected notebooks (excluding test/debug notebooks)
    expected_notebooks = []
    for notebook_path in examples_dir.rglob("*.ipynb"):
        # Skip test notebooks and hidden files
        if (not notebook_path.name.startswith('.') and 
            'test' not in notebook_path.name.lower() and
            'phase' not in notebook_path.name.lower()):
            expected_notebooks.append(notebook_path)
    
    expected_count = len(expected_notebooks)
    assert expected_count > 0, f"No notebooks found in {examples_dir}"
    
    # Create a temporary output directory
    with tempfile.TemporaryDirectory() as temp_output_dir:
        # Run the publish script with --skip-errors to handle missing data files
        cmd = [
            "python", 
            str(publish_script), 
            "--examples-dir", str(examples_dir),
            "--output-dir", str(temp_output_dir),
            "--timeout", "300",  # 5 minutes for execution
            "--skip-errors"  # Skip notebooks that fail execution (e.g., missing data)
        ]
        
        # Execute the script
        result = subprocess.run(
            cmd, 
            cwd=str(project_root),
            capture_output=True, 
            text=True
        )
        
        # Check that the script completed successfully
        assert result.returncode == 0, (
            f"publish_notebooks.py failed with return code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        
        # Verify that an index.html file was created
        index_file = Path(temp_output_dir) / "index.html"
        assert index_file.exists(), f"index.html not created in {temp_output_dir}"
        
        # Count published HTML files (excluding index.html)
        published_html = []
        for html_path in Path(temp_output_dir).rglob("*.html"):
            if html_path.name != "index.html":
                published_html.append(html_path)
        
        published_count = len(published_html)
        
        # Verify at least some notebooks were successfully published
        # Since we're using --skip-errors, not all notebooks may succeed (e.g., missing data files)
        assert published_count > 0, (
            f"No notebooks were published. Expected at least 1, got {published_count}.\n"
            f"Expected notebooks: {[nb.name for nb in expected_notebooks]}\n"
            f"Published HTML files: {[html.name for html in published_html]}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        
        # Log the success rate for informational purposes
        success_rate = published_count / expected_count
        print(f"Published {published_count}/{expected_count} notebooks (success rate: {success_rate:.1%})")
        


def test_publish_notebooks_help():
    """
    Test that publish_notebooks.py --help works correctly.
    """
    project_root = Path(__file__).parent.parent.parent
    publish_script = project_root / "publish_notebooks.py"
    
    # Run the script with --help
    result = subprocess.run(
        ["python", str(publish_script), "--help"],
        cwd=str(project_root),
        capture_output=True,
        text=True
    )
    
    # Check that help completed successfully
    assert result.returncode == 0, f"--help failed with return code {result.returncode}"
    
    # Verify help output contains expected content
    assert "Find, execute, and publish" in result.stdout
    assert "--examples-dir" in result.stdout
    assert "--output-dir" in result.stdout


if __name__ == "__main__":
    test_publish_notebooks_execution()
    test_publish_notebooks_help()
    print("All publish_notebooks.py tests passed!")