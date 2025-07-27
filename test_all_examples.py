#!/usr/bin/env python3
"""
Comprehensive test script for all shmtools example notebooks.

This script runs the complete test suite for example notebooks with
different levels of detail and error handling.
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"Return code: {result.returncode}")
        return result.returncode == 0
        
    except Exception as e:
        print(f"ERROR running command: {e}")
        return False


def main():
    """Run comprehensive notebook testing."""
    print("SHMTools Example Notebook Test Suite")
    print("=====================================")
    
    # Change to the shmtools-python directory
    script_dir = Path(__file__).parent
    print(f"Working directory: {script_dir}")
    
    # Test categories to run
    test_categories = [
        # Quick validation tests (no execution)
        {
            "name": "Quick Structure Validation",
            "cmd": ["python", "-m", "pytest", 
                   "tests/test_notebooks/test_all_notebooks.py::TestAllNotebooksQuick",
                   "-v"],
            "required": True
        },
        
        # Data availability tests
        {
            "name": "Data Availability Check", 
            "cmd": ["python", "-m", "pytest",
                   "tests/test_notebooks/test_data_validation.py::TestDataAvailability",
                   "-v", "-s"],
            "required": True
        },
        
        # Basic notebook discovery
        {
            "name": "Notebook Discovery and Categorization",
            "cmd": ["python", "-m", "pytest",
                   "tests/test_notebooks/test_all_notebooks.py::TestNotebookExecution",
                   "-v", "-s"],
            "required": True
        },
        
        # Basic notebook execution (data-independent)
        {
            "name": "Basic Notebooks (No Data Required)",
            "cmd": ["python", "-m", "pytest",
                   "-m", "basic_notebook and not requires_data",
                   "tests/test_notebooks/",
                   "-v", "-s"],
            "required": False
        },
        
        # Basic notebook execution (data-dependent)
        {
            "name": "Basic Notebooks (Data Required)",
            "cmd": ["python", "-m", "pytest", 
                   "-m", "basic_notebook and requires_data",
                   "tests/test_notebooks/",
                   "-v", "-s"],
            "required": False
        },
        
        # Intermediate notebooks
        {
            "name": "Intermediate Notebooks",
            "cmd": ["python", "-m", "pytest",
                   "-m", "intermediate_notebook", 
                   "tests/test_notebooks/",
                   "-v", "-s"],
            "required": False
        },
        
        # Advanced notebooks (structure only)
        {
            "name": "Advanced Notebooks (Structure Check)",
            "cmd": ["python", "-m", "pytest",
                   "tests/test_notebooks/test_all_notebooks.py::TestAdvancedNotebooks::test_advanced_notebooks_structure",
                   "-v", "-s"],
            "required": False
        },
        
        # Data loading tests
        {
            "name": "Data Loading Validation",
            "cmd": ["python", "-m", "pytest",
                   "tests/test_notebooks/test_data_validation.py::TestDataLoading",
                   "-v", "-s"],
            "required": False
        },
    ]
    
    # Results tracking
    results = []
    required_failures = []
    
    for test_category in test_categories:
        name = test_category["name"]
        cmd = test_category["cmd"]
        required = test_category["required"]
        
        success = run_command(cmd, name)
        results.append({
            "name": name,
            "success": success,
            "required": required
        })
        
        if required and not success:
            required_failures.append(name)
    
    # Summary report
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r["success"])
    required_tests = sum(1 for r in results if r["required"])
    successful_required = sum(1 for r in results if r["required"] and r["success"])
    
    print(f"Total test categories: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Required tests: {required_tests}")
    print(f"Required tests passed: {successful_required}")
    
    print("\nDetailed Results:")
    for result in results:
        status = "✓ PASS" if result["success"] else "✗ FAIL"
        req_marker = " (REQUIRED)" if result["required"] else ""
        print(f"  {status} {result['name']}{req_marker}")
    
    # Final assessment
    print(f"\n{'='*60}")
    if required_failures:
        print("❌ CRITICAL FAILURES DETECTED")
        print("The following required tests failed:")
        for failure in required_failures:
            print(f"  - {failure}")
        print("\nThese issues must be resolved before proceeding.")
        return False
    else:
        print("✅ ALL REQUIRED TESTS PASSED")
        optional_failures = [r["name"] for r in results if not r["success"] and not r["required"]]
        if optional_failures:
            print(f"\nOptional test failures ({len(optional_failures)}):")
            for failure in optional_failures:
                print(f"  - {failure}")
            print("\nThese failures indicate missing functionality but don't block basic operation.")
        print("\nBasic test infrastructure is working correctly!")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)