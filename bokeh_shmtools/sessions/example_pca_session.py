#!/usr/bin/env python3
"""
Example script demonstrating session management for PCA outlier detection workflow.

This script creates and saves an example session that matches the 
PCA outlier detection notebook workflow.
"""

import sys
from pathlib import Path

# Add parent directories to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent))

from bokeh_shmtools.sessions import SessionManager, WorkflowSession, ParameterSchema


def create_pca_example_session():
    """Create an example PCA outlier detection session."""
    
    # Create session manager
    manager = SessionManager()
    
    # Create new session
    session = manager.create_new_session(
        name="PCA Outlier Detection Example",
        description="Complete PCA-based outlier detection workflow using 3-story structure data. "
                   "Demonstrates AR model feature extraction, PCA training, and damage detection.",
        author="SHMTools Development Team"
    )
    
    # Step 1: Load data
    step1 = session.add_function_step(
        function_name="load_3story_data",
        category="Data Loading",
        description="Load 3-story structure dataset with multiple damage conditions"
    )
    # No parameters needed - function returns standard dataset
    
    # Step 2: Extract AR model features
    step2 = session.add_function_step(
        function_name="ar_model", 
        category="Feature Extraction",
        description="Extract AR(15) model RMSE features from time series data"
    )
    
    # Add parameters for AR model
    session.update_step_parameter(2, "data", "", "variable_reference", "Step 1: load_3story_data")
    session.update_step_parameter(2, "order", 15, "user")
    
    # Step 3: Learn PCA model
    step3 = session.add_function_step(
        function_name="learn_pca",
        category="Classification - Parametric Detectors", 
        description="Learn PCA transformation from baseline training data"
    )
    
    # PCA training uses RMSE features from AR model (baseline conditions only)
    session.update_step_parameter(3, "X", "", "variable_reference", "Step 2: ar_model_rms_residuals_fv")
    
    # Step 4: Score test data
    step4 = session.add_function_step(
        function_name="score_pca",
        category="Classification - Parametric Detectors",
        description="Score test data using learned PCA model for outlier detection"
    )
    
    session.update_step_parameter(4, "Y", "", "variable_reference", "Step 2: ar_model_rms_residuals_fv") 
    session.update_step_parameter(4, "model", "", "variable_reference", "Step 3: learn_pca")
    
    # Validate session
    errors = session.validate()
    if errors:
        print("Session validation errors:")
        for error in errors:
            print(f"  - {error}")
        return None
    
    # Save session
    file_path = manager.save_session(session)
    print(f"Created example session: {file_path}")
    
    return session, file_path


def create_comparison_session():
    """Create a session comparing multiple outlier detection methods."""
    
    manager = SessionManager()
    
    session = manager.create_new_session(
        name="Multi-Method Outlier Detection Comparison",
        description="Compare PCA, Mahalanobis, SVD, and Factor Analysis outlier detection methods "
                   "on the same 3-story structure dataset.",
        author="SHMTools Development Team"
    )
    
    # Step 1: Load data (same as PCA example)
    session.add_function_step("load_3story_data", "Data Loading")
    
    # Step 2: Extract features (same as PCA example)  
    session.add_function_step("ar_model", "Feature Extraction")
    session.update_step_parameter(2, "data", "", "variable_reference", "Step 1: load_3story_data")
    session.update_step_parameter(2, "order", 15, "user")
    
    # Steps 3-4: PCA method
    session.add_function_step("learn_pca", "Classification - Parametric Detectors")
    session.update_step_parameter(3, "X", "", "variable_reference", "Step 2: ar_model_rms_residuals_fv")
    
    session.add_function_step("score_pca", "Classification - Parametric Detectors") 
    session.update_step_parameter(4, "Y", "", "variable_reference", "Step 2: ar_model_rms_residuals_fv")
    session.update_step_parameter(4, "model", "", "variable_reference", "Step 3: learn_pca")
    
    # Steps 5-6: Mahalanobis method
    session.add_function_step("learn_mahalanobis", "Classification - Parametric Detectors")
    session.update_step_parameter(5, "X", "", "variable_reference", "Step 2: ar_model_rms_residuals_fv")
    
    session.add_function_step("score_mahalanobis", "Classification - Parametric Detectors")
    session.update_step_parameter(6, "Y", "", "variable_reference", "Step 2: ar_model_rms_residuals_fv")
    session.update_step_parameter(6, "model", "", "variable_reference", "Step 5: learn_mahalanobis")
    
    # Steps 7-8: SVD method
    session.add_function_step("learn_svd", "Classification - Parametric Detectors") 
    session.update_step_parameter(7, "X", "", "variable_reference", "Step 2: ar_model_rms_residuals_fv")
    
    session.add_function_step("score_svd", "Classification - Parametric Detectors")
    session.update_step_parameter(8, "Y", "", "variable_reference", "Step 2: ar_model_rms_residuals_fv") 
    session.update_step_parameter(8, "model", "", "variable_reference", "Step 7: learn_svd")
    
    file_path = manager.save_session(session)
    print(f"Created comparison session: {file_path}")
    
    return session, file_path


def demonstrate_session_operations():
    """Demonstrate various session management operations."""
    
    print("=== SHMTools Session Management Demo ===\n")
    
    # Create example sessions
    print("1. Creating example sessions...")
    pca_session, pca_path = create_pca_example_session()
    comp_session, comp_path = create_comparison_session()
    
    # List sessions
    print("\n2. Listing all sessions...")
    manager = SessionManager()
    sessions = manager.list_sessions()
    
    for session_info in sessions:
        print(f"  - {session_info['name']}")
        print(f"    Description: {session_info['description'][:80]}...")
        print(f"    Steps: {session_info['num_steps']}")
        print(f"    Modified: {session_info['modified'][:19]}")
        print(f"    File: {Path(session_info['file_path']).name}")
        print()
    
    # Demonstrate loading and validation
    print("3. Loading and validating PCA session...")
    loaded_session = manager.load_session(pca_path)
    
    print(f"  Loaded: {loaded_session.name}")
    print(f"  Steps: {len(loaded_session.steps)}")
    
    validation_errors = loaded_session.validate()
    if validation_errors:
        print("  Validation errors:")
        for error in validation_errors:
            print(f"    - {error}")
    else:
        print("  ✓ Session is valid")
    
    # Show step details
    print("\n4. Workflow steps:")
    for step in loaded_session.steps:
        print(f"  Step {step.step_number}: {step.function_name}")
        print(f"    Category: {step.category}")
        print(f"    Status: {step.status}")
        if step.parameters:
            print(f"    Parameters:")
            for param in step.parameters:
                if param.parameter_type == 'variable_reference':
                    print(f"      {param.name}: -> {param.source_step}")
                else:
                    print(f"      {param.name}: {param.value} ({param.parameter_type})")
        print()
    
    # Demonstrate recent sessions
    print("5. Recent sessions:")
    recent = manager.get_recent_sessions()
    for path in recent:
        print(f"  - {Path(path).name}")
    
    print(f"\nSession files saved in: {manager.sessions_dir}")
    
    return manager, loaded_session


if __name__ == "__main__":
    demonstrate_session_operations()