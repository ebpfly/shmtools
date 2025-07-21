"""
Workflow execution engine for running SHMTools analysis workflows.

This module provides the core functionality for executing sequences of
SHMTools functions with proper parameter handling and data flow.
"""

import importlib
import inspect
import numpy as np
from typing import Dict, Any, List, Optional, Callable
import traceback
from bokeh_shmtools.utils.data_loader import load_3story_data, create_synthetic_data


class WorkflowExecutor:
    """
    Executes SHMTools workflows with proper data flow and error handling.
    """
    
    def __init__(self):
        """Initialize the workflow executor."""
        self.variables = {}  # Workspace for storing intermediate results
        self.function_cache = {}  # Cache for dynamically imported functions
        
    def execute_workflow(self, workflow_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a complete workflow.
        
        Parameters
        ----------
        workflow_steps : list
            List of workflow step dictionaries.
            
        Returns
        -------
        results : dict
            Execution results and any errors.
        """
        results = {
            "success": True,
            "outputs": {},
            "errors": [],
            "step_results": []
        }
        
        try:
            for i, step in enumerate(workflow_steps):
                step_result = self.execute_step(step, step_number=i+1)
                results["step_results"].append(step_result)
                
                if not step_result["success"]:
                    results["success"] = False
                    results["errors"].append(f"Step {i+1} failed: {step_result['error']}")
                    break
                else:
                    # Store outputs for next steps
                    step_outputs = step_result.get("outputs", {})
                    results["outputs"].update(step_outputs)
                    
        except Exception as e:
            results["success"] = False
            results["errors"].append(f"Workflow execution failed: {str(e)}")
            
        return results
    
    def execute_step(self, step: Dict[str, Any], step_number: int = 1) -> Dict[str, Any]:
        """
        Execute a single workflow step.
        
        Parameters
        ----------
        step : dict
            Step definition with function name and parameters.
        step_number : int
            Step number for variable naming.
            
        Returns
        -------
        result : dict
            Step execution result.
        """
        function_name = step["function"]
        parameters = step.get("parameters", {})
        
        result = {
            "success": False,
            "outputs": {},
            "error": None,
            "function": function_name,
            "step": step_number
        }
        
        try:
            # Get the function
            func = self._get_function(function_name)
            if func is None:
                result["error"] = f"Function '{function_name}' not found"
                return result
            
            # Prepare parameters
            print(f"Step {step_number} raw parameters: {parameters}")
            prepared_params = self._prepare_parameters(func, parameters)
            
            # Execute function
            print(f"Executing {function_name} with parameters:")
            for key, value in prepared_params.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {type(value)} shape={value.shape}")
                else:
                    print(f"  {key}: {type(value)} value={value}")
            output = func(**prepared_params)
            
            # Store output with step naming convention
            output_var_name = f"{function_name}_{step_number}_out"
            self.variables[output_var_name] = output
            result["outputs"][output_var_name] = output
            
            # Debug: Show what was stored
            if hasattr(output, 'shape'):
                print(f"Stored {output_var_name}: {type(output)} shape={output.shape}")
            else:
                print(f"Stored {output_var_name}: {type(output)}")
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            print(f"Error executing step {step_number}: {e}")
            traceback.print_exc()
            
        return result
    
    def _get_function(self, function_name: str) -> Optional[Callable]:
        """
        Get a SHMTools function by name.
        
        Parameters
        ----------
        function_name : str
            Name of the function to retrieve.
            
        Returns
        -------
        function : callable or None
            The function if found, None otherwise.
        """
        if function_name in self.function_cache:
            return self.function_cache[function_name]
        
        # Handle special data loading functions
        if function_name == "load_3story_data":
            return load_3story_data
        elif function_name == "create_synthetic_data":
            return create_synthetic_data
            
        # Map of function names to their module paths
        function_modules = {
            "psd_welch": "shmtools.core.spectral",
            "bandpass_filter": "shmtools.core.filtering", 
            "rms": "shmtools.core.statistics",
            "crest_factor": "shmtools.core.statistics", 
            "ar_model": "shmtools.features.time_series",
            "arModel_shm": "shmtools.features.time_series",
            "statistical_moments": "shmtools.core.statistics",
            "stft": "shmtools.core.spectral",
            "mahalanobis_distance": "shmtools.classification.outlier_detection",
            "pca_detector": "shmtools.classification.outlier_detection",
            "learn_pca": "shmtools.classification.outlier_detection",
            "score_pca": "shmtools.classification.outlier_detection",
            "learnPCA_shm": "shmtools.classification.outlier_detection", 
            "scorePCA_shm": "shmtools.classification.outlier_detection",
            "learnMahalanobis_shm": "shmtools.classification.outlier_detection",
            "scoreMahalanobis_shm": "shmtools.classification.outlier_detection",
        }
        
        # Try to import the function
        module_path = function_modules.get(function_name)
        if module_path:
            try:
                module = importlib.import_module(module_path)
                # Try both with and without _shm suffix
                func = getattr(module, f"{function_name}_shm", None)
                if func is None:
                    func = getattr(module, function_name, None)
                    
                if func is not None:
                    self.function_cache[function_name] = func
                    return func
            except Exception as e:
                print(f"Failed to import {function_name} from {module_path}: {e}")
        
        # If not found in predefined modules, try scanning shmtools
        try:
            import shmtools
            if hasattr(shmtools, function_name):
                func = getattr(shmtools, function_name)
                self.function_cache[function_name] = func
                return func
        except:
            pass
            
        return None
    
    def _prepare_parameters(self, func: Callable, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare parameters for function execution.
        
        Parameters
        ----------
        func : callable
            The function to execute.
        parameters : dict
            Raw parameter values from UI.
            
        Returns
        -------
        prepared_params : dict
            Parameters ready for function execution.
        """
        prepared = {}
        
        # Get function signature
        try:
            sig = inspect.signature(func)
        except Exception:
            # If we can't get signature, just pass parameters as-is
            return parameters
            
        for param_name, param_value in parameters.items():
            if param_name in sig.parameters:
                # Check if this is a workflow step output reference
                if isinstance(param_value, str) and param_value.startswith("Step "):
                    # Parse "Step X: function_name" format
                    try:
                        parts = param_value.split(": ", 1)
                        if len(parts) == 2:
                            step_part = parts[0]  # "Step X"
                            function_part = parts[1]  # "function_name"
                            step_number = int(step_part.split()[1])  # Extract X
                            
                            # Create variable key: "function_name_stepnumber_out"
                            var_key = f"{function_part}_{step_number}_out"
                            
                            if var_key in self.variables:
                                prepared[param_name] = self.variables[var_key]
                                print(f"  Resolved {param_value} -> {var_key}")
                            else:
                                print(f"  Warning: Variable {var_key} not found in: {list(self.variables.keys())}")
                                prepared[param_name] = param_value  # Fallback to original value
                        else:
                            prepared[param_name] = param_value
                    except (ValueError, IndexError) as e:
                        print(f"  Error parsing step reference {param_value}: {e}")
                        prepared[param_name] = param_value
                # Check if this is a direct variable reference
                elif isinstance(param_value, str) and param_value in self.variables:
                    var_value = self.variables[param_value]
                    prepared[param_name] = var_value
                else:
                    prepared[param_name] = param_value
                    
        # Add any required parameters that weren't provided
        for param_name, param_info in sig.parameters.items():
            if param_name not in prepared:
                if param_info.default != inspect.Parameter.empty:
                    # Use default value
                    pass  # Don't add defaults, let function handle it
                elif param_name in ['data', 'X', 'x', 'signal']:
                    # Special handling for data parameter - generate test data
                    prepared[param_name] = self._generate_test_data()
                    
        return prepared
    
    def _generate_test_data(self) -> np.ndarray:
        """
        Generate test data for functions that require data input.
        
        Returns
        -------
        data : np.ndarray
            Synthetic time series data.
        """
        # Generate a simple sine wave with noise for testing
        t = np.linspace(0, 10, 1000)
        data = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(len(t))
        return data
    
    def get_variable(self, name: str) -> Any:
        """Get a variable from the workspace."""
        return self.variables.get(name)
    
    def set_variable(self, name: str, value: Any):
        """Set a variable in the workspace."""
        self.variables[name] = value
    
    def list_variables(self) -> Dict[str, str]:
        """
        List all variables in the workspace.
        
        Returns
        -------
        variables : dict
            Mapping of variable names to their type descriptions.
        """
        return {name: str(type(value).__name__) for name, value in self.variables.items()}