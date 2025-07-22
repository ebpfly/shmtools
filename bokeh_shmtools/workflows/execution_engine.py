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
import time
from datetime import datetime
from bokeh_shmtools.utils.data_loader import load_3story_data, create_synthetic_data
from bokeh_shmtools.utils.docstring_parser import parse_shmtools_docstring, parse_verbose_call


class WorkflowExecutor:
    """
    Executes SHMTools workflows with proper data flow and error handling.
    """
    
    def __init__(self):
        """Initialize the workflow executor."""
        self.variables = {}  # Workspace for storing intermediate results
        self.function_cache = {}  # Cache for dynamically imported functions
        self.progress_callback = None  # Callback for progress updates
        self.execution_metrics = {}  # Track execution timing and performance
        
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
        start_time = time.time()
        total_steps = len(workflow_steps)
        
        results = {
            "success": True,
            "outputs": {},
            "errors": [],
            "step_results": [],
            "execution_time": 0,
            "total_steps": total_steps,
            "completed_steps": 0
        }
        
        self.execution_metrics = {
            "start_time": datetime.now().isoformat(),
            "step_timings": [],
            "total_variables_created": 0
        }
        
        # Notify progress callback of workflow start
        if self.progress_callback:
            self.progress_callback({
                "type": "workflow_start",
                "total_steps": total_steps,
                "start_time": self.execution_metrics["start_time"]
            })
        
        try:
            for i, step in enumerate(workflow_steps):
                # Notify progress callback of step start
                if self.progress_callback:
                    self.progress_callback({
                        "type": "step_start",
                        "step_number": i + 1,
                        "total_steps": total_steps,
                        "function_name": step.get("function", "Unknown"),
                        "progress_percent": (i / total_steps) * 100
                    })
                
                step_start_time = time.time()
                step_result = self.execute_step(step, step_number=i+1)
                step_end_time = time.time()
                
                # Track step timing
                step_timing = {
                    "step": i + 1,
                    "function": step.get("function", "Unknown"),
                    "duration": step_end_time - step_start_time,
                    "success": step_result["success"]
                }
                self.execution_metrics["step_timings"].append(step_timing)
                
                results["step_results"].append(step_result)
                
                if not step_result["success"]:
                    results["success"] = False
                    results["errors"].append(f"Step {i+1} failed: {step_result['error']}")
                    
                    # Notify progress callback of step failure
                    if self.progress_callback:
                        self.progress_callback({
                            "type": "step_error",
                            "step_number": i + 1,
                            "error": step_result['error'],
                            "duration": step_timing["duration"]
                        })
                    break
                else:
                    # Store outputs for next steps
                    step_outputs = step_result.get("outputs", {})
                    results["outputs"].update(step_outputs)
                    results["completed_steps"] = i + 1
                    
                    # Count new variables created
                    self.execution_metrics["total_variables_created"] += len(step_outputs)
                    
                    # Notify progress callback of step completion
                    if self.progress_callback:
                        self.progress_callback({
                            "type": "step_complete",
                            "step_number": i + 1,
                            "outputs": list(step_outputs.keys()),
                            "duration": step_timing["duration"],
                            "progress_percent": ((i + 1) / total_steps) * 100
                        })
                    
        except Exception as e:
            results["success"] = False
            results["errors"].append(f"Workflow execution failed: {str(e)}")
            
            # Notify progress callback of workflow failure
            if self.progress_callback:
                self.progress_callback({
                    "type": "workflow_error",
                    "error": str(e)
                })
            
        # Calculate final execution time
        end_time = time.time()
        results["execution_time"] = end_time - start_time
        self.execution_metrics["end_time"] = datetime.now().isoformat()
        self.execution_metrics["total_duration"] = results["execution_time"]
        
        # Notify progress callback of workflow completion
        if self.progress_callback:
            self.progress_callback({
                "type": "workflow_complete" if results["success"] else "workflow_failed",
                "success": results["success"],
                "total_duration": results["execution_time"],
                "completed_steps": results["completed_steps"],
                "total_steps": total_steps,
                "variables_created": self.execution_metrics["total_variables_created"],
                "metrics": self.execution_metrics
            })
            
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
            prepared_params = self._prepare_parameters(func, parameters, function_name)
            
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
            
            # For multi-output functions, also store individual outputs with human-readable names
            if isinstance(output, tuple) and len(output) > 1:
                # Get human-readable names first, fallback to technical names
                human_readable_names = self._get_human_readable_output_names(function_name)
                technical_names = self._get_output_names(function_name)
                
                print(f"Human-readable names for {function_name}: {human_readable_names}")
                print(f"Technical names for {function_name}: {technical_names}")
                
                if human_readable_names and len(human_readable_names) == len(output):
                    # Use human-readable names as primary storage
                    for i, (human_name, val) in enumerate(zip(human_readable_names, output)):
                        # Use format: AR_Model_2_RMS_Residuals_Feature_Vectors
                        indexed_name = self._make_variable_name(function_name, step_number, human_name)
                        self.variables[indexed_name] = val
                        result["outputs"][indexed_name] = val
                        print(f"Stored human-readable output '{indexed_name}': {type(val)} shape={getattr(val, 'shape', 'no shape')}")
                        
                        # Also store with technical name for backward compatibility
                        if technical_names and i < len(technical_names):
                            tech_indexed_name = f"{function_name}_{step_number}_{technical_names[i]}"
                            self.variables[tech_indexed_name] = val
                            result["outputs"][tech_indexed_name] = val
                            
                elif technical_names and len(technical_names) == len(output):
                    print(f"Using technical output names (got {len(technical_names)} names for {len(output)} outputs)")
                    for i, (name, val) in enumerate(zip(technical_names, output)):
                        indexed_name = f"{function_name}_{step_number}_{name}"
                        self.variables[indexed_name] = val
                        result["outputs"][indexed_name] = val
                        print(f"Stored technical output {indexed_name}: {type(val)} shape={getattr(val, 'shape', 'no shape')}")
                else:
                    print(f"Using generic output names (got {len(technical_names) if technical_names else 0} technical names for {len(output)} outputs)")
                    # Fallback: store with generic output names
                    for i, val in enumerate(output):
                        indexed_name = f"{function_name}_{step_number}_output_{i}"
                        self.variables[indexed_name] = val
                        result["outputs"][indexed_name] = val
                        print(f"Stored generic output {indexed_name}: {type(val)} shape={getattr(val, 'shape', 'no shape')}")
            
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
            "statistical_moments": "shmtools.core.statistics",
            "stft": "shmtools.core.spectral",
            "mahalanobis_distance": "shmtools.classification.outlier_detection",
            "pca_detector": "shmtools.classification.outlier_detection",
            "learn_pca": "shmtools.classification.outlier_detection",
            "score_pca": "shmtools.classification.outlier_detection",
            "learn_mahalanobis": "shmtools.classification.outlier_detection",
            "score_mahalanobis": "shmtools.classification.outlier_detection",
            "learn_svd": "shmtools.classification.outlier_detection",
            "score_svd": "shmtools.classification.outlier_detection",
            "learn_factor_analysis": "shmtools.classification.outlier_detection",
            "score_factor_analysis": "shmtools.classification.outlier_detection",
        }
        
        # Try to import the function
        module_path = function_modules.get(function_name)
        if module_path:
            try:
                module = importlib.import_module(module_path)
                # Try the function name directly (new naming convention)
                func = getattr(module, function_name, None)
                # Fallback to _shm suffix for backwards compatibility
                if func is None:
                    func = getattr(module, f"{function_name}_shm", None)
                    
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
    
    def _prepare_parameters(self, func: Callable, parameters: Dict[str, Any], function_name: str = None) -> Dict[str, Any]:
        """
        Prepare parameters for function execution.
        
        Parameters
        ----------
        func : callable
            The function to execute.
        parameters : dict
            Raw parameter values from UI.
        function_name : str, optional
            Name of the function for context-specific parameter handling.
            
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
            
        # Generic parameter name mapping - no function-specific assumptions
        param_name_mapping = {
            'data': ['X', 'data', 'input_data', 'signal', 'time_series'],
            'Y': ['X', 'Y', 'data', 'test_data', 'features'],
            'X': ['X', 'data', 'features', 'input_data'], 
            'model': ['model', 'trained_model'],
            'order': ['order', 'model_order', 'ar_order', 'n_order']
        }
        
        for param_name, param_value in parameters.items():
            # Find the correct parameter name for the function signature
            target_param_names = param_name_mapping.get(param_name, [param_name])
            
            matched_param = None
            for target_name in target_param_names:
                if target_name in sig.parameters:
                    matched_param = target_name
                    break
            
            if matched_param:
                # Check if this is a workflow step output reference
                if isinstance(param_value, str) and param_value.startswith("Step "):
                    # Parse "Step X: function_name" or "Step X: function_name_output" format
                    try:
                        parts = param_value.split(": ", 1)
                        if len(parts) == 2:
                            step_part = parts[0]  # "Step X"
                            reference_part = parts[1]  # "function_name" or "function_name_output"
                            step_number = int(step_part.split()[1])  # Extract X
                            
                            # Generic approach: try to parse function and output from reference
                            # Format: "functionname_outputname" -> "functionname_stepnumber_outputname"
                            
                            var_key = None
                            
                            # Strategy 1: Try human-readable format first
                            # Look for variables that match the reference part
                            print(f"    Looking for human-readable matches for '{reference_part}'")
                            
                            # Extract the meaningful part (e.g., "rms_residuals_fv" from "ar_model_rms_residuals_fv")
                            if '_' in reference_part:
                                # Try to find the output part after the first function name
                                ref_parts = reference_part.split('_')
                                if len(ref_parts) >= 3:  # e.g., ['ar', 'model', 'rms', 'residuals', 'fv']
                                    # Look for the key phrase in human readable names
                                    key_phrase = '_'.join(ref_parts[2:])  # 'rms_residuals_fv'
                                    
                                    # Check for human-readable variables containing this concept
                                    for var_name in self.variables.keys():
                                        # Check if this variable is from the right step and contains the concept
                                        if f"_{step_number}_" in var_name:
                                            # Convert human-readable name to comparable format
                                            var_parts = var_name.split(f"_{step_number}_")
                                            if len(var_parts) == 2:
                                                var_suffix = var_parts[1].lower()
                                                # Check for semantic matches
                                                if ('rms' in key_phrase.lower() and 'rms' in var_suffix and 
                                                    'residual' in key_phrase.lower() and 'residual' in var_suffix):
                                                    var_key = var_name
                                                    print(f"    Found semantic human-readable match: {var_key}")
                                                    break
                            
                            # Strategy 2: Try parsing as function_output format 
                            if var_key is None:
                                parts = reference_part.split('_')
                                print(f"    Parsing '{reference_part}' with parts: {parts}")
                                if len(parts) >= 2:
                                    # Try different split points to find function vs output
                                    for i in range(1, len(parts)):
                                        func_name = '_'.join(parts[:i])
                                        output_name = '_'.join(parts[i:])
                                        test_key = f"{func_name}_{step_number}_{output_name}"
                                        print(f"    Trying: {test_key}")
                                        if test_key in self.variables:
                                            var_key = test_key
                                            print(f"    Found match: {test_key}")
                                            break
                            
                            # Strategy 2: Direct approach
                            if var_key is None:
                                test_key = f"{reference_part}_{step_number}"
                                if test_key in self.variables:
                                    var_key = test_key
                                    
                            # Strategy 3: Try with "_out" suffix  
                            if var_key is None:
                                test_key = f"{reference_part}_{step_number}_out"
                                if test_key in self.variables:
                                    var_key = test_key
                            
                            if var_key and var_key in self.variables:
                                var_value = self.variables[var_key]
                                prepared[matched_param] = var_value
                                print(f"  Resolved {param_name} -> {matched_param} = {var_key}")
                                if hasattr(var_value, 'shape'):
                                    print(f"    Variable shape: {var_value.shape}")
                            else:
                                print(f"  Warning: Variable not found with key {var_key}: {list(self.variables.keys())}")
                                # Try semantic fallback patterns based on variable name hints
                                fallback_keys = []
                                
                                # Extract function name from reference - try to be smarter about it
                                parts = reference_part.split('_')
                                
                                # Try to find the actual function name by checking what variables exist
                                func_name = None
                                for i in range(1, len(parts) + 1):
                                    potential_func = '_'.join(parts[:i])
                                    # Check if this function name appears in our variables
                                    if any(var.startswith(f"{potential_func}_{step_number}_") for var in self.variables.keys()):
                                        func_name = potential_func
                                        output_hint = '_'.join(parts[i:]).lower() if i < len(parts) else ""
                                        break
                                
                                # Fallback to first part if no match found
                                if func_name is None:
                                    func_name = parts[0]
                                    output_hint = '_'.join(parts[1:]).lower() if len(parts) > 1 else ""
                                
                                # Use semantic hints to map to likely output indices  
                                print(f"    Semantic analysis - func: '{func_name}', hint: '{output_hint}'")
                                if any(hint in output_hint for hint in ['rms', 'residual']):
                                    # RMS/residuals are often output 1 in multi-output functions
                                    print(f"    Detected RMS/residual pattern")
                                    fallback_keys.append(f"{func_name}_{step_number}_output_1")
                                    fallback_keys.append(f"{func_name}_{step_number}_output_0")
                                elif any(hint in output_hint for hint in ['parameters', 'params', 'coeff']):
                                    # Parameters are often output 0
                                    fallback_keys.append(f"{func_name}_{step_number}_output_0")
                                    fallback_keys.append(f"{func_name}_{step_number}_output_2")
                                elif any(hint in output_hint for hint in ['model', 'trained']):
                                    # Models are often the first or only output
                                    fallback_keys.append(f"{func_name}_{step_number}_output_0")
                                    fallback_keys.append(f"{func_name}_{step_number}_out")
                                else:
                                    # Try all outputs if no semantic hint
                                    for i in range(5):
                                        fallback_keys.append(f"{func_name}_{step_number}_output_{i}")
                                    fallback_keys.append(f"{func_name}_{step_number}_out")
                                
                                found_alternative = False
                                for alt_key in fallback_keys:
                                    if alt_key in self.variables:
                                        prepared[matched_param] = self.variables[alt_key]
                                        print(f"  Found semantic fallback: {alt_key}")
                                        if hasattr(self.variables[alt_key], 'shape'):
                                            print(f"    Fallback shape: {self.variables[alt_key].shape}")
                                        found_alternative = True
                                        break
                                        
                                if not found_alternative:
                                    print(f"  No semantic fallback found for '{output_hint}'")
                        else:
                            prepared[matched_param] = param_value
                    except (ValueError, IndexError) as e:
                        print(f"  Error parsing step reference {param_value}: {e}")
                        # Don't set the parameter on error
                # Check if this is a direct variable reference
                elif isinstance(param_value, str) and param_value in self.variables:
                    var_value = self.variables[param_value]
                    prepared[matched_param] = var_value
                else:
                    prepared[matched_param] = param_value
            else:
                print(f"  Warning: Parameter '{param_name}' not found in function signature")
                    
        # Add any required parameters that weren't provided
        for param_name, param_info in sig.parameters.items():
            if param_name not in prepared:
                if param_info.default != inspect.Parameter.empty:
                    # Use default value
                    pass  # Don't add defaults, let function handle it
                elif param_name in ['data', 'X', 'x', 'signal']:
                    # Special handling for data parameter - generate test data
                    prepared[param_name] = self._generate_test_data(function_name)
                    
        return prepared
    
    def _generate_test_data(self, function_name: str = None) -> np.ndarray:
        """
        Generate test data for functions that require data input.
        
        Parameters
        ----------
        function_name : str, optional
            Name of the function requesting data, for context-specific generation.
        
        Returns
        -------
        data : np.ndarray
            Synthetic time series data.
        """
        if function_name == "ar_model":
            # AR model expects 3D data: (time_points, channels, conditions)
            # Generate multi-channel, multi-condition test data
            t = np.linspace(0, 10, 1000)  # 1000 time points
            channels = 4  # 4 channels
            conditions = 10  # 10 conditions
            
            data = np.zeros((len(t), channels, conditions))
            
            for ch in range(channels):
                for cond in range(conditions):
                    # Different frequency and noise for each channel/condition
                    freq = 2 + ch * 0.5 + cond * 0.1
                    noise_level = 0.1 + cond * 0.01
                    data[:, ch, cond] = (np.sin(2 * np.pi * freq * t) + 
                                       noise_level * np.random.randn(len(t)))
            
            return data
        else:
            # Default: simple 1D sine wave with noise
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
    
    def set_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Set callback function for progress updates.
        
        Parameters
        ----------
        callback : callable
            Function to call with progress updates. Should accept a dictionary
            with keys like 'type', 'step_number', 'progress_percent', etc.
        """
        self.progress_callback = callback
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """
        Get execution performance metrics.
        
        Returns
        -------
        metrics : dict
            Dictionary containing timing and performance data.
        """
        return self.execution_metrics.copy()
    
    def clear_variables(self):
        """Clear all variables from the workspace."""
        self.variables.clear()
        self.execution_metrics.clear()
    
    def _get_output_names(self, function_name: str) -> Optional[List[str]]:
        """
        Get the named outputs for multi-output functions from docstring metadata.
        
        Parameters
        ----------
        function_name : str
            Name of the function.
            
        Returns
        -------
        output_names : list of str or None
            List of output variable names for the function, or None if not available.
        """
        # Get output names from docstring metadata only - no hardcoded mappings
        func = self._get_function(function_name)
        if func:
            try:
                metadata = parse_shmtools_docstring(func)
                if metadata and metadata.returns:
                    return [return_spec.name for return_spec in metadata.returns]
            except Exception as e:
                print(f"Warning: Could not parse docstring for {function_name}: {e}")
        
        # Return None if no docstring metadata available
        return None

    def _get_human_readable_output_names(self, function_name: str) -> Optional[List[str]]:
        """
        Get human-readable output names from verbose_call metadata.
        
        Parameters
        ----------
        function_name : str
            Name of the function.
            
        Returns
        -------
        output_names : list of str or None
            List of human-readable output names from verbose_call, or None if not available.
        """
        func = self._get_function(function_name)
        if func:
            try:
                metadata = parse_shmtools_docstring(func)
                if metadata and metadata.verbose_call:
                    verbose_info = parse_verbose_call(metadata.verbose_call)
                    return verbose_info.get('output_names', [])
            except Exception as e:
                print(f"Warning: Could not parse verbose call for {function_name}: {e}")
        
        # Fallback to technical names if verbose call not available
        return self._get_output_names(function_name)

    def _get_human_readable_input_names(self, function_name: str) -> Optional[List[str]]:
        """
        Get human-readable input names from verbose_call metadata.
        
        Parameters
        ----------
        function_name : str
            Name of the function.
            
        Returns
        -------
        input_names : list of str or None
            List of human-readable input names from verbose_call, or None if not available.
        """
        func = self._get_function(function_name)
        if func:
            try:
                metadata = parse_shmtools_docstring(func)
                if metadata and metadata.verbose_call:
                    verbose_info = parse_verbose_call(metadata.verbose_call)
                    return verbose_info.get('input_names', [])
            except Exception as e:
                print(f"Warning: Could not parse verbose call for {function_name}: {e}")
        
        return None

    def _get_function_display_name(self, function_name: str) -> str:
        """
        Get human-readable function display name from verbose_call metadata.
        
        Parameters
        ----------
        function_name : str
            Name of the function.
            
        Returns
        -------
        display_name : str
            Human-readable function display name, or technical name if not available.
        """
        func = self._get_function(function_name)
        if func:
            try:
                metadata = parse_shmtools_docstring(func)
                if metadata and metadata.verbose_call:
                    verbose_info = parse_verbose_call(metadata.verbose_call)
                    display_name = verbose_info.get('function_display_name', '')
                    if display_name:
                        return display_name
                # Fallback to display_name from metadata
                if metadata.display_name:
                    return metadata.display_name
            except Exception as e:
                print(f"Warning: Could not parse function display name for {function_name}: {e}")
        
        # Fallback to technical function name
        return function_name.replace('_', ' ').title()

    def _make_variable_name(self, function_name: str, step_number: int, human_readable_name: str) -> str:
        """
        Create a clean variable name from human-readable text.
        
        Converts "RMS Residuals Feature Vectors" -> "AR_Model_2_RMS_Residuals_Feature_Vectors"
        """
        # Clean up the human-readable name for use as variable name
        clean_name = human_readable_name.replace(' ', '_').replace('-', '_')
        # Remove special characters except underscores and alphanumeric
        import re
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '', clean_name)
        
        # Get function display name and clean it
        function_display = self._get_function_display_name(function_name)
        clean_function = function_display.replace(' ', '_').replace('-', '_')
        clean_function = re.sub(r'[^a-zA-Z0-9_]', '', clean_function)
        
        return f"{clean_function}_{step_number}_{clean_name}"
    
    def _get_output_descriptions(self, function_name: str) -> Dict[str, str]:
        """Get human-readable descriptions for function outputs."""
        # Try _shm version first for better docstring metadata
        shm_func = self._get_function(f"{function_name}_shm")
        if shm_func:
            try:
                metadata = parse_shmtools_docstring(shm_func)
                if metadata and metadata.returns:
                    return {return_spec.name: return_spec.description for return_spec in metadata.returns}
            except Exception:
                pass
        
        # Fallback to regular function
        func = self._get_function(function_name)
        if func:
            try:
                metadata = parse_shmtools_docstring(func)
                if metadata and metadata.returns:
                    return {return_spec.name: return_spec.description for return_spec in metadata.returns}
            except Exception:
                pass
        return {}