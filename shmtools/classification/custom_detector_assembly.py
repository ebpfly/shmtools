"""
Custom Detector Assembly System

This module provides functionality to assemble custom outlier detectors by
combining learning and scoring functions from different detector categories.
Based on the original MATLAB assembleOutlierDetector_shm function.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
import datetime
import json
from pathlib import Path

from .detector_registry import detector_registry
from .high_level_detection import detect_outlier_shm
from . import outlier_detection, nonparametric, semiparametric


def assemble_outlier_detector_shm(suffix: Optional[str] = None, 
                                 detector_type: Optional[str] = None,
                                 detector_name: Optional[str] = None,
                                 parameters: Optional[Dict[str, Any]] = None,
                                 interactive: bool = True) -> Dict[str, Any]:
    """
    Assemble custom outlier detector with interactive or programmatic configuration.
    
    .. meta::
        :category: Classification - Detector Assembly
        :display_name: Assemble Outlier Detector
        :verbose_call: [Assembled Detector Structure] = Assemble Outlier Detector (Suffix, Base Directory)
        :matlab_equivalent: assembleOutlierDetector_shm
        :complexity: Advanced
        :data_type: Configuration
        :output_type: Assembled Detector
        
    This function allows users to create custom outlier detectors by mixing 
    and matching learning/scoring function pairs from three categories:
    1. Parametric Detectors: PCA, Mahalanobis, SVD, Factor Analysis
    2. Non-parametric Detectors: Kernel density estimation with various kernels
    3. Semi-parametric Detectors: Gaussian Mixture Models with partitioning
    
    Parameters
    ----------
    suffix : str, optional
        Suffix for the generated training function name. If None, uses timestamp.
        
        .. gui::
            :widget: text_input
            :default: None
            :description: Optional suffix for detector name
            
    detector_type : str, optional
        Type of detector to assemble ('parametric', 'nonparametric', 'semiparametric').
        If None and interactive=True, user will be prompted.
        
        .. gui::
            :widget: select
            :options: ["parametric", "nonparametric", "semiparametric"]
            :default: None
            :description: Category of outlier detector
            
    detector_name : str, optional
        Specific detector to use. If None and interactive=True, user will be prompted.
        
        .. gui::
            :widget: select
            :options: []  # Will be populated based on detector_type
            :default: None
            :description: Specific detector algorithm
            
    parameters : dict, optional
        Dictionary of parameters for the detector functions. If None and 
        interactive=True, user will be prompted for parameter values.
        
        .. gui::
            :widget: json_editor
            :default: {}
            :description: Parameters for detector configuration
            
    interactive : bool, default True
        Whether to use interactive prompts for missing parameters.
        
        .. gui::
            :widget: checkbox
            :default: True
            :description: Enable interactive parameter configuration
    
    Returns
    -------
    assembled_detector : dict
        Dictionary containing:
        - 'type': detector type ('parametric', 'nonparametric', 'semiparametric')
        - 'name': detector name (e.g., 'pca', 'mahalanobis')
        - 'learn_function': name of learning function
        - 'score_function': name of scoring function
        - 'parameters': configured parameters
        - 'training_function': generated training function
        - 'suffix': detector suffix used
        - 'assembly_info': metadata about assembly process
        
    Examples
    --------
    >>> # Interactive assembly
    >>> detector = assemble_outlier_detector_shm()
    
    >>> # Programmatic assembly
    >>> detector = assemble_outlier_detector_shm(
    ...     detector_type='parametric',
    ...     detector_name='pca', 
    ...     parameters={'percentage': 99},
    ...     interactive=False
    ... )
    
    >>> # Use assembled detector
    >>> training_func = detector['training_function']
    >>> models = training_func(training_features)
    >>> results = detect_outlier_shm(test_features, models=models)
    """
    
    # Generate suffix if not provided
    if suffix is None:
        suffix = datetime.datetime.now().strftime('%m%d%yT%H%M%S')
    
    print("\\n=== SHMTools Custom Detector Assembly ===")
    print("Assembling custom outlier detector with configurable components.\\n")
    
    # Get detector type
    if detector_type is None and interactive:
        detector_type = _prompt_detector_type()
    elif detector_type is None:
        raise ValueError("detector_type must be specified when interactive=False")
    
    detector_type = detector_type.lower()
    if detector_type not in detector_registry.get_detector_types():
        raise ValueError(f"Invalid detector_type: {detector_type}")
    
    # Get detector name
    if detector_name is None and interactive:
        detector_name = _prompt_detector_name(detector_type)
    elif detector_name is None:
        raise ValueError("detector_name must be specified when interactive=False")
    
    # Validate detector availability
    if not detector_registry.validate_detector_availability(detector_type, detector_name):
        available = list(detector_registry.get_detectors_by_type(detector_type).keys())
        raise ValueError(f"Detector {detector_name} not available. Available: {available}")
    
    # Get function names
    learn_func_name, score_func_name = detector_registry.get_function_pair(
        detector_type, detector_name)
    
    # Get parameters
    if parameters is None and interactive:
        parameters = _prompt_parameters(detector_type, detector_name, learn_func_name)
    elif parameters is None:
        parameters = {}
    
    # Create the training function
    training_function = _create_training_function(
        detector_type, detector_name, learn_func_name, score_func_name, 
        parameters, suffix)
    
    # Assemble the detector structure
    assembled_detector = {
        'type': detector_type,
        'name': detector_name,
        'learn_function': learn_func_name,
        'score_function': score_func_name,
        'parameters': parameters,
        'training_function': training_function,
        'suffix': suffix,
        'assembly_info': {
            'assembly_date': datetime.datetime.now().isoformat(),
            'interactive': interactive,
            'registry_version': '1.0.0'
        }
    }
    
    print(f"\\n✅ Custom detector '{detector_name}_{suffix}' assembled successfully!")
    print(f"   Type: {detector_type}")
    print(f"   Learning function: {learn_func_name}")
    print(f"   Scoring function: {score_func_name}")
    if parameters:
        print(f"   Parameters: {parameters}")
    
    return assembled_detector


def _prompt_detector_type() -> str:
    """Prompt user to select detector type."""
    types = detector_registry.get_detector_types()
    
    print("Available detector types:")
    for i, detector_type in enumerate(types, 1):
        print(f"  [{i}] {detector_type.title()} detectors")
    
    while True:
        try:
            choice = input("\\nEnter detector type (number): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(types):
                selected_type = types[idx]
                print(f"You chose: {selected_type} detectors\\n")
                return selected_type
            else:
                print(f"Please enter a number between 1 and {len(types)}")
        except ValueError:
            print("Please enter a valid number")


def _prompt_detector_name(detector_type: str) -> str:
    """Prompt user to select specific detector."""
    detectors = detector_registry.get_detectors_by_type(detector_type)
    detector_names = list(detectors.keys())
    
    print(f"Available {detector_type} detectors:")
    for i, name in enumerate(detector_names, 1):
        info = detectors[name]
        print(f"  [{i}] {info['display_name']} ({name})")
        print(f"      {info['description']}")
    
    while True:
        try:
            choice = input("\\nChoose detector (number): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(detector_names):
                selected_name = detector_names[idx]
                info = detectors[selected_name]
                print(f"You chose: {info['display_name']}\\n")
                return selected_name
            else:
                print(f"Please enter a number between 1 and {len(detector_names)}")
        except ValueError:
            print("Please enter a valid number")


def _prompt_parameters(detector_type: str, detector_name: str, learn_func_name: str) -> Dict[str, Any]:
    """Prompt user to configure parameters for the learning function."""
    parameters = {}
    
    try:
        # Get function signature
        signature = detector_registry.get_function_signature(
            detector_type, detector_name, 'learn')
        param_names = list(signature.parameters.keys())
        
        print(f"{learn_func_name} parameters:")
        for i, param_name in enumerate(param_names, 1):
            param = signature.parameters[param_name]
            default_str = f" (default: {param.default})" if param.default != param.empty else ""
            print(f"  [{i}] {param_name}{default_str}")
        
        print("\\nConfigure parameters (press Enter to skip parameter):")
        
        for i, param_name in enumerate(param_names, 1):
            if param_name == 'features':  # Skip the main data parameter
                continue
                
            prompt = f"  Enter value for '{param_name}' (or press Enter to skip): "
            value_str = input(prompt).strip()
            
            if value_str:
                # Try to parse the value
                try:
                    # Try numeric conversion
                    if '.' in value_str:
                        parameters[param_name] = float(value_str)
                    else:
                        parameters[param_name] = int(value_str)
                except ValueError:
                    # Keep as string
                    parameters[param_name] = value_str
        
    except Exception as e:
        print(f"Warning: Could not inspect function parameters: {e}")
        print("Using default parameters.")
    
    return parameters


def _create_training_function(detector_type: str, detector_name: str, 
                            learn_func_name: str, score_func_name: str,
                            parameters: Dict[str, Any], suffix: str) -> Callable:
    """Create a training function for the assembled detector."""
    
    # Get the actual function objects
    if detector_type == 'parametric':
        module = outlier_detection
    elif detector_type == 'nonparametric':
        module = nonparametric  
    elif detector_type == 'semiparametric':
        module = semiparametric
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
    
    learn_func = getattr(module, learn_func_name)
    score_func = getattr(module, score_func_name)
    
    def assembled_training_function(features: np.ndarray, 
                                  k: Optional[int] = None,
                                  confidence: float = 0.95,
                                  model_filename: Optional[str] = None,
                                  dist_for_scores: Optional[str] = None) -> Dict[str, Any]:
        """
        Custom assembled training function.
        
        This function was automatically generated by assemble_outlier_detector_shm
        and combines the selected learning and scoring functions with configured parameters.
        
        Parameters
        ----------
        features : ndarray
            Training feature matrix (instances x features)
        k : int, optional
            Number of clusters or components (for applicable detectors)
        confidence : float, default 0.95
            Confidence level for threshold calculation
        model_filename : str, optional
            Filename to save the model
        dist_for_scores : str, optional
            Distribution type for threshold calculation
            
        Returns
        -------
        models : dict
            Dictionary containing:
            - 'detector_model': trained detector model
            - 'threshold': detection threshold
            - 'learn_function': learning function used
            - 'score_function': scoring function used  
            - 'assembly_info': detector assembly information
        """
        
        print(f"**** Training custom detector: {detector_name}_{suffix} ****")
        
        # Prepare parameters for learning function
        learn_params = parameters.copy()
        if k is not None and 'k' in learn_params:
            learn_params['k'] = k
            
        # Handle special parameter conversions
        if detector_type == 'nonparametric' and 'kernel_function' in learn_params:
            # Convert kernel function name to actual function
            kernel_name = learn_params['kernel_function']
            if isinstance(kernel_name, str):
                kernel_func_name = f"{kernel_name}_kernel_shm"
                if hasattr(nonparametric, kernel_func_name):
                    learn_params['kernel_fun'] = getattr(nonparametric, kernel_func_name)
                    del learn_params['kernel_function']
                    
        if detector_type == 'nonparametric' and 'bandwidth_method' in learn_params:
            # Convert bandwidth method string to integer
            bw_method = learn_params['bandwidth_method']
            if isinstance(bw_method, str):
                if bw_method.lower() == 'scott':
                    learn_params['bs_method'] = 2
                elif bw_method.lower() == 'silverman':
                    learn_params['bs_method'] = 1
                del learn_params['bandwidth_method']
                
        if detector_type == 'semiparametric':
            # Handle partitioning algorithm parameter
            if 'partitioning_algorithm' in learn_params:
                partition_alg = learn_params['partitioning_algorithm']
                if isinstance(partition_alg, str):
                    if partition_alg.lower() == 'kmeans':
                        # kmeans is the default, so we can pass None
                        learn_params['partition_fun'] = None
                    elif partition_alg.lower() == 'kmedians':
                        # Use k_medians_shm function if available
                        if hasattr(semiparametric, 'k_medians_shm'):
                            learn_params['partition_fun'] = getattr(semiparametric, 'k_medians_shm')
                    del learn_params['partitioning_algorithm']
                    
            # Handle n_components parameter mapping to k
            if 'n_components' in learn_params and 'k' not in learn_params:
                learn_params['k'] = learn_params['n_components']
                del learn_params['n_components']
        
        # Train the detector model
        detector_model = learn_func(features, **learn_params)
        
        # Calculate threshold using the scoring function
        score_result = score_func(features, detector_model)
        
        # Handle different return types from scoring functions
        if isinstance(score_result, tuple):
            # Some scoring functions return (scores, residuals) or similar
            training_scores = score_result[0]
        else:
            training_scores = score_result
        
        # Calculate threshold based on confidence level
        # Note: lower scores indicate outliers, so we use (1-confidence) percentile
        percentile_value = (1 - confidence) * 100
        
        if dist_for_scores == 'normal':
            # Use normal distribution assumption
            threshold = np.percentile(training_scores, percentile_value)
        else:
            # Use empirical percentile
            threshold = np.percentile(training_scores, percentile_value)
        
        # Create the complete model structure compatible with detect_outlier_shm
        models = {
            'dModel': detector_model,
            'p_threshold': threshold,
            'cModel': None,  # Confidence model not used in custom detectors
            'scoreFun': score_func,
            'distForScores': dist_for_scores,
            # Additional metadata
            'detector_type': detector_type,
            'detector_name': detector_name,
            'learn_function': learn_func_name,
            'score_function': score_func_name,
            'assembly_info': {
                'detector_type': detector_type,
                'detector_name': detector_name,
                'suffix': suffix,
                'parameters': parameters,
                'confidence': confidence,
                'dist_for_scores': dist_for_scores
            }
        }
        
        # Save model if filename provided
        if model_filename is not None:
            import pickle
            with open(model_filename, 'wb') as f:
                pickle.dump(models, f)
            print(f"Model saved to: {model_filename}")
        
        return models
    
    # Add metadata to the function
    assembled_training_function.__name__ = f"train_detector_{suffix}"
    assembled_training_function.__doc__ = f"""
    Custom training function for {detector_name} detector (suffix: {suffix}).
    
    Generated by assemble_outlier_detector_shm on {datetime.datetime.now().isoformat()}.
    Uses {learn_func_name} for learning and {score_func_name} for scoring.
    """
    
    return assembled_training_function


def save_detector_assembly(assembled_detector: Dict[str, Any], 
                         filename: Optional[str] = None) -> str:
    """
    Save assembled detector configuration to file.
    
    Parameters
    ----------
    assembled_detector : dict
        Assembled detector structure from assemble_outlier_detector_shm
    filename : str, optional
        Output filename. If None, auto-generates based on detector info.
        
    Returns
    -------
    filename : str
        Path to saved configuration file
    """
    if filename is None:
        suffix = assembled_detector['suffix']
        detector_name = assembled_detector['name']
        filename = f"assembled_detector_{detector_name}_{suffix}.json"
    
    # Create a serializable version (remove the function)
    config = assembled_detector.copy()
    config['training_function'] = f"<function train_detector_{assembled_detector['suffix']}>"
    
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Detector assembly saved to: {filename}")
    return filename


def load_detector_assembly(filename: str) -> Dict[str, Any]:
    """
    Load assembled detector configuration from file.
    
    Note: The training function will need to be regenerated using
    assemble_outlier_detector_shm with the saved parameters.
    
    Parameters
    ----------
    filename : str
        Path to saved configuration file
        
    Returns
    -------
    config : dict
        Loaded detector configuration
    """
    with open(filename, 'r') as f:
        config = json.load(f)
    
    print(f"Detector assembly loaded from: {filename}")
    print("Note: Use assemble_outlier_detector_shm to regenerate the training function.")
    
    return config