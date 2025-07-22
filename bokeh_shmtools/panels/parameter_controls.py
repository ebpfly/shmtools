"""
Parameter controls panel for configuring function parameters.

This panel provides dynamic parameter input controls for the selected
workflow step, similar to the right panel in mFUSE.
"""

from bokeh.models import (
    Div, Column, TextInput, Slider, Select, Button,
    CheckboxGroup, NumericInput
)
from bokeh.layouts import column
import inspect
from typing import Dict, Any, Optional, List
from bokeh_shmtools.utils.docstring_parser import parse_shmtools_docstring, parse_verbose_call


class ParameterControlsPanel:
    """
    Panel for configuring parameters of selected workflow steps.
    
    This panel dynamically generates appropriate input controls
    based on the selected function's parameter specifications.
    """
    
    def __init__(self):
        """Initialize the parameter controls panel."""
        self.current_function = None
        self.current_function_obj = None  # Store actual function object for introspection
        self.parameter_widgets = {}
        self.workflow_steps = []  # Reference to workflow steps for output selection
        self.panel = self._create_panel()
        
    def _create_panel(self):
        """
        Create the parameter controls panel components.
        
        Returns
        -------
        panel : bokeh.models.Panel
            Parameter controls panel.
        """
        # Panel title
        title = Div(text="<h3>Parameter Controls</h3>", width=300)
        
        # Function info
        function_info = Div(
            text="<p>Select a workflow step to configure parameters.</p>",
            width=280
        )
        
        # Parameter controls container
        params_container = Column(width=280, height=400)
        
        # Apply/Reset buttons
        apply_btn = Button(
            label="Apply Changes",
            button_type="primary",
            width=130,
            disabled=True
        )
        reset_btn = Button(
            label="Reset to Defaults", 
            width=130,
            disabled=True
        )
        
        apply_btn.on_click(self._apply_parameters)
        reset_btn.on_click(self._reset_parameters)
        
        # Store references
        self.function_info = function_info
        self.params_container = params_container
        self.apply_btn = apply_btn
        self.reset_btn = reset_btn
        
        # Create panel layout
        panel_content = column(
            title,
            function_info,
            params_container,
            apply_btn,
            reset_btn,
            min_width=350,
            sizing_mode="stretch_both"
        )
        
        return panel_content  # Return the layout directly instead of wrapping in Panel
    
    def set_workflow_steps(self, workflow_steps: List[Dict[str, Any]]):
        """Set reference to workflow steps for output selection."""
        self.workflow_steps = workflow_steps
    
    def get_available_outputs(self, before_step: int) -> List[str]:
        """Get available outputs from previous workflow steps."""
        outputs = ["[Select Input]"]  # Default option
        
        for i, step in enumerate(self.workflow_steps):
            if i >= before_step:  # Only consider steps before current one
                break
            step_name = f"Step {i+1}: {step['function']}"
            outputs.append(step_name)
            
            # Add specific output references that match session format
            function_name = step['function']
            step_number = i + 1
            
            # Add multi-output function specific references with human-readable names
            from bokeh_shmtools.workflows.execution_engine import WorkflowExecutor
            executor = WorkflowExecutor()
            
            # Get human-readable names first, fallback to technical names
            human_readable_names = executor._get_human_readable_output_names(function_name)
            technical_names = executor._get_output_names(function_name)
            function_display_name = executor._get_function_display_name(function_name)
            
            if human_readable_names:
                # Use human-readable names as primary options
                for human_name in human_readable_names:
                    display_text = f"Step {step_number}: {function_display_name} → {human_name}"
                    # Store the reference in a way that can be resolved later
                    outputs.append(display_text)
                    
                # Also add technical names for backward compatibility (marked as technical)
                if technical_names:
                    for tech_name in technical_names:
                        display_text = f"Step {step_number}: {function_name}_{tech_name} (technical)"
                        outputs.append(display_text)
            elif technical_names:
                # Fallback to technical names only
                for tech_name in technical_names:
                    outputs.append(f"Step {step_number}: {function_name}_{tech_name}")
            else:
                # Fallback for functions without multi-output
                outputs.append(f"Step {step_number}: {function_display_name}")
                    
        return outputs
    
    def _get_function_object(self, function_name: str):
        """Get the actual function object for introspection."""
        try:
            # Import and get function from execution engine mapping
            from bokeh_shmtools.workflows.execution_engine import WorkflowExecutor
            executor = WorkflowExecutor()
            return executor._get_function(function_name)
        except Exception as e:
            print(f"Could not get function object for {function_name}: {e}")
            return None
    
    def _get_display_name_for_function(self, function_name: str, func_obj=None):
        """
        Get human-readable display name for a function.
        
        Parameters
        ----------
        function_name : str
            Technical function name
        func_obj : callable, optional
            Function object for metadata parsing
            
        Returns
        -------
        str
            Human-readable display name
        """
        if func_obj:
            try:
                metadata = parse_shmtools_docstring(func_obj)
                if metadata:
                    # First try explicit display_name
                    if metadata.display_name:
                        return metadata.display_name
                        
                    # Next try parsing verbose_call
                    if metadata.verbose_call:
                        parsed = parse_verbose_call(metadata.verbose_call)
                        if parsed["function_display_name"]:
                            return parsed["function_display_name"]
            except Exception:
                pass
        
        # Fall back to technical name conversion
        return self._technical_to_readable(function_name)
    
    def _technical_to_readable(self, technical_name):
        """Convert technical function name to human-readable format."""
        # Remove _shm suffix if present
        name = technical_name.replace('_shm', '')
        
        # Split on underscores and capitalize each word
        words = name.split('_')
        
        # Handle common abbreviations
        abbrev_map = {
            'pca': 'PCA',
            'svd': 'SVD', 
            'ar': 'AR',
            'rms': 'RMS',
            'fft': 'FFT',
            'stft': 'STFT',
            'psd': 'PSD',
            'cbm': 'CBM'
        }
        
        readable_words = []
        for word in words:
            if word.lower() in abbrev_map:
                readable_words.append(abbrev_map[word.lower()])
            else:
                readable_words.append(word.capitalize())
                
        return ' '.join(readable_words)
    
    def _parameter_to_readable(self, param_name: str) -> str:
        """
        Convert parameter name to human-readable format.
        
        Parameters
        ----------
        param_name : str
            Technical parameter name like 'n_components' or 'max_iter'
            
        Returns
        -------
        str
            Human-readable name like 'Number of Components' or 'Max Iterations'
        """
        # Handle common parameter patterns
        special_cases = {
            'X': 'Input Data',
            'Y': 'Target Data',
            'n_components': 'Number of Components',
            'n_neighbors': 'Number of Neighbors',
            'max_iter': 'Max Iterations',
            'tol': 'Tolerance',
            'alpha': 'Alpha',
            'beta': 'Beta',
            'gamma': 'Gamma',
            'eps': 'Epsilon',
            'C': 'Regularization Parameter',
            'nu': 'Nu Parameter',
            'kernel': 'Kernel Type',
            'degree': 'Polynomial Degree',
            'coef0': 'Coefficient 0',
            'shrinking': 'Use Shrinking',
            'probability': 'Enable Probability',
            'cache_size': 'Cache Size',
            'class_weight': 'Class Weight',
            'verbose': 'Verbose Output',
            'n_jobs': 'Number of Jobs',
            'random_state': 'Random State',
            'shuffle': 'Shuffle Data',
            'normalize': 'Normalize',
            'fit_intercept': 'Fit Intercept',
            'copy_X': 'Copy X',
            'warm_start': 'Warm Start',
            'positive': 'Force Positive',
            'selection': 'Selection Method',
            'squared': 'Use Squared',
            'fs': 'Sampling Frequency',
            'nperseg': 'Points per Segment',
            'noverlap': 'Overlap Points',
            'window': 'Window Type',
            'detrend': 'Detrend Method',
            'scaling': 'Scaling Method',
            'return_onesided': 'Return One-Sided',
            'axis': 'Axis',
            'average': 'Average Method',
            'nfft': 'FFT Length',
            'f_min': 'Min Frequency',
            'f_max': 'Max Frequency',
            'order': 'Model Order',
            'method': 'Method',
            'criterion': 'Criterion',
            'center': 'Center Data',
            'scale': 'Scale Data',
            'stand': 'Standardize',
            'standardize': 'Standardize',
            'with_mean': 'Center to Mean',
            'with_std': 'Scale to Unit Variance',
            'copy': 'Copy Data',
            'quantile_range': 'Quantile Range',
            'n_quantiles': 'Number of Quantiles',
            'output_distribution': 'Output Distribution',
            'subsample': 'Subsample Size',
            'threshold': 'Threshold',
            'contamination': 'Contamination',
            'support_fraction': 'Support Fraction',
            'keep_dims': 'Keep Dimensions',
            'whiten': 'Whiten',
            'svd_solver': 'SVD Solver',
            'iterated_power': 'Iterated Power',
            'n_oversamples': 'Oversamples',
            'power_iteration_normalizer': 'Power Iteration Normalizer',
            'flip': 'Flip',
            'data': 'Input Data',
            'labels': 'Labels',
            'features': 'Features',
            'targets': 'Target Values',
            'weights': 'Weights',
            'scores': 'Scores',
            'predictions': 'Predictions',
            'residuals': 'Residuals',
            'coefficients': 'Coefficients',
            'intercept': 'Intercept',
            'support': 'Support Vectors',
            'dual_coef': 'Dual Coefficients',
            'coef': 'Coefficients',
            'classes': 'Classes',
            'n_support': 'Number of Support Vectors',
            'probabilities': 'Probabilities',
            'log_likelihood': 'Log Likelihood',
            'n_iter': 'Number of Iterations',
            'dual_gap': 'Dual Gap',
            'eps_abs': 'Absolute Epsilon',
            'eps_rel': 'Relative Epsilon',
            'rho': 'Rho',
            'adaptive_rho': 'Adaptive Rho',
            'psi': 'Psi',
            'max_iters': 'Max Iterations',
            'abstol': 'Absolute Tolerance',
            'reltol': 'Relative Tolerance',
            'feastol': 'Feasibility Tolerance',
            'alpha_tol': 'Alpha Tolerance',
            'lambda_tol': 'Lambda Tolerance',
            'precompute': 'Precompute',
            'xy': 'XY Product',
            'copy_Gram': 'Copy Gram Matrix',
            'overwrite_a': 'Overwrite A',
            'overwrite_b': 'Overwrite B',
            'overwrite_x': 'Overwrite X',
            'overwrite_y': 'Overwrite Y',
            'lower': 'Lower Triangle',
            'sym_pos': 'Symmetric Positive',
            'unit_diagonal': 'Unit Diagonal'
        }
        
        # Check special cases first
        if param_name in special_cases:
            return special_cases[param_name]
        
        # Handle snake_case to Title Case
        words = param_name.split('_')
        
        # Handle common abbreviations
        abbrev_map = {
            'n': 'Number of',
            'num': 'Number of',
            'max': 'Maximum',
            'min': 'Minimum',
            'tol': 'Tolerance',
            'eps': 'Epsilon',
            'coef': 'Coefficient',
            'std': 'Standard Deviation',
            'var': 'Variance',
            'cov': 'Covariance',
            'corr': 'Correlation',
            'pca': 'PCA',
            'svd': 'SVD',
            'fft': 'FFT',
            'stft': 'STFT',
            'rms': 'RMS',
            'psd': 'PSD',
            'ar': 'AR',
            'ma': 'MA',
            'arma': 'ARMA',
            'arima': 'ARIMA',
            'acf': 'ACF',
            'pacf': 'PACF',
            'ccf': 'CCF',
            'df': 'Degrees of Freedom',
            'dof': 'Degrees of Freedom',
            'aic': 'AIC',
            'bic': 'BIC',
            'rmse': 'RMSE',
            'mse': 'MSE',
            'mae': 'MAE',
            'r2': 'R²',
            'adj': 'Adjusted',
            'inv': 'Inverse',
            'sqrt': 'Square Root',
            'abs': 'Absolute',
            'rel': 'Relative',
            'cum': 'Cumulative',
            'dist': 'Distance',
            'sim': 'Similarity',
            'dissim': 'Dissimilarity',
            'init': 'Initial',
            'params': 'Parameters',
            'hyperparams': 'Hyperparameters',
            'config': 'Configuration',
            'opts': 'Options',
            'args': 'Arguments',
            'kwargs': 'Keyword Arguments',
            'attr': 'Attributes',
            'props': 'Properties',
            'vals': 'Values',
            'feat': 'Feature',
            'feats': 'Features',
            'pred': 'Prediction',
            'preds': 'Predictions',
            'prob': 'Probability',
            'probs': 'Probabilities',
            'freq': 'Frequency',
            'freqs': 'Frequencies',
            'coeff': 'Coefficient',
            'coeffs': 'Coefficients',
            'param': 'Parameter',
            'params': 'Parameters',
            'dim': 'Dimension',
            'dims': 'Dimensions',
            'vec': 'Vector',
            'vecs': 'Vectors',
            'mat': 'Matrix',
            'mats': 'Matrices',
            'arr': 'Array',
            'arrs': 'Arrays',
            'val': 'Value',
            'vals': 'Values',
            'opt': 'Option',
            'opts': 'Options',
            'est': 'Estimate',
            'ests': 'Estimates',
            'iter': 'Iteration',
            'iters': 'Iterations',
            'conv': 'Convergence',
            'diverge': 'Divergence',
            'reg': 'Regularization',
            'regul': 'Regularization',
            'norm': 'Normalization',
            'denorm': 'Denormalization',
            'preproc': 'Preprocessing',
            'postproc': 'Postprocessing',
            'xform': 'Transform',
            'trans': 'Transform',
            'inv': 'Inverse',
            'pseudo': 'Pseudo',
            'diag': 'Diagonal',
            'offdiag': 'Off-Diagonal',
            'sym': 'Symmetric',
            'asym': 'Asymmetric',
            'pos': 'Positive',
            'neg': 'Negative',
            'nonneg': 'Non-Negative',
            'nonpos': 'Non-Positive',
            'incr': 'Increasing',
            'decr': 'Decreasing',
            'mono': 'Monotonic',
            'const': 'Constant',
            'lin': 'Linear',
            'nonlin': 'Nonlinear',
            'quad': 'Quadratic',
            'cubic': 'Cubic',
            'poly': 'Polynomial',
            'exp': 'Exponential',
            'log': 'Logarithmic',
            'trig': 'Trigonometric',
            'sin': 'Sine',
            'cos': 'Cosine',
            'tan': 'Tangent',
            'asin': 'Arc Sine',
            'acos': 'Arc Cosine',
            'atan': 'Arc Tangent',
            'sinh': 'Hyperbolic Sine',
            'cosh': 'Hyperbolic Cosine',
            'tanh': 'Hyperbolic Tangent',
            'rad': 'Radians',
            'deg': 'Degrees',
            'rect': 'Rectangular',
            'polar': 'Polar',
            'cart': 'Cartesian',
            'sph': 'Spherical',
            'cyl': 'Cylindrical'
        }
        
        readable_words = []
        skip_next = False
        
        for i, word in enumerate(words):
            if skip_next:
                skip_next = False
                continue
                
            lower_word = word.lower()
            
            # Check if this word should expand to multiple words
            if lower_word in abbrev_map:
                expanded = abbrev_map[lower_word]
                # Only use expansion if it's at the beginning or makes sense in context
                if i == 0 or (i > 0 and words[i-1].lower() not in ['is', 'has', 'with', 'use']):
                    readable_words.append(expanded)
                else:
                    readable_words.append(word.capitalize())
            else:
                readable_words.append(word.capitalize())
        
        result = ' '.join(readable_words)
        
        # Clean up any double spaces
        result = ' '.join(result.split())
        
        return result
    
    def _get_parameter_metadata(self, function_name: str) -> Dict[str, Dict[str, Any]]:
        """Get human-readable parameter metadata from docstrings."""
        # Try _shm version first for better docstring metadata
        shm_func_obj = self._get_function_object(f"{function_name}_shm")
        if shm_func_obj:
            try:
                metadata = parse_shmtools_docstring(shm_func_obj)
                if metadata and metadata.parameters:
                    param_info = {}
                    for param_spec in metadata.parameters:
                        param_info[param_spec.name] = {
                            'description': param_spec.description,
                            'type_hint': param_spec.type_hint,
                            'default': param_spec.default,
                            'widget': param_spec.widget,
                            'widget_params': param_spec.widget_params
                        }
                    return param_info
            except Exception:
                pass
        
        # Fallback to regular function
        func_obj = self._get_function_object(function_name)
        if not func_obj:
            return {}
        
        try:
            # Parse docstring to get parameter metadata
            metadata = parse_shmtools_docstring(func_obj)
            if not metadata or not metadata.parameters:
                return {}
            
            param_info = {}
            for param_spec in metadata.parameters:
                param_info[param_spec.name] = {
                    'description': param_spec.description,
                    'type_hint': param_spec.type_hint,
                    'default': param_spec.default,
                    'widget': param_spec.widget,
                    'widget_params': param_spec.widget_params
                }
            
            return param_info
            
        except Exception as e:
            print(f"Error parsing docstring for {function_name}: {e}")
            return {}
    
    def _create_parameter_widget_from_signature(self, param_name: str, param_info, current_value, step_index: int):
        """Create parameter widget based on function signature introspection."""
        # Get available workflow step outputs
        available_outputs = self.get_available_outputs(step_index)
        
        # Get human-readable parameter information from docstring
        param_metadata = self._get_parameter_metadata(self.current_function)
        param_meta = param_metadata.get(param_name, {})
        
        # Use human-readable description if available
        param_description = param_meta.get('description', '')
        param_type = param_info.annotation
        has_default = param_info.default != inspect.Parameter.empty
        
        # Get type name for widget creation
        type_name = getattr(param_type, '__name__', str(param_type))
        
        # Create human-readable parameter name
        # First check if we have a human-readable name from verbose_call
        if hasattr(self, 'human_readable_param_names') and param_name in self.human_readable_param_names:
            readable_param_name = self.human_readable_param_names[param_name]
        else:
            readable_param_name = self._parameter_to_readable(param_name)
        
        # Create title with parameter name and type
        title = f"{readable_param_name} ({type_name})"
        
        if has_default:
            title += f" [default: {param_info.default}]"
        
        # Handle variable reference (from session loading)
        selected_output = available_outputs[0]  # Default to first option
        if isinstance(current_value, str) and current_value.startswith("Step "):
            # This is a variable reference from a loaded session
            if current_value in available_outputs:
                selected_output = current_value
            else:
                # Try to find matching step reference
                for output in available_outputs:
                    if current_value in output or output in current_value:
                        selected_output = output
                        break
        
        # Create dropdown for workflow step output selection
        output_select = Select(
            title=f"{title} - Data Source:",
            value=selected_output,
            options=available_outputs,
            width=280
        )
        
        # Create direct value input based on type
        value_widget = None
        
        # Determine the direct value (not variable reference)
        direct_value = None
        if current_value is not None and not (isinstance(current_value, str) and current_value.startswith("Step ")):
            direct_value = current_value
        elif has_default:
            direct_value = param_info.default
        
        if param_type == int or type_name == 'int':
            default_int = int(direct_value) if direct_value is not None else 1
            value_widget = NumericInput(
                title=f"{title} - Direct Value:",
                value=default_int,
                width=280
            )
        elif param_type == float or type_name == 'float':
            default_float = float(direct_value) if direct_value is not None else 1.0
            value_widget = NumericInput(
                title=f"{title} - Direct Value:",
                value=default_float,
                width=280
            )
        else:
            # Default to text input for other types
            default_str = str(direct_value) if direct_value is not None else ""
            value_widget = TextInput(
                title=f"{title} - Direct Value:",
                value=default_str,
                width=280
            )
        
        # Create parameter section with description
        param_header = f"<b>{readable_param_name}</b>"
        if param_description:
            param_header += f"<br><small><em>{param_description}</em></small>"
        
        # Return container with both options
        container = Column(
            Div(text=param_header),
            output_select,
            value_widget,
            width=300,
            margin=(10, 0)
        )
        
        # Create a wrapper object to store widget references
        class ParameterWidget:
            def __init__(self, container, output_select, value_widget, param_name):
                self.container = container
                self.output_select = output_select
                self.value_widget = value_widget
                self.param_name = param_name
        
        return ParameterWidget(container, output_select, value_widget, param_name)
    
    def set_function(self, function_name: str, parameters: Dict[str, Any], step_index: int = 0):
        """
        Set the current function and display its parameters.
        
        Parameters
        ----------
        function_name : str
            Name of the selected function.
        parameters : dict
            Current parameter values.
        step_index : int
            Index of this step in workflow (for output selection).
        """
        self.current_function = function_name
        self.parameter_widgets = {}
        
        # Get the actual function object for introspection
        self.current_function_obj = self._get_function_object(function_name)
        
        # Get function metadata for human-readable info
        func_obj = self._get_function_object(function_name)
        function_description = "Configure parameters for this function."
        human_readable_param_names = {}
        
        if func_obj:
            try:
                metadata = parse_shmtools_docstring(func_obj)
                if metadata:
                    function_description = metadata.brief_description or function_description
                    
                    # Extract human-readable parameter names from verbose_call
                    if metadata.verbose_call:
                        parsed = parse_verbose_call(metadata.verbose_call)
                        input_names = parsed.get('input_names', [])
                        
                        # Map parameter names to human-readable names
                        if self.current_function_obj:
                            sig = inspect.signature(self.current_function_obj)
                            param_list = list(sig.parameters.keys())
                            for i, (param_name, human_name) in enumerate(zip(param_list, input_names)):
                                if i < len(input_names):
                                    human_readable_param_names[param_name] = human_name
            except Exception:
                pass
        
        # Store human-readable parameter names for later use
        self.human_readable_param_names = human_readable_param_names
        
        # Update function info with human-readable description and name
        display_name = self._get_display_name_for_function(function_name, func_obj)
        self.function_info.text = f"<h4>{display_name}</h4><p>{function_description}</p>"
        
        # Clear existing parameter controls
        self.params_container.children = []
        
        if self.current_function_obj:
            # Introspect function signature
            try:
                sig = inspect.signature(self.current_function_obj)
                widgets = []
                
                # Map session parameter names to function signature parameters  
                param_name_mapping = {
                    'data': 'X',  # Session 'data' maps to function 'X'
                    'Y': 'X',     # Session 'Y' maps to function 'X' (for score functions)
                }
                
                for param_name, param_info in sig.parameters.items():
                    # Find the session parameter value for this function parameter
                    session_param_value = None
                    
                    # Check direct match first
                    if param_name in parameters:
                        session_param_value = parameters[param_name]
                    else:
                        # Check reverse mapping (session name -> function name)
                        for session_name, func_name in param_name_mapping.items():
                            if func_name == param_name and session_name in parameters:
                                session_param_value = parameters[session_name]
                                break
                    
                    widget_wrapper = self._create_parameter_widget_from_signature(
                        param_name, param_info, session_param_value, step_index
                    )
                    if widget_wrapper:
                        widgets.append(widget_wrapper.container)  # Add container to display
                        self.parameter_widgets[param_name] = widget_wrapper  # Store wrapper for value extraction
                
                # Update container
                self.params_container.children = widgets
                
            except Exception as e:
                error_msg = Div(text=f"<p style='color: red;'>Error inspecting function: {e}</p>")
                self.params_container.children = [error_msg]
        else:
            error_msg = Div(text=f"<p style='color: red;'>Function {function_name} not found</p>")
            self.params_container.children = [error_msg]
        
        # Enable buttons
        self.apply_btn.disabled = False
        self.reset_btn.disabled = False
    
    def _get_parameter_specs(self, function_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Get parameter specifications for a function.
        
        Parameters
        ----------
        function_name : str
            Function name.
            
        Returns
        -------
        specs : dict
            Parameter specifications.
        """
        # TODO: Dynamically introspect function signatures
        # For now, return example specifications
        
        specs = {
            "psd_welch": {
                "fs": {"type": "float", "default": 1.0, "min": 0.1, "max": 1000000, "title": "Sampling Frequency (Hz)"},
                "window": {"type": "select", "default": "hann", "options": ["hann", "hamming", "blackman"], "title": "Window Type"},
                "nperseg": {"type": "int", "default": None, "min": 8, "max": 8192, "title": "Segment Length"},
            },
            "bandpass_filter": {
                "lowcut": {"type": "float", "default": 10.0, "min": 0.1, "max": 1000, "title": "Low Cutoff (Hz)"},
                "highcut": {"type": "float", "default": 100.0, "min": 1.0, "max": 10000, "title": "High Cutoff (Hz)"},
                "order": {"type": "int", "default": 4, "min": 1, "max": 10, "title": "Filter Order"},
                "zero_phase": {"type": "bool", "default": True, "title": "Zero Phase"},
            },
            "ar_model": {
                "order": {"type": "int", "default": 10, "min": 1, "max": 50, "title": "AR Model Order"},
            },
        }
        
        return specs.get(function_name, {})
    
    def _create_parameter_widget(self, param_name: str, spec: Dict[str, Any], current_value: Any):
        """
        Create appropriate input widget for parameter.
        
        Parameters
        ----------
        param_name : str
            Parameter name.
        spec : dict
            Parameter specification.
        current_value : any
            Current parameter value.
            
        Returns
        -------
        widget : bokeh.models.Widget
            Input widget for parameter.
        """
        param_type = spec.get("type", "float")
        default_value = current_value if current_value is not None else spec.get("default")
        title = spec.get("title", param_name)
        
        if param_type == "float":
            return NumericInput(
                title=title,
                value=float(default_value) if default_value else 0.0,
                low=spec.get("min"),
                high=spec.get("max"),
                width=250
            )
        elif param_type == "int":
            return NumericInput(
                title=title,
                value=int(default_value) if default_value else 0,
                low=spec.get("min"),
                high=spec.get("max"),
                width=250
            )
        elif param_type == "select":
            return Select(
                title=title,
                value=str(default_value) if default_value else spec["options"][0],
                options=spec["options"],
                width=250
            )
        elif param_type == "bool":
            return CheckboxGroup(
                labels=[title],
                active=[0] if default_value else [],
                width=250
            )
        else:
            return TextInput(
                title=title,
                value=str(default_value) if default_value else "",
                width=250
            )
    
    def _apply_parameters(self):
        """Apply current parameter values."""
        if not self.current_function:
            return
            
        # Extract values from widgets
        parameters = {}
        for param_name, widget in self.parameter_widgets.items():
            if hasattr(widget, 'value'):
                parameters[param_name] = widget.value
            elif hasattr(widget, 'active'):
                parameters[param_name] = len(widget.active) > 0
        
        print(f"Applying parameters for {self.current_function}: {parameters}")
        # Note: Actual workflow step update is handled in app.py _setup_panel_communication()
    
    def _reset_parameters(self):
        """Reset parameters to default values."""
        if not self.current_function:
            return
            
        param_specs = self._get_parameter_specs(self.current_function)
        
        for param_name, widget in self.parameter_widgets.items():
            spec = param_specs.get(param_name, {})
            default_value = spec.get("default")
            
            if hasattr(widget, 'value') and default_value is not None:
                widget.value = default_value
            elif hasattr(widget, 'active'):
                widget.active = [0] if default_value else []
    
    def clear_selection(self):
        """Clear the current function selection."""
        self.current_function = None
        self.parameter_widgets = {}
        
        # Reset UI
        self.function_info.text = "<p>Select a workflow step to configure parameters.</p>"
        self.params_container.children = []
        self.apply_btn.disabled = True
        self.reset_btn.disabled = True
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Get current parameter values from widgets.
        
        Returns
        -------
        parameters : dict
            Current parameter values.
        """
        parameters = {}
        for param_name, widget_wrapper in self.parameter_widgets.items():
            # Handle new wrapper structure with output_select and value_widget
            if hasattr(widget_wrapper, 'output_select') and hasattr(widget_wrapper, 'value_widget'):
                # Check if user selected a workflow step output or direct value
                output_selection = widget_wrapper.output_select.value
                if output_selection != "[Select Input]":
                    # User selected a workflow step output
                    parameters[param_name] = output_selection
                else:
                    # User wants to use direct value
                    if hasattr(widget_wrapper.value_widget, 'value'):
                        parameters[param_name] = widget_wrapper.value_widget.value
                    elif hasattr(widget_wrapper.value_widget, 'active'):
                        parameters[param_name] = len(widget_wrapper.value_widget.active) > 0
            # Fallback for old widget structure
            elif hasattr(widget_wrapper, 'value'):
                parameters[param_name] = widget_wrapper.value
            elif hasattr(widget_wrapper, 'active'):
                parameters[param_name] = len(widget_wrapper.active) > 0
        return parameters