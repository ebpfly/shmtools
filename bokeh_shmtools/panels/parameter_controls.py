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
            
            # If step has specific outputs, add those too
            if 'outputs' in step and step['outputs']:
                for output_name in step['outputs'].keys():
                    outputs.append(f"{step_name}.{output_name}")
                    
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
    
    def _create_parameter_widget_from_signature(self, param_name: str, param_info, current_value, step_index: int):
        """Create parameter widget based on function signature introspection."""
        # Get available workflow step outputs
        available_outputs = self.get_available_outputs(step_index)
        
        # Determine parameter type from annotation
        param_type = param_info.annotation
        has_default = param_info.default != inspect.Parameter.empty
        
        # Create title with type info
        type_name = getattr(param_type, '__name__', str(param_type))
        title = f"{param_name}: {type_name}"
        if has_default:
            title += f" (default: {param_info.default})"
        
        # Create dropdown for workflow step output selection
        output_select = Select(
            title=f"{title} - Data Source:",
            value=current_value if isinstance(current_value, str) and current_value in available_outputs else available_outputs[0],
            options=available_outputs,
            width=280
        )
        
        # Create direct value input based on type
        value_widget = None
        if param_type == int or type_name == 'int':
            value_widget = NumericInput(
                title=f"{title} - Direct Value:",
                value=int(current_value) if current_value and not isinstance(current_value, str) else (param_info.default if has_default else 1),
                width=280
            )
        elif param_type == float or type_name == 'float':
            value_widget = NumericInput(
                title=f"{title} - Direct Value:",
                value=float(current_value) if current_value and not isinstance(current_value, str) else (param_info.default if has_default else 1.0),
                width=280
            )
        else:
            # Default to text input for other types
            default_val = str(param_info.default) if has_default else ""
            value_widget = TextInput(
                title=f"{title} - Direct Value:",
                value=str(current_value) if current_value and not isinstance(current_value, str) else default_val,
                width=280
            )
        
        # Return container with both options
        container = Column(
            Div(text=f"<b>Parameter: {param_name}</b>"),
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
        
        # Update function info
        self.function_info.text = f"<h4>{function_name}</h4><p>Configure parameters for this function.</p>"
        
        # Clear existing parameter controls
        self.params_container.children = []
        
        if self.current_function_obj:
            # Introspect function signature
            try:
                sig = inspect.signature(self.current_function_obj)
                widgets = []
                
                for param_name, param_info in sig.parameters.items():
                    widget_wrapper = self._create_parameter_widget_from_signature(
                        param_name, param_info, parameters.get(param_name), step_index
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