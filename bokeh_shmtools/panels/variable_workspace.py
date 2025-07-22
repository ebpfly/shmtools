"""
Variable workspace panel for monitoring data flow between workflow steps.

This panel provides an interactive view of all variables created during
workflow execution, with tagging and connection visualization.
"""

from bokeh.models import (
    Div, DataTable, TableColumn, ColumnDataSource, Button, Select, 
    TextInput, Tabs, TabPanel, Column, Row
)
from bokeh.layouts import column, row
from typing import Dict, Any, List, Optional
import numpy as np


class VariableWorkspacePanel:
    """
    Panel for monitoring variables and data flow in workflows.
    
    This panel shows all variables created during execution with
    metadata, connections, and tagging capabilities.
    """
    
    def __init__(self, executor=None):
        """Initialize the variable workspace panel."""
        self.executor = executor
        self.variables_data = {}
        self.variable_tags = {}  # Store user-defined tags for variables
        self.variable_connections = {}  # Track variable relationships
        
        self.panel = self._create_panel()
        
    def _create_panel(self):
        """
        Create the variable workspace panel components.
        
        Returns
        -------
        panel : bokeh.models.Panel
            Variable workspace panel.
        """
        # Panel title
        title = Div(text="<h3>Variable Workspace</h3>", width=600)
        
        # Workspace tabs
        variables_tab = self._create_variables_tab()
        connections_tab = self._create_connections_tab()
        tags_tab = self._create_tags_tab()
        
        workspace_tabs = Tabs(tabs=[variables_tab, connections_tab, tags_tab])
        
        # Control buttons
        refresh_btn = Button(label="Refresh", button_type="primary", width=100)
        clear_btn = Button(label="Clear All", button_type="warning", width=100)
        export_btn = Button(label="Export Data", button_type="success", width=100)
        
        refresh_btn.on_click(self._refresh_variables)
        clear_btn.on_click(self._clear_variables) 
        export_btn.on_click(self._export_variables)
        
        controls = row(refresh_btn, clear_btn, export_btn)
        
        # Create panel layout
        panel_content = column(
            title,
            controls,
            workspace_tabs,
            min_width=600,
            sizing_mode="stretch_both"
        )
        
        return panel_content
    
    def _create_variables_tab(self):
        """Create the variables overview tab."""
        # Variables table
        variables_data = {
            "name": [],
            "type": [], 
            "shape": [],
            "source": [],
            "tags": [],
            "created": []
        }
        
        self.variables_source = ColumnDataSource(variables_data)
        
        columns = [
            TableColumn(field="name", title="Variable Name", width=150),
            TableColumn(field="type", title="Type", width=100),
            TableColumn(field="shape", title="Shape/Size", width=120),
            TableColumn(field="source", title="Created By", width=120),
            TableColumn(field="tags", title="Tags", width=100),
            TableColumn(field="created", title="Created", width=100)
        ]
        
        self.variables_table = DataTable(
            source=self.variables_source,
            columns=columns,
            height=400,
            sizing_mode="stretch_width",
            selectable=True
        )
        
        # Variable details display
        details_title = Div(text="<h4>Variable Details</h4>")
        self.variable_details = Div(
            text="<div style='background-color: #f9f9f9; padding: 10px; border: 1px solid #ddd;'><em>Select a variable to view details</em></div>",
            width=580,
            height=150
        )
        
        self.variables_table.source.selected.on_change("indices", self._on_variable_select)
        
        variables_content = column(
            self.variables_table,
            details_title,
            self.variable_details
        )
        
        return TabPanel(child=variables_content, title="Variables")
    
    def _create_connections_tab(self):
        """Create the data flow connections tab."""
        connections_title = Div(text="<h4>Data Flow Connections</h4>")
        
        # Connection visualization
        self.connections_display = Div(
            text="<div style='background-color: #f9f9f9; padding: 10px; border: 1px solid #ddd;'><em>Execute a workflow to see data flow connections</em></div>",
            width=580,
            height=400
        )
        
        # Connection controls
        self.connection_filter = Select(
            title="Filter by Step:",
            options=[("all", "All Steps")],
            value="all",
            width=200
        )
        
        show_arrows_btn = Button(label="Update Flow", width=100)
        compact_view_btn = Button(label="Compact View", width=100)
        expand_view_btn = Button(label="Detailed View", width=100)
        
        show_arrows_btn.on_click(self._update_connections_display)
        compact_view_btn.on_click(lambda: self._update_connections_display(compact=True))
        expand_view_btn.on_click(lambda: self._update_connections_display(compact=False))
        
        connection_controls = row(self.connection_filter, show_arrows_btn, compact_view_btn, expand_view_btn)
        
        connections_content = column(
            connections_title,
            connection_controls,
            self.connections_display
        )
        
        return TabPanel(child=connections_content, title="Data Flow")
    
    def _create_tags_tab(self):
        """Create the variable tagging tab."""
        tags_title = Div(text="<h4>Variable Tags</h4>")
        
        # Tag controls
        tag_variable = Select(
            title="Variable:",
            options=[("", "Select Variable")],
            width=200
        )
        
        tag_input = TextInput(
            title="Add Tag:",
            placeholder="Enter tag name",
            width=200
        )
        
        add_tag_btn = Button(label="Add Tag", button_type="primary", width=100)
        remove_tag_btn = Button(label="Remove Tag", button_type="warning", width=100)
        
        tag_controls = row(tag_variable, tag_input, add_tag_btn, remove_tag_btn)
        
        # Predefined tags
        predefined_title = Div(text="<b>Quick Tags:</b>")
        feature_tag_btn = Button(label="Feature Data", width=120)
        model_tag_btn = Button(label="Model", width=120) 
        scores_tag_btn = Button(label="Scores", width=120)
        intermediate_tag_btn = Button(label="Intermediate", width=120)
        
        quick_tags = row(feature_tag_btn, model_tag_btn, scores_tag_btn, intermediate_tag_btn)
        
        # Tags display
        self.tags_display = Div(
            text="<div style='background-color: #f9f9f9; padding: 10px; border: 1px solid #ddd;'><em>Variable tags will appear here</em></div>",
            width=580,
            height=250
        )
        
        # Store references
        self.tag_variable_select = tag_variable
        self.tag_input = tag_input
        
        # Set up callbacks
        add_tag_btn.on_click(self._add_variable_tag)
        remove_tag_btn.on_click(self._remove_variable_tag)
        feature_tag_btn.on_click(lambda: self._quick_tag("features"))
        model_tag_btn.on_click(lambda: self._quick_tag("model"))
        scores_tag_btn.on_click(lambda: self._quick_tag("scores"))
        intermediate_tag_btn.on_click(lambda: self._quick_tag("intermediate"))
        
        tags_content = column(
            tags_title,
            tag_controls,
            predefined_title,
            quick_tags,
            self.tags_display
        )
        
        return TabPanel(child=tags_content, title="Tags")
    
    def _refresh_variables(self):
        """Refresh the variables display from the executor."""
        if not self.executor:
            return
            
        # Get variables from executor
        variables = self.executor.list_variables()
        workspace_vars = self.executor.variables
        
        # Prepare table data
        data = {
            "name": [],
            "type": [],
            "shape": [],
            "source": [],
            "tags": [],
            "created": []
        }
        
        for var_name, var_type in variables.items():
            if var_name in workspace_vars:
                var_value = workspace_vars[var_name]
                
                # Determine shape/size
                if hasattr(var_value, 'shape'):
                    shape = str(var_value.shape)
                elif hasattr(var_value, '__len__'):
                    shape = f"length {len(var_value)}"
                else:
                    shape = "scalar"
                
                # Determine source (which step created it)
                source = self._get_variable_source(var_name)
                
                # Get tags
                tags = ", ".join(self.variable_tags.get(var_name, []))
                
                data["name"].append(var_name)
                data["type"].append(var_type)
                data["shape"].append(shape)
                data["source"].append(source)
                data["tags"].append(tags)
                data["created"].append("Recent")  # TODO: Add timestamps
        
        # Update table
        self.variables_source.data = data
        
        # Update tag selector
        var_options = [("", "Select Variable")] + [(name, name) for name in data["name"]]
        self.tag_variable_select.options = var_options
        
        # Update variables data for detailed view
        self.variables_data = workspace_vars.copy()
        
        # Update connection filter options
        step_options = [("all", "All Steps")]
        step_numbers = set()
        for var_name in data["name"]:
            source = self._get_variable_source(var_name)
            step_info = self._parse_step_info(source)
            if step_info:
                step_numbers.add(step_info["step_number"])
        
        for step_num in sorted(step_numbers):
            step_options.append((str(step_num), f"Step {step_num}"))
        
        self.connection_filter.options = step_options
        
        # Auto-update connections if we have data
        if data["name"]:
            self._update_connections_display()
        
        print(f"Refreshed {len(data['name'])} variables")
    
    def _get_variable_source(self, var_name: str) -> str:
        """Determine which step created a variable."""
        if "_" not in var_name:
            return "Manual"
        
        # Parse variable names like "ar_model_2_out" or "ar_model_2_rms_residuals_fv"
        parts = var_name.split("_")
        if len(parts) >= 3:
            try:
                step_num = int(parts[-2])  # Second to last part should be step number
                func_parts = parts[:-2]   # Everything except step number and output name
                func_name = "_".join(func_parts)
                return f"Step {step_num}: {func_name}"
            except ValueError:
                pass
        
        return "Unknown"
    
    def _clear_variables(self):
        """Clear all variables from the workspace."""
        if self.executor:
            self.executor.clear_variables()
            self.variable_tags.clear()
            self.variable_connections.clear()
            self._refresh_variables()
            print("Variables cleared")
    
    def _export_variables(self):
        """Export variable data (placeholder implementation)."""
        print("Variable export functionality - to be implemented")
        # TODO: Implement variable export (CSV, MAT, JSON formats)
    
    def _on_variable_select(self, attr, old, new):
        """Handle variable selection in the table."""
        if not new or not self.variables_data:
            self.variable_details.text = "<div style='background-color: #f9f9f9; padding: 10px; border: 1px solid #ddd;'><em>Select a variable to view details</em></div>"
            return
        
        # Get selected variable
        selected_idx = new[0]
        var_names = list(self.variables_source.data["name"])
        
        if selected_idx < len(var_names):
            var_name = var_names[selected_idx]
            
            if var_name in self.variables_data:
                var_value = self.variables_data[var_name]
                details_html = self._format_variable_details(var_name, var_value)
                self.variable_details.text = f"<div style='background-color: #f9f9f9; padding: 10px; border: 1px solid #ddd;'>{details_html}</div>"
    
    def _format_variable_details(self, var_name: str, var_value: Any) -> str:
        """Format variable details for display."""
        details = [f"<b>Variable:</b> {var_name}"]
        details.append(f"<b>Type:</b> {type(var_value).__name__}")
        
        # Shape/size information
        if hasattr(var_value, 'shape'):
            details.append(f"<b>Shape:</b> {var_value.shape}")
            if hasattr(var_value, 'dtype'):
                details.append(f"<b>Data Type:</b> {var_value.dtype}")
        elif hasattr(var_value, '__len__'):
            details.append(f"<b>Length:</b> {len(var_value)}")
        
        # Statistical information for numeric data
        if isinstance(var_value, np.ndarray) and var_value.size > 0:
            try:
                details.append(f"<b>Min:</b> {np.min(var_value):.4f}")
                details.append(f"<b>Max:</b> {np.max(var_value):.4f}")
                details.append(f"<b>Mean:</b> {np.mean(var_value):.4f}")
                details.append(f"<b>Std:</b> {np.std(var_value):.4f}")
            except:
                pass
        
        # Memory usage
        if hasattr(var_value, 'nbytes'):
            memory_mb = var_value.nbytes / (1024 * 1024)
            details.append(f"<b>Memory:</b> {memory_mb:.2f} MB")
        
        # Tags
        tags = self.variable_tags.get(var_name, [])
        if tags:
            details.append(f"<b>Tags:</b> {', '.join(tags)}")
        
        return "<br>".join(details)
    
    def _add_variable_tag(self):
        """Add a tag to the selected variable."""
        var_name = self.tag_variable_select.value
        tag_name = self.tag_input.value.strip()
        
        if not var_name or not tag_name:
            print("Please select a variable and enter a tag name")
            return
        
        if var_name not in self.variable_tags:
            self.variable_tags[var_name] = []
        
        if tag_name not in self.variable_tags[var_name]:
            self.variable_tags[var_name].append(tag_name)
            self.tag_input.value = ""
            self._refresh_variables()
            self._update_tags_display()
            print(f"Added tag '{tag_name}' to variable '{var_name}'")
        else:
            print(f"Tag '{tag_name}' already exists for variable '{var_name}'")
    
    def _remove_variable_tag(self):
        """Remove a tag from the selected variable."""
        var_name = self.tag_variable_select.value
        tag_name = self.tag_input.value.strip()
        
        if not var_name or not tag_name:
            print("Please select a variable and enter a tag name to remove")
            return
        
        if var_name in self.variable_tags and tag_name in self.variable_tags[var_name]:
            self.variable_tags[var_name].remove(tag_name)
            if not self.variable_tags[var_name]:  # Remove empty list
                del self.variable_tags[var_name]
            self.tag_input.value = ""
            self._refresh_variables()
            self._update_tags_display()
            print(f"Removed tag '{tag_name}' from variable '{var_name}'")
        else:
            print(f"Tag '{tag_name}' not found for variable '{var_name}'")
    
    def _quick_tag(self, tag_name: str):
        """Apply a quick tag to the selected variable."""
        var_name = self.tag_variable_select.value
        
        if not var_name:
            print("Please select a variable first")
            return
        
        if var_name not in self.variable_tags:
            self.variable_tags[var_name] = []
        
        if tag_name not in self.variable_tags[var_name]:
            self.variable_tags[var_name].append(tag_name)
            self._refresh_variables()
            self._update_tags_display()
            print(f"Added quick tag '{tag_name}' to variable '{var_name}'")
    
    def _update_tags_display(self):
        """Update the tags overview display."""
        if not self.variable_tags:
            self.tags_display.text = "<div style='background-color: #f9f9f9; padding: 10px; border: 1px solid #ddd;'><em>No tagged variables</em></div>"
            return
        
        tags_html = ["<b>Tagged Variables:</b><br>"]
        
        for var_name, tags in self.variable_tags.items():
            tags_str = ", ".join(f"<span style='background-color: #e6f3ff; padding: 2px 6px; border-radius: 3px;'>{tag}</span>" for tag in tags)
            tags_html.append(f"• <b>{var_name}:</b> {tags_str}")
        
        content = "<br>".join(tags_html)
        self.tags_display.text = f"<div style='background-color: #f9f9f9; padding: 10px; border: 1px solid #ddd;'>{content}</div>"
    
    def update_from_executor(self, executor):
        """Update the workspace from a new executor instance."""
        self.executor = executor
        self._refresh_variables()
    
    def get_tagged_variables(self, tag: str) -> List[str]:
        """Get all variables with a specific tag."""
        tagged_vars = []
        for var_name, tags in self.variable_tags.items():
            if tag in tags:
                tagged_vars.append(var_name)
        return tagged_vars
    
    def get_variable_tags(self, var_name: str) -> List[str]:
        """Get all tags for a specific variable."""
        return self.variable_tags.get(var_name, []).copy()
    
    def _update_connections_display(self, compact: bool = False):
        """Update the data flow connections visualization."""
        if not self.executor or not self.executor.variables:
            self.connections_display.text = "<div style='background-color: #f9f9f9; padding: 10px; border: 1px solid #ddd;'><em>No variables available. Execute a workflow first.</em></div>"
            return
        
        # Analyze variable connections
        connections = self._analyze_variable_connections()
        
        if not connections:
            self.connections_display.text = "<div style='background-color: #f9f9f9; padding: 10px; border: 1px solid #ddd;'><em>No data flow connections detected.</em></div>"
            return
        
        # Generate HTML visualization
        if compact:
            html_content = self._generate_compact_flow_html(connections)
        else:
            html_content = self._generate_detailed_flow_html(connections)
        
        self.connections_display.text = f"<div style='background-color: #f9f9f9; padding: 10px; border: 1px solid #ddd;'>{html_content}</div>"
    
    def _analyze_variable_connections(self) -> Dict[str, Any]:
        """Analyze connections between variables in the workflow."""
        connections = {
            "steps": {},
            "flows": [],
            "variables": {}
        }
        
        # Group variables by their creating step
        for var_name in self.executor.variables:
            source = self._get_variable_source(var_name)
            step_info = self._parse_step_info(source)
            
            if step_info:
                step_num = step_info["step_number"]
                func_name = step_info["function_name"]
                
                if step_num not in connections["steps"]:
                    connections["steps"][step_num] = {
                        "function": func_name,
                        "inputs": [],
                        "outputs": []
                    }
                
                connections["steps"][step_num]["outputs"].append(var_name)
                connections["variables"][var_name] = {
                    "source_step": step_num,
                    "source_function": func_name,
                    "used_by": []
                }
        
        # Detect data flow connections (this is simplified - could be enhanced)
        # Look for variables that might be inputs to later steps
        step_numbers = sorted(connections["steps"].keys())
        for i, step_num in enumerate(step_numbers):
            if i < len(step_numbers) - 1:  # Not the last step
                next_steps = step_numbers[i+1:]
                step_outputs = connections["steps"][step_num]["outputs"]
                
                # Create flows from this step to later steps (simplified assumption)
                for next_step in next_steps:
                    for output_var in step_outputs:
                        # Check if this variable type might be used by next step
                        if self._could_be_input_to_step(output_var, next_step, connections):
                            connections["flows"].append({
                                "from_step": step_num,
                                "to_step": next_step,
                                "variable": output_var,
                                "confidence": "inferred"  # Could be "confirmed" if we tracked actual usage
                            })
                            
                            connections["variables"][output_var]["used_by"].append(next_step)
        
        return connections
    
    def _parse_step_info(self, source: str) -> Optional[Dict[str, Any]]:
        """Parse step information from source string."""
        if source.startswith("Step "):
            try:
                # Parse "Step 2: ar_model" format
                parts = source.split(": ", 1)
                step_part = parts[0]  # "Step 2"
                func_part = parts[1] if len(parts) > 1 else "Unknown"
                
                step_number = int(step_part.split()[1])
                return {
                    "step_number": step_number,
                    "function_name": func_part
                }
            except (ValueError, IndexError):
                pass
        return None
    
    def _could_be_input_to_step(self, var_name: str, step_num: int, connections: Dict) -> bool:
        """Determine if a variable could be an input to a step using generic heuristics."""
        # Generic heuristics based on naming patterns - no function-specific logic
        
        step_info = connections["steps"].get(step_num, {})
        func_name = step_info.get("function", "").lower()
        var_name_lower = var_name.lower()
        
        # Generic patterns for data flow
        # Data/feature patterns
        if any(pattern in var_name_lower for pattern in ["data", "features", "signal", "time_series"]):
            if any(pattern in func_name for pattern in ["process", "analyze", "transform", "learn", "score"]):
                return True
        
        # Model patterns
        if "model" in var_name_lower and any(pattern in func_name for pattern in ["score", "predict", "test", "apply"]):
            return True
            
        # Output patterns (features, parameters, etc.)
        if any(suffix in var_name_lower for suffix in ["_fv", "_features", "_params", "_parameters"]):
            if any(pattern in func_name for pattern in ["learn", "score", "classify", "detect"]):
                return True
        
        return False
    
    def _generate_compact_flow_html(self, connections: Dict[str, Any]) -> str:
        """Generate compact HTML visualization of data flow."""
        html_parts = ["<h4>Data Flow Summary</h4>"]
        
        step_numbers = sorted(connections["steps"].keys())
        
        # Create flow diagram
        flow_parts = []
        for step_num in step_numbers:
            step_info = connections["steps"][step_num]
            outputs_count = len(step_info["outputs"])
            
            step_box = f"""
            <span style="display: inline-block; background-color: #e6f3ff; padding: 5px 10px; 
                         border-radius: 5px; margin: 2px;">
                Step {step_num}: {step_info["function"]}<br>
                <small>({outputs_count} outputs)</small>
            </span>
            """
            
            flow_parts.append(step_box)
            
            if step_num < max(step_numbers):  # Add arrow if not last
                flow_parts.append(' ➡️ ')
        
        html_parts.append("<p>" + "".join(flow_parts) + "</p>")
        
        # Summary statistics
        total_variables = len(connections["variables"])
        total_flows = len(connections["flows"])
        
        html_parts.append(f"""
        <p><b>Summary:</b><br>
        • Total variables: {total_variables}<br>
        • Data flows: {total_flows}<br>
        • Processing steps: {len(connections["steps"])}
        </p>
        """)
        
        return "".join(html_parts)
    
    def _generate_detailed_flow_html(self, connections: Dict[str, Any]) -> str:
        """Generate detailed HTML visualization of data flow."""
        html_parts = ["<h4>Detailed Data Flow</h4>"]
        
        step_numbers = sorted(connections["steps"].keys())
        
        for step_num in step_numbers:
            step_info = connections["steps"][step_num]
            
            # Step header
            html_parts.append(f"""
            <div style="border: 1px solid #ccc; margin: 10px 0; padding: 10px; background-color: #f8f9fa;">
                <h5 style="margin: 0 0 10px 0; color: #2c3e50;">
                    Step {step_num}: {step_info["function"]}
                </h5>
            """)
            
            # Outputs
            if step_info["outputs"]:
                html_parts.append("<b>Outputs:</b><ul>")
                for output_var in step_info["outputs"]:
                    var_info = connections["variables"][output_var]
                    used_by = var_info["used_by"]
                    tags = ", ".join(self.variable_tags.get(output_var, []))
                    
                    used_text = f" → Used by steps: {', '.join(map(str, used_by))}" if used_by else " (unused)"
                    tags_text = f" <small>[{tags}]</small>" if tags else ""
                    
                    html_parts.append(f"<li><code>{output_var}</code>{tags_text}{used_text}</li>")
                html_parts.append("</ul>")
            else:
                html_parts.append("<p><em>No outputs</em></p>")
            
            html_parts.append("</div>")
        
        # Flow connections summary
        if connections["flows"]:
            html_parts.append("<h5>Data Flow Connections:</h5>")
            html_parts.append("<ul>")
            for flow in connections["flows"]:
                confidence_icon = "✅" if flow["confidence"] == "confirmed" else "🔍"
                html_parts.append(f"""
                <li>{confidence_icon} <b>Step {flow["from_step"]}</b> → <b>Step {flow["to_step"]}</b>: 
                    <code>{flow["variable"]}</code></li>
                """)
            html_parts.append("</ul>")
        
        return "".join(html_parts)