"""
Main Bokeh application for SHMTools workflow builder.

This is the entry point for the web-based SHM analysis interface,
providing a modern replacement for the Java-based mFUSE application.
"""

from bokeh.plotting import curdoc
from bokeh.layouts import column, row
from bokeh.models import Div, Spacer

from bokeh_shmtools.panels.function_library import FunctionLibraryPanel
from bokeh_shmtools.panels.workflow_builder import WorkflowBuilderPanel  
from bokeh_shmtools.panels.parameter_controls import ParameterControlsPanel
from bokeh_shmtools.panels.results_viewer import ResultsViewerPanel
from bokeh_shmtools.panels.variable_workspace import VariableWorkspacePanel


def create_app():
    """
    Create the main Bokeh application layout.
    
    Returns
    -------
    layout : bokeh.layouts.Layout
        Main application layout.
    """
    # Title
    title = Div(text="<h1>SHMTools Workflow Builder</h1>")
    
    # Create panels
    function_panel = FunctionLibraryPanel()
    workflow_panel = WorkflowBuilderPanel()
    params_panel = ParameterControlsPanel()
    results_panel = ResultsViewerPanel()
    workspace_panel = VariableWorkspacePanel(executor=workflow_panel.executor)
    
    # Communication
    def add_function_to_workflow():
        try:
            if hasattr(function_panel, 'selected_function') and function_panel.selected_function:
                # Get both technical and display names
                technical_name = function_panel.selected_function
                display_name = getattr(function_panel, 'selected_display_name', technical_name)
                
                workflow_panel.add_function(technical_name, display_name)
                print(f"✅ Added {display_name} to workflow")
            else:
                print("❌ No function selected")
        except Exception as e:
            print(f"❌ Error adding function: {e}")
    
    function_panel.add_button.on_click(add_function_to_workflow)
    
    # Workflow step selection -> Parameter controls  
    def on_step_select(attr, old, new):
        try:
            if new and len(new) > 0:
                step_idx = new[0]
                if step_idx < len(workflow_panel.workflow_steps):
                    step = workflow_panel.workflow_steps[step_idx]
                    # Pass workflow steps and step index to parameter panel
                    params_panel.set_workflow_steps(workflow_panel.workflow_steps)
                    params_panel.set_function(step["function"], step["parameters"], step_idx)
                    print(f"✅ Selected step {step_idx + 1}: {step['function']}")
            else:
                if hasattr(params_panel, 'clear_selection'):
                    params_panel.clear_selection()
                    print("✅ Cleared parameter selection")
        except Exception as e:
            print(f"❌ Error in step selection: {e}")
            
    workflow_panel.steps_table.source.selected.on_change("indices", on_step_select)
    
    # Workflow execution -> Results viewer and workspace update
    def update_results_viewer():
        # Get the latest workflow results and update results viewer
        if hasattr(workflow_panel, 'executor') and workflow_panel.executor.variables:
            results = {
                "success": True,
                "step_results": []
            }
            
            # Convert workflow outputs to results format
            for i, step in enumerate(workflow_panel.workflow_steps):
                if step.get("outputs"):
                    step_result = {
                        "success": True,
                        "outputs": step["outputs"]
                    }
                    results["step_results"].append(step_result)
            
            # Update results viewer
            results_panel.update_results(results)
            
            # Update workspace panel
            workspace_panel.update_from_executor(workflow_panel.executor)
    
    # Connect workflow execution to results viewer
    workflow_panel.update_results_callback = update_results_viewer
    
    # Parameter controls -> Workflow step update
    def on_apply_parameters():
        try:
            if (hasattr(params_panel, 'current_function') and params_panel.current_function and 
                hasattr(workflow_panel, 'selected_step_idx') and workflow_panel.selected_step_idx is not None):
                
                # Extract parameters from widgets
                parameters = params_panel.get_current_parameters()
                
                # Update workflow step
                step_idx = workflow_panel.selected_step_idx
                if step_idx < len(workflow_panel.workflow_steps):
                    workflow_panel.workflow_steps[step_idx]["parameters"] = parameters
                    workflow_panel.workflow_steps[step_idx]["status"] = "Configured"
                    workflow_panel._update_steps_table()
                    print(f"✅ Applied parameters to step {step_idx + 1}")
        except Exception as e:
            print(f"❌ Error applying parameters: {e}")
            
    params_panel.apply_btn.on_click(on_apply_parameters)
    
    # Use responsive layout with more generous spacing
    top_layout = row(
        function_panel.panel,
        Spacer(width=40),  # More space between panels
        workflow_panel.panel,
        Spacer(width=40),
        params_panel.panel,
        sizing_mode="stretch_width"
    )
    
    # Bottom layout with workspace and results
    bottom_layout = row(
        workspace_panel.panel,
        Spacer(width=40),
        results_panel.panel,
        sizing_mode="stretch_width"
    )
    
    # Add vertical spacer between sections
    app_layout = column(
        title,
        Spacer(height=20),
        top_layout,
        Spacer(height=30),  # More space before results
        bottom_layout,
        sizing_mode="stretch_both"
    )
    
    return app_layout


# Removed complex communication setup - using simple version in create_app()


# Use the same document setup as responsive_app.py
doc = curdoc()
doc.add_root(create_app())
doc.title = "SHMTools Workflow Builder"


# Removed main() function - using direct document setup like responsive_app.py


# Removed __main__ section - using direct document setup