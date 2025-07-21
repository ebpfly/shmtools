"""
Fixed layout version without overlapping issues.
"""

from bokeh.plotting import curdoc
from bokeh.layouts import column, row
from bokeh.models import Div

from bokeh_shmtools.panels.function_library import FunctionLibraryPanel
from bokeh_shmtools.panels.workflow_builder import WorkflowBuilderPanel  
from bokeh_shmtools.panels.parameter_controls import ParameterControlsPanel
from bokeh_shmtools.panels.results_viewer import ResultsViewerPanel

def create_fixed_app():
    """Create app with fixed layout to prevent overlapping."""
    
    # Title
    title = Div(text="<h1>SHMTools Workflow Builder (Fixed Layout)</h1>", 
                width=1200, height=60)
    
    # Create panels
    function_panel = FunctionLibraryPanel()
    workflow_panel = WorkflowBuilderPanel()
    params_panel = ParameterControlsPanel()
    results_panel = ResultsViewerPanel()
    
    # Basic communication
    def simple_add_function():
        try:
            if hasattr(function_panel, 'selected_function') and function_panel.selected_function:
                workflow_panel.add_function(function_panel.selected_function)
                print(f"✅ Added {function_panel.selected_function} to workflow")
            else:
                print("❌ No function selected")
        except Exception as e:
            print(f"❌ Error adding function: {e}")
    
    function_panel.add_button.on_click(simple_add_function)
    
    # Fixed layout with explicit spacing
    top_layout = row(
        function_panel.panel,
        Div(width=10),  # Spacer
        workflow_panel.panel,
        Div(width=10),  # Spacer 
        params_panel.panel,
        sizing_mode="fixed",
        height=480  # Fixed height to prevent overlap
    )
    
    # Add spacing between top and bottom
    spacer = Div(height=20, width=1200)
    
    app_layout = column(
        title,
        Div(height=10),  # Small spacer after title
        top_layout,
        spacer,
        results_panel.panel,
        sizing_mode="fixed",
        width=1200
    )
    
    return app_layout

doc = curdoc()
doc.add_root(create_fixed_app())
doc.title = "SHMTools Fixed"