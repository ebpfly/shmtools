"""
Working version of the main app built step by step.
"""

from bokeh.plotting import curdoc
from bokeh.layouts import column, row
from bokeh.models import Div

from bokeh_shmtools.panels.function_library import FunctionLibraryPanel
from bokeh_shmtools.panels.workflow_builder import WorkflowBuilderPanel  
from bokeh_shmtools.panels.parameter_controls import ParameterControlsPanel
from bokeh_shmtools.panels.results_viewer import ResultsViewerPanel

def create_working_app():
    """Create a working version of the main app."""
    
    # Title
    title = Div(text="<h1>SHMTools Workflow Builder (Working)</h1>", 
                width=800, height=50)
    
    # Create panels
    print("Creating function panel...")
    function_panel = FunctionLibraryPanel()
    
    print("Creating workflow panel...")
    workflow_panel = WorkflowBuilderPanel()
    
    print("Creating params panel...")
    params_panel = ParameterControlsPanel()
    
    print("Creating results panel...")
    results_panel = ResultsViewerPanel()
    
    print("Setting up basic communication...")
    
    # Only the most basic communication - just add function
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
    
    print("Creating layout...")
    
    # Layout
    top_layout = row(
        function_panel.panel,
        workflow_panel.panel, 
        params_panel.panel,
        height=500
    )
    
    app_layout = column(
        title,
        top_layout,
        results_panel.panel,
        width=1200,
        height=800
    )
    
    print("✅ App created successfully")
    return app_layout

print("Starting app creation...")
doc = curdoc()
doc.add_root(create_working_app())
doc.title = "SHMTools Working"
print("✅ Document setup complete")