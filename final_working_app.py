"""
Final working version with all layout issues fixed.
"""

from bokeh.plotting import curdoc
from bokeh.layouts import column, row
from bokeh.models import Div

from bokeh_shmtools.panels.function_library import FunctionLibraryPanel
from bokeh_shmtools.panels.workflow_builder import WorkflowBuilderPanel  
from bokeh_shmtools.panels.parameter_controls import ParameterControlsPanel
from bokeh_shmtools.panels.results_viewer import ResultsViewerPanel

def create_final_app():
    """Create final working app with all issues fixed."""
    
    # Title
    title = Div(text="<h1>SHMTools Workflow Builder</h1>", 
                width=1200, height=60)
    
    # Create panels
    function_panel = FunctionLibraryPanel()
    workflow_panel = WorkflowBuilderPanel()
    params_panel = ParameterControlsPanel()
    results_panel = ResultsViewerPanel()
    
    # Basic communication that works
    def add_function_to_workflow():
        try:
            if hasattr(function_panel, 'selected_function') and function_panel.selected_function:
                workflow_panel.add_function(function_panel.selected_function)
                print(f"✅ Added {function_panel.selected_function} to workflow")
                # Update a status display
                if hasattr(params_panel, 'function_info'):
                    params_panel.function_info.text = f"<p style='color: green;'>Added {function_panel.selected_function} to workflow!</p>"
            else:
                print("❌ No function selected")
        except Exception as e:
            print(f"❌ Error adding function: {e}")
    
    function_panel.add_button.on_click(add_function_to_workflow)
    
    # Create layout with proper sizing
    top_layout = row(
        function_panel.panel,  # 270px wide
        workflow_panel.panel,  # 370px wide  
        params_panel.panel,    # 320px wide
        width=980,  # Total: 270+370+320+margins
        height=440,
        sizing_mode="fixed"
    )
    
    app_layout = column(
        title,
        top_layout,
        results_panel.panel,
        width=1200,
        height=900,
        sizing_mode="fixed"
    )
    
    return app_layout

doc = curdoc()
doc.add_root(create_final_app())
doc.title = "SHMTools Final"