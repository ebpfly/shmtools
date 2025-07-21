"""
Workflow builder panel for creating analysis sequences.

This panel provides the main workflow editing interface, allowing users
to build sequences of SHM analysis steps similar to mFUSE.
"""

from bokeh.models import (
    Div, Button, Column, DataTable, TableColumn,
    ColumnDataSource, TextInput
)
from bokeh.layouts import column, row
from typing import List, Dict, Any
from bokeh_shmtools.workflows.execution_engine import WorkflowExecutor


class WorkflowBuilderPanel:
    """
    Panel for building and managing analysis workflows.
    
    This panel replicates the central workflow builder from mFUSE,
    allowing users to create sequences of analysis steps.
    """
    
    def __init__(self):
        """Initialize the workflow builder panel."""
        self.workflow_steps = []
        self.executor = WorkflowExecutor()
        self.update_results_callback = None  # Callback to update results viewer
        self.panel = self._create_panel()
        
    def _create_panel(self):
        """
        Create the workflow builder panel components.
        
        Returns
        -------
        panel : bokeh.models.Panel
            Workflow builder panel.
        """
        # Panel title
        title = Div(text="<h3>Workflow Builder</h3>", width=350)
        
        # Workflow name input
        workflow_name = TextInput(
            title="Workflow Name:",
            value="New Workflow",
            sizing_mode="stretch_width"
        )
        
        # Workflow steps table
        steps_data = {"step": [], "function": [], "status": []}
        steps_source = ColumnDataSource(steps_data)
        
        columns = [
            TableColumn(field="step", title="Step", width=50),
            TableColumn(field="function", title="Function", width=150),
            TableColumn(field="status", title="Status", width=100)
        ]
        
        steps_table = DataTable(
            source=steps_source,
            columns=columns,
            height=300,
            selectable=True,
            sizing_mode="stretch_width"
        )
        steps_table.source.selected.on_change("indices", self._on_step_select)
        
        # Control buttons
        move_up_btn = Button(label="Move Up", width=75)
        move_down_btn = Button(label="Move Down", width=75) 
        remove_btn = Button(label="Remove", width=75, button_type="danger")
        clear_btn = Button(label="Clear All", width=75, button_type="warning")
        
        move_up_btn.on_click(self._move_step_up)
        move_down_btn.on_click(self._move_step_down)
        remove_btn.on_click(self._remove_step)
        clear_btn.on_click(self._clear_workflow)
        
        step_controls = row(move_up_btn, move_down_btn, remove_btn, clear_btn)
        
        # Execution buttons
        run_btn = Button(
            label="Run Workflow",
            button_type="success", 
            width=150
        )
        save_btn = Button(
            label="Save Workflow",
            button_type="primary",
            width=150
        )
        
        run_btn.on_click(self._run_workflow)
        save_btn.on_click(self._save_workflow)
        
        execution_controls = row(run_btn, save_btn)
        
        # Store references
        self.workflow_name = workflow_name
        self.steps_table = steps_table
        self.selected_step_idx = None
        
        # Create panel layout
        panel_content = column(
            title,
            workflow_name,
            steps_table,
            step_controls,
            execution_controls,
            min_width=400,
            sizing_mode="stretch_both"
        )
        
        return panel_content  # Return the layout directly instead of wrapping in Panel
    
    def add_function(self, function_name: str):
        """
        Add a function to the workflow.
        
        Parameters
        ----------
        function_name : str
            Name of the function to add.
        """
        step_num = len(self.workflow_steps) + 1
        
        step = {
            "step": step_num,
            "function": function_name,
            "status": "Ready",
            "parameters": {},
            "outputs": None
        }
        
        self.workflow_steps.append(step)
        self._update_steps_table()
    
    def _update_steps_table(self):
        """Update the workflow steps table display."""
        data = {
            "step": [step["step"] for step in self.workflow_steps],
            "function": [step["function"] for step in self.workflow_steps],
            "status": [step["status"] for step in self.workflow_steps]
        }
        self.steps_table.source.data = data
    
    def _on_step_select(self, attr, old, new):
        """Handle workflow step selection."""
        if new:
            self.selected_step_idx = new[0]
        else:
            self.selected_step_idx = None
    
    def _move_step_up(self):
        """Move selected step up in the workflow."""
        if (self.selected_step_idx is not None and 
            self.selected_step_idx > 0):
            
            idx = self.selected_step_idx
            # Swap steps
            self.workflow_steps[idx], self.workflow_steps[idx-1] = \
                self.workflow_steps[idx-1], self.workflow_steps[idx]
            
            # Update step numbers
            self._renumber_steps()
            self._update_steps_table()
    
    def _move_step_down(self):
        """Move selected step down in the workflow."""
        if (self.selected_step_idx is not None and 
            self.selected_step_idx < len(self.workflow_steps) - 1):
            
            idx = self.selected_step_idx
            # Swap steps
            self.workflow_steps[idx], self.workflow_steps[idx+1] = \
                self.workflow_steps[idx+1], self.workflow_steps[idx]
            
            # Update step numbers
            self._renumber_steps()
            self._update_steps_table()
    
    def _remove_step(self):
        """Remove selected step from workflow."""
        if self.selected_step_idx is not None:
            del self.workflow_steps[self.selected_step_idx]
            self._renumber_steps()
            self._update_steps_table()
            self.selected_step_idx = None
    
    def _clear_workflow(self):
        """Clear all steps from workflow."""
        self.workflow_steps = []
        self._update_steps_table()
        self.selected_step_idx = None
    
    def _renumber_steps(self):
        """Renumber workflow steps after reordering."""
        for i, step in enumerate(self.workflow_steps):
            step["step"] = i + 1
    
    def _run_workflow(self):
        """Execute the current workflow."""
        print(f"Running workflow: {self.workflow_name.value}")
        print(f"Steps: {[step['function'] for step in self.workflow_steps]}")
        
        if not self.workflow_steps:
            print("No steps in workflow to execute")
            return
        
        # Execute workflow using the execution engine
        try:
            results = self.executor.execute_workflow(self.workflow_steps)
            
            # Update step statuses based on results
            for i, step_result in enumerate(results["step_results"]):
                if i < len(self.workflow_steps):
                    if step_result["success"]:
                        self.workflow_steps[i]["status"] = "Complete"
                        self.workflow_steps[i]["outputs"] = step_result.get("outputs", {})
                    else:
                        self.workflow_steps[i]["status"] = "Error"
                        self.workflow_steps[i]["error"] = step_result.get("error", "Unknown error")
            
            self._update_steps_table()
            
            if results["success"]:
                print("✅ Workflow completed successfully")
                print(f"Generated variables: {list(results['outputs'].keys())}")
                
                # Update results viewer if callback is set
                if self.update_results_callback:
                    self.update_results_callback()
            else:
                print("❌ Workflow failed:")
                for error in results["errors"]:
                    print(f"  {error}")
                    
        except Exception as e:
            print(f"❌ Workflow execution error: {e}")
            # Mark all steps as error
            for step in self.workflow_steps:
                step["status"] = "Error"
            self._update_steps_table()
    
    def _save_workflow(self):
        """Save the current workflow."""
        print(f"Saving workflow: {self.workflow_name.value}")
        # TODO: Implement workflow saving (session file format)