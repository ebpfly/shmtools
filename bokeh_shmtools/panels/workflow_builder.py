"""
Workflow builder panel for creating analysis sequences.

This panel provides the main workflow editing interface, allowing users
to build sequences of SHM analysis steps similar to mFUSE.
"""

from bokeh.models import (
    Div,
    Button,
    Column,
    DataTable,
    TableColumn,
    ColumnDataSource,
    TextInput,
    Select,
    FileInput,
)
from bokeh.layouts import column, row
from typing import List, Dict, Any
import base64
import json
from pathlib import Path
from bokeh_shmtools.workflows.execution_engine import WorkflowExecutor
from bokeh_shmtools.sessions import SessionManager, WorkflowSession, ParameterSchema
from bokeh_shmtools.utils.docstring_parser import parse_verbose_call


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

        # Session management
        self.session_manager = SessionManager()
        self.current_session = None

        # Progress tracking
        self.execution_in_progress = False
        self.current_step = 0

        # Set up progress callback
        self.executor.set_progress_callback(self._on_progress_update)

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

        # Session controls
        session_div = Div(text="<h4>Session Management</h4>")

        # Session selection dropdown
        session_options = self._get_session_options()
        session_select = Select(
            title="Load Session:",
            options=session_options,
            value="",
            sizing_mode="stretch_width",
        )
        session_select.on_change("value", self._on_session_load)

        # Session action buttons
        new_session_btn = Button(label="New Session", button_type="success", width=80)
        new_session_btn.on_click(self._on_new_session)

        save_session_btn = Button(label="Save", button_type="primary", width=60)
        save_session_btn.on_click(self._on_save_session)

        save_as_btn = Button(label="Save As...", button_type="default", width=80)
        save_as_btn.on_click(self._on_save_as_session)

        # File upload for session import
        file_input = FileInput(accept=".json", width=120)
        file_input.on_change("value", self._on_file_upload)

        session_buttons = row(
            new_session_btn, save_session_btn, save_as_btn, file_input
        )

        # Workflow name input
        workflow_name = TextInput(
            title="Workflow Name:", value="New Workflow", sizing_mode="stretch_width"
        )

        # Workflow steps table
        steps_data = {"step": [], "function": [], "status": []}
        steps_source = ColumnDataSource(steps_data)

        columns = [
            TableColumn(field="step", title="Step", width=50),
            TableColumn(field="function", title="Function", width=150),
            TableColumn(field="status", title="Status", width=100),
        ]

        steps_table = DataTable(
            source=steps_source,
            columns=columns,
            height=300,
            selectable=True,
            sizing_mode="stretch_width",
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
        run_btn = Button(label="Run Workflow", button_type="success", width=150)
        save_btn = Button(label="Save Workflow", button_type="primary", width=150)

        run_btn.on_click(self._run_workflow)
        save_btn.on_click(self._save_workflow)

        execution_controls = row(run_btn, save_btn)

        # Progress tracking elements
        progress_div = Div(text="<h4>Execution Progress</h4>")

        # Progress bar (custom HTML-based)
        progress_bar = Div(
            text=self._generate_progress_html(0), width=350, height=30, visible=False
        )

        # Status display
        status_display = Div(
            text="<div style='background-color: #f5f5f5; padding: 5px; border: 1px solid #ddd;'><em>Ready to execute workflow</em></div>",
            width=350,
            height=40,
        )

        # Execution metrics display
        metrics_display = Div(text="", width=350, height=60, visible=False)

        # Store references
        self.workflow_name = workflow_name
        self.steps_table = steps_table
        self.session_select = session_select
        self.selected_step_idx = None
        self.run_btn = run_btn
        self.progress_bar = progress_bar
        self.status_display = status_display
        self.metrics_display = metrics_display

        # Create panel layout
        panel_content = column(
            title,
            session_div,
            session_select,
            session_buttons,
            workflow_name,
            steps_table,
            step_controls,
            execution_controls,
            progress_div,
            progress_bar,
            status_display,
            metrics_display,
            min_width=400,
            sizing_mode="stretch_both",
        )

        return panel_content  # Return the layout directly instead of wrapping in Panel

    def add_function(self, function_name: str, display_name: str = None):
        """
        Add a function to the workflow.

        Parameters
        ----------
        function_name : str
            Technical name of the function to add (for execution).
        display_name : str, optional
            Human-readable display name. If None, will use technical name.
        """
        step_num = len(self.workflow_steps) + 1

        step = {
            "step": step_num,
            "function": function_name,  # Technical name for execution
            "display_name": display_name
            or function_name,  # Human-readable name for display
            "status": "Ready",
            "parameters": {},
            "outputs": None,
        }

        self.workflow_steps.append(step)
        self._update_steps_table()

    def _update_steps_table(self):
        """Update the workflow steps table display."""
        data = {
            "step": [step["step"] for step in self.workflow_steps],
            "function": [
                step.get("display_name", step["function"])
                for step in self.workflow_steps
            ],  # Use display name
            "status": [step["status"] for step in self.workflow_steps],
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
        if self.selected_step_idx is not None and self.selected_step_idx > 0:

            idx = self.selected_step_idx
            # Swap steps
            self.workflow_steps[idx], self.workflow_steps[idx - 1] = (
                self.workflow_steps[idx - 1],
                self.workflow_steps[idx],
            )

            # Update step numbers
            self._renumber_steps()
            self._update_steps_table()

    def _move_step_down(self):
        """Move selected step down in the workflow."""
        if (
            self.selected_step_idx is not None
            and self.selected_step_idx < len(self.workflow_steps) - 1
        ):

            idx = self.selected_step_idx
            # Swap steps
            self.workflow_steps[idx], self.workflow_steps[idx + 1] = (
                self.workflow_steps[idx + 1],
                self.workflow_steps[idx],
            )

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
        print(
            f"Steps: {[step.get('display_name', step['function']) for step in self.workflow_steps]}"
        )

        if not self.workflow_steps:
            self.status_display.text = "❌ No steps in workflow to execute"
            return

        if self.execution_in_progress:
            print("Workflow execution already in progress")
            return

        # Start execution
        self.execution_in_progress = True
        self.run_btn.disabled = True
        self.run_btn.label = "Running..."
        self.progress_bar.visible = True
        self.progress_bar.text = self._generate_progress_html(0)
        self.metrics_display.visible = False

        # Reset step statuses
        for step in self.workflow_steps:
            step["status"] = "Pending"
        self._update_steps_table()

        # Execute workflow using the execution engine
        try:
            results = self.executor.execute_workflow(self.workflow_steps)

            # Update step statuses based on results
            for i, step_result in enumerate(results["step_results"]):
                if i < len(self.workflow_steps):
                    if step_result["success"]:
                        self.workflow_steps[i]["status"] = "Complete"
                        self.workflow_steps[i]["outputs"] = step_result.get(
                            "outputs", {}
                        )
                    else:
                        self.workflow_steps[i]["status"] = "Error"
                        self.workflow_steps[i]["error"] = step_result.get(
                            "error", "Unknown error"
                        )

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

        finally:
            # Reset execution state
            self.execution_in_progress = False
            self.run_btn.disabled = False
            self.run_btn.label = "Run Workflow"

    def _save_workflow(self):
        """Save the current workflow."""
        print(f"Saving workflow: {self.workflow_name.value}")
        # TODO: Implement workflow saving (session file format)

    def _on_progress_update(self, progress_info: Dict[str, Any]):
        """
        Handle progress updates from workflow execution.

        Parameters
        ----------
        progress_info : dict
            Progress information from the execution engine.
        """
        progress_type = progress_info.get("type", "")

        if progress_type == "workflow_start":
            self.status_display.text = f"🚀 Starting workflow execution ({progress_info['total_steps']} steps)..."
            self.progress_bar.text = self._generate_progress_html(0)

        elif progress_type == "step_start":
            step_num = progress_info["step_number"]
            function_name = progress_info["function_name"]
            progress = progress_info["progress_percent"]

            # Get display name for this step
            display_name = function_name
            if step_num <= len(self.workflow_steps):
                display_name = self.workflow_steps[step_num - 1].get(
                    "display_name", function_name
                )

            self.status_display.text = f"⚙️ Step {step_num}: Executing {display_name}..."
            self.progress_bar.text = self._generate_progress_html(progress)

            # Update step status in the table
            if step_num <= len(self.workflow_steps):
                self.workflow_steps[step_num - 1]["status"] = "Running"
                self._update_steps_table()

        elif progress_type == "step_complete":
            step_num = progress_info["step_number"]
            outputs = progress_info["outputs"]
            duration = progress_info["duration"]
            progress = progress_info["progress_percent"]

            self.status_display.text = f"✅ Step {step_num}: Completed in {duration:.2f}s (Created {len(outputs)} variables)"
            self.progress_bar.text = self._generate_progress_html(progress)

            # Update step status in the table
            if step_num <= len(self.workflow_steps):
                self.workflow_steps[step_num - 1]["status"] = "Complete"
                self._update_steps_table()

        elif progress_type == "step_error":
            step_num = progress_info["step_number"]
            error = progress_info["error"]
            duration = progress_info["duration"]

            self.status_display.text = (
                f"❌ Step {step_num}: Failed after {duration:.2f}s - {error}"
            )

            # Update step status in the table
            if step_num <= len(self.workflow_steps):
                self.workflow_steps[step_num - 1]["status"] = "Error"
                self._update_steps_table()

        elif progress_type == "workflow_complete":
            total_duration = progress_info["total_duration"]
            completed_steps = progress_info["completed_steps"]
            total_steps = progress_info["total_steps"]
            variables_created = progress_info["variables_created"]

            self.status_display.text = (
                f"🎉 Workflow completed successfully in {total_duration:.2f}s"
            )
            self.progress_bar.text = self._generate_progress_html(100)

            # Show execution metrics
            metrics_html = f"""
            <b>Execution Summary:</b><br>
            • Steps completed: {completed_steps}/{total_steps}<br>
            • Total duration: {total_duration:.2f} seconds<br>
            • Variables created: {variables_created}
            """
            self.metrics_display.text = metrics_html
            self.metrics_display.visible = True

        elif progress_type == "workflow_failed":
            total_duration = progress_info["total_duration"]
            completed_steps = progress_info["completed_steps"]
            total_steps = progress_info["total_steps"]

            self.status_display.text = f"❌ Workflow failed after {total_duration:.2f}s"
            progress = (completed_steps / total_steps) * 100 if total_steps > 0 else 0
            self.progress_bar.text = self._generate_progress_html(progress)

            # Show failure metrics
            metrics_html = f"""
            <b style="color: red;">Execution Failed:</b><br>
            • Steps completed: {completed_steps}/{total_steps}<br>
            • Duration before failure: {total_duration:.2f} seconds
            """
            self.metrics_display.text = metrics_html
            self.metrics_display.visible = True

    def _generate_progress_html(self, progress: float) -> str:
        """Generate HTML for a custom progress bar."""
        progress = max(0, min(100, progress))  # Clamp between 0-100

        # Create a visual progress bar using HTML/CSS
        html = f"""
        <div style="width: 100%; background-color: #f0f0f0; border-radius: 10px; height: 20px; margin: 5px 0;">
            <div style="width: {progress}%; background-color: #4CAF50; height: 20px; border-radius: 10px; 
                        text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">
                {progress:.1f}%
            </div>
        </div>
        """
        return html

    # === Session Management Methods ===

    def _get_session_options(self):
        """Get list of available sessions for dropdown."""
        try:
            sessions = self.session_manager.list_sessions()
            options = [("", "-- Select Session --")]

            for session_info in sessions:
                label = f"{session_info['name']} ({session_info['num_steps']} steps)"
                options.append((session_info["file_path"], label))

            return options
        except Exception as e:
            print(f"Error loading session list: {e}")
            return [("", "-- No Sessions Available --")]

    def _on_new_session(self):
        """Create a new workflow session."""
        try:
            # Create new session
            self.current_session = self.session_manager.create_new_session(
                name=self.workflow_name.value or "New Workflow",
                description="",
                author="",
            )

            # Clear current workflow
            self.workflow_steps = []
            self._update_steps_table()

            # Reset UI state
            self.session_select.value = ""

            print("✅ New session created")

        except Exception as e:
            print(f"❌ Error creating new session: {e}")

    def _on_session_load(self, attr, old, new):
        """Load a selected session."""
        if not new:  # Empty selection
            return

        try:
            # Load session from file
            self.current_session = self.session_manager.load_session(new)

            # Update workflow name
            self.workflow_name.value = self.current_session.name

            # Convert session steps to workflow format
            self.workflow_steps = []
            for step in self.current_session.steps:
                workflow_step = {
                    "step": step.step_number,
                    "function": step.function_name,
                    "display_name": getattr(
                        step, "display_name", step.function_name
                    ),  # Use display name from session
                    "status": step.status.title(),
                    "parameters": {},
                    "category": step.category,
                    "description": step.description,
                }

                # Convert parameters
                for param in step.parameters:
                    if param.parameter_type == "variable_reference":
                        workflow_step["parameters"][param.name] = param.source_step
                    else:
                        workflow_step["parameters"][param.name] = param.value

                self.workflow_steps.append(workflow_step)

            # Update UI
            self._update_steps_table()
            self._refresh_session_dropdown()

            print(f"✅ Loaded session: {self.current_session.name}")
            print(f"   Steps: {len(self.workflow_steps)}")

        except Exception as e:
            print(f"❌ Error loading session: {e}")
            # Reset selection on error
            self.session_select.value = ""

    def _on_save_session(self):
        """Save the current session."""
        if not self.current_session:
            # Create new session if none exists
            self._on_new_session()

        try:
            # Update session from current workflow
            self._update_session_from_workflow()

            # Save session
            file_path = self.session_manager.save_session(self.current_session)

            # Refresh dropdown to show updated session
            self._refresh_session_dropdown()

            print(f"✅ Session saved: {Path(file_path).name}")

        except Exception as e:
            print(f"❌ Error saving session: {e}")

    def _on_save_as_session(self):
        """Save session with new name (clone current session)."""
        if not self.current_session:
            self._on_new_session()

        try:
            # Update session from current workflow
            self._update_session_from_workflow()

            # Clone session with new name
            cloned = self.current_session.clone()
            cloned.name = f"{self.workflow_name.value} (Copy)"

            # Save cloned session
            file_path = self.session_manager.save_session(cloned)

            # Switch to the cloned session
            self.current_session = cloned

            # Refresh dropdown and select the new session
            self._refresh_session_dropdown()
            self.session_select.value = file_path

            print(f"✅ Session saved as: {Path(file_path).name}")

        except Exception as e:
            print(f"❌ Error saving session as copy: {e}")

    def _on_file_upload(self, attr, old, new):
        """Handle session file upload."""
        if not new:
            return

        try:
            # Decode uploaded file
            file_content = base64.b64decode(new).decode("utf-8")

            # Parse JSON session
            session_data = json.loads(file_content)

            # Import as WorkflowSession
            from bokeh_shmtools.sessions.session_formats import SessionSchema

            schema = SessionSchema.from_dict(session_data)
            imported_session = WorkflowSession(schema)

            # Save imported session
            imported_session.name = f"Imported_{imported_session.name}"
            file_path = self.session_manager.save_session(imported_session)

            # Load the imported session
            self.current_session = imported_session
            self.session_select.value = file_path
            self._on_session_load("value", "", file_path)

            print(f"✅ Session imported: {imported_session.name}")

        except Exception as e:
            print(f"❌ Error importing session: {e}")

    def _update_session_from_workflow(self):
        """Update current session from workflow state."""
        if not self.current_session:
            return

        # Update basic session info
        self.current_session.name = self.workflow_name.value

        # Clear existing steps
        self.current_session.schema.steps = []

        # Convert workflow steps to session format
        for i, step in enumerate(self.workflow_steps):
            parameters = []

            # Convert parameters to ParameterSchema format
            for param_name, param_value in step.get("parameters", {}).items():
                if isinstance(param_value, str) and param_value.startswith("Step "):
                    # Variable reference
                    param_schema = ParameterSchema(
                        name=param_name,
                        value="",
                        parameter_type="variable_reference",
                        widget_type="text",
                        source_step=param_value,
                    )
                else:
                    # User value
                    param_schema = ParameterSchema(
                        name=param_name,
                        value=param_value,
                        parameter_type="user",
                        widget_type="text",
                    )

                parameters.append(param_schema)

            # Create step schema
            from bokeh_shmtools.sessions.session_formats import StepSchema

            step_schema = StepSchema(
                step_number=i + 1,
                function_name=step["function"],
                description=step.get("description", f"Execute {step['function']}"),
                category=step.get("category", ""),
                parameters=parameters,
                outputs=[f"{step['function']}_{i+1}_out"],
                status=step.get("status", "pending").lower(),
            )

            self.current_session.schema.steps.append(step_schema)

        # Mark as modified
        self.current_session.mark_modified()

    def _refresh_session_dropdown(self):
        """Refresh the session dropdown options."""
        try:
            new_options = self._get_session_options()
            self.session_select.options = new_options
        except Exception as e:
            print(f"Error refreshing session dropdown: {e}")
