"""
Session management for SHMTools workflows.

Provides functionality for creating, saving, loading, and managing workflow sessions
with full state persistence and restoration capabilities.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import tempfile

from .session_formats import SessionSchema, StepSchema, ParameterSchema


class WorkflowSession:
    """
    Represents a single workflow session with state management.

    This class wraps the SessionSchema with additional runtime state
    and provides methods for manipulating the workflow.
    """

    def __init__(self, schema: Optional[SessionSchema] = None):
        """Initialize workflow session."""
        self.schema = schema or SessionSchema()
        self.file_path: Optional[str] = None
        self.is_modified = False

    @property
    def name(self) -> str:
        """Get session name."""
        return self.schema.name

    @name.setter
    def name(self, value: str):
        """Set session name."""
        self.schema.name = value
        self.mark_modified()

    @property
    def description(self) -> str:
        """Get session description."""
        return self.schema.description

    @description.setter
    def description(self, value: str):
        """Set session description."""
        self.schema.description = value
        self.mark_modified()

    @property
    def steps(self) -> List[StepSchema]:
        """Get workflow steps."""
        return self.schema.steps

    def mark_modified(self):
        """Mark the session as modified."""
        self.is_modified = True
        self.schema.update_modified()

    def add_function_step(
        self,
        function_name: str,
        category: str = "",
        description: str = "",
        parameters: Optional[List[ParameterSchema]] = None,
        display_name: Optional[str] = None,
    ) -> StepSchema:
        """
        Add a function step to the workflow.

        Parameters
        ----------
        function_name : str
            Name of the SHMTools function to add.
        category : str, optional
            Function category.
        description : str, optional
            Step description.
        parameters : list of ParameterSchema, optional
            Function parameters.
        display_name : str, optional
            Human-readable function name.

        Returns
        -------
        step : StepSchema
            The created step.
        """
        step_number = len(self.schema.steps) + 1

        # Generate display name if not provided
        if not display_name:
            display_name = self._get_display_name_for_function(function_name)

        step = StepSchema(
            step_number=step_number,
            function_name=function_name,
            description=description or f"Execute {display_name}",
            category=category,
            parameters=parameters or [],
            outputs=[f"{function_name}_{step_number}_out"],
            status="pending",
            display_name=display_name,
        )

        self.schema.add_step(step)
        self.mark_modified()
        return step

    def remove_step(self, step_number: int):
        """Remove a step and update dependent parameter references."""
        # Find parameters that reference this step
        step_to_remove = self.schema.get_step(step_number)
        if not step_to_remove:
            return

        # Update parameter references in later steps
        for step in self.schema.steps:
            if step.step_number > step_number:
                for param in step.parameters:
                    if (
                        param.parameter_type == "variable_reference"
                        and param.source_step
                        and param.source_step.startswith(f"Step {step_number}:")
                    ):
                        # This parameter references the step being removed
                        param.parameter_type = "user"
                        param.source_step = None
                        param.value = (
                            ""  # Reset to empty, user will need to reconfigure
                        )

        self.schema.remove_step(step_number)
        self.mark_modified()

    def update_step_parameter(
        self,
        step_number: int,
        param_name: str,
        value: Any,
        param_type: str = "user",
        source_step: Optional[str] = None,
    ):
        """Update a parameter value for a specific step."""
        step = self.schema.get_step(step_number)
        if not step:
            raise ValueError(f"Step {step_number} not found")

        # Find existing parameter or create new one
        param = None
        for p in step.parameters:
            if p.name == param_name:
                param = p
                break

        if param is None:
            # Create new parameter
            param = ParameterSchema(
                name=param_name,
                value=value,
                parameter_type=param_type,
                widget_type="text",  # Default, should be determined by function introspection
                source_step=source_step,
            )
            step.parameters.append(param)
        else:
            # Update existing parameter
            param.value = value
            param.parameter_type = param_type
            param.source_step = source_step

        # Update step status
        if step.status == "pending":
            step.status = "configured"
        elif step.status in ["completed", "error"]:
            step.status = "configured"  # Mark as needing re-execution

        self.mark_modified()

    def get_available_variables(self, before_step: int) -> List[str]:
        """
        Get list of variables available to a specific step.

        Parameters
        ----------
        before_step : int
            Step number - only variables from earlier steps are available.

        Returns
        -------
        variables : list of str
            List of available variable references in format "Step X: function_name".
        """
        variables = []

        for step in self.schema.steps:
            if step.step_number < before_step:
                variables.append(f"Step {step.step_number}: {step.function_name}")

        return variables

    def validate(self) -> List[str]:
        """Validate the session and return any errors."""
        return self.schema.validate()

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        data = self.schema.to_dict()
        data["_metadata"] = {
            "file_path": self.file_path,
            "is_modified": self.is_modified,
        }
        return data

    def clone(self) -> "WorkflowSession":
        """Create a copy of this session."""
        new_schema = SessionSchema.from_dict(self.schema.to_dict())
        new_schema.name = f"{self.name} (Copy)"
        new_schema.created = datetime.now().isoformat()
        new_schema.modified = new_schema.created

        return WorkflowSession(new_schema)

    def _get_display_name_for_function(self, function_name: str) -> str:
        """
        Get human-readable display name for a function.

        Parameters
        ----------
        function_name : str
            Technical function name

        Returns
        -------
        str
            Human-readable display name
        """
        try:
            # Try to get function metadata from docstrings
            # Add parent directories to path if needed
            import sys
            from pathlib import Path

            current_dir = Path(__file__).parent
            if str(current_dir.parent.parent) not in sys.path:
                sys.path.insert(0, str(current_dir.parent.parent))

            from bokeh_shmtools.utils.docstring_parser import (
                parse_shmtools_docstring,
                parse_verbose_call,
            )

            # Try to get the function object
            func_obj = None
            try:
                # Try common modules
                if function_name == "load_3story_data":
                    from shmtools.utils.data_loading import load_3story_data

                    func_obj = load_3story_data
                elif function_name == "ar_model":
                    from shmtools.features.time_series import ar_model

                    func_obj = ar_model
                elif function_name == "learn_pca":
                    from shmtools.classification.outlier_detection import learn_pca

                    func_obj = learn_pca
                elif function_name == "score_pca":
                    from shmtools.classification.outlier_detection import score_pca

                    func_obj = score_pca
                elif function_name == "learn_mahalanobis":
                    from shmtools.classification.outlier_detection import (
                        learn_mahalanobis,
                    )

                    func_obj = learn_mahalanobis
                elif function_name == "score_mahalanobis":
                    from shmtools.classification.outlier_detection import (
                        score_mahalanobis,
                    )

                    func_obj = score_mahalanobis
                elif function_name == "learn_svd":
                    from shmtools.classification.outlier_detection import learn_svd

                    func_obj = learn_svd
                elif function_name == "score_svd":
                    from shmtools.classification.outlier_detection import score_svd

                    func_obj = score_svd
            except ImportError:
                pass

            if func_obj:
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

        # Fall back to converting technical name
        return self._technical_to_readable(function_name)

    def _technical_to_readable(self, technical_name: str) -> str:
        """Convert technical function name to human-readable format."""
        # Remove _shm suffix if present
        name = technical_name.replace("_shm", "")

        # Split on underscores and capitalize each word
        words = name.split("_")

        # Handle common abbreviations
        abbrev_map = {
            "pca": "PCA",
            "svd": "SVD",
            "ar": "AR",
            "rms": "RMS",
            "fft": "FFT",
            "stft": "STFT",
            "psd": "PSD",
            "cbm": "CBM",
        }

        readable_words = []
        for word in words:
            if word.lower() in abbrev_map:
                readable_words.append(abbrev_map[word.lower()])
            else:
                readable_words.append(word.capitalize())

        return " ".join(readable_words)


class SessionManager:
    """
    Manages workflow session files and provides session operations.

    Handles saving, loading, and organizing workflow sessions with
    file system operations and session history.
    """

    def __init__(self, sessions_dir: Optional[str] = None):
        """
        Initialize session manager.

        Parameters
        ----------
        sessions_dir : str, optional
            Directory to store session files. If None, uses default.
        """
        if sessions_dir is None:
            # Default to sessions subdirectory next to this file
            sessions_dir = Path(__file__).parent / "saved_sessions"

        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        # Keep track of recent sessions
        self.recent_sessions: List[str] = []
        self._load_recent_sessions()

    def create_new_session(
        self, name: str = "New Workflow", description: str = "", author: str = ""
    ) -> WorkflowSession:
        """
        Create a new workflow session.

        Parameters
        ----------
        name : str
            Session name.
        description : str, optional
            Session description.
        author : str, optional
            Session author.

        Returns
        -------
        session : WorkflowSession
            New session instance.
        """
        schema = SessionSchema(name=name, description=description, author=author)

        return WorkflowSession(schema)

    def save_session(
        self, session: WorkflowSession, file_path: Optional[str] = None
    ) -> str:
        """
        Save a session to file.

        Parameters
        ----------
        session : WorkflowSession
            Session to save.
        file_path : str, optional
            File path to save to. If None, generates from session name.

        Returns
        -------
        file_path : str
            Path where session was saved.
        """
        if file_path is None:
            # Generate filename from session name
            safe_name = "".join(
                c for c in session.name if c.isalnum() or c in " -_"
            ).strip()
            safe_name = safe_name.replace(" ", "_")
            if not safe_name:
                safe_name = "untitled_workflow"
            file_path = self.sessions_dir / f"{safe_name}.json"

            # Handle name conflicts
            counter = 1
            original_path = file_path
            while file_path.exists():
                stem = original_path.stem
                file_path = original_path.with_name(f"{stem}_{counter}.json")
                counter += 1
        else:
            file_path = Path(file_path)

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to temporary file first, then rename for atomic operation
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=file_path.parent
        ) as tmp_file:
            tmp_file.write(session.schema.to_json())
            temp_path = tmp_file.name

        # Atomic rename
        shutil.move(temp_path, str(file_path))

        # Update session state
        session.file_path = str(file_path)
        session.is_modified = False

        # Add to recent sessions
        self._add_to_recent(str(file_path))

        return str(file_path)

    def load_session(self, file_path: str) -> WorkflowSession:
        """
        Load a session from file.

        Parameters
        ----------
        file_path : str
            Path to session file.

        Returns
        -------
        session : WorkflowSession
            Loaded session.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Session file not found: {file_path}")

        try:
            with open(file_path, "r") as f:
                json_content = f.read()

            schema = SessionSchema.from_json(json_content)
            session = WorkflowSession(schema)
            session.file_path = str(file_path)
            session.is_modified = False

            # Add to recent sessions
            self._add_to_recent(str(file_path))

            return session

        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid session file format: {e}")

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all available sessions.

        Returns
        -------
        sessions : list of dict
            List of session information dictionaries.
        """
        sessions = []

        for file_path in self.sessions_dir.glob("*.json"):
            try:
                # Quick load to get metadata
                with open(file_path, "r") as f:
                    data = json.load(f)

                sessions.append(
                    {
                        "name": data.get("name", "Untitled"),
                        "description": data.get("description", ""),
                        "author": data.get("author", ""),
                        "created": data.get("created", ""),
                        "modified": data.get("modified", ""),
                        "file_path": str(file_path),
                        "num_steps": len(data.get("steps", [])),
                    }
                )

            except Exception:
                # Skip invalid files
                continue

        # Sort by modification time, newest first
        sessions.sort(key=lambda x: x["modified"], reverse=True)

        return sessions

    def delete_session(self, file_path: str):
        """
        Delete a session file.

        Parameters
        ----------
        file_path : str
            Path to session file to delete.
        """
        file_path = Path(file_path)
        if file_path.exists():
            file_path.unlink()

            # Remove from recent sessions
            if str(file_path) in self.recent_sessions:
                self.recent_sessions.remove(str(file_path))
                self._save_recent_sessions()

    def duplicate_session(self, file_path: str) -> str:
        """
        Create a duplicate of an existing session.

        Parameters
        ----------
        file_path : str
            Path to session file to duplicate.

        Returns
        -------
        new_file_path : str
            Path to the new duplicated session.
        """
        session = self.load_session(file_path)
        duplicated = session.clone()
        return self.save_session(duplicated)

    def get_recent_sessions(self) -> List[str]:
        """Get list of recently accessed session file paths."""
        # Filter out non-existent files
        existing = [path for path in self.recent_sessions if Path(path).exists()]
        if len(existing) != len(self.recent_sessions):
            self.recent_sessions = existing
            self._save_recent_sessions()

        return existing

    def _add_to_recent(self, file_path: str):
        """Add a session to recent sessions list."""
        if file_path in self.recent_sessions:
            self.recent_sessions.remove(file_path)

        self.recent_sessions.insert(0, file_path)

        # Keep only last 10
        self.recent_sessions = self.recent_sessions[:10]

        self._save_recent_sessions()

    def _load_recent_sessions(self):
        """Load recent sessions list from file."""
        recent_file = self.sessions_dir / ".recent_sessions"
        if recent_file.exists():
            try:
                with open(recent_file, "r") as f:
                    self.recent_sessions = json.load(f)
            except Exception:
                self.recent_sessions = []

    def _save_recent_sessions(self):
        """Save recent sessions list to file."""
        recent_file = self.sessions_dir / ".recent_sessions"
        try:
            with open(recent_file, "w") as f:
                json.dump(self.recent_sessions, f)
        except Exception:
            pass  # Non-critical operation
