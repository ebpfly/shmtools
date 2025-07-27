"""
Session file format definitions for SHMTools workflows.

Defines the JSON schema structure for saving and loading Bokeh native sessions.
This provides compatibility with mFUSE concepts while using modern JSON format.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json


@dataclass
class ParameterSchema:
    """Schema for workflow step parameters."""

    name: str
    value: Any
    parameter_type: str  # 'user', 'variable_reference', 'default'
    widget_type: str  # 'numeric', 'select', 'text', 'checkbox', 'file'
    source_step: Optional[str] = (
        None  # For variable references: "Step 1: function_name"
    )
    validation: Optional[Dict[str, Any]] = None  # min, max, options, etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterSchema":
        """Create from dictionary for JSON deserialization."""
        return cls(**data)


@dataclass
class StepSchema:
    """Schema for individual workflow steps."""

    step_number: int
    function_name: str
    description: str
    category: str
    parameters: List[ParameterSchema]
    outputs: List[str]  # List of output variable names
    status: str = (
        "pending"  # 'pending', 'configured', 'ready', 'running', 'completed', 'error'
    )
    execution_time: Optional[float] = None
    error_message: Optional[str] = None
    comments: Optional[str] = None
    display_name: Optional[str] = None  # Human-readable function name

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["parameters"] = [param.to_dict() for param in self.parameters]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepSchema":
        """Create from dictionary for JSON deserialization."""
        data = data.copy()
        data["parameters"] = [ParameterSchema.from_dict(p) for p in data["parameters"]]
        return cls(**data)


@dataclass
class SessionSchema:
    """Schema for complete workflow sessions."""

    version: str = "1.0.0"
    name: str = "Untitled Workflow"
    description: str = ""
    author: str = ""
    created: str = ""
    modified: str = ""

    # Workflow definition
    steps: List[StepSchema] = None

    # Session settings
    save_variables_to_workspace: bool = True
    generate_comments: bool = True
    export_format: str = "script"  # 'script', 'notebook'

    # Variable workspace state (optional)
    variables: Optional[Dict[str, Any]] = None

    # Execution history
    execution_history: List[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.steps is None:
            self.steps = []
        if self.execution_history is None:
            self.execution_history = []
        if not self.created:
            self.created = datetime.now().isoformat()
        if not self.modified:
            self.modified = self.created

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["steps"] = [step.to_dict() for step in self.steps]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionSchema":
        """Create from dictionary for JSON deserialization."""
        data = data.copy()
        data["steps"] = [StepSchema.from_dict(s) for s in data.get("steps", [])]
        return cls(**data)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "SessionSchema":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def update_modified(self):
        """Update the modification timestamp."""
        self.modified = datetime.now().isoformat()

    def add_step(self, step: StepSchema):
        """Add a step to the workflow."""
        self.steps.append(step)
        self.update_modified()

    def remove_step(self, step_number: int):
        """Remove a step by step number."""
        self.steps = [s for s in self.steps if s.step_number != step_number]
        # Renumber remaining steps
        for i, step in enumerate(self.steps):
            step.step_number = i + 1
        self.update_modified()

    def get_step(self, step_number: int) -> Optional[StepSchema]:
        """Get a step by step number."""
        for step in self.steps:
            if step.step_number == step_number:
                return step
        return None

    def get_step_outputs(self, step_number: int) -> List[str]:
        """Get the output variables for a specific step."""
        step = self.get_step(step_number)
        return step.outputs if step else []

    def validate(self) -> List[str]:
        """
        Validate the session schema.

        Returns
        -------
        errors : list of str
            List of validation errors, empty if valid.
        """
        errors = []

        # Check required fields
        if not self.name:
            errors.append("Session name is required")

        # Check step numbering
        expected_numbers = list(range(1, len(self.steps) + 1))
        actual_numbers = [step.step_number for step in self.steps]
        if sorted(actual_numbers) != expected_numbers:
            errors.append(
                f"Invalid step numbering: expected {expected_numbers}, got {actual_numbers}"
            )

        # Check for duplicate step numbers
        if len(set(actual_numbers)) != len(actual_numbers):
            errors.append("Duplicate step numbers found")

        # Validate each step
        for step in self.steps:
            step_errors = self._validate_step(step)
            errors.extend(step_errors)

        return errors

    def _validate_step(self, step: StepSchema) -> List[str]:
        """Validate a single step."""
        errors = []

        if not step.function_name:
            errors.append(f"Step {step.step_number}: function_name is required")

        # Check parameter references
        for param in step.parameters:
            if param.parameter_type == "variable_reference" and param.source_step:
                # Parse source step reference
                try:
                    if param.source_step.startswith("Step "):
                        step_num_str = param.source_step.split(": ")[0].split()[1]
                        ref_step_num = int(step_num_str)

                        # Check if referenced step exists and comes before current step
                        if ref_step_num >= step.step_number:
                            errors.append(
                                f"Step {step.step_number}: parameter '{param.name}' "
                                f"references future step {ref_step_num}"
                            )
                        elif not any(s.step_number == ref_step_num for s in self.steps):
                            errors.append(
                                f"Step {step.step_number}: parameter '{param.name}' "
                                f"references non-existent step {ref_step_num}"
                            )
                except (ValueError, IndexError):
                    errors.append(
                        f"Step {step.step_number}: invalid step reference format "
                        f"'{param.source_step}' for parameter '{param.name}'"
                    )

        return errors
