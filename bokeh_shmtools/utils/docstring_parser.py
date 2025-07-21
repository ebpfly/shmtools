"""
Docstring parser for extracting SHMTools function metadata.

This module provides utilities to parse structured docstrings and extract
metadata for the Bokeh workflow builder interface.
"""

import inspect
import re
import ast
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class ParameterSpec:
    """Parameter specification for GUI generation."""
    name: str
    type_hint: str
    description: str
    default: Any = None
    widget: str = "text_input"
    widget_params: Dict[str, Any] = field(default_factory=dict)
    
    
@dataclass 
class ReturnSpec:
    """Return value specification."""
    name: str
    type_hint: str
    description: str
    plot_specs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionMetadata:
    """Complete function metadata for workflow builder."""
    name: str
    brief_description: str
    full_description: str
    category: str = "Uncategorized"
    matlab_equivalent: Optional[str] = None
    complexity: str = "Unknown"
    data_type: str = "Unknown"
    output_type: str = "Unknown"
    parameters: List[ParameterSpec] = field(default_factory=list)
    returns: List[ReturnSpec] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    interactive_plot: bool = False
    typical_usage: List[str] = field(default_factory=list)
    notes: str = ""
    references: List[str] = field(default_factory=list)


def parse_shmtools_docstring(func) -> Optional[FunctionMetadata]:
    """
    Parse SHMTools function docstring to extract metadata.
    
    Parameters
    ----------
    func : callable
        Function to parse.
        
    Returns
    -------
    metadata : FunctionMetadata or None
        Parsed function metadata, or None if parsing fails.
    """
    docstring = inspect.getdoc(func)
    if not docstring:
        return None
    
    try:
        # Extract basic information
        lines = docstring.split('\n')
        brief_desc = lines[0].strip()
        
        # Extract meta section
        meta_info = _extract_meta_section(docstring)
        
        # Parse parameters with GUI specifications
        parameters = _parse_parameters_with_gui(docstring)
        
        # Parse returns section
        returns = _parse_returns_section(docstring)
        
        # Extract examples
        examples = _extract_examples(docstring)
        
        # Extract full description (first paragraph)
        full_desc = _extract_full_description(docstring)
        
        # Extract notes and references
        notes = _extract_section(docstring, "Notes")
        references = _extract_references(docstring)
        
        return FunctionMetadata(
            name=func.__name__,
            brief_description=brief_desc,
            full_description=full_desc,
            category=meta_info.get('category', 'Uncategorized'),
            matlab_equivalent=meta_info.get('matlab_equivalent'),
            complexity=meta_info.get('complexity', 'Unknown'),
            data_type=meta_info.get('data_type', 'Unknown'),
            output_type=meta_info.get('output_type', 'Unknown'),
            parameters=parameters,
            returns=returns,
            examples=examples,
            interactive_plot=meta_info.get('interactive_plot', False),
            typical_usage=meta_info.get('typical_usage', []),
            notes=notes,
            references=references
        )
        
    except Exception as e:
        print(f"Warning: Failed to parse docstring for {func.__name__}: {e}")
        return None


def _extract_meta_section(docstring: str) -> Dict[str, Any]:
    """Extract metadata from .. meta:: section."""
    meta_pattern = r'\.\. meta::\s*\n((?:\s{4,}.*\n?)*)'
    meta_match = re.search(meta_pattern, docstring)
    meta_info = {}
    
    if meta_match:
        meta_text = meta_match.group(1)
        for line in meta_text.strip().split('\n'):
            line = line.strip()
            if line.startswith(':') and line.endswith(':'):
                # Handle lines like ":category: Core - Spectral Analysis"
                parts = line[1:-1].split(':', 1)
                if len(parts) == 2:
                    key, value = parts
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to evaluate lists and booleans
                    if value.startswith('[') and value.endswith(']'):
                        try:
                            meta_info[key] = ast.literal_eval(value)
                        except:
                            meta_info[key] = value
                    elif value.lower() in ('true', 'false'):
                        meta_info[key] = value.lower() == 'true'
                    else:
                        meta_info[key] = value
    
    return meta_info


def _parse_parameters_with_gui(docstring: str) -> List[ParameterSpec]:
    """Parse parameters section with GUI widget specifications."""
    parameters = []
    
    # Find Parameters section
    param_pattern = r'Parameters\s*\n\s*-+\s*\n(.*?)(?=\n\s*(?:Returns|Raises|See Also|Notes|Examples|\Z))'
    param_match = re.search(param_pattern, docstring, re.DOTALL)
    
    if not param_match:
        return parameters
    
    param_text = param_match.group(1)
    
    # Split into individual parameter blocks
    param_blocks = re.split(r'\n(\w+)\s*:', param_text)[1:]  # Skip first empty element
    
    for i in range(0, len(param_blocks), 2):
        if i + 1 >= len(param_blocks):
            break
            
        param_name = param_blocks[i].strip()
        param_content = param_blocks[i + 1]
        
        # Extract parameter info
        param_spec = _parse_single_parameter(param_name, param_content)
        if param_spec:
            parameters.append(param_spec)
    
    return parameters


def _parse_single_parameter(name: str, content: str) -> Optional[ParameterSpec]:
    """Parse a single parameter with its GUI specifications."""
    lines = content.strip().split('\n')
    
    # First line contains type and description
    if not lines:
        return None
        
    first_line = lines[0].strip()
    
    # Extract type hint and description
    type_desc_match = re.match(r'([^,\n]*?)(?:,\s*(.*))?$', first_line)
    if not type_desc_match:
        return None
        
    type_hint = type_desc_match.group(1).strip()
    description_parts = [type_desc_match.group(2) or ""]
    
    # Collect multi-line description
    gui_section_start = None
    for i, line in enumerate(lines[1:], 1):
        line = line.strip()
        if line.startswith('.. gui::'):
            gui_section_start = i
            break
        elif line:
            description_parts.append(line)
    
    description = ' '.join(filter(None, description_parts)).strip()
    
    # Extract default value from type hint
    default = None
    if 'default=' in type_hint:
        default_match = re.search(r'default=([^,\)]+)', type_hint)
        if default_match:
            default_str = default_match.group(1).strip()
            try:
                default = ast.literal_eval(default_str)
            except:
                default = default_str
    
    # Parse GUI section
    widget = "text_input"
    widget_params = {}
    
    if gui_section_start:
        gui_lines = lines[gui_section_start:]
        for line in gui_lines:
            line = line.strip()
            if line.startswith(':') and ':' in line[1:]:
                key_val = line[1:].split(':', 1)
                if len(key_val) == 2:
                    key, value = key_val
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'widget':
                        widget = value
                    else:
                        # Try to evaluate the value
                        try:
                            widget_params[key] = ast.literal_eval(value)
                        except:
                            widget_params[key] = value
    
    return ParameterSpec(
        name=name,
        type_hint=type_hint,
        description=description,
        default=default,
        widget=widget,
        widget_params=widget_params
    )


def _parse_returns_section(docstring: str) -> List[ReturnSpec]:
    """Parse Returns section with plot specifications."""
    returns = []
    
    # Find Returns section
    returns_pattern = r'Returns\s*\n\s*-+\s*\n(.*?)(?=\n\s*(?:Raises|See Also|Notes|Examples|\Z))'
    returns_match = re.search(returns_pattern, docstring, re.DOTALL)
    
    if not returns_match:
        return returns
    
    returns_text = returns_match.group(1)
    
    # Split into individual return blocks
    return_blocks = re.split(r'\n(\w+)\s*:', returns_text)[1:]
    
    for i in range(0, len(return_blocks), 2):
        if i + 1 >= len(return_blocks):
            break
            
        return_name = return_blocks[i].strip()
        return_content = return_blocks[i + 1]
        
        # Parse return specification
        return_spec = _parse_single_return(return_name, return_content)
        if return_spec:
            returns.append(return_spec)
    
    return returns


def _parse_single_return(name: str, content: str) -> Optional[ReturnSpec]:
    """Parse a single return value with plot specifications."""
    lines = content.strip().split('\n')
    
    if not lines:
        return None
        
    first_line = lines[0].strip()
    
    # Extract type hint and description
    type_desc_match = re.match(r'([^,\n]*?)(?:,\s*(.*))?$', first_line)
    if not type_desc_match:
        return None
        
    type_hint = type_desc_match.group(1).strip()
    description_parts = [type_desc_match.group(2) or ""]
    
    # Collect description and parse GUI specs
    plot_specs = {}
    gui_section_start = None
    
    for i, line in enumerate(lines[1:], 1):
        line = line.strip()
        if line.startswith('.. gui::'):
            gui_section_start = i
            break
        elif line:
            description_parts.append(line)
    
    description = ' '.join(filter(None, description_parts)).strip()
    
    # Parse plot specifications
    if gui_section_start:
        gui_lines = lines[gui_section_start:]
        for line in gui_lines:
            line = line.strip()
            if line.startswith(':') and ':' in line[1:]:
                key_val = line[1:].split(':', 1)
                if len(key_val) == 2:
                    key, value = key_val
                    key = key.strip()
                    value = value.strip().strip('"')
                    plot_specs[key] = value
    
    return ReturnSpec(
        name=name,
        type_hint=type_hint,
        description=description,
        plot_specs=plot_specs
    )


def _extract_examples(docstring: str) -> List[str]:
    """Extract examples from Examples section."""
    examples_pattern = r'Examples\s*\n\s*-+\s*\n(.*?)(?=\n\s*(?:\.\. |$))'
    examples_match = re.search(examples_pattern, docstring, re.DOTALL)
    
    if not examples_match:
        return []
    
    examples_text = examples_match.group(1)
    
    # Split by example headers (lines that don't start with spaces or >>>)
    example_blocks = []
    current_block = []
    
    for line in examples_text.split('\n'):
        if line and not line.startswith((' ', '\t', '>>>')):
            # New example block
            if current_block:
                example_blocks.append('\n'.join(current_block))
            current_block = [line]
        else:
            current_block.append(line)
    
    if current_block:
        example_blocks.append('\n'.join(current_block))
    
    return example_blocks


def _extract_full_description(docstring: str) -> str:
    """Extract the full description (first paragraph after brief)."""
    lines = docstring.split('\n')
    
    # Skip brief description and empty lines
    start_idx = 1
    while start_idx < len(lines) and not lines[start_idx].strip():
        start_idx += 1
    
    # Collect description until we hit a section or meta directive
    desc_lines = []
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if (line.startswith('..') or 
            line in ['Parameters', 'Returns', 'Raises', 'See Also', 'Notes', 'Examples', 'References'] or
            (line and all(c == '-' for c in line))):
            break
        desc_lines.append(lines[i])
    
    return '\n'.join(desc_lines).strip()


def _extract_section(docstring: str, section_name: str) -> str:
    """Extract a named section from the docstring."""
    pattern = f'{section_name}\\s*\\n\\s*-+\\s*\\n(.*?)(?=\\n\\s*(?:[A-Z][a-z]+\\s*\\n\\s*-+|\\.\\.|\Z))'
    match = re.search(pattern, docstring, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_references(docstring: str) -> List[str]:
    """Extract references from References section."""
    refs_text = _extract_section(docstring, "References")
    if not refs_text:
        return []
    
    # Split by reference markers like ".. [1]"
    ref_pattern = r'\.\.\s*\[(\d+)\]\s*(.*?)(?=\n\s*\.\.\s*\[\d+\]|\Z)'
    matches = re.findall(ref_pattern, refs_text, re.DOTALL)
    
    references = []
    for ref_num, ref_text in matches:
        cleaned_ref = re.sub(r'\s+', ' ', ref_text.strip())
        references.append(f"[{ref_num}] {cleaned_ref}")
    
    return references