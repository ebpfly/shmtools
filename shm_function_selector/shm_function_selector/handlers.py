"""
Server-side handlers for SHM Function Selector JupyterLab extension.
Provides API endpoints for function discovery and variable parsing.
"""

import json
import inspect
import importlib
import math
import os
import fnmatch
from pathlib import Path
from typing import List, Dict, Any

from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join
import tornado
from tornado.web import authenticated


class InfinityJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles infinity values."""
    
    def encode(self, obj):
        # Replace infinity values in the object before encoding
        cleaned_obj = self._clean_infinity(obj)
        return super().encode(cleaned_obj)
    
    def _clean_infinity(self, obj):
        """Recursively replace infinity values with a large number."""
        if isinstance(obj, dict):
            return {k: self._clean_infinity(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_infinity(item) for item in obj]
        elif isinstance(obj, float):
            if math.isinf(obj):
                return 1e10 if obj > 0 else -1e10  # Replace infinity with large number
            elif math.isnan(obj):
                return None  # Replace NaN with null
            else:
                return obj
        else:
            return obj


class SHMFunctionHandler(APIHandler):
    """Handler for SHM function discovery and metadata."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json file."""
        if self._config is not None:
            return self._config
            
        # Try multiple locations for config file
        possible_paths = [
            # 1. In the parent directory (for installed package)
            Path(__file__).parent.parent / "config.json",
            # 2. In the same directory as handler (for development)
            Path(__file__).parent / "config.json",
            # 3. In the repository (for TLJH installations)
            Path("/srv/classrepo/shm_function_selector/config.json"),
            # 4. In site-packages root (for pip installed)
            Path(__file__).parent.parent.parent / "config.json",
        ]
        
        config_path = None
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        
        try:
            if config_path and config_path.exists():
                with open(config_path, 'r') as f:
                    self._config = json.load(f)
                self.log.info(f"Loaded configuration from {config_path}")
            else:
                # Return default configuration
                self._config = self._get_default_config()
                self.log.info("Using default configuration (config.json not found)")
        except Exception as e:
            self.log.warning(f"Error loading config: {e}, using defaults")
            self._config = self._get_default_config()
            
        return self._config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "function_discovery": {
                "enabled": True,
                "discovery_mode": "modules",
                "modules_to_scan": [],
                "include_patterns": ["*_shm", "learn_*", "apply_*", "compute_*"],
                "exclude_patterns": ["_*", "__*", "*_internal", "*_helper"],
                "custom_categories": {}
            },
            "variable_discovery": {
                "enabled": True,
                "include_notebook_variables": True,
                "include_kernel_variables": False
            },
            "gui_integration": {
                "show_parameter_hints": True,
                "show_function_descriptions": True,
                "group_by_category": True,
                "max_functions_per_category": 50
            }
        }
    
    @authenticated
    def get(self):
        """Get list of available SHM functions with metadata."""
        try:
            functions = self._discover_shm_functions()
            # Use custom JSON encoder to handle infinity values
            self.finish(json.dumps(functions, cls=InfinityJSONEncoder))
        except Exception as e:
            self.log.error(f"Error discovering SHM functions: {e}")
            self.set_status(500)
            self.finish({"error": str(e)})
    
    def _discover_shm_functions(self) -> List[Dict[str, Any]]:
        """Discover SHM functions from shmtools package."""
        config = self._load_config()
        functions = []
        
        # Check if function discovery is enabled
        if not config.get("function_discovery", {}).get("enabled", True):
            self.log.info("Function discovery disabled in configuration")
            return []
        
        try:
            # Import main shmtools package and use its introspection system
            import shmtools
            
            # Try to use the built-in introspection system first
            try:
                from shm_function_selector.introspection import discover_functions_locally
                
                # Debug: log config details
                modules_to_scan = config.get('function_discovery', {}).get('modules_to_scan', [])
                examples_modules = [m for m in modules_to_scan if 'examples' in m]
                self.log.info(f"Config has {len(modules_to_scan)} modules, examples: {examples_modules}")
                
                discovered_functions = discover_functions_locally(config)
                
                # Filter functions based on configuration
                filtered_functions = self._filter_functions_by_config(discovered_functions, config)
                
                self.log.info(f"Found {len(filtered_functions)} functions using introspection system")
                
                # The introspection system returns functions in our format already
                return filtered_functions
                        
            except ImportError:
                self.log.info("Introspection system not available, using manual discovery")
                # Fallback to manual scanning
                functions = self._manual_function_discovery()
                # Filter functions based on configuration
                functions = self._filter_functions_by_config(functions, config)
                        
        except ImportError as e:
            self.log.warning(f"SHMTools not available: {e}")
            # Return some dummy functions for testing
            functions = self._get_dummy_functions()
            
        return functions
    
    def _manual_function_discovery(self) -> List[Dict[str, Any]]:
        """Manual function discovery as fallback."""
        config = self._load_config()
        functions = []
        
        # Get modules to scan from configuration
        modules_to_scan = config.get("function_discovery", {}).get("modules_to_scan", [])
        
        for module_name in modules_to_scan:
            try:
                module = importlib.import_module(module_name)
                category = self._get_category_from_module_name(module_name)
                
                # Check if module has __all__ attribute
                if hasattr(module, '__all__'):
                    function_names = module.__all__
                else:
                    # Find functions in the module
                    function_names = [name for name in dir(module) 
                                    if not name.startswith('_')]
                
                for name in function_names:
                    try:
                        obj = getattr(module, name)
                        
                        # Check if it's a callable function and passes filtering
                        if (callable(obj) and 
                            inspect.isfunction(obj) and
                            self._should_include_function(name, config)):
                            
                            func_info = self._extract_function_info(obj, name, category, module_name)
                            if func_info:
                                functions.append(func_info)
                    except Exception as e:
                        self.log.warning(f"Could not process {name} in {module_name}: {e}")
                        continue
                            
            except ImportError as e:
                self.log.warning(f"Could not import {module_name}: {e}")
                continue
        
        return functions
    
    def _get_dummy_functions(self) -> List[Dict[str, Any]]:
        """Return dummy functions for testing when shmtools is not available."""
        return [
            {
                'name': 'psd_welch',
                'displayName': 'Welch Power Spectral Density',
                'category': 'Core - Spectral Analysis',
                'module': 'shmtools.core.spectral',
                'signature': 'psd_welch(data, fs=1000, nperseg=256)',
                'description': 'Compute power spectral density using Welch method',
                'docstring': 'Estimates power spectral density using Welch method.',
                'parameters': [
                    {'name': 'data', 'type': 'numpy.ndarray', 'optional': False, 'default': None, 'description': 'Input signal data'},
                    {'name': 'fs', 'type': 'float', 'optional': True, 'default': '1000', 'description': 'Sampling frequency'},
                    {'name': 'nperseg', 'type': 'int', 'optional': True, 'default': '256', 'description': 'Length of each segment'}
                ]
            },
            {
                'name': 'ar_model',
                'displayName': 'AR Model Parameters',
                'category': 'Features - Time Series Models',
                'module': 'shmtools.features.time_series',
                'signature': 'ar_model(data, order=10)',
                'description': 'Estimate autoregressive model parameters',
                'docstring': 'Fits an autoregressive model to time series data.',
                'parameters': [
                    {'name': 'data', 'type': 'numpy.ndarray', 'optional': False, 'default': None, 'description': 'Input time series'},
                    {'name': 'order', 'type': 'int', 'optional': True, 'default': '10', 'description': 'AR model order'}
                ]
            },
            {
                'name': 'learn_pca',
                'displayName': 'Learn PCA Model',
                'category': 'Classification - Outlier Detection',
                'module': 'shmtools.classification.outlier_detection',
                'signature': 'learn_pca(features, n_components=5)',
                'description': 'Learn PCA model for outlier detection',
                'docstring': 'Learns a PCA model from feature data.',
                'parameters': [
                    {'name': 'features', 'type': 'numpy.ndarray', 'optional': False, 'default': None, 'description': 'Feature matrix'},
                    {'name': 'n_components', 'type': 'int', 'optional': True, 'default': '5', 'description': 'Number of PCA components'}
                ]
            }
        ]
    
    def _get_category_from_module_name(self, module_name: str) -> str:
        """Map module names to human-readable categories using config only."""
        config = self._load_config()
        custom_categories = config.get("function_discovery", {}).get("custom_categories", {})
        
        # Check custom categories first - exact match
        if module_name in custom_categories:
            return custom_categories[module_name]
            
        # Check for prefix matches
        for module_pattern, category in custom_categories.items():
            if module_name.startswith(module_pattern):
                return category
        
        # No hardcoded defaults - generate generic category from module name
        parts = module_name.split('.')
        if len(parts) >= 2:
            return f"{parts[0].title()} - {parts[-1].replace('_', ' ').title()}"
        else:
            return parts[0].title() if parts else "Other"
    
    def _should_include_function(self, function_name: str, config: Dict[str, Any]) -> bool:
        """Check if a function should be included based on configuration filters."""
        function_config = config.get("function_discovery", {})
        
        # Check include patterns
        include_patterns = function_config.get("include_patterns", [])
        if include_patterns:
            included = any(fnmatch.fnmatch(function_name, pattern) for pattern in include_patterns)
            if not included:
                return False
        
        # Check exclude patterns
        exclude_patterns = function_config.get("exclude_patterns", [])
        if exclude_patterns:
            excluded = any(fnmatch.fnmatch(function_name, pattern) for pattern in exclude_patterns)
            if excluded:
                return False
        
        return True
    
    def _filter_functions_by_config(self, functions: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter discovered functions based on configuration."""
        gui_config = config.get("gui_integration", {})
        max_per_category = gui_config.get("max_functions_per_category", 50)
        
        if max_per_category <= 0:
            return functions
        
        # Group functions by category
        by_category = {}
        for func in functions:
            category = func.get("category", "Other")
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(func)
        
        # Limit functions per category
        filtered_functions = []
        for category, funcs in by_category.items():
            if len(funcs) > max_per_category:
                # Sort by name and take first N
                funcs = sorted(funcs, key=lambda f: f.get("name", ""))[:max_per_category]
                self.log.info(f"Limited {category} to {max_per_category} functions")
            filtered_functions.extend(funcs)
        
        return filtered_functions
    
    def _extract_function_info(self, func, name: str, category: str, module_name: str) -> Dict[str, Any]:
        """Extract function information from docstring and signature."""
        try:
            # Get function signature
            sig = inspect.signature(func)
            
            # Parse docstring for metadata
            docstring = inspect.getdoc(func) or ""
            
            # Extract basic info
            func_info = {
                'name': name,
                'displayName': self._extract_display_name(docstring, name),
                'category': category,
                'module': module_name,
                'signature': str(sig),
                'description': self._extract_description(docstring),
                'docstring': docstring,
                'parameters': [],
                'guiMetadata': self._extract_gui_metadata(docstring),
                'returns': self._extract_return_info(docstring)
            }
            
            # Extract parameter information
            for param_name, param in sig.parameters.items():
                param_info = {
                    'name': param_name,
                    'type': str(param.annotation) if param.annotation != param.empty else 'Any',
                    'optional': param.default != param.empty,
                    'default': str(param.default) if param.default != param.empty else None
                }
                
                # Extract parameter description from docstring
                param_description = self._extract_parameter_description(docstring, param_name)
                if param_description:
                    param_info['description'] = param_description
                
                # Extract GUI widget metadata for the parameter
                gui_widget = self._extract_parameter_gui_widget(docstring, param_name)
                if gui_widget:
                    param_info['widget'] = gui_widget
                
                # Extract validation rules
                validation = self._extract_parameter_validation(docstring, param_name)
                if validation:
                    param_info['validation'] = validation
                
                func_info['parameters'].append(param_info)
            
            return func_info
            
        except Exception as e:
            self.log.warning(f"Could not extract info for function {name}: {e}")
            return None
    
    def _extract_display_name(self, docstring: str, fallback_name: str) -> str:
        """Extract human-readable display name from docstring."""
        # Look for display_name in meta section
        lines = docstring.split('\n')
        for line in lines:
            if ':display_name:' in line:
                return line.split(':display_name:')[1].strip()
        
        # Fallback: convert function name to readable format
        display_name = fallback_name.replace('_shm', '').replace('_', ' ')
        return ' '.join(word.capitalize() for word in display_name.split())
    
    def _extract_description(self, docstring: str) -> str:
        """Extract first line of docstring as description."""
        if not docstring:
            return ""
        
        # Get first non-empty line that's not a directive
        lines = docstring.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('..') and not line.startswith(':'):
                return line
        
        return ""
    
    def _extract_parameter_description(self, docstring: str, param_name: str) -> str:
        """Extract parameter description from docstring."""
        if not docstring:
            return ""
        
        lines = docstring.split('\n')
        in_parameters = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Look for Parameters section
            if stripped.lower() in ['parameters', 'parameters:', '----------']:
                in_parameters = True
                continue
            
            # Stop at next major section
            if in_parameters and stripped.lower() in ['returns', 'returns:', 'notes', 'notes:', 'examples', 'examples:']:
                break
                
            # Look for parameter definition
            if in_parameters and param_name in stripped and ':' in stripped:
                # Try to get description from same line or next lines
                desc_parts = []
                if ':' in stripped:
                    after_colon = stripped.split(':', 1)[1].strip()
                    if after_colon:
                        desc_parts.append(after_colon)
                
                # Look at following lines for continuation
                for j in range(i + 1, min(i + 3, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and not next_line.startswith(param_name) and ':' not in next_line:
                        desc_parts.append(next_line)
                    else:
                        break
                
                if desc_parts:
                    return ' '.join(desc_parts)
        
        return ""
    
    def _extract_gui_metadata(self, docstring: str) -> Dict[str, Any]:
        """Extract GUI metadata from function docstring."""
        metadata = {}
        
        if not docstring:
            return metadata
        
        lines = docstring.split('\n')
        in_meta = False
        
        for line in lines:
            stripped = line.strip()
            
            # Look for meta section
            if stripped.startswith('.. meta::'):
                in_meta = True
                continue
            
            # Stop at next major section
            if in_meta and stripped and not stripped.startswith(':') and not stripped.startswith(' '):
                break
            
            # Extract meta properties
            if in_meta and ':' in stripped:
                if stripped.startswith(':category:'):
                    metadata['category'] = stripped.split(':', 2)[2].strip()
                elif stripped.startswith(':complexity:'):
                    metadata['complexity'] = stripped.split(':', 2)[2].strip()
                elif stripped.startswith(':data_type:'):
                    metadata['data_type'] = stripped.split(':', 2)[2].strip()
                elif stripped.startswith(':output_type:'):
                    metadata['output_type'] = stripped.split(':', 2)[2].strip()
                elif stripped.startswith(':matlab_equivalent:'):
                    metadata['matlab_equivalent'] = stripped.split(':', 2)[2].strip()
                elif stripped.startswith(':verbose_call:'):
                    metadata['verbose_call'] = stripped.split(':', 2)[2].strip()
        
        return metadata
    
    def _extract_return_info(self, docstring: str) -> List[Dict[str, Any]]:
        """Extract return value information from docstring."""
        returns = []
        
        if not docstring:
            return returns
        
        lines = docstring.split('\n')
        in_returns = False
        current_return = None
        
        for line in lines:
            stripped = line.strip()
            
            # Look for Returns section
            if stripped.lower() in ['returns', 'returns:', '-------']:
                in_returns = True
                continue
            
            # Stop at next major section
            if in_returns and stripped.lower() in ['notes', 'notes:', 'examples', 'examples:', 'see also', 'references']:
                break
            
            # Parse return value - check original line for indentation 
            # Indented lines are descriptions, not new variables
            if in_returns and ':' in stripped and not line.startswith(' ') and not line.startswith('\t'):
                if current_return:
                    returns.append(current_return)
                
                # Parse name and type (format: "name : type")
                if ' : ' in stripped:
                    name, type_str = stripped.split(' : ', 1)
                    current_return = {
                        'name': name.strip(),
                        'type': type_str.strip(),
                        'description': ""
                    }
                else:
                    # Single part, treat as name only
                    parts = stripped.split(':', 1)
                    name_part = parts[0].strip()
                    description = parts[1].strip() if len(parts) > 1 else ""
                    current_return = {
                        'name': name_part,
                        'type': 'unknown',
                        'description': description
                    }
            elif in_returns and current_return and stripped:
                # Continuation of description
                current_return['description'] += ' ' + stripped
        
        if current_return:
            returns.append(current_return)
        
        return returns
    
    def _extract_parameter_gui_widget(self, docstring: str, param_name: str) -> Dict[str, Any]:
        """Extract GUI widget metadata for a parameter."""
        widget = {}
        
        if not docstring:
            return widget
        
        lines = docstring.split('\n')
        in_param_section = False
        in_gui_section = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Look for the specific parameter
            if param_name in stripped and ':' in stripped:
                in_param_section = True
                continue
            
            # Reset if we hit another parameter
            if in_param_section and ':' in stripped and any(p in stripped for p in ['array_like', 'int', 'float', 'str', 'bool']) and param_name not in stripped:
                in_param_section = False
                continue
            
            # Look for GUI section within parameter
            if in_param_section and stripped.startswith('.. gui::'):
                in_gui_section = True
                continue
            
            # Stop GUI section at next directive or parameter
            if in_gui_section and (stripped.startswith('..') or (stripped and not stripped.startswith(':'))):
                in_gui_section = False
            
            # Extract GUI properties
            if in_gui_section and ':' in stripped:
                if stripped.startswith(':widget:'):
                    widget['widget'] = stripped.split(':', 2)[2].strip()
                elif stripped.startswith(':min:'):
                    widget['min'] = float(stripped.split(':', 2)[2].strip())
                elif stripped.startswith(':max:'):
                    widget['max'] = float(stripped.split(':', 2)[2].strip())
                elif stripped.startswith(':default:'):
                    widget['default'] = stripped.split(':', 2)[2].strip()
                elif stripped.startswith(':options:'):
                    # Parse list format [option1, option2, ...]
                    options_str = stripped.split(':', 2)[2].strip()
                    if options_str.startswith('[') and options_str.endswith(']'):
                        options = [opt.strip(' "\'') for opt in options_str[1:-1].split(',')]
                        widget['options'] = options
                elif stripped.startswith(':formats:'):
                    # Parse formats for file uploads
                    formats_str = stripped.split(':', 2)[2].strip()
                    if formats_str.startswith('[') and formats_str.endswith(']'):
                        formats = [fmt.strip(' "\'') for fmt in formats_str[1:-1].split(',')]
                        widget['formats'] = formats
        
        return widget
    
    def _extract_parameter_validation(self, docstring: str, param_name: str) -> List[Dict[str, Any]]:
        """Extract validation rules for a parameter."""
        validation = []
        
        # For now, infer basic validation from GUI widget info
        widget = self._extract_parameter_gui_widget(docstring, param_name)
        
        if 'min' in widget:
            validation.append({
                'type': 'range',
                'min': widget['min'],
                'max': widget.get('max', float('inf'))
            })
        
        if 'options' in widget:
            validation.append({
                'type': 'choice',
                'options': widget['options']
            })
        
        if 'formats' in widget:
            validation.append({
                'type': 'file_format',
                'formats': widget['formats']
            })
        
        return validation


class SHMImportsHandler(APIHandler):
    """Handler for generating required imports based on available functions."""
    
    @authenticated
    def get(self):
        """Get list of required imports organized by module."""
        try:
            imports = self._analyze_required_imports()
            self.finish(json.dumps(imports, cls=InfinityJSONEncoder))
        except Exception as e:
            self.log.error(f"Error analyzing imports: {e}")
            self.set_status(500)
            self.finish({"error": str(e)})
    
    def _analyze_required_imports(self) -> List[str]:
        """Get list of high-level module imports."""
        # Get available functions from the main handler
        function_handler = SHMFunctionHandler(self.application, self.request)
        function_handler._config = None  # Reset config cache
        functions = function_handler._discover_shm_functions()
        
        # Find all top-level modules
        top_level_modules = set()
        
        for func in functions:
            module = func.get('module', '')
            if not module:
                continue
                
            # Extract top-level module (e.g., 'shmtools' from 'shmtools.core.spectral')
            module_parts = module.split('.')
            if module_parts:
                top_level_modules.add(module_parts[0])
        
        # Return sorted list of import statements
        return sorted([f"import {module}" for module in top_level_modules])


class SHMVariableHandler(APIHandler):
    """Handler for notebook variable parsing."""
    
    @authenticated
    def post(self):
        """Parse notebook code to extract variable assignments."""
        try:
            # Get request data
            data = json.loads(self.request.body.decode('utf-8'))
            notebook_cells = data.get('cells', [])
            
            # Parse variables from all code cells
            variables = self._parse_notebook_variables(notebook_cells)
            self.finish(json.dumps(variables, cls=InfinityJSONEncoder))
            
        except Exception as e:
            self.log.error(f"Error parsing variables: {e}")
            self.set_status(500)
            self.finish({"error": str(e)})
    
    def _parse_notebook_variables(self, cells: List[Dict]) -> List[Dict[str, Any]]:
        """Parse variable assignments from notebook cells."""
        import re
        
        variables = []
        
        for cell_index, cell in enumerate(cells):
            if cell.get('cell_type') != 'code':
                continue
                
            code = cell.get('source', '')
            if isinstance(code, list):
                code = '\n'.join(code)
            
            cell_variables = self._extract_variables_from_code(code, cell_index)
            variables.extend(cell_variables)
        
        return variables
    
    def _extract_variables_from_code(self, code: str, cell_index: int) -> List[Dict[str, Any]]:
        """Extract variable assignments from code text."""
        import re
        
        variables = []
        lines = code.split('\n')
        
        for line_index, line in enumerate(lines):
            line = line.strip()
            
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue
            
            # Assignment patterns
            patterns = [
                # Simple assignment: var = expression
                r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)',
                # Tuple unpacking: var1, var2 = expression
                r'^([a-zA-Z_][a-zA-Z0-9_,\s]*)\s*=\s*(.+)',
                # Parenthesized tuple: (var1, var2) = expression
                r'^\(([a-zA-Z_][a-zA-Z0-9_,\s]*)\)\s*=\s*(.+)'
            ]
            
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    left_side = match.group(1).strip()
                    right_side = match.group(2).strip()
                    
                    # Handle tuple unpacking
                    if ',' in left_side:
                        var_names = [v.strip().replace('(', '').replace(')', '') 
                                    for v in left_side.split(',')]
                        for var_name in var_names:
                            if var_name:
                                variables.append({
                                    'name': var_name,
                                    'type': self._infer_type_from_expression(right_side),
                                    'source': f'Cell {cell_index + 1}',
                                    'cellIndex': cell_index,
                                    'lineIndex': line_index,
                                    'expression': right_side
                                })
                    else:
                        # Single variable assignment
                        variables.append({
                            'name': left_side,
                            'type': self._infer_type_from_expression(right_side),
                            'source': f'Cell {cell_index + 1}',
                            'cellIndex': cell_index,
                            'lineIndex': line_index,
                            'expression': right_side
                        })
                    break
        
        return variables
    
    def _infer_type_from_expression(self, expression: str) -> str:
        """Infer variable type from expression."""
        import re
        
        # Remove comments
        expression = expression.split('#')[0].strip()
        
        # SHM function patterns
        if 'shmtools.' in expression:
            if any(func in expression for func in ['ar_model', 'pca', 'mahalanobis']):
                return 'tuple'
            if any(func in expression for func in ['load_', 'import_']):
                return 'numpy.ndarray'
        
        # NumPy patterns
        if 'np.' in expression or 'numpy.' in expression:
            if any(func in expression for func in ['.array', '.zeros', '.ones', '.randn', '.random']):
                return 'numpy.ndarray'
            if any(func in expression for func in ['.mean', '.std', '.sum']):
                return 'float'
        
        # Literal patterns
        if re.match(r'^\d+$', expression):
            return 'int'
        if re.match(r'^\d+\.\d+$', expression):
            return 'float'
        if expression.startswith('"') or expression.startswith("'"):
            return 'str'
        if expression.startswith('[') and expression.endswith(']'):
            return 'list'
        if expression.startswith('(') and expression.endswith(')'):
            return 'tuple'
        if expression.startswith('{') and expression.endswith('}'):
            return 'dict'
        
        return 'unknown'


def setup_handlers(web_app):
    """Setup the server extension handlers."""
    host_pattern = ".*$"
    
    base_url = web_app.settings["base_url"]
    route_pattern = url_path_join(base_url, "shm-function-selector", "(.*)") 
    
    handlers = [
        (url_path_join(base_url, "shm-function-selector", "functions"), SHMFunctionHandler),
        (url_path_join(base_url, "shm-function-selector", "imports"), SHMImportsHandler),
        (url_path_join(base_url, "shm-function-selector", "variables"), SHMVariableHandler),
    ]
    
    web_app.add_handlers(host_pattern, handlers)