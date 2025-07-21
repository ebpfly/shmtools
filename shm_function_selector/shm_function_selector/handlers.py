"""
Server-side handlers for SHM Function Selector JupyterLab extension.
Provides API endpoints for function discovery and variable parsing.
"""

import json
import inspect
import importlib
from typing import List, Dict, Any

from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join
import tornado
from tornado.web import authenticated


class SHMFunctionHandler(APIHandler):
    """Handler for SHM function discovery and metadata."""
    
    @authenticated
    def get(self):
        """Get list of available SHM functions with metadata."""
        try:
            functions = self._discover_shm_functions()
            self.finish(json.dumps(functions))
        except Exception as e:
            self.log.error(f"Error discovering SHM functions: {e}")
            self.set_status(500)
            self.finish({"error": str(e)})
    
    def _discover_shm_functions(self) -> List[Dict[str, Any]]:
        """Discover SHM functions from shmtools package."""
        functions = []
        
        try:
            # Import main shmtools package
            import shmtools
            
            # Define modules to scan
            modules_to_scan = [
                'shmtools.core.spectral',
                'shmtools.core.statistics', 
                'shmtools.core.filtering',
                'shmtools.core.preprocessing',
                'shmtools.features.time_series',
                'shmtools.classification.outlier_detection',
            ]
            
            for module_name in modules_to_scan:
                try:
                    module = importlib.import_module(module_name)
                    category = self._get_category_from_module_name(module_name)
                    
                    # Find functions in the module
                    for name in dir(module):
                        obj = getattr(module, name)
                        
                        # Check if it's a callable function (not class or builtin)
                        if (callable(obj) and 
                            not name.startswith('_') and
                            inspect.isfunction(obj) and
                            hasattr(obj, '__module__') and
                            obj.__module__ == module_name):
                            
                            func_info = self._extract_function_info(obj, name, category)
                            if func_info:
                                functions.append(func_info)
                                
                except ImportError as e:
                    self.log.warning(f"Could not import {module_name}: {e}")
                    continue
                    
        except ImportError as e:
            self.log.warning(f"SHMTools not available: {e}")
            
        return functions
    
    def _get_category_from_module_name(self, module_name: str) -> str:
        """Map module names to human-readable categories."""
        category_map = {
            'shmtools.core.spectral': 'Core - Spectral Analysis',
            'shmtools.core.statistics': 'Core - Statistics', 
            'shmtools.core.filtering': 'Core - Filtering',
            'shmtools.core.preprocessing': 'Core - Preprocessing',
            'shmtools.features.time_series': 'Features - Time Series Models',
            'shmtools.classification.outlier_detection': 'Classification - Outlier Detection',
        }
        return category_map.get(module_name, 'Other')
    
    def _extract_function_info(self, func, name: str, category: str) -> Dict[str, Any]:
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
                'signature': str(sig),
                'description': self._extract_description(docstring),
                'docstring': docstring,
                'parameters': []
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
            self.finish(json.dumps(variables))
            
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
        (url_path_join(base_url, "shm-function-selector", "variables"), SHMVariableHandler),
    ]
    
    web_app.add_handlers(host_pattern, handlers)