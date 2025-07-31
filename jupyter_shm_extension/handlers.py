"""
Server-side handlers for SHM Function Selector extension.
Provides API endpoints for function discovery and metadata.
"""

import json
import inspect
import importlib
from tornado import web

# Handle different notebook versions
try:
    from notebook.base.handlers import IPythonHandler
except ImportError:
    try:
        from jupyter_server.base.handlers import JupyterHandler as IPythonHandler
    except ImportError:
        # Fallback for testing
        from tornado.web import RequestHandler as IPythonHandler


class SHMFunctionDiscoveryHandler(IPythonHandler):
    """Handler for discovering available SHM functions."""
    
    @web.authenticated
    def get(self):
        """Get list of available SHM functions with metadata."""
        try:
            functions = self._discover_shm_functions()
            self.finish(json.dumps(functions))
        except Exception as e:
            self.set_status(500)
            self.finish({"error": str(e)})
    
    def _discover_shm_functions(self):
        """Discover SHM functions from shmtools package."""
        return self._discover_shm_functions_static()
    
    @staticmethod
    def _discover_shm_functions_static():
        """Static version of function discovery for testing."""
        functions = []
        
        try:
            # Import main shmtools package
            import shmtools
            
            # Define modules to scan
            modules_to_scan = [
                'shmtools.core.spectral',
                'shmtools.core.statistics', 
                'shmtools.core.filtering',
                'shmtools.features.time_series',
                'shmtools.classification.outlier_detection',
                # Add more modules as needed
            ]
            
            for module_name in modules_to_scan:
                try:
                    module = importlib.import_module(module_name)
                    category = SHMFunctionDiscoveryHandler._get_category_from_module_name_static(module_name)
                    
                    # Find functions in the module
                    for name in dir(module):
                        obj = getattr(module, name)
                        
                        # Check if it's a callable function (not class or builtin)
                        if (callable(obj) and 
                            not name.startswith('_') and
                            inspect.isfunction(obj)):
                            
                            func_info = SHMFunctionDiscoveryHandler._extract_function_info_static(obj, name, category)
                            if func_info:
                                functions.append(func_info)
                                
                except ImportError:
                    # Skip modules that aren't available yet
                    continue
                    
        except ImportError:
            # If shmtools isn't available, return empty list
            pass
            
        return functions
    
    def _get_category_from_module_name(self, module_name):
        """Map module names to human-readable categories."""
        return self._get_category_from_module_name_static(module_name)
    
    @staticmethod
    def _get_category_from_module_name_static(module_name):
        """Static version of category mapping."""
        category_map = {
            'shmtools.core.spectral': 'Core - Spectral Analysis',
            'shmtools.core.statistics': 'Core - Statistics', 
            'shmtools.core.filtering': 'Core - Filtering',
            'shmtools.features.time_series': 'Features - Time Series Models',
            'shmtools.classification.outlier_detection': 'Classification - Outlier Detection',
        }
        return category_map.get(module_name, 'Other')
    
    def _extract_function_info(self, func, name, category):
        """Extract function information from docstring and signature."""
        return self._extract_function_info_static(func, name, category)
    
    @staticmethod
    def _extract_function_info_static(func, name, category):
        """Static version of function info extraction."""
        try:
            # Get function signature
            sig = inspect.signature(func)
            
            # Parse docstring for metadata
            docstring = inspect.getdoc(func) or ""
            
            # Extract basic info
            func_info = {
                'name': name,
                'category': category,
                'signature': str(sig),
                'docstring': docstring,
                'parameters': [],
                'display_name': SHMFunctionDiscoveryHandler._extract_display_name_static(docstring, name),
                'description': SHMFunctionDiscoveryHandler._extract_description_static(docstring)
            }
            
            # Extract parameter information
            for param_name, param in sig.parameters.items():
                param_info = {
                    'name': param_name,
                    'type': str(param.annotation) if param.annotation != param.empty else 'Any',
                    'default': str(param.default) if param.default != param.empty else None,
                    'optional': param.default != param.empty
                }
                func_info['parameters'].append(param_info)
            
            return func_info
            
        except Exception:
            # Skip functions we can't parse
            return None
    
    def _extract_display_name(self, docstring, fallback_name):
        """Extract human-readable display name from docstring."""
        return self._extract_display_name_static(docstring, fallback_name)
    
    @staticmethod
    def _extract_display_name_static(docstring, fallback_name):
        """Static version of display name extraction."""
        # Look for display_name in meta section
        lines = docstring.split('\\n')
        for line in lines:
            if ':display_name:' in line:
                return line.split(':display_name:')[1].strip()
        
        # Fallback: convert function name to readable format
        return fallback_name.replace('_shm', '').replace('_', ' ').title()
    
    def _extract_description(self, docstring):
        """Extract first line of docstring as description."""
        return self._extract_description_static(docstring)
    
    @staticmethod
    def _extract_description_static(docstring):
        """Static version of description extraction."""
        if not docstring:
            return ""
        
        # Get first non-empty line
        lines = docstring.split('\\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('..'):
                return line
        
        return ""


class SHMVariableInspectionHandler(IPythonHandler):
    """Handler for inspecting notebook variables."""
    
    @web.authenticated
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
            self.set_status(500)
            self.finish({"error": str(e)})
    
    def _parse_notebook_variables(self, cells):
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
    
    def _extract_variables_from_code(self, code, cell_index):
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
                                    'cell_index': cell_index,
                                    'line_index': line_index,
                                    'expression': right_side
                                })
                    else:
                        # Single variable assignment
                        variables.append({
                            'name': left_side,
                            'type': self._infer_type_from_expression(right_side),
                            'source': f'Cell {cell_index + 1}',
                            'cell_index': cell_index,
                            'line_index': line_index,
                            'expression': right_side
                        })
                    break
        
        return variables
    
    def _infer_type_from_expression(self, expression):
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
    host_pattern = '.*$'
    
    route_pattern = web_app.settings['base_url'] + r'shm_extension/(.*)'
    
    handlers = [
        (route_pattern.replace('(.*)', 'functions'), SHMFunctionDiscoveryHandler),
        (route_pattern.replace('(.*)', 'variables'), SHMVariableInspectionHandler),
    ]
    
    web_app.add_handlers(host_pattern, handlers)