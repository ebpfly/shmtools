#!/usr/bin/env python3
"""
Demo script to show SHM Function Selector extension capabilities.
This simulates what the JavaScript extension would do in the browser.
"""

import json
from test_function_discovery import discover_shm_functions, get_category_from_module_name

def generate_function_call_code(func_info):
    """Generate Python code for a function call with parameters."""
    lines = []
    
    # Add comment with function description
    if func_info['description']:
        lines.append(f'# {func_info["description"]}')
    
    # Generate function call
    params = []
    for param in func_info['parameters']:
        if param['name'] == 'self':
            continue  # Skip self parameter
        
        param_str = f'{param["name"]}='
        
        if param['default'] and param['default'] != 'None':
            # Use default value
            param_str += param['default']
        elif param['optional']:
            # Optional parameter with None default
            param_str += f'None  # TODO: Set {param["name"]} ({param["type"]})'
        else:
            # Required parameter
            param_str += f'None  # REQUIRED: {param["name"]} ({param["type"]})'
        
        params.append(param_str)
    
    # Determine output variables (simplified heuristics)
    outputs = 'result'
    func_name = func_info['name'].lower()
    
    if 'pca' in func_name or 'svd' in func_name or 'mahalanobis' in func_name:
        if func_name.startswith('learn'):
            outputs = 'model'
        elif func_name.startswith('score'):
            outputs = 'scores, outliers'
    elif 'ar_model' in func_name:
        outputs = 'features, residuals'
    elif 'psd' in func_name or 'spectral' in func_name:
        outputs = 'frequencies, power_spectrum'
    elif 'filter' in func_name:
        outputs = 'filtered_signal'
    
    # Build function call
    function_call = f'{outputs} = shmtools.{func_info["name"]}('
    if params:
        function_call += '\n    ' + ',\n    '.join(params) + '\n'
    function_call += ')'
    
    lines.append(function_call)
    return '\n'.join(lines)

def demo_dropdown_population():
    """Demo: Show how dropdown would be populated."""
    print("=" * 60)
    print("DEMO: SHM Function Selector Dropdown")
    print("=" * 60)
    
    functions = discover_shm_functions()
    
    # Group by category (simulating dropdown structure)
    categories = {}
    for func in functions:
        cat = func['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(func)
    
    print("Dropdown structure:")
    print("-" * 40)
    for category, funcs in categories.items():
        print(f"\n📁 {category}")
        for func in funcs:
            print(f"   • {func['display_name']}")
    
    return functions, categories

def demo_code_generation():
    """Demo: Show code generation for selected functions."""
    print("\n" + "=" * 60)
    print("DEMO: Code Generation for Selected Functions")
    print("=" * 60)
    
    functions = discover_shm_functions()
    
    # Demo with a few representative functions
    demo_functions = [
        'psd_welch',           # Spectral analysis
        'ar_model',            # Time series modeling
        'learn_pca',           # Outlier detection learning
        'score_mahalanobis',   # Outlier detection scoring
        'bandpass_filter'      # Signal filtering
    ]
    
    for func_name in demo_functions:
        # Find the function
        func_info = next((f for f in functions if f['name'] == func_name), None)
        if func_info:
            print(f"\nSelected: {func_info['display_name']}")
            print(f"Category: {func_info['category']}")
            print("\nGenerated code:")
            print("-" * 30)
            code = generate_function_call_code(func_info)
            print(code)
            print()

def demo_parameter_linking():
    """Demo: Show parameter linking capabilities."""
    print("=" * 60)
    print("DEMO: Parameter Linking (Right-click Context Menu)")
    print("=" * 60)
    
    # Simulate available variables from previous cells
    mock_variables = [
        {'name': 'acceleration_data', 'type': 'numpy.ndarray', 'shape': '(10000, 4)', 'cell': 1},
        {'name': 'sampling_freq', 'type': 'float', 'value': '1000.0', 'cell': 1},
        {'name': 'ar_features', 'type': 'numpy.ndarray', 'shape': '(100, 15)', 'cell': 2},
        {'name': 'pca_model', 'type': 'dict', 'keys': "['components', 'mean', 'std']", 'cell': 3},
        {'name': 'test_features', 'type': 'numpy.ndarray', 'shape': '(50, 15)', 'cell': 4}
    ]
    
    print("Available variables from previous cells:")
    for var in mock_variables:
        print(f"  Cell {var['cell']}: {var['name']} ({var['type']})")
        if 'shape' in var:
            print(f"           Shape: {var['shape']}")
        elif 'value' in var:
            print(f"           Value: {var['value']}")
        elif 'keys' in var:
            print(f"           Keys: {var['keys']}")
    
    print("\nExample: Right-clicking on 'X=' in score_pca() call")
    print("Context menu would show:")
    print("  🔗 Link to variable:")
    print("     • acceleration_data (numpy.ndarray) - from Cell 1")
    print("     • ar_features (numpy.ndarray) - from Cell 2")  
    print("     • test_features (numpy.ndarray) - from Cell 4")
    print("\nAfter selection: X=ar_features")

def main():
    """Run all demos."""
    print("SHM Function Selector Extension Demo")
    print("Demonstrating Phase 1 functionality")
    
    # Demo 1: Function discovery and dropdown
    functions, categories = demo_dropdown_population()
    
    # Demo 2: Code generation
    demo_code_generation()
    
    # Demo 3: Parameter linking
    demo_parameter_linking()
    
    print("\n" + "=" * 60)
    print("SUMMARY: Phase 1 Implementation Complete")
    print("=" * 60)
    print(f"✓ Function Discovery: {len(functions)} functions found")
    print(f"✓ Categorization: {len(categories)} categories")
    print("✓ Code Generation: Smart parameter handling")
    print("✓ Parameter Linking: Context menu simulation")
    print("\nReady for notebook integration!")

if __name__ == "__main__":
    main()