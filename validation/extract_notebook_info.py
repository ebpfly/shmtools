#!/usr/bin/env python3
"""Extract key information from notebooks for validation."""

import json
import os
from pathlib import Path
import re


def extract_notebook_info(notebook_path):
    """Extract structure and outputs from a notebook."""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    info = {
        'title': '',
        'sections': [],
        'numerical_outputs': [],
        'plots': [],
        'code_cells': 0,
        'markdown_cells': 0
    }
    
    # Extract sections from markdown cells
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            info['markdown_cells'] += 1
            source = ''.join(cell['source'])
            
            # Extract headers
            headers = re.findall(r'^(#+)\s+(.+)$', source, re.MULTILINE)
            for level, title in headers:
                if len(level) == 1 and not info['title']:
                    info['title'] = title.strip()
                info['sections'].append({
                    'level': len(level),
                    'title': title.strip()
                })
        
        elif cell['cell_type'] == 'code':
            info['code_cells'] += 1
            
            # Check for plot commands
            source = ''.join(cell['source'])
            if 'plt.show()' in source or 'plt.plot' in source or 'plt.scatter' in source:
                # Extract plot details
                plot_info = {
                    'type': 'unknown',
                    'title': '',
                    'xlabel': '',
                    'ylabel': ''
                }
                
                if 'plt.plot' in source:
                    plot_info['type'] = 'line'
                elif 'plt.scatter' in source:
                    plot_info['type'] = 'scatter'
                elif 'plt.bar' in source:
                    plot_info['type'] = 'bar'
                elif 'plt.hist' in source:
                    plot_info['type'] = 'histogram'
                
                # Extract labels
                title_match = re.search(r"plt\.title\(['\"](.*?)['\"]\)", source)
                if title_match:
                    plot_info['title'] = title_match.group(1)
                
                xlabel_match = re.search(r"plt\.xlabel\(['\"](.*?)['\"]\)", source)
                if xlabel_match:
                    plot_info['xlabel'] = xlabel_match.group(1)
                
                ylabel_match = re.search(r"plt\.ylabel\(['\"](.*?)['\"]\)", source)
                if ylabel_match:
                    plot_info['ylabel'] = ylabel_match.group(1)
                
                info['plots'].append(plot_info)
            
            # Check for numerical outputs in cell outputs
            if 'outputs' in cell:
                for output in cell['outputs']:
                    if 'text' in output:
                        text = ''.join(output['text'])
                        # Look for numerical values
                        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
                        if numbers and len(text) < 200:  # Short outputs likely to be results
                            info['numerical_outputs'].append({
                                'text': text.strip(),
                                'numbers': numbers
                            })
    
    return info


def generate_validation_summary():
    """Generate a summary of all notebooks for validation."""
    notebooks_dir = Path('examples/notebooks')
    summary = []
    
    for category in ['basic', 'intermediate', 'advanced']:
        category_dir = notebooks_dir / category
        if category_dir.exists():
            for nb_path in category_dir.glob('*.ipynb'):
                if 'checkpoint' not in str(nb_path):
                    info = extract_notebook_info(nb_path)
                    summary.append({
                        'name': nb_path.stem,
                        'category': category,
                        'path': str(nb_path),
                        'info': info
                    })
    
    return summary


def main():
    """Generate validation information."""
    summary = generate_validation_summary()
    
    # Create a summary report
    with open('validation/notebook_summary.md', 'w') as f:
        f.write("# Notebook Structure Summary\n\n")
        
        for nb in summary:
            f.write(f"## {nb['name']}\n")
            f.write(f"**Category**: {nb['category']}\n")
            f.write(f"**Title**: {nb['info']['title']}\n")
            f.write(f"**Code Cells**: {nb['info']['code_cells']}\n")
            f.write(f"**Markdown Cells**: {nb['info']['markdown_cells']}\n\n")
            
            f.write("### Sections\n")
            for section in nb['info']['sections']:
                indent = '  ' * (section['level'] - 1)
                f.write(f"{indent}- {section['title']}\n")
            
            f.write(f"\n### Plots ({len(nb['info']['plots'])} total)\n")
            for i, plot in enumerate(nb['info']['plots'], 1):
                f.write(f"{i}. **Type**: {plot['type']}\n")
                if plot['title']:
                    f.write(f"   - Title: {plot['title']}\n")
                if plot['xlabel']:
                    f.write(f"   - X-axis: {plot['xlabel']}\n")
                if plot['ylabel']:
                    f.write(f"   - Y-axis: {plot['ylabel']}\n")
            
            f.write(f"\n### Numerical Outputs ({len(nb['info']['numerical_outputs'])} captured)\n")
            for i, output in enumerate(nb['info']['numerical_outputs'][:3], 1):
                f.write(f"{i}. {output['text'][:100]}{'...' if len(output['text']) > 100 else ''}\n")
            
            f.write("\n---\n\n")
    
    print(f"Generated summary for {len(summary)} notebooks")
    print("Summary saved to: validation/notebook_summary.md")


if __name__ == '__main__':
    main()