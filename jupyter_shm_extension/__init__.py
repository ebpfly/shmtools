"""
SHM Function Selector Jupyter Extension

A Jupyter Notebook extension that provides:
1. Dropdown menu for SHM function selection
2. Right-click context menus for parameter linking
3. Auto-populated default values from function docstrings
"""

def _jupyter_nbextension_paths():
    """Return metadata for the nbextension."""
    return [
        {
            "section": "notebook", 
            "src": "static",
            "dest": "shm_function_selector",
            "require": "shm_function_selector/main"
        }
    ]

def _jupyter_server_extension_paths():
    """Return metadata for the server extension."""
    return [
        {
            "module": "jupyter_shm_extension.handlers"
        }
    ]

def _load_jupyter_server_extension(nb_server_app):
    """Load the server extension for Jupyter Server."""
    from .handlers import setup_handlers
    setup_handlers(nb_server_app.web_app)
    nb_server_app.log.info("SHM Function Selector extension loaded")

def load_jupyter_server_extension(nb_server_app):
    """Legacy function name for older Jupyter versions."""
    return _load_jupyter_server_extension(nb_server_app)

# Entry point for modern Jupyter
def main():
    """Entry point for console script."""
    print("SHM Jupyter Extension installed successfully!")
    print("Restart Jupyter notebook to see the extension.")