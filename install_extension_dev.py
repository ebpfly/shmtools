#!/usr/bin/env python3
"""
Development installation script for the SHM Jupyter extension.
This creates a simple way to test the extension in development.
"""

import os
import json
import shutil
from pathlib import Path

def install_extension():
    """Install the extension for development testing."""
    
    # Get Jupyter data directory
    from jupyter_core.paths import jupyter_data_dir
    
    jupyter_data = Path(jupyter_data_dir())
    nbextensions_dir = jupyter_data / "nbextensions" / "shm_function_selector"
    
    print(f"Installing extension to: {nbextensions_dir}")
    
    # Create the directory
    nbextensions_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy static files
    static_dir = Path(__file__).parent / "jupyter_shm_extension" / "static"
    
    if static_dir.exists():
        for file in static_dir.glob("*"):
            shutil.copy2(file, nbextensions_dir)
            print(f"Copied: {file.name}")
    
    # Create extension config
    config = {
        "load_extensions": {
            "shm_function_selector/main": True
        }
    }
    
    # Write config file
    config_dir = jupyter_data / "nbconfig"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    notebook_config_file = config_dir / "notebook.json"
    
    # Load existing config or create new
    if notebook_config_file.exists():
        with open(notebook_config_file, 'r') as f:
            existing_config = json.load(f)
    else:
        existing_config = {}
    
    # Merge configs
    if "load_extensions" not in existing_config:
        existing_config["load_extensions"] = {}
    
    existing_config["load_extensions"]["shm_function_selector/main"] = True
    
    # Write updated config
    with open(notebook_config_file, 'w') as f:
        json.dump(existing_config, f, indent=2)
    
    print(f"Updated notebook config: {notebook_config_file}")
    print("\n✅ Extension installed!")
    print("\nTo test:")
    print("1. Start Jupyter notebook: jupyter notebook")
    print("2. Open test_phase2_notebook.ipynb")
    print("3. Look for 'SHM Functions' dropdown in toolbar")
    print("4. Right-click on parameters to see context menu")
    
    return True

def uninstall_extension():
    """Remove the extension."""
    from jupyter_core.paths import jupyter_data_dir
    
    jupyter_data = Path(jupyter_data_dir())
    nbextensions_dir = jupyter_data / "nbextensions" / "shm_function_selector"
    
    if nbextensions_dir.exists():
        shutil.rmtree(nbextensions_dir)
        print(f"Removed: {nbextensions_dir}")
    
    # Remove from config
    config_dir = jupyter_data / "nbconfig"
    notebook_config_file = config_dir / "notebook.json"
    
    if notebook_config_file.exists():
        with open(notebook_config_file, 'r') as f:
            config = json.load(f)
        
        if "load_extensions" in config:
            config["load_extensions"].pop("shm_function_selector/main", None)
            
            with open(notebook_config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
        print(f"Updated config: {notebook_config_file}")
    
    print("✅ Extension uninstalled!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "uninstall":
        uninstall_extension()
    else:
        print("Installing SHM Jupyter Extension for Development...")
        try:
            install_extension()
        except Exception as e:
            print(f"❌ Installation failed: {e}")
            print("\nTrying alternative method...")
            
            # Alternative: just copy files to current directory
            static_dir = Path("jupyter_shm_extension/static")
            if static_dir.exists():
                print("Copying extension files for manual testing...")
                for file in static_dir.glob("*"):
                    print(f"  - {file.name}")
                
                print("\n📝 Manual testing instructions:")
                print("1. Start Jupyter notebook")
                print("2. In a notebook cell, run:")
                print("   %%html")
                print('   <script src="files/jupyter_shm_extension/static/main.js"></script>')
                print('   <link rel="stylesheet" href="files/jupyter_shm_extension/static/main.css">')
                print("3. This will load the extension for that notebook session")