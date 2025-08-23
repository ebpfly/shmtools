"""
Installation utilities for the SHM JupyterLab extension.
"""

import subprocess
import sys
import os
import atexit
from pathlib import Path

def install_extension_on_import():
    """Automatically install extension when module is imported (post-install hook)."""
    try:
        # Check if we're in a pip install process
        if 'pip' in sys.modules or any('pip' in arg for arg in sys.argv):
            # Schedule installation for after pip finishes
            atexit.register(_delayed_install)
        else:
            # Direct installation
            install_extension()
    except Exception as e:
        print(f"Warning: Could not auto-install JupyterLab extension: {e}")

def _delayed_install():
    """Install extension after pip finishes."""
    try:
        install_extension()
    except Exception as e:
        print(f"Warning: Could not auto-install JupyterLab extension: {e}")

def install_extension():
    """Install the SHM JupyterLab extension."""
    
    try:
        # Get the path to the extension directory
        current_dir = Path(__file__).parent.parent
        extension_dir = current_dir / "shm_function_selector"
        
        if not extension_dir.exists():
            print(f"❌ Extension directory not found: {extension_dir}")
            return False
        
        print("🔧 Installing SHM JupyterLab Extension...")
        print(f"Extension directory: {extension_dir}")
        
        # Change to extension directory
        old_cwd = os.getcwd()
        os.chdir(extension_dir)
        
        try:
            # Install the extension in development mode
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", "."],
                check=True,
                capture_output=True,
                text=True
            )
            
            print("✅ Extension Python package installed")
            
            # Install extension into JupyterLab
            result = subprocess.run(
                ["jupyter", "labextension", "develop", ".", "--overwrite"],
                check=True,
                capture_output=True,
                text=True
            )
            
            print("✅ Extension registered with JupyterLab")
            
            # Build JupyterLab
            print("🔨 Building JupyterLab...")
            result = subprocess.run(
                ["jupyter", "lab", "build", "--dev-build=False", "--minimize=False"],
                check=True,
                capture_output=True,
                text=True
            )
            
            print("✅ JupyterLab built successfully")
            
            print("\n🎉 SHM JupyterLab Extension installed successfully!")
            print("\nTo use the extension:")
            print("1. Start JupyterLab: jupyter lab")
            print("2. Look for the 'SHM Functions' panel in the left sidebar")
            print("3. Open a Python notebook and use the function selector")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Installation failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return False
        finally:
            os.chdir(old_cwd)
            
    except Exception as e:
        print(f"❌ Installation failed with error: {e}")
        return False


def uninstall_extension():
    """Uninstall the SHM JupyterLab extension."""
    
    try:
        print("🗑️ Uninstalling SHM JupyterLab Extension...")
        
        # Uninstall the extension from JupyterLab
        result = subprocess.run(
            ["jupyter", "labextension", "uninstall", "shm-function-selector"],
            check=True,
            capture_output=True,
            text=True
        )
        
        print("✅ Extension removed from JupyterLab")
        
        # Uninstall the Python package
        result = subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "shm-function-selector"],
            check=True,
            capture_output=True,
            text=True
        )
        
        print("✅ Extension Python package uninstalled")
        
        # Build JupyterLab
        print("🔨 Rebuilding JupyterLab...")
        result = subprocess.run(
            ["jupyter", "lab", "build"],
            check=True,
            capture_output=True,
            text=True
        )
        
        print("✅ JupyterLab rebuilt")
        print("🎉 SHM JupyterLab Extension uninstalled successfully!")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Uninstallation failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Uninstallation failed with error: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "uninstall":
        uninstall_extension()
    else:
        install_extension()