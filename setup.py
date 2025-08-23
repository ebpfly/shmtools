#!/usr/bin/env python
"""Setup script for SHMTools Python package."""

from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import subprocess
import sys

# Read the README file for long description
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f 
                   if line.strip() and not line.startswith('#')]

# Read dev requirements
with open(os.path.join(this_directory, 'requirements-dev.txt'), encoding='utf-8') as f:
    dev_requirements = [line.strip() for line in f 
                       if line.strip() and not line.startswith('#')]

# Read advanced requirements
with open(os.path.join(this_directory, 'requirements-advanced.txt'), encoding='utf-8') as f:
    advanced_requirements = [line.strip() for line in f 
                            if line.strip() and not line.startswith('#')]

# Read hardware requirements
with open(os.path.join(this_directory, 'requirements-hardware.txt'), encoding='utf-8') as f:
    hardware_requirements = [line.strip() for line in f 
                            if line.strip() and not line.startswith('#')]


class PostInstallCommand(install):
    """Custom post-installation to install JupyterLab extension."""
    
    def run(self):
        # First run the normal install
        install.run(self)
        
        # Then install the extension
        if not self.dry_run:
            self.execute(self.install_jupyter_extension, [], msg="Installing JupyterLab extension")
    
    def install_jupyter_extension(self):
        """Install the SHM JupyterLab extension after package installation."""
        print("\n" + "="*60)
        print("🔧 Installing SHM JupyterLab Extension...")
        print("="*60)
        
        try:
            # Get the extension directory path
            extension_dir = os.path.join(this_directory, 'shm_function_selector')
            
            if not os.path.exists(extension_dir):
                print(f"⚠️  Extension directory not found: {extension_dir}")
                return
                
            # Change to extension directory and install
            old_cwd = os.getcwd()
            try:
                os.chdir(extension_dir)
                
                # Install the extension package
                print("📦 Installing extension Python package...")
                result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ Extension Python package installed")
                else:
                    print(f"⚠️  Extension package install failed: {result.stderr}")
                    return
                
                # Register with JupyterLab
                print("🔗 Registering extension with JupyterLab...")
                result = subprocess.run(['jupyter', 'labextension', 'develop', '.', '--overwrite'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ Extension registered with JupyterLab")
                else:
                    print(f"⚠️  Extension registration failed: {result.stderr}")
                    return
                
                # Build JupyterLab
                print("🔨 Building JupyterLab (this may take a moment)...")
                result = subprocess.run(['jupyter', 'lab', 'build'], 
                                      capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    print("✅ JupyterLab built successfully")
                else:
                    print(f"⚠️  JupyterLab build failed: {result.stderr}")
                
            finally:
                os.chdir(old_cwd)
            
            print("\n" + "="*60)
            print("🎉 SHM JupyterLab Extension installed successfully!")
            print("="*60)
            print("To start using SHMTools:")
            print("  jupyter lab")
            print("\nLook for the 'SHM Functions' panel in the left sidebar!")
            print("="*60)
                
        except subprocess.TimeoutExpired:
            print("⚠️  JupyterLab build timed out, but extension may still work")
            print("Try running: jupyter lab build")
        except FileNotFoundError as e:
            print(f"⚠️  Could not find required command: {e}")
            print("Make sure JupyterLab is installed and in your PATH")
            print("You can install the extension manually with: shmtools-install-jupyter")
        except Exception as e:
            print(f"⚠️  Extension installation failed: {e}")
            print("You can install the extension manually with: shmtools-install-jupyter")


setup(
    name='shmtools',
    version='0.1.0',
    description='Python-based Structural Health Monitoring Toolkit',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='SHMTools Development Team',
    url='https://github.com/ebpfly/shm',
    license='BSD-3-Clause',
    packages=find_packages(),
    package_dir={'': '.'},
    python_requires='>=3.10',
    install_requires=requirements,
    extras_require={
        'dev': dev_requirements,
        'hardware': hardware_requirements,
    },
    entry_points={
        'console_scripts': [
            'install-jfuse=shmtools.jupyter_extension_installer:install_extension',
            'uninstall-jfuse=shmtools.jupyter_extension_installer:uninstall_extension',
            # 'shmtools-gui=bokeh_shmtools.app:main',  # Archived - bokeh GUI on pause
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    keywords='structural-health-monitoring signal-processing machine-learning modal-analysis',
    include_package_data=True,
    zip_safe=False,
)