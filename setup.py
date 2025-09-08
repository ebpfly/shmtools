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
        
        # Then install the extension if not in development mode
        if not self.dry_run and not self.develop:
            self.execute(self.install_jupyter_extension, [], msg="Installing JupyterLab extension")
    
    def install_jupyter_extension(self):
        """Install the SHM JupyterLab extension after package installation."""
        print("\n" + "="*70)
        print("🎆 SHMTools JupyterLab Extension Auto-Installation")
        print("="*70)
        
        try:
            # Get the extension directory path
            extension_dir = os.path.join(this_directory, 'shm_function_selector')
            
            if not os.path.exists(extension_dir):
                print(f"⚠️  Extension directory not found: {extension_dir}")
                self._print_manual_instructions()
                return
            
            # Check prerequisites
            if not self._check_prerequisites():
                self._print_manual_instructions()
                return
                
            # Change to extension directory and install
            old_cwd = os.getcwd()
            try:
                os.chdir(extension_dir)
                
                # Install npm dependencies
                print("📦 Installing npm dependencies...")
                result = subprocess.run(['npm', 'install'], capture_output=True, text=True, timeout=120)
                if result.returncode != 0:
                    print(f"⚠️  npm install failed: {result.stderr}")
                    self._print_manual_instructions()
                    return
                print("✅ npm dependencies installed")
                
                # Build TypeScript
                print("🔨 Building TypeScript...")
                result = subprocess.run(['npm', 'run', 'build:lib'], capture_output=True, text=True, timeout=60)
                if result.returncode != 0:
                    print(f"⚠️  TypeScript build failed: {result.stderr}")
                    self._print_manual_instructions()
                    return
                print("✅ TypeScript compiled")
                
                # Build labextension
                print("🔧 Building JupyterLab extension...")
                result = subprocess.run(['npm', 'run', 'build:labextension:dev'], capture_output=True, text=True, timeout=60)
                if result.returncode != 0:
                    print(f"⚠️  Extension build failed: {result.stderr}")
                    self._print_manual_instructions()
                    return
                print("✅ Extension built")
                
                # Install the extension package
                print("📦 Installing extension Python package...")
                result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    print(f"⚠️  Extension package install failed: {result.stderr}")
                    self._print_manual_instructions()
                    return
                print("✅ Extension Python package installed")
                
                # Build JupyterLab
                print("🔨 Building JupyterLab (this may take 2-3 minutes)...")
                result = subprocess.run(['jupyter', 'lab', 'build'], 
                                      capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    print(f"⚠️  JupyterLab build failed: {result.stderr}")
                    print("\nTrying alternative build command...")
                    # Try with development build flag
                    result = subprocess.run(['jupyter', 'lab', 'build', '--dev-build=False'], 
                                          capture_output=True, text=True, timeout=300)
                    if result.returncode != 0:
                        self._print_manual_instructions()
                        return
                print("✅ JupyterLab built successfully")
                
            finally:
                os.chdir(old_cwd)
            
            print("\n" + "="*70)
            print("🎉 SHMTools JupyterLab Extension Installed Successfully!")
            print("="*70)
            print("🚀 Quick Start:")
            print("  1. jupyter lab")
            print("  2. Look for the '🔍 SHM Functions' icon in the left sidebar")
            print("  3. Open a Python notebook and start exploring!")
            print("="*70)
                
        except subprocess.TimeoutExpired:
            print("⚠️  Build process timed out")
            self._print_manual_instructions()
        except FileNotFoundError as e:
            print(f"⚠️  Required tool not found: {e}")
            self._print_manual_instructions()
        except Exception as e:
            print(f"⚠️  Installation failed: {e}")
            self._print_manual_instructions()
    
    def _check_prerequisites(self):
        """Check if required tools are available."""
        required_tools = [('jupyter', 'JupyterLab'), ('npm', 'Node.js/npm'), ('node', 'Node.js')]
        missing_tools = []
        
        for tool, description in required_tools:
            try:
                subprocess.run([tool, '--version'], capture_output=True, check=True, timeout=10)
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                missing_tools.append(description)
        
        if missing_tools:
            print(f"⚠️  Missing required tools: {', '.join(missing_tools)}")
            print("\nPlease install the missing tools and try again.")
            return False
        
        print("✅ All prerequisites found")
        return True
    
    def _print_manual_instructions(self):
        """Print manual installation instructions."""
        print("\n" + "-"*70)
        print("🛠️  Manual Installation Required")
        print("-"*70)
        print("To install the jFUSE extension manually:")
        print("")
        print("  cd shm_function_selector/")
        print("  npm install")
        print("  npm run build:lib")
        print("  npm run build:labextension:dev")
        print("  cd .. && jupyter lab build")
        print("")
        print("Or use the convenience script:")
        print("  ./restart_jupyterlab.sh")
        print("-"*70)


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
            'install-jfuse=shm_function_selector.jupyter_extension_installer:install_extension',
            'uninstall-jfuse=shm_function_selector.jupyter_extension_installer:uninstall_extension',
        ],
    },
    cmdclass={
        'install': PostInstallCommand,
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