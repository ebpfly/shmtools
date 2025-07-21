#!/usr/bin/env python3
"""
Start Jupyter notebook with the SHM extension backend running.
This allows you to test the backend from within a Jupyter notebook.
"""

import subprocess
import sys
import os
import time
from threading import Thread

def start_extension_server():
    """Start the extension backend server."""
    print("🚀 Starting SHM extension backend server...")
    
    # Add extension to path
    sys.path.insert(0, 'jupyter_shm_extension')
    
    # Import and run the simple server
    from test_backend_simple import run_server_in_thread
    
    # Start server in background
    server_thread = Thread(target=run_server_in_thread, daemon=True)
    server_thread.start()
    
    time.sleep(2)
    print("✅ Extension backend running on http://localhost:8890")
    return server_thread

def start_jupyter():
    """Start Jupyter notebook."""
    print("📓 Starting Jupyter notebook...")
    
    # Activate virtual environment and start Jupyter
    cmd = [
        'bash', '-c',
        'source venv/bin/activate && jupyter notebook --ip=localhost --port=8888'
    ]
    
    return subprocess.Popen(cmd)

if __name__ == '__main__':
    print("🧪 JUPYTER + SHM EXTENSION TESTING")
    print("=" * 50)
    
    try:
        # Start extension backend
        server_thread = start_extension_server()
        
        # Start Jupyter
        jupyter_proc = start_jupyter()
        
        print("\n✅ Both servers started!")
        print("   🌐 Extension backend: http://localhost:8890")
        print("   📓 Jupyter notebook: http://localhost:8888")
        print("\nOpen the notebook and test the extension endpoints!")
        print("Press Ctrl+C to stop both servers...")
        
        # Keep running until interrupted
        jupyter_proc.wait()
        
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down servers...")
        if 'jupyter_proc' in locals():
            jupyter_proc.terminate()
    except Exception as e:
        print(f"❌ Error: {e}")