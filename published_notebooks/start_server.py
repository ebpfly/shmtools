#!/usr/bin/env python3
"""
Simple HTTP server to serve the published notebooks with proper CORS headers.

This script starts a local web server that allows the master.html file to load
individual notebook HTML files without running into CORS restrictions.
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with CORS headers to allow local file loading."""
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def main():
    """Start the server and open the master documentation."""
    
    # Change to the directory containing this script (published_notebooks)
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    port = 8000
    
    # Find an available port
    for try_port in range(8000, 8010):
        try:
            with socketserver.TCPServer(("", try_port), CORSHTTPRequestHandler) as httpd:
                port = try_port
                break
        except OSError:
            continue
    else:
        print("❌ Could not find an available port between 8000-8009")
        return 1
    
    print("SHMTools Published Notebooks Server")
    print("=" * 40)
    print(f"📁 Serving from: {script_dir}")
    print(f"🌐 Local server: http://localhost:{port}")
    print(f"📖 Master documentation: http://localhost:{port}/master.html")
    print(f"📋 Simple index: http://localhost:{port}/index.html")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 40)
    
    # Start the server
    try:
        with socketserver.TCPServer(("", port), CORSHTTPRequestHandler) as httpd:
            # Open the master documentation in the default browser
            master_url = f"http://localhost:{port}/master.html"
            print(f"🚀 Opening {master_url} in your browser...")
            webbrowser.open(master_url)
            
            # Serve indefinitely
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        return 0
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())