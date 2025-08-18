#!/usr/bin/env python3
"""
Simple HTTP server to serve the NeuralQuest progress dashboard.
Fixes CORS issues with loading CSV metrics data.
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

PORT = 8080

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def main():
    # Change to the NeuralQuest directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print("NeuralQuest Progress Dashboard Server")
    print(f"Serving from: {project_dir}")
    print(f"Server URL: http://localhost:{PORT}")
    print(f"Dashboard: http://localhost:{PORT}/progress_viewer.html")
    print(f"Metrics: http://localhost:{PORT}/pokemon_vector_logs/vector_metrics.csv")
    print("")
    
    # Start the server
    try:
        with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
            print(f"Server started successfully on port {PORT}")
            print(f"Open http://localhost:{PORT}/progress_viewer.html in your browser")
            print(f"Press Ctrl+C to stop the server")
            print("")
            
            # Optionally open browser automatically
            if len(sys.argv) > 1 and sys.argv[1] == "--open":
                webbrowser.open(f"http://localhost:{PORT}/progress_viewer.html")
                print("Dashboard opened in your default browser")
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0)
    except OSError as e:
        if e.errno == 10048:  # Port already in use on Windows
            print(f"Port {PORT} is already in use. Try a different port or close other servers.")
        else:
            print(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()