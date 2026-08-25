import os
import sys
import webbrowser
import threading
import time

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.api import start_server

PORT = 8899
URL = f"http://127.0.0.1:{PORT}/index.html"

def launch_browser():
    time.sleep(1.2)
    print(f">> Opening CineAI in browser: {URL}")
    webbrowser.open(URL)

if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        try: sys.stdout.reconfigure(encoding='utf-8')
        except Exception: pass

    print("=" * 60)
    print(">> Starting CineAI -- The Cinema Canvas...")
    print("=" * 60)
    
    # Launch browser on background thread
    threading.Thread(target=launch_browser, daemon=True).start()
    
    # Start Backend API server
    start_server(port=PORT)
