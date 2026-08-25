import os
import sys
import webbrowser
import threading
import time

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.api import start_server

PORT = int(os.environ.get('PORT', 8899))
HOST = os.environ.get('HOST', '0.0.0.0')
URL = f"http://localhost:{PORT}/index.html"

def launch_browser():
    time.sleep(1.2)
    print(f">> Opening MBMR in browser: {URL}")
    try:
        webbrowser.open(URL)
    except Exception: pass

if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        try: sys.stdout.reconfigure(encoding='utf-8')
        except Exception: pass

    print("=" * 60)
    print(f">> Starting MBMR (Mood-Based Movie Recommender) on port {PORT}...")
    print("=" * 60)
    
    # Launch browser only on local interactive environments
    if not os.environ.get('RENDER') and not os.environ.get('CI'):
        threading.Thread(target=launch_browser, daemon=True).start()
    
    # Start Backend API server
    start_server(host=HOST, port=PORT)
