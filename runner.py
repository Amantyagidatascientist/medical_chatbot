import subprocess
import sys
import time
import requests
from threading import Thread

def run_backend():
    print("🚀 Starting backend on port 8001...")
    subprocess.Popen(
        [sys.executable, "main.py"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )

def run_frontend():
    """Start Streamlit frontend after backend is ready"""
    print("⏳ Waiting for backend to be ready...")
    for _ in range(30):  # 30 retries with 1s interval
        try:
            response = requests.get("http://localhost:8001/health", timeout=1)
            if response.status_code == 200:
                print("✅ Backend is ready!")
                print("🌐 Launching frontend on port 8501...")
                subprocess.Popen(
                    [sys.executable, "-m", "streamlit", "run", "streamlit_frontend.py", "--server.port=8501"],
                    stdout=sys.stdout,
                    stderr=sys.stderr
                )
                return
        except:
            time.sleep(1)
    print("❌ Backend failed to start within 30 seconds")

if __name__ == "__main__":
    # Start backend in a separate thread
    Thread(target=run_backend).start()
    
    # Start frontend after checking backend
    run_frontend()