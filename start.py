import subprocess
import sys
import time


def run_services():
    backend = subprocess.Popen([
        sys.executable,
        "-m",
        "uvicorn",
        "backend.app:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000"
    ])

    time.sleep(3)

    frontend = subprocess.Popen([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "frontend/streamlit_app.py",
        "--server.port",
        "8501",
        "--server.address",
        "0.0.0.0"
    ])

    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        backend.terminate()
        frontend.terminate()


if __name__ == "__main__":
    run_services()