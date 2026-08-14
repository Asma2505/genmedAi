"""
GenMed AI — Frontend Launcher
Run this file from the project root:

    python run_frontend.py

OR use streamlit directly from the project root:

    streamlit run frontend/app.py
"""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "frontend/app.py"])
