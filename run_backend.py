"""
GenMed AI — Backend Launcher
Run this file from the project root:

    python run_backend.py

OR use uvicorn directly from the project root:

    uvicorn backend.main:app --reload --port 8000
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
