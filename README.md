# 🏥 GenMed AI — Project Setup & Documentation

**No External APIs — FastAPI Backend + Streamlit Frontend + JSON Data + Pre-trained ML Model**

---

## ⚠️ IMPORTANT — Run Everything from the Project Root

All commands must be run from the **`genmed_project` folder** — the folder that
directly contains `backend/`, `frontend/`, `data/`, `models/`, `requirements.txt`.

---

## ⚡ Windows Step-by-Step

### Step 1 — Navigate to the correct folder

After extracting the ZIP you will have a nested structure:
```
Downloads\genmed_project\genmed_project\   ← THIS is the project root
```

Open PowerShell and run:
```powershell
cd C:\Users\ASMA.R\Downloads\genmed_project\genmed_project
```

Confirm you are in the right place:
```powershell
dir
# Must show: backend  data  frontend  models  requirements.txt
```

### Step 2 — Install dependencies (once only)
```powershell
pip install -r requirements.txt
```

### Step 3 — Start Backend (Terminal 1)
```powershell
python run_backend.py
```
Leave this window open. You should see:
```
INFO: Uvicorn running on http://127.0.0.1:8000
```

### Step 4 — Start Frontend (New Terminal 2)
```powershell
cd C:\Users\ASMA.R\Downloads\genmed_project\genmed_project
streamlit run frontend/app.py
```
App opens at **http://localhost:8501**

---

## 📁 Project Structure

```
genmed_project/              ← RUN ALL COMMANDS FROM HERE
├── run_backend.py           ← python run_backend.py
├── run_frontend.py          ← python run_frontend.py  (or: streamlit run frontend/app.py)
├── backend/main.py          ← FastAPI REST API (15 endpoints)
├── frontend/app.py          ← Streamlit 5-page UI
├── data/patients.json       ← 1,000 patient records
├── models/model.pkl         ← Pre-trained RandomForest
├── models/train_model.py    ← Retrain if needed
├── requirements.txt
└── README.md
```

---

## 🌐 URLs

| | URL |
|--|--|
| Streamlit App | http://localhost:8501 |
| API Root | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |

---

## 🐛 Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `No module named 'backend'` | You're inside the backend/ folder. `cd ..` first |
| `File does not exist: frontend/app.py` | You're inside the frontend/ folder. `cd ..` first |
| `Cannot connect to backend` | Start backend BEFORE frontend |
| `No module named 'fastapi'` | Run `pip install -r requirements.txt` |
| `model.pkl not found` | Run `python models/train_model.py` from project root |
| Port 8000 in use | `netstat -ano \| findstr :8000` then `taskkill /PID <N> /F` |

---

## ⚠️ Disclaimer

For educational purposes only. Not a clinical tool.
