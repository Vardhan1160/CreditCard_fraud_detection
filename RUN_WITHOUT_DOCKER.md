# Running Without Docker - Simple Guide

## ✅ Step-by-Step Instructions (No Docker Required)

### Step 1: Create Virtual Environment

```powershell
python -m venv venv
```

### Step 2: Install Dependencies

**Option A: Using venv Python directly (Recommended)**
```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Option B: Activate venv first (if execution policy allows)**
```powershell
# If you get execution policy error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate:
.\venv\Scripts\Activate.ps1

# Then install:
pip install -r requirements.txt
```

### Step 3: Generate Sample Data

```powershell
.\venv\Scripts\python.exe scripts\sample_data.py
```

Or if venv is activated:
```powershell
python scripts\sample_data.py
```

### Step 4: Train the Model

```powershell
.\venv\Scripts\python.exe scripts\train_model.py
```

This will:
- Load/prepare data
- Train multiple models
- Save best model to `models\best_model.pkl`
- Generate evaluation reports

**Wait for training to complete** (5-15 minutes depending on your system)

### Step 5: Start the API Server

Open a **NEW PowerShell window** and navigate to the project folder, then:

```powershell
.\venv\Scripts\python.exe -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Or if venv is activated:
```powershell
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

**API will be available at:**
- http://localhost:8000
- Interactive docs: http://localhost:8000/docs

### Step 6: Run Streamlit Web App (Optional)

Open **ANOTHER PowerShell window** and navigate to the project folder, then:

```powershell
.\venv\Scripts\python.exe -m streamlit run webapp\app_streamlit.py
```

Or if venv is activated:
```powershell
streamlit run webapp\app_streamlit.py
```

**Web app will open at:** http://localhost:8501

---

## 📋 Quick Command Reference

| Task | Command (using venv Python directly) |
|------|--------------------------------------|
| Install dependencies | `.\venv\Scripts\python.exe -m pip install -r requirements.txt` |
| Generate data | `.\venv\Scripts\python.exe scripts\sample_data.py` |
| Train model | `.\venv\Scripts\python.exe scripts\train_model.py` |
| Start API | `.\venv\Scripts\python.exe -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000` |
| Start Streamlit | `.\venv\Scripts\python.exe -m streamlit run webapp\app_streamlit.py` |
| Run tests | `.\venv\Scripts\python.exe -m pytest tests\ -v` |

---

## 🔍 Verify Installation

Check if packages are installed:

```powershell
.\venv\Scripts\python.exe -m pip list
```

You should see: pandas, numpy, scikit-learn, xgboost, fastapi, streamlit, etc.

---

## ⚠️ Troubleshooting

### Issue: "No module named 'xxx'"

**Solution:** Make sure you're using the venv Python:
```powershell
.\venv\Scripts\python.exe -m pip install <package_name>
```

### Issue: "Model not found" when starting API

**Solution:** Train the model first:
```powershell
.\venv\Scripts\python.exe scripts\train_model.py
```

### Issue: Port already in use

**Solution:** Use a different port:
```powershell
# For API
.\venv\Scripts\python.exe -m uvicorn src.api.app:app --host 0.0.0.0 --port 8001

# For Streamlit
.\venv\Scripts\python.exe -m streamlit run webapp\app_streamlit.py --server.port 8502
```

---

## 🎯 Minimum Steps to Get Started

1. `python -m venv venv`
2. `.\venv\Scripts\python.exe -m pip install -r requirements.txt`
3. `.\venv\Scripts\python.exe scripts\sample_data.py`
4. `.\venv\Scripts\python.exe scripts\train_model.py` (wait for completion)
5. `.\venv\Scripts\python.exe -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000`

Then visit http://localhost:8000/docs to test the API!

---

**Note:** You don't need Docker at all. All commands above use Python directly from your virtual environment.

