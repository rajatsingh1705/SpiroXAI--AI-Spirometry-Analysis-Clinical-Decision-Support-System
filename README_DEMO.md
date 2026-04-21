# SpiroXAI — Live Demo Setup Guide

## Quick Start

### 1. Prerequisites

- **Python 3.10+** (with pip)
- **Supabase** account (free tier: [supabase.com](https://supabase.com))
- A modern browser (Chrome, Edge, Firefox)

### 2. Create Supabase Database

1. Go to [supabase.com](https://supabase.com) → Create a new project
2. Open the **SQL Editor** in your project dashboard
3. Copy and paste the contents of `supabase/schema.sql` and click **Run**
4. From **Project Settings → API**, copy your:
   - **Project URL** (e.g. `https://xxxx.supabase.co`)
   - **anon public key** (starts with `eyJ...`)

### 3. Configure Environment

```bash
cd backend
copy .env.example .env
```

Edit `backend/.env` with your Supabase credentials:

```
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### 4. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

> **Note:** PyTorch may take a few minutes to install. If you get memory issues, install CPU-only:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 5. Run the Backend

```bash
# From the project root:
run.bat

# Or manually:
cd backend
python -m uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.
Docs at `http://localhost:8000/docs`.

### 6. Open the Frontend

Simply open `frontend/index.html` in your browser. No npm or build tools needed!

> **Tip:** For best results, use a local HTTP server:
> ```bash
> cd frontend
> python -m http.server 3000
> ```
> Then open `http://localhost:3000`

---

## Usage

1. **Fill in demographics:** Age, Sex, Weight, Height (BMI auto-calculates)
2. **Enter spirometry values:** PEF, FEF 25-75%, Extrapolated Volume, FET, Acceptable Curves
3. **Optional:** Enter FEV1 and FVC for clinical reference (not used by the model)
4. Click **Run AI Diagnosis**
5. View the prediction, confidence distribution, and SHAP explanations
6. Check the **History** tab for past predictions stored in Supabase

### Sample Data

Use the **Load sample** buttons to quickly fill in Normal, Obstruction, or Restriction test cases.

---

## Architecture

```
┌─────────────┐     POST /predict     ┌──────────────────┐     ┌──────────┐
│  Frontend   │ ───────────────────── │  FastAPI Backend  │ ──→ │ Supabase │
│ (Vanilla JS)│ ←──────────────────── │  (Python)         │ ←── │ (Postgres)│
└─────────────┘   JSON response      └──────────────────┘     └──────────┘
                                             │
                                    ┌────────┴────────┐
                                    │   ML Ensemble   │
                                    │ XGBoost (30%)   │
                                    │ LightGBM (35%)  │
                                    │ FT-Trans (25%)  │
                                    │ DNN (10%)       │
                                    └─────────────────┘
```

## API Endpoints

| Method | Path       | Description                          |
|--------|------------|--------------------------------------|
| GET    | `/health`  | Health check + model status          |
| POST   | `/predict` | Run ensemble prediction              |
| GET    | `/records` | Fetch past predictions from Supabase |

## Files

| File                          | Purpose                              |
|-------------------------------|--------------------------------------|
| `backend/main.py`             | FastAPI app, endpoints, Supabase     |
| `backend/models.py`           | Model architectures, ensemble logic  |
| `backend/feature_engineering.py` | Feature engineering + QT scaling  |
| `backend/validators.py`       | Input validation with clinical ranges|
| `frontend/index.html`         | Main UI                              |
| `frontend/app.js`             | Frontend logic                       |
| `frontend/style.css`          | Styles                               |
| `supabase/schema.sql`         | Database table schema                |
| `saved_models/`               | Trained model weights                |
