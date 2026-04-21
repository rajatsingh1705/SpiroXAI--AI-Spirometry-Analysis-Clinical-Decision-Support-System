# 🫁 SpiroXAI – Full-Stack AI Spirometry Analysis & Clinical Decision Support System

SpiroXAI is a full-stack AI-powered web application designed for analyzing spirometry data and detecting lung diseases such as **Obstruction**, **Restriction**, and **Normal patterns**.

The system integrates modern machine learning models with an intuitive web interface and explainable AI to support clinical decision-making.

---

## 🚀 Features

- 🔍 AI-based lung disease prediction (Normal / Obstruction / Restriction)
- 📊 Confidence score visualization
- 🧠 Explainable AI using SHAP (feature importance & impact)
- 🗂️ Prediction history tracking
- ⚡ FastAPI-based high-performance backend
- 🌐 Clean and responsive frontend (Vanilla JS)
- ☁️ Cloud database integration using Supabase

---

## 🏗️ System Architecture


Frontend (HTML, CSS, JavaScript)
↓
FastAPI Backend (Python)
↓
Machine Learning Models
(FT-Transformer, XGBoost, LightGBM, DNN)
↓
SHAP Explainability Layer
↓
Supabase (PostgreSQL + JSONB Storage)


---

## 🧠 Machine Learning Models

- **FT-Transformer** (Primary Model)
- XGBoost
- LightGBM
- Deep Neural Network (DNN)

### Tasks:
- Lung disease classification
- Feature interaction modeling
- Explainability using SHAP

---

## 🖥️ Tech Stack

### Backend
- Python
- FastAPI
- Uvicorn
- Pydantic

### Frontend
- HTML5
- CSS3
- Vanilla JavaScript

### Database & Cloud
- Supabase
- PostgreSQL
- JSONB (for storing SHAP outputs)

### Machine Learning
- PyTorch / TensorFlow (for FT-Transformer)
- Scikit-learn
- XGBoost
- LightGBM
- SHAP

---

## 🔄 Workflow

1. User inputs patient and spirometry data
2. Frontend sends request to FastAPI backend
3. Backend:
   - Validates input using Pydantic
   - Runs ML models
   - Generates SHAP explanations
4. Response sent back to frontend
5. Results displayed (prediction + confidence + explanation)
6. Data stored in Supabase database
7. Users can view past predictions in History section

---

## 📊 Example Output

- Prediction: **Obstruction**
- Confidence: **82%**
- SHAP Explanation:
  - FEV1 ↓ → increases risk
  - Expiratory Time ↑ → increases risk
  - Age → moderate influence

---

## 📂 Project Structure


spiroxai/
│
├── backend/
│ ├── main.py
│ ├── models/
│ ├── schemas/
│ ├── routes/
│ └── utils/
│
├── frontend/
│ ├── index.html
│ ├── styles.css
│ ├── script.js
│
├── database/
│ └── supabase_config.sql
│
├── models/
│ ├── ft_transformer.pkl
│ ├── xgboost.pkl
│ └── lightgbm.pkl
│
└── README.md


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/spiroxai.git
cd spiroxai
2️⃣ Backend Setup
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
3️⃣ Frontend

Open directly in browser:

frontend/index.html
4️⃣ Database (Supabase)
Create a project on Supabase
Set up PostgreSQL tables
Add API keys in backend configuration
🔐 Environment Variables

Create a .env file inside backend:

SUPABASE_URL=your_url
SUPABASE_KEY=your_key
🎯 Use Cases
Clinical decision support systems
Healthcare AI applications
Spirometry data analysis tools
Academic and research projects
⚠️ Disclaimer

This project is for educational and research purposes only and is not intended for real clinical use.

👨‍💻 Author

Rajat Singh
Computer Science Engineering Student

⭐ Future Improvements
Deploy on cloud (AWS / Vercel / Render)
Add authentication system
Real-time spirometry graph upload
Mobile responsiveness improvements
Integration with hospital systems