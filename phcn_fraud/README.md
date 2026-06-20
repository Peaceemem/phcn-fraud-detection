# PHCN Fraud Detection System

## Setup & Run (3 steps)

### Step 1 — Install dependencies
pip install -r requirements.txt

### Step 2 — Train the ML models (do this once)
python ml/train_models.py

### Step 3 — Run the web server
python manage.py runserver

### Open in browser
http://127.0.0.1:8000

Login: admin / admin123

### Load demo data
Click "Load Demo Data" in the sidebar — creates 50 customers + 150 transactions automatically.

---
## Project Structure
phcn_fraud/
├── ml/
│   ├── train_models.py     ← Step 1: generates data + trains models
│   ├── rf_model.pkl        ← trained Random Forest
│   ├── dt_model.pkl        ← trained Decision Tree
│   ├── iso_model.pkl       ← trained Isolation Forest
│   └── scaler.pkl          ← MinMaxScaler
├── data/
│   └── transactions.csv    ← synthetic training dataset
├── fraud_app/
│   ├── models.py           ← database tables
│   ├── views.py            ← page logic
│   ├── predictor.py        ← ML scoring engine
│   ├── admin.py            ← Django admin config
│   └── templates/          ← HTML pages
├── config/
│   ├── settings.py         ← Django configuration
│   └── urls.py             ← URL routing
├── requirements.txt
└── manage.py
