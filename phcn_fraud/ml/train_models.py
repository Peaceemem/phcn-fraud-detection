"""
============================================================
PHCN FRAUD DETECTION SYSTEM
Step 1: Data Generation + Model Training
============================================================

This script does THREE things:
  1. Generates a realistic synthetic dataset of PHCN prepaid
     electricity transactions (since real data is hard to get)
  2. Preprocesses the data and engineers fraud-predictive features
  3. Trains Random Forest, Decision Tree, and Isolation Forest
     models and saves them as .pkl files for the web app to use

RUN THIS FIRST before starting the Django server.
Command: python ml/train_models.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)
import joblib
import os

# ── where to save models ──────────────────────────────────
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────
# PART 1: GENERATE SYNTHETIC DATASET
# ─────────────────────────────────────────────────────────
print("=" * 55)
print("  PHCN Fraud Detection — Model Training")
print("=" * 55)
print("\n[1/4] Generating synthetic transaction dataset...")

np.random.seed(42)
N_LEGIT  = 4750   # legitimate transactions
N_FRAUD  = 250    # fraudulent transactions (5% — realistic ratio)
N_TOTAL  = N_LEGIT + N_FRAUD

# ── LEGITIMATE customer features ──────────────────────────
legit = {
    # Average monthly consumption between 30 and 300 kWh
    "avg_monthly_consumption": np.random.uniform(30, 300, N_LEGIT),

    # Variance of consumption — legitimate customers vary normally
    "consumption_variance": np.random.uniform(2, 40, N_LEGIT),

    # Token purchases per month — 1 to 6 times
    "token_purchase_freq": np.random.randint(1, 6, N_LEGIT).astype(float),

    # Ratio purchased / measured ≈ 1.0 for honest customers
    "purchased_to_measured_ratio": np.random.uniform(0.85, 1.15, N_LEGIT),

    # Night consumption ratio — 10% to 40% is normal
    "night_consumption_ratio": np.random.uniform(0.10, 0.40, N_LEGIT),

    # Tamper flag count — rarely triggers for honest customers
    "tamper_flag_count": np.random.choice([0, 1], N_LEGIT, p=[0.97, 0.03]),

    # Authorized vending point — almost always yes
    "authorized_vending": np.random.choice([1, 0], N_LEGIT, p=[0.98, 0.02]),

    # Zone fraud rate — background rate 3-10%
    "zone_fraud_rate": np.random.uniform(0.03, 0.10, N_LEGIT),

    # Consumption z-score — close to 0 for normal customers
    "consumption_anomaly_score": np.random.normal(0, 0.8, N_LEGIT),

    # Token reuse — almost never for legitimate transactions
    "token_reuse_flag": np.random.choice([0, 1], N_LEGIT, p=[0.99, 0.01]),

    # Label: 0 = legitimate
    "is_fraud": np.zeros(N_LEGIT, dtype=int),
}

# ── FRAUDULENT customer features ──────────────────────────
fraud = {
    # Very LOW consumption despite purchasing tokens (bypass / tamper)
    "avg_monthly_consumption": np.random.uniform(2, 25, N_FRAUD),

    # Variance is often very low (meter stuck) or very high (erratic bypass)
    "consumption_variance": np.concatenate([
        np.random.uniform(0, 2, N_FRAUD // 2),       # stuck meter
        np.random.uniform(60, 120, N_FRAUD // 2),     # erratic bypass
    ]),

    # High purchase frequency (token duplication / resale)
    "token_purchase_freq": np.random.randint(8, 20, N_FRAUD).astype(float),

    # Ratio >> 1 means more purchased than measured (meter bypass)
    "purchased_to_measured_ratio": np.random.uniform(2.5, 8.0, N_FRAUD),

    # Night consumption is anomalously high or near zero
    "night_consumption_ratio": np.concatenate([
        np.random.uniform(0.70, 0.99, N_FRAUD // 2),  # heavy night user
        np.random.uniform(0.00, 0.02, N_FRAUD // 2),  # zero night — suspicious
    ]),

    # Tamper flags fire frequently for fraud meters
    "tamper_flag_count": np.random.randint(3, 15, N_FRAUD).astype(float),

    # Often purchased from unauthorized vending points
    "authorized_vending": np.random.choice([1, 0], N_FRAUD, p=[0.40, 0.60]),

    # High fraud zone
    "zone_fraud_rate": np.random.uniform(0.25, 0.60, N_FRAUD),

    # Strong anomaly score
    "consumption_anomaly_score": np.random.uniform(2.5, 6.0, N_FRAUD),

    # Token reuse happens frequently in fraud
    "token_reuse_flag": np.random.choice([0, 1], N_FRAUD, p=[0.30, 0.70]),

    # Label: 1 = fraud
    "is_fraud": np.ones(N_FRAUD, dtype=int),
}

# ── Combine into a single DataFrame ───────────────────────
df_legit = pd.DataFrame(legit)
df_fraud = pd.DataFrame(fraud)
df = pd.concat([df_legit, df_fraud], ignore_index=True)

# Shuffle the rows so fraud is not all at the bottom
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Add some realistic non-feature columns
df.insert(0, "transaction_id", range(1, N_TOTAL + 1))
df.insert(1, "meter_number", [f"MTR{str(i).zfill(6)}" for i in range(1, N_TOTAL + 1)])
df.insert(2, "customer_category", np.random.choice(
    ["Residential", "Commercial", "Industrial"], N_TOTAL, p=[0.70, 0.25, 0.05]
))

print(f"   Generated {N_TOTAL} transactions "
      f"({N_LEGIT} legitimate, {N_FRAUD} fraudulent)")
print(f"   Fraud rate: {N_FRAUD/N_TOTAL*100:.1f}%")

# Save raw dataset for reference
df.to_csv(os.path.join(MODEL_DIR, "../data/transactions.csv"), index=False)
print("   Saved: data/transactions.csv")

# ─────────────────────────────────────────────────────────
# PART 2: PREPROCESSING
# ─────────────────────────────────────────────────────────
print("\n[2/4] Preprocessing data...")

# The 10 features the model will use for prediction
FEATURES = [
    "avg_monthly_consumption",
    "consumption_variance",
    "token_purchase_freq",
    "purchased_to_measured_ratio",
    "night_consumption_ratio",
    "tamper_flag_count",
    "authorized_vending",
    "zone_fraud_rate",
    "consumption_anomaly_score",
    "token_reuse_flag",
]

X = df[FEATURES]
y = df["is_fraud"]

# Encode customer_category (not used in model but saved for reference)
le = LabelEncoder()
df["customer_category_encoded"] = le.fit_transform(df["customer_category"])

# Scale features to 0-1 range
# This is important for some models and helps with interpretation
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=FEATURES)

# Train/test split — 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)

print(f"   Training set: {len(X_train)} transactions")
print(f"   Test set:     {len(X_test)} transactions")

# ─────────────────────────────────────────────────────────
# PART 3: TRAIN ALL THREE MODELS
# ─────────────────────────────────────────────────────────
print("\n[3/4] Training models...")

results = {}

# ── Model A: Random Forest ─────────────────────────────
print("   Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,   # 100 decision trees in the ensemble
    max_depth=None,     # trees grow until leaves are pure
    random_state=42,
    class_weight="balanced",  # handles class imbalance (few fraud cases)
    n_jobs=-1,          # use all CPU cores
)
rf.fit(X_train, y_train)
rf_pred  = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]  # fraud probability

results["Random Forest"] = {
    "accuracy":  accuracy_score(y_test, rf_pred),
    "precision": precision_score(y_test, rf_pred),
    "recall":    recall_score(y_test, rf_pred),
    "f1":        f1_score(y_test, rf_pred),
    "auc_roc":   roc_auc_score(y_test, rf_proba),
}

# ── Model B: Decision Tree ─────────────────────────────
print("   Training Decision Tree...")
dt = DecisionTreeClassifier(
    max_depth=8,        # limit depth to prevent overfitting
    random_state=42,
    class_weight="balanced",
)
dt.fit(X_train, y_train)
dt_pred  = dt.predict(X_test)
dt_proba = dt.predict_proba(X_test)[:, 1]

results["Decision Tree"] = {
    "accuracy":  accuracy_score(y_test, dt_pred),
    "precision": precision_score(y_test, dt_pred),
    "recall":    recall_score(y_test, dt_pred),
    "f1":        f1_score(y_test, dt_pred),
    "auc_roc":   roc_auc_score(y_test, dt_proba),
}

# ── Model C: Isolation Forest (unsupervised) ───────────
print("   Training Isolation Forest...")
iso = IsolationForest(
    n_estimators=100,
    contamination=0.05,  # we expect ~5% fraud
    random_state=42,
)
iso.fit(X_train)
# Isolation Forest returns -1 (anomaly) or 1 (normal)
# We convert to 1 (fraud) and 0 (legitimate)
iso_pred_raw = iso.predict(X_test)
iso_pred = np.where(iso_pred_raw == -1, 1, 0)
# Anomaly score: more negative = more anomalous
iso_scores = -iso.decision_function(X_test)  # flip sign so higher = more fraudulent

results["Isolation Forest"] = {
    "accuracy":  accuracy_score(y_test, iso_pred),
    "precision": precision_score(y_test, iso_pred),
    "recall":    recall_score(y_test, iso_pred),
    "f1":        f1_score(y_test, iso_pred),
    "auc_roc":   roc_auc_score(y_test, iso_scores),
}

# ── Print comparison table ─────────────────────────────
print("\n" + "=" * 60)
print(f"  {'Model':<20} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
print("  " + "-" * 56)
for name, m in results.items():
    print(f"  {name:<20} {m['accuracy']:>6.3f} {m['precision']:>6.3f} "
          f"{m['recall']:>6.3f} {m['f1']:>6.3f} {m['auc_roc']:>6.3f}")
print("=" * 60)

print("\n  Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\n  Random Forest Feature Importance:")
feat_importance = pd.Series(rf.feature_importances_, index=FEATURES)
for feat, score in feat_importance.sort_values(ascending=False).items():
    bar = "█" * int(score * 40)
    print(f"  {feat:<35} {score:.3f} {bar}")

# ─────────────────────────────────────────────────────────
# PART 4: SAVE MODELS AND PREPROCESSOR
# ─────────────────────────────────────────────────────────
print("\n[4/4] Saving models and preprocessor...")

joblib.dump(rf,      os.path.join(MODEL_DIR, "rf_model.pkl"))
joblib.dump(dt,      os.path.join(MODEL_DIR, "dt_model.pkl"))
joblib.dump(iso,     os.path.join(MODEL_DIR, "iso_model.pkl"))
joblib.dump(scaler,  os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(FEATURES, os.path.join(MODEL_DIR, "features.pkl"))

print("   Saved: ml/rf_model.pkl   (Random Forest)")
print("   Saved: ml/dt_model.pkl   (Decision Tree)")
print("   Saved: ml/iso_model.pkl  (Isolation Forest)")
print("   Saved: ml/scaler.pkl     (MinMaxScaler)")
print("   Saved: ml/features.pkl   (feature list)")
print("\n  All done! Now run:  python manage.py runserver")
print("=" * 55)
