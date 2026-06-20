"""
fraud_app/predictor.py
=======================
This module loads the trained ML models and provides a single
function: predict(features_dict) that returns a fraud score.

The Django views call this whenever they need to score a transaction.
"""

import joblib
import numpy as np
from pathlib import Path
from django.conf import settings


# ── Load models once when the server starts ────────────────
# (loading from disk on every request would be very slow)
MODEL_DIR = settings.ML_MODEL_DIR

try:
    RF_MODEL  = joblib.load(MODEL_DIR / "rf_model.pkl")   # Random Forest
    ISO_MODEL = joblib.load(MODEL_DIR / "iso_model.pkl")  # Isolation Forest
    SCALER    = joblib.load(MODEL_DIR / "scaler.pkl")     # MinMaxScaler
    FEATURES  = joblib.load(MODEL_DIR / "features.pkl")   # list of feature names
    MODELS_LOADED = True
    print("[Predictor] Models loaded successfully.")
except Exception as e:
    MODELS_LOADED = False
    print(f"[Predictor] WARNING: Could not load models — {e}")
    print("[Predictor] Run:  python ml/train_models.py  first!")


# ── Fraud threshold ────────────────────────────────────────
# Transactions with combined score above this are flagged
FRAUD_THRESHOLD = 0.50


def predict(feature_dict: dict) -> dict:
    """
    Score a single transaction for fraud.

    Parameters
    ----------
    feature_dict : dict
        Keys must match the FEATURES list. Example:
        {
          "avg_monthly_consumption": 12.5,
          "consumption_variance": 0.8,
          "token_purchase_freq": 14,
          "purchased_to_measured_ratio": 5.2,
          "night_consumption_ratio": 0.85,
          "tamper_flag_count": 7,
          "authorized_vending": 0,
          "zone_fraud_rate": 0.42,
          "consumption_anomaly_score": 3.8,
          "token_reuse_flag": 1,
        }

    Returns
    -------
    dict with keys:
        fraud_probability  : float  0.0 – 1.0
        is_flagged         : bool
        risk_level         : str    "Low" | "Medium" | "High"
        alert_type         : str    "Token" | "Tamper" | "Vending" | "General"
        top_reason         : str    plain-English reason for the flag
    """
    if not MODELS_LOADED:
        return {
            "fraud_probability": 0.0,
            "is_flagged": False,
            "risk_level": "Unknown",
            "alert_type": "General",
            "top_reason": "Models not loaded — run train_models.py first",
        }

    # Build feature vector in the correct order
    values = [[feature_dict.get(f, 0) for f in FEATURES]]

    # Scale using the same scaler used during training
    values_scaled = SCALER.transform(values)

    # ── Random Forest score (supervised) ──────────────────
    rf_proba = RF_MODEL.predict_proba(values_scaled)[0][1]  # probability of fraud

    # ── Isolation Forest score (unsupervised) ─────────────
    # decision_function returns negative values for anomalies
    # We flip the sign so higher = more suspicious
    iso_score_raw = -ISO_MODEL.decision_function(values_scaled)[0]
    # Normalise to 0-1 range (rough approximation)
    iso_proba = min(max((iso_score_raw + 0.5) / 1.0, 0), 1)

    # ── Combined score (RF weighted higher) ───────────────
    combined = 0.70 * rf_proba + 0.30 * iso_proba

    # ── Risk level ────────────────────────────────────────
    if combined >= 0.75:
        risk_level = "High"
    elif combined >= FRAUD_THRESHOLD:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    is_flagged = combined >= FRAUD_THRESHOLD

    # ── Alert type — based on which feature is most suspicious ──
    alert_type = _classify_alert_type(feature_dict)

    # ── Human-readable top reason ─────────────────────────
    top_reason = _top_reason(feature_dict, combined)

    return {
        "fraud_probability": round(combined, 4),
        "is_flagged":        is_flagged,
        "risk_level":        risk_level,
        "alert_type":        alert_type,
        "top_reason":        top_reason,
    }


def _classify_alert_type(f: dict) -> str:
    """Determine the most likely fraud type from feature values."""
    if f.get("token_reuse_flag", 0) == 1:
        return "Token"
    if f.get("tamper_flag_count", 0) >= 3:
        return "Tamper"
    if f.get("authorized_vending", 1) == 0:
        return "Vending"
    return "General"


def _top_reason(f: dict, score: float) -> str:
    """Return a plain-English reason for the fraud flag."""
    reasons = []
    if f.get("purchased_to_measured_ratio", 1) > 2.5:
        reasons.append("Purchased units far exceed measured consumption (possible bypass)")
    if f.get("tamper_flag_count", 0) >= 3:
        reasons.append(f"High tamper flag count ({int(f['tamper_flag_count'])} flags)")
    if f.get("token_reuse_flag", 0) == 1:
        reasons.append("Token ID has been used more than once")
    if f.get("authorized_vending", 1) == 0:
        reasons.append("Token purchased from unauthorized vending point")
    if f.get("zone_fraud_rate", 0) > 0.25:
        reasons.append(f"High-fraud zone (zone rate: {f['zone_fraud_rate']*100:.0f}%)")
    if f.get("consumption_anomaly_score", 0) > 2.5:
        reasons.append("Consumption is a strong statistical outlier")
    if not reasons:
        reasons.append(f"Combined anomaly score ({score:.2f}) exceeds threshold")
    return "; ".join(reasons)
