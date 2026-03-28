"""
backend/main.py
---------------
FastAPI REST API for the Customer Churn Prediction System.

Endpoints:
    GET  /              → health check
    POST /predict       → returns churn probability for one customer
    GET  /insights      → returns top SHAP feature importances
    GET  /stats         → returns overall DB stats for dashboard KPIs

Run locally:
    cd backend
    uvicorn main:app --reload --port 8000
    Then open: http://localhost:8000/docs
"""

import os, pickle, sqlite3
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DB_PATH   = os.path.join(BASE_DIR, "churn.db")

CAT_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService",
    "InternetService", "Contract", "PaperlessBilling", "PaymentMethod",
]
NUM_FEATURES  = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
ALL_FEATURES  = NUM_FEATURES + CAT_FEATURES

# ── load artifacts at startup ─────────────────────────────────────────────────
model       = pickle.load(open(f"{MODEL_DIR}/model.pkl",       "rb"))
scaler      = pickle.load(open(f"{MODEL_DIR}/scaler.pkl",      "rb"))
encoder_map = pickle.load(open(f"{MODEL_DIR}/encoder_map.pkl", "rb"))
shap_imp    = pd.read_csv(f"{MODEL_DIR}/shap_importance.csv", index_col=0)
shap_imp.columns = ["importance"]

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description="XGBoost + SHAP powered churn prediction. Built by Dixitha BV.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── schemas ───────────────────────────────────────────────────────────────────
class CustomerIn(BaseModel):
    gender:           str   = "Male"
    SeniorCitizen:    int   = 0
    Partner:          str   = "No"
    Dependents:       str   = "No"
    tenure:           int   = 12
    PhoneService:     str   = "Yes"
    InternetService:  str   = "Fiber optic"
    Contract:         str   = "Month-to-month"
    PaperlessBilling: str   = "Yes"
    PaymentMethod:    str   = "Electronic check"
    MonthlyCharges:   float = 70.0
    TotalCharges:     float = 840.0


class PredictionOut(BaseModel):
    churn_probability: float
    churn_prediction:  int
    risk_level:        str
    recommendations:   list[str]


# ── helpers ───────────────────────────────────────────────────────────────────
def encode_customer(data: CustomerIn) -> np.ndarray:
    row = {
        "SeniorCitizen":    data.SeniorCitizen,
        "tenure":           data.tenure,
        "MonthlyCharges":   data.MonthlyCharges,
        "TotalCharges":     data.TotalCharges,
        "gender":           data.gender,
        "Partner":          data.Partner,
        "Dependents":       data.Dependents,
        "PhoneService":     data.PhoneService,
        "InternetService":  data.InternetService,
        "Contract":         data.Contract,
        "PaperlessBilling": data.PaperlessBilling,
        "PaymentMethod":    data.PaymentMethod,
    }
    df = pd.DataFrame([row])[ALL_FEATURES]
    for col in CAT_FEATURES:
        le = encoder_map[col]
        val = str(df[col].iloc[0])
        df[col] = le.transform([val])[0] if val in le.classes_ else 0
    return scaler.transform(df)


def get_recommendations(data: CustomerIn, prob: float) -> list[str]:
    recs = []
    if data.Contract == "Month-to-month":
        recs.append("Offer a discounted 1-year or 2-year contract to reduce churn risk.")
    if data.PaymentMethod == "Electronic check":
        recs.append("Encourage switch to auto-pay — it significantly reduces churn.")
    if data.InternetService == "Fiber optic" and prob > 0.5:
        recs.append("Fiber optic churners are price-sensitive — offer a loyalty discount.")
    if data.tenure < 12:
        recs.append("New customer — enroll in an onboarding rewards program immediately.")
    if data.MonthlyCharges > 80:
        recs.append("High spender — offer a bundled plan to improve perceived value.")
    if not recs:
        recs.append("Customer appears stable. Maintain regular engagement touchpoints.")
    return recs


def risk_label(prob: float) -> str:
    if prob >= 0.70: return "HIGH"
    if prob >= 0.40: return "MEDIUM"
    return "LOW"


# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "message": "Churn Prediction API is running ✅"}


@app.post("/predict", response_model=PredictionOut)
def predict(customer: CustomerIn):
    try:
        X    = encode_customer(customer)
        prob = float(model.predict_proba(X)[0][1])
        pred = int(prob >= 0.5)
        return PredictionOut(
            churn_probability = round(prob, 4),
            churn_prediction  = pred,
            risk_level        = risk_label(prob),
            recommendations   = get_recommendations(customer, prob),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/insights")
def insights(top_n: int = 10):
    """Return top SHAP feature importances."""
    result = (
        shap_imp.head(top_n)
        .reset_index()
        .rename(columns={"index": "feature"})
        .to_dict(orient="records")
    )
    return {"top_features": result}


@app.get("/stats")
def stats():
    """Return aggregate KPIs from the SQLite database."""
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()

        cur.execute("SELECT COUNT(*), SUM(Churn), ROUND(AVG(MonthlyCharges),2) FROM customers")
        total, churned, avg_charge = cur.fetchone()

        cur.execute("""
            SELECT Contract, COUNT(*), ROUND(SUM(Churn)*100.0/COUNT(*),2)
            FROM customers GROUP BY Contract ORDER BY 3 DESC
        """)
        by_contract = [{"contract": r[0], "total": r[1], "churn_pct": r[2]}
                       for r in cur.fetchall()]

        cur.execute("""
            SELECT ROUND(SUM(MonthlyCharges),2) FROM customers WHERE Churn=1
        """)
        revenue_at_risk = cur.fetchone()[0]

        con.close()
        return {
            "total_customers":  int(total),
            "churned":          int(churned),
            "retained":         int(total - churned),
            "churn_rate_pct":   round(churned / total * 100, 2),
            "avg_monthly":      avg_charge,
            "revenue_at_risk":  revenue_at_risk,
            "by_contract":      by_contract,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))