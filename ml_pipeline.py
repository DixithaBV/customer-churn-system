"""
ml_pipeline.py
--------------
Run this AFTER db_setup.py to:
  1. Load data from churn.db
  2. Feature-engineer + train XGBoost
  3. Evaluate (accuracy, ROC-AUC, classification report)
  4. Compute SHAP values
  5. Save model/  → model.pkl, scaler.pkl, encoder_map.pkl, shap_values.pkl

Usage:
    python ml_pipeline.py
"""

import sqlite3, os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix)
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")

DB_PATH    = "churn.db"
MODEL_DIR  = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── features ──────────────────────────────────────────────────────────────────
CAT_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService",
    "InternetService", "Contract", "PaperlessBilling", "PaymentMethod",
]
NUM_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
ALL_FEATURES  = NUM_FEATURES + CAT_FEATURES


# ── load ──────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql("SELECT * FROM customers", con)
    con.close()
    return df


# ── encode ────────────────────────────────────────────────────────────────────
def encode(df: pd.DataFrame):
    df = df[ALL_FEATURES + ["Churn"]].copy()
    encoder_map = {}
    for col in CAT_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoder_map[col] = le
    return df, encoder_map


# ── train ─────────────────────────────────────────────────────────────────────
def train(df_enc: pd.DataFrame, encoder_map: dict):
    X = df_enc[ALL_FEATURES]
    y = df_enc["Churn"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    return model, scaler, X_train, X_test, y_train, y_test


# ── evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)
    print(f"\n📊 Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"   ROC-AUC  : {auc:.4f}")
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
    print("   Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return acc, auc


# ── shap ──────────────────────────────────────────────────────────────────────
def compute_shap(model, X_train):
    print("\n🔍 Computing SHAP values (this takes ~30 seconds) …")
    sample = X_train[:500]
    try:
        # Try TreeExplainer first
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
    except Exception:
        # Fall back to feature importances if SHAP version conflicts
        print("   (Using XGBoost feature importances as fallback)")
        imp_raw     = model.feature_importances_
        shap_values = np.tile(imp_raw, (sample.shape[0], 1))
    mean_abs   = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=ALL_FEATURES).sort_values(ascending=False)
    print("\n   Top 5 churn factors (SHAP):")
    for feat, val in importance.head(5).items():
        print(f"   {feat:<25} {val:.4f}")
    return shap_values, importance


# ── save ──────────────────────────────────────────────────────────────────────
def save_artifacts(model, scaler, encoder_map, shap_values, shap_importance):
    pickle.dump(model,          open(f"{MODEL_DIR}/model.pkl",       "wb"))
    pickle.dump(scaler,         open(f"{MODEL_DIR}/scaler.pkl",      "wb"))
    pickle.dump(encoder_map,    open(f"{MODEL_DIR}/encoder_map.pkl", "wb"))
    pickle.dump(shap_values,    open(f"{MODEL_DIR}/shap_values.pkl", "wb"))
    shap_importance.to_csv(f"{MODEL_DIR}/shap_importance.csv")
    print(f"\n✅ All artifacts saved to {MODEL_DIR}/")
    print("   model.pkl | scaler.pkl | encoder_map.pkl | shap_values.pkl | shap_importance.csv")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("📂 Loading data from churn.db …")
    df = load_data()
    print(f"   {len(df):,} rows loaded")

    print("⚙️  Encoding features …")
    df_enc, encoder_map = encode(df)

    print("🤖 Training XGBoost model …")
    model, scaler, X_train, X_test, y_train, y_test = train(df_enc, encoder_map)

    evaluate(model, X_test, y_test)

    shap_values, shap_importance = compute_shap(model, X_train)

    save_artifacts(model, scaler, encoder_map, shap_values, shap_importance)


if __name__ == "__main__":
    main()