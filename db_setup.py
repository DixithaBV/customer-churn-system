"""
db_setup.py
-----------
Run this ONCE to:
  1. Read the Telco CSV
  2. Create churn.db (SQLite) with 3 tables
  3. Load all data into those tables

Usage:
    python db_setup.py
"""

import sqlite3
import pandas as pd
import numpy as np
import os

DB_PATH  = "churn.db"
CSV_PATH = "data/Telco-Customer-Churn.csv"

# ── helpers ───────────────────────────────────────────────────────────────────

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    return df


def make_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate a transactions table from billing columns."""
    rows = []
    np.random.seed(42)
    for _, row in df.iterrows():
        months = max(int(row["tenure"]), 1)
        for m in range(1, months + 1):
            rows.append({
                "customerID":     row["customerID"],
                "month_num":      m,
                "amount":         round(float(row["MonthlyCharges"]) * np.random.uniform(0.95, 1.05), 2),
                "payment_method": row["PaymentMethod"],
                "paid_on_time":   np.random.choice([1, 0], p=[0.92, 0.08]),
            })
    return pd.DataFrame(rows)


def make_support_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate support tickets — churned customers raise more tickets."""
    rows = []
    np.random.seed(7)
    categories = ["Billing", "Technical", "Service", "General", "Cancellation"]
    for _, row in df.iterrows():
        n_tickets = np.random.poisson(3 if row["Churn"] == 1 else 1)
        for _ in range(n_tickets):
            rows.append({
                "customerID": row["customerID"],
                "category":   np.random.choice(categories),
                "resolved":   np.random.choice([1, 0], p=[0.80, 0.20]),
                "rating":     np.random.randint(1, 6),
            })
    return pd.DataFrame(rows)


def make_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-aggregate segment summary — queried by the dashboard."""
    return (
        df.groupby(["Contract", "InternetService"])
        .agg(
            total_customers=("customerID", "count"),
            churned=("Churn", "sum"),
            avg_monthly=("MonthlyCharges", "mean"),
            avg_tenure=("tenure", "mean"),
        )
        .round(2)
        .reset_index()
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV not found at '{CSV_PATH}'.\n"
            "Place Telco-Customer-Churn.csv inside the data/ folder."
        )

    print("📂 Reading CSV …")
    df = clean_df(pd.read_csv(CSV_PATH))
    print(f"   {len(df):,} customers loaded")

    print("🏗  Building extra tables …")
    df_tx   = make_transactions(df)
    df_tick = make_support_tickets(df)
    df_seg  = make_segment_summary(df)

    print(f"   transactions    : {len(df_tx):,} rows")
    print(f"   support_tickets : {len(df_tick):,} rows")
    print(f"   segment_summary : {len(df_seg):,} rows")

    print(f"💾 Writing to {DB_PATH} …")
    con = sqlite3.connect(DB_PATH)

    df.to_sql("customers",        con, if_exists="replace", index=False)
    df_tx.to_sql("transactions",  con, if_exists="replace", index=False)
    df_tick.to_sql("support_tickets", con, if_exists="replace", index=False)
    df_seg.to_sql("segment_summary",  con, if_exists="replace", index=False)

    # ── indexes for faster queries ─────────────────────────────────────────
    cur = con.cursor()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_cust_churn   ON customers(Churn)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_cust_contract ON customers(Contract)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tx_cust      ON transactions(customerID)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tick_cust    ON support_tickets(customerID)")
    con.commit()
    con.close()

    print("✅ churn.db created successfully with 4 tables!")
    print("   → customers | transactions | support_tickets | segment_summary")


if __name__ == "__main__":
    main()