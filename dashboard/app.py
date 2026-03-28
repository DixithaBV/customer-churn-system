"""
dashboard/app.py
----------------
Streamlit dashboard for the Customer Churn Intelligence System.

Pages:
    🏠 Overview       — KPI cards pulled from /stats API
    🗄️  SQL Explorer  — Run pre-written SQL queries on churn.db
    📊 EDA            — Charts and segment analysis
    🤖 ML Performance — Model metrics, confusion matrix, ROC curve
    🔮 Predict        — Call /predict API, show gauge + SHAP + recommendations

Run locally:
    cd dashboard
    streamlit run app.py
"""

import sys, os, sqlite3, pickle, warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

warnings.filterwarnings("ignore")

# ── path fix so sql_queries.py is importable from parent ─────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sql_queries import QUERIES

# ── config ────────────────────────────────────────────────────────────────────
API_URL  = os.getenv("API_URL", "https://customer-churn-api-fsdd.onrender.com")
DB_PATH  = os.path.join(os.path.dirname(__file__), "..", "churn.db")
MDL_DIR  = os.path.join(os.path.dirname(__file__), "..", "model")

st.set_page_config(
    page_title="Churn Intelligence System",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.main-title { font-family:'Space Mono',monospace; font-size:2.2rem; font-weight:700;
              color:#FF4B4B; letter-spacing:-1px; margin-bottom:0; }
.sub-title  { color:#888; font-size:.95rem; margin-top:0; margin-bottom:1.5rem; }
.kpi-card   { background:linear-gradient(135deg,#1a1a2e,#16213e);
              border:1px solid #0f3460; border-radius:12px;
              padding:1.1rem 1.4rem; text-align:center;
              box-shadow:0 4px 20px rgba(0,0,0,.3); }
.kpi-val    { font-family:'Space Mono',monospace; font-size:1.9rem;
              font-weight:700; color:#FF4B4B; }
.kpi-lbl    { color:#aaa; font-size:.75rem; text-transform:uppercase; letter-spacing:1px; }
.sec        { font-family:'Space Mono',monospace; font-size:1rem; color:#FF4B4B;
              border-bottom:1px solid #FF4B4B44;
              padding-bottom:.4rem; margin:1.4rem 0 .9rem; }
.sql-box    { background:#0d1117; border:1px solid #30363d; border-radius:8px;
              padding:.9rem 1rem; font-family:'Space Mono',monospace;
              font-size:.75rem; color:#c9d1d9; white-space:pre-wrap; }
.churn-hi   { background:linear-gradient(135deg,#FF4B4B22,#FF4B4B11);
              border:2px solid #FF4B4B; border-radius:12px; padding:1rem;
              text-align:center; font-family:'Space Mono',monospace;
              color:#FF4B4B; font-size:1.4rem; }
.churn-lo   { background:linear-gradient(135deg,#00C85122,#00C85111);
              border:2px solid #00C851; border-radius:12px; padding:1rem;
              text-align:center; font-family:'Space Mono',monospace;
              color:#00C851; font-size:1.4rem; }
.stButton>button { background:linear-gradient(135deg,#FF4B4B,#c0392b)!important;
                   color:white!important; border:none!important; border-radius:8px!important;
                   font-family:'Space Mono',monospace!important; font-weight:700!important;
                   padding:.55rem 1.8rem!important; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📉 Churn Intelligence")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 Overview",
        "🗄️ SQL Explorer",
        "📊 EDA",
        "🤖 ML Performance",
        "🔮 Predict",
    ])
    st.markdown("---")
    st.markdown(
        "<small>Built by **Dixitha BV** · AIML 6th Sem<br>"
        "[GitHub](https://github.com/DixithaBV) · "
        "[LinkedIn](https://linkedin.com/in/dixitha-bv-9467a9359)</small>",
        unsafe_allow_html=True,
    )


# ── data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_customers() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql("SELECT * FROM customers", con)
    con.close()
    return df


@st.cache_resource
def load_model_artifacts():
    model       = pickle.load(open(f"{MDL_DIR}/model.pkl",       "rb"))
    scaler      = pickle.load(open(f"{MDL_DIR}/scaler.pkl",      "rb"))
    encoder_map = pickle.load(open(f"{MDL_DIR}/encoder_map.pkl", "rb"))
    shap_imp    = pd.read_csv(f"{MDL_DIR}/shap_importance.csv",  index_col=0)
    shap_imp.columns = ["importance"]
    return model, scaler, encoder_map, shap_imp


def run_query(sql: str) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    try:
        result = pd.read_sql(sql, con)
    except Exception as e:
        result = pd.DataFrame({"error": [str(e)]})
    finally:
        con.close()
    return result


def call_predict(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        return r.json()
    except Exception:
        return None


def call_stats() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/stats", timeout=10)
        return r.json()
    except Exception:
        return None


# ── load everything ───────────────────────────────────────────────────────────
df = load_customers()
try:
    model, scaler, encoder_map, shap_imp = load_model_artifacts()
    MODEL_LOADED = True
except Exception:
    MODEL_LOADED = False


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<p class="main-title">📉 Customer Churn Intelligence System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">End-to-end ML pipeline · SQLite → XGBoost → FastAPI → Streamlit · by Dixitha BV</p>', unsafe_allow_html=True)

    # Try live API first, fall back to direct DB
    stats = call_stats()

    if stats:
        total    = stats["total_customers"]
        churned  = stats["churned"]
        retained = stats["retained"]
        rate     = stats["churn_rate_pct"]
        risk     = stats["revenue_at_risk"]
        avg_m    = stats["avg_monthly"]
        src      = "🟢 Live API"
    else:
        total    = len(df)
        churned  = int(df["Churn"].sum())
        retained = total - churned
        rate     = round(churned / total * 100, 2)
        risk     = round(df[df["Churn"] == 1]["MonthlyCharges"].sum(), 2)
        avg_m    = round(df["MonthlyCharges"].mean(), 2)
        src      = "🟡 Direct DB (API offline)"

    st.caption(f"Data source: {src}")

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in zip(
        [c1, c2, c3, c4, c5],
        [f"{total:,}", f"{churned:,}", f"{retained:,}", f"{rate}%", f"${risk:,.0f}"],
        ["Total Customers", "Churned", "Retained", "Churn Rate", "Revenue at Risk"],
    ):
        col.markdown(f'<div class="kpi-card"><div class="kpi-val">{val}</div>'
                     f'<div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<p class="sec">Churn Rate by Contract Type</p>', unsafe_allow_html=True)
    contract_df = df.groupby("Contract")["Churn"].agg(["sum", "count"]).reset_index()
    contract_df["churn_pct"] = contract_df["sum"] / contract_df["count"] * 100
    fig = px.bar(contract_df, x="Contract", y="churn_pct",
                 color="churn_pct", color_continuous_scale=["#00C851", "#FF4B4B"],
                 labels={"churn_pct": "Churn Rate (%)"}, template="plotly_dark")
    fig.update_layout(coloraxis_showscale=False,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=320)
    st.plotly_chart(fig, use_container_width=True)

    if MODEL_LOADED:
        st.markdown('<p class="sec">Top Churn Factors (SHAP Feature Importance)</p>', unsafe_allow_html=True)
        fig2 = px.bar(shap_imp.reset_index().head(8), x="index", y="importance",
                      color="importance", color_continuous_scale=["#16213e", "#FF4B4B"],
                      labels={"index": "Feature", "importance": "Mean |SHAP|"},
                      template="plotly_dark")
        fig2.update_layout(coloraxis_showscale=False,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=300)
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SQL EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗄️ SQL Explorer":
    st.markdown('<p class="main-title">🗄️ SQL Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Run real SQL queries directly on the churn.db SQLite database</p>', unsafe_allow_html=True)

    query_titles = [q["title"] for q in QUERIES]
    selected_idx = st.selectbox("Choose a pre-written query:", range(len(query_titles)),
                                format_func=lambda i: query_titles[i])
    q = QUERIES[selected_idx]
    st.caption(q["description"])

    st.markdown('<p class="sec">SQL Query</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="sql-box">{q["sql"].strip()}</div>', unsafe_allow_html=True)

    st.markdown("")
    custom_sql = st.text_area("Or write your own SQL query:", height=100,
                              placeholder="SELECT Contract, COUNT(*) FROM customers GROUP BY Contract;")

    col_run, col_custom = st.columns([1, 5])
    with col_run:
        run_pre    = st.button("▶ Run selected")
        run_custom = st.button("▶ Run custom")

    if run_pre:
        with st.spinner("Running query …"):
            result = run_query(q["sql"])
        st.markdown('<p class="sec">Query Results</p>', unsafe_allow_html=True)
        st.dataframe(result, use_container_width=True)
        st.caption(f"{len(result):,} rows returned")

        if len(result) > 1 and result.select_dtypes(include=np.number).shape[1] >= 1:
            num_col = result.select_dtypes(include=np.number).columns[-1]
            str_col = result.select_dtypes(exclude=np.number).columns[0] if result.select_dtypes(exclude=np.number).shape[1] > 0 else result.columns[0]
            fig_q = px.bar(result, x=str_col, y=num_col,
                           template="plotly_dark", color_discrete_sequence=["#FF4B4B"])
            fig_q.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=300)
            st.plotly_chart(fig_q, use_container_width=True)

    if run_custom and custom_sql.strip():
        with st.spinner("Running query …"):
            result = run_query(custom_sql)
        st.markdown('<p class="sec">Custom Query Results</p>', unsafe_allow_html=True)
        st.dataframe(result, use_container_width=True)
        st.caption(f"{len(result):,} rows returned")

    st.markdown("---")
    st.markdown('<p class="sec">Database Schema</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**customers** table")
        st.code("customerID, gender, SeniorCitizen, Partner,\nDependents, tenure, PhoneService,\nInternetService, Contract, PaperlessBilling,\nPaymentMethod, MonthlyCharges, TotalCharges, Churn")
        st.markdown("**segment_summary** table")
        st.code("Contract, InternetService, total_customers,\nchurned, avg_monthly, avg_tenure")
    with col2:
        st.markdown("**transactions** table")
        st.code("customerID, month_num, amount,\npayment_method, paid_on_time")
        st.markdown("**support_tickets** table")
        st.code("customerID, category, resolved, rating")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown('<p class="main-title">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="sec">Tenure vs Monthly Charges</p>', unsafe_allow_html=True)
        fig1 = px.scatter(df.sample(1500, random_state=42), x="tenure", y="MonthlyCharges",
                          color=df.sample(1500, random_state=42)["Churn"].map({0:"Retained",1:"Churned"}),
                          color_discrete_map={"Churned":"#FF4B4B","Retained":"#00C851"},
                          opacity=0.55, template="plotly_dark")
        fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown('<p class="sec">Churn by Internet Service</p>', unsafe_allow_html=True)
        int_ch = df.groupby("InternetService")["Churn"].mean().reset_index()
        int_ch["Churn"] *= 100
        fig2 = px.pie(int_ch, names="InternetService", values="Churn",
                      color_discrete_sequence=["#FF4B4B","#0f3460","#00C851"],
                      template="plotly_dark")
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="sec">Monthly Charges Distribution by Churn Status</p>', unsafe_allow_html=True)
    fig3 = px.histogram(df, x="MonthlyCharges",
                        color=df["Churn"].map({0:"Retained",1:"Churned"}),
                        color_discrete_map={"Churned":"#FF4B4B","Retained":"#00C851"},
                        barmode="overlay", opacity=0.7, nbins=40, template="plotly_dark")
    fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<p class="sec">Churn Rate by Payment Method</p>', unsafe_allow_html=True)
    pay_ch = df.groupby("PaymentMethod")["Churn"].mean().reset_index()
    pay_ch["Churn"] = (pay_ch["Churn"] * 100).round(2)
    fig4 = px.bar(pay_ch, x="PaymentMethod", y="Churn",
                  color="Churn", color_continuous_scale=["#00C851","#FF4B4B"],
                  labels={"Churn":"Churn Rate (%)"}, template="plotly_dark")
    fig4.update_layout(coloraxis_showscale=False,
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<p class="sec">Raw Data Table</p>', unsafe_allow_html=True)
    st.dataframe(df.head(200), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ML PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Performance":
    st.markdown('<p class="main-title">🤖 ML Model Performance</p>', unsafe_allow_html=True)

    if not MODEL_LOADED:
        st.warning("⚠️ model.pkl not found. Run `python ml_pipeline.py` first.")
        st.stop()

    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, roc_auc_score,
                                 confusion_matrix, roc_curve)

    CAT_F = ["gender","Partner","Dependents","PhoneService",
             "InternetService","Contract","PaperlessBilling","PaymentMethod"]
    NUM_F = ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]
    ALL_F = NUM_F + CAT_F

    df_enc = df[ALL_F + ["Churn"]].copy()
    for col in CAT_F:
        le = encoder_map[col]
        df_enc[col] = le.transform(df_enc[col].astype(str))

    X = scaler.transform(df_enc[ALL_F])
    y = df_enc["Churn"].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "XGBoost")
    c2.metric("Accuracy", f"{acc:.2%}")
    c3.metric("ROC-AUC", f"{auc:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="sec">Confusion Matrix</p>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, template="plotly_dark",
                           color_continuous_scale=["#16213e","#FF4B4B"],
                           x=["Retained","Churned"], y=["Retained","Churned"])
        fig_cm.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.markdown('<p class="sec">ROC Curve</p>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"XGBoost AUC={auc:.4f}",
                                     line=dict(color="#FF4B4B", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     line=dict(dash="dash", color="#555"), name="Random"))
        fig_roc.update_layout(template="plotly_dark",
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               xaxis_title="False Positive Rate",
                               yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown('<p class="sec">SHAP Feature Importance</p>', unsafe_allow_html=True)
    fig_shap = px.bar(shap_imp.reset_index().head(10), x="index", y="importance",
                      color="importance", color_continuous_scale=["#16213e","#FF4B4B"],
                      labels={"index":"Feature","importance":"Mean |SHAP value|"},
                      template="plotly_dark")
    fig_shap.update_layout(coloraxis_showscale=False,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_shap, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown('<p class="main-title">🔮 Predict Customer Churn</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Enter customer details — prediction powered by the FastAPI backend</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<p class="sec">👤 Demographics</p>', unsafe_allow_html=True)
        gender     = st.selectbox("Gender", ["Male","Female"])
        senior     = st.selectbox("Senior Citizen", ["No","Yes"])
        partner    = st.selectbox("Has Partner", ["Yes","No"])
        dependents = st.selectbox("Has Dependents", ["Yes","No"])

    with c2:
        st.markdown('<p class="sec">📱 Services</p>', unsafe_allow_html=True)
        phone    = st.selectbox("Phone Service", ["Yes","No"])
        internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
        paperless= st.selectbox("Paperless Billing", ["Yes","No"])
        payment  = st.selectbox("Payment Method",
                                ["Electronic check","Mailed check",
                                 "Bank transfer (automatic)","Credit card (automatic)"])

    with c3:
        st.markdown('<p class="sec">💰 Account</p>', unsafe_allow_html=True)
        tenure   = st.slider("Tenure (months)", 0, 72, 12)
        monthly  = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
        total    = st.number_input("Total Charges ($)", min_value=0.0,
                                   value=float(tenure * monthly), step=10.0)
        contract = st.selectbox("Contract Type",
                                ["Month-to-month","One year","Two year"])

    st.markdown("")
    if st.button("🔮 Predict Churn Probability"):
        payload = {
            "gender": gender, "SeniorCitizen": 1 if senior=="Yes" else 0,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone,
            "InternetService": internet, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }

        result = call_predict(payload)

        # ── fallback: run model locally if API is offline ─────────────────────
        if result is None and MODEL_LOADED:
            st.warning("⚠️ API offline — running prediction locally.")
            CAT_F = ["gender","Partner","Dependents","PhoneService",
                     "InternetService","Contract","PaperlessBilling","PaymentMethod"]
            NUM_F = ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]
            ALL_F = NUM_F + CAT_F
            row   = {k: payload[k] for k in ALL_F}
            inp   = pd.DataFrame([row])[ALL_F]
            for col in CAT_F:
                le  = encoder_map[col]
                val = str(inp[col].iloc[0])
                inp[col] = le.transform([val])[0] if val in le.classes_ else 0
            X_in = scaler.transform(inp)
            prob = float(model.predict_proba(X_in)[0][1])
            result = {
                "churn_probability": round(prob, 4),
                "churn_prediction":  int(prob >= 0.5),
                "risk_level":        "HIGH" if prob>=0.7 else "MEDIUM" if prob>=0.4 else "LOW",
                "recommendations":   ["Run FastAPI backend for full recommendations."],
            }

        if result:
            prob = result["churn_probability"]
            risk = result["risk_level"]

            _, mid, _ = st.columns([1,2,1])
            with mid:
                cls = "churn-hi" if prob >= 0.5 else "churn-lo"
                icon = "⚠️ HIGH CHURN RISK" if prob >= 0.5 else "✅ LOW CHURN RISK"
                st.markdown(f'<div class="{cls}">{icon}<br>'
                            f'<span style="font-size:2.4rem">{prob:.1%}</span><br>'
                            f'<span style="font-size:.8rem">Risk Level: {risk}</span>'
                            f'</div>', unsafe_allow_html=True)

            st.markdown("")
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob*100,
                title={"text":"Churn Risk Score","font":{"color":"#aaa"}},
                gauge={"axis":{"range":[0,100]},
                       "bar":{"color":"#FF4B4B" if prob>=0.5 else "#00C851"},
                       "steps":[{"range":[0,40],"color":"#1a2744"},
                                {"range":[40,70],"color":"#2a1f3d"},
                                {"range":[70,100],"color":"#3d1a1a"}],
                       "threshold":{"line":{"color":"white","width":3},"value":50}},
                number={"suffix":"%","font":{"color":"#FF4B4B" if prob>=0.5 else "#00C851"}}
            ))
            fig_g.update_layout(template="plotly_dark",
                                paper_bgcolor="rgba(0,0,0,0)", height=270)
            st.plotly_chart(fig_g, use_container_width=True)

            st.markdown('<p class="sec">💡 Business Recommendations</p>', unsafe_allow_html=True)
            for r in result.get("recommendations", []):
                st.markdown(f"- {r}")
        else:
            st.error("❌ Prediction failed. Make sure the API is running and model.pkl exists.")