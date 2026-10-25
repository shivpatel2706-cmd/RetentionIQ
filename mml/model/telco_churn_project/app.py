"""RetentionIQ™ | SaaS-Grade Customer Churn Intelligence Platform (Light Edition).

Enterprise Light-Themed Executive Dashboard:
- Clean, Airy White Card Containers & Soft Drop Shadows
- High-Contrast Charcoal Typography & Royal Slate Blue Accents
- Interactive Plotly Light Visualizations (Gauges, Donut Charts & Feature Impact)
- Real-Time Single Customer Profiling & Prescriptive Retention Playbooks
- Portfolio Executive Analytics & Batch CSV Ingestion
- Model Transparency & Interactive "What-If" Scenario Simulator
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup page configuration
st.set_page_config(
    page_title="RetentionIQ™ | Enterprise Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

INR_RATE = 83.0  # Standard USD to INR conversion rate

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
PIPELINE_PATH = MODELS_DIR / "best_churn_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
EVAL_PATH = MODELS_DIR / "evaluation_metrics.json"
MEDIAN_PATH = MODELS_DIR / "median_total_charges.json"

# ==========================================
# ENTERPRISE LIGHT SAAS THEME (CUSTOM CSS)
# ==========================================
st.markdown(
    """
    <style>
    /* Google Font: Plus Jakarta Sans */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Overall App Background */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F1F5F9;
        border-right: 1px solid #E2E8F0;
    }

    /* Hero Banner Card */
    .hero-card-light {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
    }
    .hero-title-light {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #0F172A;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .hero-title-accent {
        background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle-light {
        font-size: 1.02rem;
        color: #64748B;
        font-weight: 400;
        margin: 0;
        line-height: 1.5;
    }

    /* Live Engine Ready Badge */
    .status-badge-light {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #047857;
        padding: 0.4rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .pulse-dot-light {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
        animation: pulse-light 2s infinite;
    }
    @keyframes pulse-light {
        0% { transform: scale(0.95); opacity: 0.8; }
        50% { transform: scale(1.3); opacity: 1; box-shadow: 0 0 12px #10B981; }
        100% { transform: scale(0.95); opacity: 0.8; }
    }

    /* White Glass Cards */
    .card-light {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .card-light:hover {
        border-color: #CBD5E1;
        box-shadow: 0 8px 24px -4px rgba(15, 23, 42, 0.08);
    }
    .card-title-light {
        font-size: 1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 0.6rem;
    }

    /* Executive KPI Metric Cards */
    .kpi-card-light {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.04);
    }
    .kpi-card-light::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #2563EB, #4F46E5);
    }
    .kpi-label-light {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        margin-bottom: 0.35rem;
    }
    .kpi-value-light {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .kpi-desc-light {
        font-size: 0.82rem;
        color: #94A3B8;
        font-weight: 500;
    }

    /* High Risk Badge Card */
    .badge-card-high-light {
        background: #FEF2F2;
        border: 1px solid #FCA5A5;
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.08);
    }
    .badge-title-high-light {
        color: #DC2626;
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        margin: 0;
    }
    .badge-prob-high-light {
        font-size: 3.2rem;
        font-weight: 900;
        color: #991B1B;
        margin: 0.3rem 0;
        letter-spacing: -0.03em;
    }

    /* Low Risk Badge Card */
    .badge-card-low-light {
        background: #ECFDF5;
        border: 1px solid #6EE7B7;
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.08);
    }
    .badge-title-low-light {
        color: #059669;
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        margin: 0;
    }
    .badge-prob-low-light {
        font-size: 3.2rem;
        font-weight: 900;
        color: #065F46;
        margin: 0.3rem 0;
        letter-spacing: -0.03em;
    }

    /* Recommendation Item Card */
    .rec-item-light {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.75rem;
        color: #1E293B;
        font-size: 0.93rem;
        line-height: 1.5;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }

    /* Tab navigation polish */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #F1F5F9;
        padding: 5px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        color: #64748B;
        font-weight: 600;
        font-size: 0.92rem;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        color: #2563EB !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06) !important;
    }

    /* Streamlit Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        letter-spacing: 0.01em;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_artifacts():
    """Cache and load champion pipeline, system metadata, and evaluation records."""
    if not PIPELINE_PATH.exists():
        st.error(f"Trained model not found at {PIPELINE_PATH}. Please run training first!")
        st.stop()

    pipeline = joblib.load(PIPELINE_PATH)

    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    eval_metrics = {}
    if EVAL_PATH.exists():
        with open(EVAL_PATH, "r", encoding="utf-8") as f:
            eval_metrics = json.load(f)

    median_tc = 1397.47
    if MEDIAN_PATH.exists():
        with open(MEDIAN_PATH, "r", encoding="utf-8") as f:
            median_tc = json.load(f).get("median_total_charges", 1397.47)

    return pipeline, metadata, eval_metrics, median_tc


pipeline, metadata, eval_metrics, default_median_tc = load_model_artifacts()
champion_name = metadata.get("champion_model", "XGBoost")
test_metrics = eval_metrics.get("metrics", {})
business_metrics = eval_metrics.get("business_impact", {})


def generate_prescriptive_recommendations(customer: Dict[str, Any], prob: float) -> List[str]:
    """Generate business-aligned retention strategies tailored to subscriber risk drivers."""
    recs = []

    if customer.get("Contract") == "Month-to-month":
        recs.append("🎯 **Annual Contract Transition Incentive:** Offer a guaranteed 15% rate freeze or ₹800/month bill credit in exchange for a 1-year contract commitment.")

    if customer.get("InternetService") == "Fiber optic" and customer.get("TechSupport") == "No":
        recs.append("🛠️ **VIP Tech Support Onboarding:** Bundle 3 months of complimentary 24/7 Priority Tech Support to resolve fiber optic latency and satisfaction issues.")

    if customer.get("PaymentMethod") == "Electronic check":
        recs.append("💳 **Automated ACH/Card Incentive:** Provide an immediate ₹1,250 bill credit for activating auto-pay with a credit card or direct bank transfer.")

    if customer.get("OnlineSecurity") == "No" or customer.get("OnlineBackup") == "No":
        recs.append("🛡️ **Cyber Shield Add-on Trial:** Deploy a 60-day complimentary Digital Security & Cloud Backup package to increase service stickiness.")

    if int(customer.get("tenure", 0)) <= 6:
        recs.append("🤝 **Early-Tenure Onboarding Concierge:** Enroll customer in the 90-day Customer Success check-in cadence to mitigate early tenure churn.")

    if not recs:
        recs.append("🌟 **Loyalty Rewards Tier Upgrade:** Customer exhibits loyal behavior; provide a VIP loyalty points milestone to maintain long-term advocacy.")

    return recs[:3]


def render_light_gauge_chart(churn_prob: float, is_high_risk: bool) -> go.Figure:
    """Render a crisp Plotly light-theme gauge chart."""
    prob_pct = round(churn_prob * 100, 1)
    bar_color = "#EF4444" if is_high_risk else "#10B981"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={
                "suffix": "%",
                "font": {"color": "#0F172A", "size": 46, "family": "Plus Jakarta Sans, sans-serif"},
            },
            title={
                "text": "CHURN RISK PROBABILITY",
                "font": {"color": "#64748B", "size": 13, "family": "Plus Jakarta Sans, sans-serif"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#CBD5E1",
                    "tickfont": {"color": "#64748B", "size": 11},
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "#F1F5F9",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "rgba(16, 185, 129, 0.12)"},
                    {"range": [35, 65], "color": "rgba(245, 158, 11, 0.12)"},
                    {"range": [65, 100], "color": "rgba(239, 68, 68, 0.14)"},
                ],
                "threshold": {
                    "line": {"color": bar_color, "width": 4},
                    "thickness": 0.8,
                    "value": prob_pct,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font={"color": "#0F172A", "family": "Plus Jakarta Sans"},
        margin=dict(l=25, r=25, t=35, b=25),
        height=250,
    )
    return fig


# ==========================================
# SIDEBAR NAVIGATION & TELEMETRY
# ==========================================
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
            <div style="background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%); width: 42px; height: 42px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.35rem; color: #FFFFFF; box-shadow: 0 4px 10px rgba(37,99,235,0.25);">📡</div>
            <div>
                <h3 style="margin: 0; font-size: 1.2rem; font-weight: 800; color: #0F172A;">RetentionIQ™</h3>
                <span style="font-size: 0.75rem; color: #2563EB; font-weight: 700; letter-spacing: 0.05em;">ENTERPRISE EDITION</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Production Engine")
    st.markdown(f"**Architecture:** `{champion_name} Classifier`")
    st.markdown(f"**Cross-Validation ROC-AUC:** `{metadata.get('champion_cv_roc_auc', 0.8461):.4f}`")
    st.markdown(f"**Holdout Test Recall:** `{test_metrics.get('recall', 0.7219)*100:.1f}%`")
    st.markdown(f"**Sampling Policy:** `Fold-Isolated SMOTE`")

    st.markdown("---")
    st.markdown("#### 💼 Portfolio Economics (INR)")
    annual_risk_inr = business_metrics.get('total_annual_revenue_at_risk_usd', 326578.80) * INR_RATE
    net_saved_inr = business_metrics.get('net_revenue_saved_usd', 84135.36) * INR_RATE
    st.markdown(f"**Annual Revenue At Risk:** `₹{annual_risk_inr:,.2f}`")
    st.markdown(f"**Net Revenue Saved:** `₹{net_saved_inr:,.2f}`")
    st.markdown(f"**Projected ROI:** `+{business_metrics.get('roi_percentage', 692.47):.1f}%`")

    st.markdown("---")
    st.markdown(
        """
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 0.85rem; font-size: 0.78rem; color: #64748B; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
            🛡️ <b>Strict Data Leakage Prevention:</b> Stratified splits prior to ColumnTransformer feature engineering and inference scaling.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# HEADER HERO BANNER
# ==========================================
st.markdown(
    f"""
    <div class="hero-card-light">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
            <div>
                <div class="hero-title-light">
                    <span>📡</span> RetentionIQ™ <span class="hero-title-accent">Churn Intelligence</span>
                </div>
                <p class="hero-subtitle-light">
                    Autonomous Subscriber Risk Profiling, MRR Loss Prevention, and Prescriptive Retention Operations.
                </p>
            </div>
            <div>
                <div class="status-badge-light">
                    <span class="pulse-dot-light"></span>
                    <span>AI Engine Ready ({champion_name})</span>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# MAIN TABBED WORKSPACE
# ==========================================
tab_single, tab_batch, tab_insights = st.tabs(
    [
        "⚡ Single Customer Risk Profiler",
        "📂 Executive Portfolio & Batch Analytics",
        "🔬 Model Transparency & What-If Simulator",
    ]
)


# ==============================================================================
# TAB 1: SINGLE CUSTOMER PROFILER
# ==============================================================================
with tab_single:
    st.markdown("#### ⚡ Real-Time Subscriber Risk Assessment")
    st.caption("Input subscriber characteristics to calculate churn likelihood and generate targeted retention playbooks.")

    # Profile Quick-Fill Presets
    col_pre1, col_pre2, col_pre3, col_pre4 = st.columns([1.2, 1.2, 1.2, 2.4])
    with col_pre1:
        load_high = st.button("🚨 Load High Risk Profile", use_container_width=True)
    with col_pre2:
        load_mid = st.button("⚖️ Load Moderate Profile", use_container_width=True)
    with col_pre3:
        load_loyal = st.button("🛡️ Load Loyal Profile", use_container_width=True)

    # State presets
    defaults = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50,
        "TotalCharges": 191.00,
    }

    if load_high:
        defaults.update({
            "tenure": 1,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
            "MonthlyCharges": 99.85,
            "TotalCharges": 99.85,
            "PaymentMethod": "Electronic check",
            "TechSupport": "No",
            "OnlineSecurity": "No",
            "PaperlessBilling": "Yes",
        })
    elif load_mid:
        defaults.update({
            "tenure": 18,
            "Contract": "One year",
            "InternetService": "DSL",
            "MonthlyCharges": 58.50,
            "TotalCharges": 1053.00,
            "PaymentMethod": "Mailed check",
            "TechSupport": "No",
            "OnlineSecurity": "Yes",
            "PaperlessBilling": "No",
        })
    elif load_loyal:
        defaults.update({
            "tenure": 64,
            "Contract": "Two year",
            "InternetService": "DSL",
            "MonthlyCharges": 65.00,
            "TotalCharges": 4160.00,
            "PaymentMethod": "Credit card (automatic)",
            "TechSupport": "Yes",
            "OnlineSecurity": "Yes",
            "PaperlessBilling": "No",
            "Partner": "Yes",
            "Dependents": "Yes",
        })

    with st.form("single_prediction_form"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown(
                """
                <div class="card-title-light">
                    <span>👤</span> Demographics & Contract
                </div>
                """,
                unsafe_allow_html=True,
            )
            gender = st.selectbox("Gender", ["Female", "Male"], index=0 if defaults.get("gender") == "Female" else 1)
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=defaults.get("SeniorCitizen", 0))
            partner = st.selectbox("Partner", ["Yes", "No"], index=0 if defaults.get("Partner") == "Yes" else 1)
            dependents = st.selectbox("Dependents", ["Yes", "No"], index=0 if defaults.get("Dependents") == "Yes" else 1)
            tenure = st.slider("Tenure with Company (Months)", min_value=0, max_value=72, value=int(defaults.get("tenure", 2)))
            contract = st.selectbox("Contract Terms", ["Month-to-month", "One year", "Two year"], index=0 if defaults.get("Contract") == "Month-to-month" else (1 if defaults.get("Contract") == "One year" else 2))
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if defaults.get("PaperlessBilling") == "Yes" else 1)

        with col_b:
            st.markdown(
                """
                <div class="card-title-light">
                    <span>📡</span> Connected Services & Add-Ons
                </div>
                """,
                unsafe_allow_html=True,
            )
            phone = st.selectbox("Phone Service", ["Yes", "No"], index=0 if defaults.get("PhoneService") == "Yes" else 1)
            multilines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=0)
            internet = st.selectbox("Internet Infrastructure", ["Fiber optic", "DSL", "No"], index=0 if defaults.get("InternetService") == "Fiber optic" else (1 if defaults.get("InternetService") == "DSL" else 2))
            security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=0 if defaults.get("OnlineSecurity") == "No" else 1)
            backup = st.selectbox("Online Cloud Backup", ["No", "Yes", "No internet service"], index=0 if defaults.get("OnlineBackup") == "No" else 1)
            device = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=0 if defaults.get("DeviceProtection") == "No" else 1)
            tech = st.selectbox("24/7 Tech Support", ["No", "Yes", "No internet service"], index=0 if defaults.get("TechSupport") == "No" else 1)
            stream_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=1 if defaults.get("StreamingTV") == "Yes" else 0)
            stream_movie = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=1 if defaults.get("StreamingMovies") == "Yes" else 0)

        with col_c:
            st.markdown(
                """
                <div class="card-title-light">
                    <span>💳</span> Financials & Billing
                </div>
                """,
                unsafe_allow_html=True,
            )
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                index=0 if defaults.get("PaymentMethod") == "Electronic check" else 1,
            )
            monthly_inr = st.number_input("Monthly Recurring Charges (₹)", min_value=1200.0, max_value=14000.0, value=float(defaults.get("MonthlyCharges", 95.50)) * INR_RATE, step=100.0)
            monthly = monthly_inr / INR_RATE
            calculated_approx_tc_inr = float(monthly_inr * (tenure if tenure > 0 else 1))
            total_charges_inr = st.number_input("Cumulative Total Charges (₹)", min_value=0.0, max_value=850000.0, value=float(defaults.get("TotalCharges", 95.50 * (tenure if tenure > 0 else 1))) * INR_RATE, step=500.0)
            total_charges = total_charges_inr / INR_RATE

            st.write("")
            st.write("")
            submitted = st.form_submit_button("🔮 Predict Churn Likelihood", type="primary", use_container_width=True)

    if submitted or load_high or load_mid or load_loyal:
        input_data = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multilines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_movie,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total_charges,
        }

        input_df = pd.DataFrame([input_data])
        churn_prob = float(pipeline.predict_proba(input_df)[0, 1])
        is_high_risk = churn_prob >= 0.50

        st.markdown("---")
        st.markdown("#### 🎯 Predictive Intelligence Output")

        out_col1, out_col2 = st.columns([1.1, 1.9])

        with out_col1:
            # Dynamic Light Gauge Chart
            fig_gauge = render_light_gauge_chart(churn_prob, is_high_risk)
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

            # High vs Low Risk Badge (Light Palette)
            if is_high_risk:
                st.markdown(
                    f"""
                    <div class="badge-card-high-light">
                        <div class="badge-title-high-light">⚠️ HIGH CHURN RISK</div>
                        <div class="badge-prob-high-light">{churn_prob*100:.1f}%</div>
                        <div style="color: #B91C1C; font-size: 0.85rem; font-weight: 500;">Action Required: High likelihood of subscriber cancellation within 30-60 days.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="badge-card-low-light">
                        <div class="badge-title-low-light">✅ LOW CHURN RISK</div>
                        <div class="badge-prob-low-light">{churn_prob*100:.1f}%</div>
                        <div style="color: #047857; font-size: 0.85rem; font-weight: 500;">Stable: High retention affinity and healthy engagement metrics.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")
            fin_c1, fin_c2 = st.columns(2)
            fin_c1.metric("Monthly Spend", f"₹{monthly * INR_RATE:,.2f}")
            fin_c2.metric("Annual Revenue At Risk", f"₹{monthly * 12 * INR_RATE:,.2f}")

        with out_col2:
            st.markdown(
                """
                <div class="card-title-light" style="margin-top: 0.5rem;">
                    <span>💡</span> Prescriptive Retention Playbook
                </div>
                """,
                unsafe_allow_html=True,
            )
            recommendations = generate_prescriptive_recommendations(input_data, churn_prob)
            for rec in recommendations:
                st.markdown(f'<div class="rec-item-light">{rec}</div>', unsafe_allow_html=True)

            st.write("")
            st.markdown(
                """
                <div class="card-title-light">
                    <span>🔍</span> Real-Time Engineered Signals
                </div>
                """,
                unsafe_allow_html=True,
            )
            avg_ratio = total_charges / (tenure + 1)
            discrepancy = monthly - avg_ratio
            sig1, sig2, sig3 = st.columns(3)
            sig1.metric("Historical Cost Ratio", f"₹{avg_ratio * INR_RATE:,.2f}/mo")
            sig2.metric("Monthly Price Inflation", f"₹{discrepancy * INR_RATE:+,.2f}")
            sig3.metric("Streaming Bundle", "Active" if (stream_tv == "Yes" or stream_movie == "Yes") else "None")


# ==============================================================================
# TAB 2: PORTFOLIO & BATCH ANALYTICS
# ==============================================================================
with tab_batch:
    st.markdown("#### 📂 Executive Portfolio Analytics & Batch Inference")
    st.caption("Upload multi-subscriber datasets to quantify cumulative revenue risk, detect customer segments, and trigger bulk interventions.")

    # File Upload Section
    up_col1, up_col2 = st.columns([3, 1])
    with up_col1:
        uploaded_file = st.file_uploader("Upload Customer Records CSV", type=["csv"], label_visibility="collapsed")
    with up_col2:
        load_demo_batch = st.button("📥 Load Demo Batch (100 Records)", use_container_width=True)

    batch_df = None
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
    elif load_demo_batch:
        raw_csv_path = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        if raw_csv_path.exists():
            batch_df = pd.read_csv(raw_csv_path).head(100)

    if batch_df is not None:
        # Preprocessing & Scoring
        proc_df = batch_df.copy()
        if "TotalCharges" in proc_df.columns:
            proc_df["TotalCharges"] = pd.to_numeric(
                proc_df["TotalCharges"].astype(str).str.strip(), errors="coerce"
            ).fillna(default_median_tc)

        with st.spinner("Executing batch inference across pipeline..."):
            batch_probs = pipeline.predict_proba(proc_df)[:, 1]
            batch_preds = pipeline.predict(proc_df)

            proc_df["Churn_Probability"] = np.round(batch_probs, 4)
            proc_df["Risk_Category"] = np.where(batch_probs >= 0.50, "High Risk", "Low Risk")

        total_analyzed = len(proc_df)
        high_risk_count = int((proc_df["Risk_Category"] == "High Risk").sum())
        churn_rate_pct = (high_risk_count / total_analyzed) * 100
        monthly_at_risk = float(proc_df.loc[proc_df["Risk_Category"] == "High Risk", "MonthlyCharges"].sum())
        annual_at_risk = monthly_at_risk * 12
        saved_est = annual_at_risk * 0.40 - (high_risk_count * 25.0)

        # 4-Column Executive KPI Row (Light White Cards)
        st.markdown("---")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            st.markdown(
                f"""
                <div class="kpi-card-light">
                    <div class="kpi-label-light">Subscribers Evaluated</div>
                    <div class="kpi-value-light">{total_analyzed:,}</div>
                    <div class="kpi-desc-light">Total cohort batch size</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with kpi2:
            st.markdown(
                f"""
                <div class="kpi-card-light">
                    <div class="kpi-label-light">High-Risk Churn Rate</div>
                    <div class="kpi-value-light" style="color: #EF4444;">{churn_rate_pct:.1f}%</div>
                    <div class="kpi-desc-light">{high_risk_count} subscribers flagging churn</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with kpi3:
            st.markdown(
                f"""
                <div class="kpi-card-light">
                    <div class="kpi-label-light">Annual Revenue At Risk</div>
                    <div class="kpi-value-light" style="color: #D97706;">₹{annual_at_risk * INR_RATE:,.2f}</div>
                    <div class="kpi-desc-light">₹{monthly_at_risk * INR_RATE:,.2f}/month recurring</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with kpi4:
            st.markdown(
                f"""
                <div class="kpi-card-light">
                    <div class="kpi-label-light">Preserved Revenue (Est.)</div>
                    <div class="kpi-value-light" style="color: #10B981;">₹{max(0, saved_est) * INR_RATE:,.2f}</div>
                    <div class="kpi-desc-light">Net savings from 40% retention</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        # Visual Charts Row (Light Aesthetic)
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("##### 🍩 Risk Distribution Breakdown")
            donut_fig = px.pie(
                names=["Retained / Low Risk", "At-Risk Churners"],
                values=[total_analyzed - high_risk_count, high_risk_count],
                hole=0.6,
                color=["Retained / Low Risk", "At-Risk Churners"],
                color_discrete_map={"Retained / Low Risk": "#10B981", "At-Risk Churners": "#EF4444"},
            )
            donut_fig.update_layout(
                paper_bgcolor="rgba(255,255,255,0)",
                plot_bgcolor="rgba(255,255,255,0)",
                font={"color": "#0F172A", "family": "Plus Jakarta Sans"},
                margin=dict(l=10, r=10, t=20, b=20),
                height=280,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            )
            st.plotly_chart(donut_fig, use_container_width=True, config={"displayModeBar": False})

        with chart_col2:
            st.markdown("##### 📊 Risk Distribution by Contract Type")
            if "Contract" in proc_df.columns:
                contract_risk = proc_df.groupby(["Contract", "Risk_Category"]).size().reset_index(name="Count")
                bar_fig = px.bar(
                    contract_risk,
                    x="Contract",
                    y="Count",
                    color="Risk_Category",
                    barmode="group",
                    color_discrete_map={"Low Risk": "#10B981", "High Risk": "#EF4444"},
                )
                bar_fig.update_layout(
                    paper_bgcolor="rgba(255,255,255,0)",
                    plot_bgcolor="rgba(255,255,255,0)",
                    font={"color": "#0F172A", "family": "Plus Jakarta Sans"},
                    margin=dict(l=10, r=10, t=20, b=20),
                    height=280,
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                )
                st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

        # Table & CSV Export
        st.markdown("##### 📋 Scored Customer Cohort Table")
        cols_to_show = [
            c for c in ["customerID", "Contract", "tenure", "MonthlyCharges", "PaymentMethod", "Churn_Probability", "Risk_Category"]
            if c in proc_df.columns
        ]
        st.dataframe(
            proc_df[cols_to_show].sort_values(by="Churn_Probability", ascending=False),
            use_container_width=True,
            height=320,
        )

        csv_download = proc_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Complete Scored Report (.csv)",
            data=csv_download,
            file_name="retentioniq_churn_predictions.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )
    else:
        st.info("👆 Upload a CSV file or click 'Load Demo Batch' above to view instant portfolio risk analytics.")


# ==============================================================================
# TAB 3: MODEL TRANSPARENCY & WHAT-IF SIMULATOR
# ==============================================================================
with tab_insights:
    st.markdown("#### 🔬 Model Transparency & Interactive 'What-If' Simulation")
    st.caption("Inspect the mathematical drivers of churn attrition and test real-time interventions to observe risk score alterations.")

    # Feature Importance Plotly Horizontal Bar (Light Mode)
    top_features = eval_metrics.get("top_10_features", [])
    if top_features:
        st.markdown("##### 🔑 Top 10 Drivers of Customer Churn")
        df_feat = pd.DataFrame(top_features)
        df_feat["clean_label"] = (
            df_feat["feature"]
            .str.replace("cat__", "")
            .str.replace("num__", "")
            .str.replace("_", " ")
        )
        df_feat = df_feat.sort_values(by="importance_pct", ascending=True)

        feat_fig = px.bar(
            df_feat,
            x="importance_pct",
            y="clean_label",
            orientation="h",
            labels={"importance_pct": "Relative Impact (%)", "clean_label": "Feature"},
            color="importance_pct",
            color_continuous_scale=["#93C5FD", "#2563EB"],
        )
        feat_fig.update_layout(
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font={"color": "#0F172A", "family": "Plus Jakarta Sans"},
            margin=dict(l=10, r=20, t=10, b=20),
            height=360,
            xaxis=dict(showgrid=True, gridcolor="#E2E8F0", title="Relative Importance (%)"),
            yaxis=dict(showgrid=False, title=""),
            coloraxis_showscale=False,
        )
        st.plotly_chart(feat_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")
    # Interactive What-If Simulator
    st.markdown("##### 🧪 Interactive 'What-If' Intervention Simulator")
    st.markdown("Simulate how altering subscriber attributes (such as extending contract length, upgrading tech support, or applying billing discounts) directly reduces churn probability.")

    sim_col1, sim_col2 = st.columns([1.5, 1.5])

    with sim_col1:
        st.markdown(
            """
            <div class="card-title-light">
                <span>🎛️</span> Simulated Customer Adjustments
            </div>
            """,
            unsafe_allow_html=True,
        )
        sim_contract = st.select_slider(
            "Simulated Contract Commitment",
            options=["Month-to-month", "One year", "Two year"],
            value="Month-to-month",
        )
        sim_monthly_inr = st.slider("Monthly Invoice (₹)", min_value=1500, max_value=10000, value=int(95.0 * INR_RATE), step=250)
        sim_monthly = float(sim_monthly_inr) / INR_RATE
        sim_tech = st.radio("Tech Support Active", ["No", "Yes"], index=0, horizontal=True)
        sim_security = st.radio("Online Security Active", ["No", "Yes"], index=0, horizontal=True)
        sim_payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Credit card (automatic)", "Bank transfer (automatic)", "Mailed check"],
            index=0,
        )

    with sim_col2:
        # Baseline (Worst-case) vs Adjusted simulation
        baseline_customer = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.0,
            "TotalCharges": 190.0,
        }

        adjusted_customer = baseline_customer.copy()
        adjusted_customer.update({
            "Contract": sim_contract,
            "MonthlyCharges": sim_monthly,
            "TechSupport": sim_tech,
            "OnlineSecurity": sim_security,
            "PaymentMethod": sim_payment,
            "TotalCharges": sim_monthly * 2,
        })

        base_prob = float(pipeline.predict_proba(pd.DataFrame([baseline_customer]))[0, 1])
        adj_prob = float(pipeline.predict_proba(pd.DataFrame([adjusted_customer]))[0, 1])
        prob_delta = adj_prob - base_prob

        st.markdown(
            """
            <div class="card-title-light">
                <span>📉</span> Simulated Impact Comparison
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.metric(
            label="Baseline Month-to-Month Churn Risk",
            value=f"{base_prob * 100:.1f}%",
        )

        st.metric(
            label="Adjusted Churn Risk After Strategy",
            value=f"{adj_prob * 100:.1f}%",
            delta=f"{prob_delta * 100:.1f}%",
            delta_color="inverse",
        )

        if prob_delta < -0.20:
            st.success(f"🎯 **High Impact Strategy!** This bundle intervention reduces risk by {abs(prob_delta)*100:.1f} percentage points.")
        elif prob_delta < 0:
            st.info(f"👍 **Moderate Impact:** Reduces churn risk by {abs(prob_delta)*100:.1f} percentage points.")
        else:
            st.warning("⚠️ Intervention parameters do not sufficiently decrease subscriber risk.")

    st.markdown("---")
    # Diagnostic Deep-Dive (Confusion Matrix & ROC)
    st.markdown("##### 🖼️ Validation Diagnostics (Holdout Test Partition)")
    diag_c1, diag_c2 = st.columns(2)

    with diag_c1:
        cm_file = FIGURES_DIR / "confusion_matrix.png"
        if cm_file.exists():
            st.image(str(cm_file), caption="Test Confusion Matrix (XGBoost)", use_container_width=True)

    with diag_c2:
        roc_file = FIGURES_DIR / "roc_curve.png"
        if roc_file.exists():
            st.image(str(roc_file), caption="Test ROC-AUC Curve (AUC = 0.8439)", use_container_width=True)
