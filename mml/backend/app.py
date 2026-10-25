"""RetentionIQ™ Flask Backend API.

Loads the XGBoost champion churn prediction pipeline and exposes REST endpoints
for single-customer prediction, batch inference, model metadata, and evaluation metrics.
"""

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow React dev-server requests

# Paths to model artifacts
BASE_DIR = Path(__file__).resolve().parent.parent / "model" / "telco_churn_project"
sys.path.insert(0, str(BASE_DIR))

MODELS_DIR = BASE_DIR / "models"
PIPELINE_PATH = MODELS_DIR / "best_churn_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
EVAL_PATH = MODELS_DIR / "evaluation_metrics.json"
MEDIAN_PATH = MODELS_DIR / "median_total_charges.json"
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# ──────────────────────────────────────────────
# LOAD MODEL ARTIFACTS ON STARTUP
# ──────────────────────────────────────────────
print(f"[RetentionIQ API] Adding to sys.path: {BASE_DIR}")
print(f"[RetentionIQ API] Loading pipeline from: {PIPELINE_PATH}")

if not PIPELINE_PATH.exists():
    print(f"[ERROR] Pipeline not found at {PIPELINE_PATH}")
    sys.exit(1)

pipeline = joblib.load(PIPELINE_PATH)

metadata = {}
if METADATA_PATH.exists():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

eval_metrics = {}
if EVAL_PATH.exists():
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_metrics = json.load(f)

default_median_tc = 1397.475
if MEDIAN_PATH.exists():
    with open(MEDIAN_PATH, "r", encoding="utf-8") as f:
        default_median_tc = json.load(f).get("median_total_charges", 1397.475)

print("[RetentionIQ API] [OK] All artifacts loaded successfully.")


INR_RATE = 83.0  # Standard USD to INR conversion rate

# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────
def generate_recommendations(customer: dict, prob: float) -> list:
    """Generate prescriptive retention strategies based on customer profile."""
    recs = []

    if customer.get("Contract") == "Month-to-month":
        recs.append({
            "icon": "🎯",
            "title": "Annual Contract Transition Incentive",
            "description": "Offer a guaranteed 15% rate freeze or ₹800/month bill credit in exchange for a 1-year contract commitment."
        })

    if customer.get("InternetService") == "Fiber optic" and customer.get("TechSupport") == "No":
        recs.append({
            "icon": "🛠️",
            "title": "VIP Tech Support Onboarding",
            "description": "Bundle 3 months of complimentary 24/7 Priority Tech Support to resolve fiber optic latency and satisfaction issues."
        })

    if customer.get("PaymentMethod") == "Electronic check":
        recs.append({
            "icon": "💳",
            "title": "Automated ACH/Card Incentive",
            "description": "Provide an immediate ₹1,250 bill credit for activating auto-pay with a credit card or direct bank transfer."
        })

    if customer.get("OnlineSecurity") == "No" or customer.get("OnlineBackup") == "No":
        recs.append({
            "icon": "🛡️",
            "title": "Cyber Shield Add-on Trial",
            "description": "Deploy a 60-day complimentary Digital Security & Cloud Backup package to increase service stickiness."
        })

    if int(customer.get("tenure", 0)) <= 6:
        recs.append({
            "icon": "🤝",
            "title": "Early-Tenure Onboarding Concierge",
            "description": "Enroll customer in the 90-day Customer Success check-in cadence to mitigate early tenure churn."
        })

    if not recs:
        recs.append({
            "icon": "🌟",
            "title": "Loyalty Rewards Tier Upgrade",
            "description": "Customer exhibits loyal behavior; provide a VIP loyalty points milestone to maintain long-term advocacy."
        })

    return recs[:3]


# ──────────────────────────────────────────────
# API ENDPOINTS
# ──────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model": metadata.get("champion_model", "XGBoost"),
        "version": "1.0.0"
    })


@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    """Return model metadata and summary of all trained models."""
    return jsonify(metadata)


@app.route("/api/evaluation", methods=["GET"])
def get_evaluation():
    """Return evaluation metrics, business impact, and feature importances."""
    return jsonify(eval_metrics)


@app.route("/api/predict", methods=["POST"])
def predict_single():
    """Predict churn probability for a single customer.

    Expects JSON body with customer attributes matching the model's expected features.
    Accepts MonthlyCharges in USD or INR (if > 200, automatically converts to USD for model inference).
    Returns churn probability, risk category, financial metrics in both USD and INR, and recommendations.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        model_data = data.copy()

        # Handle currency conversion if user submitted charges in INR
        raw_monthly = float(model_data.get("MonthlyCharges", 0))
        raw_total = model_data.get("TotalCharges", 0)

        if raw_monthly > 200.0:  # Value entered in Rupees (INR)
            monthly_usd = raw_monthly / INR_RATE
            model_data["MonthlyCharges"] = monthly_usd
        else:
            monthly_usd = raw_monthly

        if "TotalCharges" in model_data:
            try:
                tc = float(str(model_data["TotalCharges"]).strip())
                if tc > 10000.0 or (raw_monthly > 200.0 and tc > 200.0):
                    model_data["TotalCharges"] = tc / INR_RATE
                else:
                    model_data["TotalCharges"] = tc
            except (ValueError, TypeError):
                model_data["TotalCharges"] = default_median_tc

        input_df = pd.DataFrame([model_data])
        churn_prob = float(pipeline.predict_proba(input_df)[0, 1])
        is_high_risk = churn_prob >= 0.50
        tenure_val = int(model_data.get("tenure", 0))

        # Compute engineered signals
        avg_ratio = float(model_data.get("TotalCharges", 0)) / (tenure_val + 1)
        discrepancy = monthly_usd - avg_ratio
        has_streaming = model_data.get("StreamingTV") == "Yes" or model_data.get("StreamingMovies") == "Yes"

        recommendations = generate_recommendations(model_data, churn_prob)

        return jsonify({
            "status": "success",
            "churn_probability": round(churn_prob, 4),
            "churn_percentage": round(churn_prob * 100, 1),
            "churn_probability_pct": round(churn_prob * 100, 1),
            "prediction": 1 if is_high_risk else 0,
            "is_high_risk": is_high_risk,
            "risk_level": "High Risk" if is_high_risk else "Low Risk",
            "risk_category": "High Risk" if is_high_risk else "Low Risk",
            "threshold": 0.50,
            "monthly_spend": round(monthly_usd, 2),
            "monthly_spend_inr": round(monthly_usd * INR_RATE, 2),
            "annual_revenue_at_risk": round(monthly_usd * 12, 2),
            "annual_revenue_at_risk_inr": round(monthly_usd * 12 * INR_RATE, 2),
            "avg_monthly_cost_ratio": round(avg_ratio, 2),
            "cost_discrepancy": round(discrepancy, 2),
            "has_streaming": has_streaming,
            "recommendations": recommendations,
            "recommendation": recommendations[0]["description"] if recommendations else "Customer is in good standing."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    """Batch prediction endpoint.

    Accepts JSON array of customer records.
    Returns scored records with aggregated portfolio metrics in USD and INR.
    """
    try:
        data = request.get_json()
        if not data or not isinstance(data, list):
            return jsonify({"error": "Expected a JSON array of customer records"}), 400

        batch_df = pd.DataFrame(data)
        pred_df = batch_df.copy()
        for drop_col in ["customerID", "Churn", "prediction", "churn_probability_pct", "risk_level"]:
            if drop_col in pred_df.columns:
                pred_df = pred_df.drop(columns=[drop_col])

        # Clean TotalCharges and convert if INR
        if "TotalCharges" in pred_df.columns:
            pred_df["TotalCharges"] = pd.to_numeric(
                pred_df["TotalCharges"].astype(str).str.strip(), errors="coerce"
            ).fillna(default_median_tc)

        # Predict
        probs = pipeline.predict_proba(pred_df)[:, 1]

        batch_df["prediction"] = np.where(probs >= 0.50, 1, 0)
        batch_df["churn_probability_pct"] = np.round(probs * 100, 1)
        batch_df["risk_level"] = np.where(probs >= 0.50, "High Risk", "Low Risk")
        batch_df["Risk_Category"] = batch_df["risk_level"]
        batch_df["Churn_Probability"] = np.round(probs, 4)
        if "MonthlyCharges" in batch_df.columns:
            batch_df["MonthlyChargesINR"] = np.round(batch_df["MonthlyCharges"].astype(float) * INR_RATE, 0)

        total = len(batch_df)
        high_risk_count = int((batch_df["prediction"] == 1).sum())
        churn_rate = (high_risk_count / total) * 100 if total > 0 else 0
        monthly_at_risk = float(batch_df.loc[batch_df["prediction"] == 1, "MonthlyCharges"].sum()) if "MonthlyCharges" in batch_df.columns else 0.0
        annual_at_risk = monthly_at_risk * 12
        saved_est = annual_at_risk * 0.40 - (high_risk_count * 25.0)

        # INR conversions
        monthly_at_risk_inr = round(monthly_at_risk * INR_RATE, 2)
        annual_at_risk_inr = round(annual_at_risk * INR_RATE, 2)
        saved_est_inr = round(max(0.0, saved_est) * INR_RATE, 2)

        # Contract distribution for charts
        contract_risk = {}
        if "Contract" in batch_df.columns:
            contract_groups = batch_df.groupby(["Contract", "risk_level"]).size().reset_index(name="count")
            for _, row in contract_groups.iterrows():
                contract = row["Contract"]
                if contract not in contract_risk:
                    contract_risk[contract] = {"High Risk": 0, "Low Risk": 0}
                contract_risk[contract][row["risk_level"]] = int(row["count"])

        records = json.loads(batch_df.to_json(orient="records"))

        return jsonify({
            "status": "success",
            "total_analyzed": total,
            "high_risk_count": high_risk_count,
            "churn_rate_pct": round(churn_rate, 1),
            "monthly_at_risk": round(monthly_at_risk, 2),
            "annual_at_risk": round(annual_at_risk, 2),
            "estimated_savings": round(max(0, saved_est), 2),
            "monthly_at_risk_inr": monthly_at_risk_inr,
            "annual_at_risk_inr": annual_at_risk_inr,
            "estimated_savings_inr": saved_est_inr,
            "contract_risk_distribution": contract_risk,
            "records": records,
            "data": records
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/demo-batch", methods=["GET"])
def demo_batch():
    """Load first 100 records from the raw dataset for demonstration."""
    try:
        if not RAW_DATA_PATH.exists():
            return jsonify({"error": "Demo dataset not found"}), 404

        df = pd.read_csv(RAW_DATA_PATH).head(100)
        display_df = df.copy()

        pred_df = df.copy()
        if "customerID" in pred_df.columns:
            pred_df = pred_df.drop(columns=["customerID"])
        if "Churn" in pred_df.columns:
            pred_df = pred_df.drop(columns=["Churn"])

        # Clean TotalCharges
        if "TotalCharges" in pred_df.columns:
            pred_df["TotalCharges"] = pd.to_numeric(
                pred_df["TotalCharges"].astype(str).str.strip(), errors="coerce"
            ).fillna(default_median_tc)

        probs = pipeline.predict_proba(pred_df)[:, 1]

        display_df["prediction"] = np.where(probs >= 0.50, 1, 0)
        display_df["churn_probability_pct"] = np.round(probs * 100, 1)
        display_df["risk_level"] = np.where(probs >= 0.50, "High Risk", "Low Risk")
        display_df["Risk_Category"] = display_df["risk_level"]
        display_df["Churn_Probability"] = np.round(probs, 4)
        display_df["MonthlyChargesINR"] = np.round(display_df["MonthlyCharges"].astype(float) * INR_RATE, 0)

        total = len(display_df)
        high_risk_count = int((display_df["prediction"] == 1).sum())
        churn_rate = (high_risk_count / total) * 100 if total > 0 else 0
        monthly_at_risk = float(display_df.loc[display_df["prediction"] == 1, "MonthlyCharges"].sum())
        annual_at_risk = monthly_at_risk * 12
        saved_est = annual_at_risk * 0.40 - (high_risk_count * 25.0)

        monthly_at_risk_inr = round(monthly_at_risk * INR_RATE, 2)
        annual_at_risk_inr = round(annual_at_risk * INR_RATE, 2)
        saved_est_inr = round(max(0.0, saved_est) * INR_RATE, 2)

        contract_risk = {}
        if "Contract" in display_df.columns:
            contract_groups = display_df.groupby(["Contract", "risk_level"]).size().reset_index(name="count")
            for _, row in contract_groups.iterrows():
                contract = row["Contract"]
                if contract not in contract_risk:
                    contract_risk[contract] = {"High Risk": 0, "Low Risk": 0}
                contract_risk[contract][row["risk_level"]] = int(row["count"])

        records = json.loads(display_df.to_json(orient="records"))

        return jsonify({
            "status": "success",
            "total_analyzed": total,
            "high_risk_count": high_risk_count,
            "churn_rate_pct": round(churn_rate, 1),
            "monthly_at_risk": round(monthly_at_risk, 2),
            "annual_at_risk": round(annual_at_risk, 2),
            "estimated_savings": round(max(0, saved_est), 2),
            "monthly_at_risk_inr": monthly_at_risk_inr,
            "annual_at_risk_inr": annual_at_risk_inr,
            "estimated_savings_inr": saved_est_inr,
            "contract_risk_distribution": contract_risk,
            "records": records,
            "data": records
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/simulate", methods=["POST"])
def simulate_whatif():
    """What-If simulator endpoint.

    Compares a baseline customer against an adjusted/intervened profile
    and returns both churn probabilities, risk levels, and risk reduction delta.
    """
    try:
        data = request.get_json() or {}

        # Default at-risk customer profile
        default_baseline = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 6,
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
            "MonthlyCharges": 90.0,
            "TotalCharges": 540.0,
        }

        # Extract baseline and intervention from either nested or flat payload
        req_baseline = data.get("baseline", {})
        baseline_customer = default_baseline.copy()
        if isinstance(req_baseline, dict):
            baseline_customer.update(req_baseline)

        # Currency conversion for baseline if passed in INR
        if float(baseline_customer.get("MonthlyCharges", 0)) > 200.0:
            baseline_customer["MonthlyCharges"] = float(baseline_customer["MonthlyCharges"]) / INR_RATE
        if float(baseline_customer.get("TotalCharges", 0)) > 10000.0:
            baseline_customer["TotalCharges"] = float(baseline_customer["TotalCharges"]) / INR_RATE

        req_intervention = data.get("intervention", {})
        adjusted_customer = baseline_customer.copy()

        # Apply interventions (support both PascalCase and camelCase / snake_case)
        if isinstance(req_intervention, dict) and req_intervention:
            for k, v in req_intervention.items():
                adjusted_customer[k] = v
        else:
            # Fallback to direct keys on data root
            if "contract" in data:
                adjusted_customer["Contract"] = data["contract"]
            if "Contract" in data:
                adjusted_customer["Contract"] = data["Contract"]
            if "tech_support" in data:
                adjusted_customer["TechSupport"] = data["tech_support"]
            if "TechSupport" in data:
                adjusted_customer["TechSupport"] = data["TechSupport"]
            if "monthly_charges" in data:
                adjusted_customer["MonthlyCharges"] = float(data["monthly_charges"])
            if "MonthlyCharges" in data:
                adjusted_customer["MonthlyCharges"] = float(data["MonthlyCharges"])

        # Currency conversion for adjusted if in INR
        if float(adjusted_customer.get("MonthlyCharges", 0)) > 200.0:
            adjusted_customer["MonthlyCharges"] = float(adjusted_customer["MonthlyCharges"]) / INR_RATE
        if float(adjusted_customer.get("TotalCharges", 0)) > 10000.0:
            adjusted_customer["TotalCharges"] = float(adjusted_customer["TotalCharges"]) / INR_RATE

        base_df = pd.DataFrame([baseline_customer])
        adj_df = pd.DataFrame([adjusted_customer])

        base_prob = float(pipeline.predict_proba(base_df)[0, 1])
        adj_prob = float(pipeline.predict_proba(adj_df)[0, 1])

        # Positive delta_pct indicates risk reduction
        delta_pct = round((base_prob - adj_prob) * 100, 1)
        impact_level = "high" if delta_pct >= 20.0 else ("moderate" if delta_pct > 0 else "low")

        base_monthly = float(baseline_customer.get("MonthlyCharges", 90.0))
        adj_monthly = float(adjusted_customer.get("MonthlyCharges", 75.0))

        return jsonify({
            "status": "success",
            "baseline": {
                "churn_probability": round(base_prob, 4),
                "churn_probability_pct": round(base_prob * 100, 1),
                "risk_level": "High Risk" if base_prob >= 0.50 else "Low Risk",
                "monthly_charges": round(base_monthly, 2),
                "monthly_charges_inr": round(base_monthly * INR_RATE, 0)
            },
            "simulated": {
                "churn_probability": round(adj_prob, 4),
                "churn_probability_pct": round(adj_prob * 100, 1),
                "risk_level": "High Risk" if adj_prob >= 0.50 else "Low Risk",
                "monthly_charges": round(adj_monthly, 2),
                "monthly_charges_inr": round(adj_monthly * INR_RATE, 0)
            },
            "delta_pct": delta_pct,
            "impact_level": impact_level,
            # Backward compatibility fields
            "baseline_probability": round(base_prob, 4),
            "baseline_percentage": round(base_prob * 100, 1),
            "adjusted_probability": round(adj_prob, 4),
            "adjusted_percentage": round(adj_prob * 100, 1),
            "delta": round(adj_prob - base_prob, 4),
            "delta_percentage": round((adj_prob - base_prob) * 100, 1)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
