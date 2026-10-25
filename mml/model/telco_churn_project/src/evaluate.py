"""Model Evaluation and Business Insights Module.

Evaluates trained pipeline on test partition, generates:
1. Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
2. Top 10 Feature Importance churn drivers.
3. Quantified Business Impact & Revenue Saved Analysis based on MonthlyCharges.
4. Publication-quality plots saved to reports/figures/.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def load_test_artifacts(
    models_dir: Path = MODELS_DIR,
) -> Tuple[Any, pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Load champion model artifact, test split, and metadata.

    Args:
        models_dir: Directory containing serialized model files.

    Returns:
        Tuple of (pipeline, X_test, y_test, metadata).
    """
    pipeline_path = models_dir / "best_churn_model.joblib"
    X_test_path = models_dir / "X_test.parquet"
    y_test_path = models_dir / "y_test.parquet"
    metadata_path = models_dir / "model_metadata.json"

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Champion model not found at {pipeline_path.resolve()}. Run train.py first.")

    pipeline = joblib.load(pipeline_path)
    X_test = pd.read_parquet(X_test_path)
    y_test = pd.read_parquet(y_test_path).iloc[:, 0]

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return pipeline, X_test, y_test, metadata


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Calculate key test classification metrics.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted positive class probabilities.

    Returns:
        Dictionary of computed metric scores.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }
    return metrics


def extract_feature_importance(
    pipeline: Any,
) -> pd.DataFrame:
    """Extract and sort feature importances from champion classifier.

    Args:
        pipeline: Fitted imblearn/sklearn pipeline.

    Returns:
        pd.DataFrame containing feature names and relative importances.
    """
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # Extract feature names
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"Feature_{i}" for i in range(100)]

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        logger.warning("Classifier does not expose feature importances directly.")
        return pd.DataFrame(columns=["feature", "importance"])

    # Ensure dimensions match
    min_len = min(len(feature_names), len(importances))
    df_importance = pd.DataFrame(
        {
            "feature": feature_names[:min_len],
            "importance": importances[:min_len],
        }
    ).sort_values(by="importance", ascending=False).reset_index(drop=True)

    # Normalize to percentage
    total_imp = df_importance["importance"].sum()
    if total_imp > 0:
        df_importance["importance_pct"] = (df_importance["importance"] / total_imp) * 100
    else:
        df_importance["importance_pct"] = 0.0

    return df_importance


def calculate_business_impact(
    X_test: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    retention_success_rate: float = 0.40,
    cost_per_outreach: float = 25.0,
    months_horizon: int = 12,
) -> Dict[str, Any]:
    """Quantify telecom revenue saved by proactive churn intervention.

    Assumptions & Formulas:
    - MonthlyCharges represents recurring monthly revenue per customer.
    - True Positives (TP): Churners correctly identified by model.
    - False Positives (FP): Non-churners falsely flagged (incur campaign cost).
    - Retention Success Rate (default 40%): Share of contacted churners successfully retained.
    - Total Revenue At Risk: Total annual spend of actual churners in test set.
    - Gross Revenue Saved: (Retained TP Customers) * MonthlyCharges * Horizon.
    - Campaign Outreach Cost: (TP + FP) * cost_per_outreach.
    - Net Revenue Saved: Gross Revenue Saved - Campaign Outreach Cost.
    - ROI (%): (Net Revenue Saved / Campaign Outreach Cost) * 100.

    Args:
        X_test: Test features DataFrame containing MonthlyCharges.
        y_true: True Churn labels.
        y_pred: Model predictions.
        retention_success_rate: Probability of successfully retaining an identified churner.
        cost_per_outreach: Dollar cost of targeted loyalty incentive or retention call.
        months_horizon: Projection horizon in months (default 12 for annual projection).

    Returns:
        Dictionary of financial metrics and business impact KPIs.
    """
    monthly_charges = X_test["MonthlyCharges"].values
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    total_customers = len(y_t)
    actual_churners = int((y_t == 1).sum())
    flagged_customers = int((y_p == 1).sum())

    # Confusion matrix breakdown
    tp_mask = (y_t == 1) & (y_p == 1)
    fp_mask = (y_t == 0) & (y_p == 1)
    fn_mask = (y_t == 1) & (y_p == 0)
    tn_mask = (y_t == 0) & (y_p == 0)

    tp_count = int(tp_mask.sum())
    fp_count = int(fp_mask.sum())
    fn_count = int(fn_mask.sum())
    tn_count = int(tn_mask.sum())

    # Revenue calculations
    total_annual_revenue_at_risk = float(np.sum(monthly_charges[y_t == 1]) * months_horizon)
    tp_annual_revenue_identified = float(np.sum(monthly_charges[tp_mask]) * months_horizon)

    gross_revenue_saved = float(tp_annual_revenue_identified * retention_success_rate)
    campaign_cost = float(flagged_customers * cost_per_outreach)
    net_revenue_saved = float(gross_revenue_saved - campaign_cost)
    roi_pct = float((net_revenue_saved / campaign_cost) * 100) if campaign_cost > 0 else 0.0

    impact = {
        "total_test_customers": total_customers,
        "actual_churners": actual_churners,
        "model_flagged_churners": flagged_customers,
        "true_positives": tp_count,
        "false_positives": fp_count,
        "false_negatives": fn_count,
        "true_negatives": tn_count,
        "retention_success_rate_pct": retention_success_rate * 100,
        "cost_per_outreach_usd": cost_per_outreach,
        "projection_months": months_horizon,
        "total_annual_revenue_at_risk_usd": round(total_annual_revenue_at_risk, 2),
        "tp_annual_revenue_identified_usd": round(tp_annual_revenue_identified, 2),
        "gross_revenue_saved_usd": round(gross_revenue_saved, 2),
        "campaign_cost_usd": round(campaign_cost, 2),
        "net_revenue_saved_usd": round(net_revenue_saved, 2),
        "roi_percentage": round(roi_pct, 2),
    }

    return impact


def generate_evaluation_plots(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    df_importance: pd.DataFrame,
    champion_name: str,
    output_dir: Path = FIGURES_DIR,
):
    """Render and save confusion matrix, ROC curve, and top 10 feature importance plots.

    Args:
        y_test: True test labels.
        y_pred: Predicted test labels.
        y_prob: Predicted test probabilities.
        df_importance: DataFrame of feature importances.
        champion_name: Name of winning model.
        output_dir: Directory to save figure images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Retained (0)", "Churned (1)"],
        yticklabels=["Retained (0)", "Churned (1)"],
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title(f"Confusion Matrix: {champion_name}", fontsize=14, weight="bold", pad=12)
    plt.ylabel("Actual Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info("Saved Confusion Matrix plot to %s", cm_path.resolve())

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(6.5, 5))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"{champion_name} (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.5, linestyle="--", label="Random Baseline (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.title(f"ROC Curve: {champion_name}", fontsize=14, weight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    roc_path = output_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    logger.info("Saved ROC Curve plot to %s", roc_path.resolve())

    # 3. Top 10 Feature Importance
    top10 = df_importance.head(10).copy()
    plt.figure(figsize=(8, 5.5))
    # Prettify labels
    clean_labels = [
        feat.replace("cat__", "").replace("num__", "").replace("_", " ")
        for feat in top10["feature"]
    ]
    ax = sns.barplot(
        x="importance_pct",
        y=clean_labels,
        data=top10,
        palette="viridis",
    )
    plt.title(f"Top 10 Churn Drivers ({champion_name})", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Relative Importance (%)", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(
            f"{width:.1f}%",
            (width + 0.3, p.get_y() + p.get_height() / 2.0),
            ha="left",
            va="center",
            fontsize=10,
            color="black",
        )
    plt.tight_layout()
    fi_path = output_dir / "top10_feature_importance.png"
    plt.savefig(fi_path, dpi=300)
    plt.close()
    logger.info("Saved Top 10 Feature Importance plot to %s", fi_path.resolve())


def evaluate_champion():
    """Main function to run complete evaluation and business impact analysis."""
    logger.info("Loading evaluation artifacts...")
    pipeline, X_test, y_test, metadata = load_test_artifacts()
    champion_name = metadata.get("champion_model", "Champion")

    # Generate predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = compute_metrics(y_test.values, y_pred, y_prob)
    logger.info("=== TEST EVALUATION METRICS ===")
    for k, v in metrics.items():
        logger.info("%s: %.4f", k.upper(), v)

    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))

    # Feature Importance
    df_importance = extract_feature_importance(pipeline)
    logger.info("=== TOP 10 CHURN DRIVERS ===")
    for idx, row in df_importance.head(10).iterrows():
        logger.info("%2d. %-35s : %.2f%%", idx + 1, row["feature"], row["importance_pct"])

    # Business Impact
    business_impact = calculate_business_impact(X_test, y_test, y_pred)
    logger.info("=== BUSINESS IMPACT & REVENUE PRESERVATION ===")
    logger.info("Total Annual Revenue at Risk : $%.2f", business_impact["total_annual_revenue_at_risk_usd"])
    logger.info("Revenue from Caught Churners : $%.2f", business_impact["tp_annual_revenue_identified_usd"])
    logger.info("Gross Annual Revenue Saved   : $%.2f", business_impact["gross_revenue_saved_usd"])
    logger.info("Retention Campaign Cost      : $%.2f", business_impact["campaign_cost_usd"])
    logger.info("Net Annual Revenue Saved     : $%.2f", business_impact["net_revenue_saved_usd"])
    logger.info("Projected ROI                : %.2f%%", business_impact["roi_percentage"])

    # Plots
    generate_evaluation_plots(y_test.values, y_pred, y_prob, df_importance, champion_name)

    # Save comprehensive evaluation summary
    eval_summary = {
        "champion_model": champion_name,
        "metrics": metrics,
        "business_impact": business_impact,
        "top_10_features": df_importance.head(10).to_dict(orient="records"),
    }

    eval_path = MODELS_DIR / "evaluation_metrics.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=4)
    logger.info("Evaluation metrics and business summary saved to %s", eval_path.resolve())

    return eval_summary


if __name__ == "__main__":
    evaluate_champion()
