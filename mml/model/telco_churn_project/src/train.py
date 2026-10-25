"""Model Training and Hyperparameter Tuning Module.

Trains and evaluates 3 competitive ML models:
1. Random Forest Classifier
2. XGBoost Classifier
3. LightGBM Classifier

Features:
- Handles target class imbalance using SMOTE on training folds only (via imblearn Pipeline)
- Performs 5-fold cross-validation optimizing for ROC-AUC and Recall
- Selects the champion model based on cross-validated ROC-AUC
- Serializes the winning pipeline artifact to models/best_churn_model.joblib
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier

from src.data_loader import load_and_preprocess_raw
from src.feature_engineering import DomainFeatureEngineer
from src.preprocessing import build_preprocessor, split_data, get_feature_names_from_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = PROJECT_ROOT / "models"


def get_model_search_spaces() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """Define estimators and targeted hyperparameter tuning search grids.

    Returns:
        Dictionary mapping model names to tuples of (estimator, parameter_grid).
    """
    models = {
        "RandomForest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [8, 12, None],
                "classifier__min_samples_split": [2, 5],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            {
                "classifier__n_estimators": [100, 150],
                "classifier__max_depth": [3, 5],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__subsample": [0.8, 1.0],
            },
        ),
        "LightGBM": (
            LGBMClassifier(
                random_state=42,
                verbosity=-1,
                n_jobs=-1,
            ),
            {
                "classifier__n_estimators": [100, 150],
                "classifier__max_depth": [4, 6],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__num_leaves": [15, 31],
            },
        ),
    }
    return models


def train_and_tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Tuple[str, ImbPipeline, Dict[str, Any]]:
    """Train competitive models using 5-fold cross validation with SMOTE oversampling,
    select the best model based on ROC-AUC, and evaluate on test data.

    Args:
        X_train: Training features DataFrame.
        y_train: Training target labels Series.
        X_test: Test features DataFrame.
        y_test: Test target labels Series.
        cv_folds: Number of Stratified K-Fold splits (default 5).
        random_state: Random state for reproducibility.

    Returns:
        Tuple of (Champion Model Name, Champion Fitted Pipeline, Detailed Results Dict).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models_config = get_model_search_spaces()

    results: Dict[str, Any] = {}
    best_cv_roc_auc = -1.0
    champion_name = ""
    champion_pipeline: Any = None

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "roc_auc": "roc_auc",
        "recall": "recall",
        "f1": "f1",
        "precision": "precision",
    }

    logger.info("Initiating 5-Fold Cross-Validation & Hyperparameter Tuning across 3 models...")

    for model_name, (base_estimator, param_grid) in models_config.items():
        logger.info("--> Tuning and Training: %s", model_name)

        # Construct full imbalanced-learn pipeline
        # Strictly applies SMOTE only within training folds
        pipeline = ImbPipeline(
            steps=[
                ("feature_engineer", DomainFeatureEngineer()),
                ("preprocessor", build_preprocessor()),
                ("smote", SMOTE(random_state=random_state)),
                ("classifier", base_estimator),
            ]
        )

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            refit="roc_auc",  # Optimizing champion selection for ROC-AUC
            n_jobs=-1,
            verbose=1,
            return_train_score=False,
        )

        grid_search.fit(X_train, y_train)

        best_score = float(grid_search.best_score_)
        best_params = grid_search.best_params_
        best_pipeline = grid_search.best_estimator_

        # Extract CV recall at best index
        best_idx = grid_search.best_index_
        cv_recall = float(grid_search.cv_results_["mean_test_recall"][best_idx])
        cv_f1 = float(grid_search.cv_results_["mean_test_f1"][best_idx])
        cv_precision = float(grid_search.cv_results_["mean_test_precision"][best_idx])

        # Test set evaluation
        y_test_pred = best_pipeline.predict(X_test)
        y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]

        test_roc_auc = float(roc_auc_score(y_test, y_test_prob))
        test_recall = float(recall_score(y_test, y_test_pred))
        test_precision = float(precision_score(y_test, y_test_pred))
        test_f1 = float(f1_score(y_test, y_test_pred))
        test_accuracy = float(accuracy_score(y_test, y_test_pred))

        logger.info(
            "%s: Best CV ROC-AUC=%.4f (CV Recall=%.4f) | Test ROC-AUC=%.4f, Test Recall=%.4f, Test F1=%.4f",
            model_name,
            best_score,
            cv_recall,
            test_roc_auc,
            test_recall,
            test_f1,
        )

        results[model_name] = {
            "cv_roc_auc": best_score,
            "cv_recall": cv_recall,
            "cv_f1": cv_f1,
            "cv_precision": cv_precision,
            "test_roc_auc": test_roc_auc,
            "test_recall": test_recall,
            "test_precision": test_precision,
            "test_f1": test_f1,
            "test_accuracy": test_accuracy,
            "best_params": {k.replace("classifier__", ""): v for k, v in best_params.items()},
        }

        # Save individual candidate model artifact
        candidate_path = MODELS_DIR / f"{model_name.lower()}_model.joblib"
        joblib.dump(best_pipeline, candidate_path)

        if best_score > best_cv_roc_auc:
            best_cv_roc_auc = best_score
            champion_name = model_name
            champion_pipeline = best_pipeline

    logger.info("=== CHAMPION SELECTION ===")
    logger.info("Overall Champion: %s with CV ROC-AUC: %.4f", champion_name, best_cv_roc_auc)

    # Save champion model artifact
    champion_path = MODELS_DIR / "best_churn_model.joblib"
    joblib.dump(champion_pipeline, champion_path)
    logger.info("Champion pipeline saved successfully to %s", champion_path.resolve())

    # Extract feature names after preprocessing
    fitted_preprocessor = champion_pipeline.named_steps["preprocessor"]
    transformed_feature_names = get_feature_names_from_preprocessor(fitted_preprocessor)

    metadata = {
        "champion_model": champion_name,
        "champion_cv_roc_auc": best_cv_roc_auc,
        "models_summary": results,
        "feature_names": transformed_feature_names,
    }

    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logger.info("Model metadata saved to %s", metadata_path.resolve())

    return champion_name, champion_pipeline, metadata


def main(data_path: Path = DEFAULT_DATA_PATH):
    """Execute complete end-to-end training workflow."""
    logger.info("Loading and validating raw data...")
    X, y, median_tc = load_and_preprocess_raw(data_path)

    # Save median total charges for prediction inference
    median_path = MODELS_DIR / "median_total_charges.json"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(median_path, "w", encoding="utf-8") as f:
        json.dump({"median_total_charges": median_tc}, f)

    logger.info("Splitting dataset into stratified Train and Test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42)

    # Save test partition for evaluation script and tests
    X_test_path = MODELS_DIR / "X_test.parquet"
    y_test_path = MODELS_DIR / "y_test.parquet"
    X_test.to_parquet(X_test_path)
    y_test.to_frame().to_parquet(y_test_path)

    train_and_tune_models(X_train, y_train, X_test, y_test)
    logger.info("Training and tuning workflow completed successfully.")


if __name__ == "__main__":
    main()
