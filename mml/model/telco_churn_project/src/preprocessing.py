"""Data Preprocessing Pipeline Module.

Constructs modular Scikit-Learn transformers for scaling numeric features,
one-hot encoding categorical variables, and executing stratified data splits
to strictly prevent data leakage.
"""

from typing import List, Tuple, Optional
import logging
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import DomainFeatureEngineer

logger = logging.getLogger(__name__)

# Base feature types
NUMERICAL_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Average_Monthly_Cost_Ratio",
    "Monthly_Cost_Ratio_Discrepancy",
]

CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "Has_Streaming_Services",
    "Is_Long_Term_Contract",
    "Has_Security_Package",
]


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Execute stratified train-test split strictly BEFORE preprocessing
    transformations or SMOTE resampling to eliminate data leakage.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Proportion of dataset allocated to test set (default 0.20).
        random_state: Deterministic random seed for reproducibility.
        stratify: Whether to preserve class proportions across splits.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    stratify_target = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    logger.info(
        "Stratified Train-Test split executed: Train shape=%s (churn rate=%.2f%%), Test shape=%s (churn rate=%.2f%%)",
        X_train.shape,
        float(y_train.mean()) * 100,
        X_test.shape,
        float(y_test.mean()) * 100,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Build a ColumnTransformer that scales numeric features with StandardScaler
    and encodes categorical features using OneHotEncoder(drop='first', handle_unknown='ignore').

    Args:
        num_cols: List of numeric column names.
        cat_cols: List of categorical column names.

    Returns:
        Configured scikit-learn ColumnTransformer.
    """
    numerical_features = num_cols if num_cols is not None else NUMERICAL_COLS
    categorical_features = cat_cols if cat_cols is not None else CATEGORICAL_COLS

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="first",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_full_preprocessing_pipeline(
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
) -> Pipeline:
    """Combine DomainFeatureEngineer and ColumnTransformer into a unified
    reusable preprocessing pipeline.

    Args:
        num_cols: List of numeric columns.
        cat_cols: List of categorical columns.

    Returns:
        Unified scikit-learn Pipeline.
    """
    preprocessor = build_preprocessor(num_cols=num_cols, cat_cols=cat_cols)
    full_pipeline = Pipeline(
        steps=[
            ("feature_engineer", DomainFeatureEngineer()),
            ("preprocessor", preprocessor),
        ]
    )
    return full_pipeline


def get_feature_names_from_preprocessor(
    preprocessor: ColumnTransformer,
    feature_engineer: Optional[DomainFeatureEngineer] = None,
    sample_df: Optional[pd.DataFrame] = None,
) -> List[str]:
    """Retrieve output feature names after ColumnTransformer transformation.

    Args:
        preprocessor: Fitted ColumnTransformer instance.
        feature_engineer: Optional fitted DomainFeatureEngineer.
        sample_df: Optional sample dataframe.

    Returns:
        List of generated feature names.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback manual reconstruction if not supported
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if trans == "drop" or not cols:
                continue
            if hasattr(trans, "get_feature_names_out"):
                names.extend(trans.get_feature_names_out(cols))
            elif hasattr(trans, "named_steps") and hasattr(trans.named_steps.get("onehot", None), "get_feature_names_out"):
                names.extend(trans.named_steps["onehot"].get_feature_names_out(cols))
            else:
                names.extend(cols)
        return names
