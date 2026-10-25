"""Data Ingestion and Validation Module.

Provides robust functions for loading raw Telco Customer Churn datasets,
validating schemas, cleaning anomalies, and formatting target labels.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
]

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


def load_raw_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load raw dataset from CSV file path with path resolution.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        pd.DataFrame containing the raw ingested data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If file is empty or corrupted.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Raw data file not found at: {path.resolve()}")

    logger.info("Ingesting raw data from %s", path.resolve())
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Ingested dataset from {path.resolve()} is empty.")

    logger.info("Ingested %d rows and %d columns successfully.", df.shape[0], df.shape[1])
    return df


def clean_data(
    df: pd.DataFrame,
    median_total_charges: Optional[float] = None,
    is_training: bool = True,
) -> Tuple[pd.DataFrame, float]:
    """Clean data by handling TotalCharges anomalies, dropping identifiers,
    and encoding target variables.

    Steps:
    1. Drop customerID if present.
    2. Convert TotalCharges whitespace/empty strings to NaN, fill with median, recast to float64.
    3. Convert Churn ('Yes'/'No') to binary (1/0) if present.

    Args:
        df: Input raw or batch DataFrame.
        median_total_charges: Precomputed median for TotalCharges imputation (prevents leakage).
        is_training: Whether data is training data (computes and returns median).

    Returns:
        Tuple of (Cleaned DataFrame, Computed or used TotalCharges median).
    """
    data = df.copy()

    # 1. Drop non-predictive identifiers (e.g., customerID)
    if ID_COLUMN in data.columns:
        data = data.drop(columns=[ID_COLUMN])
        logger.info("Dropped non-predictive identifier: %s", ID_COLUMN)

    # 2. Clean TotalCharges: convert whitespace / non-numeric to NaN, fill with median, recast to float64
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"].astype(str).str.strip(), errors="coerce")
        if median_total_charges is None or is_training:
            computed_median = float(data["TotalCharges"].median())
            median_to_use = computed_median if not np.isnan(computed_median) else 0.0
        else:
            median_to_use = median_total_charges

        missing_count = int(data["TotalCharges"].isna().sum())
        if missing_count > 0:
            logger.info("Imputing %d missing TotalCharges values with median: %.2f", missing_count, median_to_use)
            data["TotalCharges"] = data["TotalCharges"].fillna(median_to_use)

        data["TotalCharges"] = data["TotalCharges"].astype("float64")
    else:
        median_to_use = 0.0

    # Ensure tenure and MonthlyCharges are numeric
    if "tenure" in data.columns:
        data["tenure"] = pd.to_numeric(data["tenure"], errors="coerce").fillna(0).astype("int64")
    if "MonthlyCharges" in data.columns:
        data["MonthlyCharges"] = pd.to_numeric(data["MonthlyCharges"], errors="coerce").fillna(0.0).astype("float64")

    # 3. Convert target variable Churn ('Yes'/'No') to binary (1/0) if present
    if TARGET_COLUMN in data.columns:
        if not pd.api.types.is_numeric_dtype(data[TARGET_COLUMN]):
            churn_mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0, "1": 1, "0": 0}
            data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(str).str.strip().map(churn_mapping)
        data[TARGET_COLUMN] = data[TARGET_COLUMN].astype("int64")
        logger.info(
            "Encoded target column '%s'. Class distribution: 0=%d, 1=%d",
            TARGET_COLUMN,
            int((data[TARGET_COLUMN] == 0).sum()),
            int((data[TARGET_COLUMN] == 1).sum()),
        )

    return data, median_to_use


def validate_schema(df: pd.DataFrame, require_target: bool = False) -> bool:
    """Validate that DataFrame contains required input features.

    Args:
        df: Input DataFrame to check.
        require_target: Whether Churn column must be present.

    Returns:
        True if validation succeeds.

    Raises:
        ValueError: If required columns are missing.
    """
    missing_features = [col for col in EXPECTED_FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Schema validation failed. Missing required feature columns: {missing_features}")

    if require_target and TARGET_COLUMN not in df.columns:
        raise ValueError(f"Schema validation failed. Required target '{TARGET_COLUMN}' is missing.")

    return True


def load_and_preprocess_raw(filepath: Union[str, Path]) -> Tuple[pd.DataFrame, pd.Series, float]:
    """Convenience function to load raw data, clean, validate, and separate
    features (X) and target (y).

    Args:
        filepath: Path to raw CSV file.

    Returns:
        Tuple of (Features DataFrame X, Target Series y, median_total_charges).
    """
    raw_df = load_raw_data(filepath)
    cleaned_df, median_tc = clean_data(raw_df, is_training=True)
    validate_schema(cleaned_df, require_target=True)

    X = cleaned_df.drop(columns=[TARGET_COLUMN])
    y = cleaned_df[TARGET_COLUMN]
    return X, y, median_tc
