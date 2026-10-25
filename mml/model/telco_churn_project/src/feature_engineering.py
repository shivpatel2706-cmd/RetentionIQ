"""Feature Engineering Module for Telco Customer Churn.

Generates custom high-signal domain features including:
- Average_Monthly_Cost_Ratio = TotalCharges / (tenure + 1)
- Has_Streaming_Services = Binary flag combining StreamingTV and StreamingMovies
- Additional high-signal domain indicators (Is_Long_Term_Contract, Has_Security_Package, Charge_Discrepancy)

Includes a scikit-learn BaseEstimator transformer for seamless pipeline integration.
"""

from typing import Optional, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate high-signal engineered domain features for Telco Churn.

    Engineered Features:
    1. Average_Monthly_Cost_Ratio: TotalCharges / (tenure + 1)
       Captures consistency/spikes in historical monthly billing vs current charges.
    2. Has_Streaming_Services: Binary indicator (1 if StreamingTV == 'Yes' or StreamingMovies == 'Yes', else 0).
    3. Is_Long_Term_Contract: Binary indicator (1 if Contract is 'One year' or 'Two year', else 0).
    4. Has_Security_Package: Binary indicator (1 if customer has OnlineSecurity, OnlineBackup,
       DeviceProtection, or TechSupport).
    5. Monthly_Cost_Ratio_Discrepancy: MonthlyCharges - Average_Monthly_Cost_Ratio
       Measures recent price inflation experienced by the customer.

    Args:
        df: Input DataFrame containing raw/cleaned Telco customer attributes.

    Returns:
        pd.DataFrame with engineered features appended.
    """
    data = df.copy()

    # Ensure numeric columns in data are cleaned and recast to float64
    data["TotalCharges"] = pd.to_numeric(
        data.get("TotalCharges", 0.0), errors="coerce"
    ).fillna(0.0).astype("float64")
    data["tenure"] = pd.to_numeric(
        data.get("tenure", 0.0), errors="coerce"
    ).fillna(0.0).astype("float64")
    data["MonthlyCharges"] = pd.to_numeric(
        data.get("MonthlyCharges", 0.0), errors="coerce"
    ).fillna(0.0).astype("float64")

    # 1. Average_Monthly_Cost_Ratio = TotalCharges / (tenure + 1)
    data["Average_Monthly_Cost_Ratio"] = (
        data["TotalCharges"] / (data["tenure"] + 1.0)
    ).astype("float64")

    # 2. Has_Streaming_Services = Binary flag combining StreamingTV and StreamingMovies
    streaming_tv = data.get("StreamingTV", pd.Series("No", index=data.index)).astype(str)
    streaming_movies = data.get("StreamingMovies", pd.Series("No", index=data.index)).astype(str)
    data["Has_Streaming_Services"] = (
        ((streaming_tv == "Yes") | (streaming_movies == "Yes")).astype("int64")
    )

    # 3. Is_Long_Term_Contract: One year or Two year vs Month-to-month
    if "Contract" in data.columns:
        data["Is_Long_Term_Contract"] = (
            data["Contract"].astype(str).isin(["One year", "Two year"]).astype("int64")
        )
    else:
        data["Is_Long_Term_Contract"] = 0

    # 4. Has_Security_Package: Any cyber/device/tech support add-ons
    sec_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    has_sec = pd.Series(0, index=data.index)
    for col in sec_cols:
        if col in data.columns:
            has_sec = has_sec | (data[col].astype(str) == "Yes")
    data["Has_Security_Package"] = has_sec.astype("int64")

    # 5. Monthly_Cost_Ratio_Discrepancy
    monthly_charges = pd.to_numeric(
        data.get("MonthlyCharges", 0.0), errors="coerce"
    ).fillna(0.0).astype("float64")
    data["Monthly_Cost_Ratio_Discrepancy"] = (
        monthly_charges - data["Average_Monthly_Cost_Ratio"]
    ).astype("float64")

    return data


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-Learn Transformer for modular Telco Churn feature engineering.

    Compatible with scikit-learn Pipeline and ColumnTransformer.
    """

    def __init__(self, include_extra_domain_features: bool = True):
        self.include_extra_domain_features = include_extra_domain_features
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit transformer (records feature schema)."""
        transformed = engineer_domain_features(X)
        self.feature_names_ = transformed.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        transformed = engineer_domain_features(X)
        return transformed

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Get output feature names."""
        if self.feature_names_ is not None:
            return np.array(self.feature_names_, dtype=object)
        return np.array([], dtype=object)
