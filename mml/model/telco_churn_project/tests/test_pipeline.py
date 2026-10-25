"""Unit and Integration Test Suite for Telco Churn Prediction System.

Tests cover:
- Data loader validation, cleaning, and schema checks
- Domain feature engineering calculations and outputs
- ColumnTransformer encoding, scaling, and output dimensions
- End-to-end inference with serialized champion pipeline
- Non-null predictions and valid probability bounds
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.data_loader import clean_data, validate_schema
from src.feature_engineering import DomainFeatureEngineer, engineer_domain_features
from src.preprocessing import build_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CHAMPION_MODEL_PATH = MODELS_DIR / "best_churn_model.joblib"


@pytest.fixture
def sample_customer_dict():
    """Provides a realistic single customer record."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 55.20,
        "TotalCharges": "662.40",
    }


@pytest.fixture
def sample_dataframe(sample_customer_dict):
    """Provides a sample multi-row DataFrame."""
    cust1 = sample_customer_dict.copy()
    cust2 = sample_customer_dict.copy()
    cust2.update({
        "gender": "Male",
        "tenure": 1,
        "TotalCharges": " ",  # test whitespace anomaly
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "StreamingTV": "Yes",
        "Churn": "Yes",
    })
    cust1["Churn"] = "No"
    return pd.DataFrame([cust1, cust2])


def test_clean_data_handles_whitespace_and_target(sample_dataframe):
    """Verify clean_data converts whitespace TotalCharges to median and maps Churn."""
    cleaned_df, median_tc = clean_data(sample_dataframe, is_training=True)

    assert not cleaned_df["TotalCharges"].isna().any()
    assert cleaned_df["TotalCharges"].dtype == "float64"
    assert cleaned_df.loc[1, "TotalCharges"] == median_tc
    assert list(cleaned_df["Churn"]) == [0, 1]
    assert cleaned_df["Churn"].dtype == "int64"


def test_schema_validation_success_and_failure(sample_dataframe):
    """Verify schema validator accepts valid features and rejects missing columns."""
    cleaned_df, _ = clean_data(sample_dataframe)
    assert validate_schema(cleaned_df, require_target=True) is True

    # Missing column should raise ValueError
    incomplete_df = cleaned_df.drop(columns=["tenure"])
    with pytest.raises(ValueError, match="Missing required feature columns"):
        validate_schema(incomplete_df)


def test_domain_feature_engineering_calculations(sample_dataframe):
    """Verify high-signal custom features are engineered accurately."""
    cleaned_df, _ = clean_data(sample_dataframe)
    engineered_df = engineer_domain_features(cleaned_df)

    assert "Average_Monthly_Cost_Ratio" in engineered_df.columns
    assert "Has_Streaming_Services" in engineered_df.columns
    assert "Is_Long_Term_Contract" in engineered_df.columns
    assert "Has_Security_Package" in engineered_df.columns
    assert "Monthly_Cost_Ratio_Discrepancy" in engineered_df.columns

    # Check Average_Monthly_Cost_Ratio = TotalCharges / (tenure + 1)
    expected_ratio_0 = cleaned_df.loc[0, "TotalCharges"] / (cleaned_df.loc[0, "tenure"] + 1.0)
    assert np.isclose(engineered_df.loc[0, "Average_Monthly_Cost_Ratio"], expected_ratio_0)

    # Check Has_Streaming_Services
    assert engineered_df.loc[0, "Has_Streaming_Services"] == 0
    assert engineered_df.loc[1, "Has_Streaming_Services"] == 1


def test_domain_feature_engineer_transformer(sample_dataframe):
    """Verify DomainFeatureEngineer scikit-learn transformer compatibility."""
    cleaned_df, _ = clean_data(sample_dataframe)
    transformer = DomainFeatureEngineer()
    transformed = transformer.fit_transform(cleaned_df)

    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape[1] > cleaned_df.shape[1]
    assert "Average_Monthly_Cost_Ratio" in transformer.get_feature_names_out()


def test_preprocessor_transformation_dimensions(sample_dataframe):
    """Verify ColumnTransformer scales and one-hot encodes into fixed 2D array."""
    cleaned_df, _ = clean_data(sample_dataframe)
    engineered_df = engineer_domain_features(cleaned_df)
    preprocessor = build_preprocessor()

    X_transformed = preprocessor.fit_transform(engineered_df)

    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.ndim == 2
    assert X_transformed.shape[0] == len(sample_dataframe)
    assert not np.isnan(X_transformed).any()
    assert X_transformed.shape[1] >= 10  # Numeric + one-hot columns for 2-row sample


def test_champion_model_file_exists():
    """Verify serialized champion pipeline artifact exists on disk."""
    assert CHAMPION_MODEL_PATH.exists(), f"Champion model not found at {CHAMPION_MODEL_PATH}"


def test_champion_pipeline_inference_and_bounds(sample_customer_dict):
    """Verify champion pipeline produces valid, non-null probabilities bounded in [0, 1]."""
    pipeline = joblib.load(CHAMPION_MODEL_PATH)
    input_df = pd.DataFrame([sample_customer_dict])

    preds = pipeline.predict(input_df)
    probs = pipeline.predict_proba(input_df)

    assert preds is not None
    assert len(preds) == 1
    assert preds[0] in [0, 1]

    assert probs is not None
    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 1] <= 1.0
    assert np.isclose(probs[0, 0] + probs[0, 1], 1.0)


def test_batch_prediction_consistency(sample_dataframe):
    """Verify batch predictions preserve ordering and return correct array length."""
    pipeline = joblib.load(CHAMPION_MODEL_PATH)
    preds = pipeline.predict(sample_dataframe)
    probs = pipeline.predict_proba(sample_dataframe)

    assert len(preds) == len(sample_dataframe)
    assert probs.shape == (len(sample_dataframe), 2)
    assert not np.isnan(preds).any()
    assert not np.isnan(probs).any()


def test_missing_and_corrupt_total_charges():
    """Verify inference pipeline survives corrupt/empty TotalCharges strings."""
    pipeline = joblib.load(CHAMPION_MODEL_PATH)
    corrupt_df = pd.DataFrame([{
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 0,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 20.0,
        "TotalCharges": "   ",  # whitespace
    }])

    prob = pipeline.predict_proba(corrupt_df)[0, 1]
    assert 0.0 <= prob <= 1.0
