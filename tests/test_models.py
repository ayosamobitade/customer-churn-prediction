import pytest
import pandas as pd
from src import data_preprocessing, predict
from sklearn.ensemble import RandomForestClassifier
from typing import Dict

@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    # Minimal raw sample data similar to raw telco dataset structure
    data = {
        'customerID': ['0001'],
        'gender': ['Female'],
        'SeniorCitizen': [0],
        'Partner': ['Yes'],
        'Dependents': ['No'],
        'tenure': [12],
        'PhoneService': ['Yes'],
        'MultipleLines': ['No'],
        'InternetService': ['DSL'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['Yes'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['No'],
        'StreamingMovies': ['No'],
        'Contract': ['Month-to-month'],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'],
        'MonthlyCharges': [70.35],
        'TotalCharges': ['845.5'],
        'Churn': ['No']
    }
    return pd.DataFrame(data)

def test_preprocess_data(sample_raw_data: pd.DataFrame) -> None:
    df_processed = data_preprocessing.preprocess_data_from_df(sample_raw_data)
    # Check 'Churn' is encoded as 0 or 1
    assert set(df_processed['Churn'].unique()).issubset({0, 1})
    # Check no missing values in numeric columns
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        assert not df_processed[col].isnull().any()
    # Check binary columns are encoded 0/1
    for col in ['Partner', 'Dependents', 'PhoneService']:
        assert set(df_processed[col].unique()).issubset({0, 1})

def test_predictor_predict() -> None:
    # Create dummy model that always predicts 1
    class DummyModel:
        def predict(self, X):
            return [1] * len(X)

    dummy_predictor = predict.ChurnPredictor.__new__(predict.ChurnPredictor)
    dummy_predictor.model = DummyModel()

    sample_input = {
        'tenure': 12.0,
        'MonthlyCharges': 70.35,
        'TotalCharges': 845.5,
        'Partner': 1,
        'Dependents': 0,
        'PhoneService': 1,
        'PaperlessBilling': 1,
        'OnlineSecurity': 0,
        'OnlineBackup': 1,
        'DeviceProtection': 0,
        'TechSupport': 0,
        'StreamingTV': 0,
        'StreamingMovies': 0,
        'MultipleLines_No': 1,
        'MultipleLines_Yes': 0,
        'InternetService_Fiber optic': 1,
        'InternetService_No': 0,
        'Contract_One year': 0,
        'Contract_Two year': 0,
        'PaymentMethod_Credit card (automatic)': 0,
        'PaymentMethod_Electronic check': 1,
        'PaymentMethod_Mailed check': 0,
    }
    prediction = dummy_predictor.predict(sample_input)
    assert prediction == 1

