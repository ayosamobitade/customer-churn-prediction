import pandas as pd
from typing import Any, Dict
import joblib
import numpy as np

class ChurnPredictor:
    """
    Class to load trained model and perform churn predictions.
    """
    def __init__(self, model_path: str) -> None:
        """
        Initialize and load the trained model.

        Args:
            model_path (str): Path to the saved model file (.pkl).
        """
        self.model = joblib.load(model_path)

    def predict(self, input_data: Dict[str, Any]) -> int:
        """
        Predict churn for a single customer input.

        Args:
            input_data (Dict[str, Any]): Dictionary with customer features.

        Returns:
            int: 1 if churn predicted, 0 otherwise.
        """
        # Convert input dictionary to DataFrame
        df = pd.DataFrame([input_data])

        # Important: Make sure features order and encoding matches training preprocessing

        # You may need to preprocess the input data here exactly as training (encoding, scaling)
        # For simplicity, assume input_data is already preprocessed and ready

        prediction = self.model.predict(df)
        return int(prediction[0])

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict churn for a batch of customers.

        Args:
            df (pd.DataFrame): DataFrame with preprocessed customer data.

        Returns:
            np.ndarray: Array of predictions (1 or 0).
        """
        predictions = self.model.predict(df)
        return predictions

# Example usage (for testing only)
if __name__ == "__main__":
    model_path = "../app/model.pkl"
    predictor = ChurnPredictor(model_path)

    # Example of preprocessed customer data (feature names must match training)
    example_customer = {
        # Fill in all features exactly as in training data, example keys:
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
        # ... add all other one-hot encoded columns as 0 or 1
    }

    prediction = predictor.predict(example_customer)
    print(f"Predicted churn: {'Yes' if prediction == 1 else 'No'}")
