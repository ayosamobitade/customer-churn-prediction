import streamlit as st
import pandas as pd
import joblib
from typing import Dict

# Load the trained model
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return joblib.load(model_path)

# Preprocess user inputs into model-ready dataframe
def preprocess_input(user_input: Dict[str, any]) -> pd.DataFrame:
    """
    Convert and preprocess raw user inputs to model input features.

    Args:
        user_input (Dict[str, any]): Raw input from user.

    Returns:
        pd.DataFrame: Preprocessed input suitable for prediction.
    """
    df = pd.DataFrame([user_input])

    # Example preprocessing steps:
    # - Map Yes/No to 1/0 for binary fields
    # - One-hot encode categorical manually (or use same approach as training)
    # - Scale numeric features (if scaler saved, load and apply here)
    # For demo, assume user_input already has encoded numeric features except categorical

    # Map binary inputs
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Replace 'No internet service' with 'No' in certain fields
    no_internet_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in no_internet_cols:
        if df[col].iloc[0] == 'No internet service':
            df[col] = 0

    # One-hot encode categorical variables manually (Contract, PaymentMethod, InternetService, MultipleLines)
    # Initialize all dummy columns to 0
    dummies = {
        'Contract_One year': 0,
        'Contract_Two year': 0,
        'PaymentMethod_Credit card (automatic)': 0,
        'PaymentMethod_Electronic check': 0,
        'PaymentMethod_Mailed check': 0,
        'InternetService_Fiber optic': 0,
        'InternetService_No': 0,
        'MultipleLines_No': 0,
        'MultipleLines_Yes': 0,
    }
    # Set dummy variables based on input
    contract = user_input.get('Contract', '')
    if contract == 'One year':
        dummies['Contract_One year'] = 1
    elif contract == 'Two year':
        dummies['Contract_Two year'] = 1

    payment = user_input.get('PaymentMethod', '')
    if payment in ['Credit card (automatic)', 'Electronic check', 'Mailed check']:
        dummies[f'PaymentMethod_{payment}'] = 1

    internet = user_input.get('InternetService', '')
    if internet == 'Fiber optic':
        dummies['InternetService_Fiber optic'] = 1
    elif internet == 'No':
        dummies['InternetService_No'] = 1

    multiple_lines = user_input.get('MultipleLines', '')
    if multiple_lines == 'No':
        dummies['MultipleLines_No'] = 1
    elif multiple_lines == 'Yes':
        dummies['MultipleLines_Yes'] = 1

    # Drop original categorical columns from df if exist
    for cat_col in ['Contract', 'PaymentMethod', 'InternetService', 'MultipleLines']:
        if cat_col in df.columns:
            df.drop(columns=[cat_col], inplace=True)

    # Add dummy columns to df
    for k, v in dummies.items():
        df[k] = v

    # Convert tenure, MonthlyCharges, TotalCharges to float and optionally scale (if scaler saved, load & apply)
    df['tenure'] = float(df['tenure'])
    df['MonthlyCharges'] = float(df['MonthlyCharges'])
    df['TotalCharges'] = float(df['TotalCharges'])

    # For simplicity, assume model expects unscaled numeric inputs or scale externally
    return df

def main():
    st.title("Customer Churn Prediction")

    st.write(
        """
        Input customer details below to predict if they are likely to churn.
        """
    )

    # User inputs
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=850.0)

    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])

    user_input = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "PaperlessBilling": paperless_billing,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "Contract": contract,
        "PaymentMethod": payment_method,
    }

    model = load_model("model.pkl")  # Adjust path if needed

    if st.button("Predict Churn"):
        input_df = preprocess_input(user_input)
        prediction = model.predict(input_df)[0]
        result = "will churn" if prediction == 1 else "will NOT churn"
        st.success(f"Prediction: Customer {result}.")

if __name__ == "__main__":
    main()
