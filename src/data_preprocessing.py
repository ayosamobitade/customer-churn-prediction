import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw data from CSV file.

    Args:
        filepath (str): Path to the raw CSV file.

    Returns:
        pd.DataFrame: Loaded raw dataframe.
    """
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe.

    - Convert 'TotalCharges' to numeric, coerce errors to NaN.
    - Fill missing 'TotalCharges' with median.
    - Strip whitespace in column names.

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df.columns = df.columns.str.strip()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    median_val = df['TotalCharges'].median()
    df['TotalCharges'].fillna(median_val, inplace=True)
    return df

def encode_binary_features(df: pd.DataFrame, binary_cols: List[str]) -> pd.DataFrame:
    """
    Encode binary categorical columns from 'Yes'/'No' to 1/0.

    Args:
        df (pd.DataFrame): Dataframe.
        binary_cols (List[str]): List of column names to encode.

    Returns:
        pd.DataFrame: Dataframe with encoded binary columns.
    """
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    return df

def encode_categorical_features(df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    """
    One-hot encode categorical columns with multiple categories.

    Args:
        df (pd.DataFrame): Dataframe.
        categorical_cols (List[str]): List of categorical columns.

    Returns:
        pd.DataFrame: Dataframe with one-hot encoded columns.
    """
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

def scale_numeric_features(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numeric columns using StandardScaler.

    Args:
        df (pd.DataFrame): Dataframe.
        numeric_cols (List[str]): List of numeric columns.

    Returns:
        Tuple[pd.DataFrame, StandardScaler]: Dataframe with scaled features and scaler object.
    """
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler

def preprocess_data(raw_filepath: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    - Load data
    - Clean data
    - Encode categorical features
    - Scale numeric features

    Args:
        raw_filepath (str): Path to raw CSV file.

    Returns:
        pd.DataFrame: Fully preprocessed dataframe ready for modeling.
    """
    df = load_data(raw_filepath)
    df = clean_data(df)

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Binary columns to encode
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

    # Columns with 'No internet service' to replace with 'No'
    replace_no_internet = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in replace_no_internet:
        df[col] = df[col].replace({'No internet service': 'No'})
    binary_cols += replace_no_internet

    # Encode binary features
    df = encode_binary_features(df, binary_cols)

    # Replace 'No phone service' with 'No' in 'MultipleLines'
    df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})

    # Categorical columns for one-hot encoding
    categorical_cols = ['InternetService', 'Contract', 'PaymentMethod', 'MultipleLines']

    df = encode_categorical_features(df, categorical_cols)

    # Scale numeric features
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df, scaler = scale_numeric_features(df, numeric_cols)

    return df
