import pandas as pd
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import os

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load preprocessed data from CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    return pd.read_csv(filepath)

def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features (X) and target (y).

    Args:
        df (pd.DataFrame): Input dataframe.
        target_col (str): Name of target column.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        RandomForestClassifier: Trained model.
    """
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Evaluate the model and print metrics.

    Args:
        model (RandomForestClassifier): Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def save_model(model: RandomForestClassifier, filepath: str) -> None:
    """
    Save the trained model to a file.

    Args:
        model (RandomForestClassifier): Trained model.
        filepath (str): Path to save the model file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

if __name__ == "__main__":
    processed_data_path = "../data/processed/processed_data.csv"
    model_save_path = "../app/model.pkl"
    
    # Load preprocessed data
    df = load_data(processed_data_path)
    
    # Split features and target
    X, y = split_features_target(df, target_col='Churn')
    
    # Split into train/test sets (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the model
    print("Training Random Forest model...")
    model = train_random_forest(X_train, y_train)
    
    # Evaluate the model
    print("Evaluating model on test data...")
    evaluate_model(model, X_test, y_test)
    
    # Save the trained model
    save_model(model, model_save_path)
