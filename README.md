# Customer Churn Prediction System

## Overview

This project aims to predict customer churn — whether a customer will stop using a service — based on their account and usage data. Using machine learning techniques, the system analyzes customer behavior and churn patterns to help businesses proactively retain customers.

The project includes:
- Data exploration and visualization
- Feature engineering and preprocessing
- Model training with Random Forest and XGBoost
- A web app for live churn prediction

---

## Folder Structure

```bash
customer-churn-prediction/
│
├── data/
│ ├── raw/ # Original raw dataset files
│ └── processed/ # Cleaned and processed datasets
│
├── notebooks/
│ ├── 01_eda.ipynb # Exploratory Data Analysis and Visualization
│ ├── 02_feature_engineering.ipynb # Feature encoding, scaling, and preprocessing
│ └── 03_model_training.ipynb # Model training and evaluation
│
├── src/
│ ├── init.py
│ ├── data_preprocessing.py # Data loading, cleaning, encoding, scaling
│ ├── train_model.py # Model training and saving
│ └── predict.py # Model loading and prediction functions
│
├── app/
│ ├── app.py # Streamlit web app for prediction
│ ├── model.pkl # Saved trained model
│ └── requirements.txt # Python dependencies
│
├── tests/
│ └── test_model.py # Unit tests for preprocessing and prediction
│
├── .gitignore
├── README.md
├── LICENSE
└── setup.py

```

---

## Dataset

The dataset used is [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn), containing customer demographic information, account details, and churn labels.

The dataset CSV should be placed in `data/raw/telco_customer_churn.csv`.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:

```bash
git clone https://github.com/ayosamobitade/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```


3. Install dependencies:

```bash
pip install -r app/requirements.txt
```

4. Place the raw dataset CSV file in data/raw/.

## Usage
### Data Processing and Model Training

Run the notebooks sequentially:

- notebooks/01_eda.ipynb — Explore and visualize data.

- notebooks/02_feature_engineering.ipynb — Preprocess and engineer features.

- notebooks/03_model_training.ipynb — Train models and save the best one.

Alternatively, you can run scripts from the src/ folder for preprocessing and training.
Launch the Web App

Run the Streamlit app to interactively predict churn:
```bash
streamlit run app/app.py
```

This will open a web interface to input customer details and get a churn prediction.

## Technologies Used
- Python 3.8+
- Pandas, NumPy — Data manipulation
- Scikit-learn — Machine learning models and utilities
- XGBoost — Gradient boosting classifier
- Streamlit — Web application framework for deployment
- Joblib — Model serialization

## Testing

Run tests to verify functionality:
```bash
pytest tests/
```
## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or support, please contact [Ayobami Samuel Obitade](ayosamobitade@gmail.com).

## Acknowledgments
- Kaggle Telco Customer Churn Dataset
- Streamlit community for great UI tools
- Open source contributors

