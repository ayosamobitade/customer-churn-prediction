from setuptools import setup, find_packages

setup(
    name="customer_churn_prediction",
    version="0.1.0",
    author="Your Name",
    author_email="youremail@example.com",
    description="A machine learning project to predict customer churn and deploy a web app.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/customer-churn-prediction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0",
        "xgboost>=1.6.0",
        "joblib>=1.1.0",
        "streamlit>=1.18.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
