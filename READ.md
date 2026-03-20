# Customer Churn Prediction System

An end-to-end machine learning system to predict customer churn using the Telco Customer Churn dataset. Built with production-grade engineering practices including experiment tracking, containerization, and unit testing.

## Results

| Model | AUC-ROC | F1 Score |
|---|---|---|
| Logistic Regression | 0.8336 | 0.6141 |
| XGBoost | 0.8306 | 0.6148 |
| Random Forest | 0.8114 | 0.5737 |

## Tech Stack

- **ML**: scikit-learn, XGBoost, SHAP, imbalanced-learn
- **Experiment Tracking**: MLflow
- **Dashboard**: Streamlit
- **API**: Flask
- **Testing**: pytest (16 tests)
- **Containerization**: Docker
- **Config Management**: YAML

## Project Structure
```
churn-prediction/
├── src/
│   ├── data_prep.py       # Data cleaning, encoding, SMOTE
│   ├── train.py           # Model training with MLflow logging
│   └── evaluate.py        # Metrics, ROC curves, SHAP plots
├── tests/
│   └── test_pipeline.py   # 16 unit tests
├── data/                  # Dataset and generated plots
├── models/                # Saved model artifacts
├── app.py                 # Streamlit dashboard
├── config.yaml            # All hyperparameters and paths
├── Dockerfile             # Container definition
└── requirements.txt       # Dependencies
```

## Setup & Run

**Clone the repo:**
```bash
git clone https://github.com/Purvaladhav/churn-prediction.git
cd churn-prediction
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the full pipeline:**
```bash
python src/data_prep.py
python src/train.py
python src/evaluate.py
```

**Launch Streamlit dashboard:**
```bash
streamlit run app.py
```

**View MLflow experiment tracking:**
```bash
mlflow ui
```
Then open `http://localhost:5000`

**Run tests:**
```bash
pytest tests/ -v
```

**Run with Docker:**
```bash
docker build -t churn-prediction .
docker run -p 8501:8501 churn-prediction
```

## Key Features

- **SMOTE** to handle 73/27 class imbalance
- **SHAP values** for model explainability and feature importance
- **MLflow** for experiment tracking — every run logged with params and metrics
- **config.yaml** for centralized hyperparameter management
- **Dockerized** Streamlit app for reproducible deployment
- **16 pytest unit tests** covering data, models, and config

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 features, 26.5% churn rate.