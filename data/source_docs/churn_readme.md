# Churn Prediction MLOps Pipeline
End-to-end ML project: data pipeline, model training, FastAPI serving, Docker deployment.

## Results

Baseline Logistic Regression achieved 80.4% accuracy, 0.60 F1-score, and 0.842 ROC-AUC on held-out test data. XGBoost was further tuned via Optuna (30-trial hyperparameter search) and tracked using MLflow.

## Explainability

SHAP was used to interpret individual model predictions beyond built-in feature importance. See notebooks/plots_shap_summary.png and notebooks/plots_feature_importance.png.

## API

The trained model is served via a FastAPI endpoint (/predict). Input: customer attributes (tenure, contract type, monthly charges, etc.). Output: churn_prediction (true/false) and churn_probability (0-1 confidence score).

## Docker

The API is containerized with Docker for portable deployment. See the Dockerfile in this repository.

## Tech Stack

Python, Scikit-learn, XGBoost, MLflow, Optuna, SHAP, FastAPI, Docker, Pandas

## Sample Predictions

Model tested on 3 synthetic customer profiles:

![Sample Predictions](notebooks/plots_sample_customer_predictions.png)

## Model Evaluation

![Confusion Matrix](notebooks/plots_confusion_matrix.png)
![ROC Curve](notebooks/plots_roc_curve.png)

## Feature Importance & Explainability

![Feature Importance](notebooks/plots_feature_importance.png)
![SHAP Summary](notebooks/plots_shap_summary.png)
