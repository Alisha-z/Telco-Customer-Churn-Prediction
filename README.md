# Telco-Customer-Churn-Prediction

# 📊 Telco Customer Churn Prediction – End-to-End ML Pipeline

## 📌 Project Overview
This project implements an **end-to-end Machine Learning pipeline** for predicting customer churn using the **Telco Customer Churn dataset**. The pipeline is built with **Scikit-learn’s Pipeline API**, making it **reusable, modular, and production-ready**.  

The model predicts whether a customer is likely to churn (leave the service) based on demographics, account details, and service usage.  

---

## 🚀 Features
- **Data Preprocessing with Pipeline**
  - Handling missing values
  - Scaling numerical features with `StandardScaler`
  - Encoding categorical features with `OneHotEncoder`
- **Models Implemented**
  - Logistic Regression
  - Random Forest Classifier
- **Hyperparameter Tuning**
  - `GridSearchCV` with cross-validation
- **Model Export**
  - Final pipeline (preprocessing + model) saved using `joblib`

---

## 🛠️ Tech Stack & Tools
- **Programming Language:** Python 3.x  
- **Libraries & Frameworks:**
  - `pandas`, `numpy` → data handling
  - `scikit-learn` → preprocessing, ML models, pipeline, grid search
  - `joblib` → model persistence
- **Dataset:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 📂 Project Structure

TelcoChurnPipeline/
│── churn_pipeline.py # Main script for pipeline, training & saving model
│── telco_churn.csv # Dataset
│── churn_model.pkl # Saved ML pipeline (after training)
│── README.md # Project documentation

