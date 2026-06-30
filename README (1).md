# 💰 Loan Default Risk Predictor

A machine learning project that predicts whether a loan applicant is likely to default, built as part of a Credit Risk & Loan Default Prediction capstone project. The model is trained on historical loan data and deployed as an interactive web app.

## 🚀 Live App

👉 **[Try the live app here](https://loan-default-predictor-nx8ce8oeohtthwtsf3tpsg.streamlit.app/)**

Enter an applicant's details (age, income, loan amount, interest rate, loan grade, etc.) and instantly get a default risk prediction with probability score.

## 📊 Project Overview

Banks need a fast, consistent way to flag high-risk loan applications instead of relying purely on manual review. This project builds a supervised classification model to predict `loan_status` (0 = Paid, 1 = Default) using applicant financial and demographic data.

**Dataset:** [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (Kaggle) — 32,581 loan applications, 12 features.

## 🔧 What Was Done

1. **Data Cleaning** — Imputed missing values (`person_emp_length`, `loan_int_rate`) with median; removed illogical outliers (e.g. age 144, employment length 123 years).
2. **EDA** — Analyzed how income, interest rate, and loan grade relate to default rate.
3. **Class Imbalance Handling** — Used `class_weight='balanced'` since only ~22% of loans default.
4. **Feature Encoding** — One-Hot Encoding for categorical columns (loan_intent, loan_grade, home_ownership, etc.).
5. **Modeling** — Trained and compared Logistic Regression (baseline) and Random Forest (final model).
6. **Evaluation** — Focused on Precision and Recall (not just accuracy) since missing a default is costly for a bank.
7. **Deployment** — Final Random Forest model deployed via Streamlit Cloud.

## 📈 Results

| Model | Accuracy | Precision (Default) | Recall (Default) | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 80% | 0.54 | 0.76 | 0.86 |
| **Random Forest (final)** | **91%** | **0.87** | **0.72** | **0.92** |

**Top predictors of default:** loan-to-income ratio, applicant income, interest rate, and loan grade.

## 🗂️ Repository Contents

| File | Description |
|---|---|
| `app.py` | Streamlit web app source code |
| `model.pkl` | Trained Random Forest model |
| `feature_cols.pkl` | Feature column order used by the model |
| `options.pkl` | Dropdown options for categorical inputs |
| `requirements.txt` | Python dependencies |

## 🛠️ Tech Stack

Python · Pandas · Scikit-learn · Streamlit · Joblib

## 🏃 Run Locally

```bash
git clone https://github.com/Aditifulare/loan-default-predictor.git
cd loan-default-predictor
pip install -r requirements.txt
streamlit run app.py
```

## 👤 Author

**Aditi Fulare**
🔗 [LinkedIn](https://linkedin.com/in/aditi-fulare-6b0b993a9/) · [GitHub](https://github.com/Aditifulare) · [Portfolio](https://aditi-fulare.netlify.app)
