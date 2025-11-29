# Springleaf Marketing Response — Machine Learning Capstone Project

## 🎯 Project Objective
This project builds a complete end-to-end machine learning solution for the Springleaf Marketing Response dataset (Kaggle).
The goal is to predict whether a customer will respond to a marketing offer using high-dimensional, anonymized customer attributes.

The project demonstrates:
- Problem understanding
- Industry context
- Data cleaning & preprocessing
- Feature engineering
- Modeling (traditional ML + deep learning requirement)
- Hyperparameter tuning
- Model stacking
- Model interpretation
- Business insights & recommendations

## 📊 Dataset Overview
- 145,231 rows
- ~1,934 anonymized features
- Heavy missingness & placeholder values (e.g., 98/99/9999)
- Mixed types: numeric, categorical, encoded timestamps, location/job categories
- Marketing response target variable
  - 76.7% non-responders
  - 23.3% responders

##  🛠️ Project Workflow

| Stage | Description |
|---|---|
Data Cleaning | Handle missingness, placeholders, constant & duplicate columns, type conversions
DA | Target imbalance, distributions, correlations, MI scores, PCA
Feature Engineering | MI-based selection, row-level engineered features, categorical prep
Modeling | Logistic Regression, XGBoost, CatBoost, tuning, stacking
Model Evaluation | ROC-AUC, PR-AUC, F1, accuracy, comparison plots
Model Interpretation | Feature importances, permutation importance
Business Insights | How a bank would use the model in real targeting
Final Recommendations | What the marketing team should do

## 🤖 Models Implemented
Baseline Models
- Dummy Classifier
- Logistic Regression with balanced class weights

Advanced Models
- XGBoost (baseline)
- CatBoost (native categorical handling)
- XGBoost (RandomizedSearchCV tuned)
- Stacked Ensemble (XGBoost + CatBoost + Logistic Regression meta-model)

Planned / Optional
- Simple feed-forward deep learning model

## 📈 Key Metrics
Used due to class imbalance (23% positive):
- ROC-AUC
- Average Precision (PR-AUC)
- F1 Score
- Accuracy (shown but not used for decisions)

📌 Best Model (Validation Set)

Stacked XGBoost + CatBoost
- AUC: ~0.773
- PR-AUC: ~0.532
- F1: ~0.404
- Accuracy: ~0.794

## 🔍 Model Interpretation
Because SHAP had compatibility issues with XGBoost in this environment:
- Global Feature Importance (gain-based from XGBoost)
- Permutation Importance (validation-based)

Key Predictive Drivers
- Encoded date/time behavior patterns (VAR_0073, VAR_0075)
- Customer segment categories (VAR_0001)
- Job / state-like categories
- Missingness-based signals (zero_count)

These suggest:

✔ timing of interactions matters
✔ customer segment affects likelihood of response
✔ geographic/occupational groups show different response patterns

## 💼 Business Impact & Insights
For a financial marketing team, the model helps:
- Target customers most likely to respond
- Reduce campaign costs
- Increase ROI
- Optimize contact timing
- Identify high-value customer segments

This model could be deployed as:
- A batch-scoring pipeline (monthly campaign lists)
- A real-time scoring service (API)
- A segmentation engine for marketing managers

## 📂 Project Structure
Because this project is developed inside one Jupyter notebook, the structure is intentionally simple:

```bash
springleaf-ml-capstone/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│
├── notebook/
│   └── springleaf_capstone.ipynb       # full project pipeline in one notebook
│
├── results/
│   ├── model_metrics.csv               # saved metrics (optional)
│   ├── feature_importance.png          # plots
│   ├── evaluation_plots/               # ROC, PR, F1 comparisons
│
├── README.md
└── requirements.txt
```

## ⚙️ Tech Stack
Python, pandas, numpy,
scikit-learn, xgboost, catboost,
matplotlib, seaborn, plotly,
shap (attempted), permutation_importance

## 🚀 Future Work
- Add a deep learning model (TabNet or FNN)
- Reduce dimensionality more efficiently (Autoencoders, PCA, feature hashing)
- Deploy model using FastAPI or Flask
- Build a marketing simulation dashboard (Streamlit)
- Try SHAP with older XGBoost versions or GPU-based TreeExplainer

---
