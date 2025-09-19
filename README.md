# Payment Fraud Detection — Supervised Learning (Classification)

Project documentation for Milestone 2 Phase 1 — **Fraud transaction detection** in digital payment systems using several classification models (baseline) and the main model **XGBoost** with *pipeline* and *hyperparameter tuning*.

---

## Project Objectives
- Build an end-to-end workflow from EDA --> Feature Engineering --> Modeling --> Evaluation --> Model Storage.
- Compare several algorithms (KNN, SVM, Decision Tree, Random Forest, XGBoost) in a **class-imbalance** case.
- Optimize the best model and **adjust the threshold** to maximize relevant metrics (recall for the fraud class).

## Repository Outline
.
├── app.py                   # Main Streamlit script for the app
├── home.py                  # Landing page of the app
├── eda.py                   # Streamlit page for Exploratory Data Analysis
├── prediction.py            # Streamlit prediction page
├── model.pkl                # Stored machine learning model
├── xgboost_best_model.pkl   # Best XGBoost model from training
├── best_threshold.pkl       # Optimal probability threshold
├── numerical_features.pkl   # List of numerical features
├── categorical_features.pkl # List of categorical features
├── model_meta.json          # Model metadata
├── payment_fraud.csv        # Main dataset
├── requirements.txt         # Python dependencies
├── description.md           # Project description
├── README.md                # Task instructions from instructor
├── README_deploy.md         # Deployment guide for the app
├── url.txt                  # Deployment URL
├── P1M2_Muhammad_Luthfi_Alfatih.pdf / .pptx  # Project report and presentation
└── __pycache__/             # Python cache

> Note: *.pkl* files are generated after running the notebook.

## Problem Background
Fraud in digital transactions directly impacts financial losses and the reputation of service providers. The dataset shows a high **class imbalance**, with only ~**1.09%** of transactions labeled as *fraud*. Under this condition, _accuracy_ becomes a **misleading** metric, so the evaluation focus is shifted to **recall** (capturing as many fraud cases as possible) and **ROC-AUC**.

## Project Output
This project produces a machine learning model for fraud detection in digital payment systems.  
The best model is XGBoost, built within a pipeline including preprocessing, optimized through hyperparameter tuning and threshold adjustment to maximize recall on the fraud class.  
Besides the model, this project also delivers:

- `.pkl` artifact files for the model pipeline, optimal threshold, and feature lists.  
- Analysis notebooks for EDA and training.  
- Inference scripts for predicting on new data.  

## Data
- Source: [Payment Fraud: Empowering Financial Security (Kaggle)](https://www.kaggle.com/datasets/younusmohamed/payment-fraud-empowering-financial-security)  
- Size: 39,221 rows × 8 original columns before feature engineering.  
- Target: `label` (0 = legitimate, 1 = fraud)  
- Characteristics:  
  - Highly imbalanced: fraud only ~1.09% of data.  
  - Missing values:  
    - Category: 95 missing  
    - isWeekend: 560 missing  
  - Categorical features: `paymentMethod`, `Category`  
  - Numerical features: `numItems`, `localTime`, `hour`, `risk_score`, `transaction_velocity`, `payment_age_ratio`, `category_deviation`, `temporal_risk_window` (after feature engineering)  

## Method
This project applies a **supervised learning (classification)** approach with the following steps:  
1. **EDA**: Analyze target distribution, feature correlations, fraud rate per category.  
2. **Feature Engineering & Preprocessing**:  
   - Derived features such as `risk_score`, `transaction_velocity`, `payment_age_ratio`, `category_deviation`, `temporal_risk_window`.  
   - Missing value imputation (median/mode).  
   - Robust scaling (numerical).  
   - One-hot encoding (categorical).  
3. **Baseline Modeling**: KNN, SVM, Decision Tree, Random Forest, XGBoost.  
4. **Model Selection & Tuning**: XGBoost with GridSearchCV.  
5. *Threshold Tuning*: Adjusting default threshold (0.5) to 0.10 to improve recall.  
6. **Evaluation**:  
   - Main metrics: ROC-AUC and recall for fraud class.  
   - Accuracy ignored due to imbalance.  

## Stacks
- **Programming Language**: Python 3.10+  
- **Tools**: Jupyter Notebook, GitHub  
- **Python Libraries**:  
  - Data Analysis: `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`  
  - Machine Learning: `scikit-learn`, `xgboost`  
  - Model Saving/Loading: `joblib`  
  - Utilities: `json`, `pathlib`  

## Reference
- Dataset: [Kaggle – Payment Fraud: Empowering Financial Security](https://www.kaggle.com/datasets/younusmohamed/payment-fraud-empowering-financial-security)  
- Library Documentation:  
  - [Scikit-learn](https://scikit-learn.org/stable/)  
  - [XGBoost](https://xgboost.readthedocs.io/en/stable/)  
- Markdown Guide: [GitHub Docs](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)  

---

**Additional references:**  
- [Basic Writing and Syntax on Markdown](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)  
- [Example readme](https://github.com/fahmimnalfrzki/Swift-XRT-Automation)  
- [Another example](https://github.com/sanggusti/final_bangkit) (**Must read**)  
- [Additional reference](https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/)  
