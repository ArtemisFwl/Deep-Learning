Telecom Customer Churn Prediction using Deep Learning
Project Overview

Customer churn is a critical challenge in the telecom industry, where acquiring new customers is significantly more expensive than retaining existing ones.
This project builds an end-to-end deep learning pipeline to predict customer churn using business-driven feature engineering and an Artificial Neural Network (ANN).

The project follows industry best practices, including:

Clear separation of EDA, preprocessing, modeling, and evaluation

Feature engineering based on telecom domain logic

Business-focused evaluation beyond raw accuracy

🎯 Objectives

Identify customers likely to churn

Build a robust deep learning model using engineered features

Optimize model evaluation based on business impact, not just accuracy

Demonstrate a production-style ML workflow

🗂️ Project Structure
telecom-churn-deep-learning/
│
├── data/
│   ├── raw/
│   │   └── telco-Customer-Churn.csv
│   └── processed/
│       ├── telecom_churn_initial_clean.csv
│       ├── telecom_churn_feature_engineered.csv
│       ├── X_test_final.npy
│       └── y_test_final.npy
│
├── notebooks/
│   ├── 01_business_eda.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling_ann.ipynb
│   └── 05_evaluation_insights.ipynb
│
├── models/
│   └── telecom_churn_ann.keras
│
├── requirements.txt
└── README.md

📒 Notebook Workflow
1️⃣ Business EDA

Notebook: 01_business_eda.ipynb

Exploratory analysis of churn behavior

Identification of churn patterns across tenure, contract type, charges, and services

Business-driven insights for feature creation

2️⃣ Baseline Data Preprocessing

Notebook: 02_data_preprocessing.ipynb

Missing value handling

Encoding categorical variables

Feature scaling

Baseline train–test split (benchmark pipeline)

3️⃣ Feature Engineering (Telecom Logic)

Notebook: 03_feature_engineering.ipynb

Key engineered features:

tenure_group – customer lifecycle stages

high_monthly_charge – revenue risk indicator

long_term_contract – customer commitment

num_services – service usage depth

electronic_check – payment risk signal

Feature engineering is performed before final modeling to avoid data leakage.

4️⃣ Deep Learning Model (ANN)

Notebook: 04_modeling_ann.ipynb

Rebuilt preprocessing pipeline using engineered features

Artificial Neural Network with:

Two hidden layers

Dropout regularization

Model trained using validation monitoring

Final trained model saved in native .keras format

5️⃣ Model Evaluation & Business Insights

Notebook: 05_evaluation_insights.ipynb

Evaluation on final, versioned test data

Metrics analyzed:

Accuracy

Confusion Matrix

Precision, Recall, F1-score

ROC–AUC

Threshold tuning applied to improve churn recall

📈 Model Performance (Final)

Test Accuracy: ~80%

Churn Recall (default threshold): ~53%

ROC–AUC: ~0.79

Threshold tuning increased churn recall by accepting controlled false positives

In churn prediction, missing a churner is more costly than targeting a loyal customer.
Therefore, recall optimization was prioritized over raw accuracy.

🧠 Key Business Insights

Early-tenure customers are significantly more likely to churn

Customers with multiple services show lower churn probability

Month-to-month contracts are high-risk churn segments

Payment method plays a role in churn behavior

Threshold tuning aligns model behavior with real-world retention strategies

🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

TensorFlow / Keras

Matplotlib, Seaborn

Git & GitHub