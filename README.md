# 💳 Customer Transaction Prediction

Predicting whether a customer will make a transaction using machine learning, based on customer-related features — helping financial institutions identify potential customers and improve targeted marketing strategies.

**🔗 Live App:** https://your-streamlit-app-url.streamlit.app/

---

## Overview

This project predicts whether a customer is likely to make a transaction using historical customer data. Multiple machine learning classification models were trained and compared to identify the best-performing model. The final model is deployed as an interactive Streamlit web application where users can enter customer feature values and instantly receive a transaction prediction.

## Dataset

* Customer Transaction Prediction dataset
* **Features:** 200 anonymized numerical customer features (`var_0` to `var_199`)
* **Target:**

  * **0** – No Transaction
  * **1** – Transaction

## Approach

1. **Data Cleaning** — verified and removed missing values and duplicate records.
2. **Exploratory Data Analysis (EDA)** — examined feature distributions, class imbalance, and data characteristics.
3. **Data Preprocessing** — applied feature scaling using StandardScaler where required.
4. **Handling Class Imbalance** — balanced the training dataset using SMOTE.
5. **Modeling** — trained and compared four classification models:

   * Logistic Regression
   * Decision Tree
   * XGBoost
   * LightGBM
6. **Hyperparameter Tuning** — optimized model performance using GridSearchCV.
7. **Evaluation** — compared models using Accuracy, Precision, Recall, F1 Score, ROC-AUC Score, Confusion Matrix, and Classification Report.

## Results

| Model                | Performance                                     |
| -------------------- | ----------------------------------------------- |
| Logistic Regression  | Strong baseline with good generalization        |
| Decision Tree        | High training accuracy but prone to overfitting |
| XGBoost              | High predictive performance                     |
| **LightGBM (Tuned)** | **Best overall performance** ✅                  |

**Final Model:** **LightGBM Classifier**, selected based on its overall performance, generalization ability, and evaluation metrics on the test dataset.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, LightGBM, XGBoost, Imbalanced-learn (SMOTE), Matplotlib, Streamlit, Joblib

## How to Run Locally

```bash
git clone https://github.com/rakesh30-05/customer-transaction-prediction.git
cd customer-transaction-prediction
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```text
customer-transaction-prediction/
├── app.py                     # Streamlit web application
├── Customer_Transaction.ipynb # Data analysis & model training
├── model.pkl                  # Trained LightGBM model
├── scaler.pkl                 # Saved StandardScaler
├── requirements.txt
├── train.csv                  # Dataset
└── README.md
```

## Features

* Interactive Streamlit web application
* Real-time customer transaction prediction
* Machine learning model comparison
* Class imbalance handling using SMOTE
* Hyperparameter tuning for improved performance
* Clean and user-friendly interface

## Limitations

* The model is trained on historical data and may not reflect future customer behavior.
* Dataset features are anonymized, limiting feature interpretation.
* Predictions are intended for educational and demonstration purposes and should not be used as the sole basis for business decisions.

## Author

**D N Rakesh**

* **LinkedIn:** https://www.linkedin.com/in/dnrakesh8008
* **GitHub:** https://github.com/rakesh30-05
