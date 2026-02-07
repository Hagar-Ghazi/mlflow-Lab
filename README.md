# Bank Consumer Churn Prediction Pipeline

This repository contains a professional end-to-end machine learning pipeline for predicting bank consumer churn. The project focuses on data preprocessing, addressing class imbalance, and comparing multiple classification algorithms using **MLflow** for experiment tracking and model management.

## 🚀 Key Features

* **Data Rebalancing**: Implements downsampling logic to handle imbalanced churn classes, ensuring the model doesn't favor the majority class.
* **Feature Engineering**: Uses a `ColumnTransformer` to automate `StandardScaling` for numerical features and `OneHotEncoding` for categorical variables.
* **Experiment Tracking**: Logs parameters (like `max_iter`, `C`, `n_estimators`), metrics (Accuracy, Precision, Recall, F1), and artifacts (Confusion Matrices) directly to MLflow.
* **Model Registry**: High-performing versions are registered for version control and lifecycle management.

## 🛠 Tech Stack

* **Language**: Python 3.x
* **Library**: Scikit-Learn
* **Tracking**: MLflow
* **Data Handling**: Pandas, Joblib
* **Visualization**: Matplotlib

---

## 📊 Models Evaluated

We compared three distinct approaches to identify the most effective predictor for consumer behavior:

1. **Logistic Regression**: Used as a baseline for high interpretability.
2. **Random Forest**: Leveraged to capture non-linear relationships and feature importance.
3. **Support Vector Machine (SVM)**: Optimized with an RBF kernel to handle complex decision boundaries.

---

## ⚙️ How to Run

1. **Clone & Install**:
```bash
git clone https://github.com/Hagar-Ghazi/mlflow-Lab.git
pip install pandas scikit-learn mlflow matplotlib joblib

```


2. **Execute Training**:
```bash
python train.py

```


3. **Launch MLflow UI**:
```bash
mlflow ui

```


View the results at `http://localhost:5000` to compare the metrics between the logged runs.

---

## 📁 Project Structure

* `preprocess.py`: Contains `rebalance()` and `preprocess()` functions.
* `train.py`: Main script for model training and MLflow logging.
* `mlruns/`: Directory containing all tracked experiment data.


### 📊 Model Results

The models were evaluated on a test set (30% of the balanced data) to ensure they generalize well to unseen consumer behavior. Below are the performance metrics recorded in the MLflow Model Registry.

| Model | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| **Random Forest** |0.7677| 0.7718| 0.7487 |**0.7601**|
| **SVM (RBF)** | 0.76287 | 0.7723 | 0.7337 | 0.7525|
| **Logistic Regression** | 0.705 |0.708 | 0.682 | 0.694 |

> **Analysis**: The **Random Forest** model currently serves as the "Champion" model due to its superior F1-Score. It effectively balances the need to identify customers who are likely to churn (Recall) without over-flagging loyal customers (Precision).

---

### 📈 Visualizations

The following artifacts are automatically generated and logged to MLflow for every run to assist in error analysis:

* **Confusion Matrix**: Visualizes the True Positives and False Positives to understand where the model is misclassifying users.
* **Feature Importance (RF only)**: Identifies which variables (e.g., Age, Balance, or Number of Products) are the strongest predictors of churn.

---

