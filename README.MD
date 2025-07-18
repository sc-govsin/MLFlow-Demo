# 🚀 IoT Water Meter Anomaly Detection

This repository implements a complete anomaly detection system for IoT water meters using various machine learning models. It includes data preprocessing, feature engineering, model training with imbalance handling, experiment tracking, and deployment-ready wrappers.

---

## 📁 Folder Structure

```

├── Models/
│   ├── LogisticRegression/
│   │   ├── LogisticRegression.pkl
│   │   ├── standard\_scaler.pkl
│   │   ├── reference\_data.csv
│   │   └── confusion\_matrix.png
│   ├── RandomForest/
│   ├── XGBoost/
│   └── LSTM/
│
├── logistic\_regression\_pipeline.txt   # Notebook-style script for LR
├── random\_forest\_pipeline.ipynb
├── xgboost\_pipeline.ipynb
├── lstm\_pipeline.ipynb
├── README.md

````

---

## ✅ Models Used

| Model              | Notes                                  |
|-------------------|----------------------------------------|
| Logistic Regression | Baseline linear model with SMOTE       |
| Random Forest       | Tree-based ensemble classifier         |
| XGBoost             | Boosted gradient trees for better ROC  |
| LSTM                | Deep learning model on sequence data   |

---

## 📊 Key Features

- 📥 Download + preprocess water meter data
- 🧬 Feature engineering (e.g., part of day, diff, encoded time)
- ⚖️ Imbalance handling with **SMOTE**
- 📈 Model training + scoring
- 🧪 MLflow logging:
  - Parameters
  - Metrics (precision, recall, F1)
  - Confusion matrix
  - `reference_data.csv` for monitoring
- 🧩 PyFunc wrappers for structured prediction
- ✅ Compatible with **Evidently AI** (`column_mapping` logged in each run)

---

## 🧪 Running Experiments

Each experiment is a standalone notebook or script.  
After editing or training:

```bash
# Run Logistic Regression
python logistic_regression_pipeline.txt

# For notebooks
jupyter notebook random_forest_pipeline.ipynb
````

---

## 🧰 Requirements

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn mlflow xgboost
```

> For LSTM:

```bash
pip install tensorflow
```

---

## 🔮 Column Mapping for Evidently

Each model directory contains `reference_data.csv`, and each MLflow run logs:

```json
{
  "numerical_features": [...],
  "categorical_features": [],
  "text_features": [],
  "input_features": [...],
  "predicted_features": ["prediction"]
}
```

---

## 🧪 Example Results

| Model              | Precision | Recall | F1 Score |
| ------------------ | --------- | ------ | -------- |
| LogisticRegression | 0.88      | 0.83   | 0.85     |
| RandomForest       | 0.91      | 0.89   | 0.90     |
| XGBoost            | 0.92      | 0.90   | 0.91     |
| LSTM               | 0.93      | 0.91   | 0.92     |
