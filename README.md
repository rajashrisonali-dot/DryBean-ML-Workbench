# Dry Bean ML Workbench

## Machine Learning Assignment 2 – Multi-Class Classification

### 1. Project Overview
This project implements and evaluates five Machine Learning classification algorithms using the UCI Dry Bean Dataset.

The objective is to classify dry beans into seven different categories using numerical features.

### 2. Dataset Details
- Dataset: UCI Dry Bean Dataset
- Total Instances: 13,611
- Total Features: 16
- Target Variable: Class
- Number of Classes: 7

### 3. Machine Learning Models
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### 4. Evaluation Metrics
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### 5. Project Structure
- `Dry_Bean_ML_Assignment2_2026aj05996.ipynb` – Model training and evaluation
- `SampleDataset.csv` – Dataset
- `test_data.csv` – Test dataset
- `model_comparison.csv` – Model comparison results
- `model/` – Saved trained models and metadata
- `app.py` – Streamlit application
- `requirements.txt` – Required Python libraries

### 6. Streamlit Application
The Streamlit application allows users to:
- Upload test CSV data
- Select a trained ML model
- View prediction results
- View evaluation metrics
- View confusion matrix and classification report

The application loads previously trained models and does not retrain models on uploaded test data.

### 7. How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

### 8. Dataset Source
UCI Machine Learning Repository – Dry Bean Dataset:

https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

### 9. Conclusion
Five classification algorithms were implemented and evaluated on the same dataset using six evaluation metrics. Their performance was compared to identify the best-performing model.
