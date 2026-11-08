# Dry Bean ML Workbench

## Machine Learning Assignment 2 – Multi-Class Classification

**Student Name:** TUPE RAJASHRI VISHWAS  
**BITS ID:** 2026AJ05996  
**Programme:** M.Tech Artificial Intelligence and Machine Learning

---

## 1. Problem Statement

The objective of this project is to develop and evaluate multiple Machine Learning classification models to classify dry beans into seven different categories using their physical characteristics.

Five classification algorithms are implemented on the same dataset. Their performance is evaluated using six standard classification metrics to identify the best-performing model.

An interactive Streamlit web application is also developed and deployed to demonstrate the trained models and their prediction results.

## 2. Dataset Description

- **Dataset Name:** UCI Dry Bean Dataset
- **Dataset Source:** UCI Machine Learning Repository
- **Total Instances:** 13,611
- **Total Features:** 16
- **Target Variable:** Class
- **Number of Classes:** 7

The dataset contains numerical measurements describing the shape and size of dry beans.

The seven bean categories are:

1. BARBUNYA
2. BOMBAY
3. CALI
4. DERMASON
5. HOROZ
6. SEKER
7. SIRA

**Dataset Source:**  
https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

## 3. GitHub Repository Link

**GitHub Repository:**  
https://github.com/rajashrisonali-dot/DryBean-ML-Workbench

The repository contains the Jupyter Notebook, dataset, test data, saved trained models, Streamlit application source code, requirements file and project documentation.

## 4. Machine Learning Models Used

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

All five models were trained and evaluated using the same dataset and the same train-test split.

### Evaluation Metrics

The following six metrics were used:

- **Accuracy:** Measures the proportion of correctly classified instances.
- **AUC:** Measures the model's ability to distinguish between classes.
- **Precision:** Measures how many predicted positive classifications are correct.
- **Recall:** Measures how many actual positive classifications are correctly identified.
- **F1 Score:** Represents the balance between precision and recall.
- **MCC:** Measures the overall quality of classification predictions.

### Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | 0.9934 | 0.9215 | 0.9207 | 0.9209 | 0.9042 |
| Decision Tree | 0.8917 | 0.9330 | 0.8912 | 0.8917 | 0.8913 | 0.8691 |
| KNN | 0.9166 | 0.9812 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Gaussian Naive Bayes | 0.7639 | 0.9644 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| Support Vector Machine | **0.9221** | **0.9937** | **0.9224** | **0.9221** | **0.9222** | **0.9059** |

## 5. Observations on Model Performance

### Logistic Regression
Logistic Regression achieved an accuracy of 92.07% and an AUC score of 0.9934. It demonstrated strong classification performance and maintained a good balance between precision and recall.

### Decision Tree
Decision Tree achieved an accuracy of 89.17%. It performed reasonably well but showed lower overall performance compared with Logistic Regression, KNN and SVM.

### K-Nearest Neighbors (KNN)
KNN achieved an accuracy of 91.66% and an AUC score of 0.9812. It demonstrated good classification performance and was one of the stronger models in this experiment.

### Gaussian Naive Bayes
Gaussian Naive Bayes achieved an accuracy of 76.39%. Although its AUC score was 0.9644, its accuracy, F1 score and MCC were lower than those of the other models.

### Support Vector Machine (SVM)
Support Vector Machine achieved the highest accuracy of 92.21%, an AUC score of 0.9937 and an MCC score of 0.9059. It provided the best overall classification performance among the five evaluated models.

### Overall Winner

**Support Vector Machine (SVM)** is the best-performing model for the UCI Dry Bean Dataset in this experiment.

It achieved the highest accuracy, AUC, precision, recall, F1 score and MCC among the five evaluated models.

## 6. Project Structure

```text
DryBean-ML-Workbench/
│
├── app.py
├── requirements.txt
├── README.md
├── Dry_Bean_ML_Assignment2_2026aj05996.ipynb
├── SampleDataset.csv
├── test_data.csv
├── model_comparison.csv
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── k_nearest_neighbors.joblib
    ├── gaussian_naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── metadata.joblib
```

## 7. Streamlit Application

The project is deployed using Streamlit Community Cloud.

**Live Application:**  
https://drybean-ml-workbench-05996.streamlit.app/

The application provides the following features:

- Upload test data in CSV format.
- Select a trained Machine Learning model using a dropdown.
- Preview the uploaded dataset.
- Generate predictions using the selected saved model.
- Display six evaluation metrics: Accuracy, AUC, Precision, Recall, F1 Score and MCC.
- Display the confusion matrix.
- View classification results.

The application loads previously trained models from the saved model files. It does not retrain the models using uploaded test data.

The uploaded test CSV must contain the required 16 input features and the target column `Class` for evaluation.

## 8. How to Run the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in a web browser.

## 9. Conclusion

In this project, five Machine Learning classification algorithms were implemented and evaluated using the UCI Dry Bean Dataset.

The models were compared using six evaluation metrics. Support Vector Machine achieved the highest overall performance with an accuracy of 92.21%.

An interactive Streamlit application was successfully developed and deployed to demonstrate model selection, predictions and evaluation results.

This project demonstrates the complete Machine Learning workflow, including dataset validation, model training, performance evaluation, model saving, GitHub repository management and web application deployment.

---

**Student:** TUPE RAJASHRI VISHWAS  
**BITS ID:** 2026AJ05996  
**Course:** M.Tech AIML – Machine Learning Assignment 2
