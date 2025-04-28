
# 🏥 Multi-Class Obesity Risk Prediction

This project focuses on predicting **obesity risk categories** based on individuals' physical, lifestyle, and behavioral attributes using machine learning techniques.

---

## 📚 Project Overview

The goal of this project is to build and evaluate multiple machine learning models to classify individuals into different obesity risk levels.  
The data includes features such as gender, age, height, weight, eating habits, physical activity, technology usage, and more.

---

## 🛠️ Techniques Used

- **Data Preprocessing**:
  - Handling missing data
  - Encoding categorical variables
  - Feature scaling (Normalization/Standardization)
- **Feature Engineering**:
  - Label Encoding
  - One-Hot Encoding
  - Ordinal Encoding
- **Machine Learning Models**:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest Classifier
  - XGBoost Classifier

---

## 📈 Model Performance

- **XGBoost** achieved the **highest performance** with an **accuracy of around 90%**.
- Other models like Random Forest, SVM, and Logistic Regression also performed well, but XGBoost was the most accurate and robust against overfitting.
- Evaluation metrics used:
  - Accuracy
  - F1-Score
  - Confusion Matrix
  - Precision & Recall

---

## 📊 Results Summary

| Model                  | Accuracy  | Notes                     |
|-------------------------|-----------|----------------------------|
| Logistic Regression     | 85%       | Good but slightly underfitting |
| SVM                     | 86%       | Better generalization      |
| Random Forest           | 88%       | Strong with some variance  |
| **XGBoost**             | **90%**   | Best overall performance   |




