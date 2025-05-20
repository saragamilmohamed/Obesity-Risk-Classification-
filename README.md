# 🧠 Obesity Risk Prediction Project

This repository contains a machine learning project aimed at predicting **obesity risk** in individuals using a variety of personal, dietary, and lifestyle attributes.

## 📌 Project Overview

We’ve built a Streamlit web application that allows users to interactively input their personal and lifestyle data and receive instant obesity risk predictions.

🚀 Try the App Online
👉 Live Streamlit App:[https://saragamilmohamed-obesity-risk-classification--app-mlobt9.streamlit.app]


**Objective**:  
Build a robust machine learning model to classify individuals into different obesity risk categories based on input features.

## 💡 Business & Health Significance

- **Healthcare Providers**: Enables early interventions and personalized recommendations.
- **Policy Makers**: Helps target high-risk populations for public health campaigns.
- **Individuals**: Encourages awareness and preventive actions through risk estimation.

---

## 🧾 Dataset Description

The dataset includes multiple features categorized as:

### 🔹 Personal Attributes
- Gender, Age, Height, Weight
- Family history of obesity
- Smoking habits

### 🔹 Eating Habits
- High caloric food consumption
- Vegetable intake
- Number of daily meals
- Snacking frequency
- Water and alcohol consumption

### 🔹 Physical Condition & Lifestyle
- Physical activity frequency
- Use of technology (daily time)
- Calorie monitoring habits
- Transportation mode

### 🧮 Target Labels: BMI-Based Obesity Categories
- Insufficient Weight
- Normal Weight
- Overweight
- Obesity I
- Obesity II
- Obesity III

---

## 📊 Data Preprocessing & EDA

- Transformed and rounded numerical features (e.g., Age, Height).
- Handled outliers using IQR method.
- Addressed imbalance in categorical variables (e.g., Transportation, Smoking).
- Encoding:
  - Binary: Label Encoding
  - Ordinal: Ordinal Encoding
  - Nominal: One-Hot Encoding

---

## 🧠 Models Implemented

| Model              | Accuracy | Notes                        |
|-------------------|----------|------------------------------|
| Decision Tree      | 83.96%   | Basic model, lowest accuracy |
| Random Forest      | 89.28%   | Good generalization          |
| XGBoost (Untuned)  | 90.03%   | Best pre-tuning performance  |
| XGBoost (Tuned)    | 90.94%   | Best overall performance     |

### 🔧 Hyperparameter Tuning
- **Method**: `RandomizedSearchCV` (10 iterations, 5-fold CV)
- **Best Params**:
  - `n_estimators=300`
  - `max_depth=3`
  - `learning_rate=0.2`
  - `gamma=0`
  - `reg_lambda=1`
  - `reg_alpha=1`

---

## 📈 Evaluation & Results

- **Confusion Matrix**: High accuracy across all 7 categories.
- **Learning Curve**: Stable training with minimal overfitting.
- **Class-wise Precision**:
  - Obesity Type I & II: ~98–100%
  - Obesity Type II & III: ~80–82%


