# Smart Chennai Flood Risk Analytics

## 📌 Project Overview

Smart Chennai Flood Risk Analytics is a machine learning-based system developed to analyze and predict flood risk across different wards of Chennai.

The project uses historical rainfall, flood-related, and ward-level data to create a structured dataset and applies machine learning techniques to estimate flood probability and classify wards into different flood-risk categories.

The system uses Random Forest, Gradient Boosting, and XGBoost models and provides the results through a Flask-based web dashboard.

---

## 🎯 Objectives

- Analyze historical flood and rainfall-related data for Chennai.
- Prepare a ward-level dataset for flood-risk analysis.
- Develop machine learning models for flood-risk prediction.
- Compare Random Forest, Gradient Boosting, and XGBoost models.
- Estimate flood probability for Chennai wards.
- Classify wards into Low, Moderate, High, and Very High risk categories.
- Provide an easy-to-use web dashboard for viewing ward-level flood risk.
- Support better flood preparedness and urban disaster management.

---

## 📊 Dataset

The project uses historical Chennai rainfall and flood-related datasets obtained from publicly available sources.

The final processed dataset contains:

- **200 Chennai wards**
- **46 model features**
- Ward-level flood-related information
- Flood probability
- Flood risk classification

The raw datasets include rainfall and flood-related information such as inundation points, flood hotspots, and flood extent data.

The raw datasets are not included in this repository when they are large. The processed dataset required by the application is included.

---

## 🤖 Machine Learning Models

Three ensemble machine learning algorithms were used:

### 1. Random Forest

Random Forest combines multiple decision trees to produce a robust prediction. It can handle nonlinear relationships and multiple input features effectively.

### 2. Gradient Boosting

Gradient Boosting builds decision trees sequentially, where each new tree attempts to improve the errors made by the previous trees.

### 3. XGBoost

XGBoost is an optimized gradient boosting algorithm designed for efficient and accurate prediction on structured datasets.

---

## 📈 Model Evaluation

The models were evaluated using Accuracy, Precision, Recall, and F1-Score.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Random Forest | 0.9992 | 0.9992 | 0.9992 | 0.9992 |
| Gradient Boosting | 0.9929 | 0.9929 | 0.9929 | 0.9929 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Grid-aware Random Forest | 0.9647 | 0.9650 | 0.9647 | 0.9648 |
| Grid-aware Gradient Boosting | 0.9989 | 0.9989 | 0.9989 | 0.9989 |
| Grid-aware XGBoost | 0.9967 | 0.9967 | 0.9967 | 0.9967 |

> Note: These evaluation results are based on the prepared project dataset and evaluation setup. They should not be interpreted as guaranteed real-world flood prediction accuracy.

---

## 🌊 Flood Risk Categories

The system classifies wards into four flood-risk categories:

- 🟢 **Low**
- 🟡 **Moderate**
- 🟠 **High**
- 🔴 **Very High**

The dashboard displays the predicted flood probability and corresponding risk category for the selected ward.

---

## 🖥️ Web Application

The machine learning models are integrated into a Flask web application.

The dashboard allows users to:

- View overall flood-risk information.
- Select a Chennai ward.
- View the ward's flood probability.
- View the predicted risk category.
- View relevant risk information.
- View recommended preventive actions.

---

## 🏗️ System Architecture

```text
Historical Rainfall & Flood Data
              ↓
      Data Preprocessing
              ↓
       Feature Engineering
              ↓
      Ward-Level Dataset
              ↓
     Machine Learning Models
       ↙       ↓       ↘
 Random Forest  GB    XGBoost
       ↘       ↓       ↙
       Model Evaluation
              ↓
       Flood Risk Prediction
              ↓
       Flask Backend / API
              ↓
        Web Dashboard
              ↓
     Ward-Level Risk Results
