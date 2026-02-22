# 📊 Credit Card Default Prediction

## 📌 Project Overview

This project predicts whether a credit card client will default on their payment next month using supervised machine learning models.

The dataset used is the **Default of Credit Card Clients Dataset** from the UCI Machine Learning Repository.

<img width="945" height="837" alt="image" src="https://github.com/user-attachments/assets/7f5df71f-8ce1-44d8-addd-d7cbf9f24820" />

### 🎯 Goals

- Compare multiple classification models  
- Perform feature engineering and aggregation  
- Evaluate models using accuracy, AUC, confusion matrix, and lift curves  
- Analyze overfitting vs. generalization performance  

---

## 🗂 Dataset

**Source:** UCI Machine Learning Repository  
**Samples:** 30,000 credit card clients  
**Features:** 23 original features  
**Target variable:** `default payment next month` (binary classification)

### Feature Categories

#### Demographics
- `SEX`
- `EDUCATION`
- `MARRIAGE`
- `AGE`

#### Credit Information
- `LIMIT_BAL`

#### Payment History
- `PAY_0` – `PAY_6`

#### Bill Statements
- `BILL_AMT1` – `BILL_AMT6`

#### Previous Payments
- `PAY_AMT1` – `PAY_AMT6`

---

## ⚙️ Data Preprocessing

The following preprocessing steps were applied:

- Set correct column headers  
- Convert object columns to numeric  
- Handle missing values (mean imputation)  
- One-hot encoding of categorical variables  
- Train / validation / test split:
  - 64% Training  
  - 16% Validation  
  - 20% Test  
- Feature scaling using `StandardScaler` (for models requiring normalization)

---

## 🧠 Feature Engineering

Additional aggregated features were created:

### Payment Behavior
- `Mean_Payment_Status`
- `Worst_Payment_Status`
- `Count_Late_Payments`

### Billing Behavior
- `Mean_Bill_Amount`
- `Max_Bill_Amount`
- `Total_Bill_Amount`
- `Std_Bill_Amount`

### Payment Amount Behavior
- `Mean_Payment_Amount`
- `Max_Payment_Amount`
- `Total_Payment_Amount`
- `Payment_Bill_Ratio`

These features were analyzed using:

- Correlation heatmaps  
- Boxplots  
- Pairplots  
- Distribution plots  

---

## 🤖 Models Implemented

The following models were trained and compared:

- K-Nearest Neighbors (with GridSearch)  
- Logistic Regression (`class_weight="balanced"`)  
- Linear Discriminant Analysis  
- Gaussian Naive Bayes  
- Artificial Neural Network (Keras)  
- Decision Tree  

---

## 📈 Model Evaluation

Models were evaluated using:

- Accuracy  
- Precision / Recall / F1-score  
- ROC-AUC  
- Confusion Matrix  
- Lift Charts  
- Train vs. Validation Error (overfitting detection)  

---

## 🔍 Validation AUC Comparison

| Model | Validation AUC |
|-------|----------------|
| ANN | **0.771** |
| Logistic Regression | 0.721 |
| LDA | 0.713 |
| KNN | 0.711 |
| Gaussian NB | 0.682 |
| Decision Tree | 0.609 |

---

## 🏆 Best Performing Model

The **Artificial Neural Network** achieved:

- Test Accuracy ≈ 82%  
- Validation AUC ≈ 0.77  
- Good balance between precision and recall  

The Decision Tree severely overfitted (Train AUC = 1.00).

---

## 📊 Key Insights

- Default prediction is a class-imbalanced problem.  
- Logistic Regression improved recall using `class_weight="balanced"`.  
- Neural networks provided the best AUC.  
- Decision Trees overfit without pruning.  
- Payment behavior aggregation features added predictive power.  

---

## 🛠 Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Keras / TensorFlow  
- Matplotlib  
- Seaborn  
