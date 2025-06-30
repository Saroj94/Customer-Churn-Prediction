# 🧠 Customer Churn Prediction using Voting Classifier (ML Project)
On **Customer Churn Prediction** project — I tackle the challenge of identifying customers likely to leave a service using machine learning techniques.

This project includes **exploratory data analysis**, **feature engineering**, **statistical testing**, **model training**, and an **ensemble learning approach** using Voting Classifier.

---

## 📌 Problem Statement

Customer churn is one of the most critical KPIs for any subscription-based business. The goal is to develop a machine learning model that accurately predicts whether a customer will churn, allowing businesses to take proactive steps to retain them.

---

## 📊 Exploratory Data Analysis (EDA)

I started with thorough descriptive and statistical analysis:

- Measured central tendency (mean, median, mode) of numerical features.
- Assessed **skewness** to identify asymmetry in distributions.
- Detected **outliers** using minimum and maximum values.
- Formed hypotheses about transformations required for better feature performance.

### 🔍 Univariate & Bivariate Analysis

- Analyzed each feature distribution individually and in relation to the target variable `Churn`.
- Visualized feature relationships with **heatmaps, bar plots,pie plots and box plots**.
- Performed **Chi-Square Hypothesis Testing** between the `Contract` feature and `Churn` to evaluate dependency.

---

## 🧪 Feature Engineering

- Transformed categorical features using:
  - `LabelEncoder` for binary variables.
  - `OneHotEncoder` for multi-class categorical features.
- Reasoning:
  1. Reduced high-dimensionality in sparse features.
  2. Adapted to both binary and high-cardinality features.
- Applied `PowerTransformer` to reduce skewness in numerical data and improve model performance.

---

## 🤖 Models Trained

I trained a total of **seven classification models**:

1. `Support Vector Classifier (SVC)`
2. `Random Forest Classifier`
3. `Logistic Regression`
4. `AdaBoost Classifier`
5. `XGBoost Classifier`
6. `Gradient Boosting Classifier`
7. `LightGBM Classifier`

After evaluating their performance, I selected the **top 4 models** based on accuracy and generalization capability for ensemble learning.

---

## 🧩 Voting Classifier Ensemble

I used a **VotingClassifier** to combine predictions from the best-performing models:

- `Support Vector Classifier`
- `Logistic Regression`
- `Gradient Boosting`
- `AdaBoost`

### ✅ Ensemble Accuracy: **78%**

This was my first time using VotingClassifier for this type of problem, and it significantly improved the overall stability of the predictions.

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- XGBoost, LightGBM

---

## 📈 Future Improvements

- Hyperparameter tuning using GridSearchCV for ensemble.
- Model deployment via FastAPI or Streamlit for real-time churn prediction.
- Deploying the project using the concepts of MLOps.
