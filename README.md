# Adult Income Classification: End-to-End Machine Learning Pipeline with EDA, Feature Engineering & KNN (GridSearchCV)

## Project Overview

This project demonstrates a complete end-to-end machine learning classification pipeline using the Adult Income dataset. It covers every stage of a real-world ML workflow, including data exploration, preprocessing, feature engineering, model training, hyperparameter tuning, model evaluation, and cross-validation.

The primary objective is to predict whether an individual's annual income is **>50K** or **<=50K** based on demographic and employment-related attributes.

---

## Dataset

**Dataset:** Adult Income Dataset

**Target Variable:** `income`

Classification Labels:

* <=50K
* > 50K

---

# Project Workflow

## 1. Exploratory Data Analysis (EDA)

The project begins with a detailed exploratory analysis of the dataset.

Performed analyses include:

* Dataset shape and structure
* Feature information
* Descriptive statistics
* Missing value detection
* Duplicate record detection
* Categorical value inspection
* Target distribution analysis
* Correlation analysis

Visualizations include:

* Count Plots
* KDE Plots
* Histograms
* Box Plots
* Correlation Heatmap

---

## 2. Data Cleaning

The dataset was cleaned before model development.

Steps performed:

* Removed duplicate records
* Replaced `?` with missing values (`NaN`)
* Removed unnecessary spaces from categorical values
* Standardized categorical text formatting

---

## 3. Feature Engineering & Preprocessing

A preprocessing pipeline was built using **Scikit-Learn Pipeline** and **ColumnTransformer**.

### Missing Value Handling

* **workclass**

  * Most Frequent Imputation

* **native-country**

  * Most Frequent Imputation

* **occupation**

  * Constant Value Imputation

### Categorical Features

* One-Hot Encoding
* `drop='first'`
* `handle_unknown='ignore'`

### Numerical Features

* StandardScaler

### Target Variable

* Label Encoding

---

## Design Decisions

Several preprocessing strategies were experimentally evaluated before selecting the final pipeline.

* RobustScaler was not used because it resulted in lower classification accuracy on this dataset.
* Outlier handling was intentionally skipped because it did not produce meaningful performance improvements. Additionally, features such as **capital-gain**, **capital-loss**, and **hours-per-week** contain dominant real-world values that occur frequently, and removing them as outliers would discard valid observations.
* Feature dropping was also evaluated but did not provide a significant improvement in model performance.

---

## 4. Machine Learning Models

Two classification models were implemented.

### Logistic Regression

Used as the baseline model.

Evaluation Metrics:

* Accuracy
* Precision
* Recall

---

### K-Nearest Neighbors (KNN)

The KNN classifier was optimized using **GridSearchCV**.

Hyperparameters searched:

* `n_neighbors`
* `weights`
* `p`

GridSearchCV performed an exhaustive search using **5-fold Cross Validation** to identify the best-performing hyperparameter combination.

---

## 5. Model Comparison

The performance of both models was compared using:

* Accuracy
* Precision
* Recall

A visual comparison chart was also generated.

---

## 6. Cross Validation

Both models were evaluated using **5-fold Cross Validation** to estimate their generalization performance.

Cross-validation results include:

* Fold-wise Accuracy
* Mean Accuracy
* Standard Deviation

---

# Key Findings

* Education level is one of the strongest indicators of income.
* Capital Gain has a strong relationship with higher income.
* Married individuals are more likely to belong to the >50K income group.
* Proper preprocessing significantly improves model performance.
* Hyperparameter tuning with GridSearchCV improved the KNN model over the baseline Logistic Regression model.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

---

# Repository Structure

```
├── adult_income.csv
├── Adult_Income_Classification.ipynb
├── README.md
```

---

# Future Improvements

* Try additional classification algorithms such as Random Forest, XGBoost, and Support Vector Machine.
* Perform feature selection using model-based techniques.
* Experiment with advanced preprocessing and feature engineering methods.
* Deploy the trained model as a web application using Streamlit or Flask.

---

# License

This project is created for learning, experimentation, and portfolio purposes.
