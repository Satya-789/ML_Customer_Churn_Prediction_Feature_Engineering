# Customer Churn Prediction: Feature Engineering and Model Building

## Project Overview
This Jupyter Notebook, `Customerr_Churn_Prediction_Feature_Engineering_.ipynb`, documents a machine learning project focused on predicting customer churn for a telecommunications company. The process includes comprehensive data cleaning, exploratory data analysis (EDA), feature engineering, preprocessing using Scikit-learn Pipelines, and training and evaluating classification models.

## Data Source
The project uses the "Telco Customer Churn" dataset, which is read from the file:
* `/content/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Key Steps and Analysis

### 1. Data Loading and Cleaning
* **Initial Check**: The dataset is loaded and inspected.
* **Data Type Correction**: The `TotalCharges` column is converted from an object type to a numeric (float) type, with errors coerced to `NaN`.
* **Missing Value Handling**: The 11 missing values in `TotalCharges` are imputed using the column's median.
* **Target Encoding**: The target variable `Churn` is converted from categorical ('Yes', 'No') to binary (1, 0).
* **Feature Removal**: The `customerID` column is dropped as it is non-predictive.

### 2. Feature Engineering and Preprocessing
* **Data Splitting**: The data is split into training and testing sets.
* **Preprocessing Pipelines**: Scikit-learn `Pipeline` and `ColumnTransformer` are used to standardize the workflow:
    * **Numerical Features**: Handled by median imputation and `StandardScaler` for scaling.
    * **Categorical Features**: Handled by `OneHotEncoder` for conversion to a numerical format.

### 3. Model Training and Evaluation
Two models are built and evaluated for comparison:
1.  **Logistic Regression** (Base Model)
2.  **Random Forest Classifier** (Selected Features Model)

The final model's performance on the test set is reported:

| Model | Metric | Value |
| :--- | :--- | :--- |
| Random Forest | Accuracy | 0.80 |

The classification report for the Random Forest model shows:
* **Precision (Class 1 - Churn)**: 0.67
* **Recall (Class 1 - Churn)**: 0.52

## Prerequisites

The following Python libraries are required to run the notebook:
* `pandas`
* `numpy`
* `matplotlib.pyplot`
* `seaborn`
* `sklearn` (specifically `model_selection`, `preprocessing`, `compose`, `pipeline`, `linear_model`, `ensemble`, and `metrics`)
