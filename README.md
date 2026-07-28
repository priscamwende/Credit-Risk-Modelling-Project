# Machine Learning for Credit Risk Modeling: Predicting Customer Loan Default Using Financial and Behavioral Data 


### Ploblem statement

Financial institutions face significant challenges in identifying loan applicants who are likely to default while ensuring that creditworthy customers are not unfairly rejected. Traditional credit risk assessment methods may fail to capture complex relationships within large and diverse financial datasets, leading to poor lending decisions and increased financial losses. This project aims to develop a machine learning model that predicts the probability of customer loan default using the Home Credit Default Risk dataset. By leveraging customer demographic information, financial history, credit bureau records, previous loan applications, installment payments, credit card activity, and POS/Cash loan history, the model seeks to improve credit risk assessment and support more accurate, data-driven lending decisions.


### Objectives

#### Main Objective

Develop a machine learning model capable of accurately predicting customer loan default risk.

#### Specific Objectives

* Clean and preprocess the Home Credit dataset.
* Engineer meaningful customer-level features from multiple related datasets.
* Perform exploratory data analysis (EDA) to understand customer characteristics.
* Handle missing values, anomalies, and outliers.
* Encode categorical variables and scale numerical features where necessary.
* Select the most informative features.
* Build and compare multiple machine learning models.
* Evaluate model performance using appropriate classification metrics.
* Provide business recommendations to improve credit risk assessment.
###  Dataset

The project uses the **Home Credit Default Risk** dataset from Kaggle.

It consists of several relational tables:

| Dataset                   | Description                                      |
| ------------------------- | ------------------------------------------------ |
| application_train.csv     | Customer loan applications with default labels   |
| application_test.csv      | Customer applications used for prediction        |
| bureau.csv                | Previous loans from other financial institutions |
| bureau_balance.csv        | Monthly history of bureau loans                  |
| previous_application.csv  | Previous Home Credit loan applications           |
| POS_CASH_balance.csv      | Monthly POS and cash loan balances               |
| credit_card_balance.csv   | Monthly credit card balances                     |
| installments_payments.csv | Historical installment payment records           |


#  Project Workflow

```
Data Collection
        │
        ▼
Feature Engineering
        │
        ▼
Data Cleaning
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Missing Value Handling
        │
        ▼
Categorical Encoding
        │
        ▼
Feature Scaling
        │
        ▼
Feature Selection
        │
        ▼
Train-Test Split
        │
        ▼
Class Imbalance Handling
        │
        ▼
Machine Learning Models
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Model Evaluation
        │
        ▼
Feature Importance
        │
        ▼
Final Prediction
```

---

# Feature Engineering

A significant portion of this project focuses on transforming multiple relational datasets into a single customer-level dataset.

The following tables were aggregated:

#### Bureau Balance

Monthly bureau loan records were summarized into one record per loan by calculating:

* Number of months recorded
* Earliest month
* Latest month
* Frequency of each loan status
#### Bureau

Customer bureau records were aggregated into one row per customer by computing:

* Average credit amounts
* Maximum credit amounts
* Outstanding debt
* Credit limits
* Overdue balances
* Loan counts
* Proportion of loan categories
#### POS/Cash Balance

Monthly POS/Cash loan records were summarized into customer-level features including:

* Average installments
* Maximum installments
* Days past due
* Contract status proportions
* Number of previous POS loans
#### Credit Card Balance

Customer credit card behavior was summarized using:

* Credit utilization
* Credit limits
* Payment behavior
* Drawing behavior
* Delinquency measures
* Number of previous credit cards
#### Previous Applications

Historical loan applications were summarized using:

* Loan amounts
* Interest rates
* Payment information
* Application outcomes
* Previous application counts
#### Installment Payments

Repayment behavior was summarized by calculating:

* Average payment amount
* Payment ratios
* Days paid late
* Underpayments
* Payment variability
* Number of installments
* Number of previous installment loans

### Exploratory Data Analysis

EDA included:

* Target distribution
* Missing value analysis
* Numerical feature distributions
* Boxplots
* Correlation analysis
* Outlier detection
* Default rates across demographic groups
* Income distribution
* Age distribution
* Credit amount distribution
* 
### Data Cleaning

Several preprocessing techniques were applied:

* Removed unnecessary columns
* Corrected anomalous values (`DAYS_EMPLOYED = 365243`)
* Replaced infinite values
* Removed constant features
* Checked duplicate columns
* Verified train/test consistency

### Missing Value Handling

Missing values were handled using:

* Median imputation for numerical variables
* "Missing" category for categorical variables
* Missing-value indicator variables for highly incomplete features
### Categorical Encoding

Categorical variables were transformed into numerical representations using:

* One-Hot Encoding

This ensures compatibility with machine learning algorithms while preserving categorical information.
### Feature Scaling

StandardScaler was applied to numerical features for algorithms that require standardized inputs, such as:

* Logistic Regression
* Support Vector Machines
* Neural Networks

Unscaled datasets were also retained for tree-based algorithms.


### Feature Selection

Feature selection methods include:

* Constant feature removal
* Highly correlated feature removal
* Mutual Information
* Random Forest Feature Importance
* Recursive Feature Elimination (RFE)

### Machine Learning Models

The following models are evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* LightGBM
* CatBoost

### Evaluation Metrics

Model performance is assessed using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* Precision-Recall Curve
### Technologies Used

Programming Language

* Python

**Libraries**

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* SciPy
###  Expected Outcomes

The project is expected to:

* Improve loan default prediction accuracy.
* Identify the most influential factors contributing to customer default.
* Compare the effectiveness of several machine learning algorithms.
* Support more informed, data-driven lending decisions.

### Business Impact

The developed model can help financial institutions:

* Reduce loan default risk.
* Improve credit approval decisions.
* Minimize financial losses.
* Identify high-risk applicants early.
* Enhance customer risk profiling.
* Increase operational efficiency through automated credit assessment.


