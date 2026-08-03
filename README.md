# Machine Learning for Credit Risk Modeling: Predicting Customer Loan Default Using Financial and Behavioral Data

## Project Overview

Credit risk assessment is one of the most important tasks performed by financial institutions. Every lending decision involves uncertainty, and approving loans for customers who are unlikely to repay them can lead to significant financial losses. At the same time, rejecting customers who are capable of repaying their loans reduces business opportunities and financial inclusion.

The objective of this project is to develop a machine learning model that predicts whether a customer is likely to default on a loan using the Home Credit Default Risk dataset. The model combines customer demographic information with historical financial behaviour collected from multiple data sources to improve the accuracy of credit risk assessment.

Rather than relying on traditional rule-based credit scoring, this project applies machine learning techniques to identify complex relationships between customer characteristics and repayment behaviour. The resulting model can support financial institutions in making more informed lending decisions while reducing the risk associated with loan defaults.

# Business Problem

Financial institutions receive thousands of loan applications from customers with different financial backgrounds and repayment histories. Determining which applicants are likely to repay their loans is a challenging task, especially when customer information comes from multiple data sources.

The Home Credit Default Risk dataset contains customer application information together with credit bureau records, previous loan applications, installment payment history, POS/Cash loan history, and credit card balances. These datasets provide valuable information about a customer's financial behaviour but require extensive preprocessing before they can be used for predictive modelling.

The goal of this project is therefore to build a classification model capable of estimating the probability that a customer will default on a loan, enabling lenders to make more accurate, data-driven credit decisions.

# Dataset Description

The project uses the Home Credit Default Risk dataset.

The data consists of one primary application dataset together with several secondary relational datasets containing customer financial history.

The secondary datasets include:

* Bureau credit history
* Bureau monthly loan balance
* Previous loan applications
* Credit card balance history
* POS/Cash loan balance history
* Installment payment history

Since each customer may appear multiple times in the secondary datasets, these records were aggregated into customer-level features before being merged with the main application dataset. This produced a single modelling dataset containing one record per applicant.

# Exploratory Data Analysis (EDA)

Exploratory Data Analysis was conducted to understand the structure, quality and characteristics of the data before any preprocessing or modelling.

Several important observations were made during this stage.

### Class Imbalance

The target variable was highly imbalanced.

Approximately 92% of customers successfully repaid their loans, while only about 8% defaulted.

This imbalance indicated that relying on accuracy alone would produce misleading results. Consequently, ROC-AUC was selected as the primary evaluation metric, and class weighting was incorporated into the machine learning models.

### Missing Values

Many numerical and categorical variables contained substantial missing values.

The missing values were not random. In several cases, the absence of information represented meaningful customer characteristics rather than poor data quality.

This finding guided the choice of appropriate imputation methods and the creation of missing-value indicator features during feature engineering.

### Data Anomalies

Exploratory analysis identified abnormal values in variables such as `DAYS_EMPLOYED`, where an unusually large repeated value represented missing or unavailable employment information rather than actual employment duration.

These anomalies were identified and appropriately handled during preprocessing.

### Feature Relationships

Correlation analysis showed that most individual variables had relatively weak linear relationships with the target variable.

This suggested that loan default prediction depends on complex interactions between multiple features rather than any single variable.

A separate linear separability assessment confirmed that the classes were not perfectly separable using simple linear boundaries, supporting the inclusion of more advanced machine learning algorithms capable of learning non-linear relationships.

# Data Cleaning and Preprocessing

Several preprocessing techniques were applied to improve data quality and prepare the dataset for machine learning.

## Missing Value Treatment

Numerical variables were imputed using the median because it is robust to extreme values and preserves the overall distribution of the data.

Categorical variables were imputed using the most frequent category or appropriate missing indicators, allowing the models to retain potentially useful information contained in missing values.

After preprocessing, no missing values remained in the modelling dataset.

## Feature Engineering

The Home Credit dataset consists of multiple related tables containing repeated observations for each customer.

To create a single modelling dataset, these records were aggregated using statistical measures such as:

* Mean
* Maximum
* Minimum
* Sum
* Count

These engineered variables summarised each customer's historical financial behaviour, previous borrowing patterns, repayment history and credit utilisation.

Feature engineering substantially increased the amount of predictive information available for model training.

## Encoding

Machine learning algorithms require numerical input.

Categorical variables were therefore converted into numerical variables using one-hot encoding.

This approach preserves all category information while avoiding artificial ordering between categories.

## Duplicate Feature Removal

Following feature engineering and encoding, duplicate feature detection was performed.

Candidate duplicate columns were first identified using sampled fingerprints before being verified against the complete dataset.

Only columns that were confirmed to contain identical information across the full dataset were removed.

Removing duplicate features reduced redundancy without losing predictive information.

## Feature Scaling

Numerical variables were standardised using StandardScaler.

Scaling ensures that variables measured on different scales contribute equally to algorithms that are sensitive to feature magnitude, such as Logistic Regression.

## Correlation-Based Feature Selection

Highly correlated variables were identified using a correlation threshold of 0.90.

For each correlated pair, the feature with the stronger relationship to the target variable was retained while the weaker feature was removed.

This reduced multicollinearity, simplified the dataset and improved model efficiency.

The feature space was reduced from 659 features to 510 features without removing useful predictive information.

# Model Development

The processed dataset was divided into training and validation datasets using an 80:20 stratified split.

Stratified sampling ensured that both datasets preserved the original distribution of the target classes despite the class imbalance.

Several supervised machine learning algorithms were evaluated.

## Logistic Regression

Logistic Regression was used as the baseline model because it is simple, computationally efficient and highly interpretable. It provides an important benchmark against which more advanced machine learning models can be compared.

## Decision Tree

Decision Trees were included because they capture non-linear relationships between variables while producing interpretable decision rules.

## Random Forest

Random Forest was selected because it combines multiple decision trees to improve predictive performance and reduce overfitting through ensemble learning.

## XGBoost

XGBoost was chosen because gradient boosting algorithms consistently achieve excellent performance on structured tabular datasets. It is capable of modelling complex feature interactions while maintaining strong predictive accuracy.

## CatBoost

CatBoost was included because it performs well on structured datasets containing categorical information and often requires relatively little parameter tuning.

## LightGBM

LightGBM was evaluated separately because it does not support feature names containing certain special characters introduced during one-hot encoding.

The feature names were therefore cleaned before training the model. This preprocessing step affected only the feature names and did not modify the underlying data.


# Feature Selection Strategy

Feature selection was performed to reduce redundancy and improve model efficiency while retaining the most informative predictors.

The first stage involved removing duplicate features created during feature engineering and encoding.

The second stage removed highly correlated variables while retaining the feature with the stronger relationship to the target variable.

# Model Evaluation

Each machine learning model was evaluated using multiple classification metrics.

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC

Because the dataset is highly imbalanced, ROC-AUC was selected as the primary evaluation metric. It provides a more reliable measure of a model's ability to distinguish between customers who are likely to default and those who are likely to repay their loans.

The baseline Logistic Regression model achieved a ROC-AUC score of approximately 0.77, providing a strong benchmark for comparison with the remaining machine learning algorithms.

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* Matplotlib
* Seaborn

# Conclusion

This project demonstrates how machine learning can improve credit risk assessment by combining customer demographic information with historical financial behaviour collected from multiple financial sources.

Through comprehensive exploratory data analysis, feature engineering, data cleaning, feature selection and model comparison, the project develops predictive models capable of estimating the probability of customer loan default.

The final stage of the project focuses on selecting the best-performing model through comprehensive model comparison and hyperparameter tuning before generating the final loan default predictions. The resulting model aims to support more accurate lending decisions, reduce financial risk and improve the overall effectiveness of credit risk management.

