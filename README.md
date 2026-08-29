# Credit Risk Assessment: Predicting Loan Default with Machine Learning

## An End-to-End Machine Learning Project

# Project Overview

Credit risk assessment is one of the most important decisions in lending. Approving a customer who later defaults can result in financial loss, while incorrectly identifying a creditworthy customer as high risk can lead to lost business opportunities and unnecessary scrutiny.

This project is my end-to-end exploration of a real-world credit risk problem using the **Home Credit Default Risk dataset**. https://www.kaggle.com/competitions/home-credit-default-risk

I transformed large, relational financial datasets into a machine learning pipeline capable of estimating the probability that an applicant will default on a loan.

The project goes beyond simply training a machine learning model. I explored the practical challenges involved in building a credit-risk system, including:

- Combining multiple financial data sources.
- Managing large datasets.
- Handling missing values and anomalies.
- Creating financial and behavioural features.
- Addressing severe class imbalance.
- Comparing multiple machine learning algorithms.
- Performing hyperparameter tuning.
- Calibrating predicted probabilities.
- Selecting a threshold using business cost.
- Analysing model errors.
- Explaining predictions using SHAP.
- Deploying the final model through an interactive Streamlit application.

The final system uses a **Tuned Class-Weighted CatBoost model**, calibrated using **Isotonic Regression** and deployed with a **business-cost-optimized decision threshold of 0.14**

# The Business Problem

Lenders receive large numbers of loan applications from customers with different financial situations, employment histories, family structures, credit histories, and repayment behaviours.

The challenge is determining which applicants are likely to default.

The objective of this project is to build a machine learning model that estimates:

 **The probability that a loan applicant will default.**

Rather than replacing human decision-making, the model is designed as a **decision-support tool**.

## Dataset

This project uses the **Home Credit Default Risk dataset**, originally published as a Kaggle machine learning competition.

The dataset contains information about:

- Current loan applications.
- Customer demographics.
- Previous Home Credit loans.
- External credit bureau records.
- Monthly loan payment behaviour.
- Credit card activity.
- POS and cash loan history.
- Installment payment records.

The original dataset contains **eight relational tables**:

| Table | Description |
|---|---|
| `application_train` | Main applicant-level training data containing the target variable |
| `application_test` | Applicant-level test data without target labels |
| `bureau` | Previous loans from other financial institutions |
| `bureau_balance` | Monthly repayment status of bureau loans |
| `previous_application` | Previous loan applications with Home Credit |
| `POS_CASH_balance` | Monthly POS and cash loan history |
| `credit_card_balance` | Monthly credit card history |
| `installments_payments` | Historical installment payment records |

The data was accessed through a Hugging Face mirror because of Kaggle API account verification issues.

### Relational Data Challenge

The relational nature of the dataset created one of the major challenges in this project.

A single customer could appear:

- Once in the main application table.
- Multiple times in previous loan applications.
- Multiple times in credit bureau records.
- Hundreds of times across monthly financial history.

Therefore, the secondary datasets could not be merged directly with the main applicant-level dataset. Each secondary dataset first had to be **aggregated to the applicant level using `SK_ID_CURR`** before being merged with the main application data.

This aggregation process transformed historical financial behaviour into applicant-level features, including summary statistics such as:

- Mean values.
- Minimum and maximum values.
- Total amounts.
- Counts of previous records.
- Payment behaviour.
- Loan repayment history.
- Credit utilisation patterns.
- Contract status distributions.

The final aggregated features were then merged with the main application dataset to create a single machine learning dataset where each row represents one loan applicant.

# Notebook 1 - Data Loading and Aggregation

This notebook focused on transforming the raw Home Credit relational datasets into applicant-level modelling data. The process involved memory optimisation, aggregation of historical records, creation of behavioural and recency-based features, and merging the resulting datasets with the main application data.

#### Memory Optimisation

Because some datasets are very large, particularly *bureau_balance* with approximately 27 million rows, memory-efficient data types were used during loading. A custom *downcast()* function converted:
- float64 to float32
- int64 to smaller integer types
- bool to int8

Large intermediate objects were also deleted and garbage collection was used after processing.

#### Aggregation of Historical Data

The following secondary datasets were aggregated to the applicant level using **SK_ID_CURR**:

- *bureau_balance*
- *bureau*
- *POS_CASH_balance*
- *credit_card_balance*
- *previous_application*
- *installments_payments*

Standard aggregation functions such as **mean, sum, minimum, maximum, count, standard deviation, and number of unique values** were used depending on the dataset.

* ***bureau_balance*** was first aggregated by *SK_ID_BUREAU* before being merged with *bureau* and subsequently aggregated by applicant.

Categorical variables were one-hot encoded where appropriate, with means of the dummy variables representing category proportions.

#### Behavioural and Recency Features

Additional features were created to capture recent financial behaviour rather than relying only on lifetime averages.

An exponential decay weighting scheme was used so that recent records received greater importance.

Examples included:

- Recency-weighted bureau debt, overdue amounts, and days overdue
- Recent POS/Cash days-past-due behaviour
- Recent credit card utilisation and delinquency
- Recent previous-application refusal rates and credit amounts
- Recent installment lateness and late-payment rates

Additional trend and recent-period features were created using windows such as the **last 6 months, 12 months, and 2 years**, depending on the dataset.

#### Previous Applications and Installment Behaviour

The **previous_application** dataset was aggregated at applicant level using numerical summaries, categorical proportions, application counts, and recency-based features.

For `installments_payments`, repayment behaviour features were created:
- `DAYS_LATE` — whether payments were made late or early
- `AMT_PAYMENT_DIFF` — difference between expected and actual payment
- `AMT_PAYMENT_RATIO` — proportion of the installment amount that was paid

These features helped capture repayment quality and delinquency behaviour.

#### Merging and Validation

The aggregated datasets were saved as Parquet files and left-merged onto the main application datasets using **SK_ID_CURR**.

After every merge, row counts were checked to ensure that applicants were neither duplicated nor lost.

The training and test datasets were then aligned to ensure they contained the same modelling features. Missing test columns were added and filled with zero.

## Final Outputs

The final applicant-level datasets were saved as:
- `train_final.parquet`
- `test_final.parquet`

These datasets combined the main application information with historical credit, loan, credit-card, POS/Cash, previous-application, and installment-payment behaviour and were used as the foundation for the subsequent data cleaning and feature engineering stages.


# Notebook 2 _ Exploratory Data Analysis and Data Cleaning

In this notebook I focused on understanding the structure, quality, distribution, and predictive characteristics of the applicant-level datasets produced in Notebook 1. The analysis covered missing values, data anomalies, outliers, feature relationships, distributions, constant features, and class imbalance before model development.

#### 1. Dataset Overview

The processed datasets were loaded from Notebook 1:

- train_final.csv
- test_final.csv

The initial datasets contained:

- Training: **307,511 rows × 617 columns**
- Testing: **48,744 rows × 616 columns**

The difference of one column is due to ***TARGET***, which is available only in the training data.

No duplicate rows were found in the training dataset.

#### 2. Target Distribution

The target variable was highly imbalanced:

- **91.93%** non-defaulters (*TARGET = 0*)
- **8.07%** defaulters (*TARGET = 1*)

This imbalance was an important consideration for model development and evaluation because a model could achieve high overall accuracy while performing poorly at identifying actual defaulters.
<img width="610" height="469" alt="image" src="https://github.com/user-attachments/assets/830d0fb4-e054-4eed-8d6f-ed236d30c57f" />


#### 3. Missing Value Analysis and Treatment

Missing-value analysis showed that missing data was widespread:

- Training: **562 of 617 columns** contained missing values
- Testing: **559 of 616 columns** contained missing values

Different imputation strategies were used depending on the meaning of the feature.

- Count, sum, and unique-count features were generally filled with **0**, where missingness represented no recorded historical activity.
- Day-based and other continuous numerical features were filled using the **median**.
- Categorical variables were filled with a *"Missing"* category.
- For numerical features with more than **70% missing values**, additional *_WAS_MISSING* indicators were created to preserve information about the original missingness.

The imputation strategy was applied consistently to both training and testing data, with the median values learned from the training data.

After treatment, both datasets contained **zero remaining missing values**.

#### 4. Data Anomaly Detection

Anomaly checks identified the known *365243* placeholder in *DAYS_EMPLOYED*. This value represents unavailable employment information rather than a genuine employment duration.

A binary feature, *DAYS_EMPLOYED_ANOM*, was created to preserve this information. The placeholder was then replaced with *NaN* and the resulting missing value was filled using the median of valid *DAYS_EMPLOYED* values.

This prevented the placeholder from being treated as a real numerical value while retaining information about its occurrence.
<img width="599" height="434" alt="image" src="https://github.com/user-attachments/assets/d98bcbf2-6b65-401d-91a9-025b754d4531" />

#### 5. Outlier Analysis

Outliers were examined using the **Interquartile Range (IQR)** method.

The analysis identified several features with substantial numbers of extreme observations, particularly property-related variables and aggregated installment-payment features.

Rather than automatically removing all detected outliers, the analysis was used to understand the distributions and determine whether extreme values represented genuine observations or potential data-quality issues.
<img width="1790" height="789" alt="image" src="https://github.com/user-attachments/assets/adbcfd0a-3262-4eb5-ad55-7d1fde174913" />


## 6. Feature Distribution Analysis

Key numerical features were examined using descriptive statistics, histograms, and skewness analysis.

Most financial variables were positively skewed. In particular:

- *AMT_INCOME_TOTAL* was extremely right-skewed.
- *AMT_CREDIT*, *AMT_ANNUITY*, and *AMT_GOODS_PRICE* were moderately right-skewed.
- *CNT_CHILDREN* was also positively skewed.
- *DAYS_BIRTH* was comparatively close to symmetric.

These distributions were considered when determining appropriate preprocessing and modelling strategies.

#### 7. Correlation Analysis

Feature correlations with ***TARGET*** were examined to identify variables associated with loan default.

The strongest negative correlations included:

- ***EXT_SOURCE_2***: *-0.1603*
- ***EXT_SOURCE_3***: *-0.1559*
- ***EXT_SOURCE_1***: *-0.0989*

Positive relationships included features related to recent repayment behaviour and previous application outcomes, such as installment late-payment rates and previous application refusal rates.

The correlations were generally modest, indicating that default risk is influenced by multiple factors rather than a single feature.

#### 8. Constant Feature Removal

Constant features were identified because they provide no variation and therefore no useful information for model learning.

One constant feature was found:***PREV_NAME_GOODS_CATEGORY_HOUSE CONSTRUCTION_MEAN***

It was removed from both training and testing datasets.


The cleaned datasets provided the foundation for the subsequent feature engineering, model development, and evaluation stages.

INCOMPLETE
