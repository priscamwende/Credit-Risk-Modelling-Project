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


INCOMPLETE
