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


# Notebook 2 - Exploratory Data Analysis and Data Cleaning

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

## Notebook 3 - Feature Engineering

This notebook focused on transforming the cleaned applicant-level dataset into a more informative modelling dataset by creating new features based on **financial relationships, household structure, age, employment history, external credit information, loan characteristics, social-risk indicators, and applicant contact information**.

The objective was to move beyond the raw variables and create features that better represent the financial circumstances and potential credit risk of each applicant.

The notebook starts with the cleaned datasets produced during the EDA and data-cleaning stage. At this point, missing values, missingness indicators, data-quality issues, and constant features had already been addressed.

The cleaned datasets contained:

- Training: **307,511 rows × 733 columns**
- Testing: **48,744 rows × 732 columns**

`TARGET` was separated from the training features and `SK_ID_CURR` was temporarily removed so that neither the target nor the applicant identifier would be used as an input feature during feature engineering.

The initial feature set contained:

- **715 numerical features**
- **16 categorical features**

#### 1. Financial Ratio Features

Several features were created by combining existing financial variables into ratios.

These included:

- ***CREDIT_INCOME_RATIO*** - credit amount relative to annual income
- ***ANNUITY_INCOME_RATIO*** - annual loan payment relative to income
- ***CREDIT_GOODS_RATIO*** - credit amount relative to the value of the goods being financed
- ***GOODS_INCOME_RATIO*** - goods price relative to income
- ***ANNUITY_CREDIT_RATIO*** - annual annuity relative to the credit amount

These ratios provide more context than absolute financial amounts alone.

For example, a loan of the same size can represent very different levels of financial burden for applicants with different incomes. Ratio-based features therefore provide the model with information about **affordability, repayment burden, and borrowing behaviour**.

A ***safe_divide()*** helper function was used to avoid division-by-zero problems when creating these features.

#### 2. Household and Family Features

The notebook created features that account for household size and family composition.

These included:

- *INCOME_PER_PERSON*
- *INCOME_PER_CHILD*
- *CREDIT_PER_PERSON*
- *CHILDREN_RATIO*

These features provide additional information about an applicant's financial resources relative to the number of people or dependants in the household.

For example, total income alone does not provide the same level of information as income considered relative to family size. These engineered variables therefore provide a more contextual measure of household financial capacity.

#### 3. Age and Time-Based Features

Several variables in the Home Credit dataset are represented as negative numbers of days relative to the loan application date.

To make these variables easier to interpret, they were converted from days into years.

The following features were created:

- *AGE_YEARS*
- *EMPLOYMENT_YEARS*
- *REGISTRATION_YEARS*
- *ID_PUBLISH_YEARS*

These transformations preserve the underlying information while making the variables easier to interpret in terms of an applicant's age, employment history, registration history, and identification-record history.

#### 4. Employment and Stability Features

The notebook also created relative measures of employment and registration history.

These included:

- *EMPLOYMENT_AGE_RATIO*
- *REGISTRATION_AGE_RATIO*

These ratios compare employment or registration duration with the applicant's age.

Rather than looking only at how many years an applicant has been employed or registered, the ratios provide additional context about how significant those periods are relative to the applicant's lifetime.

#### 5. External Credit Score Features

The three external credit variables:

- *EXT_SOURCE_1*
- *EXT_SOURCE_2*
- *EXT_SOURCE_3*

were combined into several summary features:

- *EXT_SOURCE_MEAN*
- *EXT_SOURCE_MAX*
- *EXT_SOURCE_MIN*
- *EXT_SOURCE_STD*
- *EXT_SOURCE_SUM*

These features were particularly important because the exploratory analysis in Notebook 2 showed that the external credit variables had some of the strongest relationships with the target.

Combining the external scores allowed the model to capture an applicant's overall external credit profile rather than relying only on the individual scores.


#### 6. Loan Duration Feature

A ***LOAN_TERM*** feature was created using the relationship between:

- *AMT_CREDIT*
- *AMT_ANNUITY*

This provides an approximate measure of the repayment duration implied by the loan amount and annuity.

The purpose was to give the model an additional representation of the loan's repayment structure rather than treating the credit amount and annuity independently.

#### 7. Social Risk Features

The Home Credit dataset contains information about observed payment problems and defaults within an applicant's social circle.

Two aggregate features were created:

- *TOTAL_SOCIAL_OBS*
- *TOTAL_SOCIAL_DEF*

These combine the 30-day and 60-day social-circle observations and default indicators.

The purpose was to create broader measures of the applicant's social credit environment rather than using the individual 30-day and 60-day variables separately.

#### 8. Contact Information Feature

Several binary contact indicators were combined into:

`TOTAL_CONTACT_FLAGS`

This feature represents the total number of contact methods associated with an applicant, including indicators such as:

- Mobile phone
- Employer phone
- Work phone
- Contact mobile
- Phone
- Email

Combining these indicators creates a compact summary of the applicant's available contact information.

## 9. Engineered Feature Summary

A total of **24 new features** were created.

| Feature Group | Examples |
|---|---|
| Financial ratios | *CREDIT_INCOME_RATIO*, *ANNUITY_INCOME_RATIO*, *CREDIT_GOODS_RATIO* |
| Household affordability | *INCOME_PER_PERSON*, *INCOME_PER_CHILD*, *CREDIT_PER_PERSON* |
| Family composition | *CHILDREN_RATIO* |
| Age and time | *AGE_YEARS*, *EMPLOYMENT_YEARS*, *REGISTRATION_YEARS* |
| Employment stability | *EMPLOYMENT_AGE_RATIO*, *REGISTRATION_AGE_RATIO* |
| External credit | *EXT_SOURCE_MEAN*, *EXT_SOURCE_MIN*, *EXT_SOURCE_MAX*, *EXT_SOURCE_STD*, *EXT_SOURCE_SUM* |
| Loan structure | *LOAN_TERM* |
| Social risk | *TOTAL_SOCIAL_OBS*, *TOTAL_SOCIAL_DEF* |
| Contact information | *TOTAL_CONTACT_FLAGS* |

#### 10. Correlation Analysis of Engineered Features

The newly created features were evaluated against ***TARGET*** to understand whether they provided additional linear signal for predicting loan default.

The strongest negative correlations were:

| Feature | Correlation with TARGET |
|---|---:|
| EXT_SOURCE_SUM | **-0.221** |
| EXT_SOURCE_MEAN | **-0.221** |
| EXT_SOURCE_MIN | **-0.193** |
| EXT_SOURCE_MAX | **-0.174** |
| AGE_YEARS | **-0.078** |
| EMPLOYMENT_YEARS | **-0.063** |
| ID_PUBLISH_YEARS | **-0.051** |
| EMPLOYMENT_AGE_RATIO | **-0.050** |
| REGISTRATION_YEARS | **-0.042** |
| LOAN_TERM | **-0.032** |

The strongest positive relationships included:

| Feature | Correlation with TARGET |
|---|---:|
| EXT_SOURCE_STD | **0.078** |
| CREDIT_GOODS_RATIO | **0.068** |
| TOTAL_SOCIAL_DEF | **0.033** |
| CHILDREN_RATIO | **0.021** |
| TOTAL_CONTACT_FLAGS | **0.021** |
| ANNUITY_INCOME_RATIO | **0.014** |

The combined external-credit features produced the strongest relationships among the engineered variables. In particular, *EXT_SOURCE_SUM* and *EXT_SOURCE_MEAN* both had a correlation of approximately **-0.221** with the target.

A negative correlation indicates that higher values were associated with a lower observed likelihood of default, while a positive correlation indicates a higher observed likelihood of default.

However, most correlations were relatively small. This does not necessarily mean that the features are unimportant because correlation measures only a **linear relationship between an individual feature and the target**. Machine learning models can capture non-linear relationships and interactions that simple correlation analysis cannot.

#### 11. Feature Engineering Validation

Before saving the final datasets, several quality checks were performed.

The notebook verified that:

- There were **no remaining missing values**.
- There were **no infinite numerical values**.
- The engineered features had been successfully created.
- Summary statistics were inspected for the newly created variables.

These checks ensured that the feature-engineered datasets were suitable for the subsequent modelling stage.

#### 12. Restoring Target and Applicant IDs

After feature engineering and validation, the original *TARGET* and *SK_ID_CURR* columns were added back to the final datasets.

The resulting datasets contained:

- Training: **307,511 rows × 757 columns**
- Testing: **48,744 rows × 756 columns**

The difference of one column is due to *TARGET*, which exists only in the training dataset.

## 13. Final Outputs

The final feature-engineered datasets were saved in Parquet format:

text
output/
├── train_feature_engineered.parquet
└── test_feature_engineered.parquet

INCOMPLETE
