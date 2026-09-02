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

## Notebook 4 - Model Selection, Training, Calibration and Business Optimization

This notebook represents the **model development and decision-making stage** of the credit risk project.

Using the feature-engineered dataset produced in Notebook 3, my objective was to identify a machine learning model capable of distinguishing between customers who are likely to default and those who are unlikely to default.

The modelling process went beyond simply selecting the model with the highest accuracy. Because loan default is a highly imbalanced classification problem and the cost of missing a genuine defaulter can be significantly higher than incorrectly flagging a good applicant, the modelling process considered:

- Model discrimination
- Class imbalance
- Cross-validation performance
- Hyperparameter tuning
- Precision and recall
- F1-score
- Probability calibration
- Business-cost-sensitive threshold selection
- Final model packaging and saving

The final selected model was a **Tuned Class-Weighted CatBoost classifier**, calibrated using **Isotonic Regression**, with a **decision threshold of 0.15** under the assumed business cost structure.

#### 1. Modelling Dataset

The notebook begins by loading the feature-engineered datasets created in Notebook 3.

The datasets contained:

- **Training:** 307,511 rows × 757 columns
- **Testing:** 48,744 rows × 756 columns

The *TARGET* variable was separated from the training features, while *SK_ID_CURR* was excluded from model inputs because it is an applicant identifier rather than a predictive feature.

The resulting feature set contained a large mixture of numerical and encoded categorical information derived from the original application data and historical financial records.

#### 2. Train-Validation Split

The training data was divided into training and validation subsets using a **stratified train-validation split**.

Stratification was important because only approximately 8% of applicants were defaulters. Maintaining a similar class distribution across the training and validation datasets provided a more reliable evaluation of model performance.

The validation dataset was kept separate from model fitting so that it could be used to evaluate how well the models performed on previously unseen observations.

#### 3. Baseline Model Development

Several machine learning algorithms were initially evaluated to establish baseline performance.

The models included:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

These models were selected because they provide a useful combination of:

- Linear modelling
- Bagging-based tree models
- Gradient boosting
- High-performance categorical-feature modelling

The initial comparison used metrics including:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Because of the strong class imbalance, **ROC-AUC was given more importance than accuracy** when comparing the models.

#### 4. Baseline Model Results

The initial validation comparison produced the following results:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| LightGBM | 91.99% | 58.53% | 2.56% | 4.90% | **0.7823** |
| CatBoost | 92.02% | **65.96%** | 2.50% | 4.81% | 0.7803 |
| Logistic Regression | 92.01% | 56.76% | 4.23% | 7.87% | 0.7786 |
| XGBoost | 91.88% | 48.13% | **6.75%** | **11.84%** | 0.7739 |
| Random Forest | 91.93% | 50.77% | 0.66% | 1.31% | 0.7352 |

LightGBM achieved the highest baseline ROC-AUC of **0.7823**, while CatBoost followed closely with **0.7803**.

However, the very low recall values demonstrated an important problem: using the default classification threshold caused the models to identify very few actual defaulters.

This showed why **accuracy alone was not an appropriate model-selection criterion** for this problem.

#### 5. Handling Class Imbalance

The dataset contains substantially more non-defaulters than defaulters.

To investigate the effect of class imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** was tested.

SMOTE was used to generate additional synthetic observations for the minority class.

Importantly, SMOTE was incorporated into modelling pipelines so that oversampling was applied to the training data rather than the validation data.

This helped prevent information from the validation set from influencing the model training process.

#### 6. Five-Fold Stratified Cross-Validation

After the initial model comparison, the strongest candidate models were evaluated using **five-fold Stratified Cross-Validation**.

The purpose was to determine whether the initial results were stable across different subsets of the data.

The models were evaluated using ROC-AUC.

| Model | Mean CV ROC-AUC | Standard Deviation |
|---|---:|---:|
| **LightGBM** | **0.7770** | 0.0026 |
| Logistic Regression | 0.7751 | 0.0025 |
| CatBoost | 0.7734 | 0.0027 |
| XGBoost | 0.7655 | 0.0025 |

LightGBM achieved the highest mean cross-validation ROC-AUC of **0.7770**.

However, the differences between LightGBM, Logistic Regression and CatBoost were relatively small.

As a result, Logistic Regression and CatBoost were retained as candidates for further investigation, while XGBoost was excluded from subsequent tuning because it produced the lowest cross-validation ROC-AUC.

The cross-validation results also demonstrated that model selection should not depend on a single train-validation split.

#### 7. SMOTE versus Class Weighting

Two approaches for addressing class imbalance were considered:

### SMOTE

SMOTE creates synthetic examples of the minority class.

### Class Weighting

Class weighting increases the penalty associated with misclassifying observations from the minority class.

For the later tuning stage, **class weighting was preferred** because it avoids creating synthetic observations and allows the models to learn directly from the original observations while placing greater importance on defaulters.

This was particularly useful for the tree-based models used later in the project.

#### 8. Hyperparameter Tuning

Hyperparameter tuning was performed on the strongest candidate models.

**RandomizedSearchCV** was used to search through combinations of model parameters.

This approach was selected instead of testing every possible combination because the dataset contains hundreds of features and the computational cost of exhaustive grid searches can become very high.

The main models tuned were:

- LightGBM
- CatBoost
- Logistic Regression

The tuning process focused on parameters that influence model complexity, learning behaviour and class imbalance.

#### 9. Tuned LightGBM

The best LightGBM configuration achieved a cross-validation ROC-AUC of:

**0.7836**

The selected parameters included:

- `n_estimators = 300`
- `learning_rate = 0.1`
- `num_leaves = 15`
- `max_depth = 10`
- `min_child_samples = 50`
- `colsample_bytree = 0.8`

LightGBM remained one of the strongest models and demonstrated particularly strong recall when class weighting was applied.

#### 10. Tuned CatBoost

CatBoost achieved a slightly higher cross-validation ROC-AUC:

**0.7839**

The best configuration included:

- `iterations = 300`
- `learning_rate = 0.05`
- `depth = 8`
- `l2_leaf_reg = 7`
- `class_weights = [1, 5]`

CatBoost produced the strongest overall balance between precision and recall among the tuned models.

Its cross-validation ROC-AUC of **0.7839** was also slightly higher than the tuned LightGBM result of **0.7836**.

#### 11. Tuned Model Comparison

After tuning, the models were compared using both discrimination and classification metrics.

The results showed an important trade-off.

**LightGBM** produced substantially higher recall, meaning it identified more of the actual defaulters.

However, this came at the cost of lower precision and accuracy.

**CatBoost** achieved a better balance between precision and recall and produced the highest F1-score among the tuned models.

This was particularly important because the objective was not simply to maximize the number of detected defaulters.

The model also needed to avoid generating an unnecessarily large number of false alarms.

The notebook therefore moved beyond model performance at the default threshold and investigated **probability calibration and business-cost optimization**.

#### 12. Probability Calibration

Machine learning models produce probability estimates, but those probabilities are not always reliable.

For example, a model predicting a default probability of 0.80 should ideally be correct approximately 80% of the time for observations receiving that probability.

To evaluate and improve probability reliability, calibration was performed on:

- Tuned Logistic Regression
- Tuned LightGBM
- Tuned CatBoost

Two calibration approaches were compared:

- **Sigmoid calibration**
- **Isotonic calibration**

The **Brier Score** was used as the main measure of probability calibration.

A lower Brier Score indicates better-calibrated probabilities.

#### 13. Calibration Results

Before calibration:

| Model | Brier Score |
|---|---:|
| Logistic Regression | 0.0808 |
| CatBoost | 0.1009 |
| LightGBM | 0.1718 |

After calibration, the probability estimates improved substantially.

The best calibrated results were:

| Model | Calibration Method | Brier Score | ROC-AUC |
|---|---|---:|---:|
| **Tuned CatBoost** | **Isotonic** | **0.0651** | **0.7968** |
| Tuned LightGBM | Isotonic | 0.0663 | 0.7812 |
| Tuned Logistic Regression | Isotonic | 0.0670 | 0.7753 |

Tuned CatBoost with Isotonic calibration achieved the **lowest Brier Score and highest calibrated ROC-AUC** among the evaluated models.

This made it the strongest candidate for the final credit-risk system.

#### 14. Why a 0.50 Threshold Was Not Used

A major finding from the project was that the default classification threshold of **0.50** was inappropriate for this credit-risk problem.

At a threshold of 0.50, the calibrated models identified very few actual defaulters.

This occurred because the dataset is highly imbalanced and because the probability estimates represent relatively low default probabilities for most applicants.

Therefore, simply asking whether:

Predicted Probability >= 0.50

#### 15. Business Cost Framework

The project introduced a simple business-cost framework to make the threshold selection more realistic.

The assumptions were:

- **False Positive Cost = 1**
- **False Negative Cost = 5**

This means that incorrectly flagging a low-risk customer was considered to have a cost of 1, while failing to identify a genuine defaulter was considered five times more costly.

The total business cost was calculated as:

Total Cost = (False Positives × 1) + (False Negatives × 5)

#### 16. Cost-Sensitive Threshold Optimization

Multiple probability thresholds were tested to determine how the classification decision changed as the threshold was adjusted.

The analysis showed that lowering the threshold increased recall but also increased the number of false positives.

This created the expected trade-off:

- **Higher threshold** → fewer false positives but more missed defaulters.
- **Lower threshold** → more detected defaulters but more false positives.

The objective was to identify the threshold that minimized the assumed total business cost.

Rather than automatically using the standard **0.50 threshold**, the project selected the threshold based on the financial consequences of false-positive and false-negative predictions.

## 17. Final Cost-Sensitive Comparison

The best results from the three calibrated models were:

| Model | Best Threshold | Precision | Recall | F1-score | False Positives | False Negatives | Total Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Tuned Class-Weighted CatBoost** | **0.15** | **28.11%** | **44.81%** | **34.55%** | 1,138 | 548 | **3,878** |
| Tuned Class-Weighted LightGBM | 0.20 | 33.02% | 31.92% | 32.46% | 643 | 676 | 4,023 |
| Tuned Class-Weighted Logistic Regression | 0.15 | 25.17% | 43.71% | 31.95% | 1,290 | 559 | 4,085 |

Based on the assumed cost structure, **Tuned Class-Weighted CatBoost achieved the lowest total business cost of 3,878**.

It also achieved:

- **44.81% recall**
- **28.11% precision**
- **34.55% F1-score**
- **548 false negatives**
- **1,138 false positives**

Therefore, **CatBoost was selected as the final model**.

## 18. Final Model Selection

The final model was selected based on a combination of:

1. Strong discrimination
2. Good probability calibration
3. Competitive precision
4. Strong recall
5. High F1-score
6. Lowest assumed business cost
7. Practical suitability for a credit-risk decision-support system

The final configuration was:

| Component | Final Selection |
|---|---|
| Model | **Tuned Class-Weighted CatBoost** |
| Calibration | **Isotonic Regression** |
| Decision Threshold | **0.15** |
| False Positive Cost | **1** |
| False Negative Cost | **5** |
| Total Business Cost | **3,878** |
| Precision | **28.11%** |
| Recall | **44.81%** |
| F1-score | **34.55%** |

The model was therefore selected based on **business cost rather than accuracy alone**.


## Notebook 5 - Final Model Evaluation, Interpretation and Deployment

This notebook represents the **final evaluation and validation stage** of the credit risk project.

The purpose of this stage was to evaluate the final selected model on unseen validation data, assess the reliability of its predicted probabilities, apply the business-optimized decision threshold, analyse classification errors, calculate the final business cost, interpret the model using SHAP, and prepare the model for use in a credit-risk application.

The final model package contains the trained **Class-Weighted CatBoost model**, **Isotonic calibration model**, selected decision threshold, training feature columns, categorical-column information, and the top organization categories required for consistent predictions.

### 1. Loading the Final Model Package

The saved final model package was loaded from: **output/final_chosen_credit_risk_model.pkl**

The package contains:

* Final Class-Weighted CatBoost model
* Isotonic calibration model
* Decision threshold of **0.15**
* Training feature columns
* Categorical column information
* Top 15 organization types

The saved model expects **842 training features** and contains **16 categorical columns** used during preprocessing.

The notebook also verified that the saved model package loaded successfully and that the final threshold was **0.15**.

#### 2. Preparing the Evaluation Data

The feature-engineered training dataset from the previous stage was loaded and separated into:

* **Features (`X`)**
* **Target (`y`)**

*SK_ID_CURR* was removed from the model inputs because it is an applicant identifier rather than a predictive feature.

The categorical variables were identified and the *ORGANIZATION_TYPE* variable was grouped into its **15 most common categories**, with the remaining categories grouped into `"Other"`.

I then One hot encoded the categorical variables. After encoding, the evaluation dataset contained:

**307,511 observations × 842 features**

#### 3. Feature Consistency Validation

Before making predictions,  I compared the evaluation features with the features used to train final model. 
The validation confirmed:

* Model expected: **842 columns**
* Evaluation dataset: **842 columns**
* Missing columns: **0**
* Extra columns: **0**
* Exact feature match: **True**

This step was important because a mismatch between the training and evaluation feature structure could produce incorrect model predictions.

#### 4. Train-Validation Split

I divided thr  data  into training and validation sets using a **stratified 80/20 split**.

The resulting datasets were:

| Dataset    |    Rows | Features |
| ---------- | ------: | -------: |
| Training   | 246,008 |      842 |
| Validation |  61,503 |      842 |

I used Stratification to preserve the proportion of defaulters and non-defaulters in both datasets.

I checked the  validation data to ensure that:

* There were **0 missing values**
* There were **0 non-numeric columns**
* All **842 expected model features** were present

#### 5. Raw Probability Predictions

The final CatBoost model generated a probability of default for each of the **61,503 validation customers**.

The raw probabilities ranged from approximately:

* **Minimum:** 1.04%
* **Maximum:** 92.96%
* **Average:** 24.86%

The purpose of generating probabilities was to estimate **how likely each applicant was to default**, rather than producing only a simple default/non-default classification.

These raw probabilities were not immediately used for the final lending decision because probability calibration was applied first.

#### 6. Isotonic Probability Calibration

The raw CatBoost probabilities were passed through the selected **Isotonic Calibration** model.

The purpose of calibration was to make the predicted probabilities more closely aligned with the actual observed default rate.

After calibration:

* Minimum probability: **0%**
* Maximum probability: **100%**
* Mean calibrated probability: **8.06%**

The average probability decreased from approximately **24.86% before calibration to 8.06% after calibration**.

The calibrated average was much closer to the dataset's actual default rate of approximately **8.07%**.

This suggests that the raw CatBoost model was overestimating default probabilities and that the calibration step substantially improved the reliability of the probability estimates.

#### 7. Evaluation of Calibrated Probabilities

The calibrated probabilities were evaluated using:

* **Brier Score**
* **ROC-AUC**

The final results were:

| Metric      |     Result |
| ----------- | ---------: |
| Brier Score | **0.0655** |
| ROC-AUC     | **0.7880** |

The Brier Score of **0.0655** indicates reasonably reliable probability estimates, with lower values representing better calibration.

The ROC-AUC of **0.7880** shows that the model retained good ability to distinguish between customers who defaulted and those who did not after calibration.

Therefore, calibration improved probability reliability while maintaining useful discriminatory performance.

#### 8. Applying the Final Decision Threshold

I converted the  final calibrated probabilities  into classification decisions using the selected business-optimized threshold of:

**0.15**

Applicants with a calibrated probability of default of **15% or higher** were classified as high risk.

The validation predictions were:

| Prediction |  Customers | Percentage |
| ---------- | ---------: | ---------: |
| Lower Risk |     53,611 |     87.17% |
| High Risk  |      7,892 |     12.83% |
| **Total**  | **61,503** |   **100%** |

This demonstrates how the calibrated probabilities were converted into actionable risk classifications.

#### 9. Final Confusion Matrix

The final model was evaluated using the confusion matrix at the **0.15 decision threshold**.

|                          | Predicted Non-Defaulter | Predicted Defaulter |
| ------------------------ | ----------------------: | ------------------: |
| **Actual Non-Defaulter** |              **50,869** |           **5,669** |
| **Actual Defaulter**     |               **2,742** |           **2,223** |

The results can be interpreted as follows:

* **True Negatives (50,869):** Customers who did not default and were correctly classified as lower risk.
* **False Positives (5,669):** Customers who did not default but were classified as high risk.
* **False Negatives (2,742):** Customers who defaulted but were classified as lower risk.
* **True Positives (2,223):** Customers who defaulted and were correctly classified as high risk.

The confusion matrix demonstrates the trade-off created by the 0.15 threshold.

The model identifies a substantial number of actual defaulters, but it also produces false positives. At the same time, some actual defaulters remain undetected.

#### 10. Final Validation Metrics

At the selected threshold of **0.15**, the final model achieved:

| Metric      |     Result |
| ----------- | ---------: |
| Accuracy    | **86.32%** |
| Precision   | **28.17%** |
| Recall      | **44.77%** |
| F1-Score    | **34.58%** |
| ROC-AUC     | **0.7880** |
| Brier Score | **0.0655** |

The model achieved a recall of **44.77%**, meaning that it identified approximately 45% of the actual defaulters in the validation dataset.

The precision of **28.17%** means that approximately 28% of the applicants classified as high risk were actual defaulters.

The relatively low precision is partly a result of the lower decision threshold, which was intentionally selected to identify more potential defaulters.

#### 11. Final Business Cost

The project assumed that the consequences of the two types of classification errors were different:

* **False Positive Cost = 1**
* **False Negative Cost = 5**

The final business cost was calculated as:

**Total Business Cost = (False Positives × 1) + (False Negatives × 5)**

Using the final confusion matrix:

**Total Business Cost = (5,669 × 1) + (2,742 × 5)**

**Total Business Cost = 19,379**

The higher cost assigned to false negatives reflects the assumption that failing to identify an actual defaulter is more costly to the lender than incorrectly flagging a non-defaulter.

This business-cost framework was therefore used alongside statistical performance metrics when evaluating the final model.

#### 12. SHAP Model Interpretation

 I used SHAP  to understand which features contributed most strongly to the final CatBoost model's predictions.

The analysis helped identify the variables that had the greatest influence on predicted credit risk.

Important features included variables related to:

* External credit information
* Previous repayment behaviour
* Installment payment lateness
* Loan and credit structure
* Previous loan repayment timing
* Loan affordability

Examples of important features included:

* *EXT_SOURCE_MEAN*
* *EXT_SOURCE_SU*`
* *INSTAL_RECENCY_WEIGHTED_LATE_RATE*
* *CREDIT_GOODS_RATIO*
* *PREV_DAYS_LAST_DUE_1ST_VERSION_MAX*
* *AMT_ANNUITY*

The SHAP analysis provides an explanation of how the model uses these variables when making predictions. These relationships should be interpreted as **model associations rather than causal relationships**.

#### 13. Final Model Package

After evaluation, the final model package was retained with the components required to reproduce predictions consistently.

The final package includes:

| Component                   | Final Selection             |
| --------------------------- | --------------------------- |
| Model                       | **Class-Weighted CatBoost** |
| Calibration                 | **Isotonic Regression**     |
| Decision Threshold          | **0.15**                    |
| Training Features           | **842**                     |
| Categorical Columns         | **16**                      |
| Top Organization Categories | **15**                      |

The model was saved as:

*final_chosen_credit_risk_model.pkl*

This package provides the model and supporting preprocessing information required for deployment.

### 14. Deployment Readiness

The final model package was prepared for use in an interactive credit-risk application.

The deployment workflow uses the saved model package to:

1. Receive applicant information.
2. Apply the required preprocessing.
3. Align the applicant features with the model's **842 training columns**.
4. Generate a raw probability of default.
5. Apply Isotonic calibration.
6. Compare the calibrated probability with the **0.15 threshold**.
7. Produce a final risk classification.

This allows the machine learning model to move beyond notebook-based analysis into a practical decision-support application.

#### 15. Final Project Outcome

The completed project developed an end-to-end credit-risk modelling workflow beginning with raw relational financial data and ending with a calibrated and deployable machine learning model.

The final workflow includes:

**Data Aggregation → Data Cleaning → Feature Engineering → Model Comparison → Class Imbalance Handling → Hyperparameter Tuning → Probability Calibration → Threshold Optimization → Model Evaluation → SHAP Interpretation → Model Packaging → Deployment**

The final **Class-Weighted CatBoost model with Isotonic calibration** achieved a **ROC-AUC of 0.7880** and a **Brier Score of 0.0655** on the validation data.

Using a business-optimized threshold of **0.15**, the model achieved:

* **86.32% Accuracy**
* **28.17% Precision**
* **44.77% Recall**
* **34.58% F1-Score**
* **0.7880 ROC-AUC**
* **0.0655 Brier Score**
* **19,379 Total Business Cost**

The project demonstrates that credit-risk modelling should not focus on accuracy alone. A practical lending model must also consider **probability reliability, class imbalance, false-positive and false-negative costs, model interpretability, and deployment requirements**.

The final model is therefore intended as a **decision-support system**, helping lenders identify potentially higher-risk applicants while allowing human decision-makers to consider the model's predictions alongside other relevant credit information and business policies.

