# Machine Learning for Credit Risk Modeling: Predicting Customer Loan Default Using Financial and Behavioral Data

## Project Overview

Credit risk assessment is one of the most important tasks performed by financial institutions. Every lending decision involves uncertainty - approving loans for customers who are unlikely to repay them can lead to significant financial losses, while rejecting customers who are capable of repaying reduces business opportunities and financial inclusion.

In this project, I built a machine learning model that predicts whether a customer is likely to default on a loan, using the Home Credit Default Risk dataset. I combined customer demographic information with historical financial behaviour collected from multiple data sources to improve the accuracy of credit risk assessment.

Rather than relying on traditional rule-based credit scoring, I applied machine learning techniques to uncover the more complex relationships between customer characteristics and repayment behaviour. I split the work into four notebooks that mirror the natural stages of the project - data loading, exploratory data analysis, feature engineering, and model selection/training - so each stage could be developed, checked, and re-run independently. The final notebook goes beyond a baseline model comparison to include class-imbalance handling, hyperparameter tuning, threshold tuning, calibration, error analysis, SHAP-based interpretability, and a fairness check, ending with a saved, production-ready model and a set of test-set predictions.

## Business Problem

Financial institutions receive thousands of loan applications from customers with very different financial backgrounds and repayment histories. Figuring out which applicants are likely to repay is a genuinely hard problem, especially once customer information is scattered across multiple data sources.

The Home Credit Default Risk dataset gave me customer application information alongside credit bureau records, previous loan applications, installment payment history, POS/Cash loan history, and credit card balances. These sources hold a lot of valuable signal about a customer's financial behaviour, but they needed extensive preprocessing before I could use them for modelling.

My goal was to build a classification model capable of estimating the probability that a customer will default, so that lenders can make more accurate, data-driven credit decisions.

## Dataset Description

I worked with the Home Credit Default Risk dataset, published on Kaggle in 2018 by Home Credit Group, a consumer lender that serves largely unbanked and underbanked populations. The competition challenged participants to predict how capable each applicant is of repaying a loan, using their current application together with previous credit and repayment history. Since I ran into account verification issues with the Kaggle API, I downloaded the data from a Hugging Face mirror of the competition instead, which preserved the same structure and features as the original.

I pulled down 8 tables in total:

- **application_train / application_test** - the primary applicant-level data
- **bureau** - an applicant's past loans at other credit institutions
- **bureau_balance** - monthly status history for those bureau loans (~27 million rows, the largest table in the project)
- **previous_application** - an applicant's previous loan applications with Home Credit itself
- **POS_CASH_balance** - monthly balance history on POS/cash loans
- **credit_card_balance** - monthly credit card account history
- **installments_payments** - the detailed record of every installment payment made

Since customers can appear many times in the five secondary tables (once per past loan, once per month of history, and so on), I aggregated each of these down to one row per applicant before merging everything onto the main application table. Before I began the aggregation, I ran two sanity checks: confirming all 8 files were present and that the TARGET column existed, then counting the rows in each file so I understood how much work each table would need.

## Notebook 1 - Data Loading & Aggregation

This is where I turned eight raw, messy tables into a single modelling-ready file.

**Downcasting for memory.** Because some of these tables are enormous (bureau_balance alone has ~27 million rows), I wrote a `downcast()` helper that shrinks each column to the smallest dtype that can hold it without losing information - float64 becomes float32, oversized integer types get trimmed down, and so on. I used this throughout the pipeline to keep memory usage manageable.

**Aggregating the five secondary tables.** For each table, I one-hot encoded the categorical columns and grouped by the applicant (or loan) ID, computing summary statistics - mean, sum, min, max, and count - that describe each customer's history:

- `bureau_balance` first gets aggregated up to one row per `SK_ID_BUREAU`, since it's keyed at the loan level, not the applicant level.
- `bureau` then folds in those `bureau_balance` summaries and gets aggregated again, this time down to one row per `SK_ID_CURR`.
- `POS_CASH_balance` already carries `SK_ID_CURR` directly, so I aggregated it straight to one row per applicant, capturing things like average/maximum installments remaining and the share of loans in each contract status.
- `credit_card_balance` gets the same treatment - average, maximum, and total balances, payments, withdrawals, and receivables per applicant.
- `previous_application` needed one extra cleaning step first: the value 365243 is used as a placeholder for "no date available" in several date columns, so I replaced it with NaN before aggregating, otherwise it would have distorted every average I computed from those columns.
- `installments_payments` got two new features before aggregation - `DAYS_LATE` (how many days early or late a payment was) and `AMT_PAYMENT_DIFF` (the gap between the expected and actual payment amount) -
which let me summarise each applicant's payment discipline, not just their raw payment totals.

**Merging everything together.** With each secondary table reduced to one row per applicant, I left-merged all five onto the main application table and checked the row count after every single merge to make sure nothing was silently duplicated or dropped.

**Aligning train and test.** One-hot encoding can occasionally produce a category in the training set that never shows up in the test set (or vice versa), so I reindexed the test set to match the training set's exact columns, filling any gaps with zero, and asserted the column sets matched before saving.

I saved the results as `train_final.csv` and `test_final.csv` - 307,511 applicants × 568 columns in training, and 48,744 applicants × 567 columns in test (the difference being the missing TARGET column in test).

## Notebook 2 - Exploratory Data Analysis

Once I had a single modelling file, I spent this notebook getting to know the data properly before touching it with any preprocessing.

**Class imbalance.** I confirmed the target was heavily imbalanced - 91.93% of applicants repaid their loans, and 8.07% defaulted. This shaped almost every decision from here on: I chose ROC-AUC as my primary evaluation metric rather than accuracy, and planned to handle the imbalance directly during model training.

**Missing values.** I built a `missing_value_report()` function to profile missingness across every column, and the picture was substantial: 513 of 568 features had at least one missing value, and only 55 were complete. But the missingness wasn't evenly spread - only 6 features were missing more than 90% of their values, 145 features fell between 50-90% missing, and the majority (362 features) were under 50% missing. The features with the worst missingness were mostly aggregated previous-loan interest rate variables, while important predictors like `EXT_SOURCE_2`, `AMT_GOODS_PRICE`, and `AMT_ANNUITY` were almost entirely complete. Crucially, the missingness was concentrated in the `PREV_`, `CC_`, and `BUREAU_` feature groups - meaning a missing value there often just reflects an applicant who never had that kind of previous financial activity, rather than a data quality problem. That insight directly shaped how I imputed things.

**Handling the missingness.** I split my imputation strategy by what each feature actually represents:

- Features ending in `_SUM`, `_COUNT`, or `_NUNIQUE` got filled with 0, since a missing value there usually means "no previous activity" rather than "unknown value."
- Everything else numeric (financial amounts, ratios, averages) got filled with the median, since these variables are heavily skewed and a mean or zero-fill would distort them.
- For any feature missing more than 70% of its values, I created a `_WAS_MISSING` binary indicator column before imputing, so the model could still use the fact that the value was absent as a signal in itself. This added 110 new indicator columns.
- Categorical columns got filled with a new "Missing" category rather than being dropped.

After all of this, both datasets had zero remaining missing values.

**Anomaly detection.** I found that `DAYS_EMPLOYED` contained a suspicious repeated value - 365243, which works out to roughly 1,000 years. This is a known Home Credit placeholder meaning "not currently employed," not a real duration. I created a `DAYS_EMPLOYED_ANOM` flag to preserve that signal, then replaced the placeholder with the median of the genuine values.

**Outliers.** I ran an IQR-based outlier scan across the numeric columns and visualised the worst offenders (income, credit amount, annuity, goods price) with boxplots. The outliers I found looked like genuine variation in customer circumstances rather than data errors, so I left them in - the tree-based models I planned to use aren't especially sensitive to them anyway.

**Correlation analysis.** I correlated every numeric feature against TARGET and found that no single feature has a strong linear relationship with default - the strongest correlation I found was only -0.160, from `EXT_SOURCE_2`. The three external credit score features (`EXT_SOURCE_1/2/3`) were consistently the strongest protective factors, while previous-loan refusals, active bureau accounts, and regional risk ratings showed weak positive relationships with default. This told me default risk is driven by a combination of many weak signals rather than any one strong predictor - which is exactly the kind of pattern non-linear models like Random Forest, XGBoost, and LightGBM are built to capture.

**Feature distributions & skewness.** Most of the financial variables were positively skewed - `AMT_INCOME_TOTAL` especially so (skewness ≈ 391.6), since a small number of applicants earn far more than the rest. This confirmed median imputation was the right call, and reinforced that tree-based models (which don't assume normally distributed inputs) were a good fit here.

**Constant features.** I checked for columns with only one unique value across the entire dataset - there was just one, and I dropped it since it carried no information.

**Linear separability.** I plotted random pairs of features against each other, colouring by default status, to see whether a simple straight-line boundary could separate defaulters from non-defaulters. It couldn't - the classes overlapped substantially across almost every pair I looked at, confirming that this problem needs models capable of learning non-linear interactions.

I finished with a final quality check - no missing values, no leftover 365243 placeholders, no infinite values - before saving the cleaned data as `train_clean_eda.parquet` and `test_clean_eda.parquet`.

## Notebook 3 - Feature Engineering

With clean data in hand, I turned to building new features that would give the models more to work with than the raw variables alone.

I created 24 new features across seven themes:

- **Financial ratios** (`CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_GOODS_RATIO`, `GOODS_INCOME_RATIO`, and a fifth affordability ratio) - these capture affordability and repayment burden far better than the raw loan and income figures on their own.
- **Family/household features** (`INCOME_PER_PERSON`, `INCOME_PER_CHILD`, `CREDIT_PER_PERSON`, `CHILDREN_RATIO`) - since income and debt are usually shared across a household, these adjust for family size rather than treating every applicant as a single-income unit.
- **Age features** (`AGE_YEARS`, `EMPLOYMENT_YEARS`, `REGISTRATION_YEARS`, `ID_PUBLISH_YEARS`) - I converted the raw "days before application" fields into more interpretable years.
- **Life-stage ratios** (`EMPLOYMENT_AGE_RATIO`, `REGISTRATION_AGE_RATIO`) - comparing employment and registration duration to age, to capture stability relative to an applicant's lifetime rather than in absolute terms.
- **External credit score aggregates** (`EXT_SOURCE_MEAN`, `MAX`, `MIN`, `STD`, `SUM`) - since the three individual `EXT_SOURCE` columns were already my strongest predictors from the EDA, I combined them into summary statistics to capture an applicant's overall external creditworthiness.
- **Loan term** (`LOAN_TERM`) - an estimate of how long a loan would take to repay, from credit amount over annuity.
- **Social risk features** (`TOTAL_SOCIAL_OBS`, `TOTAL_SOCIAL_DEF`) - combining the 30-day and 60-day social circle observation/default counts into single measures.
- **Contact accessibility** (`TOTAL_CONTACT_FLAGS`) - summing up the six individual contact-method flags into one count.

I validated every new feature for missing values, infinite values (a real risk with ratio features when the denominator is zero), and reasonable distributions before moving on. When I checked their correlation with TARGET, the engineered `EXT_SOURCE_SUM` and `EXT_SOURCE_MEAN` features came out on top at -0.221 - stronger than any of the three original `EXT_SOURCE` columns individually, which was a nice confirmation that the engineering was adding real value rather than just noise.

After a final quality check, I saved the result as `train_feature_engineered.parquet` and `test_feature_engineered.parquet` - 307,511 rows × 702 columns for training.

## Notebook 4 - Model Selection & Training

This is where everything comes together into a trained, tuned, evaluated, and saved production model.

**Cardinality check & encoding.** I profiled the number of unique categories in each categorical feature. Most were low-cardinality (2–19 categories, e.g. `NAME_CONTRACT_TYPE`, `FLAG_OWN_CAR`, and `FLAG_OWN_REALTY` had only 2, while `OCCUPATION_TYPE` had 19), but `ORGANIZATION_TYPE` had 58, so I kept the 15 most frequent categories and grouped everything else into "Other" before one-hot encoding, to avoid creating dozens of rarely-used dummy columns. I one-hot encoded the remaining categorical columns with `pd.get_dummies()` and aligned train/test to guarantee identical columns. Since LightGBM doesn't tolerate certain special characters in feature names (which one-hot encoding introduces), I also ran a column-name cleaning step across both datasets.

**Train/validation split.** I split the training data 80/20 using a stratified split on TARGET, so both sets preserved the original 92/8 class balance - 246,008 rows for training and 61,503 for validation (56,538 non-defaulters and 4,965 defaulters).

**Handling class imbalance with SMOTE.** As a first approach, I applied SMOTE (Synthetic Minority Oversampling Technique) to the training set only, generating synthetic examples of the minority (default) class. I made sure to apply SMOTE after the train/validation split, so no synthetic information could leak into the validation set and inflate my scores artificially.

**Scaling.** I standardised the features with `StandardScaler`, but only for Logistic Regression, since it's the one model in this line-up that's sensitive to feature magnitude - the tree-based models don't need it.

**Baseline model training & comparison.** I trained five models on the SMOTE-balanced training data and evaluated all of them on the untouched validation set using accuracy, precision, recall, F1-score, and ROC-AUC:

| Model | ROC-AUC | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| LightGBM | 0.780 | 0.920 | 0.565 | 0.023 | 0.044 |
| Logistic Regression | 0.777 | 0.920 | 0.546 | 0.036 | 0.067 |
| XGBoost | 0.773 | 0.919 | 0.477 | 0.061 | 0.108 |
| Random Forest | 0.736 | 0.919 | 0.514 | 0.004 | 0.007 |
| Decision Tree | 0.544 | 0.845 | 0.143 | 0.185 | 0.161 |

LightGBM came out on top, with Logistic Regression a close second — a good sign that my engineered features carried enough signal that even a simple linear model could compete with the ensemble methods. I also plotted ROC curves (LightGBM 0.780, Logistic Regression 0.777, and XGBoost 0.773 were nearly indistinguishable, with Random Forest at 0.736 and Decision Tree only slightly better than random guessing at 0.544) and precision-recall curves (LightGBM led with an average precision of 0.274, ahead of Logistic Regression at 0.264 and XGBoost at 0.261).

Despite the strong accuracy numbers (~92% across the top three models), recall was consistently poor - Logistic Regression caught only 4% of true defaulters, and even LightGBM caught just 2.3%. Because defaults are the minority class, all five models were much better at recognising repayers than at flagging the customers who will actually default, which is the group that matters most for a lender.

**Cross-validated baseline comparison.** Because the leaderboard above came from a single 80/20 split, and the top three scores were close enough to be noise, I re-ran the comparison with 5-fold cross-validation (SMOTE applied within each fold to avoid leakage). This confirmed the ranking: LightGBM averaged 0.775 ROC-AUC, Logistic Regression 0.772, and XGBoost 0.762, with small standard deviations showing all three were consistent across folds.

**SMOTE vs. class-weighting.** SMOTE is also considerably more expensive than simply telling a model to weight the minority class more heavily, so I compared the two approaches directly - training Logistic Regression with `class_weight="balanced"` and LightGBM with `is_unbalance=True` on the original, non-resampled training data:

| Model | ROC-AUC | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| LightGBM (is_unbalance) | 0.787 | 0.726 | 0.186 | 0.709 | 0.295 |
| LightGBM (SMOTE) | 0.780 | 0.920 | 0.565 | 0.023 | 0.044 |
| Logistic Regression (class_weight) | 0.777 | 0.711 | 0.177 | 0.705 | 0.283 |
| Logistic Regression (SMOTE) | 0.777 | 0.920 | 0.546 | 0.036 | 0.067 |

Class weighting was a clear win for the goal that actually matters here: class-weighted LightGBM lifted recall from 2.3% to 70.9% and edged ROC-AUC up to 0.787, at the cost of precision dropping to 18.6% (more false alarms). Given that missing an actual defaulter is typically far costlier to a lender than flagging a safe applicant for review, I moved forward with class-weighted LightGBM rather than the SMOTE version.

**Hyperparameter tuning.** I tuned the class-weighted LightGBM with `RandomizedSearchCV` (10 candidates, 3-fold stratified CV, optimising ROC-AUC) over `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, and `colsample_bytree`. The best configuration (`num_leaves=31`, `n_estimators=300`, `min_child_samples=100`, `max_depth=10`, `learning_rate=0.03`, `colsample_bytree=1.0`) reached a cross-validated ROC-AUC of **0.7831**, and on the untouched validation set it scored **0.19 precision / 0.72 recall / 0.29 F1** for the defaulter class at the default 0.5 threshold - essentially matching the pre-tuning class-weighted model's recall while giving a small, consistent lift in ranking quality.

**Threshold tuning.** Since class-weighted LightGBM already outputs well-separated probabilities, I swept the decision threshold from 0.5 down to 0.1 to see how the precision/recall trade-off moved:

| Threshold | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| 0.50 | 0.723 | 0.186 | 0.718 | 0.295 |
| 0.40 | 0.617 | 0.151 | 0.812 | 0.255 |
| 0.30 | 0.485 | 0.125 | 0.896 | 0.219 |
| 0.20 | 0.328 | 0.103 | 0.955 | 0.187 |
| 0.10 | 0.150 | 0.086 | 0.993 | 0.159 |

Lowering the threshold keeps pushing recall toward 100%, but at a steep cost to precision, accuracy, and F1. The default 0.5 threshold gave the best balance (F1 = 0.295) of the options tested, so I kept it as the final operating threshold rather than pushing further into the "flag almost everyone" regime.

**Calibration.** I checked how trustworthy the model's predicted probabilities are (not just its rankings) with a calibration curve and Brier score. The class-weighted LightGBM produced a Brier score of **0.1781** - a reasonable but imperfect calibration, meaning the raw probabilities are directionally useful but shouldn't be read as exact default likelihoods without further calibration if the business use case demands it.

**Confusion matrix & error analysis.** At the final 0.5 threshold, the model produced 15,400 false positives (safe applicants incorrectly flagged) and 1,445 false negatives (defaulters missed) on the validation set. Comparing correctly classified vs. misclassified applicants showed misclassified customers had, on average, slightly lower income and a lower `AMT_GOODS_PRICE`, with the largest behavioural gap being in the model's own predicted probability (0.29 average for correct predictions vs. 0.62 for misclassified ones - i.e. most errors are borderline cases near the decision boundary, not confident model failures). Age and income distribution plots for correct vs. misclassified customers showed broadly similar shapes, without one extreme subgroup driving all the errors.

**Feature importance & SHAP.** Built-in LightGBM feature importance ranked `EXT_SOURCE_MEAN`, `LOAN_TERM`, `ANNUITY_CREDIT_RATIO`, `AMT_ANNUITY`, and `PREV_DAYS_LAST_DUE_1ST_VERSION_MAX` as the top drivers - reinforcing that the engineered ratio and external-score features were pulling real weight alongside the raw application fields. Because raw importance doesn't show *direction* or explain individual decisions, I added SHAP analysis: a global summary plot over a 2,000-row sample of the validation set for overall feature impact and direction, plus individual force-plot explanations for one concrete false-negative and one false-positive case, in the style of a reason code a credit risk team could review for a specific declined or flagged applicant.

**Fairness check.** I checked defaulter recall (the model's ability to catch actual defaulters) across gender and age groups, since an uneven false-negative rate across protected groups would be a real business and regulatory concern:

- By gender: recall was 65.3% for female applicants vs. 78.3% for male applicants (female false-negative rate 34.7% vs. male 21.7%) - a meaningful gap worth further investigation before production use.
- By age group: recall declined steadily with age, from 82.5% for 18–29 year-olds down to 47.8% for applicants 60+ (false-negative rate rising from 17.5% to 52.2%) - the model is noticeably worse at catching defaults among older applicants.

These gaps don't invalidate the model, but they flag it as an area needing attention (e.g. group-specific threshold review or additional fairness-aware training) before any real-world deployment.

**Final model.** I saved the tuned, class-weighted LightGBM together with its 0.50 decision threshold, the exact training column list, the top-15 organization-type grouping, the categorical column list, and the column-cleaning function as a single `credit_risk_model.pkl` package, so the whole preprocessing-to-prediction pipeline can be reproduced on new data. The saved package records a cross-validated ROC-AUC of 0.7831 and a validation ROC-AUC of **0.7886**, across 787 training columns. I then used this final model to generate default probabilities and predictions for the held-out Kaggle test set, saving them to `test_predictions.csv` - about 29.9% of test applicants were flagged as predicted defaulters at the 0.5 threshold.

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost, LightGBM
- imbalanced-learn (SMOTE)
- SHAP
- Matplotlib, Seaborn
- Hugging Face datasets (for sourcing the data)

## Conclusion

Across these four notebooks, I took eight raw, relational tables and turned them into a single, clean, feature-rich dataset capable of predicting loan default, then carried that all the way through to a tuned, interpretable, and fairness-checked production model. Along the way, I learned that missingness in this dataset is informative rather than random, that no single feature can separate defaulters from non-defaulters on its own, and that the engineered `EXT_SOURCE` aggregates outperformed any of the original external credit score columns individually.

LightGBM was consistently the strongest model family. As a plain baseline it reached a ROC-AUC of 0.780, but the bigger lesson came from comparing imbalance-handling strategies: switching from SMOTE to class weighting turned a model that caught only 2.3% of defaulters into one that caught 70.9% of them, for a small gain in ROC-AUC and an acceptable trade-off in precision - a much more useful model for a lender than the higher-accuracy, low-recall SMOTE version. Hyperparameter tuning on top of that pushed cross-validated ROC-AUC to 0.7831 and validation ROC-AUC to 0.7886.

The error analysis and SHAP work showed the model's mistakes cluster around genuinely borderline applicants rather than one broken subgroup, while the fairness check surfaced a real gap - lower defaulter recall for female applicants and for older applicants - that would need further investigation before this model could be trusted in production. The final tuned model, its threshold, and its full preprocessing metadata are saved as `credit_risk_model.pkl`, and test-set predictions have been generated and saved, closing out the modelling pipeline end to end.



