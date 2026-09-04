import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap



# PAGE CONFIGURATION


st.set_page_config(
    page_title="Credit Risk Assessment",
    layout="wide",
    initial_sidebar_state="expanded"
)



# COLUMN NAME CLEANING


def clean_column_names(df):

    df = df.copy()

    df.columns = [
        re.sub(
            r"[^A-Za-z0-9_]",
            "_",
            str(col)
        )
        for col in df.columns
    ]

    return df



# SAFE DIVISION


def safe_divide(a, b):

    return a / (
        b.replace(0, np.nan) + 1e-8
    )


# SHAP EXPLAINER


@st.cache_resource
def create_shap_explainer(_model):
    # Leading underscore tells st.cache_resource not to try to
    # hash the model object itself (it's not hashable); the
    # explainer is still cached for the app's lifetime so we
    # don't rebuild a TreeExplainer on every single prediction.
    return shap.TreeExplainer(_model)


def get_shap_explanation(model, X):

    explainer = create_shap_explainer(model)

    raw_values = explainer.shap_values(X)

    if isinstance(raw_values, list):

        values = np.asarray(
            raw_values[-1]
        )

    else:

        values = np.asarray(
            raw_values
        )

    if values.ndim == 3:

        class_index = (
            1
            if values.shape[-1] > 1
            else 0
        )

        values = values[
            0,
            :,
            class_index
        ]

    elif values.ndim == 2:

        values = values[0]

    elif values.ndim != 1:

        raise ValueError(
            f"Unexpected SHAP output shape: {values.shape}"
        )

    if len(values) != X.shape[1]:

        raise ValueError(
            "SHAP returned a different number of values "
            f"({len(values)}) than model features "
            f"({X.shape[1]})."
        )

    explanation = pd.DataFrame({

        "Feature": X.columns,

        "Value": X.iloc[0].values,

        "SHAP": values

    })

    explanation["Impact"] = (
        explanation["SHAP"].abs()
    )

    return (
        explanation
        .sort_values(
            "Impact",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# USER-FRIENDLY FEATURE NAMES
# ============================================================

def pretty_feature_name(feature):

    replacements = {

        "EXT_SOURCE_1":
            "External credit information",

        "EXT_SOURCE_2":
            "External credit information",

        "EXT_SOURCE_3":
            "External credit information",

        "CREDIT_INCOME_RATIO":
            "Credit compared with income",

        "ANNUITY_INCOME_RATIO":
            "Annual payment compared with income",

        "CREDIT_GOODS_RATIO":
            "Credit compared with goods price",

        "GOODS_INCOME_RATIO":
            "Goods price compared with income",

        "ANNUITY_CREDIT_RATIO":
            "Annual payment compared with credit amount",

        "AGE_YEARS":
            "Age",

        "EMPLOYMENT_YEARS":
            "Employment duration",

        "REGISTRATION_YEARS":
            "Years at current address",

        "ID_PUBLISH_YEARS":
            "Time since identification document was published",

        "LOAN_TERM":
            "Loan term",

        "CREDIT_PER_PERSON":
            "Credit amount per family member",

        "INCOME_PER_PERSON":
            "Income per family member",

        "INCOME_PER_CHILD":
            "Income per child",

        "CHILDREN_RATIO":
            "Children compared with family size",

        "EXT_SOURCE_MEAN":
            "Average external credit information",

        "EXT_SOURCE_MAX":
            "Maximum external credit information",

        "EXT_SOURCE_MIN":
            "Minimum external credit information",

        "EXT_SOURCE_STD":
            "Variation in external credit information",

        "EXT_SOURCE_SUM":
            "Combined external credit information",

        "TOTAL_SOCIAL_OBS":
            "Social circle information",

        "TOTAL_SOCIAL_DEF":
            "Social circle repayment information",

        "TOTAL_CONTACT_FLAGS":
            "Contact information",

        "INSTAL_RECENCY_WEIGHTED_LATE_RATE":
            "Recent late payment rate",

        "INSTAL_AMT_PAYMENT_SUM":
            "Total instalment payments",

        "PREV_DAYS_LAST_DUE_1ST_VERSION_MAX":
            "Previous loan repayment timing",

        "AMT_ANNUITY":
            "Annual loan payment",

        "AMT_CREDIT":
            "Credit amount requested",

        "AMT_GOODS_PRICE":
            "Value of goods being financed"

    }

    if feature in replacements:

        return replacements[feature]

    feature_lower = feature.lower()

    if "contract_type" in feature_lower:

        return "Loan contract information"

    if "education_type" in feature_lower:

        return "Education information"

    if "income_type" in feature_lower:

        return "Income information"

    if "organization_type" in feature_lower:

        if "business_entity_type_1" in feature_lower:

            return (
                "Employment sector: Business Entity Type 1 "
                "(dataset-defined category)"
            )

        if "business_entity_type_2" in feature_lower:

            return (
                "Employment sector: Business Entity Type 2 "
                "(dataset-defined category)"
            )

        if "business_entity_type_3" in feature_lower:

            return (
                "Employment sector: Business Entity Type 3 "
                "(dataset-defined category)"
            )

        if "self_employed" in feature_lower:

            return "Employment sector: Self-employed"

        if "government" in feature_lower:

            return "Employment sector: Government"

        return "Employment sector"

    if "prev_" in feature_lower:

        return "Previous loan history"

    if "instal_" in feature_lower:

        return "Instalment payment history"

    if "bureau_" in feature_lower:

        return "Credit bureau history"

    if "pos_" in feature_lower:

        return "Previous credit account history"

    if "credit_card_" in feature_lower:

        return "Credit card history"

    if "flag_" in feature_lower:

        return "Account information"

    if "days_" in feature_lower:

        return "Applicant history"

    return feature.replace(
        "_",
        " "
    ).title()


# ============================================================
# FEATURES HIDDEN FROM SHAP EXPLANATION
# ============================================================

HIDDEN_EXPLANATION_FEATURES = [

    "CODE_GENDER",

    "NAME_FAMILY_STATUS",

    "NAME_EDUCATION_TYPE",

    "CNT_CHILDREN",

    "CNT_FAM_MEMBERS",

    "NAME_CONTRACT_TYPE",

    "NAME_INCOME_TYPE",

    "ORGANIZATION_TYPE"

]


def should_hide_feature(feature):

    feature_upper = feature.upper()

    for hidden in HIDDEN_EXPLANATION_FEATURES:

        if hidden in feature_upper:

            return True

    return False


# ============================================================
# FEATURE GUIDANCE
# ============================================================

def get_feature_guidance(feature):

    feature_upper = feature.upper()

    guidance = {

        "EXT_SOURCE_1": {
            "meaning":
                "This is an external credit-related score used by the model.",

            "improve":
                "Maintaining a positive repayment history may help strengthen similar credit indicators over time."
        },

        "EXT_SOURCE_2": {
            "meaning":
                "This is an external credit-related score used by the model.",

            "improve":
                "Maintaining a positive repayment history may help strengthen similar credit indicators over time."
        },

        "EXT_SOURCE_3": {
            "meaning":
                "This is an external credit-related score used by the model.",

            "improve":
                "Maintaining a positive repayment history may help strengthen similar credit indicators over time."
        },

        "EXT_SOURCE_MEAN": {
            "meaning":
                "This represents the average of the available external credit-related scores.",

            "improve":
                "A stronger overall credit profile may improve this indicator over time."
        },

        "EXT_SOURCE_MAX": {
            "meaning":
                "This represents the strongest available external credit-related score.",

            "improve":
                "Maintaining positive credit behaviour may help strengthen external credit information."
        },

        "EXT_SOURCE_MIN": {
            "meaning":
                "This represents the weakest available external credit-related score.",

            "improve":
                "Improving weaker areas of the overall credit profile may reduce the impact of this indicator."
        },

        "EXT_SOURCE_STD": {
            "meaning":
                "This measures how different the available external credit-related scores are from each other.",

            "improve":
                "More consistent and reliable credit information may reduce large differences between indicators."
        },

        "EXT_SOURCE_SUM": {
            "meaning":
                "This combines the available external credit-related scores.",

            "improve":
                "A stronger overall credit profile may improve the combined indicator."
        },

        "CREDIT_INCOME_RATIO": {
            "meaning":
                "This compares the requested credit amount with the applicant's income.",

            "improve":
                "A lower requested credit amount relative to income may improve affordability."
        },

        "ANNUITY_INCOME_RATIO": {
            "meaning":
                "This compares the expected loan payment with the applicant's income.",

            "improve":
                "Reducing the repayment burden relative to income may improve affordability."
        },

        "CREDIT_GOODS_RATIO": {
            "meaning":
                "This compares the requested credit amount with the value of the goods being financed.",

            "improve":
                "A financing request that is more proportionate to the goods value may reduce financial risk."
        },

        "GOODS_INCOME_RATIO": {
            "meaning":
                "This compares the value of the goods with the applicant's income.",

            "improve":
                "Choosing a purchase that is more affordable relative to income may improve the risk assessment."
        },

        "ANNUITY_CREDIT_RATIO": {
            "meaning":
                "This compares the expected loan payment with the total credit amount.",

            "improve":
                "A more manageable repayment structure may improve affordability."
        },

        "LOAN_TERM": {
            "meaning":
                "This estimates the relationship between the total credit amount and the regular loan payment.",

            "improve":
                "A repayment structure that better matches the applicant's financial capacity may improve affordability."
        },

        "AMT_ANNUITY": {
            "meaning":
                "This represents the expected regular loan repayment amount.",

            "improve":
                "A lower repayment amount may reduce pressure on the applicant's available income."
        },

        "AMT_CREDIT": {
            "meaning":
                "This represents the total amount of credit requested.",

            "improve":
                "Requesting an amount that is appropriate for income and repayment capacity may reduce risk."
        },

        "AMT_GOODS_PRICE": {
            "meaning":
                "This represents the value of the goods associated with the credit request.",

            "improve":
                "The financing amount should remain reasonable relative to the value of the goods."
        },

        "AGE_YEARS": {
            "meaning":
                "This represents the applicant's age.",

            "improve":
                "Age itself cannot and should not be changed for credit assessment purposes."
        },

        "EMPLOYMENT_YEARS": {
            "meaning":
                "This represents the applicant's length of employment.",

            "improve":
                "Longer and more stable employment history may provide stronger evidence of financial stability."
        },

        "EMPLOYMENT_AGE_RATIO": {
            "meaning":
                "This compares the applicant's employment duration with their age.",

            "improve":
                "A longer and more stable employment history may strengthen this type of indicator."
        },

        "REGISTRATION_YEARS": {
            "meaning":
                "This represents how long the applicant has been registered at the current address.",

            "improve":
                "Longer residential stability may provide additional evidence of stability."
        },

        "INSTAL_RECENCY_WEIGHTED_LATE_RATE": {
            "meaning":
                "This measures recent late repayment behaviour, giving more importance to recent payment history.",

            "improve":
                "Making future repayments on time may gradually improve this indicator."
        },

        "INSTAL_AMT_PAYMENT_SUM": {
            "meaning":
                "This summarizes historical instalment payment activity.",

            "improve":
                "Consistent repayment behaviour may strengthen the overall repayment history."
        },

        "PREV_DAYS_LAST_DUE_1ST_VERSION_MAX": {
            "meaning":
                "This relates to repayment timing in previous credit applications.",

            "improve":
                "Meeting future repayment obligations on time may strengthen repayment history."
        },

        "TOTAL_SOCIAL_DEF": {
            "meaning":
                "This summarizes default-related information recorded in the applicant's available social credit data.",

            "improve":
                "This factor may not be directly controllable by the applicant and should be interpreted carefully."
        },

        "TOTAL_SOCIAL_OBS": {
            "meaning":
                "This summarizes available social credit information in the dataset.",

            "improve":
                "This factor may not be directly controllable by the applicant."
        },

        "INCOME_PER_PERSON": {
            "meaning":
                "This estimates the amount of income available per family member.",

            "improve":
                "Higher available household income relative to household size may improve affordability."
        },

        "CREDIT_PER_PERSON": {
            "meaning":
                "This estimates the requested credit burden per family member.",

            "improve":
                "A lower credit burden relative to household financial capacity may improve affordability."
        },

        "CHILDREN_RATIO": {
            "meaning":
                "This compares the number of children with the total family size.",

            "improve":
                "This is a household characteristic and should not be treated as something the applicant must change."
        },

        "INCOME_PER_CHILD": {
            "meaning":
                "This estimates income relative to the number of children.",

            "improve":
                "Greater financial capacity relative to household responsibilities may improve affordability."
        }

    }

    if feature_upper in guidance:

        return guidance[feature_upper]

    if feature_upper.startswith("PREV_"):

        return {
            "meaning":
                "This factor is related to the applicant's previous loan history.",

            "improve":
                "Consistent repayment and responsible use of credit may strengthen future credit history."
        }

    if feature_upper.startswith("INSTAL_"):

        return {
            "meaning":
                "This factor is related to the applicant's instalment payment history.",

            "improve":
                "Maintaining timely payments may improve future repayment indicators."
        }

    if feature_upper.startswith("BUREAU_"):

        return {
            "meaning":
                "This factor is based on information from previous or existing credit bureau records.",

            "improve":
                "Maintaining positive credit behaviour may improve future credit bureau information."
        }

    if feature_upper.startswith("POS_"):

        return {
            "meaning":
                "This factor is related to previous point-of-sale or consumer credit account history.",

            "improve":
                "Consistent repayment behaviour may strengthen future account history."
        }

    if feature_upper.startswith("CREDIT_CARD_"):

        return {
            "meaning":
                "This factor is related to historical credit card account information.",

            "improve":
                "Managing credit responsibly and paying obligations on time may strengthen future history."
        }

    return {

        "meaning":
            "This is a financial or credit-related factor that influenced the model's estimated risk.",

        "improve":
            "This factor should be reviewed together with the applicant's complete financial and credit information."
    }


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    # FIX: the training notebook saves the package to
    # "final_chosen_credit_risk_model.pkl" (cell 101). The
    # previous filename here ("final_credit_risk_model.pkl")
    # doesn't exist on disk and would raise FileNotFoundError.
    model_path = (
        Path("output")
        / "final_chosen_credit_risk_model.pkl"
    )

    package = joblib.load(
        model_path
    )

    return package


# ============================================================
# LOAD FINAL CHOSEN MODEL PACKAGE
# ============================================================

try:

    model_package = load_model()

    final_chosen_model = (
        model_package["model"]
    )

    final_chosen_calibrator = (
        model_package["calibrator"]
    )

    final_chosen_threshold = float(
        model_package["threshold"]
    )

    final_chosen_training_columns = (
        model_package["training_columns"]
    )

    final_chosen_categorical_columns = (
        model_package["categorical_columns"]
    )

    final_chosen_calibration_method = (
        model_package["calibration_method"]
    )

    # FIX: the training notebook groups every ORGANIZATION_TYPE
    # value outside the 15 most frequent training categories into
    # "Other" *before* one-hot encoding (cell 19), and this exact
    # list is saved in the package as "top_15_organization_types"
    # (cell 101) — but it was never loaded or used anywhere in the
    # app. Without it, any applicant (individual or batch) whose
    # organization type isn't one of those 15 exact strings ends up
    # with every ORGANIZATION_TYPE_* dummy column set to 0, which is
    # a different feature representation than anything the model
    # was trained on (every training row has exactly one such column
    # set to 1, since "Other" is itself one of the dummy columns).
    final_chosen_top15_organizations = (
        model_package["top_15_organization_types"]
    )

except Exception as e:

    st.error(
        "Unable to load the final chosen credit risk model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# FINAL THRESHOLD VERIFICATION
# ============================================================

EXPECTED_FINAL_THRESHOLD = 0.15

if not np.isclose(
    final_chosen_threshold,
    EXPECTED_FINAL_THRESHOLD
):

    st.error(
        f"Unexpected model threshold: "
        f"{final_chosen_threshold:.2f}. "
        f"The final chosen model should use a threshold "
        f"of {EXPECTED_FINAL_THRESHOLD:.2f}."
    )

    st.stop()


# ============================================================
# REFERENCE DATA
# ============================================================

@st.cache_data
def load_reference_data():

    reference_path = (
        Path("output")
        / "train_feature_engineered.parquet"
    )

    return pd.read_parquet(
        reference_path
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):

    df = df.copy()

    df["CREDIT_INCOME_RATIO"] = safe_divide(
        df["AMT_CREDIT"],
        df["AMT_INCOME_TOTAL"]
    )

    df["ANNUITY_INCOME_RATIO"] = safe_divide(
        df["AMT_ANNUITY"],
        df["AMT_INCOME_TOTAL"]
    )

    df["CREDIT_GOODS_RATIO"] = safe_divide(
        df["AMT_CREDIT"],
        df["AMT_GOODS_PRICE"]
    )

    df["GOODS_INCOME_RATIO"] = safe_divide(
        df["AMT_GOODS_PRICE"],
        df["AMT_INCOME_TOTAL"]
    )

    df["ANNUITY_CREDIT_RATIO"] = safe_divide(
        df["AMT_ANNUITY"],
        df["AMT_CREDIT"]
    )

    df["INCOME_PER_PERSON"] = safe_divide(
        df["AMT_INCOME_TOTAL"],
        df["CNT_FAM_MEMBERS"]
    )

    df["INCOME_PER_CHILD"] = safe_divide(
        df["AMT_INCOME_TOTAL"],
        df["CNT_CHILDREN"] + 1
    )

    df["CREDIT_PER_PERSON"] = safe_divide(
        df["AMT_CREDIT"],
        df["CNT_FAM_MEMBERS"]
    )

    df["CHILDREN_RATIO"] = safe_divide(
        df["CNT_CHILDREN"],
        df["CNT_FAM_MEMBERS"]
    )

    df["AGE_YEARS"] = (
        -df["DAYS_BIRTH"] / 365
    )

    df["EMPLOYMENT_YEARS"] = (
        -df["DAYS_EMPLOYED"] / 365
    )

    df["REGISTRATION_YEARS"] = (
        -df["DAYS_REGISTRATION"] / 365
    )

    df["ID_PUBLISH_YEARS"] = (
        -df["DAYS_ID_PUBLISH"] / 365
    )

    df["EMPLOYMENT_AGE_RATIO"] = safe_divide(
        df["EMPLOYMENT_YEARS"],
        df["AGE_YEARS"]
    )

    df["REGISTRATION_AGE_RATIO"] = safe_divide(
        df["REGISTRATION_YEARS"],
        df["AGE_YEARS"]
    )

    ext_cols = [
        col
        for col in [
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3"
        ]
        if col in df.columns
    ]

    if ext_cols:

        df["EXT_SOURCE_MEAN"] = (
            df[ext_cols].mean(axis=1)
        )

        df["EXT_SOURCE_MAX"] = (
            df[ext_cols].max(axis=1)
        )

        df["EXT_SOURCE_MIN"] = (
            df[ext_cols].min(axis=1)
        )

        df["EXT_SOURCE_STD"] = (
            df[ext_cols].std(axis=1)
        )

        df["EXT_SOURCE_SUM"] = (
            df[ext_cols].sum(axis=1)
        )

    df["LOAN_TERM"] = safe_divide(
        df["AMT_CREDIT"],
        df["AMT_ANNUITY"]
    )

    if (
        "OBS_30_CNT_SOCIAL_CIRCLE" in df.columns
        and
        "OBS_60_CNT_SOCIAL_CIRCLE" in df.columns
    ):

        df["TOTAL_SOCIAL_OBS"] = (
            df["OBS_30_CNT_SOCIAL_CIRCLE"]
            +
            df["OBS_60_CNT_SOCIAL_CIRCLE"]
        )

    if (
        "DEF_30_CNT_SOCIAL_CIRCLE" in df.columns
        and
        "DEF_60_CNT_SOCIAL_CIRCLE" in df.columns
    ):

        df["TOTAL_SOCIAL_DEF"] = (
            df["DEF_30_CNT_SOCIAL_CIRCLE"]
            +
            df["DEF_60_CNT_SOCIAL_CIRCLE"]
        )

    contact_columns = [

        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_CONT_MOBILE",
        "FLAG_PHONE",
        "FLAG_EMAIL"

    ]

    existing_contact = [

        col
        for col in contact_columns
        if col in df.columns

    ]

    if existing_contact:

        df["TOTAL_CONTACT_FLAGS"] = (
            df[existing_contact].sum(axis=1)
        )

    return df


# ============================================================
# PREPARE DATA FOR MODEL
# ============================================================

def prepare_for_model(applicant):

    df = applicant.copy()

    df = engineer_features(df)

    # FIX: replicate the training notebook's top-15 grouping
    # (cell 19) before one-hot encoding, otherwise any
    # organization type outside that exact list of 15 strings
    # silently loses all ORGANIZATION_TYPE signal instead of
    # correctly falling back to the "Other" bucket the model
    # was actually trained on.
    if "ORGANIZATION_TYPE" in df.columns:

        df["ORGANIZATION_TYPE"] = np.where(
            df["ORGANIZATION_TYPE"].isin(
                final_chosen_top15_organizations
            ),
            df["ORGANIZATION_TYPE"],
            "Other"
        )

    df = df.drop(
        columns=[
            "SK_ID_CURR",
            "TARGET"
        ],
        errors="ignore"
    )

    existing_categorical = [

        col
        for col in final_chosen_categorical_columns
        if col in df.columns

    ]

    if existing_categorical:

        df = pd.get_dummies(
            df,
            columns=existing_categorical,
            drop_first=False,
            dtype="int8"
        )

    df = clean_column_names(df)

    df = df.reindex(
        columns=final_chosen_training_columns,
        fill_value=0
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.fillna(0)

    return df


# ============================================================
# REFERENCE VALUES
# ============================================================

@st.cache_data
def get_reference_values():

    reference = load_reference_data()

    reference = reference.drop(
        columns=[
            "TARGET",
            "SK_ID_CURR"
        ],
        errors="ignore"
    )

    values = {}

    for col in reference.columns:

        if pd.api.types.is_numeric_dtype(
            reference[col]
        ):

            values[col] = (
                reference[col].median()
            )

        else:

            mode = (
                reference[col].mode()
            )

            if len(mode) > 0:

                values[col] = (
                    mode.iloc[0]
                )

            else:

                values[col] = "Missing"

    return values


# ============================================================
# HEADER
# ============================================================

st.title(
    "Credit Risk Assessment"
)

st.write(
    "AI-powered assessment of loan default risk "
    "using historical credit information."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "Credit Risk Assessment"
    )

    st.write(
        "### How to use this tool"
    )

    st.write(
        "**1. Enter applicant information**\n\n"
        "Provide the available applicant and loan details.\n\n"

        "**2. Assess the application**\n\n"
        "Select **Assess Credit Risk** to generate an estimate.\n\n"

        "**3. Review the result**\n\n"
        "Review the estimated probability and the main "
        "factors influencing the assessment."
    )

    st.divider()

    st.write(
        "### Important"
    )

    st.caption(
        "This tool provides a machine-learning estimate. "
        "It should support professional credit assessment "
        "rather than replace human judgement."
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "Individual Assessment",
        "Batch Assessment"
    ]
)


# ============================================================
# INDIVIDUAL ASSESSMENT
# ============================================================

with tab1:

    st.header(
        "Applicant Information"
    )

    st.info(
        "Enter the applicant's information below. "
        "Information not directly entered is completed "
        "using typical values from the training data."
    )

    reference_values = (
        get_reference_values()
    )


    # ========================================================
    # PERSONAL INFORMATION
    # ========================================================

    st.subheader(
        "Personal Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["F", "M"]
        )

    with col2:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1
        )

    with col3:

        children = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=20,
            value=0,
            step=1
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        family_members = st.number_input(
            "Family Members",
            min_value=1,
            max_value=30,
            value=2,
            step=1
        )

    with col2:

        income = st.number_input(
            "Annual Income",
            min_value=0.0,
            value=150000.0,
            step=10000.0
        )

    with col3:

        education = st.selectbox(
            "Education",
            [
                "Secondary / secondary special",
                "Higher education",
                "Incomplete higher",
                "Lower secondary",
                "Academic degree"
            ]
        )


    # ========================================================
    # LOAN INFORMATION
    # ========================================================

    st.subheader(
        "Loan Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        credit = st.number_input(
            "Credit Amount",
            min_value=0.0,
            value=500000.0,
            step=10000.0
        )

    with col2:

        annuity = st.number_input(
            "Annual Annuity",
            min_value=0.0,
            value=30000.0,
            step=1000.0
        )

    with col3:

        goods_price = st.number_input(
            "Goods Price",
            min_value=0.0,
            value=500000.0,
            step=10000.0
        )

    col1, col2 = st.columns(2)

    with col1:

        contract_type = st.selectbox(
            "Contract Type",
            [
                "Cash loans",
                "Revolving loans"
            ]
        )

    with col2:

        income_type = st.selectbox(
            "Income Type",
            [
                "Working",
                "Commercial associate",
                "Pensioner",
                "State servant",
                "Student",
                "Businessman",
                "Maternity leave",
                "Unemployed"
            ]
        )


    # ========================================================
    # EMPLOYMENT AND STABILITY
    # ========================================================

    st.subheader(
        "Employment and Stability"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        employment_years = st.number_input(
            "Employment Years",
            min_value=0.0,
            max_value=70.0,
            value=5.0,
            step=1.0
        )

    with col2:

        registration_years = st.number_input(
            "Years at Current Address",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=1.0
        )

    with col3:

        # FIX: these five options are a subset of the 15
        # categories the model actually saw by name during
        # training (plus "Other", which correctly captures
        # everything else). Selecting any of them, or "Other",
        # now maps correctly through the top-15 grouping fix
        # above and matches an ORGANIZATION_TYPE_* dummy column
        # the model was trained on.
        organization = st.selectbox(
            "Employment Sector",
            [
                "Business Entity Type 3",
                "Business Entity Type 2",
                "Self-employed",
                "Government",
                "Other"
            ]
        )

    st.caption(
        "Business Entity categories are original dataset "
        "categories and do not have a more specific industry description."
    )


    # ========================================================
    # EXTERNAL CREDIT SCORES
    # ========================================================

    st.subheader(
        "External Credit Scores"
    )

    st.caption(
        "Values generally range from 0 to 1."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        ext1 = st.slider(
            "External Score 1",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01
        )

    with col2:

        ext2 = st.slider(
            "External Score 2",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01
        )

    with col3:

        ext3 = st.slider(
            "External Score 3",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01
        )


    # ========================================================
    # CONTACT INFORMATION
    # ========================================================

    st.subheader(
        "Contact Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        mobile = st.checkbox(
            "Mobile Phone",
            value=True
        )

    with col2:

        work_phone = st.checkbox(
            "Work Phone",
            value=True
        )

    with col3:

        email = st.checkbox(
            "Email",
            value=True
        )


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    st.divider()

    predict_button = st.button(
        "Assess Credit Risk",
        type="primary",
        use_container_width=True
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        with st.spinner(
            "Assessing applicant risk..."
        ):

            applicant = pd.DataFrame(
                [reference_values]
            )

            applicant["CODE_GENDER"] = gender

            applicant["CNT_CHILDREN"] = int(
                children
            )

            applicant["CNT_FAM_MEMBERS"] = int(
                family_members
            )

            applicant["AMT_INCOME_TOTAL"] = float(
                income
            )

            applicant["AMT_CREDIT"] = float(
                credit
            )

            applicant["AMT_ANNUITY"] = float(
                annuity
            )

            applicant["AMT_GOODS_PRICE"] = float(
                goods_price
            )

            applicant["NAME_CONTRACT_TYPE"] = (
                contract_type
            )

            applicant["NAME_INCOME_TYPE"] = (
                income_type
            )

            applicant["NAME_EDUCATION_TYPE"] = (
                education
            )

            applicant["ORGANIZATION_TYPE"] = (
                organization
            )

            applicant["EXT_SOURCE_1"] = float(
                ext1
            )

            applicant["EXT_SOURCE_2"] = float(
                ext2
            )

            applicant["EXT_SOURCE_3"] = float(
                ext3
            )

            applicant["FLAG_MOBIL"] = int(
                mobile
            )

            applicant["FLAG_WORK_PHONE"] = int(
                work_phone
            )

            applicant["FLAG_EMAIL"] = int(
                email
            )

            applicant["DAYS_BIRTH"] = (
                -float(age) * 365
            )

            applicant["DAYS_EMPLOYED"] = (
                -float(employment_years) * 365
            )

            applicant["DAYS_REGISTRATION"] = (
                -float(registration_years) * 365
            )

            X_input = prepare_for_model(
                applicant
            )

            # ------------------------------------------------
            # RAW MODEL PROBABILITY
            # ------------------------------------------------

            raw_probability = (
                final_chosen_model
                .predict_proba(
                    X_input
                )[0, 1]
            )

            # ------------------------------------------------
            # CALIBRATED PROBABILITY
            # ------------------------------------------------

            probability = (
                final_chosen_calibrator
                .predict(
                    np.array([raw_probability])
                )[0]
            )

            probability = float(
                np.clip(
                    probability,
                    0.0,
                    1.0
                )
            )

            # ------------------------------------------------
            # FINAL DECISION
            # ------------------------------------------------

            prediction = int(
                probability >= final_chosen_threshold
            )


        # ====================================================
        # RISK ASSESSMENT
        # ====================================================

        st.divider()

        st.header(
            "Risk Assessment"
        )

        probability_pct = (
            probability * 100
        )

        if prediction == 1:

            risk_level = (
                "Higher Estimated Risk"
            )

            recommendation = (
                "The model estimates that this applicant "
                "has a relatively higher likelihood of default. "
                "Further credit assessment is recommended "
                "before approval."
            )

        else:

            risk_level = (
                "Lower Estimated Risk"
            )

            recommendation = (
                "The model estimates that this applicant "
                "has a relatively lower likelihood of default. "
                "The application may proceed to normal "
                "credit assessment."
            )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Estimated Default Probability",
                f"{probability_pct:.2f}%"
            )

        with col2:

            st.metric(
                "Model Classification",
                (
                    "Higher Risk"
                    if prediction == 1
                    else "Lower Risk"
                )
            )

        st.caption(
            f"Decision threshold: "
            f"{final_chosen_threshold:.2f}"
        )

        st.progress(
            min(
                max(
                    probability,
                    0.0
                ),
                1.0
            )
        )

        st.subheader(
            "What does this mean?"
        )

        st.write(
            recommendation
        )

        st.caption(
            "This is a machine-learning estimate and should "
            "support, not replace, human credit assessment."
        )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        st.divider()

        st.header(
            "Why did the model make this decision?"
        )

        st.write(
            "The following factors had the greatest influence "
            "on this applicant's estimated default risk."
        )

        try:

            shap_explanation = (
                get_shap_explanation(
                    final_chosen_model,
                    X_input
                )
            )

            shap_explanation = (
                shap_explanation[
                    ~shap_explanation[
                        "Feature"
                    ].apply(
                        should_hide_feature
                    )
                ].copy()
            )

            risk_increasing = (
                shap_explanation[
                    shap_explanation["SHAP"] > 0
                ]
                .sort_values(
                    "SHAP",
                    ascending=False
                )
                .head(5)
            )

            risk_reducing = (
                shap_explanation[
                    shap_explanation["SHAP"] < 0
                ]
                .sort_values(
                    "SHAP",
                    ascending=True
                )
                .head(5)
            )


            # =================================================
            # SHAP FACTORS
            # =================================================

            left_column, right_column = st.columns(2)


            # =================================================
            # FACTORS INCREASING RISK
            # =================================================

            with left_column:

                st.subheader(
                    "Factors Increasing Estimated Risk"
                )

                if risk_increasing.empty:

                    st.write(
                        "No major risk-increasing factors "
                        "were identified."
                    )

                else:

                    for index, row in (
                        risk_increasing.iterrows()
                    ):

                        feature = row["Feature"]

                        feature_name = (
                            pretty_feature_name(
                                feature
                            )
                        )

                        guidance = (
                            get_feature_guidance(
                                feature
                            )
                        )

                        st.markdown(
                            f"**{feature_name}**"
                        )

                        st.write(
                            f"**What it means:** "
                            f"{guidance['meaning']}"
                        )

                        st.write(
                            f"**What may help:** "
                            f"{guidance['improve']}"
                        )

                        if (
                            index
                            !=
                            risk_increasing.index[-1]
                        ):

                            st.divider()


            # =================================================
            # FACTORS REDUCING RISK
            # =================================================

            with right_column:

                st.subheader(
                    "Factors Reducing Estimated Risk"
                )

                if risk_reducing.empty:

                    st.write(
                        "No major risk-reducing factors "
                        "were identified."
                    )

                else:

                    for index, row in (
                        risk_reducing.iterrows()
                    ):

                        feature = row["Feature"]

                        feature_name = (
                            pretty_feature_name(
                                feature
                            )
                        )

                        guidance = (
                            get_feature_guidance(
                                feature
                            )
                        )

                        st.markdown(
                            f"**{feature_name}**"
                        )

                        st.write(
                            f"**Why it helped:** "
                            f"{guidance['meaning']}"
                        )

                        st.write(
                            "This factor moved the model's "
                            "estimate toward a lower probability "
                            "of default."
                        )

                        if (
                            index
                            !=
                            risk_reducing.index[-1]
                        ):

                            st.divider()


            # =================================================
            # INTERPRETATION
            # =================================================

            st.divider()

            st.subheader(
                "How should this explanation be interpreted?"
            )

            st.write(
                "Factors listed under increasing risk moved "
                "the model's estimate toward a higher "
                "probability of default."
            )

            st.write(
                "Factors listed under reducing risk moved "
                "the model's estimate toward a lower "
                "probability of default."
            )

            st.caption(
                "These factors describe how the model arrived "
                "at its estimate. They should not be interpreted "
                "as proof that any individual factor caused "
                "the applicant's risk."
            )

        except Exception as shap_error:

            st.warning(
                "The risk assessment was completed, but the "
                "individual explanation could not be generated."
            )

            st.caption(
                f"SHAP error: {shap_error}"
            )


# ============================================================
# BATCH ASSESSMENT
# ============================================================

with tab2:

    st.header(
        "Batch Credit Risk Assessment"
    )

    st.write(
        "Upload a CSV containing applicant records. "
        "The application will process the applicants and "
        "generate calibrated estimated default probabilities."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(
                uploaded_file
            )

            st.write(
                f"Uploaded {len(batch_df):,} applicants."
            )

            st.dataframe(
                batch_df.head(),
                use_container_width=True
            )

            if st.button(
                "Run Batch Assessment",
                type="primary"
            ):

                with st.spinner(
                    "Processing applicants..."
                ):

                    batch_input = (
                        batch_df.copy()
                    )

                    if "TARGET" in batch_input.columns:

                        batch_input = (
                            batch_input.drop(
                                columns=["TARGET"]
                            )
                        )

                    if "SK_ID_CURR" in batch_input.columns:

                        applicant_ids = (
                            batch_input[
                                "SK_ID_CURR"
                            ].copy()
                        )

                    else:

                        applicant_ids = pd.Series(
                            range(
                                len(batch_input)
                            ),
                            name="SK_ID_CURR"
                        )

                    X_batch = (
                        prepare_for_model(
                            batch_input
                        )
                    )

                    raw_probabilities = (
                        final_chosen_model
                        .predict_proba(
                            X_batch
                        )[:, 1]
                    )

                    probabilities = (
                        final_chosen_calibrator
                        .predict(
                            raw_probabilities
                        )
                    )

                    probabilities = np.clip(
                        probabilities,
                        0.0,
                        1.0
                    )

                    predictions = (
                        probabilities
                        >= final_chosen_threshold
                    ).astype(int)

                    results = pd.DataFrame({

                        "Applicant ID":
                            applicant_ids.values,

                        "Estimated Default Probability":
                            probabilities,

                        "Risk Classification":
                            np.where(
                                predictions == 1,
                                "Higher Risk",
                                "Lower Risk"
                            )

                    })

                st.success(
                    "Batch assessment completed successfully."
                )

                display_results = (
                    results.copy()
                )

                display_results[
                    "Estimated Default Probability"
                ] = (

                    display_results[
                        "Estimated Default Probability"
                    ]

                    * 100

                ).round(2).astype(str) + "%"

                st.dataframe(
                    display_results,
                    use_container_width=True,
                    hide_index=True
                )

                higher_risk_count = int(
                    predictions.sum()
                )

                lower_risk_count = (
                    len(predictions)
                    - higher_risk_count
                )

                st.subheader(
                    "Assessment Summary"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Applicants Assessed",
                        f"{len(results):,}"
                    )

                with col2:

                    st.metric(
                        "Higher Risk",
                        f"{higher_risk_count:,}"
                    )

                with col3:

                    st.metric(
                        "Lower Risk",
                        f"{lower_risk_count:,}"
                    )

                csv = (
                    results
                    .to_csv(
                        index=False
                    )
                    .encode("utf-8")
                )

                st.download_button(
                    "Download Assessment Results",
                    data=csv,
                    file_name="credit_risk_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as batch_error:

            st.error(
                "The uploaded file could not be processed. "
                "Please check that the CSV contains the required "
                "applicant information."
            )

            st.caption(
                f"Error: {batch_error}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Credit Risk Assessment • Machine Learning Decision Support"
)