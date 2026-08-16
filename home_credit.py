import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import shap


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
    }

    .metric-card {
        background-color: #f7f9fc;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e6eaf0;
        text-align: center;
    }

    .title-text {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle-text {
        color: #6b7280;
        font-size: 17px;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# COLUMN NAME CLEANING
# ============================================================

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


# ============================================================
# SAFE DIVISION
# ============================================================

def safe_divide(a, b):

    return a / (
        b.replace(0, np.nan) + 1e-8
    )


# ============================================================
# SHAP EXPLAINER
# ============================================================

@st.cache_resource
def create_shap_explainer(_model):

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

        "Feature":
            X.columns,

        "Value":
            X.iloc[0].values,

        "SHAP":
            values

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
# PRETTY FEATURE NAMES
# ============================================================

def pretty_feature_name(feature):

    replacements = {

        "EXT_SOURCE_1":
            "External Credit Score 1",

        "EXT_SOURCE_2":
            "External Credit Score 2",

        "EXT_SOURCE_3":
            "External Credit Score 3",

        "CREDIT_INCOME_RATIO":
            "Credit-to-Income Ratio",

        "ANNUITY_INCOME_RATIO":
            "Annuity-to-Income Ratio",

        "CREDIT_GOODS_RATIO":
            "Credit-to-Goods Ratio",

        "GOODS_INCOME_RATIO":
            "Goods-to-Income Ratio",

        "ANNUITY_CREDIT_RATIO":
            "Annuity-to-Credit Ratio",

        "AGE_YEARS":
            "Age (Years)",

        "EMPLOYMENT_YEARS":
            "Employment Duration",

        "REGISTRATION_YEARS":
            "Years at Current Address",

        "ID_PUBLISH_YEARS":
            "Years Since ID Published",

        "LOAN_TERM":
            "Loan Term",

        "CREDIT_PER_PERSON":
            "Credit per Family Member",

        "INCOME_PER_PERSON":
            "Income per Family Member",

        "INCOME_PER_CHILD":
            "Income per Child",

        "CHILDREN_RATIO":
            "Children-to-Family Ratio",

        "EXT_SOURCE_MEAN":
            "Average External Credit Score",

        "EXT_SOURCE_MAX":
            "Maximum External Credit Score",

        "EXT_SOURCE_MIN":
            "Minimum External Credit Score",

        "EXT_SOURCE_STD":
            "External Credit Score Variation",

        "EXT_SOURCE_SUM":
            "Total External Credit Score",

        "TOTAL_SOCIAL_OBS":
            "Social Circle Observations",

        "TOTAL_SOCIAL_DEF":
            "Social Circle Defaults",

        "TOTAL_CONTACT_FLAGS":
            "Contact Information Flags",

    }

    if feature in replacements:

        return replacements[feature]

    return feature.replace(
        "_",
        " "
    ).title()


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    model_path = (
        Path("output")
        / "credit_risk_model.pkl"
    )

    with open(
        model_path,
        "rb"
    ) as file:

        package = pickle.load(file)

    return package


try:

    model_package = load_model()

    final_model = (
        model_package["model"]
    )

    final_threshold = (
        model_package["threshold"]
    )

    training_columns = (
        model_package["training_columns"]
    )

    top_15_org = (
        model_package[
            "top_15_organization_types"
        ]
    )

    categorical_columns = (
        model_package[
            "categorical_columns"
        ]
    )

    metadata = (
        model_package.get(
            "metadata",
            {}
        )
    )

except Exception as e:

    st.error(
        "Unable to load the credit risk model."
    )

    st.exception(e)

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

    # --------------------------------------------------------
    # FINANCIAL RATIOS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # FAMILY FEATURES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # EMPLOYMENT RATIOS
    # --------------------------------------------------------

    df["EMPLOYMENT_AGE_RATIO"] = safe_divide(
        df["EMPLOYMENT_YEARS"],
        df["AGE_YEARS"]
    )

    df["REGISTRATION_AGE_RATIO"] = safe_divide(
        df["REGISTRATION_YEARS"],
        df["AGE_YEARS"]
    )

    # --------------------------------------------------------
    # EXTERNAL CREDIT SCORES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # LOAN TERM
    # --------------------------------------------------------

    df["LOAN_TERM"] = safe_divide(
        df["AMT_CREDIT"],
        df["AMT_ANNUITY"]
    )

    # --------------------------------------------------------
    # SOCIAL CIRCLE
    # --------------------------------------------------------

    df["TOTAL_SOCIAL_OBS"] = (

        df["OBS_30_CNT_SOCIAL_CIRCLE"]

        +

        df["OBS_60_CNT_SOCIAL_CIRCLE"]

    )

    df["TOTAL_SOCIAL_DEF"] = (

        df["DEF_30_CNT_SOCIAL_CIRCLE"]

        +

        df["DEF_60_CNT_SOCIAL_CIRCLE"]

    )

    # --------------------------------------------------------
    # CONTACT FLAGS
    # --------------------------------------------------------

    df["TOTAL_CONTACT_FLAGS"] = (

        df["FLAG_MOBIL"]

        +

        df["FLAG_EMP_PHONE"]

        +

        df["FLAG_WORK_PHONE"]

        +

        df["FLAG_CONT_MOBILE"]

        +

        df["FLAG_PHONE"]

        +

        df["FLAG_EMAIL"]

    )

    return df


# ============================================================
# PREPARE DATA FOR MODEL
# ============================================================

def prepare_for_model(applicant):

    df = applicant.copy()

    # --------------------------------------------------------
    # ORGANIZATION GROUPING
    # --------------------------------------------------------

    if "ORGANIZATION_TYPE" in df.columns:

        df["ORGANIZATION_TYPE"] = np.where(

            df["ORGANIZATION_TYPE"].isin(
                top_15_org
            ),

            df["ORGANIZATION_TYPE"],

            "Other"

        )

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    df = engineer_features(
        df
    )

    # --------------------------------------------------------
    # REMOVE ID AND TARGET
    # --------------------------------------------------------

    for col in [

        "SK_ID_CURR",
        "TARGET"

    ]:

        if col in df.columns:

            df = df.drop(
                columns=[col]
            )

    # --------------------------------------------------------
    # CATEGORICAL ENCODING
    # --------------------------------------------------------

    existing_categorical = [

        col

        for col in categorical_columns

        if col in df.columns

    ]

    if existing_categorical:

        df = pd.get_dummies(

            df,

            columns=existing_categorical,

            drop_first=False,

            dtype="int8"

        )

    # --------------------------------------------------------
    # CLEAN COLUMN NAMES
    # --------------------------------------------------------

    df = clean_column_names(
        df
    )

    # --------------------------------------------------------
    # EXACT TRAINING SCHEMA
    # --------------------------------------------------------

    df = df.reindex(

        columns=training_columns,

        fill_value=0

    )

    # --------------------------------------------------------
    # HANDLE INFINITY / MISSING VALUES
    # --------------------------------------------------------

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
                reference[col]
                .median()
            )

        else:

            mode = (
                reference[col]
                .mode()
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

st.markdown(

    '<div class="title-text">'
    'Credit Risk Assessment'
    '</div>',

    unsafe_allow_html=True

)

st.markdown(

    """
    <div class="subtitle-text">
    Intelligent loan default risk assessment powered by a
    tuned LightGBM machine learning model.
    </div>
    """,

    unsafe_allow_html=True

)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "About the Model"
    )

    st.write(

        """
        This application uses the final tuned LightGBM
        credit-risk model developed for the Home Credit
        default prediction project.
        """

    )

    st.divider()

    st.metric(
        "Model",
        "Tuned LightGBM"
    )

    st.metric(
        "Model Features",
        str(len(training_columns))
    )

    st.metric(
        "Decision Threshold",
        f"{final_threshold:.2f}"
    )

    if "validation_roc_auc" in metadata:

        st.metric(

            "Validation ROC-AUC",

            f"{metadata['validation_roc_auc']:.4f}"

        )

    st.divider()

    st.caption(

        "Higher predicted probability indicates "
        "higher estimated default risk."

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

    st.subheader(
        "Applicant Information"
    )

    st.info(

        """
        Enter the applicant's information below.
        Fields not directly entered are handled using
        reference values from the training data.
        """

    )

    reference_values = (
        get_reference_values()
    )

    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    st.markdown(
        "### Personal Information"
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

            value=35

        )

    with col3:

        children = st.number_input(

            "Number of Children",

            min_value=0,

            max_value=20,

            value=0

        )

    col1, col2, col3 = st.columns(3)

    with col1:

        family_members = st.number_input(

            "Family Members",

            min_value=1.0,

            max_value=30.0,

            value=2.0

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

    # --------------------------------------------------------
    # LOAN INFORMATION
    # --------------------------------------------------------

    st.markdown(
        "### Loan Information"
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

    # --------------------------------------------------------
    # EMPLOYMENT
    # --------------------------------------------------------

    st.markdown(
        "### Employment and Stability"
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

        organization = st.selectbox(

            "Organization Type",

            [

                "Business Entity Type 3",

                "Business Entity Type 2",

                "Self-employed",

                "Government",

                "Other"

            ]

        )

    # --------------------------------------------------------
    # EXTERNAL CREDIT SCORES
    # --------------------------------------------------------

    st.markdown(
        "### External Credit Scores"
    )

    st.caption(
        "Values should generally fall between 0 and 1."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        ext1 = st.slider(

            "External Score 1",

            0.0,

            1.0,

            0.5

        )

    with col2:

        ext2 = st.slider(

            "External Score 2",

            0.0,

            1.0,

            0.5

        )

    with col3:

        ext3 = st.slider(

            "External Score 3",

            0.0,

            1.0,

            0.5

        )

    # --------------------------------------------------------
    # CONTACT INFORMATION
    # --------------------------------------------------------

    st.markdown(
        "### Contact Information"
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

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

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

            # Start from training reference values

            applicant = pd.DataFrame(
                [reference_values]
            )

            # ------------------------------------------------
            # USER INPUTS
            # ------------------------------------------------

            applicant["CODE_GENDER"] = gender

            applicant["CNT_CHILDREN"] = (
                children
            )

            applicant["CNT_FAM_MEMBERS"] = (
                family_members
            )

            applicant["AMT_INCOME_TOTAL"] = (
                income
            )

            applicant["AMT_CREDIT"] = (
                credit
            )

            applicant["AMT_ANNUITY"] = (
                annuity
            )

            applicant["AMT_GOODS_PRICE"] = (
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

            applicant["EXT_SOURCE_1"] = (
                ext1
            )

            applicant["EXT_SOURCE_2"] = (
                ext2
            )

            applicant["EXT_SOURCE_3"] = (
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

            # ------------------------------------------------
            # CONVERT YEARS TO ORIGINAL HOME CREDIT FORMAT
            # ------------------------------------------------

            applicant["DAYS_BIRTH"] = (
                -age * 365
            )

            applicant["DAYS_EMPLOYED"] = (
                -employment_years * 365
            )

            applicant["DAYS_REGISTRATION"] = (
                -registration_years * 365
            )

            # ------------------------------------------------
            # PREPARE MODEL INPUT
            # ------------------------------------------------

            X_input = prepare_for_model(
                applicant
            )

            # ------------------------------------------------
            # VERIFY MODEL SCHEMA
            # ------------------------------------------------

            if X_input.shape[1] != len(
                training_columns
            ):

                raise ValueError(

                    "Model input contains "
                    f"{X_input.shape[1]} features, "
                    f"but the model expects "
                    f"{len(training_columns)}."

                )

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            probability = (

                final_model
                .predict_proba(
                    X_input
                )[0, 1]

            )

            prediction = int(

                probability
                >= final_threshold

            )

        # ====================================================
        # RESULTS
        # ====================================================

        st.markdown(
            "## Risk Assessment Result"
        )

        probability_pct = (
            probability * 100
        )

        if prediction == 1:

            risk_level = "HIGH RISK"

            recommendation = (

                "The model estimates a relatively "
                "high probability of loan default. "
                "Further credit assessment is recommended "
                "before approval."

            )

            st.error(
                risk_level
            )

        else:

            risk_level = "LOWER RISK"

            recommendation = (

                "The model estimates a relatively "
                "lower probability of loan default. "
                "The application may proceed to "
                "normal credit assessment."

            )

            st.success(
                risk_level
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(

                "Default Probability",

                f"{probability_pct:.2f}%"

            )

        with col2:

            st.metric(

                "Decision Threshold",

                f"{final_threshold * 100:.0f}%"

            )

        with col3:

            st.metric(

                "Model Decision",

                (
                    "Default Risk"
                    if prediction == 1
                    else "Lower Risk"
                )

            )

        st.progress(
            min(probability, 1.0)
        )

        st.markdown(
            "### Interpretation"
        )

        st.write(
            recommendation
        )

        st.caption(

            "This prediction is a machine-learning "
            "risk estimate and should support, not replace, "
            "human credit assessment."

        )

        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        st.divider()

        st.markdown(
            "### Why did the model make this decision?"
        )

        st.write(

            "SHAP explains the individual prediction by "
            "showing which features pushed the model toward "
            "higher or lower estimated default risk."

        )

        try:

            shap_explanation = (
                get_shap_explanation(
                    final_model,
                    X_input
                )
            )

            meaningful = (
                shap_explanation[
                    shap_explanation["Impact"] > 0
                ]
                .copy()
            )

            risk_increasing = (

                meaningful[
                    meaningful["SHAP"] > 0
                ]

                .sort_values(
                    "SHAP",
                    ascending=False
                )

                .head(5)

            )

            risk_reducing = (

                meaningful[
                    meaningful["SHAP"] < 0
                ]

                .sort_values(
                    "SHAP",
                    ascending=True
                )

                .head(5)

            )

            col1, col2 = st.columns(2)

            # ------------------------------------------------
            # RISK INCREASING
            # ------------------------------------------------

            with col1:

                st.markdown(
                    "#### Factors Increasing Estimated Risk"
                )

                if risk_increasing.empty:

                    st.info(
                        "No material risk-increasing "
                        "factors were identified."
                    )

                else:

                    max_impact = max(

                        float(
                            risk_increasing[
                                "SHAP"
                            ]
                            .abs()
                            .max()
                        ),

                        1e-9

                    )

                    for _, row in (
                        risk_increasing.iterrows()
                    ):

                        st.write(

                            f"**{pretty_feature_name(row['Feature'])}**"

                        )

                        st.caption(

                            f"Model impact: +{row['SHAP']:.4f} | "
                            f"Input value: {row['Value']:.4g}"

                        )

                        st.progress(

                            min(

                                float(
                                    abs(row["SHAP"])
                                )
                                / max_impact,

                                1.0

                            )

                        )

            # ------------------------------------------------
            # RISK REDUCING
            # ------------------------------------------------

            with col2:

                st.markdown(
                    "#### Factors Reducing Estimated Risk"
                )

                if risk_reducing.empty:

                    st.info(
                        "No material risk-reducing "
                        "factors were identified."
                    )

                else:

                    max_impact = max(

                        float(
                            risk_reducing[
                                "SHAP"
                            ]
                            .abs()
                            .max()
                        ),

                        1e-9

                    )

                    for _, row in (
                        risk_reducing.iterrows()
                    ):

                        st.write(

                            f"**{pretty_feature_name(row['Feature'])}**"

                        )

                        st.caption(

                            f"Model impact: {row['SHAP']:.4f} | "
                            f"Input value: {row['Value']:.4g}"

                        )

                        st.progress(

                            min(

                                float(
                                    abs(row["SHAP"])
                                )
                                / max_impact,

                                1.0

                            )

                        )

            # ------------------------------------------------
            # TOP DRIVERS TABLE
            # ------------------------------------------------

            st.markdown(
                "#### Top Model Drivers"
            )

            top_drivers = (
                shap_explanation
                .head(10)
                .copy()
            )

            top_drivers["Feature"] = (
                top_drivers["Feature"]
                .apply(
                    pretty_feature_name
                )
            )

            top_drivers["Direction"] = np.where(

                top_drivers["SHAP"] > 0,

                "Higher default risk",

                "Lower default risk"

            )

            top_drivers["SHAP Impact"] = (
                top_drivers["SHAP"]
                .round(4)
            )

            st.dataframe(

                top_drivers[
                    [
                        "Feature",
                        "Direction",
                        "SHAP Impact"
                    ]
                ],

                hide_index=True,

                use_container_width=True

            )

            st.caption(

                "SHAP impact describes the direction and "
                "relative contribution of each feature to "
                "this individual model prediction. It does "
                "not represent a causal relationship."

            )

        except Exception as shap_error:

            st.warning(

                "The credit-risk prediction was completed "
                "successfully, but the local explanation "
                "could not be generated."

            )

            st.caption(
                f"Explanation details: {shap_error}"
            )


# ============================================================
# BATCH ASSESSMENT
# ============================================================

with tab2:

    st.subheader(
        "Batch Credit Risk Assessment"
    )

    st.write(

        """
        Upload a CSV containing applicant records.
        The application will align the columns with the
        model's training schema and generate predictions.
        """

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
                "Uploaded data:",
                batch_df.shape
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

                    if "TARGET" in (
                        batch_input.columns
                    ):

                        batch_input = (
                            batch_input.drop(
                                columns=["TARGET"]
                            )
                        )

                    if "SK_ID_CURR" in (
                        batch_input.columns
                    ):

                        applicant_ids = (
                            batch_input[
                                "SK_ID_CURR"
                            ]
                            .copy()
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

                    probabilities = (

                        final_model
                        .predict_proba(
                            X_batch
                        )[:, 1]

                    )

                    predictions = (

                        probabilities
                        >= final_threshold

                    ).astype(int)

                    results = pd.DataFrame({

                        "SK_ID_CURR":
                            applicant_ids.values,

                        "Default_Probability":
                            probabilities,

                        "Prediction":
                            predictions

                    })

                st.success(
                    "Batch assessment completed successfully."
                )

                st.dataframe(

                    results,

                    use_container_width=True

                )

                csv = (
                    results
                    .to_csv(index=False)
                    .encode("utf-8")
                )

                st.download_button(

                    "Download Predictions",

                    data=csv,

                    file_name=
                        "credit_risk_predictions.csv",

                    mime="text/csv",

                    use_container_width=True

                )

        except Exception as e:

            st.error(
                "The uploaded file could not be processed."
            )

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(

    "Credit Risk Modelling Project • "
    "Tuned LightGBM • "
    f"{len(training_columns)} model features • "
    f"Decision threshold: {final_threshold:.2f}"

)