"""
Loan Approval Assessment App

Streamlit web app for evaluating applicant loan eligibility based on 
a trained logistic regression model (~45k historical records).
"""

import pandas as pd
import joblib
import streamlit as st

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for clean UI styling
st.markdown(
    """
    <style>
    .main { padding-top: 2rem; }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        height: 3rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e9ecef;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load trained model pipeline and cached evaluation metrics
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("model.pkl")
    metrics = joblib.load("metrics.pkl")
    return pipeline, metrics


try:
    pipeline, metrics = load_artifacts()
except Exception as e:
    st.error("Could not load model artifacts. Make sure model.pkl exists in the directory.")
    st.stop()


# Header section
st.title("🏦 Loan Approval Assessor")
st.write(
    "Estimate loan approval chances based on historical lending patterns. "
    "This tool analyzes applicant demographics, income stability, credit score, and requested loan terms using a model trained on over 45,000 application records."
)

# Expandable benchmark metrics
with st.expander(f"📊 View Model Performance Benchmark (Test set: {metrics['n_test']:,} applications)"):
    m1, m2, m3 = st.columns(3)
    m1.metric("Model Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
    m2.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
    m3.metric("Recall", f"{metrics['recall'] * 100:.1f}%")

    m4, m5 = st.columns(2)
    m4.metric("F1 Score", f"{metrics['f1'] * 100:.1f}%")
    m5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    st.markdown("##### Confusion Matrix")
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Rejected", "Actual: Approved"],
        columns=["Predicted: Rejected", "Predicted: Approved"],
    )
    st.dataframe(cm_df, use_container_width=True)
    st.caption(
        f"Evaluated on {metrics['n_test']:,} unseen records ({metrics['n_train']:,} training samples). "
        "Because approval data is naturally imbalanced (~22% baseline approval rate), the model is tuned "
        "with weighted classes to emphasize recall—minimizing missed qualified applicants."
    )

st.divider()

# Input form section
st.subheader("📋 Applicant & Loan Profile")
st.write("Fill out the details below to test a loan scenario:")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Personal Information**")
    person_age = st.number_input("Age", min_value=18, max_value=80, value=28, step=1, help="Applicant age in years")
    person_gender = st.selectbox("Gender", ["male", "female"])
    person_education = st.selectbox(
        "Education Level", ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
    )
    person_income = st.number_input("Annual Income ($)", min_value=0, value=60000, step=1000)
    person_emp_exp = st.number_input("Employment History (Years)", min_value=0, max_value=60, value=3, step=1)
    person_home_ownership = st.selectbox("Home Ownership Status", ["RENT", "MORTGAGE", "OWN", "OTHER"])

with col2:
    st.markdown("**Loan & Credit Profile**")
    loan_amnt = st.number_input("Requested Loan Amount ($)", min_value=500, value=10000, step=500)
    loan_intent = st.selectbox(
        "Purpose of Loan",
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
    )
    loan_int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=25.0, value=11.0, step=0.1)
    cb_person_cred_hist_length = st.number_input("Credit History (Years)", min_value=0, max_value=40, value=5, step=1)
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=640, step=5)
    previous_loan_defaults_on_file = st.selectbox("Prior Defaults on Record?", ["No", "Yes"])

# Automatically calculate debt-to-income ratio for feedback
loan_percent_income = round(loan_amnt / person_income, 4) if person_income > 0 else 0.0
st.info(f"💡 **Calculated Loan-to-Income Ratio:** {loan_percent_income * 100:.1f}% of annual income")

st.divider()

# Prediction execution
if st.button("Run Approval Assessment", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "person_gender": person_gender,
        "person_education": person_education,
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file,
    }])

    prediction = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0]
    approval_prob = proba[1]

    st.subheader("Assessment Result")
    if prediction == 1:
        st.success(f"✅ **High Probability of Approval** — Estimated approval probability: **{approval_prob * 100:.1f}%**")
    else:
        st.error(f"❌ **Risk of Rejection** — Estimated rejection probability: **{(1 - approval_prob) * 100:.1f}%**")

    st.progress(float(approval_prob))

    # Analytical takeaways based on statistical modeling findings
    st.markdown("##### Key Risk Observations")
    if loan_percent_income > 0.35:
        st.warning(
            "⚠️ **High Loan-to-Income:** The requested loan exceeds 35% of the applicant's annual income. "
            "In statistical analysis, high loan-to-income is one of the strongest drivers of loan rejection."
        )
    if previous_loan_defaults_on_file == "Yes":
        if loan_int_rate >= 15:
            st.info(
                "ℹ️ **Prior Default & High Rate:** Prior defaults correlate with higher interest rates assigned by lenders. "
                "The elevated interest rate reflects priced-in credit risk."
            )
        else:
            st.info(
                "ℹ️ **Prior Default Note:** While prior default increases risk, model coefficients show that interest rate "
                "and debt burden play a more dominant role in the final approval threshold."
            )
    if credit_score >= 700:
        st.success("🌟 **Strong Credit Rating:** A credit score of 700+ provides a strong positive boost to approval odds.")

st.divider()
st.caption(
    "Note: Built for demonstration & statistical portfolio analysis. Model results should not be used as official financial decisions."
)

