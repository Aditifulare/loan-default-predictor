import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="💰", layout="centered")

# ---------- Light sky-blue theme (matching your style) ----------
st.markdown("""
<style>
.stApp { background-color: #EAF4FB; }
h1, h2, h3 { color: #0B3D5C; }
div.stButton > button {
    background-color: #1B6CA8; color: white; border-radius: 8px;
    font-weight: 600; padding: 0.5em 1.5em; border: none;
}
div.stButton > button:hover { background-color: #0B3D5C; color: white; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    options = joblib.load("options.pkl")
    return model, feature_cols, options

model, feature_cols, options = load_artifacts()

st.title("💰 Loan Default Risk Predictor")
st.write("Enter applicant details to estimate the probability of loan default. (Random Forest model, ~91% accuracy)")

st.divider()

col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("Applicant Age", min_value=18, max_value=80, value=28)
    person_income = st.number_input("Annual Income (₹)", min_value=1000, max_value=1000000, value=50000, step=1000)
    person_emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=60, value=5)
    person_home_ownership = st.selectbox("Home Ownership", options['person_home_ownership'])
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=40, value=4)

with col2:
    loan_amnt = st.number_input("Loan Amount (₹)", min_value=500, max_value=100000, value=10000, step=500)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=40.0, value=11.5, step=0.1)
    loan_intent = st.selectbox("Loan Purpose", options['loan_intent'])
    loan_grade = st.selectbox("Loan Grade", options['loan_grade'])
    cb_person_default_on_file = st.selectbox("Prior Default on File?", options['cb_person_default_on_file'])

loan_percent_income = round(loan_amnt / person_income, 4) if person_income > 0 else 0

st.divider()
st.caption(f"Calculated Loan-to-Income Ratio: **{loan_percent_income:.2%}**")

if st.button("🔍 Predict Default Risk"):
    raw = {
        'person_age': person_age,
        'person_income': person_income,
        'person_emp_length': person_emp_length,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'person_home_ownership': person_home_ownership,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'cb_person_default_on_file': cb_person_default_on_file,
    }
    input_df = pd.DataFrame([raw])
    input_encoded = pd.get_dummies(input_df, columns=[
        'person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'
    ])

    # Align columns with training feature set
    for col in feature_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_cols]

    proba = model.predict_proba(input_encoded)[0][1]
    pred = model.predict(input_encoded)[0]

    st.divider()
    if pred == 1:
        st.error(f"⚠️ HIGH RISK — Predicted probability of default: **{proba:.1%}**")
        st.write("This applicant matches patterns seen in past defaults. Recommend extra scrutiny or stricter terms.")
    else:
        st.success(f"✅ LOW RISK — Predicted probability of default: **{proba:.1%}**")
        st.write("This applicant matches patterns seen in past loans that were repaid in full.")

    st.progress(min(proba, 1.0))

st.divider()
st.caption("Built as part of the Credit Risk & Loan Default Prediction capstone project.")
