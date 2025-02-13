import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("loan_model.pkl")

# Set Page Config
st.set_page_config(page_title="Loan Default Predictor", page_icon="🏦", layout="wide")

# Main Title
st.markdown(
    "<h1 style='text-align: center; color: #1E3A8A;'>🏦 Loan Default Prediction App</h1>",
    unsafe_allow_html=True,
)

st.markdown("#### Enter the required details below to check the loan risk prediction.")

# Layout with Columns
col1, col2 = st.columns([2, 2])

with st.form("loan_form"):
    with col1:
        st.subheader("📋 Applicant Information")
        age = st.slider("🎂 Age", min_value=18, max_value=70, value=30)
        income = st.number_input("💰 Monthly Income (₹)", min_value=1000, max_value=1000000, step=1000, value=30000)
        employment_status = st.selectbox("👔 Employment Type", ["Salaried", "Self-Employed", "Unemployed"])
        credit_score = st.slider("📊 Credit Score", min_value=300, max_value=900, value=700)

    with col2:
        st.subheader("🏡 Loan Information")
        loan_amount = st.number_input("🏦 Loan Amount (₹)", min_value=5000, max_value=5000000, step=5000, value=500000)
        loan_duration = st.selectbox("⏳ Loan Duration", ["6 months", "1 year", "3 years", "5 years", "10 years"])
        interest_rate = st.slider("📈 Interest Rate (%)", min_value=1.0, max_value=20.0, step=0.1, value=7.5)
        previous_loans = st.selectbox("📜 Previous Loan History", ["No Previous Loans", "Paid on Time", "Missed Payments"])

    # Financial Details
    st.subheader("🏠 Financial & Other Details")
    savings = st.number_input("📉 Total Debt (₹)", min_value=0, max_value=10000000, step=1000, value=50000)
    debt = st.number_input("💵 Savings (₹)", min_value=0, max_value=10000000, step=1000, value=100000)

    # Generate remaining 12 missing features with default values
    default_features = [0] * 12

    # Submit Button
    submitted = st.form_submit_button("🔍 Predict Loan Default", use_container_width=True)

# Feature Encoding (Convert categorical values to numbers)
employment_dict = {"Salaried": 0, "Self-Employed": 1, "Unemployed": 2}
loan_duration_dict = {"6 months": 0, "1 year": 1, "3 years": 2, "5 years": 3, "10 years": 4}
previous_loans_dict = {"No Previous Loans": 0, "Paid on Time": 1, "Missed Payments": 2}

# Run Prediction when Submit Button Clicked
if submitted:
    features = np.array([[age, income, employment_dict[employment_status], credit_score, loan_amount, 
                          loan_duration_dict[loan_duration], interest_rate, previous_loans_dict[previous_loans], 
                          savings, debt] + default_features])  # Add missing 12 features
    
    with st.spinner("Processing Prediction... 🔄"):
        prediction = model.predict(features)

    st.subheader("📊 Prediction Result:")
    
    if prediction[0] == 1:
        st.error("🚨 High Risk! Loan Likely to Default.")
        st.progress(85)
    else:
        st.success("✅ Low Risk! Loan Likely to be Repaid.")
        st.progress(20)
