# employee_app.py

import streamlit as st
import pandas as pd
from joblib import load

# =========================
# Load model & encoders
# =========================
model = load("employee_lgbm_model.joblib")
encoders = load("employee_encoders.joblib")

# Feature order from training
FEATURE_ORDER = model.feature_name_

st.set_page_config(page_title="Employee Churn Prediction")
st.title("Employee Churn Prediction App")

st.write("Fill employee details to predict churn risk.")

# =========================
# User Inputs
# =========================
city = st.text_input("City", "city_103")
city_dev_index = st.slider("City Development Index", 0.0, 1.0, 0.75)

gender = st.selectbox("Gender", ["Male", "Female", "0"])
relevant_exp = st.selectbox(
    "Relevant Experience",
    ["Has relevent experience", "No relevent experience"]
)

enrolled_univ = st.selectbox(
    "Enrolled University",
    ["no_enrollment", "Full time course", "Part time course", "0"]
)

education = st.selectbox(
    "Education Level",
    ["Graduate", "Masters", "High School", "Primary School", "0"]
)

major = st.selectbox(
    "Major Discipline",
    ["STEM", "Business Degree", "Arts", "Humanities", "Other", "0"]
)

experience = st.selectbox(
    "Experience",
    ["<1", "1", "2", "3", "4", "5", ">20", "0"]
)

company_size = st.selectbox(
    "Company Size",
    ["<10", "10-49", "50-99", "100-500", "500-999", "1000+", "0"]
)

company_type = st.selectbox(
    "Company Type",
    ["Pvt Ltd", "Funded Startup", "Public Sector", "NGO", "Other", "0"]
)

last_new_job = st.selectbox(
    "Last New Job",
    ["never", "1", "2", "3", "4", ">4"]
)

training_hours = st.number_input("Training Hours", 0, 500, 40)

# =========================
# Prediction
# =========================
if st.button("Predict Employee Churn"):

    # Create dataframe
    df = pd.DataFrame([{
        "city": city,
        "city_development_index": city_dev_index,
        "gender": gender,
        "relevent_experience": relevant_exp,
        "enrolled_university": enrolled_univ,
        "education_level": education,
        "major_discipline": major,
        "experience": experience,
        "company_size": company_size,
        "company_type": company_type,
        "last_new_job": last_new_job,
        "training_hours": training_hours
    }])

    # Replace any missing with 0
    df.fillna(0, inplace=True)

    # Encode categorical columns
    for col, le in encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col].astype(str))
            except ValueError:
                df[col] = -1  # unseen category

    # Match training feature order
    df = df[FEATURE_ORDER]

    # Force numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    # Output
    if prediction == 1:
        st.error(f"⚠️ Employee likely to LEAVE (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Employee likely to STAY (Probability: {probability:.2%})")
