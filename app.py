import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

st.set_page_config(page_title="Employee Attrition Predictor")

st.title("Employee Attrition Prediction")
st.write("Enter Employee Details")

# Load files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

with open("columns.json", "r") as f:
    model_columns = json.load(f)

# Basic Inputs
age = st.slider("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
distance = st.slider("Distance From Home", 1, 30, 5)
job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
years_at_company = st.slider("Years At Company", 0, 40, 5)
overtime = st.selectbox("OverTime", ["Yes", "No"])

if st.button("Predict"):

    input_dict = {
        "Age": age,
        "MonthlyIncome": monthly_income,
        "DistanceFromHome": distance,
        "JobSatisfaction": job_satisfaction,
        "YearsAtCompany": years_at_company,
        "OverTime_Yes": 1 if overtime == "Yes" else 0
    }

    input_df = pd.DataFrame([input_dict])

    # Add missing columns
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠ Employee Likely to Leave (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Employee Likely to Stay (Probability: {1-probability:.2f})")