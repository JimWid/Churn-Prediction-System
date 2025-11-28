import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Loading the model and scaler
MODEL = joblib.load('xgb_churn_model.pkl')
SCALER = joblib.load('scaler.pkl')
FEATURES = joblib.load('feature_names.pkl')

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("Customer Churn Prediction App")

st.write("Enter customer details to predict whether they are likely to churn!!")

# Input fields:
tenure = st.number_input("Tenure (in months) max. ", min_value=0, max_value=100)
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=300.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=1_000_000.0)

online_security = st.selectbox("Online Security", options=["No", "Yes"])
tech_support = st.selectbox("Tech Support", options=["No", "Yes"])
contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Input data preparation
input_data = {
    'tenure': tenure,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges,
    'OnlineSecurity': online_security,
    'TechSupport': tech_support,
    'Contract': contract,
    'PaymentMethod': payment_method
}

# Button to predict
if st.button("Predict Churn"):

    input_df = pd.DataFrame([input_data])

    st.write("Input (raw):")
    st.dataframe(input_df.T)

    input_df['OnlineSecurity'] = input_df['OnlineSecurity'].replace({'Yes': 1, 'No': 0})
    input_df['TechSupport'] = input_df['TechSupport'].replace({'Yes': 1, 'No': 0})

    # Scaling numerical features
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[numerical_features] = SCALER.transform(input_df[numerical_features])

    # One-hot encoding categorical features
    input_dummies = pd.get_dummies(input_df, columns=['Contract', 'PaymentMethod'], drop_first=False)

    X = input_dummies.reindex(columns=FEATURES, fill_value=0)
    
    st.write("Prepared input (after dummies & reindex):")
    st.dataframe(X.head().T)
    

    # Prediction
    prediction = MODEL.predict(X)
    prediction_proba = MODEL.predict_proba(X)[:, 1]

    st.write("prediction: ", prediction)
    st.write("prediction_proba: ", prediction_proba)

    if prediction[0] == 1:
        st.error(f"The customer is likely to churn D: (Probability: {prediction_proba[0]:.2f})")
    else:
        st.success(f"The customer is unlikely to churn :D (Probability: {1 - prediction_proba[0]:.2f})")