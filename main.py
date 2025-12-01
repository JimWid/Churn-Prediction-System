import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Loading the model and scaler
MODEL = joblib.load('models/xgb_churn_model.pkl')
# MODEL = joblib.load('models/logistic_regression_model.pkl')
SCALER = joblib.load('models/scaler.pkl')
FEATURES = joblib.load('models/feature_names.pkl')

st.set_page_config(page_title="Customer Churn Prediction", layout="centered", page_icon="😸")
st.markdown("""
<div style="display:flex;align-items:center;gap:10px;">
  <div style="flex:1">
    <h2 style="margin:0;">Customer Churn Prediction</h2>
    <div style="color:gray;font-size:0.9rem;margin-top:2px;">
      Predicting Churn Risk with XGBoost / Logistic Regression model
    </div>
  </div>
  <div style="text-align:right;">
    <small style="color:gray">Quick demo • Streamlit</small>
  </div>
</div>
""", unsafe_allow_html=True)

st.write("---")

# Input fields:
with st.sidebar.form("input_form"):
    st.header("Customer Details Input")
    tenure = st.number_input("Tenure (in months) (max. 200) ", min_value=0, max_value=200)
    MonthlyCharges = st.number_input("Monthly Charges (max. 300.00)", min_value=0.0, max_value=300.0)
    TotalCharges = st.number_input("Total Charges (max. 1,000,000)", min_value=0.0, max_value=1_000_000.0)

    online_security = st.selectbox("Online Security", options=["No", "Yes"])
    tech_support = st.selectbox("Tech Support", options=["No", "Yes"])
    contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    submit_button = st.form_submit_button(label="Predict")

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
if submit_button:

    input_df = pd.DataFrame([input_data])

    input_df['OnlineSecurity'] = input_df['OnlineSecurity'].replace({'Yes': 1, 'No': 0})
    input_df['TechSupport'] = input_df['TechSupport'].replace({'Yes': 1, 'No': 0})

    # If you want to use Logistic Regression model + Scaler:
    # Scaling numerical features
    # numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    # input_df[numerical_features] = SCALER.transform(input_df[numerical_features])

    # One-hot encoding categorical features
    input_dummies = pd.get_dummies(input_df, columns=['Contract', 'PaymentMethod'], drop_first=False)

    X = input_dummies.reindex(columns=FEATURES, fill_value=0)

    # Prediction
    prediction = MODEL.predict(X)
    prediction_proba = MODEL.predict_proba(X)[:, 1].item()


    left, right = st.columns([1, 1])

    # Left column: prediction result
    if prediction == 1:
        left.metric(label="Churn Prediction", delta="Likely", delta_color="inverse", value=f"{prediction_proba:.2f}")
    else:
        left.metric(label="Churn Prediction", delta="Unlikely", value=f"{prediction_proba:.2f}")

    # Right column: Input summary
    with right:
        st.subheader("Input summary")
        st.write(input_df.T)

    st.write("---")
    with st.expander("Show raw model output & debug"):
        st.write("Prediction:", int(prediction))
        st.write("Churn probability:", float(prediction_proba))

st.caption("@ Developed by JimWid")