import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved components
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
ordinal_encoders = joblib.load('ordinal_encoders.pkl')

# Define feature list and matching order
selected_features = ['Transaction_Amount', 'Account_Balance', 'Transaction_Device', 'Device_Type',
                     'Gender', 'State', 'Merchant_Category', 'Account_Type']

st.title("🏦 Bank Transaction Fraud Detection App")

st.subheader("Enter Transaction Details:")

# Collect user input
input_data = {}

for feature in selected_features:
    if feature == 'Transaction_Amount':
        # Allow user to input Transaction_Amount manually
        input_data[feature] = st.number_input(f"Enter {feature} (in USD):", min_value=0.0, value=100.0, step=10.0)
    elif feature == 'Account_Balance':
        # Allow user to input Account_Balance manually
        input_data[feature] = st.number_input(f"Enter {feature} (in USD):", min_value=0.0, value=1000.0, step=50.0)
    elif feature in ordinal_encoders:
        # For categorical features, use selectbox
        categories = list(ordinal_encoders[feature].categories_[0])
        selected = st.selectbox(f"{feature}:", categories)
        encoded = ordinal_encoders[feature].transform([[selected]])[0][0]
        input_data[feature] = encoded
    else:
        # For other numerical features, use number_input
        input_data[feature] = st.number_input(f"{feature}:", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Scale numerical features
scaled_input = scaler.transform(input_df)

# Make prediction
if st.button("Predict Fraud"):
    prediction = model.predict(scaled_input)[0]
    result = "🚨 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
    st.success(f"Prediction: {result}")