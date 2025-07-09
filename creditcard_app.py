#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import joblib

# Load scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("best_model.pkl")

st.title("Credit Card Fraud Detection")
st.write("Enter transaction details to predict fraud")

# Define the exact columns the scaler/model expect
expected_columns = [
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

# Build the input form
user_input = {}

st.sidebar.header("Input Features")

for col in expected_columns:
    user_input[col] = st.sidebar.number_input(col, value=0.0)

# Build DataFrame
input_df = pd.DataFrame([user_input])

st.write("### Input Data", input_df)

# Make sure columns are in correct order
input_df = input_df[expected_columns]

# Scale
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)[0]
proba = model.predict_proba(input_scaled)[0][1]

if prediction == 1:
    st.error(f"⚠️ Fraudulent Transaction Detected! (Fraud probability: {proba:.2%})")
else:
    st.success(f"✅ Legitimate Transaction (Fraud probability: {proba:.2%})")



# In[ ]:




