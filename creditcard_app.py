#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('best_model.pkl')

st.title("Credit Card Fraud Detection")
st.write("Enter transaction details to predict fraud")

# Define all feature columns (the ones used to train the model)
feature_columns = [
    f'V{i}' for i in range(1, 29)
] + ['Amount']  # adjust if you also include 'Time'

# Build form for user input
user_input = {}
st.sidebar.header("Input Transaction Data")

for col in feature_columns:
    if col == 'Amount':
        user_input[col] = st.sidebar.number_input(col, min_value=0.0, step=0.01, value=0.0)
    else:
        user_input[col] = st.sidebar.number_input(col, step=0.01, value=0.0)

# Convert user input to DataFrame
input_df = pd.DataFrame([user_input])

st.write("### Input Data", input_df)

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)[0]
prediction_proba = model.predict_proba(input_scaled)[0][1]

if prediction == 1:
    st.error(f"⚠️ Fraudulent Transaction Detected! (Probability: {prediction_proba:.2%})")
else:
    st.success(f"✅ Legitimate Transaction (Probability of Fraud: {prediction_proba:.2%})")


# In[ ]:




