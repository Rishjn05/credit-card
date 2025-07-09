#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model & scaler
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Credit Card Fraud Detection")

st.write("Enter transaction details below to predict if it’s fraudulent or not.")

# Input fields
feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]

user_input = {}
for feat in feature_names:
    user_input[feat] = st.number_input(feat, value=0.0)

input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    pred = model.predict(input_scaled)[0]
    probab = model.predict_probab(input_scaled)[0][1]

    if pred == 1:
        st.error(f"Fraudulent Transaction Detected! (probability: {probab:.2f})")
    else:
        st.success(f"Legitimate Transaction. (probability: {1-probab:.2f})")


# In[ ]:




