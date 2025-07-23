import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained SVR model
with open("svr_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load encoders and column transformer
with open("encoders.pkl", "rb") as f:
    le_sex, le_smoker, le_time, ct = pickle.load(f)

st.title("Total Bill Prediction (SVR)")
st.markdown("Predict the **total bill** in a restaurant based on tip, size, and other attributes.")

# User inputs
tip = st.number_input("Tip Amount", min_value=0.0, format="%.2f")
sex = st.selectbox("Sex", ["Male", "Female"])
smoker = st.selectbox("Smoker", ["Yes", "No"])
day = st.selectbox("Day", ["Thur", "Fri", "Sat", "Sun"])
time = st.selectbox("Time", ["Lunch", "Dinner"])
size = st.slider("Table Size", 1, 10, 2)

# Encode input as a DataFrame
input_df = pd.DataFrame({
    "tip": [tip],
    "sex": [le_sex.transform([sex])[0]],
    "smoker": [le_smoker.transform([smoker])[0]],
    "day": [day],
    "time": [le_time.transform([time])[0]],
    "size": [size]
})

# Transform features
X_transformed = ct.transform(input_df)

# Predict
if st.button("Predict Total Bill"):
    prediction = model.predict(X_transformed)
    st.success(f"Predicted Total Bill: ${prediction[0]:.2f}")
